"""Common training utilities — 扩散采样 + prompt 编码 + PPO 辅助。

所有实验共享的基础设施:
    - pipeline_sd3_train_sample: 扩散去噪 + 双路 log_prob (θ 和 θ_base)
    - total_log_prob_from_list: 跨步 log_prob 求和
    - zscore_normalize: 组内 z-score 归一化
    - encode_prompt: SD3 三重 text encoder 编码
    - make_chain_generators: 确定性种子生成器

PPO 相关统一由 srdm_pytorch_exp.ppo_trainer 负责:
    - compute_log_prob_at_step (re-export)
    - ppo_update_mini_batch
    - TrainingAlerter
"""

import math
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from srdm_pytorch_exp.ppo_trainer import compute_log_prob_at_step  # re-export, 单一实现在 ppo_trainer.py
from srdm_pytorch_exp.reward_rin import zscore_normalize  # re-export, 单一实现在 reward_rin.py


@torch.no_grad()
def pipeline_sd3_train_sample(
    pipeline: StableDiffusion3Pipeline,
    base_transformer: torch.nn.Module,
    prompt_embeds: torch.FloatTensor,
    pooled_prompt_embeds: torch.FloatTensor,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
):
    """Run SD3 denoising and record log_probs from BOTH θ (policy) and θ_base (frozen).

    At each step:
      1. Forward θ (policy)       → v_θ   → SDE step → x_{t-1} + log_p_θ
      2. Forward θ_base (frozen)  → v_θb  → SDE step with prev_sample → log_p_θbase

    Returns:
        (images, all_latents, log_probs_old, log_probs_base)
        - images: [B, C, H, W] decoded images
        - all_latents: list of [x_T, x_{T-1}, ..., x_0], each [B, C_lat, H_lat, W_lat]
        - log_probs_old: list[Tensor[B]] — log_p_θ per step (incl. log p(x_T))
        - log_probs_base: list[Tensor[B]] — log_p_θbase per step (incl. log p(x_T))
    """
    scheduler: StochasticFlowMatchScheduler = pipeline.scheduler
    device = pipeline.transformer.device
    batch_size = prompt_embeds.shape[0]

    if height is None:
        height = pipeline.default_sample_size * pipeline.vae_scale_factor
    if width is None:
        width = pipeline.default_sample_size * pipeline.vae_scale_factor
    num_channels_latents = pipeline.transformer.config.in_channels

    # --- x_T ~ N(0, I) ---
    latents_shape = (
        batch_size,
        num_channels_latents,
        height // pipeline.vae_scale_factor,
        width // pipeline.vae_scale_factor,
    )
    if generator is not None:
        if isinstance(generator, list):
            gen = generator[0]  # first generator for x_T
        else:
            gen = generator
        if gen.device != device:
            gen = torch.Generator(device=device).manual_seed(gen.initial_seed())
    else:
        gen = None
    latents = torch.randn(
        latents_shape, generator=gen, device=device, dtype=prompt_embeds.dtype
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    do_cfg = guidance_scale > 1.0

    all_latents = [latents]
    log_probs_old = []
    log_probs_base = []

    # log p(x_T): same for both θ and θ_base
    xT = latents
    D = xT[0].numel()
    log_p_xT = (
        -0.5 * (xT ** 2).flatten(1).float().sum(dim=1)
        - 0.5 * D * math.log(2 * math.pi)
    )
    log_probs_old.append(log_p_xT)
    log_probs_base.append(log_p_xT)

    for i, t in enumerate(tqdm(timesteps, desc="      denoise", leave=False)):
        # --- Build CFG inputs ---
        if do_cfg:
            latent_input = torch.cat([latents] * 2)
            embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        else:
            latent_input = latents
            embeds_input = prompt_embeds
            pooled_input = pooled_prompt_embeds

        t_batch = t.expand(latent_input.shape[0])

        # --- Forward θ (policy) → v_θ ---
        v_theta = pipeline.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=embeds_input,
            pooled_projections=pooled_input,
            timestep=t_batch,
            return_dict=False,
        )[0]
        if do_cfg:
            v_uncond, v_text = v_theta.chunk(2)
            v_theta = v_uncond + guidance_scale * (v_text - v_uncond)

        # --- Forward θ_base (frozen) → v_θbase ---
        v_theta_base = base_transformer(
            hidden_states=latent_input,
            encoder_hidden_states=embeds_input,
            pooled_projections=pooled_input,
            timestep=t_batch,
            return_dict=False,
        )[0]
        if do_cfg:
            vb_uncond, vb_text = v_theta_base.chunk(2)
            v_theta_base = vb_uncond + guidance_scale * (vb_text - vb_uncond)

        # --- SDE step with θ → actual x_{t-1} ---
        latents_prev = latents  # save x_t before update
        step_gen = generator[i + 1] if isinstance(generator, list) else generator
        latents, log_prob_old = scheduler.step(
            v_theta, t, latents, generator=step_gen, return_dict=False
        )
        log_probs_old.append(log_prob_old)

        # --- SDE step with θ_base → log_p_θbase of actual x_{t-1} ---
        _, log_prob_base = scheduler.step(
            v_theta_base, t, latents_prev, prev_sample=latents, return_dict=False
        )
        log_probs_base.append(log_prob_base)

        all_latents.append(latents.float())  # fp32 存储, 保证 log_prob 重算精度

    # --- VAE decode ---
    latents_for_decode = latents / pipeline.vae.config.scaling_factor
    if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor is not None:
        latents_for_decode = latents_for_decode + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents_for_decode, return_dict=False)[0]

    return images, all_latents, log_probs_old, log_probs_base


def total_log_prob_from_list(log_prob_list: List[torch.Tensor]) -> torch.Tensor:
    """Sum log_probs across all steps for each chain.

    Args:
        log_prob_list: list of Tensor[B], length = num_steps + 1

    Returns:
        Tensor[B] — total log_prob per chain
    """
    return torch.stack([lp for lp in log_prob_list], dim=0).sum(dim=0)


@torch.no_grad()
def encode_prompt(
    pipeline: StableDiffusion3Pipeline,
    prompt: Union[str, List[str]],
    device: torch.device,
) -> tuple:
    """Encode a text prompt using SD3's triple text encoders.

    Returns:
        (prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds)
    """
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        device=device,
        do_classifier_free_guidance=True,
    )
    return (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def make_chain_generators(
    base_seed: int, chain_idx: int, num_steps: int, device: torch.device
) -> tuple:
    """Create independent generators for one chain.

    Returns:
        (latents_gen, step_generators_list)
        - latents_gen: Generator for x_T
        - step_generators_list: List[Generator] for each denoising step
    """
    stride = 1000  # safe margin to avoid collisions across chains
    latents_seed = base_seed + chain_idx * stride
    latents_gen = torch.Generator(device=device).manual_seed(latents_seed)
    step_generators = [
        torch.Generator(device=device).manual_seed(latents_seed + 1 + i)
        for i in range(num_steps)
    ]
    return latents_gen, step_generators
