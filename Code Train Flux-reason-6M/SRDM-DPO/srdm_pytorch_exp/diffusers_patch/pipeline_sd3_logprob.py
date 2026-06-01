"""
SD3 Pipeline with LogProb.

仿照 DDPO 的 pipeline_with_logprob.py，适配 Stable Diffusion 3 (流匹配模型)。

关键差异 vs DDPO (SD1.5):
    - SD3 Transformer 取代 UNet (forward 签名不同)
    - CFG 需同时复制 prompt_embeds 和 pooled_prompt_embeds
    - 不需要 scale_model_input (SD3 scheduler 无此步骤)
    - 使用随机性流匹配 SDE step 替代 DDIM step
"""

import math
from typing import List, Optional, Union

import torch
from diffusers import StableDiffusion3Pipeline

from .flow_match_sde import StochasticFlowMatchScheduler


@torch.no_grad()
def pipeline_sd3_with_logprob(
    pipeline: StableDiffusion3Pipeline,
    prompt_embeds: torch.FloatTensor,
    pooled_prompt_embeds: torch.FloatTensor,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    output_type: str = "pt",
    latents: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
):
    """
    运行 SD3 管道并记录所有中间潜变量和对数概率。

    Args:
        pipeline: 已加载的 StableDiffusion3Pipeline (scheduler 须为 StochasticFlowMatchScheduler)。
        prompt_embeds: 正向 prompt embeddings [B, L, D]。
        pooled_prompt_embeds: 正向 pooled prompt embeddings [B, D]。
        negative_prompt_embeds: 负向 prompt embeddings [B, L, D] (CFG 时使用)。
        negative_pooled_prompt_embeds: 负向 pooled prompt embeddings [B, D]。
        height: 图像高度 (None 则用默认)。
        width: 图像宽度 (None 则用默认)。
        num_inference_steps: 推理步数。
        guidance_scale: CFG 引导强度 (> 1.0 启用 CFG)。
        output_type: 输出类型 ("pt" = 原始 tensor)。
        latents: 初始噪声 (None 则随机生成)。
        generator: 随机数生成器。单个 Generator 表示所有步共享同一状态 (链式延续)；
                   List[Generator] 表示每步使用独立的生成器 (List[Generator])。

    Returns:
        (images, all_latents, all_log_probs)
        - images: 解码后的图像 tensor [B, C, H, W]。
        - all_latents: 所有步的潜变量列表 [x_T, x_{T-1}, ..., x_0]。
        - all_log_probs: 所有步的 log_prob 列表 [log_p_T, ..., log_p_1]。
    """
    scheduler: StochasticFlowMatchScheduler = pipeline.scheduler
    device = pipeline.transformer.device
    batch_size = prompt_embeds.shape[0]

    # 1. 确定图像尺寸
    if height is None:
        height = pipeline.default_sample_size * pipeline.vae_scale_factor
    if width is None:
        width = pipeline.default_sample_size * pipeline.vae_scale_factor
    num_channels_latents = pipeline.transformer.config.in_channels

    # 2. 初始化潜变量 x_T ~ N(0, I)
    if latents is None:
        latents_shape = (
            batch_size,
            num_channels_latents,
            height // pipeline.vae_scale_factor,
            width // pipeline.vae_scale_factor,
        )
        # 新版本 torch 要求 generator 和 device 一致
        if generator is not None and generator.device != device:
            gen = torch.Generator(device=device).manual_seed(
                generator.initial_seed()
            )
        else:
            gen = generator
        latents = torch.randn(
            latents_shape, generator=gen, device=device, dtype=prompt_embeds.dtype
        )

    # 3. 设置时间步
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 4. CFG 判断
    do_cfg = guidance_scale > 1.0

    # 5. 去噪循环
    all_latents = [latents]
    all_log_probs = []

    # 5a. 计算 log p(x_T): x_T ~ N(0, I) 的 log 密度
    #     log p(x_T) = -||x_T||²/2 - D/2 * log(2π)
    #     沿 batch 外维度取均值，与 scheduler.step() 返回的 log_prob 维度一致
    xT = latents
    D = xT[0].numel()  # 单个样本的维度 (C × H × W)
    # float16 下 sum(65536 个元素) 会溢出，先转 float32 再累加
    log_p_xT = (
        -0.5 * (xT ** 2).flatten(1).float().sum(dim=1)
        - 0.5 * D * math.log(2 * math.pi)
    )
    all_log_probs.append(log_p_xT)

    for i, t in enumerate(timesteps):
        # 5a. CFG: 同时复制 latent 和 text embeddings
        if do_cfg:
            latent_input = torch.cat([latents] * 2)
            embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        else:
            latent_input = latents
            embeds_input = prompt_embeds
            pooled_input = pooled_prompt_embeds

        # 5b. Transformer forward (不是 UNet!)
        # SD3 要求 timestep 为 1D tensor，expand 到 batch size
        t_batch = t.expand(latent_input.shape[0])
        noise_pred = pipeline.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=embeds_input,
            pooled_projections=pooled_input,
            timestep=t_batch,
            return_dict=False,
        )[0]

        # 5c. CFG 组合
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # 5d. 随机性流匹配 SDE step (支持 per-step generator)
        step_gen = generator[i] if isinstance(generator, list) else generator
        latents, log_prob = scheduler.step(
            noise_pred, t, latents, generator=step_gen, return_dict=False
        )

        all_latents.append(latents)
        all_log_probs.append(log_prob)

    # 6. VAE 解码
    latents_for_decode = latents / pipeline.vae.config.scaling_factor
    if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor is not None:
        latents_for_decode = latents_for_decode + pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents_for_decode, return_dict=False)[0]

    if output_type == "pt":
        return images, all_latents, all_log_probs
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")
