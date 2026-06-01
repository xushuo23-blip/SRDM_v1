"""
实验二 Part2：DDPO 训练 (r_in 内生奖励 + PPO Clip).

训练流程 (per epoch):
    1. 随机选取 2 个 prompt, 各 6 条链 = 12 chains
    2. 用 frozen base model 计算 total_log_p_base
    3. r_in = zscore_normalize(total_log_p_base) 组内 6 条归一化
    4. PPO: per-step ratio = exp(log_p_new - log_p_old), clip + advantage
    5. LoRA fine-tuning, 4 gradient updates per epoch (mini_batch=3)

WandB 输出:
    - r_in 分布 (histogram + mean/std)
    - PPO loss, ratio mean
    - 每 N epoch 采样图像
"""

# 运行方式 (在 SRDM-DDPO/ 目录下):
#     python experiments/exp2/exp2_part2.py --config config/exp2_part2_config.py

import copy
import os
import random
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srdm_pytorch_exp.sde_sampling import (
    compute_log_prob_at_step,
    encode_prompt,
    make_chain_generators,
    pipeline_sd3_train_sample,
    total_log_prob_from_list,
    zscore_normalize,
)
from srdm_pytorch_exp.reward_rin import compute_reward_rin
from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from prompts import load_prompts_from_file


def get_config_from_path(config_path):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return __import__(config_name).get_config()


def run_training(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.float32

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # ================================================================
    # 1. WandB
    # ================================================================
    wandb.init(
        entity="xushuo23-sorbonne-universit-",
        project="SRDM-DDPO",
        name=cfg.run_name,
        config=cfg.to_dict(),
        reinit=True,
    )

    # ================================================================
    # 2. 加载 SD3
    # ================================================================
    print("加载 SD3 模型...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        cfg.pretrained_model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipeline = pipeline.to(device)
    pipeline.vae.requires_grad_(False)
    if hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder.requires_grad_(False)
    if hasattr(pipeline, "text_encoder_2"):
        pipeline.text_encoder_2.requires_grad_(False)
    if hasattr(pipeline, "text_encoder_3"):
        pipeline.text_encoder_3.requires_grad_(False)
    pipeline.safety_checker = None

    # ================================================================
    # 3. 冻结 base model (用于 r_in)
    # ================================================================
    base_transformer = copy.deepcopy(pipeline.transformer)
    base_transformer.requires_grad_(False)
    base_transformer.eval()
    print("base model 已冻结.")

    # ================================================================
    # 4. LoRA 注入 (仅 trainable transformer)
    # ================================================================
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    pipeline.transformer.requires_grad_(False)
    pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
    trainable_params = sum(p.numel() for p in pipeline.transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipeline.transformer.parameters())
    print(f"LoRA: trainable={trainable_params:,} / total={total_params:,}")

    # ================================================================
    # 5. Scheduler + Optimizer
    # ================================================================
    original_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    orig_dict = dict(original_scheduler.config)
    orig_dict = {k: v for k, v in orig_dict.items() if not k.startswith('_')}
    scheduler = StochasticFlowMatchScheduler(a=cfg.a, **orig_dict)
    pipeline.scheduler = scheduler

    optimizer = torch.optim.AdamW(
        [p for p in pipeline.transformer.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # ================================================================
    # 6. 加载 prompt 列表
    # ================================================================
    all_prompts = load_prompts_from_file(cfg.prompt_file)
    print(f"加载 {len(all_prompts)} 条 prompts.")

    # ================================================================
    # 7. 训练循环
    # ================================================================
    num_chains_total = cfg.num_prompts_per_epoch * cfg.num_chains_per_prompt
    num_updates_per_epoch = num_chains_total // cfg.ppo_mini_batch_size
    print(f"\n训练: {cfg.num_epochs} epochs, "
          f"{num_chains_total} chains/epoch ({cfg.num_prompts_per_epoch} prompts x {cfg.num_chains_per_prompt}), "
          f"{num_updates_per_epoch} PPO updates/epoch")

    epoch_stride = 10000
    chain_stride = 1000

    for epoch in range(1, cfg.num_epochs + 1):
        # ---- 7a. 选取 prompts ----
        selected = random.sample(all_prompts, cfg.num_prompts_per_epoch)

        # ---- 7b. 采样 ----
        all_chain_data = []  # list of dict per chain

        for p_idx, prompt_text in enumerate(selected):
            prompt_embeds, pooled_embeds, neg_embeds, neg_pooled = encode_prompt(
                pipeline, prompt_text, device
            )

            for c_idx in range(cfg.num_chains_per_prompt):
                latents_gen, step_gens = make_chain_generators(
                    cfg.seed + epoch * epoch_stride,
                    p_idx * cfg.num_chains_per_prompt + c_idx,
                    cfg.num_inference_steps,
                    device,
                )
                all_gens = [latents_gen] + step_gens  # gen[0]=x_T, gen[i+1]=step_i

                images, all_latents, log_probs_old, log_probs_base = (
                    pipeline_sd3_train_sample(
                        pipeline,
                        base_transformer,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_embeds,
                        negative_prompt_embeds=neg_embeds,
                        negative_pooled_prompt_embeds=neg_pooled,
                        height=cfg.height, width=cfg.width,
                        num_inference_steps=cfg.num_inference_steps,
                        guidance_scale=cfg.guidance_scale,
                        generator=all_gens,
                    )
                )

                total_lp_base = total_log_prob_from_list(log_probs_base).item()

                all_chain_data.append({
                    "prompt": prompt_text,
                    "image": images[0].detach().cpu(),
                    "all_latents": all_latents,
                    "log_probs_old": log_probs_old,       # list[Tensor[1]]
                    "log_probs_base": log_probs_base,     # list[Tensor[1]]
                    "total_lp_base": total_lp_base,
                    "prompt_embeds": prompt_embeds,
                    "pooled_embeds": pooled_embeds,
                    "neg_embeds": neg_embeds,
                    "neg_pooled": neg_pooled,
                })

        # ---- 7c. r_in = zscore(total_log_p_base) per prompt (组内6条链归一化) ----
        total_lps = torch.tensor([d["total_lp_base"] for d in all_chain_data], device=device)
        r_in = compute_reward_rin(total_lps, cfg.num_chains_per_prompt)
        # Clip advantage
        r_in_clipped = r_in.clamp(-cfg.adv_clip_max, cfg.adv_clip_max)

        # ---- 7d. PPO 更新 ----
        ppo_metrics = defaultdict(list)

        # Shuffle chain indices
        chain_indices = list(range(num_chains_total))
        random.shuffle(chain_indices)

        for update_i in range(num_updates_per_epoch):
            batch_idx = chain_indices[update_i * cfg.ppo_mini_batch_size:
                                      (update_i + 1) * cfg.ppo_mini_batch_size]

            step_losses = []
            step_ratios = []

            for step_t in range(cfg.num_inference_steps):
                batch_log_p_new = []
                batch_log_p_old = []

                for idx in batch_idx:
                    d = all_chain_data[idx]
                    t = pipeline.scheduler.timesteps[step_t]
                    x_t = d["all_latents"][step_t]        # [1, C, H, W]
                    x_tm1 = d["all_latents"][step_t + 1]  # [1, C, H, W]

                    # log_p_new: forward with trainable (LoRA) transformer
                    lp_new = compute_log_prob_at_step(
                        pipeline, x_t, x_tm1, t,
                        d["prompt_embeds"], d["pooled_embeds"],
                        d["neg_embeds"], d["neg_pooled"],
                        cfg.guidance_scale,
                    )
                    batch_log_p_new.append(lp_new.squeeze(0))

                    # log_p_old from sampling
                    batch_log_p_old.append(d["log_probs_old"][step_t + 1].squeeze(0))

                # Stack: [mini_batch_size]
                lp_new = torch.stack(batch_log_p_new)
                lp_old = torch.stack(batch_log_p_old)

                # Ratio
                log_ratio = lp_new - lp_old
                ratio = torch.exp(log_ratio)
                clipped_ratio = ratio.clamp(1.0 - cfg.ppo_clip_range, 1.0 + cfg.ppo_clip_range)

                # Advantage
                adv = torch.stack([r_in_clipped[idx] for idx in batch_idx])

                # PPO clipped objective (negative because we maximize)
                ppo_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

                # Normalize by num_steps for gradient accumulation
                total_loss = ppo_loss / cfg.num_inference_steps

                total_loss.backward()

                step_losses.append(total_loss.item())
                step_ratios.append(ratio.mean().item())

            # Gradient clipping + optimizer step
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipeline.transformer.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

            ppo_metrics["loss"].append(np.mean(step_losses))
            ppo_metrics["ratio"].append(np.mean(step_ratios))

        # ---- 7e. W&B 日志 ----
        log_dict = {
            "epoch": epoch,
            "r_in_mean": r_in.mean().item(),
            "r_in_std": r_in.std().item(),
            "r_in_min": r_in.min().item(),
            "r_in_max": r_in.max().item(),
            "total_lp_mean": total_lps.mean().item(),
            "total_lp_std": total_lps.std().item(),
            "ppo_loss": np.mean(ppo_metrics["loss"]),
            "ratio_mean": np.mean(ppo_metrics["ratio"]),
            "lr": optimizer.param_groups[0]["lr"],
        }

        # r_in histogram every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            log_dict["r_in_hist"] = wandb.Histogram(r_in.tolist())

        # sample images every 10 epochs (grid: 2 prompts x 6 chains = 12 images)
        if epoch % 10 == 0 or epoch == 1:
            from PIL import Image
            thumb = 128
            # Group by prompt
            prompt_imgs = defaultdict(list)
            for d in all_chain_data:
                prompt_imgs[d["prompt"]].append(d["image"])
            for pi, (prompt_text, imgs) in enumerate(prompt_imgs.items()):
                row = []
                for img_tensor in imgs:
                    img = (img_tensor / 2 + 0.5).clamp(0, 1)
                    img_np = (img.permute(1, 2, 0).float().numpy() * 255).round().astype("uint8")
                    row.append(Image.fromarray(img_np).resize((thumb, thumb)))
                grid = Image.new("RGB", (thumb * len(row), thumb))
                for j, im in enumerate(row):
                    grid.paste(im, (j * thumb, 0))
                log_dict[f"images/prompt{pi}_epoch{epoch}"] = wandb.Image(
                    grid, caption=f"epoch {epoch} | {prompt_text[:80]}"
                )

        wandb.log(log_dict)

        # ---- 7f. Checkpoint 保存 (每 50 epoch) ----
        if epoch % 50 == 0:
            ckpt_dir = cfg.work_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "transformer_state_dict": get_peft_model_state_dict(pipeline.transformer),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "random": random.getstate(),
                },
            }, ckpt_path)
            print(f"  checkpoint saved: {ckpt_path}")

        # Console
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{cfg.num_epochs} | "
                  f"r_in: {r_in.mean().item():+.3f} ± {r_in.std().item():.3f} | "
                  f"loss: {np.mean(ppo_metrics['loss']):.4f} | "
                  f"ratio: {np.mean(ppo_metrics['ratio']):.4f}")

    # ================================================================
    print(f"\n{'='*60}")
    print("实验二 Part2 完成!")
    print(f"{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = get_config_from_path(args.config)
    run_training(cfg)
