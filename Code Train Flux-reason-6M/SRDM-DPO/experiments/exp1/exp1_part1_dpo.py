"""Exp1 Part1 DPO: r_in only + Pairwise SDE-DPO — SD3.5 base model.

DPO Loss:
    For each prompt, sample 6 SDE chains → compute r_in (z-score of log_probs_base)
    → sort by reward → 3 pairs: (1st,6th), (2nd,5th), (3rd,4th)
    → DPO loss per pair: -log σ(β * (Δ_θ(τ_w) - Δ_θ(τ_l)))
    → mean over 3 pairs → one backward → one optimizer step

Usage:
    python experiments/exp1/exp1_part1_dpo.py --config config/exp1_part1_dpo_config.py
    python experiments/exp1/exp1_part1_dpo.py --config config/exp1_part1_dpo_config.py --resume 300
"""

import math
import os
import random
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb
from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from srdm_pytorch_exp.dpo_trainer import dpo_update
from srdm_pytorch_exp.reward_rin import zscore_normalize
from srdm_pytorch_exp.sde_sampling import (
    encode_prompt,
    make_chain_generators,
    pipeline_sd3_train_sample,
    total_log_prob_from_list,
)


def tensor_to_pil(img_tensor):
    """Convert a [C, H, W] tensor to PIL Image."""
    from PIL import Image
    import numpy as np
    img = (img_tensor.detach().cpu() / 2 + 0.5).clamp(0, 1)
    img_np = (img.permute(1, 2, 0).float().numpy() * 255).round().astype("uint8")
    return Image.fromarray(img_np)


def get_config_from_path(config_path):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return __import__(config_name).get_config()


def load_prompts_txt(file_path):
    """Load prompts from plain text file, one prompt per line."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _build_model(model_key, model_path, dtype, cfg, device):
    """Load SD3.5 base model, apply LoRA, return state dict."""
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # LoRA
    lora_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=["attn.to_k", "attn.to_q", "attn.to_v",
                        "attn.to_out.0", "attn.add_k_proj", "attn.add_q_proj",
                        "attn.add_v_proj", "attn.to_add_out"],
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config).to(device)

    # SDE scheduler
    orig = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    orig_dict = {k: v for k, v in dict(orig.config).items() if not k.startswith('_')}
    pipe.scheduler = StochasticFlowMatchScheduler(a=cfg.a, **orig_dict)

    # Frozen base transformer (reference model for DPO) — load only transformer, not full pipeline
    from diffusers import SD3Transformer2DModel
    base = SD3Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype,
    ).to(device)
    for p in base.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon,
    )

    n_trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    print(f"  [{model_key}] GPU={device}  LoRA trainable={n_trainable:,}  "
          f"path={os.path.basename(model_path)}")
    return {
        "pipeline": pipe, "base": base, "optimizer": optimizer,
        "device": device,
    }


def _run_model_epoch(state, selected_prompts, epoch, cfg, model_key):
    """Sample (all) → r_in Reward → DPO Update (per prompt)."""

    pipeline = state["pipeline"]
    base_transformer = state["base"]
    optimizer = state["optimizer"]
    device = state["device"]

    N = cfg.num_prompts_per_epoch          # 3
    M = cfg.num_chains_per_prompt          # 6
    n_pairs = cfg.dpo_num_pairs            # 3

    all_chain_data = []
    pipeline.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # ---- Phase 1: Sample all chains ----
    for p_idx, prompt_text in enumerate(selected_prompts):
        prompt_embeds, pooled_embeds, neg_embeds, neg_pooled = encode_prompt(
            pipeline, prompt_text, device)
        prompt_embeds = prompt_embeds.to(device)
        pooled_embeds = pooled_embeds.to(device)
        neg_embeds = neg_embeds.to(device)
        neg_pooled = neg_pooled.to(device)

        p_chains = []
        for c_idx in range(M):
            latents_gen, step_gens = make_chain_generators(
                cfg.seed + epoch * 10000, p_idx * M + c_idx,
                cfg.num_inference_steps, device)
            all_gens = [latents_gen] + step_gens

            images, all_latents, log_probs_old, log_probs_base = pipeline_sd3_train_sample(
                pipeline=pipeline, base_transformer=base_transformer,
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_embeds,
                negative_prompt_embeds=neg_embeds, negative_pooled_prompt_embeds=neg_pooled,
                height=cfg.height, width=cfg.width,
                num_inference_steps=cfg.num_inference_steps, guidance_scale=cfg.guidance_scale,
                generator=all_gens,
            )

            d = {
                "prompt": prompt_text, "prompt_idx": p_idx, "chain_idx": c_idx,
                "image": images[0].detach().cpu(),
                "pil_image": tensor_to_pil(images[0]),
                "all_latents": all_latents,
                "log_probs_old": log_probs_old, "log_probs_base": log_probs_base,
                "total_lp_base": total_log_prob_from_list(log_probs_base).item(),
                "prompt_embeds": prompt_embeds, "pooled_embeds": pooled_embeds,
                "neg_embeds": neg_embeds, "neg_pooled": neg_pooled,
            }
            p_chains.append(d)
            all_chain_data.append(d)

        # ---- r_in reward (z-score across 6 chains) ----
        valid_lp = torch.tensor([d["total_lp_base"] for d in p_chains], device=device)
        r_in = zscore_normalize(valid_lp)
        for j, d in enumerate(p_chains):
            d["r_in"] = r_in[j].item()
            d["reward"] = r_in[j].item()

        # ---- Sort by reward, create pairs ----
        sorted_indices = sorted(range(M), key=lambda i: p_chains[i]["reward"], reverse=True)
        pairs = []
        for k in range(n_pairs):
            w_idx = sorted_indices[k]           # k-th best
            l_idx = sorted_indices[M - 1 - k]   # k-th worst
            # Convert to global index in all_chain_data
            global_w = p_idx * M + w_idx
            global_l = p_idx * M + l_idx
            pairs.append((global_w, global_l))

        # ---- DPO Update ----
        metrics = dpo_update(
            transformer_trainable=pipeline.transformer,
            transformer_ref=base_transformer,
            chain_data=all_chain_data,
            pairs=pairs,
            optimizer=optimizer,
            beta=cfg.dpo_beta,
            num_inference_steps=cfg.num_inference_steps,
            max_grad_norm=cfg.max_grad_norm,
        )
        # Store per-prompt metrics
        p_chains[0]["_dpo_metrics"] = metrics

    # ---- Build aggregate metrics ----
    pfx = lambda name: f"{name}/{model_key}"

    all_rewards = torch.tensor([d.get("reward", 0.0) for d in all_chain_data])
    all_r_in = torch.tensor([d.get("r_in", 0.0) for d in all_chain_data])
    all_log_p_base = torch.tensor([d.get("total_lp_base", 0.0) for d in all_chain_data])

    losses = [d[0].get("_dpo_metrics", {}).get("loss", 0.0) for d in
              [all_chain_data[p_idx * M:p_idx * M + M] for p_idx in range(N)]
              for _ in [0]]
    losses = []
    grad_norms = []
    for p_idx in range(N):
        m = all_chain_data[p_idx * M].get("_dpo_metrics", {})
        if m:
            losses.append(m.get("loss", 0.0))
            grad_norms.append(m.get("grad_norm", 0.0))

    total_chains = N * M
    metrics = {
        pfx("reward_mean"): all_rewards.mean().item(),
        pfx("reward_std"): all_rewards.std().item(),
        pfx("reward_max"): all_rewards.max().item(),
        pfx("reward_min"): all_rewards.min().item(),
        pfx("r_in_mean"): all_r_in.mean().item(),
        pfx("log_p_base_mean"): all_log_p_base.mean().item(),
        pfx("log_p_base_std"): all_log_p_base.std().item(),
        pfx("loss"): np.mean(losses) if losses else 0.0,
        pfx("grad_norm"): np.mean(grad_norms) if grad_norms else 0.0,
        pfx("lr"): optimizer.param_groups[0]["lr"],
        pfx("n_chains"): total_chains,
    }

    return metrics, all_chain_data


def run_training(cfg, resume_epoch=None):
    """Main training loop for Exp1 Part1 DPO.

    Args:
        cfg: config object
        resume_epoch: if not None, load checkpoint from this epoch and continue
    """

    device_str = f"cuda:{cfg.gpu_sd35}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if cfg.mixed_precision == "fp32" else torch.bfloat16

    # ---- Load prompts ----
    prompt_file = cfg.prompt_file
    all_prompts = load_prompts_txt(prompt_file)
    print(f"Loaded {len(all_prompts)} prompts from {prompt_file}")

    # ---- WandB init ----
    run = wandb.init(
        entity="xushuo23-sorbonne-universit-",
        project="SRDM-DPO",
        name=f"{cfg.run_name}_sd35",
        config=cfg.to_dict(),
        dir="/root/autodl-tmp" if os.path.exists("/root/autodl-tmp") else None,
        resume="allow",
    )

    # ---- Build models ----
    model_key = "sd35"
    model_path = cfg.pretrained_model_paths[model_key]
    print(f"\n{'='*50}")
    print(f"  Loading: {model_key} → {device_str}")
    print(f"{'='*50}")
    state = _build_model(model_key, model_path, dtype, cfg, device_str)
    pipeline = state["pipeline"]

    # ---- Resume from checkpoint ----
    checkpoint_dir = os.path.join(cfg.work_dir, f"{model_key}_dpo")
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 1

    if resume_epoch is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{resume_epoch}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"\n  Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device_str)
        missing, unexpected = pipeline.transformer.load_state_dict(
            ckpt["transformer_state_dict"], strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        state["optimizer"].load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        epoch_loss = ckpt.get("loss", "?")
        epoch_r_in = ckpt.get("r_in_mean", "?")
        print(f"  Resumed epoch {ckpt['epoch']} | loss={epoch_loss} | r_in_mean={epoch_r_in}")
        print(f"  Continuing from epoch {start_epoch}")

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        # Select prompts for this epoch
        selected_idx = random.sample(range(len(all_prompts)), cfg.num_prompts_per_epoch)
        selected_prompts = [all_prompts[i] for i in selected_idx]

        # Run epoch (encoding done on-the-fly per prompt)
        metrics, all_chain_data = _run_model_epoch(state, selected_prompts, epoch, cfg, model_key)
        elapsed = time.time() - t0

        # Console
        print(f"\n[Epoch {epoch:3d}/{cfg.num_epochs}] "
              f"loss={metrics.get(f'loss/{model_key}', 0):.4f} | "
              f"r_in_mean={metrics.get(f'r_in_mean/{model_key}', 0):.4f} | "
              f"grad={metrics.get(f'grad_norm/{model_key}', 0):.2f} | "
              f"time={elapsed:.1f}s")

        # ---- WandB logging ----
        pfx = lambda name: f"{name}/{model_key}"
        log_dict = {**metrics, "epoch": epoch}

        # Image tables: only every log_interval epochs (expensive to upload)
        if epoch % cfg.log_interval == 0 or epoch == 1:
            for p_idx in range(cfg.num_prompts_per_epoch):
                p_chains = [d for d in all_chain_data if d["prompt_idx"] == p_idx]
                p_chains_sorted = sorted(p_chains, key=lambda d: d["reward"], reverse=True)

                columns = ["Rank", "Image", "log_p_base", "r_in"]
                rows = []
                for rank, d in enumerate(p_chains_sorted):
                    rows.append([
                        rank + 1,
                        wandb.Image(d["pil_image"]),
                        round(d["total_lp_base"], 2),
                        round(d["r_in"], 3),
                    ])
                table = wandb.Table(columns=columns, data=rows)
                prompt_snip = p_chains[0]["prompt"][:120]
                log_dict[pfx(f"sorted_chains/prompt{p_idx}")] = table
                log_dict[pfx(f"sorted_chains/prompt{p_idx}_caption")] = prompt_snip

        wandb.log(log_dict)

        # Save checkpoint
        if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "transformer_state_dict": get_peft_model_state_dict(pipeline.transformer),
                "optimizer_state_dict": state["optimizer"].state_dict(),
                "config": cfg.to_dict(),
                "loss": metrics.get(f"loss/{model_key}", 0.0),
                "r_in_mean": metrics.get(f"r_in_mean/{model_key}", 0.0),
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    run.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/exp1_part1_dpo_config.py")
    parser.add_argument("--resume", type=int, default=None,
                        help="Resume from epoch N checkpoint")
    args = parser.parse_args()

    cfg = get_config_from_path(args.config)
    run_training(cfg, resume_epoch=args.resume)
