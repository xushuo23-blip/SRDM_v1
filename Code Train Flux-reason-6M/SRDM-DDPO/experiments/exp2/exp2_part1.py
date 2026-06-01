"""
实验二 Part1：r_in 内生奖励验证.

目标:
    1. 使用 a=0.7 (实验一选定的最优值), 1 prompt x 6 chains
    2. 计算每条链的 total_log_p
    3. r_in = z-score normalize(total_log_p) across 6 chains
    4. 显示 r_in 分布和 log_prob 轨迹

种子方案 (与实验一完全一致):
    seed(chain_j, step_i) = base_seed + j * N + i
    step_i=0: x_T 初始噪声, step_i=1..N: 各步随机噪声

WandB 输出:
    - 最终图像网格 (1 行 x 6 列)
    - 7 线图: 6 chain log_prob + 1 mean
    - r_in 柱状图 + 数值表
    - Per-step 详细表格
"""

# 运行方式 (在 SRDM-DDPO/ 目录下):
#     python experiments/exp2/exp2_part1.py --config config/exp2_part1_config.py

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from absl import app, flags
from ml_collections import config_flags
import numpy as np
import torch
import wandb
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

from prompts import load_prompts_from_file
from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from srdm_pytorch_exp.diffusers_patch.pipeline_sd3_logprob import pipeline_sd3_with_logprob

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/exp2_part1_config.py", "实验二 Part1 配置文件。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _encode_prompt_sd3(pipeline, prompt, device):
    (prompt_embeds, neg_prompt_embeds,
     pooled_prompt_embeds, neg_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt="",
    )
    return prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds


def _tensor_to_pil(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in images]


def main(_):
    config = FLAGS.config

    # ================================================================
    # 1. WandB
    # ================================================================
    wandb.init(
        entity="xushuo23-sorbonne-universit-",
        project="SRDM-DDPO",
        name=config.run_name,
        config=config.to_dict(),
        reinit=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32

    print(f"设备: {device}, 精度: {dtype}")

    # ================================================================
    # 2. 加载 SD3
    # ================================================================
    print("\n加载 SD3 模型...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained_model_path,
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
    pipeline.transformer.requires_grad_(False)
    pipeline.safety_checker = None

    original_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )

    # ================================================================
    # 3. 加载 & 编码 prompt
    # ================================================================
    print("\n加载测试 prompt...")
    all_prompts = load_prompts_from_file(config.prompt_file)
    test_prompt = all_prompts[0]
    print(f"  Prompt: {test_prompt[:100]}...")

    pos_emb, pos_pooled, neg_emb, neg_pooled = _encode_prompt_sd3(
        pipeline, test_prompt, device
    )

    # ================================================================
    # 4. 设置 SDE scheduler (a=0.7)
    # ================================================================
    orig_dict = dict(original_scheduler.config)
    orig_dict = {k: v for k, v in orig_dict.items() if not k.startswith('_')}
    scheduler = StochasticFlowMatchScheduler(a=config.a, **orig_dict)
    pipeline.scheduler = scheduler

    # ================================================================
    # 5. 采样 6 条链
    # ================================================================
    base_seed = config.seed
    num_steps = config.num_inference_steps
    num_chains = config.num_chains_per_prompt

    latents_shape = (
        1, pipeline.transformer.config.in_channels,
        config.height // pipeline.vae_scale_factor,
        config.width // pipeline.vae_scale_factor,
    )

    print(f"\n采样 a={config.a}, {num_chains} 条链, base_seed={base_seed}")
    print(f"链条 seed 范围: {base_seed}--{base_seed + num_chains * num_steps}")

    chain_results = []

    for c_idx in range(num_chains):
        seed_0 = base_seed + c_idx * num_steps
        # x_T
        gen_xT = torch.Generator(device=device).manual_seed(seed_0)
        xT = torch.randn(latents_shape, generator=gen_xT, device=device, dtype=dtype)
        # step generators
        step_gens = [
            torch.Generator(device=device).manual_seed(seed_0 + i + 1)
            for i in range(num_steps)
        ]

        images, all_latents, all_log_probs = pipeline_sd3_with_logprob(
            pipeline,
            prompt_embeds=pos_emb,
            pooled_prompt_embeds=pos_pooled,
            negative_prompt_embeds=neg_emb,
            negative_pooled_prompt_embeds=neg_pooled,
            height=config.height, width=config.width,
            num_inference_steps=num_steps,
            guidance_scale=config.guidance_scale,
            output_type="pt",
            latents=xT,
            generator=step_gens,
        )

        has_nan = any(torch.isnan(lat).any() for lat in all_latents)
        has_inf = any(torch.isinf(lat).any() for lat in all_latents)
        log_probs_tensor = torch.stack(all_log_probs)  # [num_steps+1]
        total_log_prob = log_probs_tensor.sum().item()

        print(f"  chain[{c_idx}] seed_range=[{seed_0}, {seed_0 + num_steps}]: "
              f"total_log_prob={total_log_prob:.4f}, NaN={has_nan}, Inf={has_inf}")

        chain_results.append({
            "image": images[0].detach().cpu(),
            "all_log_probs": log_probs_tensor.detach().cpu(),
            "total_log_prob": total_log_prob,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "seed": seed_0,
        })

    # ================================================================
    # 6. 计算 r_in = z-score normalize(total_log_p) across 6 chains
    # ================================================================
    total_log_ps = torch.tensor([r["total_log_prob"] for r in chain_results])
    mean_lp = total_log_ps.mean().item()
    std_lp = total_log_ps.std().item()

    if std_lp > 0:
        r_in = ((total_log_ps - mean_lp) / std_lp).tolist()
    else:
        r_in = [0.0] * num_chains

    print(f"\nr_in 计算:")
    print(f"  total_log_p: mean={mean_lp:.4f}, std={std_lp:.4f}")
    for c_idx in range(num_chains):
        print(f"  chain[{c_idx}]: total_log_p={total_log_ps[c_idx].item():.4f}, r_in={r_in[c_idx]:.4f}")

    # ================================================================
    # 7. WandB 输出
    # ================================================================
    print("\n上传结果到 WandB...")

    thumb_size = 256
    a_val = config.a
    steps = [-1] + list(range(num_steps))

    # -------- A: 最终图像网格 (1 行 x 6 列) --------
    row_imgs = [_tensor_to_pil(r["image"].unsqueeze(0))[0].resize((thumb_size, thumb_size))
               for r in chain_results]
    img_grid = Image.new("RGB", (thumb_size * num_chains, thumb_size))
    for j, im in enumerate(row_imgs):
        img_grid.paste(im, (j * thumb_size, 0))
    wandb.log({"final_images": wandb.Image(img_grid, caption=f"a={a_val}, 6 chains")})

    # -------- B: 7 线图 (6 chain + mean) --------
    traces = np.stack([r["all_log_probs"].numpy() for r in chain_results], axis=0)  # [6, 31]
    mean_trace = traces.mean(axis=0)
    chain_colors = plt.cm.tab10(np.linspace(0, 1, num_chains))

    fig, ax = plt.subplots(figsize=(14, 7))
    for c_idx in range(num_chains):
        ax.plot(steps, traces[c_idx], color=chain_colors[c_idx],
                linewidth=1.2, marker=".", markersize=2, label=f"chain {c_idx}")
    ax.plot(steps, mean_trace, color="black", linewidth=2.2,
            linestyle="--", label="mean", zorder=10)
    ax.set_xlabel("Step (-1 = log p(x_T), 0..N-1 = denoising steps)")
    ax.set_ylabel("log_prob (per-step sum over D dims)")
    ax.set_title(f"Log Prob Traces (a={a_val})")
    ax.legend(fontsize=8, ncol=num_chains + 1, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    wandb.log({"log_prob_traces": wandb.Image(fig)})
    plt.close(fig)

    # -------- C: r_in 柱状图 --------
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(num_chains), r_in, color=chain_colors, edgecolor="white")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    for c_idx, (val, bar) in enumerate(zip(r_in, bars)):
        y_pos = val + 0.02 if val >= 0 else val - 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    ax.set_xlabel("Chain Index")
    ax.set_ylabel("r_in (z-score)")
    ax.set_title(f"r_in = zscore(total_log_p) across 6 chains (a={a_val})")
    ax.set_xticks(range(num_chains))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    wandb.log({"r_in_bar_chart": wandb.Image(fig)})
    plt.close(fig)

    # -------- D: r_in + total_log_p 汇总表 --------
    r_in_table = wandb.Table(
        data=[[c_idx, r["seed"],
               round(r["total_log_prob"], 4),
               round(r_in[c_idx], 4),
               r["has_nan"], r["has_inf"]]
              for c_idx, r in enumerate(chain_results)],
        columns=["chain", "seed", "total_log_p", "r_in", "NaN", "Inf"],
    )
    wandb.log({"r_in_summary": r_in_table})

    # -------- E: Per-step 详细表格 --------
    table_data = []
    for i, s in enumerate(steps):
        row = [int(s)]
        for c_idx in range(num_chains):
            row.append(round(traces[c_idx, i].item(), 4))
        row.append(round(mean_trace[i].item(), 4))
        table_data.append(row)
    columns = ["step"] + [f"chain_{c}_lp" for c in range(num_chains)] + ["mean_lp"]
    wandb.log({"per_step_table": wandb.Table(data=table_data, columns=columns)})

    # -------- F: 数值稳定性 --------
    nan_count = sum(1 for r in chain_results if r["has_nan"])
    inf_count = sum(1 for r in chain_results if r["has_inf"])
    wandb.log({"numerical_stability": wandb.Table(
        data=[[a_val, nan_count, inf_count, num_chains]],
        columns=["a", "num_NaN", "num_Inf", "total"],
    )})

    print(f"\n{'='*60}")
    print("实验二 Part1 完成!")
    print(f"{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
