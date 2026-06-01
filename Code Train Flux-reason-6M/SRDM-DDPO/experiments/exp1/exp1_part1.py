"""
实验一 Part1：SDE 噪声系数 a 对 log_prob 和生成图像的影响.

核心研究问题:
    1. log_prob raw 数值随 a 的变化规律:
       - a=0 (确定性 ODE): σ_t=0, log_prob 理论上是多少? 跨链方差是否收敛到 0?
       - a>0 (随机 SDE): 噪声幅度增大后, log_prob 如何变化?
       - 跨链方差: 不同 chain (同 prompt 不同种子) 之间的 log_prob 方差有多大?
    2. a 对生成图像质量的影响:
       - 不同 a 值生成的图像质量是否有肉眼可见差异?
       - a 越大是否引入更多随机性/多样性?

种子方案:
    seed(chain_j, step_i) = base_seed + j * num_steps + i
    - step_i=0: x_T 初始噪声, 预缓存保证不同 a 值相同
    - step_i=1..N: 每 a 值重新创建 Generator(seed), 保证相同随机数
    - 不同 a 值之间同一 (prompt, chain_idx) 的随机数完全一致 → 公平对比

WandB 输出:
    Part A: 每个 prompt 一张最终图像网格 (行=a, 列=chain) → 人眼对比生成效果
    Part B: 每个 (a, prompt) 一张 7 线图 (6 chain + 1 mean) → 观察 raw log_prob 曲线
    Part C: 每个 (a, prompt) 一张 6 线偏差图 (chain - mean) → 观察跨链方差
    Part D: 每个 (a, prompt) 一张 per-step 详细表格 → 精确数值
"""

# 运行方式 (在 SRDM-DDPO/ 目录下):
#     python experiments/exp1/exp1_part1.py --config config/exp1_part1_config.py

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
config_flags.DEFINE_config_file("config", "config/exp1_part1_config.py", "实验一 Part1 配置文件。")

# ---------- matplotlib ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- helper ----------

def _encode_prompt_sd3(pipeline, prompt, device):
    """编码单个 prompt，返回 CFG 所需四元组."""
    (
        prompt_embeds, neg_prompt_embeds,
        pooled_prompt_embeds, neg_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        device=device, num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt="",
    )
    return prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds


def _tensor_to_pil(images):
    """[-1, 1] tensor -> PIL Image 列表."""
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in images]


# ---------- main ----------

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
    print(f"模型路径: {config.pretrained_model_path}")

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
    # 3. 加载 & 编码 prompts
    # ================================================================
    print("\n加载测试 prompts...")
    all_prompts = load_prompts_from_file(config.prompt_file)
    test_prompts = all_prompts[:config.num_test_prompts]
    print(f"测试 prompts (共 {len(test_prompts)} 条):")
    for i, p in enumerate(test_prompts):
        print(f"  [{i}] {p[:100]}...")

    print("\n预编码 prompts...")
    encoded_prompts = []
    for prompt in test_prompts:
        pos_emb, pos_pooled, neg_emb, neg_pooled = _encode_prompt_sd3(
            pipeline, prompt, device
        )
        encoded_prompts.append({
            "prompt": prompt,
            "prompt_embeds": pos_emb,
            "pooled_prompt_embeds": pos_pooled,
            "neg_prompt_embeds": neg_emb,
            "neg_pooled_prompt_embeds": neg_pooled,
        })

    # ================================================================
    # 4. 预计算种子 & 缓存 x_T
    #    seed(chain_j, step_i) = base_seed + j * N + i
    #    step_i=0 -> x_T,  step_i=1..N -> denoising steps
    # ================================================================
    base_seed = config.seed
    num_steps = config.num_inference_steps
    num_chains = config.num_chains_per_prompt
    num_prompts = config.num_test_prompts

    latents_shape = (
        1, pipeline.transformer.config.in_channels,
        config.height // pipeline.vae_scale_factor,
        config.width // pipeline.vae_scale_factor,
    )

    print(f"\n种子方案: base_seed={base_seed}, num_steps={num_steps}, num_chains={num_chains}")
    print(f"缓存 x_T ({num_prompts} prompts x {num_chains} chains)...")

    latents_init_cache = []  # [p_idx][c_idx] = tensor
    for p_idx in range(num_prompts):
        prompt_cache = []
        for c_idx in range(num_chains):
            seed_0 = base_seed + c_idx * num_steps
            gen = torch.Generator(device=device).manual_seed(seed_0)
            xT = torch.randn(latents_shape, generator=gen, device=device, dtype=dtype)
            prompt_cache.append(xT)
        latents_init_cache.append(prompt_cache)

    # ================================================================
    # 5. 对每个 a 值运行采样
    # ================================================================
    results = {}  # results[a_val][p_idx][c_idx] = dict

    for a_val in config.a_values:
        print(f"\n{'='*60}")
        print(f"测试 a = {a_val}")
        print(f"{'='*60}")

        # 创建 scheduler
        orig_dict = dict(original_scheduler.config)
        orig_dict = {k: v for k, v in orig_dict.items() if not k.startswith('_')}
        scheduler = StochasticFlowMatchScheduler(a=a_val, **orig_dict)
        pipeline.scheduler = scheduler

        a_results = []

        for p_idx, enc in enumerate(encoded_prompts):
            chain_results = []

            for c_idx in range(num_chains):
                # 每 a 值重新创建 step generators (相同种子 -> 相同随机数)
                seed_0 = base_seed + c_idx * num_steps
                step_gens = [
                    torch.Generator(device=device).manual_seed(seed_0 + i + 1)
                    for i in range(num_steps)
                ]

                latents_init = latents_init_cache[p_idx][c_idx]

                images, all_latents, all_log_probs = pipeline_sd3_with_logprob(
                    pipeline,
                    prompt_embeds=enc["prompt_embeds"],
                    pooled_prompt_embeds=enc["pooled_prompt_embeds"],
                    negative_prompt_embeds=enc["neg_prompt_embeds"],
                    negative_pooled_prompt_embeds=enc["neg_pooled_prompt_embeds"],
                    height=config.height, width=config.width,
                    num_inference_steps=num_steps,
                    guidance_scale=config.guidance_scale,
                    output_type="pt",
                    latents=latents_init,
                    generator=step_gens,
                )

                has_nan = any(torch.isnan(lat).any() for lat in all_latents)
                has_inf = any(torch.isinf(lat).any() for lat in all_latents)
                log_probs_tensor = torch.stack(all_log_probs)  # [num_steps+1]
                total_log_prob = log_probs_tensor.sum().item()

                print(f"  prompt[{p_idx}] chain[{c_idx}] "
                      f"seed_range=[{seed_0}, {seed_0 + num_steps}]: "
                      f"total_log_prob={total_log_prob:.4f}, "
                      f"NaN={has_nan}, Inf={has_inf}")

                chain_results.append({
                    "image": images[0].detach().cpu(),
                    "all_log_probs": log_probs_tensor.detach().cpu(),
                    "total_log_prob": total_log_prob,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "seed": seed_0,
                })

            # 链条汇总
            total_lps = [r["total_log_prob"] for r in chain_results]
            print(f"  prompt[{p_idx}] 汇总: "
                  f"mean={np.mean(total_lps):.2f} std={np.std(total_lps):.4f} "
                  f"min={np.min(total_lps):.2f} max={np.max(total_lps):.2f}")

            a_results.append(chain_results)

        results[a_val] = a_results

    # ================================================================
    # 6. WandB 日志
    # ================================================================
    print(f"\n{'='*60}")
    print("上传结果到 WandB...")
    print(f"{'='*60}")

    thumb_size = 256
    a_vals = config.a_values
    steps = [-1] + list(range(num_steps))  # 31 points: x_T + N denoising steps

    # -------- Part A: 最终图像网格 (per prompt) --------
    print("\n[A] 上传最终图像网格...")
    for p_idx in range(num_prompts):
        grid_rows = []
        for a_val in a_vals:
            row_imgs = []
            for c_idx in range(num_chains):
                img = results[a_val][p_idx][c_idx]["image"]
                pil = _tensor_to_pil(img.unsqueeze(0))[0].resize((thumb_size, thumb_size))
                row_imgs.append(pil)
            row_pil = Image.new("RGB", (thumb_size * num_chains, thumb_size))
            for j, im in enumerate(row_imgs):
                row_pil.paste(im, (j * thumb_size, 0))
            grid_rows.append(row_pil)

        full_grid = Image.new("RGB", (thumb_size * num_chains, thumb_size * len(a_vals)))
        for i, row in enumerate(grid_rows):
            full_grid.paste(row, (0, i * thumb_size))

        wandb.log({
            f"final_images/prompt_{p_idx}": wandb.Image(
                full_grid,
                caption=(
                    f"Prompt[{p_idx}]: {test_prompts[p_idx][:100]}\n"
                    f"行 = a ({', '.join(str(a) for a in a_vals)}), "
                    f"列 = chain (6 chains, base_seed={base_seed})"
                ),
            )
        })

    # -------- Part B: 7 线图 (6 chain + 1 mean) per (a, prompt) --------
    print("\n[B] 上传 7 线 log_prob 轨迹图...")
    chain_colors = plt.cm.tab10(np.linspace(0, 1, num_chains))

    for a_val in a_vals:
        for p_idx in range(num_prompts):
            # 收集 6 条链的 log_prob: [num_chains, num_steps+1]
            traces = []
            for c_idx in range(num_chains):
                traces.append(results[a_val][p_idx][c_idx]["all_log_probs"].numpy())
            traces = np.stack(traces, axis=0)  # [6, 31]
            mean_trace = traces.mean(axis=0)    # [31]

            fig, ax = plt.subplots(figsize=(14, 7))
            for c_idx in range(num_chains):
                ax.plot(steps, traces[c_idx], color=chain_colors[c_idx],
                        linewidth=1.2, marker=".", markersize=2,
                        label=f"chain {c_idx}")
            ax.plot(steps, mean_trace, color="black", linewidth=2.2,
                    linestyle="--", label="mean", zorder=10)

            ax.set_xlabel("Step (-1 = log p(x_T), 0..N-1 = denoising steps)")
            ax.set_ylabel("log_prob (per-step sum over D dims)")
            ax.set_title(f"Log Prob Traces (a={a_val}, prompt[{p_idx}])")
            ax.legend(fontsize=8, ncol=num_chains + 1, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            wandb.log({f"log_prob_traces/a{a_val}_prompt{p_idx}": wandb.Image(fig)})
            plt.close(fig)

    # -------- Part C: 6 线偏差图 (chain - mean) per (a, prompt) --------
    print("\n[C] 上传偏差图...")
    for a_val in a_vals:
        for p_idx in range(num_prompts):
            traces = []
            for c_idx in range(num_chains):
                traces.append(results[a_val][p_idx][c_idx]["all_log_probs"].numpy())
            traces = np.stack(traces, axis=0)  # [6, 31]
            mean_trace = traces.mean(axis=0)

            fig, ax = plt.subplots(figsize=(14, 7))
            for c_idx in range(num_chains):
                dev = traces[c_idx] - mean_trace
                ax.plot(steps, dev, color=chain_colors[c_idx],
                        linewidth=1.2, marker=".", markersize=2,
                        label=f"chain {c_idx}")

            ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
            ax.set_xlabel("Step (-1 = log p(x_T), 0..N-1 = denoising steps)")
            ax.set_ylabel("log_prob - mean(log_prob)")
            ax.set_title(f"Deviation from Mean (a={a_val}, prompt[{p_idx}])")
            ax.legend(fontsize=8, ncol=num_chains, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            wandb.log({f"deviation/a{a_val}_prompt{p_idx}": wandb.Image(fig)})
            plt.close(fig)

    # -------- Part D: Per-step 详细表格 per (a, prompt) --------
    print("\n[D] 上传 per-step 详细表格...")
    for a_val in a_vals:
        for p_idx in range(num_prompts):
            traces = []
            for c_idx in range(num_chains):
                traces.append(results[a_val][p_idx][c_idx]["all_log_probs"].numpy())
            traces = np.stack(traces, axis=0)  # [6, 31]
            mean_trace = traces.mean(axis=0)

            table_data = []
            for i, s in enumerate(steps):
                row = [int(s)]
                for c_idx in range(num_chains):
                    row.append(round(traces[c_idx, i].item(), 4))
                row.append(round(mean_trace[i].item(), 4))
                table_data.append(row)

            columns = ["step"] + [f"chain_{c}_lp" for c in range(num_chains)] + ["mean_lp"]
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({f"tables/a{a_val}_prompt{p_idx}": table})

    # -------- Part E: x_T log_prob 初始值汇总表 (跨 a 对比) --------
    print("\n[E] 上传 x_T 初始 log_prob 汇总...")
    xT_data = []
    for a_val in a_vals:
        for p_idx in range(num_prompts):
            for c_idx in range(num_chains):
                lp_xT = results[a_val][p_idx][c_idx]["all_log_probs"][0].item()
                xT_data.append([a_val, p_idx, c_idx, round(lp_xT, 4)])
    xT_table = wandb.Table(data=xT_data, columns=["a", "prompt_idx", "chain_idx", "log_p_xT"])
    wandb.log({"xT_log_prob_summary": xT_table})

    # -------- Part F: 数值稳定性 --------
    print("\n[F] 数值稳定性汇总...")
    stability_data = []
    for a_val in a_vals:
        nan_count = inf_count = 0
        total = 0
        for p_idx in range(num_prompts):
            for c_idx in range(num_chains):
                r = results[a_val][p_idx][c_idx]
                total += 1
                if r["has_nan"]:
                    nan_count += 1
                if r["has_inf"]:
                    inf_count += 1
        stability_data.append([a_val, nan_count, inf_count, total])
    stability_table = wandb.Table(
        data=stability_data, columns=["a", "num_NaN", "num_Inf", "total"]
    )
    wandb.log({"numerical_stability": stability_table})

    # -------- Part G: 跨 a 总 log_prob 汇总 --------
    print("\n[G] 跨 a 总 log_prob 汇总...")
    summary_data = []
    for a_val in a_vals:
        all_totals = []
        for p_idx in range(num_prompts):
            for c_idx in range(num_chains):
                all_totals.append(results[a_val][p_idx][c_idx]["total_log_prob"])
        summary_data.append([
            a_val,
            round(np.mean(all_totals), 2),
            round(np.std(all_totals), 4),
            round(np.min(all_totals), 2),
            round(np.max(all_totals), 2),
        ])
    summary_table = wandb.Table(
        data=summary_data,
        columns=["a", "log_prob_mean", "log_prob_std", "log_prob_min", "log_prob_max"],
    )
    wandb.log({"total_log_prob_summary": summary_table})

    # ================================================================
    print(f"\n{'='*60}")
    print("实验一 Part1 完成!")
    print(f"{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
