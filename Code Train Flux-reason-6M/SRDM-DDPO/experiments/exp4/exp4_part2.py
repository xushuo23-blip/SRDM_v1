"""
实验四 Part 2：DDPO 训练 (r_SSR) — fp32 + VLM 压缩模式 — SD3 vs SD3.5 同时对比.

与 Part 1 的关键区别:
    - fp32 精度: ratio 严格 = 1.0, 无 clip 误杀, 训练稳定但慢 ~2x
    - VLM: thinking=enabled + 图片压缩 (max 256px), 不用 no_thinking
      - fp32 生成慢 → VLM 有充足时间完成 thinking → 解析精度更高
      - 图片压缩减小 vision token 数量 → 适度加速

GPU 策略: 每个 GPU 跑一个完整模型, 两模型并行 → 双倍吞吐.
  - GPU {gpu_sd3}:  SD3 (LoRA transformer + frozen base + VAE + text encoders)
  - GPU {gpu_sd35}: SD3.5 (同上)

用法:
    python experiments/exp4/exp4_part2.py --config config/exp4_part2_config.py
"""

import copy, os, random, sys, time
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srdm_pytorch_exp.sde_sampling import (
    encode_prompt, make_chain_generators, pipeline_sd3_train_sample,
    total_log_prob_from_list, zscore_normalize,
)
from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch
from prompts import load_prompts_from_file
from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch, make_distance_plot
from srdm_pytorch_exp.structure_features import phi_dicts_simplified
from vlm_client import VLMClient, draw_structure_annotations, validate_structure_bboxes


def get_config_from_path(config_path):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return __import__(config_name).get_config()


def tensor_to_pil(img_tensor):
    img = (img_tensor.detach().cpu() / 2 + 0.5).clamp(0, 1)
    img_np = (img.permute(1, 2, 0).float().numpy() * 255).round().astype("uint8")
    return Image.fromarray(img_np)


def _get_font(size: int = 11):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_feature_image(pil_image, structure, phi_dict, distances, chain_idx, total_d,
                       lambda_count, lambda_coverage, lambda_relation,
                       bbox_width=256, panel_width=320):
    """Side-by-side: bbox annotated image + phi feature values panel. From exp3_part3."""
    w_img, h_img = bbox_width, bbox_width
    pad = 4
    total_w = w_img + pad + panel_width

    canvas = Image.new("RGB", (total_w, h_img), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(11)
    font_title = _get_font(13)

    ann = draw_structure_annotations(pil_image.copy(), structure, line_width=2, font_size=12)
    ann = ann.resize((w_img, h_img))
    canvas.paste(ann, (0, 0))

    count_vec = phi_dict.get("count", torch.tensor([]))
    cov_val = phi_dict.get("coverage", torch.tensor([])).item() if phi_dict.get("coverage", torch.tensor([])).numel() > 0 else 0.0
    rel_vec = phi_dict.get("relation", torch.tensor([]))

    d_count = distances.get("d_count", [0.0])[chain_idx]
    d_cov = distances.get("d_coverage", [0.0])[chain_idx]
    d_rel = distances.get("d_relation", [0.0])[chain_idx]
    d_count_n = distances.get("d_count_norm", [0.0])[chain_idx]
    d_cov_n = distances.get("d_coverage_norm", [0.0])[chain_idx]
    d_rel_n = distances.get("d_relation_norm", [0.0])[chain_idx]

    px = w_img + pad + 8
    py = 6
    line_h = 16

    draw.text((px, py), f"Chain {chain_idx}", fill=(30, 30, 30), font=font_title)
    py += 20

    color = (0, 130, 0) if total_d < 0.5 else (180, 80, 0) if total_d < 1.0 else (180, 0, 0)
    draw.text((px, py), f"d = {total_d:.3f}", fill=color, font=font_title)
    py += line_h + 4

    draw.line([(px, py), (px + panel_width - 24, py)], fill=(200, 200, 200))
    py += 6

    count_str = ", ".join(f"{v:.0f}" for v in count_vec.tolist()) if count_vec.numel() > 0 else "-"
    draw.text((px, py), f"phi_count: [{count_str}]", fill=(60, 60, 60), font=font)
    py += line_h
    draw.text((px + 8, py), f"d_raw={d_count:.3f}  d_norm={d_count_n:.3f}  lambda={lambda_count}", fill=(120, 120, 120), font=font)
    py += line_h + 2

    draw.text((px, py), f"phi_cov: {cov_val:.3f}", fill=(60, 60, 60), font=font)
    py += line_h
    draw.text((px + 8, py), f"d_raw={d_cov:.3f}  d_norm={d_cov_n:.3f}  lambda={lambda_coverage}", fill=(120, 120, 120), font=font)
    py += line_h + 2

    rel_str = ", ".join(f"{v:.0f}" for v in rel_vec.tolist()) if rel_vec.numel() > 0 else "-"
    draw.text((px, py), f"phi_rel: [{rel_str}]", fill=(60, 60, 60), font=font)
    py += line_h
    draw.text((px + 8, py), f"d_raw={d_rel:.3f}  d_norm={d_rel_n:.3f}  lambda={lambda_relation}", fill=(120, 120, 120), font=font)
    py += line_h + 6

    draw.line([(px, py), (px + panel_width - 24, py)], fill=(200, 200, 200))
    py += 6
    draw.text((px, py), "phi* = uniform avg (1/M)", fill=(100, 100, 100), font=font)

    return canvas


# ============================================================
# Model loading (每个模型独享一个 GPU)
# ============================================================

def _load_pipeline(model_path, dtype, device):
    """Load one SD3/SD3.5 pipeline on the given device."""
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=True)
    pipe.safety_checker = None
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    for attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
        if hasattr(pipe, attr):
            getattr(pipe, attr).requires_grad_(False)
    return pipe


def _build_model_state(model_key, model_path, dtype, cfg, device):
    """Load pipeline + create LoRA / frozen base / optimizer / alerter."""
    pipe = _load_pipeline(model_path, dtype, device)

    # Frozen base model (same GPU)
    base = copy.deepcopy(pipe.transformer)
    base.requires_grad_(False).eval().to(device)

    # LoRA
    lora_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
        lora_dropout=0.0, bias="none")
    pipe.transformer.requires_grad_(False)
    pipe.transformer = get_peft_model(pipe.transformer, lora_config).to(device)
    n_trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)

    # Scheduler
    orig = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    orig_dict = {k: v for k, v in dict(orig.config).items() if not k.startswith('_')}
    pipe.scheduler = StochasticFlowMatchScheduler(a=cfg.a, **orig_dict)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon)

    # Alerter
    alerter = TrainingAlerter(
        window=cfg.alert_window, threshold=cfg.alert_threshold,
        ratio_bad_pct=cfg.alert_ratio_bad_pct)

    print(f"  [{model_key}] GPU={device}  LoRA trainable={n_trainable:,}  "
          f"path={os.path.basename(model_path)}")
    return {
        "pipeline": pipe, "base": base, "optimizer": optimizer, "alerter": alerter,
        "device": device,
    }


# ============================================================
# Per-model epoch (单 GPU, 无跨 GPU 传输)
# ============================================================

def _run_model_epoch(state, selected_prompts, prompt_encodings, epoch, cfg,
                     vlm_client, model_key):
    """Sample → VLM (thinking=enabled + 压缩) → r_SSR → PPO. All on state['device']."""

    pipeline = state["pipeline"]
    base_transformer = state["base"]
    optimizer = state["optimizer"]
    alerter = state["alerter"]
    device = state["device"]

    N = cfg.num_prompts_per_epoch          # 3
    M = cfg.num_chains_per_prompt          # 8
    B = cfg.ppo_mini_batch_size            # 4
    updates_per_prompt = M // B            # 2

    all_chain_data = []
    vlm_futures = []

    def _vlm_for_prompt(p_chains, prompt_text):
        """VLM extraction: thinking=enabled + 图片压缩 (max_image_size).

        fp32 生成慢 → VLM 有充足时间, 不需要 no_thinking 极端加速。
        图片压缩到 256px 减少 vision token, 适度加速 API 响应。
        """
        t0 = time.time()
        schema = vlm_client.extract_schema(prompt_text)
        structures = vlm_client.extract_structures_batch(
            [d["pil_image"] for d in p_chains], schema,
            original_prompt=prompt_text, max_workers=cfg.vlm_max_workers,
            stagger_delay=cfg.vlm_stagger_delay, max_image_size=cfg.vlm_max_image_size)
        return schema, structures, time.time() - t0

    # ---- 采样 + VLM 流水线 ----
    with ThreadPoolExecutor(max_workers=N) as vlm_pool:
        for p_idx, prompt_text in enumerate(selected_prompts):
            prompt_embeds, pooled_embeds, neg_embeds, neg_pooled = prompt_encodings[p_idx]
            # Move embeddings to this model's GPU
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

            vlm_futures.append((p_idx, vlm_pool.submit(_vlm_for_prompt, p_chains, prompt_text), p_chains))

    # ---- 收集 VLM + r_SSR ----
    vlm_elapsed = 0.0
    all_advantages = torch.zeros(N * M, device=device)
    json_bad_total = 0
    batch_skip_total = 0

    for p_idx, future, p_chains in vlm_futures:
        schema, structures, elapsed = future.result()
        vlm_elapsed += elapsed

        for d, s in zip(p_chains, structures):
            d["schema"] = schema; d["structure"] = s
            d["json_ok"] = not (
                "_error" in s
                or not isinstance(s.get("objects"), list)
                or len(s.get("objects", [])) == 0
            )
            if d["json_ok"]:
                d["json_ok"] = validate_structure_bboxes(s)
                if not d["json_ok"]:
                    print(f"  WARNING chain {d['chain_idx']}: bbox validation failed — "
                          f"rejecting chain (VLM returned malformed bbox data)")

        valid_chains = [d for d in p_chains if d["json_ok"]]
        bad_chains = [d for d in p_chains if not d["json_ok"]]
        json_bad_total += len(bad_chains)

        if len(valid_chains) >= 2:
            valid_structures = [d["structure"] for d in valid_chains]
            phi_dicts, active_nouns, top2, dead_nouns = phi_dicts_simplified(valid_structures, schema)
            valid_lp = torch.tensor([d["total_lp_base"] for d in valid_chains], device=device)
            r_ssr_valid, debug_info = compute_r_ssr_batch(
                phi_dicts, valid_lp, lambda_count=cfg.lambda_count,
                lambda_coverage=cfg.lambda_coverage, lambda_relation=cfg.lambda_relation,
                uniform_weights=cfg.phi_uniform_weights)

            adv_valid = zscore_normalize(r_ssr_valid.to(device))
            for j, d in enumerate(valid_chains):
                d["r_ssr"] = r_ssr_valid[j].item()
                d["advantage"] = adv_valid[j].item()
                d["phi_dict"] = phi_dicts[j]
                d["d_combined"] = debug_info["d_combined"][j].item()

            min_adv = adv_valid.min().item()
            for d in bad_chains:
                d["r_ssr"] = min_adv
                d["advantage"] = min_adv
                d["phi_dict"] = {}
                d["d_combined"] = 999.0

            debug_info_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in debug_info.items()}
            p_chains[0]["_debug_info"] = debug_info_cpu
            p_chains[0]["_valid_indices"] = [j for j, d in enumerate(p_chains) if d["json_ok"]]
        else:
            for d in p_chains:
                d["r_ssr"] = 0.0
                d["advantage"] = 0.0
                d["phi_dict"] = {}
                d["d_combined"] = 999.0
            p_chains[0]["_debug_info"] = None
            p_chains[0]["_valid_indices"] = []

        start, end = p_idx * M, (p_idx + 1) * M
        all_advantages[start:end] = torch.tensor([d["advantage"] for d in p_chains], device=device)

        for d in p_chains:
            d.setdefault("active_nouns", [])
            d.setdefault("dead_nouns", [])

    # ---- PPO (per prompt batch, 过滤坏链) ----
    timesteps = pipeline.scheduler.timesteps
    ppo_metrics = defaultdict(list)
    ppo_total = N * updates_per_prompt
    ppo_pbar = tqdm(total=ppo_total, desc=f"      PPO {model_key}", leave=False)
    for p_idx in range(N):
        p_indices = list(range(p_idx * M, (p_idx + 1) * M))
        random.shuffle(p_indices)
        for u in range(updates_per_prompt):
            ppo_pbar.update(1)
            raw_batch = p_indices[u * B:(u + 1) * B]

            m = ppo_update_mini_batch(
                pipeline, all_chain_data, raw_batch, timesteps,
                advantages=all_advantages, guidance_scale=cfg.guidance_scale,
                optimizer=optimizer, ppo_clip_range=cfg.ppo_clip_range,
                max_grad_norm=cfg.max_grad_norm,
                num_inference_steps=cfg.num_inference_steps, alerter=alerter)
            if m.get("batch_skipped"):
                batch_skip_total += 1
                ppo_metrics["json_batch_skipped"].append(1)
                continue

            ppo_metrics["json_batch_skipped"].append(0)
            for k, v in m.items():
                ppo_metrics[k].append(v)
    ppo_pbar.close()

    # ---- Build metrics dict with model_key suffix ----
    all_r_ssr = torch.tensor([d.get("r_ssr", 0.0) for d in all_chain_data])
    pfx = lambda name: f"{name}/{model_key}"

    # Per-component raw distances (collected across all valid chains from all prompts)
    all_d_count_raw = []
    all_d_cov_raw = []
    all_d_rel_raw = []
    all_d_combined_vals = []
    for p_idx in range(N):
        p_chains = [d for d in all_chain_data if d["prompt_idx"] == p_idx]
        debug_p = p_chains[0].get("_debug_info")
        if debug_p is not None:
            v_idx = p_chains[0].get("_valid_indices", list(range(len(p_chains))))
            for j in v_idx:
                all_d_count_raw.append(debug_p["d_count"][j].item())
                if debug_p.get("has_coverage", False):
                    all_d_cov_raw.append(debug_p["d_coverage"][j].item())
                if debug_p.get("has_relation", False):
                    all_d_rel_raw.append(debug_p["d_relation"][j].item())
                all_d_combined_vals.append(debug_p["d_combined"][j].item())

    metrics = {
        pfx("advantage_mean"): all_advantages.mean().item(),
        pfx("advantage_std"): all_advantages.std().item(),
        pfx("r_ssr_mean"): all_r_ssr.mean().item(),
        pfx("r_ssr_std"): all_r_ssr.std().item(),
        pfx("r_ssr_max"): all_r_ssr.max().item(),
        pfx("r_ssr_p50"): torch.quantile(all_r_ssr.float(), 0.5).item(),
        pfx("r_ssr_p90"): torch.quantile(all_r_ssr.float(), 0.9).item(),
        pfx("vlm_elapsed"): vlm_elapsed,
        pfx("json_bad_total"): json_bad_total,
        pfx("json_batch_skipped"): batch_skip_total,
        pfx("lr"): optimizer.param_groups[0]["lr"],
    }
    for k in ["loss", "ratio_mean", "ratio_clip_rate", "grad_norm"]:
        if k in ppo_metrics:
            metrics[pfx(f"ppo/{k}")] = np.mean(ppo_metrics[k])
    for k in ["alert/ratio_fired", "alert/grad_fired", "alert/any_fired"]:
        if k in ppo_metrics:
            metrics[pfx(k)] = ppo_metrics[k][-1]

    # Per-component mean raw distances (lower = chains agree more on structure)
    if all_d_count_raw:
        metrics[pfx("comp/d_count_raw")] = np.mean(all_d_count_raw)
    if all_d_cov_raw:
        metrics[pfx("comp/d_coverage_raw")] = np.mean(all_d_cov_raw)
    if all_d_rel_raw:
        metrics[pfx("comp/d_relation_raw")] = np.mean(all_d_rel_raw)

    # ---- Images (每 epoch 生成, 检测 VLM 工作状态) ----
    images_dict = {}

    # 缩略图网格 (每个 prompt 8 条链)
    thumb = 128
    prompt_imgs = defaultdict(list)
    for d in all_chain_data:
        prompt_imgs[d["prompt"]].append(d["image"])
    for pi, (pt, imgs) in enumerate(prompt_imgs.items()):
        row = [Image.fromarray(
            ((t / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).float().numpy() * 255
             ).round().astype("uint8")).resize((thumb, thumb)) for t in imgs]
        grid = Image.new("RGB", (thumb * len(row), thumb))
        for j, im in enumerate(row):
            grid.paste(im, (j * thumb, 0))
        images_dict[pfx(f"images/prompt{pi}")] = wandb.Image(
            grid, caption=f"[{model_key}] e{epoch} | {pt[:80]}")

    # r_SSR 直方图
    images_dict[pfx("r_ssr_hist")] = wandb.Histogram(all_r_ssr.tolist())

    # Per-prompt: feature images + distance scatter + best/worst bbox
    for p_idx in range(N):
        p_chains = [d for d in all_chain_data if d["prompt_idx"] == p_idx]
        valid_p = [d for d in p_chains if d.get("json_ok")]

        # --- Feature images ---
        feat_imgs = []
        for j, d in enumerate(p_chains[:M]):
            if d.get("json_ok") and d.get("phi_dict"):
                debug_info_p = p_chains[0].get("_debug_info")
                if debug_info_p is not None:
                    distances = {
                        "d_count": debug_info_p["d_count"].tolist(),
                        "d_coverage": debug_info_p["d_coverage"].tolist(),
                        "d_relation": debug_info_p["d_relation"].tolist(),
                        "d_count_norm": debug_info_p["d_count_norm"].tolist(),
                        "d_coverage_norm": debug_info_p["d_coverage_norm"].tolist(),
                        "d_relation_norm": debug_info_p["d_relation_norm"].tolist(),
                    }
                    v_idx = p_chains[0].get("_valid_indices", list(range(len(p_chains))))
                    dist_idx = v_idx.index(j) if j in v_idx else 0
                    feat_img = make_feature_image(
                        d["pil_image"], d["structure"], d["phi_dict"], distances, dist_idx,
                        d.get("d_combined", 999.0), cfg.lambda_count, cfg.lambda_coverage,
                        cfg.lambda_relation)
                    feat_imgs.append(feat_img)
        if feat_imgs:
            fw, fh = feat_imgs[0].size
            cols = min(3, len(feat_imgs))
            rows_f = (len(feat_imgs) + cols - 1) // cols
            grid_f = Image.new("RGB", (fw * cols, fh * rows_f), (245, 245, 245))
            for i, img in enumerate(feat_imgs):
                r, c = divmod(i, cols)
                grid_f.paste(img, (c * fw, r * fh))
            images_dict[pfx(f"features/prompt{p_idx}")] = wandb.Image(
                grid_f, caption=f"[{model_key}] e{epoch} p{p_idx} phi features")

        # --- Distance scatter plot ---
        debug_info_p = p_chains[0].get("_debug_info")
        if debug_info_p is not None and len(valid_p) >= 2:
            try:
                v_indices = p_chains[0].get("_valid_indices", list(range(len(p_chains))))
                dist_plot = make_distance_plot(
                    debug_info_p, chain_indices=v_indices,
                    variant_label=f"[{model_key}] e{epoch}",
                    prompt_short=p_chains[0]["prompt"])
                images_dict[pfx(f"distance/prompt{p_idx}")] = wandb.Image(
                    dist_plot, caption=f"[{model_key}] e{epoch} p{p_idx} PCA distances")
            except Exception as e:
                print(f"  [{model_key}] distance plot error p{p_idx}: {e}")

        # --- Best/worst chain bbox ---
        if len(valid_p) >= 2:
            p_sorted = sorted(valid_p, key=lambda d: d.get("r_ssr", -999))
            for label, chain in [("best", p_sorted[-1]), ("worst", p_sorted[0])]:
                try:
                    pil_ann = draw_structure_annotations(chain["pil_image"].copy(), chain["structure"])
                    obj_info = ",".join(f"{o.get('label','?')}:{o.get('count',0)}"
                                        for o in chain["structure"].get("objects", [])[:5])
                    caption = f"[{model_key}] e{epoch} p{p_idx} {label} | r_SSR={chain.get('r_ssr',0):.3f} | {obj_info}"
                    images_dict[pfx(f"bbox/prompt{p_idx}_{label}")] = wandb.Image(pil_ann, caption=caption)
                except Exception as e:
                    print(f"  [{model_key}] bbox draw error p{p_idx} {label}: {e}")

    # ---- Console data ----
    grad_norms = ppo_metrics.get("grad_norm", [0])
    lr = optimizer.param_groups[0]["lr"]
    console = {
        "adv_mean": all_advantages.mean().item(),
        "adv_std": all_advantages.std().item(),
        "loss": np.mean(ppo_metrics.get("loss", [0])),
        "ratio_mean": np.mean(ppo_metrics.get("ratio_mean", [0])),
        "clip_rate": np.mean(ppo_metrics.get("ratio_clip_rate", [0])),
        "grad_norm": np.mean(grad_norms),
        "lr_step": lr * np.mean(grad_norms),
        "vlm_elapsed": vlm_elapsed,
        "json_bad": json_bad_total,
        "batch_skip": batch_skip_total,
    }

    return metrics, images_dict, console


# ============================================================
# Main
# ============================================================

def _run_model_thread(model_key, state, selected_prompts, prompt_encodings, epoch, cfg, vlm_client):
    """Thread wrapper: set CUDA device, then run epoch."""
    device = state["device"]
    with torch.cuda.device(device):
        return _run_model_epoch(state, selected_prompts, prompt_encodings, epoch, cfg, vlm_client, model_key)


def run_training(cfg, resume_epoch=0):
    # ---- Per-model devices ----
    gpu_map = {"sd3": getattr(cfg, "gpu_sd3", 0), "sd35": getattr(cfg, "gpu_sd35", 1)}

    if cfg.mixed_precision == "fp16":
        dtype = torch.float16
    elif cfg.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    # ---- WandB ----
    model_tag = "+".join(cfg.model_ids)
    run_name = f"{cfg.run_name}_{model_tag}_{cfg.reward_mode}"
    if resume_epoch > 0:
        run_name += f"_resume{resume_epoch}"
    wandb.init(entity="xushuo23-sorbonne-universit-", project="SRDM-DDPO",
               name=run_name, config=cfg.to_dict(), reinit=True)

    # ---- Load models (each on its own GPU, no sharing) ----
    model_states = {}
    for mk in cfg.model_ids:
        path = cfg.pretrained_model_paths.get(mk)
        if path is None or "REPLACE" in path:
            print(f"  [{mk}] SKIP: model path not configured (REPLACE placeholder)")
            continue
        device = torch.device(f"cuda:{gpu_map[mk]}")
        print(f"加载模型: {mk} → GPU {gpu_map[mk]} | {path}")
        model_states[mk] = _build_model_state(mk, path, dtype, cfg, device)
    print(f"已加载 {len(model_states)} 个模型: {list(model_states.keys())}")

    if not model_states:
        raise RuntimeError("没有成功加载任何模型, 请检查 pretrained_model_paths 配置.")

    # ---- Shared: VLM + Prompts ----
    vlm_client = VLMClient(backend=cfg.vlm_backend, model=cfg.vlm_model,
                           max_retries=cfg.vlm_max_retries)
    all_prompts = load_prompts_from_file(cfg.prompt_file)
    vlm_mode = "compressed" if cfg.vlm_max_image_size <= 256 else "thinking"
    print(f"VLM: {cfg.vlm_backend} / {vlm_mode} (max={cfg.vlm_max_image_size}px, thinking=enabled)")
    print(f"Prompts: {len(all_prompts)}")

    N = cfg.num_prompts_per_epoch
    M = cfg.num_chains_per_prompt
    B = cfg.ppo_mini_batch_size
    updates_per_epoch = N * (M // B)

    # ---- Resume from checkpoint ----
    start_epoch = 1
    if resume_epoch > 0:
        for model_key, state in model_states.items():
            ckpt_dir = os.path.join(cfg.work_dir, f"{model_key}_{cfg.reward_mode}")
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{resume_epoch}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=state["device"])
            missing, unexpected = state["pipeline"].transformer.load_state_dict(
                ckpt["transformer_state_dict"], strict=False)
            state["optimizer"].load_state_dict(ckpt["optimizer_state_dict"])
            print(f"  [{model_key}] loaded checkpoint: epoch {ckpt['epoch']} from {ckpt_path}")
            if missing:
                print(f"  [{model_key}]   missing keys: {len(missing)} (expected: LoRA prefix mismatch)")
            if unexpected:
                print(f"  [{model_key}]   unexpected keys: {len(unexpected)}")
        first_mk = cfg.model_ids[0]
        first_ckpt_dir = os.path.join(cfg.work_dir, f"{first_mk}_{cfg.reward_mode}")
        first_ckpt = torch.load(os.path.join(first_ckpt_dir, f"epoch_{resume_epoch}.pt"), map_location="cpu")
        torch.set_rng_state(first_ckpt["rng_state"]["torch"])
        np.random.set_state(first_ckpt["rng_state"]["numpy"])
        random.setstate(first_ckpt["rng_state"]["random"])
        start_epoch = resume_epoch + 1
        print(f"\n=== 从 epoch {start_epoch} 继续训练 (已恢复 checkpoint epoch_{resume_epoch}.pt) ===\n")

    remaining = cfg.num_epochs - start_epoch + 1
    print(f"\n训练: {cfg.num_epochs} epochs (剩余 {remaining}), {N}×{M}={N*M} chains/epoch/model, "
          f"{updates_per_epoch} PPO updates/epoch/model")
    print(f"精度: {cfg.mixed_precision} | 模型: {list(model_states.keys())} | 奖励: {cfg.reward_mode}")
    print(f"GPU 分配: sd3=cuda:{gpu_map.get('sd3','?')}, sd35=cuda:{gpu_map.get('sd35','?')} (并行)")

    # ---- Training loop ----
    history = defaultdict(lambda: defaultdict(list))

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        selected = random.sample(all_prompts, N)

        first_model = cfg.model_ids[0]
        encode_device = model_states[first_model]["device"]
        prompt_encodings_cpu = []
        for p in selected:
            pe, po, ne, np_ = encode_prompt(model_states[first_model]["pipeline"], p, encode_device)
            prompt_encodings_cpu.append((pe.cpu(), po.cpu(), ne.cpu(), np_.cpu()))

        all_metrics = {"epoch": epoch}
        all_images = {}

        with ThreadPoolExecutor(max_workers=len(model_states)) as ex:
            futures = {}
            for model_key, state in model_states.items():
                futures[ex.submit(
                    _run_model_thread, model_key, state,
                    selected, prompt_encodings_cpu, epoch, cfg, vlm_client
                )] = model_key

            for future in as_completed(futures):
                model_key = futures[future]
                metrics, images, console = future.result()
                all_metrics.update(metrics)
                all_images.update(images)

                for k in ["r_ssr_mean", "r_ssr_std", "r_ssr_max", "r_ssr_p50", "r_ssr_p90",
                          "advantage_mean", "advantage_std",
                          "comp/d_count_raw", "comp/d_coverage_raw", "comp/d_relation_raw",
                          "ppo/loss", "ppo/ratio_mean", "ppo/ratio_clip_rate", "ppo/grad_norm",
                          "vlm_elapsed", "json_bad_total", "json_batch_skipped",
                          "alert/ratio_fired", "alert/grad_fired", "alert/any_fired"]:
                    metric_key = f"{k}/{model_key}"
                    if metric_key in metrics:
                        history[k][model_key].append(metrics[metric_key])

                alert_keys = []
                pfx = lambda name: f"{name}/{model_key}"
                for k in ["alert/ratio_fired", "alert/grad_fired"]:
                    if metrics.get(pfx(k)):
                        alert_keys.append(k.split("/")[-1].replace("_fired", ""))
                alerts_str = "".join(f" [{a.upper()}]" for a in alert_keys)
                json_str = f" | JSON bad:{console['json_bad']} skip:{console['batch_skip']}" if console["json_bad"] > 0 else ""
                print(f"[{model_key}] e{epoch:3d}/{cfg.num_epochs} | "
                      f"adv: {console['adv_mean']:+.3f}±{console['adv_std']:.3f} | "
                      f"loss: {console['loss']:.4f} | "
                      f"ratio: {console['ratio_mean']:.2f} | clip: {console['clip_rate']:.0%} | "
                      f"|grad|: {console['grad_norm']:.2f} | step: {console['lr_step']:.2e} | "
                      f"VLM: {console['vlm_elapsed']:.1f}s{alerts_str}{json_str}")

        all_metrics.update(all_images)

        # Combined charts
        model_keys = list(model_states.keys())
        epochs_list = list(range(start_epoch, epoch + 1))
        for metric_base in ["r_ssr_mean", "r_ssr_p50", "ppo/loss", "ppo/ratio_mean",
                            "ppo/ratio_clip_rate", "ppo/grad_norm",
                            "comp/d_count_raw", "comp/d_coverage_raw", "comp/d_relation_raw"]:
            if all(mk in history[metric_base] and len(history[metric_base][mk]) == (epoch - start_epoch + 1)
                   for mk in model_keys):
                ys = [history[metric_base][mk] for mk in model_keys]
                chart = wandb.plot.line_series(
                    xs=epochs_list, ys=ys, keys=model_keys,
                    title=metric_base, xname="Epoch")
                all_metrics[f"compare/{metric_base}"] = chart

        wandb.log(all_metrics)

        # Checkpoints
        if epoch % cfg.save_interval == 0:
            for model_key, state in model_states.items():
                ckpt_dir = os.path.join(cfg.work_dir, f"{model_key}_{cfg.reward_mode}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "epoch": epoch, "model_id": model_key, "reward_mode": cfg.reward_mode,
                    "transformer_state_dict": get_peft_model_state_dict(state["pipeline"].transformer),
                    "optimizer_state_dict": state["optimizer"].state_dict(),
                    "rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "random": random.getstate()},
                }, os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))
            print(f"  checkpoints saved: epoch_{epoch}.pt × {len(model_states)} models")

    print(f"\n{'='*60}\n实验四 Part 2 完成! models={list(model_states.keys())}, reward={cfg.reward_mode}\n{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from epoch N (loads epoch_N.pt, continues from N+1)")
    args = parser.parse_args()
    cfg = get_config_from_path(args.config)
    run_training(cfg, resume_epoch=args.resume)
