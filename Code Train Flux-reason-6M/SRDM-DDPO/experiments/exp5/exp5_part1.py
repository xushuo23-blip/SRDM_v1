"""
exp5 Part 1: r_gt + r_in DDPO 训练 — GT 对齐奖励 + 组内 z-score 归一化.

奖励组合规则:
    - r_gt = ±lambda_gt (默认 ±2) → 直接用 r_gt, 不组合 r_in
    - r_gt = 0 → 使用组内 z-score 归一化的 r_in

与 exp4_part3 的关键区别:
    - GT 监督信号: flux_reason_spatial_short_1k_gt.jsonl (DeepSeek 三级提取)
    - 奖励: r_gt (count+direction 两阶段门控) + r_in (base model log_prob z-score)
    - VLM: Doubao-pro + thinking 开启 (深度思考)
    - VLMClientNoun: schema 来自 GT objects keys, phi* 来自 get_gt_phi_star()
    - M=6 chains, B=3 mini_batch_size, lr=1e-5, 50 epochs

用法:
    python experiments/exp5/exp5_part1.py --config config/exp5_part1_config.py
"""

import os, random, sys, time
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import wandb
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srdm_pytorch_exp.sde_sampling import (
    encode_prompt, make_chain_generators, pipeline_sd3_train_sample,
    total_log_prob_from_list,
)
from srdm_pytorch_exp.reward_rin import zscore_normalize
from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch
from srdm_pytorch_exp.prompts_noun import load_prompts_from_file
from srdm_pytorch_exp.vlm_client_noun import VLMClientNoun, draw_structure_annotations, validate_structure_bboxes
from experiments.exp5.gt_utils import load_prompt_gt
from experiments.exp5.reward_gt import compute_r_gt_single


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


# ============================================================
# Model loading
# ============================================================

def _load_pipeline(model_path, dtype, device):
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
    pipe = _load_pipeline(model_path, dtype, device)
    import copy
    base = copy.deepcopy(pipe.transformer)
    base.requires_grad_(False).eval().to(device)

    lora_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
        target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
        lora_dropout=0.0, bias="none")
    pipe.transformer.requires_grad_(False)
    pipe.transformer = get_peft_model(pipe.transformer, lora_config).to(device)
    n_trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)

    orig = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    orig_dict = {k: v for k, v in dict(orig.config).items() if not k.startswith('_')}
    pipe.scheduler = StochasticFlowMatchScheduler(a=cfg.a, **orig_dict)

    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=cfg.learning_rate, betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay, eps=cfg.adam_epsilon)

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
# Per-model epoch
# ============================================================

def _run_model_epoch(state, selected_prompts, prompt_encodings, prompt_gt_map, epoch, cfg,
                     vlm_client, model_key):
    """Sample (all) → VLM (parallel) → Hard-Gate Reward + PPO (per prompt).

    Phase 1: 采样所有 prompt 的 M 条 chain，后台提交 VLM。
    Phase 2: 逐 prompt 等 VLM → 硬性门控 → r_in z-score → PPO update。
    """

    pipeline = state["pipeline"]
    base_transformer = state["base"]
    optimizer = state["optimizer"]
    alerter = state["alerter"]
    device = state["device"]

    N = cfg.num_prompts_per_epoch          # 3
    M = cfg.num_chains_per_prompt          # 6
    B = cfg.ppo_mini_batch_size            # 3
    updates_per_prompt = M // B            # 2

    all_chain_data = []
    pipeline.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    def _vlm_for_prompt(p_chains, prompt_text):
        t0 = time.time()
        schema = vlm_client.extract_schema(prompt_text)
        structures = vlm_client.extract_structures_batch(
            [d["pil_image"] for d in p_chains], schema,
            original_prompt=prompt_text, max_workers=cfg.vlm_max_workers,
            stagger_delay=cfg.vlm_stagger_delay, max_image_size=cfg.vlm_max_image_size,
            disable_thinking=cfg.vlm_disable_thinking)
        return schema, structures, time.time() - t0

    # ---- Phase 1: Sample all + submit VLM ----
    vlm_futures = []

    vlm_pool = ThreadPoolExecutor(max_workers=N)
    try:
        for p_idx, prompt_text in enumerate(selected_prompts):
            prompt_embeds, pooled_embeds, neg_embeds, neg_pooled = prompt_encodings[p_idx]
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

        # ---- Phase 2: Per-prompt VLM → r_gt + r_in → PPO ----
        vlm_elapsed = 0.0
        all_rewards = torch.zeros(N * M, device=device)
        json_bad_total = 0
        r_gt_pos_total = 0     # r_gt = +lambda_gt
        r_gt_zero_total = 0    # r_gt = 0
        r_gt_neg_total = 0     # r_gt = -lambda_gt
        ppo_metrics = defaultdict(list)
        batch_skip_total = 0
        ppo_total = N * updates_per_prompt
        ppo_pbar = tqdm(total=ppo_total, desc=f"      PPO {model_key}", leave=False)

        for p_idx, future, p_chains in vlm_futures:
            schema, structures, elapsed = future.result()
            vlm_elapsed += elapsed

            # Validate structures
            for d, s in zip(p_chains, structures):
                d["schema"] = schema
                d["structure"] = s
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

            prompt_gt = prompt_gt_map.get(p_chains[0]["prompt"], {})

            if len(valid_chains) >= 1 and prompt_gt:
                # ---- r_gt: GT 对齐奖励 (count + direction 两阶段门控) ----
                lambda_gt = cfg.lambda_gt
                r_gt_list = []
                for d in valid_chains:
                    r_gt, r_gt_debug = compute_r_gt_single(
                        d["structure"], prompt_gt,
                        count_penalty=-lambda_gt,
                        direction_reward=lambda_gt,
                        direction_threshold=cfg.direction_threshold,
                        return_debug=True,
                    )
                    d["r_gt"] = r_gt
                    d["r_gt_debug"] = r_gt_debug  # vlm_counts, gt_counts, vlm_direction, gt_direction
                    r_gt_list.append(r_gt)

                # ---- r_in: 组内 z-score 归一化 (base model log_prob) ----
                valid_lp = torch.tensor([d["total_lp_base"] for d in valid_chains], device=device)
                r_in = zscore_normalize(valid_lp)

                # ---- 组合: r_gt != 0 → 直接用 r_gt; r_gt == 0 → 用 r_in ----
                for j, d in enumerate(valid_chains):
                    d["r_in"] = r_in[j].item()
                    if d["r_gt"] != 0:
                        d["reward"] = float(d["r_gt"])
                    else:
                        d["reward"] = r_in[j].item()

                # 统计 r_gt 分布
                for d in valid_chains:
                    if d["r_gt"] > 0:
                        r_gt_pos_total += 1
                    elif d["r_gt"] < 0:
                        r_gt_neg_total += 1
                    else:
                        r_gt_zero_total += 1

                # Bad chains: 保守兜底 = min reward from valid chains
                min_reward = min(d["reward"] for d in valid_chains)
                for d in bad_chains:
                    d["reward"] = min_reward
                    d["r_gt"] = None
                    d["r_in"] = None
                    d["r_gt_debug"] = None

                p_chains[0]["_debug_info"] = {
                    "r_gt_list": r_gt_list,
                    "r_in": r_in.tolist(),
                    "n_valid": len(valid_chains),
                }
            else:
                for d in p_chains:
                    d["reward"] = 0.0
                    d["r_gt"] = 0.0
                    d["r_in"] = 0.0
                    d["r_gt_debug"] = None
                p_chains[0]["_debug_info"] = None

            # Advantage = reward (r_gt ±λ 或 r_in z-score，均无需再归一化)
            start, end = p_idx * M, (p_idx + 1) * M
            all_rewards[start:end] = torch.tensor([d["reward"] for d in p_chains], device=device)

            # ---- PPO for this prompt ----
            p_indices = list(range(p_idx * M, (p_idx + 1) * M))
            random.shuffle(p_indices)
            for u in range(updates_per_prompt):
                ppo_pbar.update(1)
                raw_batch = p_indices[u * B:(u + 1) * B]
                good_idx = [i for i in raw_batch if all_chain_data[i].get("json_ok", True)]
                n_bad_batch = len(raw_batch) - len(good_idx)

                if n_bad_batch >= 2:
                    batch_skip_total += 1
                    ppo_metrics["json_batch_skipped"].append(1)
                    continue

                ppo_metrics["json_batch_skipped"].append(0)

                if len(good_idx) < 2:
                    batch_skip_total += 1
                    continue

                m = ppo_update_mini_batch(
                    pipeline, all_chain_data, good_idx, timesteps,
                    advantages=all_rewards, guidance_scale=cfg.guidance_scale,
                    optimizer=optimizer, ppo_clip_range=cfg.ppo_clip_range,
                    max_grad_norm=cfg.max_grad_norm,
                    num_inference_steps=cfg.num_inference_steps, alerter=alerter)
                for k, v in m.items():
                    ppo_metrics[k].append(v)
    finally:
        vlm_pool.shutdown(wait=False)
    ppo_pbar.close()

    # ---- Build metrics ----
    all_rewards_np = all_rewards.cpu()
    pfx = lambda name: f"{name}/{model_key}"

    total_chains = N * M
    total_valid = total_chains - json_bad_total
    r_gt_sign_rate = (r_gt_pos_total + r_gt_neg_total) / max(total_valid, 1)

    # Per-component reward statistics
    all_log_p_base = torch.tensor([d.get("total_lp_base", 0.0) for d in all_chain_data])
    all_r_gt = torch.tensor([d.get("r_gt", 0.0) for d in all_chain_data if d.get("r_gt") is not None])
    all_r_in = torch.tensor([d.get("r_in", 0.0) for d in all_chain_data if d.get("r_in") is not None])

    metrics = {
        pfx("reward_mean"): all_rewards_np.mean().item(),
        pfx("reward_std"): all_rewards_np.std().item(),
        pfx("log_p_base_mean"): all_log_p_base.mean().item(),
        pfx("log_p_base_std"): all_log_p_base.std().item(),
        pfx("r_gt_mean"): all_r_gt.mean().item() if all_r_gt.numel() > 0 else 0.0,
        pfx("r_gt_std"): all_r_gt.std().item() if all_r_gt.numel() > 0 else 0.0,
        pfx("r_gt_pos"): r_gt_pos_total,
        pfx("r_gt_zero"): r_gt_zero_total,
        pfx("r_gt_neg"): r_gt_neg_total,
        pfx("r_gt_sign_rate"): r_gt_sign_rate,
        pfx("r_in_mean"): all_r_in.mean().item() if all_r_in.numel() > 0 else 0.0,
        pfx("r_in_std"): all_r_in.std().item() if all_r_in.numel() > 0 else 0.0,
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

    # ---- Images ----
    images_dict = {}

    # Thumbnail grid (every epoch)
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

    # Reward histograms (every epoch)
    images_dict[pfx("reward_hist")] = wandb.Histogram(all_rewards_np.tolist())
    if all_r_gt.numel() > 0:
        images_dict[pfx("r_gt_hist")] = wandb.Histogram(all_r_gt.tolist())
    if all_r_in.numel() > 0:
        images_dict[pfx("r_in_hist")] = wandb.Histogram(all_r_in.tolist())
    images_dict[pfx("log_p_base_hist")] = wandb.Histogram(all_log_p_base.tolist())

    # Bbox annotations: epoch 1 + every log_interval epochs
    # Show ALL chains with VLM features vs GT standard answer
    if epoch == 1 or epoch % cfg.log_interval == 0:
        for p_idx in range(N):
            p_chains = [d for d in all_chain_data if d["prompt_idx"] == p_idx]
            prompt_text = p_chains[0]["prompt"] if p_chains else ""
            gt_info = prompt_gt_map.get(prompt_text, {})
            gt_counts = gt_info.get("objects", {})
            gt_direction = gt_info.get("spatial_relation", {}).get("direction", [0, 0])
            gt_top2 = gt_info.get("top2_objects", [])

            # GT standard answer string
            gt_count_str = "{" + ", ".join(f"{k}:{v}" for k, v in gt_counts.items()) + "}"
            gt_dir_str = f"[{gt_direction[0]},{gt_direction[1]}]" if len(gt_direction) == 2 else "[0,0]"
            gt_answer_str = f"GT*: count={gt_count_str} dir({gt_top2[0] if len(gt_top2)>0 else '?'}→{gt_top2[1] if len(gt_top2)>1 else '?'})={gt_dir_str}"

            for chain in p_chains:
                label = f"c{chain['chain_idx']}"
                try:
                    pil_ann = draw_structure_annotations(chain["pil_image"].copy(), chain["structure"]) if chain.get("json_ok") else chain["pil_image"].copy()

                    # VLM detection string
                    debug = chain.get("r_gt_debug", {}) or {}
                    vlm_counts = debug.get("vlm_counts", {})
                    vlm_direction = debug.get("vlm_direction", [])
                    vlm_count_str = "{" + ", ".join(f"{k}:{v}" for k, v in vlm_counts.items()) + "}" if vlm_counts else "N/A"
                    vlm_dir_str = f"[{vlm_direction[0]},{vlm_direction[1]}]" if vlm_direction and len(vlm_direction) == 2 else "N/A"
                    vlm_str = f"VLM: count={vlm_count_str} dir={vlm_dir_str}"

                    # Reward string
                    r_gt = chain.get("r_gt")
                    r_in_val = chain.get("r_in")
                    reward = chain.get("reward", 0)
                    if r_gt is not None:
                        if r_gt != 0:
                            reward_str = f"r_gt={r_gt:+.0f}"
                        else:
                            reward_str = f"r_gt=0 r_in={r_in_val:+.3f}"
                    else:
                        reward_str = f"reward={reward:.3f} (bad)"

                    caption = (
                        f"[{model_key}] e{epoch} p{p_idx} {label} | {reward_str}\n"
                        f"  {gt_answer_str}\n"
                        f"  {vlm_str}"
                    )
                    images_dict[pfx(f"bbox/prompt{p_idx}_{label}")] = wandb.Image(pil_ann, caption=caption)
                except Exception as e:
                    print(f"  [{model_key}] bbox draw error p{p_idx} c{chain['chain_idx']}: {e}")

    # ---- Console ----
    grad_norms = ppo_metrics.get("grad_norm", [0])
    lr = optimizer.param_groups[0]["lr"]
    console = {
        "reward_mean": all_rewards_np.mean().item(),
        "reward_std": all_rewards_np.std().item(),
        "r_gt_dist": f"+{r_gt_pos_total}/0:{r_gt_zero_total}/-{r_gt_neg_total}",
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
# Training loop
# ============================================================

def run_training(cfg, resume_epoch=0):
    device = torch.device(f"cuda:{cfg.gpu_sd35}")

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

    # ---- VLM (GT 监督信号模式) ----
    prompt_gt = load_prompt_gt(cfg.prompt_gt_file)
    all_prompts = load_prompts_from_file(cfg.prompt_file)
    n_with_gt = sum(1 for p in all_prompts if p in prompt_gt)
    print(f"VLM: {cfg.vlm_backend} / {cfg.vlm_model} (GT gate mode) | "
          f"Prompts: {len(all_prompts)} ({n_with_gt} with GT)")

    prompt_objects = {p: list(gt["objects"].keys()) for p, gt in prompt_gt.items()}
    vlm_client = VLMClientNoun(
        prompt_objects=prompt_objects,
        backend=cfg.vlm_backend, model=cfg.vlm_model,
        max_retries=cfg.vlm_max_retries,
    )

    # ---- Load model ----
    model_key = cfg.model_ids[0]
    model_path = cfg.pretrained_model_paths.get(model_key)
    if model_path is None or "REPLACE" in model_path:
        raise RuntimeError(f"Model path not configured for {model_key}")
    print(f"加载模型: {model_key} -> GPU {cfg.gpu_sd35} | {model_path}")
    state = _build_model_state(model_key, model_path, dtype, cfg, device)

    N = cfg.num_prompts_per_epoch
    M = cfg.num_chains_per_prompt
    B = cfg.ppo_mini_batch_size
    updates_per_epoch = N * (M // B)

    # ---- Resume ----
    start_epoch = 1
    if resume_epoch > 0:
        ckpt_dir = os.path.join(cfg.work_dir, f"{model_key}_{cfg.reward_mode}")
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{resume_epoch}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        missing, unexpected = state["pipeline"].transformer.load_state_dict(
            ckpt["transformer_state_dict"], strict=False)
        state["optimizer"].load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  [{model_key}] loaded checkpoint: epoch {ckpt['epoch']} from {ckpt_path}")
        if missing:
            print(f"  [{model_key}]   missing keys: {len(missing)}")
        if unexpected:
            print(f"  [{model_key}]   unexpected keys: {len(unexpected)}")
        torch.set_rng_state(ckpt["rng_state"]["torch"])
        np.random.set_state(ckpt["rng_state"]["numpy"])
        random.setstate(ckpt["rng_state"]["random"])
        start_epoch = resume_epoch + 1
        print(f"\n=== 从 epoch {start_epoch} 继续训练 ===\n")

    remaining = cfg.num_epochs - start_epoch + 1
    print(f"\n训练: {cfg.num_epochs} epochs (剩余 {remaining}), "
          f"{N}x{M}={N*M} chains/epoch, {updates_per_epoch} PPO updates/epoch")
    print(f"模型: {model_key} | 奖励: {cfg.reward_mode} | 精度: {cfg.mixed_precision}")
    print(f"GPU: cuda:{cfg.gpu_sd35} | VLM: {cfg.vlm_model}")
    print(f"Lambda GT: ±{cfg.lambda_gt} | Direction threshold: {cfg.direction_threshold}")

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        selected = random.sample(all_prompts, N)

        # Encode prompts
        prompt_encodings_cpu = []
        for p in selected:
            pe, po, ne, np_ = encode_prompt(state["pipeline"], p, device)
            prompt_encodings_cpu.append((pe.cpu(), po.cpu(), ne.cpu(), np_.cpu()))

        # Run epoch (single model, no ThreadPoolExecutor)
        metrics, images, console = _run_model_epoch(
            state, selected, prompt_encodings_cpu, prompt_gt, epoch, cfg, vlm_client, model_key)

        # Log
        all_metrics = {"epoch": epoch}
        all_metrics.update(metrics)
        all_metrics.update(images)

        alert_keys = []
        pfx = lambda name: f"{name}/{model_key}"
        for k in ["alert/ratio_fired", "alert/grad_fired"]:
            if metrics.get(pfx(k)):
                alert_keys.append(k.split("/")[-1].replace("_fired", ""))
        alerts_str = "".join(f" [{a.upper()}]" for a in alert_keys)
        json_str = f" | JSON bad:{console['json_bad']} skip:{console['batch_skip']}" if console["json_bad"] > 0 else ""
        print(f"[{model_key}] e{epoch:3d}/{cfg.num_epochs} | "
              f"reward: {console['reward_mean']:+.3f}+/-{console['reward_std']:.3f} | "
              f"r_gt: {console['r_gt_dist']} | "
              f"loss: {console['loss']:.4f} | "
              f"ratio: {console['ratio_mean']:.2f} | clip: {console['clip_rate']:.0%} | "
              f"|grad|: {console['grad_norm']:.2f} | step: {console['lr_step']:.2e} | "
              f"VLM: {console['vlm_elapsed']:.1f}s{alerts_str}{json_str}")

        wandb.log(all_metrics)

        # Checkpoint
        if epoch % cfg.save_interval == 0:
            ckpt_dir = os.path.join(cfg.work_dir, f"{model_key}_{cfg.reward_mode}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch, "model_id": model_key, "reward_mode": cfg.reward_mode,
                "transformer_state_dict": get_peft_model_state_dict(state["pipeline"].transformer),
                "optimizer_state_dict": state["optimizer"].state_dict(),
                "rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "random": random.getstate()},
            }, os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))
            print(f"  checkpoint saved: epoch_{epoch}.pt")

    print(f"\n{'='*60}\nexp5 完成! model={model_key}, reward={cfg.reward_mode}\n{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from epoch N (loads checkpoint epoch_N.pt)")
    args = parser.parse_args()
    cfg = get_config_from_path(args.config)
    run_training(cfg, resume_epoch=args.resume)
