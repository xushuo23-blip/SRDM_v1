"""
实验四 Part 3：DDPO 训练 (r_SSR) — 仅 SD3.5 + fp32 + no_thinking + spaCy.

与 Part 1/2 的关键区别:
    - 仅 SD3.5 单模型 (单 GPU)
    - fp32 精度: ratio 严格 = 1.0, 训练稳定
    - VLM: no_thinking 模式 (4.4x 加速)
    - spaCy 实时名词提取 → schema 秒级构建, 零 API 费用
    - 1K prompt 数据集

用法:
    python experiments/exp4/exp4_part3.py --config config/exp4_part3_config.py
"""

import copy, os, random, sys, time
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srdm_pytorch_exp.sde_sampling import (
    encode_prompt, make_chain_generators, pipeline_sd3_train_sample,
    total_log_prob_from_list, zscore_normalize,
)
from vlm_client import draw_structure_annotations, validate_structure_bboxes
from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch
from prompts import load_prompts_from_file
from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch
from srdm_pytorch_exp.structure_features import phi_dicts_simplified
from vlm_client import VLMClient


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
    """Load SD3.5 pipeline on the given device."""
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

def _run_model_epoch(state, selected_prompts, prompt_encodings, epoch, cfg,
                     vlm_client, model_key):
    """Sample (all prompts) → VLM (parallel) → r_SSR + PPO (per prompt, as VLM ready).

    VLM 在每个 prompt 的 8 条 chain 采样完后立即后台提交，与后续 prompt 采样重叠。
    采样全部结束后，逐 prompt 处理：block 等待该 prompt 的 VLM → 算 r_SSR → PPO update。
    如果 VLM 在采样期间已跑完，future.result() 立即返回，GPU 无缝衔接 PPO。
    """

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
    pipeline.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    def _vlm_for_prompt(p_chains, prompt_text):
        t0 = time.time()
        schema = vlm_client.extract_schema(prompt_text)
        structures = vlm_client.extract_structures_batch(
            [d["pil_image"] for d in p_chains], schema,
            original_prompt=prompt_text, max_workers=cfg.vlm_max_workers,
            stagger_delay=cfg.vlm_stagger_delay, max_image_size=cfg.vlm_max_image_size,
            disable_thinking=True)
        return schema, structures, time.time() - t0

    # ---- Phase 1: Sample all prompts + submit VLM in background ----
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

        # ---- Phase 2: Per-prompt VLM → r_SSR → PPO (blocks only on unfinished VLM) ----
        vlm_elapsed = 0.0
        all_advantages = torch.zeros(N * M, device=device)
        json_bad_total = 0
        ppo_metrics = defaultdict(list)
        batch_skip_total = 0
        ppo_total = N * updates_per_prompt
        ppo_pbar = tqdm(total=ppo_total, desc=f"      PPO {model_key}", leave=False)

        for p_idx, future, p_chains in vlm_futures:
            # future.result() 阻塞直到该 prompt 的 VLM 完成。
            # VLM 在 Phase 1 采样期间已后台运行，此时大概率立即返回。
            schema, structures, elapsed = future.result()
            vlm_elapsed += elapsed

            # ---- r_SSR ----
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

            # ---- PPO for this prompt ----
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
    finally:
        vlm_pool.shutdown(wait=False)
    ppo_pbar.close()

    # ---- Build metrics ----
    all_r_ssr = torch.tensor([d.get("r_ssr", 0.0) for d in all_chain_data])
    pfx = lambda name: f"{name}/{model_key}"

    metrics = {
        pfx("advantage_mean"): all_advantages.mean().item(),
        pfx("advantage_std"): all_advantages.std().item(),
        pfx("r_ssr_mean"): all_r_ssr.mean().item(),
        pfx("r_ssr_std"): all_r_ssr.std().item(),
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

    # r_SSR histogram (every epoch)
    images_dict[pfx("r_ssr_hist")] = wandb.Histogram(all_r_ssr.tolist())

    # Bbox annotations: epoch 1 + every log_interval epochs
    if epoch == 1 or epoch % cfg.log_interval == 0:
        for p_idx in range(N):
            p_chains = [d for d in all_chain_data if d["prompt_idx"] == p_idx]
            valid_p = [d for d in p_chains if d.get("json_ok")]

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

    # ---- Console ----
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

    # ---- VLM (spaCy 实时名词提取) ----
    all_prompts = load_prompts_from_file(cfg.prompt_file)
    print(f"VLM: {cfg.vlm_backend} / no_thinking (spaCy) | "
          f"Prompts: {len(all_prompts)}")

    vlm_client = VLMClient(
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
    print(f"GPU: cuda:{cfg.gpu_sd35}")

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
            state, selected, prompt_encodings_cpu, epoch, cfg, vlm_client, model_key)

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
              f"adv: {console['adv_mean']:+.3f}+/-{console['adv_std']:.3f} | "
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

    print(f"\n{'='*60}\n实验四 Part 3 完成! model={model_key}, reward={cfg.reward_mode}\n{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from epoch N (loads checkpoint epoch_N.pt)")
    args = parser.parse_args()
    cfg = get_config_from_path(args.config)
    run_training(cfg, resume_epoch=args.resume)
