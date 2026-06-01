"""实验三 Part 2: VLM 结构提取 — 三种策略对比.

三种方法:
    0. baseline_512        — 512px, RGB, thinking=enabled (参照组)
    1. no_thinking         — 512px, RGB, thinking=disabled (速度优先)
    2. compress_grayscale  — 256px + Grayscale (压缩 + 灰度 结合)

与 Part 1 的区别:
    - 5 个新的 prompt (不同于 Part 1)
    - 只保留 3 种有意义的 variant
    - VLM prompt 已加入「忽略大背景」提醒
    - 同样的 variant → 并行 6 图 → 输出结构

Usage:
    python experiments/exp3/exp3_part2.py --config config/exp3_part2_config.py
"""

import json
import os
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import wandb
from diffusers import StableDiffusion3Pipeline
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srdm_pytorch_exp.sde_sampling import (
    encode_prompt,
    make_chain_generators,
    pipeline_sd3_train_sample,
    total_log_prob_from_list,
)
from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler
from vlm_client import (
    VLMClient,
    VLMVariant,
    compare_structures,
    draw_structure_annotations,
)

# ============================================================
# Variants (3 种)
# ============================================================

VARIANTS = [
    VLMVariant("baseline_512",        "Baseline (512px)",           "reference",               512),
    VLMVariant("no_thinking",         "No Thinking",                "关闭思考轮次",            512, disable_thinking=True),
    VLMVariant("compress_grayscale",  "Compress 256 + Grayscale",   "压缩大小 + 灰度化",       256, grayscale=True),
]

# ============================================================
# Prompts (5 个新 prompt，不同于 Part 1)
# ============================================================

PROMPTS = [
    "Two red apples on the left, three green pears on the right, "
    "a wooden basket between them. Still life photography.",

    "A tall black lamp on the left side of a desk, a white laptop "
    "on the right side. Modern office setup.",

    "One large blue book on top of two smaller red books, "
    "stacked vertically. Library shelf background.",

    "A brown dog sitting on the left, a white cat standing on the "
    "right, facing each other. Living room scene.",

    "Four yellow tennis balls scattered on a green tennis court, "
    "with a tennis racket on the left side. Sports photography.",
]


# ============================================================
# Helpers
# ============================================================

def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    img = (image_tensor / 2 + 0.5).clamp(0, 1)
    img_np = img[0].cpu().permute(1, 2, 0).float().numpy()
    img_np = (img_np * 255).round().astype("uint8")
    return Image.fromarray(img_np)


def make_result_grid(images: list, thumb_size: int = 256) -> Image.Image:
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (thumb_size * cols, thumb_size * rows))
    for i in range(n):
        pil = images[i].resize((thumb_size, thumb_size))
        grid.paste(pil, ((i % cols) * thumb_size, (i // cols) * thumb_size))
    return grid


def _load_config(config_path: str):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return __import__(config_name).get_config()


# ============================================================
# Generate 6 SDE chains for one prompt
# ============================================================

def generate_chains(pipeline, prompt: str, device: str, config) -> tuple:
    prompt_embeds, pooled_embeds, neg_embeds, neg_pooled = encode_prompt(
        pipeline, prompt, device,
    )
    pil_images = []
    log_p_base_totals = []

    for c in range(config.num_chains_per_prompt):
        latents_gen, step_gens = make_chain_generators(
            config.seed, c, config.num_inference_steps, device,
        )
        all_gens = [latents_gen] + step_gens

        images, _, _, log_probs_base = pipeline_sd3_train_sample(
            pipeline=pipeline,
            base_transformer=pipeline.transformer,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            negative_prompt_embeds=neg_embeds,
            negative_pooled_prompt_embeds=neg_pooled,
            height=config.height, width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=all_gens,
        )
        pil = tensor_to_pil(images)
        pil_images.append(pil)
        lp_total = total_log_prob_from_list(log_probs_base).item()
        log_p_base_totals.append(lp_total)

    return pil_images, log_p_base_totals


# ============================================================
# Run one variant across all prompts (6 images parallel per prompt)
# ============================================================

def run_variant_across_prompts(
    vlm_client: VLMClient,
    variant: VLMVariant,
    all_prompt_data: list,
    stagger_delay: float = 2.0,
) -> list:
    print(f"\n  {'='*60}")
    print(f"  Variant: {variant.label} ({variant.key})")
    print(f"  {'='*60}")

    for p_idx, pdata in enumerate(all_prompt_data):
        pil_images = pdata["pil_images"]
        schema = pdata["schema"]
        M = len(pil_images)

        results: list = [None] * M

        def _extract_one(img_idx: int) -> tuple:
            try:
                structure, elapsed = vlm_client.extract_structure_variant(
                    pil_images[img_idx], schema, variant,
                    original_prompt=pdata["prompt"],
                )
                return img_idx, structure, elapsed, None
            except Exception as e:
                return img_idx, {"objects": [], "_error": str(e)}, -1.0, str(e)

        with ThreadPoolExecutor(max_workers=min(6, M)) as ex:
            futures = []
            for i in range(M):
                futures.append(ex.submit(_extract_one, i))
                if stagger_delay > 0 and i < M - 1:
                    time.sleep(stagger_delay)
            for future in as_completed(futures):
                idx, structure, elapsed, err = future.result()
                results[idx] = {
                    "variant_key": variant.key,
                    "variant_label": variant.label,
                    "strategy": variant.strategy,
                    "structure": structure,
                    "elapsed": elapsed,
                    "is_baseline": variant.is_baseline,
                }
                if err:
                    print(f"    WARNING prompt {p_idx} chain {idx}: {err}")

        pdata[f"results_{variant.key}"] = results

        times = [r["elapsed"] for r in results if r["elapsed"] > 0]
        errors = sum(1 for r in results if "_error" in r["structure"])
        avg_t = np.mean(times) if times else -1
        print(f"    Prompt {p_idx}: {M} images, avg {avg_t:.1f}s, "
              f"errors={errors}  ({pdata['prompt'][:50]}...)")

    return all_prompt_data


# ============================================================
# Main
# ============================================================

def main(config_path: str):
    config = _load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"{'='*70}")
    print(f"  Experiment 3 Part 2: VLM — 3-Strategy Comparison")
    print(f"{'='*70}")
    print(f"  Device:    {device}")
    print(f"  Prompts:   {len(PROMPTS)}")
    print(f"  Chains:    {config.num_chains_per_prompt} per prompt "
          f"-> {len(PROMPTS) * config.num_chains_per_prompt} total images")
    print(f"  Variants:  {len(VARIANTS)} "
          f"({', '.join(v.key for v in VARIANTS)})")

    # ================================================================
    # 1. Load SD3
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  1. Loading SD3 pipeline")
    print(f"{'='*70}")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    sched_cfg = {k: v for k, v in dict(pipeline.scheduler.config).items()
                 if not k.startswith("_")}
    pipeline.scheduler = StochasticFlowMatchScheduler(a=config.a, **sched_cfg)
    pipeline.transformer.requires_grad_(False)
    pipeline.transformer.eval()

    # ================================================================
    # 2. Init VLM + wandb
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  2. Initializing VLM & wandb")
    print(f"{'='*70}")
    vlm = VLMClient(model=config.vlm_model, base_url=config.vlm_base_url)

    wandb.init(project=config.wandb_project, name=config.run_name, config={
        "num_prompts": len(PROMPTS),
        "prompts": PROMPTS,
        "num_chains_per_prompt": config.num_chains_per_prompt,
        "num_variants": len(VARIANTS),
        "variant_keys": [v.key for v in VARIANTS],
        "sde_a": config.a,
        "num_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "seed": config.seed,
        "note": "VLM prompt includes background-ignore instruction",
    })

    # ================================================================
    # 3. Generate images for all prompts
    # ================================================================
    n_total = len(PROMPTS) * config.num_chains_per_prompt
    print(f"\n{'='*70}")
    print(f"  3. Generating {len(PROMPTS)} prompts x "
          f"{config.num_chains_per_prompt} chains = {n_total} images")
    print(f"{'='*70}")

    os.makedirs(config.output_dir, exist_ok=True)
    all_prompt_data = []

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"\n  --- Prompt {p_idx}: {prompt[:80]}... ---")
        schema = vlm.extract_schema(prompt)
        canonical_labels = [obj["label"] for obj in schema.get("canonical_objects", [])]
        print(f"  spaCy nouns ({len(canonical_labels)}): {canonical_labels}")

        pil_images, log_p_base_totals = generate_chains(pipeline, prompt, device, config)

        r_in_raw = torch.tensor(log_p_base_totals)
        print(f"  log_p_base: mean={r_in_raw.mean().item():.2f}  "
              f"std={r_in_raw.std().item():.2f}  "
              f"rel_range={(r_in_raw.max()-r_in_raw.min())/abs(r_in_raw.mean())*100:.3f}%")

        for i, pil_img in enumerate(pil_images):
            pil_img.save(os.path.join(config.output_dir, f"p{p_idx}_chain_{i}_raw.png"))

        if config.visualize:
            grid = make_result_grid(pil_images)
            wandb.log({f"exp3_part2/p{p_idx}/overview_raw": wandb.Image(
                grid, caption=f"P{p_idx}: {prompt[:120]}",
            )})

        all_prompt_data.append({
            "prompt": prompt,
            "schema": schema,
            "canonical_labels": canonical_labels,
            "pil_images": pil_images,
            "log_p_base_totals": log_p_base_totals,
        })

    # ================================================================
    # 4. VLM Extraction — variant-first loop
    # ================================================================
    n_calls = len(PROMPTS) * config.num_chains_per_prompt * len(VARIANTS)
    print(f"\n{'='*70}")
    print(f"  4. VLM Extraction — {len(VARIANTS)} variants x "
          f"{len(PROMPTS)} prompts x {config.num_chains_per_prompt} chains")
    print(f"     Total: {n_calls} VLM calls")
    print(f"{'='*70}")

    variant_order = sorted(VARIANTS, key=lambda v: (0 if v.is_baseline else 1))

    for variant in variant_order:
        t0 = time.time()
        run_variant_across_prompts(vlm, variant, all_prompt_data)
        elapsed_v = time.time() - t0
        print(f"  [Variant {variant.key} total: {elapsed_v:.1f}s]")

        # Quality vs baseline
        if not variant.is_baseline:
            baseline_key = f"results_{variant_order[0].key}"
            for pdata in all_prompt_data:
                baseline_results = pdata.get(baseline_key, [])
                variant_results = pdata.get(f"results_{variant.key}", [])
                for i, vr in enumerate(variant_results):
                    if i < len(baseline_results):
                        vr["quality_vs_baseline"] = compare_structures(
                            baseline_results[i]["structure"], vr["structure"],
                        )

        # ---- Wandb logging ----
        for p_idx, pdata in enumerate(all_prompt_data):
            pil_images = pdata["pil_images"]
            results = pdata.get(f"results_{variant.key}", [])

            for c_idx, vr in enumerate(results):
                tag = f"exp3_part2/p{p_idx}/c{c_idx}/variants/{variant.key}"

                if config.visualize:
                    pil_ann = draw_structure_annotations(pil_images[c_idx].copy(), vr["structure"])
                    obj_summary = ", ".join(
                        f"{o.get('label','?')}:{o.get('count',0)}"
                        for o in vr["structure"].get("objects", [])
                    )
                    err = vr["structure"].get("_error", "")
                    caption = (f"{variant.label} | {vr['elapsed']:.1f}s | {obj_summary}")
                    if err:
                        caption += f" | ERR: {err}"
                    wandb.log({
                        f"{tag}/annotated": wandb.Image(pil_ann, caption=caption),
                        f"{tag}/time_sec": vr["elapsed"],
                    })

                    objects = vr["structure"].get("objects", [])
                    if objects:
                        obj_rows = [[
                            o.get("label", "?"), o.get("count", 0),
                            len(o.get("instances", [])),
                            json.dumps(o.get("instances", []), ensure_ascii=False)[:300],
                        ] for o in objects]
                        wandb.log({f"{tag}/json_objects": wandb.Table(
                            columns=["label", "count", "n_instances", "instances_truncated"],
                            data=obj_rows,
                        )})

                # Quality metrics (always logged)
                q = vr.get("quality_vs_baseline")
                if q:
                    wandb.log({
                        f"{tag}/quality/count_agreement": q["count_agreement"],
                        f"{tag}/quality/bbox_iou_mean": q["bbox_iou_mean"],
                    })

                json_path = os.path.join(
                    config.output_dir, f"p{p_idx}_chain_{c_idx}_{variant.key}.json",
                )
                with open(json_path, "w") as f:
                    json.dump({
                        "prompt_idx": p_idx,
                        "chain_idx": c_idx,
                        "variant": variant.key,
                        "strategy": variant.strategy,
                        "elapsed_sec": vr["elapsed"],
                        "log_p_base": pdata["log_p_base_totals"][c_idx],
                        "prompt": pdata["prompt"],
                        "canonical_labels": pdata["canonical_labels"],
                        "quality_vs_baseline": vr.get("quality_vs_baseline"),
                        "structure": vr["structure"],
                    }, f, indent=2, ensure_ascii=False)

            # Per-prompt comparison table
            tag = f"exp3_part2/p{p_idx}/variant_comparison"
            comp_rows = []
            for c_idx, vr in enumerate(results):
                q = vr.get("quality_vs_baseline")
                comp_rows.append([
                    c_idx, vr["variant_label"], vr["strategy"],
                    round(vr["elapsed"], 2),
                    q["count_agreement"] if q else None,
                    q["bbox_iou_mean"] if q else None,
                    int("_error" in vr["structure"]),
                ])
            wandb.log({f"{tag}/summary": wandb.Table(
                columns=["chain", "variant", "strategy", "time_sec",
                         "count_agree", "bbox_iou", "has_error"],
                data=comp_rows,
            )})

    # ================================================================
    # 5. Cross-variant summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  5. Cross-Variant Summary (across all {len(PROMPTS)} prompts)")
    print(f"{'='*70}")

    avg_times, avg_count_agree, avg_bbox_iou = {}, {}, {}
    for variant in VARIANTS:
        times, count_agreements, bbox_ious = [], [], []
        for pdata in all_prompt_data:
            for vr in pdata.get(f"results_{variant.key}", []):
                if vr["elapsed"] > 0:
                    times.append(vr["elapsed"])
                q = vr.get("quality_vs_baseline")
                if q:
                    count_agreements.append(q["count_agreement"])
                    bbox_ious.append(q["bbox_iou_mean"])
        avg_times[variant.key] = np.mean(times) if times else -1.0
        avg_count_agree[variant.key] = np.mean(count_agreements) if count_agreements else None
        avg_bbox_iou[variant.key] = np.mean(bbox_ious) if bbox_ious else None

    baseline_time = avg_times.get("baseline_512", 1.0)
    print(f"\n  {'Variant':<28} {'Avg Time':>10} {'Speedup':>10}  "
          f"{'Count Agr':>10}  {'Bbox IoU':>10}")
    print(f"  {'-'*72}")
    for variant in VARIANTS:
        t = avg_times[variant.key]
        speedup = baseline_time / t if t > 0 else 0
        ca = avg_count_agree.get(variant.key)
        bi = avg_bbox_iou.get(variant.key)
        ca_str = f"{ca:.3f}" if ca is not None else "-"
        bi_str = f"{bi:.3f}" if bi is not None else "-"
        print(f"  {variant.label:<28} {t:>8.2f}s  {speedup:>8.2f}x  "
              f"{ca_str:>10}  {bi_str:>10}")

    # Wandb summary
    timing_summary = [[v.label, avg_times[v.key]] for v in VARIANTS if avg_times[v.key] > 0]
    wandb.log({"exp3_part2/summary/variant_avg_time": wandb.plot.bar(
        wandb.Table(columns=["variant", "avg_time_sec"], data=timing_summary),
        "variant", "avg_time_sec",
        title="Part 2: Average VLM Extraction Time per Variant",
    )})

    speedup_rows = []
    for v in VARIANTS:
        t = avg_times[v.key]
        sp = round(baseline_time / t, 2) if t > 0 else 0
        ca = avg_count_agree.get(v.key)
        bi = avg_bbox_iou.get(v.key)
        speedup_rows.append([
            v.label, v.strategy, round(t, 2), sp,
            round(ca, 4) if ca is not None else None,
            round(bi, 4) if bi is not None else None,
        ])
    wandb.log({"exp3_part2/summary/speedup_quality_table": wandb.Table(
        columns=["variant", "strategy", "avg_time_sec", "speedup_vs_baseline",
                 "count_agreement", "bbox_iou_mean"],
        data=speedup_rows,
    )})

    # ================================================================
    # 6. Save aggregate results
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  6. Saving Aggregate Results")
    print(f"{'='*70}")

    aggregate = {
        "prompts": PROMPTS,
        "config": {
            "seed": config.seed,
            "num_chains_per_prompt": config.num_chains_per_prompt,
            "sde_a": config.a,
            "num_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "vlm_model": config.vlm_model,
            "num_variants": len(VARIANTS),
            "note": "VLM prompt includes background-ignore instruction",
        },
        "variant_avg_times_sec": {k: round(v, 3) for k, v in avg_times.items()},
        "variant_avg_count_agreement": {k: round(v, 4) if v is not None else None
                                        for k, v in avg_count_agree.items()},
        "variant_avg_bbox_iou": {k: round(v, 4) if v is not None else None
                                 for k, v in avg_bbox_iou.items()},
        "per_prompt": [],
    }
    for p_idx, pdata in enumerate(all_prompt_data):
        prompt_entry = {
            "prompt_idx": p_idx,
            "prompt": pdata["prompt"],
            "canonical_labels": pdata["canonical_labels"],
            "log_p_base_totals": pdata["log_p_base_totals"],
            "variants": {},
        }
        for variant in VARIANTS:
            results = pdata.get(f"results_{variant.key}", [])
            prompt_entry["variants"][variant.key] = [
                {"chain_idx": c_idx, "elapsed_sec": vr["elapsed"],
                 "quality_vs_baseline": vr.get("quality_vs_baseline"),
                 "structure": vr["structure"]}
                for c_idx, vr in enumerate(results)
            ]
        aggregate["per_prompt"].append(prompt_entry)

    agg_path = os.path.join(config.output_dir, "all_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {agg_path}")
    print(f"\n{'='*70}")
    print(f"  Done. Wandb: {config.wandb_project}/{config.run_name}")
    print(f"  Total VLM calls: {n_calls}")
    print(f"{'='*70}\n")

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file, e.g. config/exp3_part2_config.py")
    args = parser.parse_args()
    main(args.config)
