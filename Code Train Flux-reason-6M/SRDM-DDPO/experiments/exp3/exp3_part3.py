"""实验三 Part 3: φ 结构特征计算 + 距离可视化.

读取 Part 2 的 VLM JSON 数据，计算:
  - φ_count / φ_coverage / φ_relation (structure_features)
  - φ* 均匀平均原型
  - 各分量 z-score 归一化 → 加权合并距离
  - λ_count=0.5, λ_coverage=0.25, λ_relation=0.25

Wandb 输出 (per prompt):
  - 每条链的结构 bbox 图 + 特征值文本 (3 variants × 6 chains)
  - 距离散点图: φ* 为中心, 各链距离标记

Usage:
    python experiments/exp3/exp3_part3.py --config config/exp3_part3_config.py
    python experiments/exp3/exp3_part3.py  # uses defaults
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from srdm_pytorch_exp.structure_features import phi_dicts_simplified
from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch, make_distance_plot
from vlm_client import draw_structure_annotations, validate_structure_bboxes

# ============================================================
# Defaults (overridden by --config)
# ============================================================

_DEFAULTS = {
    "run_name": "exp3_part3_phi_features",
    "seed": 42,
    "input_json": "experiments/exp3/exp3_part2_output/all_results.json",
    "image_dir": "experiments/exp3/exp3_part2_output",
    "lambda_count": 0.5,
    "lambda_coverage": 0.25,
    "lambda_relation": 0.25,
    "visualize": True,
    "bbox_width": 256,
    "panel_width": 320,
    "wandb_project": "SRDM-DDPO",
}

VARIANT_KEYS = ["baseline_512", "no_thinking", "compress_grayscale"]
VARIANT_LABELS = {
    "baseline_512": "Baseline 512",
    "no_thinking": "No Thinking",
    "compress_grayscale": "Compress+Grayscale",
}


# ============================================================
# Font helper
# ============================================================

def _get_font(size: int = 14):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ============================================================
# Feature text panel (experiment-specific: bbox + φ values)
# ============================================================

def make_feature_image(
    pil_image: Image.Image,
    structure: dict,
    phi_dict: dict,
    distances: dict,
    chain_idx: int,
    total_d: float,
    lambda_count: float,
    lambda_coverage: float,
    lambda_relation: float,
    bbox_width: int = 256,
    panel_width: int = 320,
) -> Image.Image:
    """Side-by-side: bbox annotated structure + feature value panel."""
    w_img, h_img = bbox_width, bbox_width
    pad = 4
    total_w = w_img + pad + panel_width

    canvas = Image.new("RGB", (total_w, h_img), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(11)
    font_title = _get_font(13)

    # Left: bbox annotated
    ann = draw_structure_annotations(pil_image.copy(), structure, line_width=2, font_size=12)
    ann = ann.resize((w_img, h_img))
    canvas.paste(ann, (0, 0))

    # Right: feature panel
    count_vec = phi_dict.get("count", torch.tensor([]))
    cov_val = phi_dict.get("coverage", torch.tensor([])).item() if phi_dict.get("coverage", torch.tensor([])).numel() > 0 else 0.0
    rel_vec = phi_dict.get("relation", torch.tensor([]))

    d_count = distances.get("d_count", [0.0] * 6)[chain_idx]
    d_cov = distances.get("d_coverage", [0.0] * 6)[chain_idx]
    d_rel = distances.get("d_relation", [0.0] * 6)[chain_idx]
    d_count_n = distances.get("d_count_norm", [0.0] * 6)[chain_idx]
    d_cov_n = distances.get("d_coverage_norm", [0.0] * 6)[chain_idx]
    d_rel_n = distances.get("d_relation_norm", [0.0] * 6)[chain_idx]

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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Exp3 Part 3: phi features + distance viz")
    parser.add_argument("--config", type=str, default="", help="Path to ml_collections config .py file")
    args = parser.parse_args()

    # Load config
    if args.config:
        from ml_collections import ConfigDict
        import importlib.util
        spec = importlib.util.spec_from_file_location("exp3_part3_config", args.config)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        cfg = cfg_mod.get_config()
        config = {k: cfg.get(k, _DEFAULTS[k]) if k in cfg else _DEFAULTS[k] for k in _DEFAULTS}
    else:
        config = dict(_DEFAULTS)

    print("=" * 60)
    print(f"  Experiment 3 Part 3: phi Features + Distance Viz")
    print(f"  Config: {config}")
    print("=" * 60)

    with open(config["input_json"]) as f:
        data = json.load(f)

    prompts = data["prompts"]
    per_prompt = data["per_prompt"]

    wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config,
    )

    for p_idx, pdata in enumerate(per_prompt):
        prompt = pdata["prompt"]
        canonical_labels = pdata["canonical_labels"]
        log_p_base = pdata["log_p_base_totals"]
        schema = {"canonical_objects": [{"label": l} for l in canonical_labels]}

        print(f"\n{'='*60}")
        print(f"  Prompt {p_idx}: {prompt[:70]}...")
        print(f"  Labels: {canonical_labels}")
        print(f"{'='*60}")

        for v_key in VARIANT_KEYS:
            chains = pdata["variants"].get(v_key, [])
            if not chains:
                continue

            structures = []
            valid_indices = []
            for c in chains:
                struct = c["structure"]
                json_ok = not (
                    "_error" in struct
                    or not isinstance(struct.get("objects"), list)
                    or len(struct.get("objects", [])) == 0
                )
                if json_ok:
                    json_ok = validate_structure_bboxes(struct)
                if json_ok:
                    structures.append(struct)
                    valid_indices.append(c["chain_idx"])

            if len(structures) < 2:
                print(f"  {v_key}: <2 valid chains, skip")
                continue

            # ---- Compute phi + distances ----
            phi_dicts, active_nouns, top2, dead_nouns = phi_dicts_simplified(structures, schema)
            r_in_raw = torch.tensor([log_p_base[i] for i in valid_indices])

            r_ssr, debug = compute_r_ssr_batch(
                phi_dicts,
                r_in_raw,
                lambda_count=config["lambda_count"],
                lambda_coverage=config["lambda_coverage"],
                lambda_relation=config["lambda_relation"],
                uniform_weights=True,
            )

            print(f"\n  {v_key} ({len(structures)} valid chains):")
            print(f"    active_nouns: {active_nouns}")
            print(f"    top2:         {top2}")
            print(f"    dead_nouns:   {dead_nouns}")
            print(f"    d_combined:   {[f'{v:.3f}' for v in debug['d_combined'].tolist()]}")
            print(f"    r_ssr:        {[f'{v:.3f}' for v in r_ssr.tolist()]}")

            distances = {
                "d_count": debug["d_count"].tolist(),
                "d_coverage": debug["d_coverage"].tolist(),
                "d_relation": debug["d_relation"].tolist(),
                "d_count_norm": debug["d_count_norm"].tolist(),
                "d_coverage_norm": debug["d_coverage_norm"].tolist(),
                "d_relation_norm": debug["d_relation_norm"].tolist(),
            }

            if config["visualize"]:
                # ---- Per-chain feature images ----
                feature_images = []
                for i, (struct, c_idx) in enumerate(zip(structures, valid_indices)):
                    pil_img = Image.open(
                        f"{config['image_dir']}/p{p_idx}_chain_{c_idx}_raw.png"
                    )
                    feat_img = make_feature_image(
                        pil_img, struct, phi_dicts[i], distances,
                        chain_idx=c_idx, total_d=debug["d_combined"][i].item(),
                        lambda_count=config["lambda_count"],
                        lambda_coverage=config["lambda_coverage"],
                        lambda_relation=config["lambda_relation"],
                        bbox_width=config["bbox_width"],
                        panel_width=config["panel_width"],
                    )
                    feature_images.append(feat_img)

                # Grid: 3 cols × 2 rows
                grid_w = max(img.width for img in feature_images)
                grid_h = max(img.height for img in feature_images)
                grid_img = Image.new("RGB", (grid_w * 3, grid_h * 2), (245, 245, 245))
                for i, img in enumerate(feature_images):
                    row, col = divmod(i, 3)
                    grid_img.paste(img, (col * grid_w, row * grid_h))

                wandb.log({
                    f"p{p_idx}/{v_key}/feature_grid": wandb.Image(
                        grid_img,
                        caption=f"P{p_idx} {VARIANT_LABELS.get(v_key, v_key)}: "
                                f"bbox + phi features (active={active_nouns}, top2={top2})"
                    ),
                })

                # ---- Distance scatter plot ----
                dist_img = make_distance_plot(
                    debug, valid_indices,
                    variant_label=VARIANT_LABELS.get(v_key, v_key),
                    prompt_short=prompt,
                )
                wandb.log({
                    f"p{p_idx}/{v_key}/distance_plot": wandb.Image(
                        dist_img,
                        caption=f"P{p_idx} {VARIANT_LABELS.get(v_key, v_key)}: "
                                f"chain distances from phi*"
                    ),
                })

    print(f"\n{'='*60}")
    print(f"  Done. Wandb: {config['wandb_project']}/{config['run_name']}")
    print(f"{'='*60}\n")
    wandb.finish()


if __name__ == "__main__":
    main()
