"""Exp6 柱状图可视化 — checkpoint vs 基线, 独立脚本.

Usage (from Code Test PRISM-Bench/):
    python Rcombine_exp6_part1/doubao-seed-2.0-pro/visualize_exp6.py \
        --variant sd35_exp6_epoch300 \
        --scores_dir Rcombine_exp6_part1/doubao-seed-2.0-pro/scores/sd35_exp6_epoch300_doubao-seed-2.0-pro \
        --baseline baseline_scores/baseline_scores_doubao-seed-2.0-pro.txt \
        --base_model sd35 \
        --output Rcombine_exp6_part1/doubao-seed-2.0-pro/results/comparison_epoch300.png \
        --title "Exp6 Part1 Epoch 300 vs SD3.5 Baseline (Doubao-seed-2.0-pro)"
"""

import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent  # PRISM-Bench root

TRACKS = ["imagination", "entity", "text_rendering", "style", "affection", "composition", "long_text"]
CATEGORIES = ["overall"] + TRACKS

_PRIMARY_TRACK = "composition"
_REASONING_TRACKS = {"entity", "long_text"}

BAR_COLORS = ["#5B9BD5", "#ED7D31", "#2ca02c"]


def parse_baseline_file(filepath: Path, base_model: str) -> dict:
    """Parse baseline txt, return only the matching model's scores.

    Args:
        filepath: path to baseline_scores_*.txt
        base_model: "sd3" or "sd35"
    """
    target = "SD3.5 Baseline" if base_model == "sd35" else "SD3 Baseline"
    text = filepath.read_text(encoding="utf-8")
    results = {}
    current_model = None

    for line in text.split("\n"):
        m = re.match(r"Model\s+:\s+(.*)", line)
        if m:
            model_full = m.group(1).strip()
            if "SD3.5" in model_full:
                current_model = "SD3.5 Baseline"
            elif "SD3" in model_full:
                current_model = "SD3 Baseline"
            else:
                current_model = model_full
            results[current_model] = {}
            continue

        if current_model:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                nums = [float(p) for p in parts[-3:]]
            except ValueError:
                continue
            cat_name = " ".join(parts[:-3]).lower().replace(" ", "_")
            if cat_name in CATEGORIES:
                results[current_model][cat_name] = {
                    "alignment": nums[0], "aesthetic": nums[1], "avg": nums[2],
                }

    # Only return the matching baseline
    if target in results:
        return {target: results[target]}
    return {}


def load_scores(scores_dir: Path) -> dict:
    """Load variant scores from scores_dir (alignment/ + aesthetic/)."""
    results = {}
    all_align, all_aes = [], []

    for track in TRACKS:
        align_scores, aes_scores = [], []
        for i in range(100):
            f_align = scores_dir / "alignment" / track / f"{i}.jsonl"
            f_aes = scores_dir / "aesthetic" / track / f"{i}.jsonl"
            for fpath, collector in [(f_align, align_scores), (f_aes, aes_scores)]:
                if fpath.exists():
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        collector.append(float(data["score"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        pass

        align_mean = np.mean(align_scores) * 10 if align_scores else 0.0
        aes_mean = np.mean(aes_scores) * 10 if aes_scores else 0.0
        results[track] = {
            "alignment": align_mean, "aesthetic": aes_mean,
            "avg": (align_mean + aes_mean) / 2,
        }
        all_align.extend([s * 10 for s in align_scores])
        all_aes.extend([s * 10 for s in aes_scores])

    results["overall"] = {
        "alignment": np.mean(all_align) if all_align else 0.0,
        "aesthetic": np.mean(all_aes) if all_aes else 0.0,
    }
    results["overall"]["avg"] = (results["overall"]["alignment"] + results["overall"]["aesthetic"]) / 2
    return results


def print_table(data: dict, variant_name: str, baseline_key: str, display_name: str):
    """Print scores table + delta vs baseline."""
    categories = ["overall"] + TRACKS

    for metric_name, metric in [("Combined Average", "avg"), ("Alignment", "alignment"), ("Aesthetic", "aesthetic")]:
        print(f"\n{'='*90}")
        print(f"  {metric_name} Scores (0-100)")
        print(f"{'='*90}")
        header = f"{'Category':<20}  {display_name:>10}  {baseline_key:>10}  {'Delta':>8}"
        print(header)
        print("-" * 90)
        for c in categories:
            label = c.replace("_", " ").title()
            v = data[variant_name].get(c, {}).get(metric, 0.0)
            b = data[baseline_key].get(c, {}).get(metric, 0.0) if baseline_key in data else 0.0
            d = v - b
            sig = " +" if d > 0.05 else (" -" if d < -0.05 else "  ")
            print(f"{label:<20}  {v:>9.1f}  {b:>9.1f}  {d:+8.1f}{sig}")

    # Summary
    print(f"\n{'='*90}")
    print(f"  Summary: {display_name} vs {baseline_key}")
    print(f"{'='*90}")
    overall_v = data[variant_name].get("overall", {}).get("avg", 0.0)
    overall_b = data[baseline_key].get("overall", {}).get("avg", 0.0) if baseline_key in data else 0.0
    delta_overall = overall_v - overall_b
    print(f"  Overall:    {overall_v:.1f} vs {overall_b:.1f}  (delta={delta_overall:+.1f})")

    for c in _REASONING_TRACKS:
        v = data[variant_name].get(c, {}).get("avg", 0.0)
        b = data[baseline_key].get(c, {}).get("avg", 0.0) if baseline_key in data else 0.0
        d = v - b
        flag = " ↑↑" if d > 0.5 else (" ↓" if d < -0.5 else "")
        print(f"  {c:<12}: {v:.1f} vs {b:.1f}  (delta={d:+.1f}){flag}")

    print(f"  + / - /   : above / below / tied with baseline")


def plot_chart(data: dict, variant_name: str, baseline_key: str,
               display_name: str, output_path: str, suptitle: str):
    """Bar chart: baseline + checkpoint side by side."""
    categories = ["overall"] + TRACKS
    cat_labels = ["Overall", "Imag.", "Entity", "TextR.", "Style", "Affect.", "Comp.", "LongT."]

    x = np.arange(len(categories))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    for ax, metric, title in [
        (ax1, "alignment", "Alignment"),
        (ax2, "aesthetic", "Aesthetic"),
    ]:
        baseline_vals = [data[baseline_key].get(c, {}).get(metric, 0.0) for c in categories] if baseline_key in data else [0]*len(categories)
        variant_vals = [data[variant_name].get(c, {}).get(metric, 0.0) for c in categories]

        ax.bar(x - w/2, baseline_vals, w, label=baseline_key, color="#5B9BD5", edgecolor="white")
        ax.bar(x + w/2, variant_vals, w, label=display_name, color="#ED7D31", edgecolor="white")

        for i, (b_val, v_val) in enumerate(zip(baseline_vals, variant_vals)):
            if b_val > 0.5:
                ax.text(i - w/2, b_val + 0.3, f"{b_val:.1f}", ha="center", fontsize=7.5)
            if v_val > 0.5:
                ax.text(i + w/2, v_val + 0.3, f"{v_val:.1f}", ha="center", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=30, ha="right", fontsize=9)

        for i, tick in enumerate(ax.get_xticklabels()):
            cat_lower = categories[i]
            if cat_lower == _PRIMARY_TRACK:
                tick.set_fontweight("bold"); tick.set_color("#2ca02c")
            elif cat_lower in _REASONING_TRACKS:
                tick.set_fontweight("semibold"); tick.set_color("#1f77b4")

        ax.set_ylabel("Score (0-100)")
        ax.set_title(f"{title} Scores", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved: {output_path}")


def main():
    parser = ArgumentParser(description="Exp6 checkpoint vs baseline bar chart")
    parser.add_argument("--variant", type=str, required=True, help="Variant name")
    parser.add_argument("--scores_dir", type=str, required=True, help="Score directory for the variant")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline scores txt file")
    parser.add_argument("--base_model", type=str, required=True, choices=["sd3", "sd35"],
                        help="Which baseline model to compare: sd3 or sd35")
    parser.add_argument("--output", type=str, required=True, help="Output chart path")
    parser.add_argument("--title", type=str, default="PRISM-Bench: Checkpoint vs Baseline", help="Chart suptitle")
    parser.add_argument("--name", type=str, default=None, help="Display name for the variant")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = SCRIPT_DIR / args.baseline

    scores_path = Path(args.scores_dir)
    if not scores_path.is_absolute():
        scores_path = SCRIPT_DIR / args.scores_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / args.output

    display_name = args.name or args.variant.replace("_", " ").title()
    baseline_key = "SD3.5 Baseline" if args.base_model == "sd35" else "SD3 Baseline"

    # Load baseline (only the matching model)
    data = parse_baseline_file(baseline_path, args.base_model)
    if baseline_key not in data:
        print(f"ERROR: {baseline_key} not found in {baseline_path}")
        return
    print(f"Loaded baseline: {baseline_key} from {baseline_path}")

    # Load variant scores
    data[args.variant] = load_scores(scores_path)
    overall = data[args.variant]["overall"]
    print(f"Loaded variant: {args.variant}  align={overall['alignment']:.1f}  aes={overall['aesthetic']:.1f}")

    print_table(data, args.variant, baseline_key, display_name)
    plot_chart(data, args.variant, baseline_key, display_name, str(output_path), args.title)


if __name__ == "__main__":
    main()
