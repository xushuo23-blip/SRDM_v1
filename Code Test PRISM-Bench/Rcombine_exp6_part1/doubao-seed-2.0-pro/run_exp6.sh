#!/bin/bash
# Exp6 Part1: SD3.5 r_in+r_SSR Epoch 300 — Doubao 打分 + 柱状图可视化
#
# Usage (from Code Test PRISM-Bench/):
#   bash Rcombine_exp6_part1/doubao-seed-2.0-pro/run_exp6.sh
#   (运行时输入 ARK_API_KEY，不回显)
#
# 前提: images/sd35_exp6_epoch300/ 图片已存在
#
# Output:
#   Rcombine_exp6_part1/doubao-seed-2.0-pro/scores/sd35_exp6_epoch300_doubao-seed-2.0-pro/
#   Rcombine_exp6_part1/doubao-seed-2.0-pro/results/comparison_epoch300.png
#   Rcombine_exp6_part1/doubao-seed-2.0-pro/results/scores_table.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRISM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# === Config ===
EVALUATOR="Doubao-seed-2.0-pro"
EVALUATOR_LOWER="doubao-seed-2.0-pro"
MODEL_NAME="doubao-seed-2-0-pro-260215"
BASE_URL="https://ark.cn-beijing.volces.com/api/v3"

VARIANT="sd35_exp6_epoch300"
BASELINE_MODEL="SD3.5 Baseline"
# ==============

# 运行时输入 API Key
if [ -z "$ARK_API_KEY" ]; then
    echo -n "Enter ARK_API_KEY: "
    read -s API_KEY
    echo ""
    if [ -z "$API_KEY" ]; then
        echo "ERROR: API Key cannot be empty."
        exit 1
    fi
else
    API_KEY="$ARK_API_KEY"
    echo "Using ARK_API_KEY from environment."
fi

IMAGES_DIR="Rcombine_exp6_part1/images"
SCORES_DIR="Rcombine_exp6_part1/doubao-seed-2.0-pro/scores"
RESULTS_DIR="Rcombine_exp6_part1/doubao-seed-2.0-pro/results"
WORKERS=12
NAMES="${VARIANT}=SD3.5 r_in+r_SSR Epoch 300"
TITLE="Exp6 Part1 r_in+r_SSR Epoch 300 vs SD3.5 Baseline (Doubao-seed-2.0-pro)"

cd "$PRISM_DIR"

echo "========================================"
echo "  Exp6 Part1: r_in+r_SSR Epoch 300 — SD3.5"
echo "  Evaluator: $EVALUATOR ($MODEL_NAME)"
echo "  Workers: $WORKERS"
echo "========================================"

# ============================================================
# Step 1/2: Doubao 打分
# ============================================================
echo ""
echo "========================================"
echo "  Step 1/2: Doubao 打分 (1 variant x 700 imgs x 2 = 1400 calls)"
echo "========================================"

SCORE_OUT="$SCORES_DIR/${VARIANT}_${EVALUATOR_LOWER}"

# eval_parallel.py auto-skips existing scores, no delete needed
python Rcombine_exp6_part1/doubao-seed-2.0-pro/eval_parallel.py \
    --image_path "$IMAGES_DIR/$VARIANT" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --api_model "$MODEL_NAME" \
    --workers $WORKERS \
    --output_dir "$SCORE_OUT"

# ============================================================
# Step 2/2: 柱状图可视化 (对比 baseline)
# ============================================================
echo ""
echo "========================================"
echo "  Step 2/2: 柱状图可视化 (vs $BASELINE_MODEL)"
echo "========================================"

BASELINE_FILE="$PRISM_DIR/baseline_scores/baseline_scores_${EVALUATOR_LOWER}.txt"
mkdir -p "$RESULTS_DIR"

python Rcombine_exp6_part1/doubao-seed-2.0-pro/visualize_exp6.py \
    --variant "$VARIANT" \
    --scores_dir "$SCORE_OUT" \
    --baseline "$BASELINE_FILE" \
    --base_model sd35 \
    --name "SD3.5 r_in+r_SSR Epoch 300" \
    --output "$RESULTS_DIR/comparison_epoch300.png" \
    --title "$TITLE" \
    2>&1 | tee "$RESULTS_DIR/scores_table.txt"

echo ""
echo "========================================"
echo "  全部完成!"
echo ""
echo "  Scores:  $SCORE_OUT/"
echo "  Chart:   $RESULTS_DIR/comparison_epoch300.png"
echo "  Table:   $RESULTS_DIR/scores_table.txt"
echo ""
echo "  柱状图标注:"
echo "    Composition — 绿色加粗 (SRDM 首要目标)"
echo "    Entity      — 蓝色半粗 (SRDM 第二目标)"
echo "    Long Text   — 蓝色半粗 (SRDM 第三目标)"
echo "========================================"
