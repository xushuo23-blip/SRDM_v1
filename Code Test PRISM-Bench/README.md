# PRISM-Bench 测评框架

## 目录总览

```
Code Test PRISM-Bench/
├── README.md                           # 本文档
├── prism-bench-main/                   # PRISM-Bench 官方库 (不改)
├── baseline_scores/                    # 基线分数 txt (各 VLM 各一份)
│   ├── baseline_scores_doubao-seed-2.0-pro.txt
│   └── baseline_scores_qwen2.5-vl-7b-instruct.txt
├── baseline_eval/                      # 基线测评 (SD3/SD3.5 base model)
│   ├── gen_baseline_images.py          # Step 1: 生成基线图片
│   ├── images/                         # 图片 (所有 VLM 共用)
│   ├── doubao-seed-2.0-pro/           # Doubao VLM 专属
│   │   ├── run_baseline.sh
│   │   ├── eval_parallel.py
│   │   ├── visualize_baseline.py
│   │   ├── scores/   → 中间分数
│   │   └── results/  → 表格 + 柱状图
│   └── qwen2.5-vl-7b-instruct/        # Qwen VLM 专属
│       ├── run_baseline.sh
│       ├── run_evals.py                # 本地打分入口
│       ├── visualize_baseline.py
│       ├── scores/
│       └── results/
└── Rcombine_exp6_part1/                # 实验: Exp6 checkpoint 测评
    ├── gen_images.py                   # Step 1: 生成 checkpoint 图片
    ├── images/                         # 图片 (所有 VLM 共用)
    └── doubao-seed-2.0-pro/           # Doubao VLM 专属 (完全自包含)
        ├── run_exp6.sh                 # 一键打分+可视化
        ├── eval_parallel.py            # API 并行打分 (从 baseline_eval 复制)
        ├── visualize_exp6.py           # 柱状图 vs baseline (含 --base_model)
        ├── scores/   → 中间分数
        └── results/  → 图表
```

**核心原则**:
1. **按 VLM 模型分文件夹** — 第三方 VLM 叫什么名字，文件夹就叫什么
2. **scores 不在 images 里** — 打分结果存 VLM 文件夹下的 `scores/`，与图片分离，不再需要删旧分数
3. **VLM 文件夹名用小写完整名** — `doubao-seed-2.0-pro`、`qwen2.5-vl-7b-instruct`
4. **每个实验完全自包含** — 实验的 VLM 文件夹从 baseline_eval 复制三件套 (`eval_parallel.py` + `visualize_expN.py` + `run_exp.sh`)，不跨目录引用，通过 `--base_model sd3|sd35` 指定比对基线

---

## 场景一: 生成新 VLM 的基线分数

当你有了一个新的 VLM 评估器（如 Doubao、Qwen），需要生成它视角下的 SD3/SD3.5 基线分数。

### 1. 创建 VLM 文件夹

```bash
mkdir -p baseline_eval/<vlm_name>/{scores,results}
```

### 2. 放入三件套

从已有 VLM 文件夹复制：

| 文件 | 作用 | 需要改什么 |
|------|------|-----------|
| `run_baseline.sh` | 驱动三步流程 | `EVALUATOR`, `EVALUATOR_LOWER`, API 配置 |
| `eval_parallel.py` | API 并行打分 | `--api_model` 默认值 |
| `visualize_baseline.py` | 表格 + 柱状图 | `EVALUATOR`, `MODEL_NAME` |

- **API VLM** (如 Doubao): 三个文件都要

### 3. 运行

```bash
bash baseline_eval/<vlm_name>/run_baseline.sh
```

### 4. 输出

- `baseline_eval/<vlm_name>/scores/` — 中间打分 JSON
- `baseline_eval/<vlm_name>/results/baseline_scores_<vlm_name>.txt` — 基线分数表
- `baseline_eval/<vlm_name>/results/baseline_comparison_<vlm_name>.png` — SD3 vs SD3.5 柱状图

基线 txt 文件也应拷贝一份到根目录 `baseline_scores/` 下，方便后续实验引用。

---

## 场景二: 测评训练好的 checkpoint

当你有一个训练好的 checkpoint，想在 PRISM-Bench 上打分并与基线对比。

### 速查: 用户语言 → 变量

当你说 **"给 exp6 part1 的 sd35 做 Doubao 测评"**：

| 你说的 | 对应变量 | 值 |
|--------|---------|-----|
| exp6 part1 | `exp_dir` | `Rcombine_exp6_part1` |
| sd35 | `BASE_MODEL` | `sd35` |
| Doubao | VLM 文件夹 | `doubao-seed-2.0-pro` |

当你说 **"给 exp7 part2 的 sd3 做 Qwen 测评"**：

| 你说的 | 对应变量 | 值 |
|--------|---------|-----|
| exp7 part2 | `exp_dir` | `Rcombine_exp7_part2` |
| sd3 | `BASE_MODEL` | `sd3` |
| Qwen | VLM 文件夹 | `qwen2.5-vl-7b-instruct` |

**通用规则**: 用户说 "exp N part M" → 目录名 `Rcombine_exp{N}_part{M}`；"sd3/sd35" → `--base_model sd3/sd35`；"Doubao/Qwen" → 从 `baseline_eval/<vlm_name>/` 复制三件套。

### 1. 创建实验目录

```bash
mkdir -p <exp_dir>/<vlm_name>/{scores,results}
```

目录结构：
```
<exp_dir>/
├── gen_images.py                    # Step 1: 生成图片
├── images/                          # 生成图片存放处
│   └── <variant_name>/
└── <vlm_name>/                      # VLM 专属, 完全自包含 (可以有多个 VLM)
    ├── run_exp.sh                   # 一键打分+可视化
    ├── eval_parallel.py             # API 并行打分 (从 baseline_eval 复制)
    ├── visualize_expN.py            # 柱状图 vs baseline
    ├── scores/                      # 中间打分
    └── results/                     # 最终图表
```

### 2. 从 baseline_eval 复制三件套

```bash
# 复制模板文件到实验 VLM 文件夹
cp baseline_eval/<vlm_name>/eval_parallel.py <exp_dir>/<vlm_name>/
cp baseline_eval/<vlm_name>/visualize_baseline.py <exp_dir>/<vlm_name>/visualize_expN.py
cp baseline_eval/<vlm_name>/run_baseline.sh <exp_dir>/<vlm_name>/run_exp.sh
```

`eval_parallel.py` 无需修改，`visualize_expN.py` 和 `run_exp.sh` 按下面模板改。

### 3. 写入 `gen_images.py`

参考 `Rcombine_exp6_part1/gen_images.py` 或 `baseline_eval/gen_baseline_images.py`。

你需要改的部分：
```python
VARIANTS = [
    # (variant_name,  base_model_path,  checkpoint_path)
    ("sd35_exp6_epoch300", SD35_MODEL_PATH, "logs_checkpoints/sd35_xxx/epoch_300.pt"),
]
```

- Variant 名前缀决定基线匹配：`sd35_*` → SD3.5 Baseline，`sd3_*` → SD3 Baseline
- `base_model_path`: SD3 用 `SD3_MODEL_PATH`，SD3.5 用 `SD35_MODEL_PATH`
- `checkpoint_path`: 设为 `None` 表示不加载（baseline）

### 4. 写入 `visualize_expN.py` — 自包含柱状图

从 `Rcombine_exp6_part1/doubao-seed-2.0-pro/visualize_exp6.py` 复制，核心参数：

```
--variant      Variant 名 (如 sd35_exp6_epoch300)
--scores_dir   打分 JSON 目录
--baseline     baseline_scores/baseline_scores_<vlm>.txt
--base_model   sd3 或 sd35 (决定从基线文件取哪个模型的分数)
--name         图表中显示的 checkpoint 名称
--output       图表输出路径 (.png)
--title        图表标题
```

`--base_model` 是关键参数：显式指定要对比的基线模型（`sd3` 或 `sd35`），脚本从基线 txt 中只提取对应的分数，不靠前缀自动匹配。

### 5. 写入 `<vlm_name>/run_exp.sh`

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRISM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# === VLM 配置 ===
EVALUATOR_LOWER="doubao-seed-2.0-pro"           # 小写名，用于文件路径
MODEL_NAME="doubao-seed-2-0-pro-260215"          # API 模型名
BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
# ================

VARIANT="sd35_exp6_epoch300"                     # 与 gen_images.py 一致
BASE_MODEL="sd35"                                # sd3 或 sd35
DISPLAY_NAME="SD3.5 r_in+r_SSR Epoch 300"        # 图表显示名称
IMAGES_DIR="<exp_dir>/images"
SCORES_DIR="<exp_dir>/${EVALUATOR_LOWER}/scores"
RESULTS_DIR="<exp_dir>/${EVALUATOR_LOWER}/results"
BASELINE_FILE="$PRISM_DIR/baseline_scores/baseline_scores_${EVALUATOR_LOWER}.txt"

cd "$PRISM_DIR"

# Step 1: 打分 (本地 eval_parallel.py, 自动跳过已有分数)
SCORE_OUT="$SCORES_DIR/${VARIANT}_${EVALUATOR_LOWER}"
python <exp_dir>/${EVALUATOR_LOWER}/eval_parallel.py \
    --image_path "$IMAGES_DIR/$VARIANT" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --api_model "$MODEL_NAME" \
    --output_dir "$SCORE_OUT"

# Step 2: 可视化 (实验自带的 visualize_expN.py)
mkdir -p "$RESULTS_DIR"
python <exp_dir>/${EVALUATOR_LOWER}/visualize_expN.py \
    --variant "$VARIANT" \
    --scores_dir "$SCORE_OUT" \
    --baseline "$BASELINE_FILE" \
    --base_model "$BASE_MODEL" \
    --name "$DISPLAY_NAME" \
    --output "$RESULTS_DIR/comparison.png" \
    --title "<图表标题>" \
    2>&1 | tee "$RESULTS_DIR/scores_table.txt"
```

**关键变量**：

| 变量 | 说明 |
|------|------|
| `EVALUATOR_LOWER` | VLM 小写名，与文件夹名一致 |
| `VARIANT` | 图片子目录名 |
| `BASE_MODEL` | `sd3` 或 `sd35`，显式指定对比哪个基线 |
| `BASELINE_FILE` | 引用 `baseline_scores/` 下的基线 txt |
| `SCORES_DIR` | 打分输出位置 (不在 images 里) |

### 6. 运行

```bash
# 先确保 API Key
export ARK_API_KEY=xxx

# 生成图片 (GPU)
python <exp_dir>/gen_images.py

# 打分 + 可视化
bash <exp_dir>/<vlm_name>/run_exp.sh
```

---

## 关于删除旧分数

**不再需要。** 以前 scores 放在 `images/*/score/` 里，切 VLM 需要删掉重跑。现在 scores 放在各 VLM 文件夹的 `scores/` 下，不同 VLM 的分数互不干扰，`eval_parallel.py` 内部也已支持断点续跑（已有 score JSON 自动跳过）。

---

## 评分尺度

- VLM 输出 **0-10 分**
- `visualize_baseline.py` 和实验的 `visualize_expN.py` 自动 **×10 → 0-100**
- 基线文件 (`baseline_scores/*.txt`) 已是 0-100，无需转换
