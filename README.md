# SRDM: Self-Rewarding Diffusion Model

基于自奖励机制的文本到图像扩散模型优化。核心思想：**让扩散模型在训练过程中通过内生奖励（r_in）和外生结构奖励（r_SSR）自我评估生成质量，并用 PPO/DPO 优化 LoRA 参数，从而提升文本-图像对齐能力**。

---

## 📁 仓库结构

```
第6期：初期SRDM代码/
├── README.md                          # 本文档
├── SRDM训练实验/
│   └── Code Train Flux-reason-6M/     # 训练代码
│       ├── SRDM-DDPO/                 # DDPO (PPO-based) 训练框架 ★ 主力
│       │   ├── config/                # 各实验超参数配置
│       │   ├── experiments/           # 训练脚本 (exp1 ~ exp6)
│       │   ├── srdm_pytorch_exp/      # 共享训练基础设施
│       │   ├── scripts/               # 诊断工具
│       │   └── data/                  # 训练 prompt 数据
│       ├── SRDM-DPO/                  # DPO (Direct Preference Optimization) 训练
│       │   ├── config/                # DPO 实验配置
│       │   ├── experiments/           # DPO 训练脚本
│       │   └── srdm_pytorch_exp/      # DPO 专用模块
│       └── VLM结构图识别方法说明.md    # VLM 结构识别方案
└── SRDM测评实验/
    └── Code Test PRISM-Bench/         # PRISM-Bench 测评代码
        ├── prism-bench-main/          # PRISM-Bench 官方评测库
        ├── baseline_eval/             # 基线测评 (SD3/SD3.5)
        ├── baseline_scores/           # 各 VLM 下的基线分数
        ├── Rcombine_exp6_part1/       # Exp6 P1 checkpoint 测评
        ├── Rcombine_exp6_part2/       # Exp6 P2 checkpoint 测评
        ├── Rgt_exp5_part1/            # Exp5 GT 奖励测评
        ├── Rin_exp2/                  # Exp2 内生奖励测评
        ├── Rssr_exp4_part1/           # Exp4 r_SSR 测评 (SD3.5)
        ├── Rssr_exp4_part2/           # Exp4 r_SSR 测评 (SD3)
        ├── Rssr_exp4_part3/           # Exp4 r_SSR 测评 (SD3.5 扩展)
        └── README_测评指标说明.md       # 测评指标详细说明
```

---

## 🧠 SRDM 是什么？

SRDM (Self-Rewarding Diffusion Model) 在 Stable Diffusion 3/3.5 的基础上，用强化学习的方法微调 Transformer 的 LoRA 参数。核心特点是**不需要人工标注数据**：

| 奖励信号 | 类型 | 来源 | 作用 |
|----------|------|------|------|
| **r_in** | 内生奖励 | frozen base model 的 log-likelihood | 评估扩散链的"自然程度"，引导生成质量 |
| **r_SSR** | 外生奖励 | VLM + 结构特征提取 | 评估物体数量/空间关系的结构一致性 |

训练流程（一条 prompt → 一次梯度更新）：

```
Prompt → SDE 采样 M 条链 → VLM 检测物体 bbox → 提取 φ 特征
    → r_in (z-score of log_p_base) + r_SSR (距离 mode prototype)
    → PPO update (clip ratio, 梯度累积)
```

---

## 🏋️ 训练实验 (SRDM训练实验)

### SRDM-DDPO（主力，PPO 算法）

基于 DDPO (Denoising Diffusion Policy Optimization) 的 PPO 训练框架。包含 6 个实验的渐进迭代：

| 实验 | 奖励信号 | 训练数据 | 核心改进 |
|------|----------|----------|----------|
| Exp1 | 无奖励 | SD3 | 验证 SDE 采样的 log_prob 计算正确性 |
| Exp2 | r_in (内生) | SD3 | 引入内生奖励，验证 PPO 训练可行性 |
| Exp3 | r_SSR 分析 | SD3 | 离线分析 φ 特征分布，调试 r_SSR 计算 |
| Exp4 | r_SSR (V1) | SD3/SD3.5 | 首次引入外生结构奖励，PPO 优化 |
| Exp5 | r_gt + r_in | SD3.5 | 尝试 GT 边界框信号（不稳定） |
| **Exp6** ★ | **r_in + r_SSR (V2)** | SD3.5 | **组合奖励首次实现 Composition 正向突破** |

#### 共享训练基础设施 (`srdm_pytorch_exp/`)

```
srdm_pytorch_exp/
├── ppo_trainer.py              # PPO 更新 + TrainingAlerter
├── prompts_noun.py             # Prompt 与 noun 标签加载
├── reward_rin.py               # r_in 内生奖励 (z-score)
├── reward_ssr.py               # r_SSR V1 (softmax 加权原型)
├── reward_ssr_v2.py            # r_SSR V2 (mode 原型 + deviation ratio)
├── sde_sampling.py             # SDE 采样编排
├── structure_features.py       # φ(G) 特征提取 (count/coverage/relation)
├── vlm_client_noun.py          # VLM 客户端 (Doubao/Qwen)
├── vis_utils.py                # bbox 可视化
└── diffusers_patch/
    ├── flow_match_sde.py       # SDE Scheduler (可控噪声 a)
    └── pipeline_sd3_logprob.py # 分析用全量 log_prob 记录
```

> 详细的模块文档见 `SRDM-DDPO/srdm_pytorch_exp/README.md`

#### VLM 结构识别方案

采用 **三阶段纯 API 方法**，让 VLM 识别生成图像中的物体结构：

1. **Phase 1 — LLM 预提取名词**：离线用 LLM 将所有 prompt 中的名词提取为 `_gt.jsonl`，训练时直接查表
2. **Phase 2 — VLM 检测 bbox**：将生成的图片 + schema 标签列表发给 VLM（Doubao），只检测 schema 里的物体
3. **Phase 3 — 纯数学打分**：从 count/bbox 计算 φ_count、φ_coverage、φ_relation，与 mode prototype 对比

> 详见 `VLM结构图识别方法说明.md`

### SRDM-DPO（Pairwise DPO 算法）

将 DPO (Direct Preference Optimization) 应用于扩散模型。核心创新是**单步 velocity matching DPO**，避免回传整条 SDE trajectory 导致 OOM：

- **Part1**：标准 Pairwise DPO（每 prompt 构造 3 对 preference pairs）
- **Part2**：极值对 + Batched DPO（只用 best/worst 极值对，更稳定）

> 详见 `SRDM-DPO/README_DPO.md`

---

## 📊 测评实验 (SRDM测评实验)

### PRISM-Bench 测评框架

使用 [PRISM-Bench](https://flux-reason-6m.github.io/)（ICLR 2026）对 SRDM 训练 checkpoint 进行七维度评测：

| 维度 | 英文 | SRDM 关注度 | 说明 |
|------|------|-------------|------|
| **Composition** | composition | ★★★ 首要 | 物体空间关系的正确性 |
| **Entity** | entity | ★★ 次要 | 单个物体的类型/数量 |
| **Long Text** | long_text | ★ 第三 | 复杂长 prompt 的全局理解 |
| Imagination | imagination | 参考 | 超现实/幻想场景 |
| Style | style | 参考 | 艺术风格还原 |
| Affection | affection | 参考 | 情绪/氛围表达 |
| Text Rendering | text_rendering | 参考 | 图中文字拼写 |

### 各实验 Composition + Entity 结果总览

以下均以 **Alignment** 分（与 prompt 的对齐度）为基准：

| 实验 | Reward | Baseline | Composition Δ | Entity Δ | 结论 |
|------|--------|----------|---------------|----------|------|
| Exp2 P2 E200 | r_in Z-Score | SD3 | **-0.4** | **-0.9** | 纯内生奖励不提升对齐 |
| Exp4 P3 E100 | r_SSR | SD3.5 | **0.0** | **-2.3** | 一致性不提升准确性 |
| Exp5 P1 E50 | r_gt+r_in | SD3.5 | **-8.0** | **-11.8** | GT 信号不可靠，崩溃 |
| **Exp6 P1 E300** | **r_in+r_SSR** | SD3.5 | **+2.6** | **-3.2** | **Composition 首次正向突破 ★** |

### Exp6 详细分析（当前最优）

vs SD3.5 Baseline (Alignment)：

| 维度 | Δ | 解读 |
|------|---|------|
| **composition** | **+2.6** | SRDM 首要目标首次转正 |
| long_text | +0.9 | 第三目标轻微提升 |
| text_rendering | +3.1 | 无关维度，意外提升 |
| entity | -3.2 | 第二目标：r_SSR 权重过大导致 count 准确性退化 |
| style | -1.6 | 无关维度，轻微退化 |

**关键结论**：r_in + r_SSR 组合奖励的设计在 Composition（SRDM 核心目标）上首次取得了正收益，验证了方法的有效性。Entity 退化是已知问题，后续改进方向是引入 entity-level 对齐约束。

> 详细的测评指标说明见 `SRDM测评实验/README_测评指标说明.md`

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.0+
- Stable Diffusion 3 / 3.5 Medium
- LoRA (rank=8, PEFT)
- VLM API 访问 (Doubao / Qwen2.5-VL)

### 训练 (DDPO + r_in + r_SSR V2)

```bash
# 进入训练目录
cd SRDM训练实验/Code\ Train\ Flux-reason-6M/SRDM-DDPO

# 运行 Exp6 训练（组合奖励，当前最优配置）
python experiments/exp6/exp6_part1.py --config config/exp6_part1_config.py

# 从 checkpoint 恢复
python experiments/exp6/exp6_part1.py --config config/exp6_part1_config.py --resume 300
```

**关键超参数**：
- `ppo_clip_range`: 0.2
- `max_grad_norm`: 5.0
- `λ_count`: 1.0, `λ_coverage`: 0.5, `λ_relation`: 0.5 (r_SSR V2)
- LoRA rank: 8, alpha: 16
- 精度: bf16

### 测评 (PRISM-Bench)

```bash
# 进入测评目录
cd SRDM测评实验/Code\ Test\ PRISM-Bench

# Step 1: 生成 checkpoint 图片 (GPU)
python <exp_dir>/gen_images.py

# Step 2: VLM 打分 + 可视化
bash <exp_dir>/<vlm_name>/run_exp.sh
```

---

## 📝 核心发现总结

1. **组合奖励 (r_in + r_SSR V2) 有效**：Exp6 首次在 Composition 上取得 +2.6 的正向提升，证明内生+外生奖励的组合策略优于单一奖励
2. **内生奖励偏向审美**：纯 r_in 训练提升 Aesthetic（视觉质量 ↑），但不改善 Alignment（文本对齐），需要外部对齐信号
3. **KL 正则化是安全基础**：无 KL 约束时 PPO 会在 100 epoch 内崩溃；β=0.1 + 低 lr 是可行配置
4. **Entity 退化是已知局限**：r_SSR 的一致性信号（2×权重）压过了 r_in 的准确性信号，导致 count 精度下降，后续需引入 entity-level 对齐约束

---

## 📚 参考资源

- [FLUX-Reason-6M & PRISM-Bench 论文 (ICLR 2026)](https://arxiv.org/pdf/2509.09680)
- [PRISM-Bench 官方仓库](https://flux-reason-6m.github.io/)
- [FLUX-Reason-6M 数据集 (HuggingFace)](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M)
- [DDPO 参考实现](https://github.com/kvablack/ddpo-pytorch)

---

## 📄 子模块文档索引

| 文档 | 位置 |
|------|------|
| DDPO 共享模块详细文档 | `SRDM训练实验/Code Train Flux-reason-6M/SRDM-DDPO/srdm_pytorch_exp/README.md` |
| DPO 训练代码解读 | `SRDM训练实验/Code Train Flux-reason-6M/SRDM-DPO/README_DPO.md` |
| VLM 结构图识别方法 | `SRDM训练实验/Code Train Flux-reason-6M/VLM结构图识别方法说明.md` |
| PRISM-Bench 测评框架 | `SRDM测评实验/Code Test PRISM-Bench/README.md` |
| 测评指标详细说明 | `SRDM测评实验/README_测评指标说明.md` |
| 实验二结论 (r_in) | `SRDM测评实验/1. R_in/exp2_old/exp2_conclusion.md` |
