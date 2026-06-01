# SRDM-DDPO 共享模块 (`srdm_pytorch_exp/`)

所有实验共享的训练基础设施。本文档分两部分：

- **第一部分（读者）**：按训练流程的 7 件事组织，逐件说明哪些文件在干什么
- **第二部分（AI）**：每个 `.py` 文件的函数级 I/O 速查，供新会话快速理解代码

---

# 第一部分：训练流程七件事

一条 prompt 从输入到完成一次梯度下降，共经历 7 个步骤。每步标注了负责文件。

## 事 1：调取 prompt JSON 文件，得到 prompt 和对应的 noun 标签

**文件：`prompts_noun.py`**

```
JSON/JSONL 文件 (含 prompt + objects)
        │
        ▼
  load_prompt_objects()
        │
        ▼
  {prompt_text: ["noun1", "noun2", ...]}
```

从训练数据文件（`.json` 或 `.jsonl`）中读取每条 prompt 及其对应的目标物体名词列表。objects 支持两种格式：`{"noun": count}` (dict) 或 `["noun1", "noun2"]` (list)。也提供 `load_prompts_from_file()` 兼容旧版纯文本 prompt 文件。

## 事 2：调取基准模型，SDE 采样训练，记录 log p

**文件：`sde_sampling.py`（根在 `diffusers_patch/`）**

`diffusers_patch/` 下的两个文件是 `sde_sampling.py` 的底层依赖，不单独调用：

| 底层文件 | 做了什么 |
|----------|---------|
| `diffusers_patch/flow_match_sde.py` | 将 SD3 的确定性 ODE scheduler 替换为 SDE scheduler，注入可控噪声 $a$ |
| `diffusers_patch/pipeline_sd3_logprob.py` | 纯分析用途，跑一次完整去噪并记录每步 log_prob（exp1/exp2_part1 用） |

上层入口是 `sde_sampling.py` 的 `pipeline_sd3_train_sample()`：

```
prompt → encode_prompt() → pipeline_sd3_train_sample()
                                │
                   每步同时跑两个 forward:
                        ├── LoRA θ       → log_probs_old  (训练目标)
                        └── frozen θ_base → log_probs_base (内生奖励信号)
                                │
                                ▼
                   返回: images, all_latents, log_probs_old[T], log_probs_base[T]
```

每步 log p 对 D=65536 维求和（非取均值），精度 float32。种子方案用 stride=1000 隔离不同链，6 条链永不碰撞。

## 事 3：计算 r_in（内生奖励）

**文件：`reward_rin.py`**

```
log_probs_base (per chain)
        │
        ▼
  zscore_normalize() 按 prompt 分组
        │
        ▼
  r_in ∈ ℝ (组内 z-score)
```

每条 prompt 的 M 条链独立做 z-score 归一化：$r_{\text{in}} = (x - \mu_{\text{prompt}}) / \sigma_{\text{prompt}}$。$\sigma \approx 0$ 时返回零向量。消除 prompt 间 log p 量级差异，使奖励仅反映链间相对质量。

## 事 4：将一个 prompt 的所有图并行送入 VLM，生成 bbox

**文件：`vlm_client_noun.py`**

```
一个 prompt 的 M 张图 + 目标 noun 列表
        │
        ▼
  VLMClientNoun.extract_schema(prompt)   ← 从预提取 noun 构建 canonical schema
  VLMClientNoun.extract_structures_batch(images, schema)
        │
        ▼
  ThreadPoolExecutor 并行调用 VLM API（doubao / qwen / qwen_local）
        │
        ▼
  M 个 structure JSON (objects[].count + instances[].bbox)
```

VLM 的职责是纯 API 翻译：接收 (图片, noun 标签列表) → 返回 bbox JSON。**不做** count 向量对齐（那是事 5 的职责）。

**REMARK — bbox 质量检测与坏样本标注：**

`validate_structure_bboxes()` 在 VLM 返回 JSON 后逐一检查每个 instance 的 bbox 是否合法（四个数值、非空、可转 float）。非法 bbox 占比 > 50% → 该链标记为 `json_ok = False`。此外，`_error` 字段、空 objects、非 list 类型也会触发拒绝。被标记为 `json_ok = False` 的链在后续 PPO 优化中被自动过滤。

## 事 5：bbox + noun → 定长 φ 特征向量

**文件：`structure_features.py`**

```
structure JSON (per image) + canonical schema
        │
        ▼
  phi_dicts_simplified()
        │
        ├── φ_count:     [K] 每个 canonical label 的 instance count（定长，缺失补 0）
        ├── φ_coverage:  [1] Top-2 物体的 per-instance intersection ratio
        └── φ_relation:  [2] Top-2 物体质心方向 sign 编码
```

`schema` 是 canonical label 的固定列表（从事 1 的 noun 列表构建）。`phi_count` 输出定长 [K] 向量，每个位置固定对应一个 schema label，缺失标签补 0。这保证了不同图片的 φ 向量维度一致，后续 r_SSR 才能批量计算。

Top-2 按所有链的 total count 排序选出，用于 coverage 和 relation。

也提供 `phi_full()`（三分量拼接）和 bbox 几何工具函数（normalize_bbox, bbox_iou, bbox_area 等）。

## 事 6：计算 φ* 原型并计算 r_SSR（外生结构奖励）

**文件：`reward_ssr.py`（V1）+ `reward_ssr_v2.py`（V2，当前主力）**

### V1（`reward_ssr.py`）

```
φ_dicts (M 条链) + r_in (M)
        │
        ▼
  compute_r_ssr_batch()
        │
        ├── φ* = Σ softmax(r_in/τ)_i · φ_i    (以 r_in 为权重的软原型)
        ├── d_j = L1(φ_j, φ*)                   (每条链到原型的 L1 距离)
        ├── d_combined = λ_count·d_count + λ_coverage·d_coverage + λ_relation·d_relation
        └── r_SSR = -d_combined                 (越近分越高)
```

$r_{\text{in}}$ 大的链贡献更多到原型 φ*，然后所有链向原型靠拢。三分量（count/coverage/relation）的 L1 距离分别 z-score 归一化后加权求和。

### V2（`reward_ssr_v2.py`）

```
φ_dicts (M 条链) + r_in (M)
        │
        ▼
  compute_r_ssr_v2_batch()
        │
        ├── Step 0: 存在性惩罚 — 任意 noun count == 0 → r_SSR_i = -λ_exist
        │
        ├── φ*_count    = mode_prototype: 每个分量取出现次数最多的值
        │                  (1,1,2,3)→1.0  (1,1,2,2)→1.5  (1,2,3,4)→2.5
        ├── φ*_relation = mode_prototype: 同上（sign 值为 -1/0/1）
        ├── φ*_coverage = weighted average: Σ softmax(r_in/τ)_i · φ_i（不变）
        │
        ├── d_count    = mean_k |φ_ik - φ*_k| / max(φ_ik, φ*_k)   (偏差比)
        ├── d_cov      =       |C_i - C*|    / max(C_i, C*)
        ├── d_rel      = mean_k |R_ik - R*_k|                      (纯 L1)
        │
        ├── z-score 归一化 → d_count_norm, d_cov_norm, d_rel_norm
        ├── d_combined = λ_count*d_count_norm + λ_cov*d_cov_norm + λ_rel*d_rel_norm
        └── r_SSR = -d_combined
```

**V2 核心改进**：

1. **存在性惩罚**：任意物体缺失直接给 -λ_exist，不给"部分偏差"的机会
2. **Mode 原型**：φ* 是"多数共识"而非"加权平均"，好处是当两条链同样常见（如 1 和 2 各两条），原型取 1.5，对所有链的 count 奖励对称——优化器不会朝某个方向推
3. **偏差比（max 分母）**：生成 3 个期望 2 个 vs 生成 1 个期望 2 个，惩罚比例都是 1/3÷max=1/3，对称处理过生成和欠生成
4. **Relation 用纯 L1**：sign 值域 {-1,0,1}，不需归一化分母

## 事 7：完成梯度下降优化（PPO Update）

**文件：`ppo_trainer.py`**

```
advantages = r_in + r_SSR (per chain)
        │
        ▼
  ppo_update_mini_batch()
        │
        ├── 过滤 json_ok = False 的坏链
        ├── for t in 0..T-1:
        │     ratio = exp(log_p_new - log_p_old)
        │     loss = -min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
        │     (loss/T).backward()     ← 梯度累积
        ├── clip_grad_norm_(max=5.0)  ← 裁剪 ∇θ L 整体范数
        └── optimizer.step()
```

**核心机制：**

- **PPO min-clip**：ratio 超出 [0.8, 1.2] 的样本步梯度为零，不设 KL 惩罚项
- **AdamW 外部创建**：优化器由实验脚本创建并传入，lr/β/wd 等参数外部控制
- **TrainingAlerter**：滑动窗口双重警报，分别监控 ratio 裁剪率和 grad_norm 裁剪频率

**REMARK — VLM、采样、优化三者并发执行，互不等待：**

在实际训练脚本（如 exp6_part1.py）中，Phase 1 先采样所有 prompt 的所有 chain 并把 VLM 任务提交到 `ThreadPoolExecutor`，Phase 2 逐 prompt 等待 VLM 结果 → 计算 reward → 立即 PPO update。这意味着：

```
时间线:
  prompt 0: [采样] ──→ [VLM 后台跑] ─────────────→ [等 VLM] [reward] [PPO]
  prompt 1: [采样] ──→ [VLM 后台跑] ─────────────→ [等 VLM] [reward] [PPO]
  prompt 2: [采样] ──→ [VLM 后台跑] ─────────────→ [等 VLM] [reward] [PPO]
                         ↑ VLM 在后台并发运行，不阻塞其他 prompt 的采样
```

当 VLM 耗时 > 采样+优化耗时时，VLM 成为瓶颈（PPO 等 VLM）。当优化很快时，必须等待 VLM 结果返回后才能继续。三者不会互相阻塞等待，除非前一步的输出是后一步的输入（数据依赖）。

**REMARK — bbox 坏样本自动过滤，batch 减少超过半数则跳过：**

`ppo_update_mini_batch()` 内部有两层过滤：

1. 检查每条链的 `json_ok` 字段（由事 4 的 `validate_structure_bboxes` 设置），过滤掉 VLM 返回了非法 bbox 的链
2. 若有效链数量 < `min_valid_ratio` (默认 0.5，即超过半数被过滤) → `batch_skipped = True`，整个 mini-batch 不更新，跳过本次优化

exp6 在此之上还有一层预过滤：单 batch 中 bad ≥ 2 条就直接跳过，不进入 `ppo_update_mini_batch()`。

---

# 第二部分：AI 速查 — 每个文件的函数 I/O

以下是每个 `.py` 文件的所有对外函数/类的输入输出，供新会话快速理解代码。

## `prompts_noun.py` — Prompt 与 Noun 标签加载

| 函数 | 输入 | 输出 |
|------|------|------|
| `load_prompts_from_file(file_path)` | `.txt` 文件路径 | `List[str]` |
| `load_prompt_objects(file_path)` | `.json` 或 `.jsonl` 文件路径 | `Dict[str, List[str]]` |

`load_prompt_objects` 自动检测文件类型，支持 objects 为 `dict` 或 `list` 两种格式。

## `diffusers_patch/flow_match_sde.py` — SDE 噪声注入 Scheduler

类 `StochasticFlowMatchScheduler`，继承 `FlowMatchEulerDiscreteScheduler`。唯一对外方法：

| 方法 | 输入 | 输出 |
|------|------|------|
| `step(v_θ, t, x_t, generator, prev_sample)` | 向量场 + 时间步 + latent + 噪声源 | `(x_{t-1}, log_prob)` |

核心公式：$x_{t-1} = x_t + f_\theta \cdot \Delta t + a\sqrt{t/(1-t)} \cdot \sqrt{|\Delta t|} \cdot \varepsilon$。log_prob 全程 float32。

## `diffusers_patch/pipeline_sd3_logprob.py` — 分析用全量 log_prob 记录

| 函数 | 输入 | 输出 |
|------|------|------|
| `pipeline_sd3_with_logprob(pipeline, prompt_embeds, ...)` | pipeline + embeddings + 步数 + CFG + generator | `(images[B,C,H,W], all_latents[T+1], all_log_probs[T+1])` |

## `sde_sampling.py` — 训练采样编排

| 函数 | 输入 | 输出 |
|------|------|------|
| `pipeline_sd3_train_sample(pipeline, base_transformer, prompt_embeds, ...)` | 训练 pipeline + frozen base + embeddings + 步数 + CFG + generators | `(images, all_latents[T+1], log_probs_old[T], log_probs_base[T])` |
| `encode_prompt(pipeline, prompt, device)` | pipeline + text + device | `(pos_embeds, pooled, neg_embeds, neg_pooled)` |
| `make_chain_generators(base_seed, chain_idx, num_steps, device)` | 种子 + 链索引 + 步数 | `(latents_gen, [step_gen_0..step_gen_{T-1}])` |
| `total_log_prob_from_list(log_prob_list)` | `List[Tensor[B]]` | `Tensor[B]` |

## `reward_rin.py` — 内生奖励

| 函数 | 输入 | 输出 |
|------|------|------|
| `zscore_normalize(values)` | `Tensor[M]` | `Tensor[M]`（σ≈0 返回零向量） |
| `compute_reward_rin(total_log_p_base, group_size)` | log_p_base [M] + 每组链数 | `Tensor[M]` |

## `vlm_client_noun.py` — VLM 客户端（唯一入口）

合并了原 `vlm_client.py`。包含 `VLMClient`（spaCy 基类）+ `VLMClientNoun`（预提取名词）+ 工具函数 + benchmark 框架。

### 客户端类

| 类 | 职责 | schema 来源 |
|----|------|------------|
| `VLMClient(backend, model, **kwargs)` | API 调用 / JSON 解析 / 并行批处理 / spaCy 名词提取 | spaCy POS tagging |
| `VLMClientNoun(prompt_objects, backend, model, **kwargs)` | 继承 VLMClient，覆盖 `extract_schema()` | 预提取 `{prompt: [nouns]}` 映射 |

支持三种后端：`doubao`（豆包 Seed API）、`qwen`（Qwen2-VL via vLLM）、`qwen_local`（Qwen2.5-VL 本地）。

### 核心方法（VLMClient / VLMClientNoun 共用）

| 方法 | 输入 | 输出 |
|------|------|------|
| `extract_schema(prompt)` | prompt 文本 | canonical schema dict（含 `canonical_objects` 列表） |
| `extract_structure(image, schema)` | PIL Image + schema | structure JSON（`objects[].count` + `instances[].bbox`） |
| `extract_structures_batch(images, schema, max_workers, stagger_delay, ...)` | PIL Image 列表 + schema + 预处理参数 | `List[structure]`（ThreadPoolExecutor 并行，stagger_delay 防 429） |

### 工具函数

| 函数 | 用途 |
|------|------|
| `validate_structure_bboxes(structure, max_bad_ratio=0.5)` | 校验 bbox 合法性：非法比例 > 0.5 → False，标记坏样本 |
| `draw_structure_annotations(image, structure)` | 在 PIL Image 上绘制 bbox + 质心标注（re-export from vis_utils） |
| `extract_nouns_spacy(prompt)` | spaCy POS tagging 提取名词（去重/词形还原/过滤抽象词） |
| `preprocess_image(image, variant)` | 按 variant 配置预处理（crop/grayscale/resize） |

### Benchmark 框架

| 类/函数 | 用途 |
|---------|------|
| `VLMVariant(key, label, strategy, ...)` | 定义一种加速策略（分辨率/crop/灰度/thinking 开关） |
| `benchmark_variants(image, schema, variants, client)` | 单图所有 variant 对比 |
| `benchmark_variants_batch(images, schema, variants, client, ...)` | 多图并行 benchmark，baseline 先行 |

### JSON 解析容错链

`_extract_json()` → 直接解析 → \`\`\`json fences → 正则匹配 → truncation 修复 → 正则部分提取（label+count 容错）→ None

## `structure_features.py` — 结构特征 φ(G)

纯数学计算，不调用 VLM。

| 函数 | 输入 | 输出 |
|------|------|------|
| `phi_count(structure, schema)` | structure + schema | `Tensor[K]` 每个 canonical label 的 instance count |
| `phi_coverage(structure, schema)` | structure + schema | `Tensor[K(K-1)/2]` 逐实例对 intersection ratio |
| `phi_relation(structure, schema)` | structure + schema | `Tensor[K(K-1)×2]` 质心方向 sign 编码 |
| `phi_full(structure, schema)` | structure + schema | `Tensor[D]` concat 以上三者 |
| `phi_to_dict(structure, schema)` | structure + schema | `{"count": T, "coverage": T, "relation": T, "full": T}` |
| `phi_dicts_simplified(structures, schema)` | `List[structure]` + schema | `(phi_dicts[M], active_nouns, top2, dead_nouns)` |

`phi_count()` 实现定长对齐：只输出 schema 中定义的标签，缺失补 0，多余忽略。

### bbox 几何工具

`normalize_bbox`, `union_bbox`, `bbox_area`, `bbox_intersection`, `bbox_iou`

## `reward_ssr.py` — 结构相似度奖励 r_SSR V1

| 函数 | 输入 | 输出 |
|------|------|------|
| `compute_r_ssr_batch(phi_dicts, r_in_raw, λ_count, λ_coverage, λ_relation, temperature, uniform_weights)` | phi_dicts[M] + r_in[M] + λ权重 + τ | `r_ssr[M]` + `debug_info`（含 φ*原型、L1 距离、z-score 归一化距离） |
| `compute_component_distance_l1(phi_list, weights)` | phi tensors[M, D_c] + softmax weights[M] | `(φ*, distances[M], raw_diffs[M, D_c])` |
| `make_distance_plot(debug_info, ...)` | debug_info from compute_r_ssr_batch | PIL Image（PCA 2D，φ* 为原点，RdYlGn_r 着色） |

计算流程：$φ^* = \sum \text{softmax}(r_{\text{in}}/\tau)_i \cdot φ_i$ → 每分量 L1 距离 → z-score 归一化 → λ 加权求和 → $r_{\text{SSR}} = -d_{\text{combined}}$

## `reward_ssr_v2.py` — 结构相似度奖励 r_SSR V2（当前主力）

| 函数 | 输入 | 输出 |
|------|------|------|
| `compute_r_ssr_v2_batch(phi_dicts, r_in_raw, λ_exist, λ_count, λ_coverage, λ_relation, temperature, uniform_weights)` | phi_dicts[M] + r_in[M] + λ权重 + τ | `r_ssr[M]` + `debug_info`（含 φ*、偏差比、归一化距离） |
| `_mode_prototype(phi_list)` | phi tensors[M, D] | `φ*[D]` 众数原型 |
| `_weighted_average_prototype(phi_list, weights)` | phi tensors[M, D] + weights[M] | `φ*[D]` 加权平均原型 |
| `_component_deviation_ratio(phi_list, phi_star)` | phi tensors[M, D] + pre-computed φ* | `(losses[M], raw_diffs[M, D])` |
| `_plain_l1_distance(phi_list, phi_star)` | phi tensors[M, D] + pre-computed φ* | `(losses[M], raw_diffs[M, D])` |

计算流程：
1. 存在性惩罚：任意 noun count == 0 → `-λ_exist`
2. φ* 区分计算：count/relation → mode 原型，coverage → 加权平均
3. Count/Coverage：`|φ-φ*| / max(φ,φ*)`，Relation：`|R-R*|`
4. z-score 归一化 → λ 加权 → `r_SSR = -d_combined`

## `ppo_trainer.py` — PPO 优化

### 类 `TrainingAlerter`

滑动窗口双重警报（ratio 裁剪率 + grad_norm 裁剪率），参数：`window=10, threshold=3, ratio_bad_pct=0.5`。

| 方法 | 输入 | 输出 |
|------|------|------|
| `check(ratio_clip_rate, grad_clipped)` | 本次 update 的被 clip 比例 + 梯度是否被裁剪 | dict（含 fired/raised/recent_count/is_bad） |

### 函数

| 函数 | 输入 | 输出 |
|------|------|------|
| `compute_log_prob_at_step(pipeline, x_t, x_{t-1}, t, prompt_embeds, pooled_embeds, neg_embeds, neg_pooled, guidance_scale)` | 单样本所有状态 | log_prob 标量（梯度流过 transformer） |
| `ppo_update_mini_batch(pipeline, chain_data, batch_indices, timesteps, advantages, optimizer, ppo_clip_range, max_grad_norm, num_inference_steps, alerter, min_valid_ratio)` | 见下方参数表 | metrics dict（loss, ratio_mean, ratio_clip_rate, grad_norm, batch_skipped, n_bad_in_batch, 警报字段） |

**`ppo_update_mini_batch` 关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ppo_clip_range` | 0.2 | ratio 约束在 [0.8, 1.2] |
| `max_grad_norm` | 5.0 | 梯度范数裁剪阈值 |
| `min_valid_ratio` | 0.5 | 有效链低于此比例 → 跳过整个 batch |

## `vis_utils.py` — 可视化

| 函数 | 输入 | 输出 |
|------|------|------|
| `draw_structure_annotations(image, structure)` | PIL Image + structure JSON | 标注 PIL Image（bbox + 质心 + 标签） |

---

## 实验与模块对应

| 实验 | 采样 | 奖励 | VLM | PPO |
|------|------|------|-----|-----|
| exp1 | `pipeline_sd3_with_logprob` | 无 | 无 | 无 |
| exp2_part1 | `pipeline_sd3_with_logprob` | 内联 z-score | 无 | 无 |
| exp2_part2 | `pipeline_sd3_train_sample` | `compute_reward_rin` | 无 | 内联 |
| exp2_part3 | `pipeline_sd3_train_sample` | tanh 内联 | 无 | 内联 |
| exp3 | `pipeline_sd3_train_sample` | `compute_r_ssr_batch` (分析) | `extract_structures_batch` | 无 |
| exp4 | `pipeline_sd3_train_sample` | `compute_r_ssr_batch` | `extract_structures_batch` + json_ok | `ppo_update_mini_batch` |
| exp5 | `pipeline_sd3_train_sample` | `compute_r_gt_single` (exp5/) | `VLMClientNoun` + GT utils (exp5/) | `ppo_update_mini_batch` |
| exp6 | `pipeline_sd3_train_sample` | `compute_r_ssr_v2_batch` + r_in | `VLMClientNoun` + `load_prompt_objects` | `ppo_update_mini_batch` |

## 文件清单

```
srdm_pytorch_exp/
├── __init__.py
├── ppo_trainer.py              # 事 7: PPO 更新 + TrainingAlerter
├── prompts_noun.py             # 事 1: prompt/objects 加载
├── reward_rin.py               # 事 3: r_in 内生奖励
├── reward_ssr.py               # 事 6: r_SSR V1 结构相似度奖励
├── reward_ssr_v2.py            # 事 6: r_SSR V2 (mode φ* + deviation ratio)
├── sde_sampling.py             # 事 2: SDE 采样编排
├── structure_features.py       # 事 5: φ(G) 定长特征提取
├── vis_utils.py                # bbox/质心可视化
├── vlm_client_noun.py          # 事 4: VLM 客户端 (含 bbox 校验)
└── diffusers_patch/
    ├── flow_match_sde.py       # 事 2 底层: SDE Scheduler
    └── pipeline_sd3_logprob.py # 事 2 底层: 分析用 log_prob 记录
```
