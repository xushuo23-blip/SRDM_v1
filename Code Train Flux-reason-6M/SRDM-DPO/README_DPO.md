# SRDM-DPO: Pairwise SDE-DPO 代码解读

## 1. 算法总览

### Part1: 标准 Pairwise DPO

```
每个 epoch (3 prompts, 3 optimizer steps):
  for 3 个随机 prompt:
    1. 用当前 LoRA 模型 + SDE scheduler 采样 6 条轨迹 (no_grad)
    2. 算 r_in = z-score(log_p_base) → 6 条链排序
    3. 构造 3 对: (1st,6th), (2nd,5th), (3rd,4th)
    4. DPO update: 单步 velocity matching loss → backward → optimizer.step()
```

### Part2: 极值对 + Batched DPO (更稳定)

```
每个 epoch (6 prompts, 2 optimizer steps):
  1. 采样 6 prompts × 6 chains = 36 条轨迹 (no_grad)
  2. 每个 prompt: z-score → 只取 (1st, 6th) 极值对
  3. Batch 1: prompts 0,1,2 → 3 extreme pairs → DPO backward → step
  4. Batch 2: prompts 3,4,5 → 3 extreme pairs → DPO backward → step
```

**Part2 设计动机**: 只用最极端的好/坏样本作为训练信号，去掉中间对 (2nd,5th)、(3rd,4th) 的噪声，提升训练稳定性。3 个 prompt 组成一个 batch，每个 epoch 执行 2 次 optimizer.step()。

**关键设计**: 不回传整个 SDE trajectory（会 OOM），而是只回传**单步 rectified flow interpolation**。SDE 采样得到的 x_0 当固定数据用，DPO 阶段随机抽一个 t 算 velocity MSE。

---

## 2. 文件结构

```
SRDM-DPO/
├── srdm_pytorch_exp/
│   ├── dpo_trainer.py              # DPO loss 核心
│   ├── sde_sampling.py             # SDE 采样 (从 SRDM-DDPO 复制)
│   ├── reward_rin.py               # r_in = z-score of log_p_base
│   └── diffusers_patch/
│       └── flow_match_sde.py       # SDE scheduler
├── config/
│   ├── exp1_part1_dpo_config.py    # Part1 配置: 3 prompts, 3 pairs/prompt
│   └── exp1_part2_dpo_config.py    # Part2 配置: 6 prompts, extreme pair only, batch=3
└── experiments/exp1/
    ├── exp1_part1_dpo.py           # Part1 训练脚本
    └── exp1_part2_dpo.py           # Part2 训练脚本 (极值对 + batched DPO)
```

---

## 3. Phase 1: SDE 采样 (Rollout)

**文件**: `srdm_pytorch_exp/sde_sampling.py` → `pipeline_sd3_train_sample()`

```
输入: prompt c
1. 用 SDE scheduler 从 t=1 走到 t=0 (20 步)
2. 每步同时跑 trainable transformer (v_θ) 和 base transformer (v_ref)
3. 返回: 最终图片, all_latents[x_T..x_0], log_probs_old, log_probs_base
```

**SDE 公式**:
```
σ_t = a · √(t/(1-t))
f_θ = v_θ + σ_t²/(2t) · (x_t + (1-t)·v_θ)
x_{t-Δt} = x_t + f_θ·Δt + σ_t·√Δt·ε,  ε～N(0,I)
log_prob = -||x_{t-Δt} - μ||²/(2σ²) - D·log(σ) - D/2·log(2π)
```

采样全程在 `@torch.no_grad()` 下进行，不存梯度。

---

## 4. Phase 2: r_in Reward

**文件**: `srdm_pytorch_exp/reward_rin.py` → `zscore_normalize()`

```python
r_in_i = (log_p_base_i - mean(log_p_base)) / std(log_p_base)
```

- `log_p_base` = frozen reference model 对当前轨迹的 log_prob（sum over T steps）
- z-score 取在 6 条链之间：log_p_base 高的 → r_in 高 → "好"链
- **只用 reference model 判断质量**，不涉及 VLM 或 r_SSR

---

## 5. Phase 3: Pair Construction

6 条链按 r_in 排序：
```
r_in: [2.1, 1.5, 0.8, -0.3, -1.2, -2.9]
        ↓    ↓    ↓     ↓     ↓     ↓
      1st  2nd  3rd   4th   5th   6th
```

构造 3 对 (winner, loser)：
```
pair 1: (1st, 6th)  → Δr = 2.1 - (-2.9) = 5.0
pair 2: (2nd, 5th)  → Δr = 1.5 - (-1.2) = 2.7
pair 3: (3rd, 4th)  → Δr = 0.8 - (-0.3) = 1.1
```

---

## 6. Phase 4: DPO Update (核心)

**文件**: `srdm_pytorch_exp/dpo_trainer.py` → `dpo_update()`

### 6.1 为什么不能用 trajectory log_prob？

旧方案尝试回传 20 步 SDE:
```
log p_θ(τ|c) = Σ_{t=1}^{T} log p_θ(x_{t-1}|x_t, c)
```
每个 transformer forward 都存激活 → 20x 显存 → OOM。

### 6.2 新方案: 单步 Velocity Matching DPO

**思路**: SDE rollout 只用来产生 x_0（好/坏图）。DPO 阶段随机抽一个 t，用 rectified flow 插值构造 x_t，直接比两个模型的 velocity prediction 误差。

```
对每个 (winner, loser) pair:

1. 取 SDE trajectory 的最终 latent: x_0^w, x_0^l

2. 随机抽 t ~ U(0, 1)

3. 构造 rectified flow interpolation:
   x_t = (1-t) · x_0 + t · ε,  ε～N(0,I)
   
4. 目标 velocity (rectified flow 的真值):
   u_t = ε - x_0

5. Trainable model 预测:
   ℓ_θ = ||u_t - v_θ(x_t, t, c)||²  (mean over all dims)

6. Reference model 预测 (no_grad):
   ℓ_ref = ||u_t - v_ref(x_t, t, c)||²

7. DPO loss (single pair):
   Δ = (ℓ_θ^w - ℓ_ref^w) - (ℓ_θ^l - ℓ_ref^l)
   L_pair = -log σ(-β · T · Δ)

8. 3 对平均 → loss.backward() → clip_grad → optimizer.step()
```

### 6.3 Loss 直觉

| 场景 | ℓ_θ - ℓ_ref | 含义 |
|------|------------|------|
| Winner, 模型进步 | 负值 (ℓ_θ < ℓ_ref) | 模型对 winner 的 velocity 预测更准了 |
| Loser, 模型不退 | 接近 0 | 模型对 loser 保持 baseline 水平 |

如果 winner 比 loser 进步更多:
```
(ℓ_θ^w - ℓ_ref^w) - (ℓ_θ^l - ℓ_ref^l) < 0
→ -βT · [负值] > 0
→ σ(正) ≈ 1
→ -log(1) ≈ 0  ✓ loss 低
```

### 6.4 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| β | 1.0 | DPO temperature，适配 velocity MSE 量级 |
| T | 20 | num_inference_steps，作为 scaling factor |
| max_grad_norm | 5.0 | 梯度裁剪，防止爆炸 |
| K_t | 1 | 每对采几个 t（当前仅 1 个，可增大） |

### 6.5 内存优势

| | 旧方案 (trajectory log_prob) | 新方案 (velocity matching) |
|---|---|---|
| Transformer forward/update | 20 × 2 × 3 = 120 次 (with grad) | 2 × 3 = 6 次 (with grad) |
| 激活存储 | 20 步全部保留 | 1 步 |
| 显存 | O(T) → OOM | O(1) |
| 额外 no_grad forward | 0 | 2 × 3 = 6 次 (ref model) |

---

## 7. 模型架构

```
┌─────────────────────────────────────────┐
│           SD3.5 Pipeline (pipe)          │
│  ├── VAE (frozen)                       │
│  ├── Text Encoders x3 (frozen)          │
│  │   CLIP-L, CLIP-G, T5                 │
│  ├── Transformer (LoRA trainable)       │
│  └── SDE Scheduler (a=0.7)              │
├─────────────────────────────────────────┤
│  Base Transformer (frozen reference)     │
│  SD3Transformer2DModel only, no VAE/TE  │
└─────────────────────────────────────────┘
```

- **LoRA**: rank=8, alpha=16, target=Q/K/V/out projections
- **Precision**: bf16 (SD3 原生)
- **CFG**: 仅在 SDE 采样阶段用 (guidance_scale=5.0)，DPO update 不用

---

## 8. 训练循环

### Part1: Per-Prompt DPO

```python
for epoch in range(1, 1001):
    # Step A: 从 20k prompt 中随机选 3 条
    prompts = random.sample(all_prompts, 3)
    
    for prompt in prompts:
        # Step B: SDE 采样 6 条链 (no_grad)
        chains = [sample_sde(prompt) for _ in range(6)]
        
        # Step C: r_in z-score 排序
        r_in = zscore([c.log_p_base for c in chains])
        
        # Step D: 构造 3 对
        pairs = make_pairs(chains, r_in)  # (1st,6th), (2nd,5th), (3rd,4th)
        
        # Step E: DPO update (1 backward + 1 optimizer step)
        dpo_update(trainable, ref, chains, pairs, optimizer)
    
    # Step F: Logging + Checkpoint (每 20 epoch)
    wandb.log(metrics)
    if epoch % 20 == 0: save_lora_checkpoint()
```

每个 epoch = 3 prompt × 1 backward = 3 次 optimizer.step()。

### Part2: Batched Extreme-Pair DPO (更稳定)

```python
for epoch in range(1, 1001):
    # Step A: 从 20k prompt 中随机选 6 条
    prompts = random.sample(all_prompts, 6)
    
    # Step B: 采样所有 36 条链 (no_grad)
    all_chains = []
    for prompt in prompts:
        chains = [sample_sde(prompt) for _ in range(6)]
        r_in = zscore([c.log_p_base for c in chains])
        extreme_pair = (best_idx, worst_idx)  # 只用 (1st, 6th)
        all_chains.append(chains)
    
    # Step C: Batch 1 — prompts 0,1,2 的 extreme pairs → 1 backward
    batch1_pairs = [all_chains[0].extreme_pair,
                    all_chains[1].extreme_pair,
                    all_chains[2].extreme_pair]
    dpo_update(trainable, ref, all_chains, batch1_pairs, optimizer)
    
    # Step D: Batch 2 — prompts 3,4,5 的 extreme pairs → 1 backward
    batch2_pairs = [all_chains[3].extreme_pair,
                    all_chains[4].extreme_pair,
                    all_chains[5].extreme_pair]
    dpo_update(trainable, ref, all_chains, batch2_pairs, optimizer)
    
    # Step E: Logging + Checkpoint (每 20 epoch)
    wandb.log(metrics)
    if epoch % 20 == 0: save_lora_checkpoint()
```

每个 epoch = 2 batches × 1 backward = 2 次 optimizer.step()。

### Part1 vs Part2 对比

| 参数 | Part1 | Part2 |
|------|-------|-------|
| prompts/epoch | 3 | 6 |
| pairs/prompt | 3 (1st/6th, 2nd/5th, 3rd/4th) | 1 (1st/6th only) |
| optimizer steps/epoch | 3 | 2 |
| 训练信号 | 全部 3 对 (含弱信号) | 仅极值对 (强信号) |
| 设计目标 | 标准 DPO baseline | 降低噪声，提升稳定性 |

---

## 9. Checkpoint 格式

**保存** (`get_peft_model_state_dict`):
```python
torch.save({
    "epoch": epoch,
    "transformer_state_dict": get_peft_model_state_dict(transformer),  # ~18MB LoRA only
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": ...,
    "r_in_mean": ...,
}, "epoch_N.pt")
```

**恢复**:
```bash
python experiments/exp1/exp1_part1_dpo.py --config config/exp1_part1_dpo_config.py --resume 200
```

---

## 10. 已知问题和改进方向

| 问题 | 状态 | 改进方向 |
|------|------|---------|
| 单 t 采样方差大 (K_t=1) | 当前 | 增大 K_t 取平均，或用 importance sampling |
| β 可能不最优 | 待观察 | 根据 loss 曲线调 β |
| 训练可能崩塌 | 上一次训练 log_p 从 -50k 掉到 -150k | 降 LR 到 1e-5，观察 WandB |
| r_in 只是 proxy reward | 设计如此 | Exp1 纯 r_in，后续实验加 r_SSR |
