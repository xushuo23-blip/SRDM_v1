"""
PPO Trainer — 标准 PPO min-clip + 梯度范数裁剪 + 警报.

仅有两个安全机制:
  1. PPO ratio clip (ε=0.2):     ratio ∉ [0.8, 1.2] → 该样本梯度为零
  2. 梯度范数裁剪 (max_norm=5.0): 裁剪对象是 ∇θ L (除学习率外的整体)

警报装置监控两者是否频繁触发:
  - 比率裁剪警报: 最近 W 次 update 中, ratio 裁剪率过高的次数
  - 梯度裁剪警报: 最近 W 次 update 中, grad_norm 达到阈值的次数
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from diffusers import StableDiffusion3Pipeline

from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler


# ============================================================
# Training Alerter (比率裁剪 + 梯度裁剪 双重监控)
# ============================================================

class TrainingAlerter:
    """滑动窗口警报: 同时监控 ratio 裁剪率和梯度裁剪频率.

    Args:
        window:         滑动窗口大小 (update 次数), 默认 10
        threshold:      窗口内触发警报的最小次数, 默认 3
        ratio_bad_pct:  单次 update 中 ratio 被裁剪的样本-step 比例超过此值 → "坏 update"
    """

    def __init__(
        self,
        window: int = 10,
        threshold: int = 3,
        ratio_bad_pct: float = 0.5,
    ):
        self.window = window
        self.threshold = threshold
        self.ratio_bad_pct = ratio_bad_pct

        # 分别追踪两种事件
        self.ratio_history: List[bool] = []   # ratio 裁剪率过高
        self.grad_history: List[bool] = []    # grad_norm 被裁剪

        self.ratio_alert = False
        self.grad_alert = False
        self.total_updates = 0

    def check(
        self,
        ratio_clip_rate: float,     # 本次 update 中被 clip 的 sample-step 占比
        grad_clipped: bool,         # 本次 update 梯度是否触发裁剪
    ) -> dict:
        """记录一次 PPO update 并返回警报状态."""
        self.total_updates += 1

        # 记录
        ratio_bad = ratio_clip_rate > self.ratio_bad_pct
        grad_bad = grad_clipped

        self.ratio_history.append(ratio_bad)
        self.grad_history.append(grad_bad)
        if len(self.ratio_history) > self.window:
            self.ratio_history = self.ratio_history[-self.window:]
            self.grad_history = self.grad_history[-self.window:]

        ratio_count = sum(self.ratio_history)
        grad_count = sum(self.grad_history)

        # 警报边沿检测
        ratio_prev = self.ratio_alert
        grad_prev = self.grad_alert
        self.ratio_alert = ratio_count >= self.threshold
        self.grad_alert = grad_count >= self.threshold

        return {
            "alert/ratio_bad": ratio_bad,
            "alert/ratio_recent": ratio_count,
            "alert/ratio_fired": self.ratio_alert,
            "alert/ratio_raised": self.ratio_alert and not ratio_prev,
            "alert/grad_bad": grad_bad,
            "alert/grad_recent": grad_count,
            "alert/grad_fired": self.grad_alert,
            "alert/grad_raised": self.grad_alert and not grad_prev,
            "alert/any_fired": self.ratio_alert or self.grad_alert,
            "alert/total_updates": self.total_updates,
        }

    def reset(self):
        self.ratio_history.clear()
        self.grad_history.clear()
        self.ratio_alert = False
        self.grad_alert = False
        self.total_updates = 0


# ============================================================
# log_prob 计算 (per sample, 含 CFG)
# ============================================================

def compute_log_prob_at_step(
    pipeline: StableDiffusion3Pipeline,
    x_t: torch.Tensor,
    x_tm1: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    guidance_scale: float = 5.0,
    transformer: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """Compute log p_θ(x_{t-1} | x_t) for a single sample.

    Gradients flow through transformer parameters (no @torch.no_grad).
    """
    if transformer is None:
        transformer = pipeline.transformer

    t = t.to(device=x_t.device)

    scheduler: StochasticFlowMatchScheduler = pipeline.scheduler
    do_cfg = guidance_scale > 1.0

    if do_cfg:
        latent_input = torch.cat([x_t] * 2, dim=0)
        embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        latent_input = x_t
        embeds_input = prompt_embeds
        pooled_input = pooled_prompt_embeds

    t_batch = t.expand(latent_input.shape[0])
    v_theta = transformer(
        hidden_states=latent_input,
        encoder_hidden_states=embeds_input,
        pooled_projections=pooled_input,
        timestep=t_batch,
        return_dict=False,
    )[0]

    if do_cfg:
        v_uncond, v_text = v_theta.chunk(2)
        v_theta = v_uncond + guidance_scale * (v_text - v_uncond)

    _, log_prob = scheduler.step(
        v_theta, t, x_t, prev_sample=x_tm1, return_dict=False
    )
    return log_prob


# ============================================================
# PPO Update (单次 mini-batch: T 步梯度累积 → clip → step)
# ============================================================

def ppo_update_mini_batch(
    pipeline: StableDiffusion3Pipeline,
    chain_data: List[dict],
    batch_indices: List[int],
    timesteps: torch.Tensor,
    advantages: torch.Tensor,               # [num_chains_total], 已 z-score 归一化
    guidance_scale: float,
    optimizer: torch.optim.Optimizer,
    ppo_clip_range: float = 0.2,
    max_grad_norm: float = 5.0,
    num_inference_steps: int = 30,
    alerter: Optional[TrainingAlerter] = None,
    min_valid_ratio: float = 0.5,
) -> dict:
    """标准 PPO 更新: min(A·ratio, A·clip(ratio)) + 梯度范数裁剪 + 坏样本过滤.

    坏样本过滤:
      每条链 chain_data[i] 有一个 json_ok 属性 (True/False).
      VLM bbox 校验失败等原因会将其标记为 False.
      trainer 内部自动过滤: n_bad >= half of batch → 跳过整个 batch.

    裁剪对象:
      1. ratio clip (PPO 内置): ratio ∉ [1-ε, 1+ε] → 该样本梯度为零
      2. grad_norm clip:       ∇θ L 整体范数 > max_grad_norm → 等比缩回

    流程:
      filter bad samples by json_ok
      for each step t:
          计算 per-sample ratio
          PPO min-clip loss → backward (梯度累积)
      clip_grad_norm_(max_grad_norm)  → optimizer.step()

    Returns:
        metrics: dict 含 loss, ratio, grad_norm, batch_skipped, 警报状态
    """
    # ---- 坏样本过滤 ----
    good_idx = [i for i in batch_indices if chain_data[i].get("json_ok", True)]
    n_bad = len(batch_indices) - len(good_idx)

    if len(good_idx) / max(len(batch_indices), 1) < min_valid_ratio:
        return {
            "loss": 0.0, "ratio_mean": 0.0, "ratio_clip_rate": 0.0,
            "n_clipped_total": 0, "grad_norm": 0.0, "grad_clipped": False,
            "max_grad_norm": max_grad_norm,
            "batch_skipped": True, "n_bad_in_batch": n_bad,
        }

    B = len(good_idx)
    T = num_inference_steps
    lo, hi = 1.0 - ppo_clip_range, 1.0 + ppo_clip_range

    step_losses = []
    total_clipped = 0                     # 被 ratio clip 的 sample-step 总数
    total_ratio_mean = 0.0

    for step_t in range(T):
        batch_lp_new = []
        batch_lp_old = []

        for idx in good_idx:
            d = chain_data[idx]
            t = timesteps[step_t]
            x_t = d["all_latents"][step_t]
            x_tm1 = d["all_latents"][step_t + 1]

            lp_new = compute_log_prob_at_step(
                pipeline, x_t, x_tm1, t,
                d["prompt_embeds"], d["pooled_embeds"],
                d["neg_embeds"], d["neg_pooled"],
                guidance_scale,
            )
            batch_lp_new.append(lp_new.squeeze(0))
            batch_lp_old.append(d["log_probs_old"][step_t + 1].squeeze(0))

        lp_new = torch.stack(batch_lp_new)     # [B]
        lp_old = torch.stack(batch_lp_old)     # [B]

        ratio = torch.exp(lp_new - lp_old)     # [B]
        clipped_ratio = ratio.clamp(lo, hi)    # [B]

        # 统计被 clip 的样本数
        n_clipped = (ratio != clipped_ratio).sum().item()
        total_clipped += n_clipped

        adv = advantages[good_idx]             # [B]

        # PPO min-clip objective
        ppo_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
        (ppo_loss / T).backward()

        step_losses.append(ppo_loss.item())
        total_ratio_mean += ratio.mean().item()

    # ---- 梯度范数裁剪 (裁剪 ∇θ L 整体, lr 之前) ----
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in pipeline.transformer.parameters() if p.requires_grad],
        max_grad_norm,
    )
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
    grad_clipped = grad_norm_val >= max_grad_norm * 0.99

    optimizer.step()
    optimizer.zero_grad()

    # ---- 警报 ----
    total_sample_steps = B * T
    ratio_clip_rate = total_clipped / total_sample_steps

    alert_info = {}
    if alerter is not None:
        alert_info = alerter.check(ratio_clip_rate, grad_clipped)

    # ---- 汇总 ----
    metrics = {
        "loss": sum(step_losses) / T,
        "ratio_mean": total_ratio_mean / T,
        "ratio_clip_rate": ratio_clip_rate,           # 被 clip 的 sample-step 比例
        "n_clipped_total": total_clipped,
        "grad_norm": grad_norm_val,
        "grad_clipped": grad_clipped,
        "max_grad_norm": max_grad_norm,
        "batch_skipped": False,
        "n_bad_in_batch": n_bad,
    }
    metrics.update(alert_info)

    return metrics
