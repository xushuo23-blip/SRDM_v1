"""
r_in 内生奖励 — 基于 base model log_prob 的 z-score 归一化。

这是所有后续奖励的基础模块。实验二已证明 z-score 归一化优于 tanh 锚定。

用法:
    from srdm_pytorch_exp.reward_rin import zscore_normalize, compute_reward_rin

    # 单组归一化
    r_in = zscore_normalize(total_log_p_base)

    # 多 prompt 分组归一化
    rewards = compute_reward_rin(total_log_p_base, group_size=6)
"""

import torch


def zscore_normalize(values: torch.Tensor) -> torch.Tensor:
    """Z-score normalize: (x - mean) / std.

    Returns zeros if std ≈ 0.
    """
    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        return torch.zeros_like(values)
    return (values - mean) / std


def compute_reward_rin(
    total_log_p_base: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Per-group z-score normalize of base model log_prob.

    Args:
        total_log_p_base: [M] tensor, total log_prob from frozen base model.
        group_size: chains per prompt (normalization is within each group).

    Returns:
        rewards: [M] tensor, z-score normalized within each group.
    """
    M = total_log_p_base.shape[0]
    n_groups = M // group_size
    rewards = torch.zeros(M, device=total_log_p_base.device, dtype=torch.float32)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        rewards[start:end] = zscore_normalize(total_log_p_base[start:end])

    return rewards
