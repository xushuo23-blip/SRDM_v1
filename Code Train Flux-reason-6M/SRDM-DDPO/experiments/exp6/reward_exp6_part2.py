"""Exp6 Part2 奖励: r_total = 0.5 * r_in + 1.0 * r_SSR (V2).

与 Part1 的区别: r_SSR 升级为 V2 版本:
    - 存在性惩罚: 任何 noun count == 0 → r_SSR = -lambda_exist
    - φ* (count/relation): mode-based — per component, 最多出现的值
    - φ* (coverage): weighted average (r_in softmax)
    - Count/Coverage: deviation ratio |φ - φ*| / max(φ, φ*)
    - Relation: plain L1
    - z-score → weighted combination
"""

import torch

from srdm_pytorch_exp.reward_rin import zscore_normalize
from srdm_pytorch_exp.reward_ssr_v2 import compute_r_ssr_v2_batch
from srdm_pytorch_exp.structure_features import phi_dicts_simplified


def compute_reward_exp6_part2(valid_chains, cfg):
    """对单个 prompt 的 M 条链计算 r_in + r_SSR_v2 组合奖励.

    Args:
        valid_chains: list of dict, 每条链含 total_lp_base, structure.
        cfg: ConfigDict, 含 r_SSR v2 参数和权重.

    Returns:
        r_in: [M] z-score normalized.
        r_ssr: [M] structural similarity reward (V2).
        rewards: [M] combined = cfg.r_in_weight * r_in + cfg.r_ssr_weight * r_ssr.
        ssr_debug: dict from compute_r_ssr_v2_batch.
        phi_dicts: list of M dicts with "count"/"coverage"/"relation" tensors.
    """
    M = len(valid_chains)
    if M < 2:
        return torch.zeros(M), torch.zeros(M), torch.zeros(M), {}, []

    valid_lp = torch.tensor([d["total_lp_base"] for d in valid_chains])
    r_in = zscore_normalize(valid_lp)

    phi_dicts, _, _, _ = phi_dicts_simplified(
        [d["structure"] for d in valid_chains],
        valid_chains[0]["schema"],
    )

    r_ssr, ssr_debug = compute_r_ssr_v2_batch(
        phi_dicts, valid_lp,
        lambda_exist=cfg.lambda_exist,
        lambda_count=cfg.lambda_count,
        lambda_coverage=cfg.lambda_coverage,
        lambda_relation=cfg.lambda_relation,
        temperature=cfg.r_ssr_temperature,
        uniform_weights=cfg.phi_uniform_weights,
    )

    rewards = cfg.r_in_weight * r_in + cfg.r_ssr_weight * r_ssr
    return r_in, r_ssr, rewards, ssr_debug, phi_dicts
