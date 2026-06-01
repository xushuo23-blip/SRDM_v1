"""Exp6 奖励: r_total = 0.5 * r_in + 1.0 * r_SSR.

调用公共模块:
    - srdm_pytorch_exp.reward_rin.zscore_normalize → r_in
    - srdm_pytorch_exp.reward_ssr.compute_r_ssr_batch → r_SSR
    - srdm_pytorch_exp.structure_features.phi_dicts_simplified → φ 向量
"""

import torch

from srdm_pytorch_exp.reward_rin import zscore_normalize
from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch
from srdm_pytorch_exp.structure_features import phi_dicts_simplified


def compute_reward_exp6(valid_chains, cfg):
    """对单个 prompt 的 M 条链计算组合奖励.

    Args:
        valid_chains: list of dict, 每条链含 total_lp_base, structure.
        cfg: ConfigDict, 含 r_SSR 参数 (lambda_*, phi_uniform_weights)
             和权重 (r_in_weight, r_ssr_weight).

    Returns:
        r_in: [M] z-score normalized.
        r_ssr: [M] structural similarity reward.
        rewards: [M] combined = cfg.r_in_weight * r_in + cfg.r_ssr_weight * r_ssr.
        ssr_debug: dict from compute_r_ssr_batch.
    """
    M = len(valid_chains)
    if M < 2:
        return torch.zeros(M), torch.zeros(M), torch.zeros(M), {}

    valid_lp = torch.tensor([d["total_lp_base"] for d in valid_chains])
    r_in = zscore_normalize(valid_lp)

    phi_dicts, _, _, _ = phi_dicts_simplified(
        [d["structure"] for d in valid_chains],
        valid_chains[0]["schema"],
    )

    r_ssr, ssr_debug = compute_r_ssr_batch(
        phi_dicts, valid_lp,
        lambda_count=cfg.lambda_count,
        lambda_coverage=cfg.lambda_coverage,
        lambda_relation=cfg.lambda_relation,
        uniform_weights=cfg.phi_uniform_weights,
    )

    rewards = cfg.r_in_weight * r_in + cfg.r_ssr_weight * r_ssr
    return r_in, r_ssr, rewards, ssr_debug
