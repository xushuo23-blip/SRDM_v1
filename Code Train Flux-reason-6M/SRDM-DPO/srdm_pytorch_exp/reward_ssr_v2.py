"""
Structural Similarity Reward V2 — r_SSR with deviation ratio + mode-based φ*.

Key changes from V1:
    1. Existence penalty: r_SSR = -λ_exist if any noun count == 0
    2. φ* (count/relation): mode-based — per component, the most frequent value wins.
       Tie → average tied values. All unique → average all (median-like).
    3. φ* (coverage): weighted average (r_in softmax) — unchanged.
    4. Count/Coverage: deviation ratio |φ - φ*| / max(φ, φ*)
    5. Relation: plain L1 distance |R - R*| (values are {-1,0,1})
    6. Per-component z-score before weighted combination (same as V1)

Usage:
    from srdm_pytorch_exp.reward_ssr_v2 import compute_r_ssr_v2_batch
    r_ssr, debug_info = compute_r_ssr_v2_batch(phi_dicts, r_in_raw, ...)
"""

from typing import Dict, List

import torch

from srdm_pytorch_exp.reward_rin import zscore_normalize


# ============================================================
# Softmax temperature (for coverage weighted-average φ*)
# ============================================================

def _softmax_temperature(values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax with temperature scaling. Lower tau = sharper."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    v = values / temperature
    return torch.softmax(v, dim=0)


# ============================================================
# φ* prototype: mode-based (count, relation) + weighted avg (coverage)
# ============================================================

def _mode_prototype(phi_list: List[torch.Tensor]) -> torch.Tensor:
    """Mode-based prototype φ* — per dimension, the most frequent value wins.

    For each component (column) across M chains:
        - Find the most frequent value(s)
        - Tie → average the tied values
        - All unique → average all (acts as natural center)

    Examples for one column:
        (1,1,2,3) → mode=1 → φ*=1.0
        (1,1,2,2) → tie {1,2} → φ*=1.5 (symmetric, no optimization pressure)
        (1,2,3,4) → all unique → φ*=2.5 (extremes 1,4 penalized more)

    Args:
        phi_list: list of M tensors, each [D] for one component.

    Returns:
        phi_star: [D] mode-based prototype.
    """
    M = len(phi_list)
    if M == 0:
        return torch.tensor([])

    phi_matrix = torch.stack(phi_list, dim=0).float()  # [M, D]
    D = phi_matrix.shape[1]
    dev = phi_matrix.device
    phi_star = torch.zeros(D, device=dev)

    for d in range(D):
        col = phi_matrix[:, d]  # [M]
        unique_vals, counts = torch.unique(col, return_counts=True)
        max_count = counts.max()
        mode_vals = unique_vals[counts == max_count]
        phi_star[d] = mode_vals.mean()

    return phi_star


def _weighted_average_prototype(
    phi_list: List[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted-average prototype φ* = Σ w_i · φ_i (for coverage)."""
    if len(phi_list) == 0:
        return torch.tensor([])
    phi_matrix = torch.stack(phi_list, dim=0).float()  # [M, D]
    return (weights.unsqueeze(1) * phi_matrix).sum(dim=0)  # [D]


# ============================================================
# Distance functions (take pre-computed φ*)
# ============================================================

def _plain_l1_distance(
    phi_list: List[torch.Tensor],
    phi_star: torch.Tensor,
) -> tuple:
    """Plain L1 distance to prototype: d_i = mean_k |φ_ik - φ*_k|.

    Used for relation (sign values in {-1,0,1}, raw diff in [0,2]).
    """
    M = len(phi_list)
    if M == 0 or phi_star.numel() == 0:
        D = phi_star.shape[0] if phi_star.numel() > 0 else 0
        return torch.tensor([]), torch.tensor([]), torch.zeros(M, max(D, 1))

    phi_matrix = torch.stack(phi_list, dim=0).float()  # [M, D]
    raw_diffs = phi_matrix - phi_star.unsqueeze(0)  # [M, D]
    abs_diffs = raw_diffs.abs()
    losses = abs_diffs.mean(dim=1)  # [M]
    return losses, raw_diffs


def _component_deviation_ratio(
    phi_list: List[torch.Tensor],
    phi_star: torch.Tensor,
) -> tuple:
    """Deviation ratio to prototype: |φ_ik - φ*_k| / max(φ_ik, φ*_k).

    Symmetric — over/under-generation penalized equally relative to larger value.
    Edge case: max(0, 0) = 0 → ratio = 0.
    """
    M = len(phi_list)
    if M == 0 or phi_star.numel() == 0:
        D = phi_star.shape[0] if phi_star.numel() > 0 else 0
        return torch.tensor([]), torch.tensor([]), torch.zeros(M, max(D, 1))

    phi_matrix = torch.stack(phi_list, dim=0).float()  # [M, D]
    raw_diffs = phi_matrix - phi_star.unsqueeze(0)  # [M, D]
    abs_diffs = raw_diffs.abs()

    denom = torch.max(phi_matrix, phi_star.unsqueeze(0))  # [M, D]
    ratio = torch.where(denom > 1e-8, abs_diffs / denom, torch.zeros_like(abs_diffs))

    losses = ratio.mean(dim=1)  # [M]
    return losses, raw_diffs


# ============================================================
# Main: compute_r_ssr_v2_batch
# ============================================================

def compute_r_ssr_v2_batch(
    phi_dicts: List[dict],
    r_in_raw: torch.Tensor,
    lambda_exist: float = 2.0,
    lambda_count: float = 1.0,
    lambda_coverage: float = 1.0,
    lambda_relation: float = 1.0,
    temperature: float = 1.0,
    uniform_weights: bool = False,
) -> tuple:
    """Compute r_SSR v2 for a batch of M chains.

    Algorithm:
        0. Existence check: if any noun count == 0 → r_SSR_i = -lambda_exist
        1. φ*_count    = mode-based (most frequent per object)
           φ*_relation = mode-based (most frequent per sign component)
           φ*_coverage = weighted average (r_in softmax, unchanged)
        2. Count:   d_count_i    = mean_k |φ_ik - φ*_k| / max(φ_ik, φ*_k)
        3. Coverage: d_coverage_i =       |C_i - C*|      / max(C_i, C*)
        4. Relation: d_rel_i      = mean_k |R_ik - R*_k|
        5. z-score each d vector → d_count_norm, d_cov_norm, d_rel_norm
        6. d_combined = λ_count*d_count_norm + λ_cov*d_cov_norm + λ_rel*d_rel_norm
        7. r_SSR_i = -d_combined, with existence override → -lambda_exist

    Args:
        phi_dicts: list of M dicts, each from structure_features.phi_to_dict().
        r_in_raw: [M] raw total_log_p_base values.
        lambda_exist: penalty for chains with zero-count nouns.
        lambda_count / lambda_coverage / lambda_relation: component weights.
        temperature: softmax temperature (for coverage φ* only).
        uniform_weights: if True, φ*_coverage = (1/M) Σ φ_i.

    Returns:
        r_ssr: [M] structural reward per chain.
        debug_info: dict with phi_star, per-component losses, and penalty mask.
    """
    M = len(phi_dicts)
    dev = r_in_raw.device

    empty_debug = {
        "phi_star_count": torch.tensor([]),
        "phi_star_coverage": torch.tensor([]),
        "phi_star_relation": torch.tensor([]),
        "d_count": torch.zeros(M, device=dev),
        "d_coverage": torch.zeros(M, device=dev),
        "d_relation": torch.zeros(M, device=dev),
        "diff_count": torch.zeros(M, 1, device=dev),
        "diff_coverage": torch.zeros(M, 1, device=dev),
        "diff_relation": torch.zeros(M, 1, device=dev),
        "d_count_norm": torch.zeros(M, device=dev),
        "d_coverage_norm": torch.zeros(M, device=dev),
        "d_relation_norm": torch.zeros(M, device=dev),
        "d_combined": torch.zeros(M, device=dev),
        "r_ssr": torch.zeros(M, device=dev),
        "exist_penalty_mask": torch.zeros(M, dtype=torch.bool, device=dev),
        "weights": torch.ones(M, device=dev) / max(M, 1),
        "has_count": False,
        "has_coverage": False,
        "has_relation": False,
    }

    if M < 2:
        return torch.zeros(M, device=dev), empty_debug

    # 1. Weights (for coverage φ* only)
    if uniform_weights:
        weights = torch.ones(M, device=dev) / M
    else:
        weights = _softmax_temperature(r_in_raw.float(), temperature)

    # 2. Move phi tensors to device
    count_phis = [d["count"].to(dev) for d in phi_dicts]
    coverage_phis = [d["coverage"].to(dev) for d in phi_dicts]
    relation_phis = [d["relation"].to(dev) for d in phi_dicts]

    has_count = all(p.numel() > 0 for p in count_phis)
    has_coverage = all(p.numel() > 0 for p in coverage_phis)
    has_relation = all(p.numel() > 0 for p in relation_phis)

    # 3. Step 0: existence penalty mask (triggered if any noun has count == 0)
    exist_mask = torch.zeros(M, dtype=torch.bool, device=dev)
    if has_count:
        for i, cp in enumerate(count_phis):
            if (cp == 0).any():
                exist_mask[i] = True

    # 4. Compute φ* (mode-based for count/relation, weighted avg for coverage)
    phi_star_count = torch.tensor([], device=dev)
    phi_star_coverage = torch.tensor([], device=dev)
    phi_star_relation = torch.tensor([], device=dev)
    d_count = torch.zeros(M, device=dev)
    d_coverage = torch.zeros(M, device=dev)
    d_relation = torch.zeros(M, device=dev)
    diff_count = torch.zeros(M, 1, device=dev)
    diff_coverage = torch.zeros(M, 1, device=dev)
    diff_relation = torch.zeros(M, 1, device=dev)

    if has_count:
        phi_star_count = _mode_prototype(count_phis)
        d_count, diff_count = _component_deviation_ratio(count_phis, phi_star_count)

    if has_coverage:
        phi_star_coverage = _weighted_average_prototype(coverage_phis, weights)
        d_coverage, diff_coverage = _component_deviation_ratio(coverage_phis, phi_star_coverage)

    if has_relation:
        phi_star_relation = _mode_prototype(relation_phis)
        d_relation, diff_relation = _plain_l1_distance(relation_phis, phi_star_relation)

    # 5. Per-component z-score normalization
    d_count_norm = zscore_normalize(d_count)
    d_coverage_norm = zscore_normalize(d_coverage)
    d_relation_norm = zscore_normalize(d_relation)

    # 6. Weighted combination
    d_combined = (
        lambda_count * d_count_norm +
        lambda_coverage * d_coverage_norm +
        lambda_relation * d_relation_norm
    )

    r_ssr = -d_combined

    # 7. Apply existence penalty (override computed r_ssr)
    r_ssr[exist_mask] = -lambda_exist

    debug_info = {
        "phi_star_count": phi_star_count,
        "phi_star_coverage": phi_star_coverage,
        "phi_star_relation": phi_star_relation,
        "d_count": d_count,
        "d_coverage": d_coverage,
        "d_relation": d_relation,
        "diff_count": diff_count,
        "diff_coverage": diff_coverage,
        "diff_relation": diff_relation,
        "d_count_norm": d_count_norm,
        "d_coverage_norm": d_coverage_norm,
        "d_relation_norm": d_relation_norm,
        "d_combined": d_combined,
        "r_ssr": r_ssr,
        "exist_penalty_mask": exist_mask,
        "weights": weights,
        "has_count": has_count,
        "has_coverage": has_coverage,
        "has_relation": has_relation,
    }
    return r_ssr, debug_info
