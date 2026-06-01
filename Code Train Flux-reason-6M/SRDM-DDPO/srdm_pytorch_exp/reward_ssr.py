"""
Structural Similarity Reward — r_SSR 计算.

核心逻辑:
    1. 对 M 条链, r_in 软加权得到内生结构原型 ϕ*
    2. 每条链的 ϕ 各分量分别计算到 ϕ* 的 L1 距离
    3. 每分量距离在 M 条链内 z-score 归一化
    4. 加权求和: d = λ_count * d_count_norm + λ_coverage * d_coverage_norm + λ_relation * d_relation_norm
    5. r_SSR = -d (d 已是归一化结果的加权和，无需再归一化)

用法:
    r_ssr, debug_info = compute_r_ssr_batch(phi_dicts, r_in_raw, lambdas, temperature)
    debug_info 包含 ϕ* 原型、原始 L1 距离、归一化距离、每链原始差异等
"""

from typing import Dict, List, Optional

import torch

from srdm_pytorch_exp.reward_rin import zscore_normalize


def _softmax_temperature(values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax with temperature scaling. Lower τ → sharper (closer to hard argmax)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    v = values / temperature
    return torch.softmax(v, dim=0)


def compute_component_distance_l1(
    phi_list: List[torch.Tensor],
    weights: torch.Tensor,
) -> tuple:
    """Compute L1 distance of each chain's ϕ to the weighted prototype ϕ*.

    L1 distance (Manhattan): d_j = mean(|ϕ_j - ϕ*|) across dimensions.
    比 L2 更可解释——计数差 1 个单位就是 1，不会被平方放大。

    Args:
        phi_list: list of M tensors, each [D_c] for one component.
        weights: [M] softmax weights (from r_in).

    Returns:
        phi_star: [D_c] weighted prototype.
        distances: [M] mean L1 distance per chain.
        raw_diffs: [M, D_c] per-dimension signed difference (ϕ_j - ϕ*).
    """
    M = len(phi_list)
    if M == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    phi_matrix = torch.stack(phi_list, dim=0).float()  # [M, D_c]
    phi_star = (weights.unsqueeze(1) * phi_matrix).sum(dim=0)  # [D_c]

    raw_diffs = phi_matrix - phi_star.unsqueeze(0)  # [M, D_c]
    distances = raw_diffs.abs().mean(dim=1)  # [M]  mean L1 across dims

    return phi_star, distances, raw_diffs


def compute_r_ssr_batch(
    phi_dicts: List[dict],
    r_in_raw: torch.Tensor,
    lambda_count: float = 1.0,
    lambda_coverage: float = 1.0,
    lambda_relation: float = 1.0,
    temperature: float = 1.0,
    uniform_weights: bool = False,
) -> tuple:
    """Compute r_SSR for a batch of M chains.

    Args:
        phi_dicts: list of M dicts, each from structure_features.phi_to_dict().
        r_in_raw: [M] raw total_log_p_base values.
        lambda_count / lambda_coverage / lambda_relation: 分量权重.
        temperature: softmax temperature (ignored when uniform_weights=True).
        uniform_weights: if True, φ* = (1/M) Σ φ_i (均匀平均).
                         if False, φ* = Σ softmax(r_in/τ)_i · φ_i.

    Returns:
        r_ssr: [M] normalized structural reward.
        debug_info: dict with:
            - phi_star_count / phi_star_coverage / phi_star_relation: 原型向量
            - d_count / d_coverage / d_relation: 原始 L1 距离 [M]
            - diff_count / diff_coverage / diff_relation: 每链每维原始差异 [M, D]
            - d_count_norm / d_coverage_norm / d_relation_norm: 归一化距离 [M]
            - d_combined: 加权合并距离 [M]
            - r_ssr_raw: 负合并距离 [M]
            - weights: 权重 [M]
    """
    M = len(phi_dicts)
    empty_debug = {
        "phi_star_count": torch.tensor([]), "phi_star_coverage": torch.tensor([]),
        "phi_star_relation": torch.tensor([]),
        "d_count": torch.zeros(M), "d_coverage": torch.zeros(M), "d_relation": torch.zeros(M),
        "diff_count": torch.zeros(M, 1), "diff_coverage": torch.zeros(M, 1),
        "diff_relation": torch.zeros(M, 1),
        "d_count_norm": torch.zeros(M), "d_coverage_norm": torch.zeros(M),
        "d_relation_norm": torch.zeros(M),
        "d_combined": torch.zeros(M), "r_ssr_raw": torch.zeros(M), "r_ssr": torch.zeros(M),
        "weights": torch.ones(M) / max(M, 1),
        "has_count": False, "has_coverage": False, "has_relation": False,
    }
    if M < 2:
        return torch.zeros(M), empty_debug

    # 1. Weights: uniform or softmax(r_in)
    if uniform_weights:
        weights = torch.ones(M, device=r_in_raw.device) / M
    else:
        weights = _softmax_temperature(r_in_raw.float(), temperature)

    # 2. Per-component L1 distances + phi* + raw diffs
    # Move all phi tensors to the same device as weights
    dev = weights.device
    count_phis = [d["count"].to(dev) for d in phi_dicts]
    coverage_phis = [d["coverage"].to(dev) for d in phi_dicts]
    relation_phis = [d["relation"].to(dev) for d in phi_dicts]

    has_count = all(p.numel() > 0 for p in count_phis)
    has_coverage = all(p.numel() > 0 for p in coverage_phis)
    has_relation = all(p.numel() > 0 for p in relation_phis)

    phi_star_count = torch.tensor([])
    phi_star_coverage = torch.tensor([])
    phi_star_relation = torch.tensor([])
    d_count = torch.zeros(M)
    d_coverage = torch.zeros(M)
    d_relation = torch.zeros(M)
    diff_count = torch.zeros(M, 1)
    diff_coverage = torch.zeros(M, 1)
    diff_relation = torch.zeros(M, 1)

    if has_count:
        phi_star_count, d_count, diff_count = compute_component_distance_l1(count_phis, weights)
    if has_coverage:
        phi_star_coverage, d_coverage, diff_coverage = compute_component_distance_l1(coverage_phis, weights)
    if has_relation:
        phi_star_relation, d_relation, diff_relation = compute_component_distance_l1(relation_phis, weights)

    # 3. Per-component z-score normalization
    d_count_norm = zscore_normalize(d_count)
    d_coverage_norm = zscore_normalize(d_coverage)
    d_relation_norm = zscore_normalize(d_relation)

    # 4. Weighted combination
    d_combined = (
        lambda_count * d_count_norm +
        lambda_coverage * d_coverage_norm +
        lambda_relation * d_relation_norm
    )

    # 5. r_SSR = -d_combined (closer = higher reward)
    #    d_combined 已是各分量归一化距离的加权和，无需再归一化
    r_ssr = -d_combined

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
        "r_ssr_raw": r_ssr,
        "r_ssr": r_ssr,
        "weights": weights,
        "has_count": has_count,
        "has_coverage": has_coverage,
        "has_relation": has_relation,
    }
    return r_ssr, debug_info


# ============================================================
# Optional visualization: PCA distance scatter plot
# ============================================================

def make_distance_plot(
    debug_info: dict,
    chain_indices: list = None,
    variant_label: str = "",
    prompt_short: str = "",
) -> "Image.Image":
    """2D PCA scatter plot of chain distances from phi*.

    Projects the 3D normalized distance vector [d_count_norm, d_cov_norm, d_rel_norm]
    to 2D via PCA. phi* is at origin. Each point = one chain, annotated with
    chain index and d_combined value. Colored by d_combined (RdYlGn_r: green=close).

    Lazy-imports matplotlib — no dependency unless this function is called.

    Args:
        debug_info: dict from compute_r_ssr_batch().
        chain_indices: optional list of chain index labels.
        variant_label: chart title prefix.
        prompt_short: chart title suffix (truncated to 60 chars).

    Returns:
        PIL Image of the scatter plot.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d_count_n = debug_info.get("d_count_norm", None)
    d_cov_n = debug_info.get("d_coverage_norm", None)
    d_rel_n = debug_info.get("d_relation_norm", None)
    d_combined = debug_info.get("d_combined", None)

    if d_count_n is None or d_cov_n is None or d_rel_n is None or d_combined is None:
        raise ValueError("debug_info missing required distance keys")

    if isinstance(d_count_n, torch.Tensor):
        d_count_n = d_count_n.numpy()
    if isinstance(d_cov_n, torch.Tensor):
        d_cov_n = d_cov_n.numpy()
    if isinstance(d_rel_n, torch.Tensor):
        d_rel_n = d_rel_n.numpy()
    if isinstance(d_combined, torch.Tensor):
        d_combined = d_combined.numpy()

    M = len(d_count_n)
    if chain_indices is None:
        chain_indices = list(range(M))

    if M < 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "Need >= 2 chains for PCA", ha="center", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return _pil_from_buf(buf)

    X = np.stack([d_count_n, d_cov_n, d_rel_n], axis=1)  # [M, 3]
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_2d = X_centered @ Vt[:2].T  # [M, 2]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.axhline(0, color="#ddd", linewidth=0.8)
    ax.axvline(0, color="#ddd", linewidth=0.8)

    d_max = max(d_combined.max(), 1e-6)
    colors = plt.cm.RdYlGn_r(d_combined / d_max)

    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=120, edgecolors="#333",
               linewidths=0.8, zorder=5)

    for i in range(M):
        offset = 0.05 * (X_2d[:, :2].std(axis=0).max() or 0.1)
        ax.annotate(
            f"c{chain_indices[i]}\nd={d_combined[i]:.2f}",
            (X_2d[i, 0], X_2d[i, 1]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color="#333",
        )

    # phi* at origin
    ax.scatter([0], [0], marker="*", s=250, c="blue", edgecolors="navy",
               linewidths=1.5, zorder=10)
    ax.annotate("phi*", (0, 0), textcoords="offset points", xytext=(8, -12),
                fontsize=10, fontweight="bold", color="navy")

    ax.set_title(f"{variant_label}  |  {prompt_short[:60]}", fontsize=10)
    ax.set_xlabel("PC1 (normalized distance space)")
    ax.set_ylabel("PC2 (normalized distance space)")
    ax.set_aspect("equal", adjustable="datalim")

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, d_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("d_combined (lower = closer to phi*)", fontsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return _pil_from_buf(buf)


def _pil_from_buf(buf) -> "Image.Image":
    from PIL import Image
    buf.seek(0)
    return Image.open(buf)
