"""r_gt — GT 对齐奖励，基于 count + direction 两阶段门控 (Exp5 专用).

单张图像 vs GT 标准答案:
    Phase 1 — Count 检查:
        对 GT objects 中的每个物体，VLM 检测数量必须与 GT 完全一致。
        任一不一致 → r_gt = -2（硬惩罚）。

    Phase 2 — Direction 检查 (仅 count 全通过时):
        GT direction = [dx, dy], dx,dy ∈ {-1, 0, 1}。
        - 完全一致 → r_gt = +2
        - 最多一个轴不一致，且不一致轴 GT=0 → r_gt = +2（容忍）
        - 其他 → r_gt = 0（中间值）

用法:
    from experiments.exp5.reward_gt import compute_r_gt_single

    score = compute_r_gt_single(structure, prompt_gt)
"""

from typing import Dict, List

from experiments.exp5.gt_utils import (
    compute_top2_direction_from_structure,
)


def _get_vlm_counts(structure: dict, labels: List[str]) -> Dict[str, int]:
    """Extract VLM-detected counts for given labels from structure JSON."""
    counts = {lbl: 0 for lbl in labels}
    for obj in structure.get("objects", []):
        lbl = obj.get("label", "")
        if lbl in counts:
            counts[lbl] = obj.get("count", 0)
    return counts


def _check_count(vlm_counts: Dict[str, int], gt_objects: Dict[str, int]) -> bool:
    """True if all VLM counts match GT exactly."""
    for label, gt_count in gt_objects.items():
        if vlm_counts.get(label, 0) != gt_count:
            return False
    return True


def _check_direction(
    vlm_direction: List[int],
    gt_direction: List[int],
) -> float:
    """Direction gate. Returns +2 (pass), or 0 (fail).

    Pass conditions:
        - Exact match: vlm == gt on both axes → +2
        - Tolerated: at most 1 axis differs, and differing axis GT=0 → +2
        - Otherwise → 0
    """
    if vlm_direction == gt_direction:
        return 2.0

    if not isinstance(gt_direction, (list, tuple)) or len(gt_direction) != 2:
        return 0.0
    if not isinstance(vlm_direction, (list, tuple)) or len(vlm_direction) != 2:
        return 0.0

    diff_axes = [ax for ax in (0, 1) if vlm_direction[ax] != gt_direction[ax]]
    n_diff = len(diff_axes)

    if n_diff >= 2:
        return 0.0

    # n_diff == 1: tolerated only if GT[diff_axis] == 0
    if gt_direction[diff_axes[0]] == 0:
        return 2.0

    return 0.0


def compute_r_gt_single(
    structure: dict,
    prompt_gt: dict,
    count_penalty: float = -2.0,
    direction_reward: float = 2.0,
    direction_threshold: float = 0.15,
    return_debug: bool = False,
):
    """Compute r_gt for a single image.

    Args:
        structure: VLM structure JSON.
        prompt_gt: GT dict with "objects", "top2_objects", "spatial_relation".
        count_penalty: penalty when count check fails (default -2).
        direction_reward: reward when direction passes (default +2).
        direction_threshold: centroid direction threshold.
        return_debug: if True, return (score, debug_dict).

    Returns:
        score (float), or (score, debug_dict) if return_debug=True.
    """
    gt_objects = prompt_gt.get("objects", {})
    gt_top2 = prompt_gt.get("top2_objects", [])
    gt_spatial = prompt_gt.get("spatial_relation", {})

    # ---- Phase 1: Count check ----
    gt_labels = list(gt_objects.keys())
    vlm_counts = _get_vlm_counts(structure, gt_labels)

    count_pass = _check_count(vlm_counts, gt_objects)

    if not count_pass:
        score = count_penalty
        if return_debug:
            return score, {
                "phase": "count_fail",
                "vlm_counts": vlm_counts,
                "gt_counts": gt_objects,
                "vlm_direction": None,
                "gt_direction": None,
                "score": score,
            }
        return score

    # ---- Phase 2: Direction check ----
    gt_direction = gt_spatial.get("direction", [0, 0])
    vlm_direction = compute_top2_direction_from_structure(
        structure, gt_top2, direction_threshold)

    dir_score = _check_direction(vlm_direction, gt_direction)
    score = direction_reward if dir_score > 0 else 0.0

    if return_debug:
        return score, {
            "phase": "direction_" + ("pass" if dir_score > 0 else "fail"),
            "vlm_counts": vlm_counts,
            "gt_counts": gt_objects,
            "vlm_direction": vlm_direction,
            "gt_direction": gt_direction,
            "score": score,
        }

    return score
