"""
Structure Features — ϕ(G) 提取.

从 VLM structure JSON 提取三类结构化特征向量:
    ϕ_count    — 数量特征: 每个 canonical object 的 count
    ϕ_coverage — 覆盖特征: object 对之间的 IoU / 重叠度量
    ϕ_relation — 方位特征: object 对之间的方向关系编码

所有特征均为纯数学计算，VLM 不参与打分。
"""

from typing import Dict, List

import torch


# ============================================================
# Bbox geometry utilities (公共工具，vlm_client 也从此 import)
# ============================================================

def normalize_bbox(bbox):
    """Normalize bbox to [x1,y1,x2,y2] with x1<=x2, y1<=y2, clamped to [0,1].

    Returns None if bbox is not a list/tuple of exactly 4 numeric values.
    Callers must handle None — no silent default.
    """
    try:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted([max(0.0, min(x1, 1.0)), max(0.0, min(x2, 1.0))])
    y1, y2 = sorted([max(0.0, min(y1, 1.0)), max(0.0, min(y2, 1.0))])
    return [x1, y1, x2, y2]


def union_bbox(bboxes):
    """Compute union bounding box over a list of bboxes. Skips invalid bboxes."""
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    normed = [n for b in bboxes if (n := normalize_bbox(b)) is not None]
    if not normed:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = min(b[0] for b in normed)
    y1 = min(b[1] for b in normed)
    x2 = max(b[2] for b in normed)
    y2 = max(b[3] for b in normed)
    return [x1, y1, x2, y2]


def bbox_area(bbox) -> float:
    nb = normalize_bbox(bbox)
    if nb is None:
        return 0.0
    x1, y1, x2, y2 = nb
    return (x2 - x1) * (y2 - y1)


def bbox_intersection(bbox_a, bbox_b) -> float:
    nb_a = normalize_bbox(bbox_a)
    nb_b = normalize_bbox(bbox_b)
    if nb_a is None or nb_b is None:
        return 0.0
    x1_a, y1_a, x2_a, y2_a = nb_a
    x1_b, y1_b, x2_b, y2_b = nb_b
    x1 = max(x1_a, x1_b)
    y1 = max(y1_a, y1_b)
    x2 = min(x2_a, x2_b)
    y2 = min(y2_a, y2_b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    inter = bbox_intersection(bbox_a, bbox_b)
    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)
    union = area_a + area_b - inter
    if union < 1e-8:
        return 0.0
    return inter / union


def _get_canonical_labels(schema: dict) -> List[str]:
    """Extract ordered canonical object labels from schema."""
    objects = schema.get("canonical_objects", [])
    if not objects:
        return []
    return [obj["label"] for obj in objects]


def _get_centroid(inst: dict) -> List[float]:
    """Get [cx, cy] from instance. Uses centroid if present, else computes from bbox."""
    if "centroid" in inst:
        c = inst["centroid"]
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            try:
                return [float(c[0]), float(c[1])]
            except (TypeError, ValueError):
                pass
    bbox = inst.get("bbox", [0.0, 0.0, 0.0, 0.0])
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return [(float(bbox[0]) + float(bbox[2])) / 2.0,
                    (float(bbox[1]) + float(bbox[3])) / 2.0]
        except (TypeError, ValueError):
            pass
    return [0.0, 0.0]


def _mean_centroid(instances: List[dict]) -> List[float]:
    """Average centroid over all instances of an object type."""
    if not instances:
        return [0.0, 0.0]
    centroids = [_get_centroid(inst) for inst in instances]
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def phi_count(structure: dict, schema: dict) -> torch.Tensor:
    """ϕ_count: [count(label_0), count(label_1), ..., count(label_{K-1})] ∈ R^K.

    For each canonical object label, extract its count from the structure JSON.
    Missing objects get count 0.
    """
    labels = _get_canonical_labels(schema)
    if not labels:
        return torch.tensor([])

    obj_map = {}
    for obj in structure.get("objects", []):
        obj_map[obj.get("label", "")] = obj.get("count", 0)

    counts = [float(obj_map.get(label, 0)) for label in labels]
    return torch.tensor(counts)


def phi_coverage(structure: dict, schema: dict) -> torch.Tensor:
    """ϕ_coverage: per-instance pairwise intersection ratio.

    For each pair (i, j) with i < j:
        coverage = count(intersecting instance pairs) / (N_i × N_j)

    where N_i is the number of instances of class i, and two instances
    "intersect" if their bbox_intersection > 0. Length = K*(K-1)/2.
    """
    labels = _get_canonical_labels(schema)
    K = len(labels)
    if K < 2:
        return torch.tensor([])

    # Collect all bboxes per label (per-instance, not merged)
    label_bboxes: Dict[str, List[List[float]]] = {}
    for obj in structure.get("objects", []):
        lbl = obj.get("label", "")
        bboxes = [inst.get("bbox", [0, 0, 0, 0]) for inst in obj.get("instances", [])]
        if bboxes:
            label_bboxes[lbl] = bboxes

    ratios = []
    for i in range(K):
        for j in range(i + 1, K):
            bboxes_i = label_bboxes.get(labels[i], [])
            bboxes_j = label_bboxes.get(labels[j], [])
            if not bboxes_i or not bboxes_j:
                ratios.append(0.0)
            else:
                total_pairs = len(bboxes_i) * len(bboxes_j)
                intersecting = sum(
                    1 for ba in bboxes_i for bb in bboxes_j
                    if bbox_intersection(ba, bb) > 0
                )
                ratios.append(intersecting / total_pairs)
    return torch.tensor(ratios)


def phi_relation(structure: dict, schema: dict) -> torch.Tensor:
    """ϕ_relation: pairwise directional relation encoding.

    For each pair (i, j) with i < j, encode relative position:
        x_sign = sign(centroid_i_x - centroid_j_x)  ∈ {-1, 0, +1}
        y_sign = sign(centroid_i_y - centroid_j_y)  ∈ {-1, 0, +1}

    Flattened: [x_01, y_01, x_02, y_02, ...] ∈ R^{K*(K-1)}.
    """
    labels = _get_canonical_labels(schema)
    K = len(labels)
    if K < 2:
        return torch.tensor([])

    # Get mean centroid per label
    label_centroids: Dict[str, List[float]] = {}
    for obj in structure.get("objects", []):
        lbl = obj.get("label", "")
        instances = obj.get("instances", [])
        if instances:
            label_centroids[lbl] = _mean_centroid(instances)

    encodings = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            ci = label_centroids.get(labels[i], [0.0, 0.0])
            cj = label_centroids.get(labels[j], [0.0, 0.0])
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            x_sign = 1.0 if dx > 0.1 else (-1.0 if dx < -0.1 else 0.0)
            y_sign = 1.0 if dy > 0.1 else (-1.0 if dy < -0.1 else 0.0)
            encodings.extend([x_sign, y_sign])
    return torch.tensor(encodings)


def phi_full(structure: dict, schema: dict) -> torch.Tensor:
    """ϕ_full = concat([ϕ_count, ϕ_coverage, ϕ_relation]) ∈ R^D."""
    pc = phi_count(structure, schema)
    pv = phi_coverage(structure, schema)
    pr = phi_relation(structure, schema)
    parts = [p for p in [pc, pv, pr] if p.numel() > 0]
    if not parts:
        return torch.tensor([])
    return torch.cat(parts)


def phi_to_dict(structure: dict, schema: dict) -> dict:
    """Return ϕ components as a dict with torch tensors.

    Returns:
        {"count": tensor, "coverage": tensor, "relation": tensor, "full": tensor}
    """
    return {
        "count": phi_count(structure, schema),
        "coverage": phi_coverage(structure, schema),
        "relation": phi_relation(structure, schema),
        "full": phi_full(structure, schema),
    }


# ============================================================
# Simplified batch phi (Top-2, active nouns only)
# 对应 DDPO_TRAINING_METHODOLOGY_CN Section 3.2
# ============================================================

def phi_dicts_simplified(
    structures: List[dict],
    schema: dict,
) -> tuple:
    """Build simplified phi dicts for M chains: filter dead nouns, Top-2 for coverage/relation.

    Per the methodology:
        φ_count:    [K] — all canonical labels (fixed dimension), missing → 0
        φ_coverage: [1] — per-instance intersection ratio between Top-2 objects
        φ_relation: [2] — direction encoding from Top2[0] to Top2[1]

    Args:
        structures: list of M VLM structure JSONs.
        schema: canonical schema dict.

    Returns:
        phi_dicts: list of M dicts with keys "count", "coverage", "relation".
        active_nouns: list of noun labels with total count > 0.
        top2: list of 2 noun labels with highest total count.
        dead_nouns: list of noun labels with total count == 0.
    """
    import numpy as np

    labels = _get_canonical_labels(schema)
    M = len(structures)
    if M == 0 or not labels:
        return [], [], [], []

    # Total counts per noun across all chains
    total_counts = {lbl: 0 for lbl in labels}
    for s in structures:
        for obj in s.get("objects", []):
            lbl = obj.get("label", "")
            if lbl in total_counts:
                total_counts[lbl] += obj.get("count", 0)

    dead_nouns = [l for l, c in total_counts.items() if c == 0]
    active_nouns = [l for l, c in total_counts.items() if c > 0]

    # Top-2 active nouns
    if len(active_nouns) >= 2:
        sorted_active = sorted(active_nouns, key=lambda l: total_counts[l], reverse=True)
        top_count = total_counts[sorted_active[0]]
        second_count = total_counts[sorted_active[1]]
        first_tied = [l for l in active_nouns if total_counts[l] == top_count]
        if len(first_tied) >= 2:
            chosen = list(np.random.choice(first_tied, size=2, replace=False))
            top2 = [str(c) for c in chosen]
        else:
            top2 = [str(first_tied[0]), str(np.random.choice(
                [l for l in active_nouns if total_counts[l] == second_count]))]
    elif len(active_nouns) == 1:
        top2 = [active_nouns[0], active_nouns[0]]
    else:
        top2 = []

    phi_dicts = []
    for s in structures:
        obj_map = {}
        inst_map = {}
        for obj in s.get("objects", []):
            lbl = obj.get("label", "")
            obj_map[lbl] = obj.get("count", 0)
            inst_map[lbl] = obj.get("instances", [])

        # φ_count: [K] — all schema labels, fixed dimension (missing → 0)
        count_vec = torch.tensor([float(obj_map.get(lbl, 0)) for lbl in labels])

        # φ_coverage: [1] — per-instance pairwise intersection ratio
        if len(top2) >= 2 and top2[0] != top2[1]:
            inst_a = inst_map.get(top2[0], [])
            inst_b = inst_map.get(top2[1], [])
            bboxes_a = [i.get("bbox", [0, 0, 0, 0]) for i in inst_a]
            bboxes_b = [i.get("bbox", [0, 0, 0, 0]) for i in inst_b]
            if bboxes_a and bboxes_b:
                total_pairs = len(bboxes_a) * len(bboxes_b)
                intersecting = sum(
                    1 for ba in bboxes_a for bb in bboxes_b
                    if bbox_intersection(ba, bb) > 0
                )
                cov = intersecting / total_pairs
            else:
                cov = 0.0
            cov_vec = torch.tensor([cov])
        else:
            cov_vec = torch.tensor([0.0])

        # φ_relation: [2]
        if len(top2) >= 2 and top2[0] != top2[1]:
            inst_a = inst_map.get(top2[0], [])
            inst_b = inst_map.get(top2[1], [])
            if inst_a and inst_b:
                ca = _mean_centroid(inst_a)
                cb = _mean_centroid(inst_b)
                dx = ca[0] - cb[0]
                dy = ca[1] - cb[1]
                x_sign = 1.0 if dx > 0.1 else (-1.0 if dx < -0.1 else 0.0)
                y_sign = 1.0 if dy > 0.1 else (-1.0 if dy < -0.1 else 0.0)
            else:
                x_sign, y_sign = 0.0, 0.0
            rel_vec = torch.tensor([x_sign, y_sign])
        else:
            rel_vec = torch.tensor([0.0, 0.0])

        phi_dicts.append({
            "count": count_vec,
            "coverage": cov_vec,
            "relation": rel_vec,
        })

    return phi_dicts, active_nouns, top2, dead_nouns
