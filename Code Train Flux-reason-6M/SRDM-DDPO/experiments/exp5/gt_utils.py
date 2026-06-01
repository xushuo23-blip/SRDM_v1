"""Exp5 专用: GT 对齐工具 — DeepSeek 三级提取 (objects + top2 + spatial_relation).

搬自 srdm_pytorch_exp/prompts_noun.py 和 vlm_client_noun.py 的 exp5 特有函数。
仅 exp5 使用，其他实验不依赖此文件。
"""

import json
import os
from typing import Dict, List, Optional


# ============================================================
# GT 数据加载 (原 prompts_noun.py)
# ============================================================

def load_prompt_gt(file_path: str) -> Dict[str, dict]:
    """Load GT supervision signals from _gt.jsonl (DeepSeek 三级提取).

    Each line: {
        "prompt": "...",
        "objects": {"noun": count, ...},
        "top2_objects": ["obj_a", "obj_b"],
        "spatial_relation": {"from": "obj_a", "to": "obj_b", "direction": [dx, dy]}
    }

    Returns:
        dict of {prompt_text: {"objects": {name: count}, "top2_objects": [...], "spatial_relation": {...}}}
        Empty dict if file doesn't exist.
    """
    if not os.path.exists(file_path):
        return {}
    mapping = {}
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt_text = item.get("prompt", "")
            if not prompt_text:
                continue
            mapping[prompt_text] = {
                "objects": item.get("objects", {}),
                "top2_objects": item.get("top2_objects", []),
                "spatial_relation": item.get("spatial_relation", {}),
            }
    return mapping


def extract_gt_phi_star(prompt_gt: dict) -> dict:
    """从 GT dict 提取 phi*（标准答案），供 Hard Gate 对齐使用。

    GT 目前仅提供 count 和 direction 两个分量:
        - count: {object_label: count, ...} — GT 物体数量
        - direction: [dx, dy] — GT Top-2 空间方位向量, dx,dy ∈ {-1, 0, 1}

    Args:
        prompt_gt: load_prompt_gt() 返回的单个 prompt GT dict。

    Returns:
        {"count": {str: int}, "direction": [int, int]}
    """
    return {
        "count": prompt_gt.get("objects", {}),
        "direction": prompt_gt.get("spatial_relation", {}).get("direction", [0, 0]),
    }


# ============================================================
# Top-2 方向 / 数量提取 (原 vlm_client_noun.py)
# ============================================================

def compute_top2_direction_from_structure(
    structure: dict,
    top2_objects: List[str],
    threshold: float = 0.15,
) -> List[int]:
    """从 VLM structure JSON 计算 Top-2 物体的实际方向向量。

    对 Top-2 两物体的所有 instance bbox 求平均质心，编码方向:
        dx = sign(cx_A - cx_B)  ∈ {-1, 0, 1}
        dy = sign(cy_A - cy_B)  ∈ {-1, 0, 1}

    Args:
        structure: VLM 返回的 structure JSON。
        top2_objects: [obj_a, obj_b] 两个 canonical label。
        threshold: 质心差小于此阈值视为 0（无偏移）。

    Returns:
        [dx, dy] 方向向量。若 Top-2 不足两个不同物体，返回 [0, 0]。
    """
    if len(top2_objects) < 2 or top2_objects[0] == top2_objects[1]:
        return [0, 0]

    obj_a, obj_b = top2_objects[0], top2_objects[1]

    def _get_instances(label: str) -> List[dict]:
        for obj in structure.get("objects", []):
            if obj.get("label", "") == label:
                return obj.get("instances", [])
        return []

    def _mean_centroid(instances: List[dict]) -> List[float]:
        if not instances:
            return [0.0, 0.0]
        cxs, cys = [], []
        for inst in instances:
            bbox = inst.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    cxs.append((float(bbox[0]) + float(bbox[2])) / 2.0)
                    cys.append((float(bbox[1]) + float(bbox[3])) / 2.0)
                except (TypeError, ValueError):
                    pass
        if not cxs:
            return [0.0, 0.0]
        return [sum(cxs) / len(cxs), sum(cys) / len(cys)]

    inst_a = _get_instances(obj_a)
    inst_b = _get_instances(obj_b)
    if not inst_a or not inst_b:
        return [0, 0]

    ca = _mean_centroid(inst_a)
    cb = _mean_centroid(inst_b)
    dx = ca[0] - cb[0]
    dy = ca[1] - cb[1]
    x_sign = 1 if dx > threshold else (-1 if dx < -threshold else 0)
    y_sign = 1 if dy > threshold else (-1 if dy < -threshold else 0)
    return [x_sign, y_sign]


def get_top2_counts_from_structure(
    structure: dict,
    top2_objects: List[str],
) -> Dict[str, int]:
    """从 VLM structure JSON 提取 Top-2 物体的检测数量。

    Args:
        structure: VLM 返回的 structure JSON。
        top2_objects: [obj_a, obj_b] 两个 canonical label。

    Returns:
        {obj_a: count, obj_b: count}。未检测到的物体 count=0。
    """
    counts = {obj: 0 for obj in top2_objects}
    for obj in structure.get("objects", []):
        lbl = obj.get("label", "")
        if lbl in counts:
            counts[lbl] = obj.get("count", 0)
    return counts
