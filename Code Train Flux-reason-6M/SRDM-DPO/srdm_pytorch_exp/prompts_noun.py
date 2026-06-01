"""通用 JSON Prompt 加载器 — 从任意含 prompt + objects 的 JSON/JSONL 文件提取.

用法:
    from srdm_pytorch_exp.prompts_noun import load_prompts_from_file, load_prompt_objects

    prompts = load_prompts_from_file("data/train_prompts/xxx.txt")
    objects_map = load_prompt_objects("data/train_prompts/xxx_gt.jsonl")
    # objects_map: {prompt_text: ["noun1", "noun2", ...]}
"""

import json
import os
from typing import Dict, List


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file, one prompt per line."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_prompt_objects(file_path: str) -> Dict[str, List[str]]:
    """从 JSON/JSONL 文件加载 prompt → object labels 映射.

    自动检测文件格式:
        - .jsonl: 每行一个 JSON (如 _gt.jsonl)
        - .json:  单个 JSON 对象或数组

    每条记录只需含:
        - "prompt": str  — 提示词文本
        - "objects": dict 或 list
            - dict: {"noun": count, ...}  → 取 .keys() 作为标签
            - list: ["noun1", "noun2", ...] → 直接作为标签

    Returns:
        {prompt_text: [noun_label_list]}
        文件不存在时返回 {}。
    """
    if not os.path.exists(file_path):
        return {}

    ext = os.path.splitext(file_path)[1].lower()
    mapping: Dict[str, List[str]] = {}

    if ext == ".jsonl":
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item.get("prompt", "")
                objects = item.get("objects", {})
                if prompt and objects:
                    mapping[prompt] = _objects_to_labels(objects)
    elif ext == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for item in items:
            prompt = item.get("prompt", "")
            objects = item.get("objects", {})
            if prompt and objects:
                mapping[prompt] = _objects_to_labels(objects)

    return mapping


def _objects_to_labels(objects) -> List[str]:
    """Normalize objects field to a list of label strings.

    Args:
        objects: dict {label: count} or list [label, ...]

    Returns:
        list of label strings.
    """
    if isinstance(objects, dict):
        return list(objects.keys())
    if isinstance(objects, list):
        return [str(x) for x in objects]
    return []
