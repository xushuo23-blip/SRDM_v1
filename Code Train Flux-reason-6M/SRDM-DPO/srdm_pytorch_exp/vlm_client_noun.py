"""
VLM Client (Noun-based) — Multi-backend support (Doubao + Qwen) + speed benchmarking.

两阶段结构提取 (v2):
    Phase 1: 提取名词 → schema JSON (spaCy 或预提取, <0.01s, 零费用)
    Phase 2: VLM 图像结构提取 → structure JSON (含 bbox / count)

VLM 只负责 count + bbox — centroid 和 spatial_relations 由数学计算得出。

支持后端:
    - doubao:    豆包 Seed API (默认, 需要 ARK_API_KEY)
    - qwen:      Qwen2-VL via vLLM / OpenAI-compatible API
    - qwen_local: Qwen2.5-VL 本地加载 (transformers, 不需要额外服务)

Variant benchmarking (VLMVariant + benchmark_variants):
    测试不同加速策略 (压缩/灰度/裁剪/关闭thinking) 的速度与质量对比。

两种客户端:
    - VLMClient:     spaCy POS tagging 提取名词 (基类，向后兼容)
    - VLMClientNoun: 预提取名词模式，从 JSON 加载 prompt→objects 映射

用法:
    # spaCy 模式 (基类)
    client = VLMClient(backend="doubao", model="doubao-seed-2-0-pro-260215")
    schema = client.extract_schema("three cats and two dogs")
    structure = client.extract_structure(pil_image, schema)

    # 预提取名词模式
    client = VLMClientNoun(
        prompt_objects={"a red cube": ["cube", "sphere"], ...},
        backend="doubao",
    )
    schema = client.extract_schema("a red cube")     # 秒级，无 API 调用
    structure = client.extract_structure(pil, schema)  # 调 API

    # Speed benchmark
    from srdm_pytorch_exp.vlm_client_noun import VLMVariant, benchmark_variants
    results = benchmark_variants(image, schema, variants, client)
"""

import base64
import gc
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import requests
from PIL import Image, ImageOps

# spaCy 可选: 未安装时给出友好提示
try:
    import spacy
except ImportError:
    spacy = None

# transformers 可选: 仅 qwen_local 需要
try:
    import torch
except ImportError:
    torch = None

def _format_schema_for_prompt(schema: dict) -> str:
    """Render canonical_objects as a compact label list for the VLM prompt.

    Produces: "cube", "frosting", "sphere"
    """
    labels = [obj["label"] for obj in schema.get("canonical_objects", [])]
    return ", ".join(f'"{l}"' for l in labels) if labels else "(none)"


_STRUCTURE_PROMPT_TEMPLATE = """Detect each object from {OBJECT_SCHEMA} in this image. For each object, record:
- count: integer, how many instances are visible. If the object is NOT present, count=0 and instances=[].
  This is completely normal — not every noun appears in every image.
- For each visible instance: bbox [x1, y1, x2, y2] in normalized [0,1] coordinates (top-left origin, x→right, y→down).

CRITICAL RULES — read carefully before detecting:

1. COUNTABLE vs UNCOUNTABLE: per noun, decide countable or uncountable in this image. Tiny fragments and sub-5%-area specks are uncountable — ignore or merge into one region.
   - COUNTABLE (cube, snowman, tree...): count each visible instance normally.
   - UNCOUNTABLE (frosting, snow, grass, crumbs...): count by REGION. ONE bbox covering the WHOLE contiguous area. Do NOT split into tiny pieces.

2. BACKGROUND: ignore walls, floors, sky, ground, water. Detect if touching a foreground object. When in doubt, detect.

3. BBOX QUALITY:
   - Each bbox must tightly enclose the visible extent of ONE instance.
   - For uncountable nouns, the single bbox covers the ENTIRE contiguous region of that substance.
   - Coordinates must be in [0,1] range. Be precise.

Output ONLY valid JSON following this template exactly:
{{
  "objects": [
    {{"label": "cube", "count": 1, "instances": [{{"bbox": [0.2, 0.6, 0.4, 0.8]}}]}},
    {{"label": "frosting", "count": 1, "instances": [{{"bbox": [0.1, 0.3, 0.9, 0.7]}}]}},
    {{"label": "sphere", "count": 0, "instances": []}}
  ]
}}"""


# ---- spaCy noun extraction ----

_NOUN_POS = {"NOUN", "PROPN"}
# Abstract/semantic nouns that are rarely visual objects — filtered out
_ABSTRACT_NOUNS = {
    "time", "way", "day", "man", "woman", "child", "people", "thing",
    "world", "life", "hand", "part", "place", "case", "week", "company",
    "group", "number", "problem", "fact", "moment", "night", "year",
    "morning", "evening", "afternoon", "kind", "sort", "lot", "bit",
}

# spaCy model name — en_core_web_sm is lightweight and sufficient for POS tagging
_SPACY_MODEL = "en_core_web_sm"
_nlp = None


def _get_nlp():
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is not None:
        return _nlp
    if spacy is None:
        raise ImportError(
            "spaCy 未安装。请运行: pip install spacy && python -m spacy download en_core_web_sm"
        )
    try:
        _nlp = spacy.load(_SPACY_MODEL)
    except OSError:
        raise OSError(
            f"spaCy 模型 '{_SPACY_MODEL}' 未下载。请运行: python -m spacy download {_SPACY_MODEL}"
        )
    return _nlp


def extract_nouns_spacy(prompt: str, filter_abstract: bool = True) -> List[str]:
    """Extract all nouns from a prompt using spaCy POS tagging.

    Returns deduplicated, lemmatized list preserving original order of first occurrence.
    """
    nlp = _get_nlp()
    doc = nlp(prompt)

    seen = set()
    nouns = []
    for token in doc:
        if token.pos_ not in _NOUN_POS:
            continue
        lemma = token.lemma_.lower().strip()
        if len(lemma) < 2:
            continue
        if filter_abstract and lemma in _ABSTRACT_NOUNS:
            continue
        if lemma not in seen:
            seen.add(lemma)
            nouns.append(lemma)

    return nouns


def _build_schema_from_nouns(prompt: str, nouns: List[str]) -> dict:
    """Build a canonical schema JSON from a list of extracted nouns."""
    canonical_objects = [
        {"label": noun, "description": f"'{noun}' mentioned in prompt", "is_primary": True}
        for noun in nouns
    ]
    return {
        "original_prompt": prompt,
        "canonical_objects": canonical_objects,
        "notes": {
            "allowed_labels": nouns,
            "extraction_method": "spacy",
        },
    }


# ---- Image processing utilities ----

def _image_to_base64(image: Image.Image, max_size: int = 512) -> str:
    """PIL Image → base64 data URI, resizing if needed."""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _base64_to_pil(data_uri: str) -> Image.Image:
    """base64 data URI → PIL Image."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    img_bytes = base64.b64decode(data_uri)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _fix_truncated_json(text: str) -> str:
    """Attempt to repair common VLM JSON errors: missing brackets, truncation.

    Common VLM mistakes:
      1. bbox arrays missing ] before }:  [0.1, 0.2, 0.3, 0.4} → [0.1, 0.2, 0.3, 0.4]}
      2. Trailing comma before } or ]:  {"a": 1,} → {"a": 1}
      3. Truncated JSON — missing closing brackets/braces.
    """
    # Fix 1: "number}" → "number]}" inside bbox-like patterns
    # Match: [ number , number , number , number } (missing ])
    text = re.sub(
        r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*([\d.]+)\s*\}',
        r'[\1, \2, \3, \4]}',
        text,
    )
    # Fix 2: trailing comma before } or ]
    text = re.sub(r',\s*\n?\s*\}', '}', text)
    text = re.sub(r',\s*\n?\s*\]', ']', text)
    # Fix 3: close all open brackets/braces (truncation recovery)
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    if open_braces > 0 or open_brackets > 0:
        # Only add closers if text looks like it was truncated (ends mid-structure)
        if not text.rstrip().endswith(('}', ']', '"')):
            text = text.rstrip().rstrip(',')  # remove trailing comma if any
        text += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
    return text


def _extract_json(text: str) -> Optional[dict]:
    """Robustly extract JSON object from VLM response text.

    Tries in order:
      1. Direct parse
      2. Extract from ```json ... ``` fences
      3. Find outermost { ... } via regex
      4. Apply bracket/truncation fixes, then re-try steps 1-3
      5. Regex-based partial extraction (label + count only)
    """
    text = text.strip()

    # Step 1-3: standard extraction
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Step 4: try fixing common errors
    fixed = _fix_truncated_json(text)
    if fixed != text:
        for candidate in _json_candidates(fixed):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    # Step 5: regex-based partial extraction (last resort)
    partial = _extract_partial_objects(text)
    if partial is not None:
        return partial

    return None


def _json_candidates(text: str):
    """Yield candidate JSON strings from text."""
    yield text
    # Extract from ``` fences
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text):
        yield m.group(1).strip()
    # Find outermost { ... }
    # Greedy match: find the first { and the last }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        yield text[start:end + 1]


def _extract_partial_objects(text: str) -> Optional[dict]:
    """Regex-based fallback: extract label + count when JSON is irreparable.

    Matches patterns like: {"label": "table", "count": N, ...}
    and builds minimal objects list.
    """
    # Match: "label": "xxx", "count": N
    pattern = re.compile(
        r'"label"\s*:\s*"([^"]+)"\s*,\s*"count"\s*:\s*(\d+)'
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    objects = []
    for label, count_str in matches:
        # Avoid duplicates
        if not any(o["label"] == label for o in objects):
            objects.append({
                "label": label,
                "count": int(count_str),
                "instances": [],
            })

    # Also try to extract bbox data: "bbox": [x1, y1, x2, y2]
    bbox_pattern = re.compile(
        r'"label"\s*:\s*"([^"]+)".*?"bbox"\s*:\s*\[([^\]]+)\]'
    )
    label_bboxes = {}
    for label, bbox_str in bbox_pattern.findall(text):
        try:
            coords = [float(x.strip()) for x in bbox_str.split(',')]
            if len(coords) == 4:
                if label not in label_bboxes:
                    label_bboxes[label] = []
                label_bboxes[label].append(coords)
        except ValueError:
            continue

    for obj in objects:
        lbl = obj["label"]
        if lbl in label_bboxes:
            obj["instances"] = [{"bbox": b} for b in label_bboxes[lbl]]
            obj["count"] = max(obj["count"], len(label_bboxes[lbl]))

    return {"objects": objects}


from srdm_pytorch_exp.vis_utils import draw_structure_annotations  # re-export

# ---- VLM Variant & Benchmarking ----

@dataclass
class VLMVariant:
    """One VLM extraction variant for speed/quality benchmarking."""
    key: str
    label: str
    strategy: str                      # which acceleration strategy
    max_image_size: int = 512
    grayscale: bool = False
    center_crop_ratio: float = 1.0     # 1.0 = no crop
    disable_thinking: bool = False

    @property
    def is_baseline(self) -> bool:
        return self.key == "baseline_512"


def preprocess_image(image: Image.Image, variant: VLMVariant) -> Image.Image:
    """Apply image-level preprocessing for a variant (crop -> grayscale -> resize)."""
    img = image.copy()

    if variant.center_crop_ratio < 1.0:
        r = variant.center_crop_ratio
        w, h = img.size
        cw, ch = int(w * r), int(h * r)
        left = (w - cw) // 2
        top = (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))

    if variant.grayscale:
        img = ImageOps.grayscale(img).convert("RGB")

    if max(img.size) > variant.max_image_size:
        ratio = variant.max_image_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    return img


def compare_structures(ref_structure: dict, cmp_structure: dict) -> dict:
    """Compare variant structure output vs baseline reference.

    Computes quality metrics to detect if speed optimizations degrade accuracy:
      - count_agreement: fraction of labels with matching counts
      - bbox_iou_mean: mean IoU between matching label union bboxes
      - per_label: per-label count match and bbox IoU

    Returns dict suitable for wandb logging.
    """
    ref_objs = {o.get("label", ""): o for o in ref_structure.get("objects", [])}
    cmp_objs = {o.get("label", ""): o for o in cmp_structure.get("objects", [])}

    all_labels = sorted(set(ref_objs.keys()) | set(cmp_objs.keys()))
    if not all_labels:
        return {
            "count_agreement": 1.0, "bbox_iou_mean": 1.0,
            "n_labels": 0, "per_label": {},
        }

    per_label = {}
    count_matches = 0
    ious = []

    for lbl in all_labels:
        ref_o = ref_objs.get(lbl, {})
        cmp_o = cmp_objs.get(lbl, {})
        ref_count = ref_o.get("count", 0)
        cmp_count = cmp_o.get("count", 0)
        count_ok = (ref_count == cmp_count)

        ref_bboxes = [i.get("bbox", [0, 0, 0, 0]) for i in ref_o.get("instances", [])]
        cmp_bboxes = [i.get("bbox", [0, 0, 0, 0]) for i in cmp_o.get("instances", [])]

        if ref_bboxes and cmp_bboxes:
            u_ref = union_bbox(ref_bboxes)
            u_cmp = union_bbox(cmp_bboxes)
            iou = bbox_iou(u_ref, u_cmp)
            ious.append(iou)
        elif not ref_bboxes and not cmp_bboxes:
            ious.append(1.0)

        if count_ok:
            count_matches += 1

        per_label[lbl] = {
            "ref_count": ref_count, "cmp_count": cmp_count,
            "count_match": count_ok,
            "iou": ious[-1] if ious else None,
        }

    count_agreement = count_matches / len(all_labels) if all_labels else 1.0
    bbox_iou_mean = np.mean(ious) if ious else 1.0

    return {
        "count_agreement": round(count_agreement, 4),
        "bbox_iou_mean": round(bbox_iou_mean, 4),
        "n_labels": len(all_labels),
        "per_label": per_label,
    }


def benchmark_variants(
    image: Image.Image,
    schema: dict,
    variants: List[VLMVariant],
    vlm_client: "VLMClient",
    original_prompt: str = "",
) -> List[dict]:
    """Run all VLM variants on one image, with quality comparison vs baseline.

    Baseline (key="baseline_512") is always run first. Other variants are compared
    against it for quality degradation detection.

    Args:
        image: PIL image to analyze.
        schema: canonical schema dict.
        variants: list of VLMVariant definitions (must include baseline_512).
        vlm_client: VLMClient instance.
        original_prompt: passed through to VLM.

    Returns:
        List of dicts, each with keys:
            variant_key, variant_label, strategy, structure, elapsed,
            is_baseline, quality_vs_baseline (None for baseline itself).
    """
    ordered = sorted(variants, key=lambda v: (0 if v.is_baseline else 1))

    results = []
    baseline_structure = None

    for variant in ordered:
        try:
            structure, elapsed = vlm_client.extract_structure_variant(
                image, schema, variant, original_prompt=original_prompt,
            )
        except Exception as e:
            structure = {"objects": [], "_error": str(e)}
            elapsed = -1.0

        quality = None
        if not variant.is_baseline and baseline_structure is not None:
            quality = compare_structures(baseline_structure, structure)

        result = {
            "variant_key": variant.key,
            "variant_label": variant.label,
            "strategy": variant.strategy,
            "structure": structure,
            "elapsed": elapsed,
            "is_baseline": variant.is_baseline,
            "quality_vs_baseline": quality,
        }
        results.append(result)

        if variant.is_baseline:
            baseline_structure = structure

    return results


def benchmark_variants_batch(
    images: List[Image.Image],
    schema: dict,
    variants: List[VLMVariant],
    vlm_client: "VLMClient",
    original_prompt: str = "",
    max_workers: int = 6,
    stagger_delay: float = 2.0,
) -> List[List[dict]]:
    """Run all VLM variants across multiple images with parallel API calls.

    For each variant, all images are preprocessed and sent to VLM in parallel
    via ThreadPoolExecutor. Baseline variant always runs first (its results
    serve as quality reference for other variants).

    Args:
        images: list of PIL images (one per chain).
        schema: canonical schema dict.
        variants: list of VLMVariant definitions (must include baseline_512).
        vlm_client: VLMClient instance.
        original_prompt: passed through to VLM.
        max_workers: max concurrent VLM calls per variant.
        stagger_delay: delay between launching threads (avoids 429).

    Returns:
        List of per-image results, each a list of per-variant dicts with keys:
            variant_key, variant_label, strategy, structure, elapsed,
            is_baseline, quality_vs_baseline.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ordered = sorted(variants, key=lambda v: (0 if v.is_baseline else 1))
    M = len(images)

    # per_image_results[img_idx][variant_key] = result dict
    per_image_results: List[dict] = [{} for _ in range(M)]
    baseline_structures: List[Optional[dict]] = [None] * M

    for variant in ordered:
        variant_key = variant.key

        def _extract_one(idx: int) -> tuple:
            try:
                structure, elapsed = vlm_client.extract_structure_variant(
                    images[idx], schema, variant, original_prompt=original_prompt,
                )
                return idx, structure, elapsed, None
            except Exception as e:
                return idx, {"objects": [], "_error": str(e)}, -1.0, str(e)

        # Parallel VLM calls for all images under this variant
        with ThreadPoolExecutor(max_workers=min(max_workers, M)) as ex:
            futures = []
            for i in range(M):
                futures.append(ex.submit(_extract_one, i))
                if stagger_delay > 0 and i < M - 1:
                    time.sleep(stagger_delay)

            for future in as_completed(futures):
                idx, structure, elapsed, err = future.result()
                quality = None
                if not variant.is_baseline and baseline_structures[idx] is not None:
                    quality = compare_structures(baseline_structures[idx], structure)

                result = {
                    "variant_key": variant_key,
                    "variant_label": variant.label,
                    "strategy": variant.strategy,
                    "structure": structure,
                    "elapsed": elapsed,
                    "is_baseline": variant.is_baseline,
                    "quality_vs_baseline": quality,
                }
                per_image_results[idx][variant_key] = result

                if variant.is_baseline:
                    baseline_structures[idx] = structure

                if err:
                    print(f"  WARNING img {idx} variant {variant_key}: {err}")

    # Convert dict-of-dicts to list-of-lists (preserving variant order)
    output = []
    for img_idx in range(M):
        img_results = [per_image_results[img_idx][v.key] for v in ordered]
        output.append(img_results)

    return output


class VLMClient:
    """Multi-backend VLM client for structure extraction.

    Phase 1 (schema): spaCy noun extraction (local, fast, free).
    Phase 2 (structure): VLM API call per image.

    Supports:
        - doubao:    豆包 Seed API (default, needs ARK_API_KEY)
        - qwen:      Qwen2-VL via vLLM / OpenAI-compatible API
        - qwen_local: Qwen2.5-VL loaded locally via transformers (no server needed)

    Schema extraction is always done via spaCy (local POS tagging, <0.01s).
    """

    def __init__(
        self,
        backend: str = "doubao",
        api_key: Optional[str] = None,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3/responses",
        model: str = "doubao-seed-2-0-pro-260215",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        backend = backend.lower()
        if backend not in ("doubao", "qwen", "qwen_local"):
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'doubao', 'qwen', or 'qwen_local'."
            )
        self.backend = backend

        if api_key:
            self.api_key = api_key
        elif backend == "doubao":
            self.api_key = os.environ.get("ARK_API_KEY", "")
        elif backend in ("qwen", "qwen_local"):
            self.api_key = os.environ.get("QWEN_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

        if not self.api_key:
            if backend in ("qwen", "qwen_local"):
                pass
            else:
                import getpass
                print("=" * 50)
                print("Doubao VLM API Key 未设置 (env: ARK_API_KEY)。")
                self.api_key = getpass.getpass("请输入 ARK_API_KEY: ").strip()
                print("=" * 50)
                if not self.api_key:
                    raise ValueError("ARK_API_KEY 不能为空。请设置环境变量 ARK_API_KEY 或运行时输入。")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._schema_cache: Dict[str, dict] = {}

        # qwen_local: load model once at init time
        self._local_model = None
        self._local_processor = None
        if backend == "qwen_local":
            self._init_qwen_local()

    def _init_qwen_local(self):
        """Load Qwen2.5-VL model via transformers for local inference."""
        if torch is None:
            raise ImportError("PyTorch is required for qwen_local backend. pip install torch")
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers is required for qwen_local backend. "
                "pip install transformers"
            )
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            print("Warning: qwen_vl_utils not installed. Attempting without it.")
            process_vision_info = None

        print(f"Loading Qwen2.5-VL from {self.model} ...")
        self._local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._local_processor = AutoProcessor.from_pretrained(
            self.model,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self._local_process_vision_info = process_vision_info
        self._local_gen_config = {"max_new_tokens": 1024}
        print("Qwen2.5-VL loaded successfully.")

    def _call_api(self, messages: List[dict], thinking: Optional[dict] = None) -> str:
        """Dispatch to the correct backend call.

        Args:
            messages: VLM API messages.
            thinking: optional Doubao thinking config, e.g. {"type": "disabled"}.
                      None = API default (thinking enabled at full depth).
        """
        if self.backend == "doubao":
            return self._call_doubao(messages, thinking=thinking)
        elif self.backend == "qwen":
            return self._call_qwen(messages, thinking=thinking)
        elif self.backend == "qwen_local":
            return self._call_qwen_local(messages, thinking=thinking)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def _call_doubao(self, messages: List[dict], thinking: Optional[dict] = None) -> str:
        """Call Doubao Seed API with retry logic.

        Doubao-Seed-2.0-pro returns output as a list of items:
          - "reasoning" blocks: model's internal monologue (IGNORED)
          - "message" blocks: actual response with output_text (USED)

        Only message/output_text is returned — reasoning text contaminates JSON parsing.

        Args:
            messages: VLM API messages.
            thinking: optional thinking config, e.g. {"type": "disabled"}.
            reasoning_effort: "minimal"|"low"|"medium"|"high" ("" = default).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": messages,
        }
        if thinking is not None:
            payload["thinking"] = thinking
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                output = data.get("output", data)

                if isinstance(output, list):
                    message_texts = []
                    for item in output:
                        if item.get("type") != "message":
                            continue
                        content = item.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    t = c.get("text", "")
                                    if t:
                                        message_texts.append(t)
                        elif isinstance(content, str):
                            message_texts.append(content)

                    combined = "".join(message_texts)
                    if combined.strip():
                        return combined

                    all_texts = []
                    for item in output:
                        item_type = item.get("type", "")
                        if item_type == "reasoning":
                            summary = item.get("summary", [])
                            if isinstance(summary, list):
                                for s in summary:
                                    if isinstance(s, dict) and s.get("type") == "summary_text":
                                        all_texts.append(s.get("text", ""))
                        else:
                            content = item.get("content", item)
                            if isinstance(content, list):
                                for c in content:
                                    if isinstance(c, dict):
                                        all_texts.append(c.get("text", ""))
                            elif isinstance(content, str):
                                all_texts.append(content)
                    return "".join(all_texts) if all_texts else str(output)

                if isinstance(output, dict):
                    text = output.get("text", "") or output.get("content", "")
                    if text:
                        return str(text)
                return str(output)
            except requests.RequestException as e:
                last_error = e
                # Print full response body for debugging (esp. 429 errors)
                if hasattr(e, 'response') and e.response is not None:
                    print(f"    [DEBUG] HTTP {e.response.status_code}: {e.response.text[:500]}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError(f"Doubao VLM API failed after {self.max_retries} attempts: {last_error}")

    def _call_qwen(self, messages: List[dict], thinking: Optional[dict] = None) -> str:
        """Call Qwen2-VL via OpenAI-compatible API (vLLM).

        Converts Doubao-format messages to OpenAI chat format:
          input_text  → {"type": "text", "text": ...}
          input_image → {"type": "image_url", "image_url": {"url": ...}}
        """
        openai_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content_list = []
            for part in msg.get("content", []):
                if part.get("type") == "input_text":
                    content_list.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "input_image":
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": part.get("image_url", "")},
                    })
            openai_messages.append({"role": role, "content": content_list})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": openai_messages,
        }

        api_url = f"{self.base_url}/v1/chat/completions"
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if text.strip():
                    return text
                return str(data)
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError(f"Qwen VLM API failed after {self.max_retries} attempts: {last_error}")

    def _call_qwen_local(self, messages: List[dict], thinking: Optional[dict] = None) -> str:
        """Call locally loaded Qwen2.5-VL via transformers.

        Converts Doubao-format messages to Qwen chat format and runs inference.
        """
        # Extract text and image from Doubao-format messages
        text_parts = []
        pil_image = None
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "input_text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "input_image":
                    image_uri = part.get("image_url", "")
                    # Convert base64 data URI back to PIL Image
                    pil_image = _base64_to_pil(image_uri)

        query = "\n".join(text_parts)

        # Build Qwen chat format
        content = []
        if pil_image is not None:
            content.append({"type": "image", "image": pil_image})
        if query:
            content.append({"type": "text", "text": query})
        messages_qwen = [{"role": "user", "content": content}]

        # Apply chat template
        text = self._local_processor.apply_chat_template(
            messages_qwen, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        # Process vision info
        image_inputs, video_inputs = None, None
        if self._local_process_vision_info is not None:
            image_inputs, video_inputs = self._local_process_vision_info(messages_qwen)

        inputs = self._local_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to GPU
        device = next(self._local_model.parameters()).device
        inputs = inputs.to(device).to(torch.float16)

        # Generate
        generated_ids = self._local_model.generate(**inputs, **self._local_gen_config)

        # Trim prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self._local_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()

        return response

    def extract_schema(self, prompt: str) -> dict:
        """Extract canonical object schema via spaCy POS tagging (<0.01s).

        Results are cached per prompt.
        """
        if prompt in self._schema_cache:
            return self._schema_cache[prompt]

        nouns = extract_nouns_spacy(prompt)
        result = _build_schema_from_nouns(prompt, nouns)
        self._schema_cache[prompt] = result
        return result

    def extract_structure(self, image: Image.Image, schema: dict, original_prompt: str = "",
                          max_image_size: int = 512) -> dict:
        """Extract structured representation from an image given a canonical schema.

        Sends image + noun list to VLM, returns JSON with objects, counts,
        and per-instance bounding boxes.

        Args:
            max_image_size: max pixel size for image (longest edge). Smaller = faster.
        """
        image_uri = _image_to_base64(image, max_size=max_image_size)

        schema_text = _format_schema_for_prompt(schema)
        user_text = _STRUCTURE_PROMPT_TEMPLATE.format(OBJECT_SCHEMA=schema_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_uri},
                    {"type": "input_text", "text": user_text},
                ],
            }
        ]
        response_text = self._call_api(messages)
        result = _extract_json(response_text)
        if result is None:
            raise ValueError(f"Failed to parse structure JSON from VLM response: {response_text[:500]}")
        return result

    def extract_structure_variant(
        self,
        image: Image.Image,
        schema: dict,
        variant: VLMVariant,
        original_prompt: str = "",
    ) -> tuple:
        """Extract structure with variant-specific preprocessing and API params.

        Applies image preprocessing (crop/gray/resize) then calls VLM with
        optional thinking=disabled for speed benchmarking.

        Args:
            image: PIL image to analyze.
            schema: canonical schema dict.
            variant: VLMVariant with preprocessing + API settings.
            original_prompt: passed to VLM (unused in current prompt template).

        Returns:
            (structure_dict, elapsed_seconds)
        """
        img = preprocess_image(image, variant)
        image_uri = _image_to_base64(img, max_size=variant.max_image_size)

        schema_text = _format_schema_for_prompt(schema)
        user_text = _STRUCTURE_PROMPT_TEMPLATE.format(OBJECT_SCHEMA=schema_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_uri},
                    {"type": "input_text", "text": user_text},
                ],
            }
        ]

        thinking = {"type": "disabled"} if variant.disable_thinking else None
        t0 = time.time()
        response_text = self._call_api(messages, thinking=thinking)
        elapsed = time.time() - t0

        result = _extract_json(response_text)
        if result is None:
            result = {
                "objects": [],
                "_error": "JSON parse failed",
                "_raw_response": response_text[:500],
            }
        return result, elapsed

    def extract_structures_batch(
        self, images: List[Image.Image], schema: dict, original_prompt: str = "",
        max_workers: int = 6,
        stagger_delay: float = 2.0,
        max_image_size: int = 512,
        disable_thinking: bool = False,
        grayscale: bool = False,
    ) -> List[dict]:
        """Extract structures for multiple images in parallel.

        Uses ThreadPoolExecutor for concurrent VLM API calls.
        stagger_delay prevents thundering-herd 429 errors.
        Returns list of structure dicts in the same order as input images.

        Args:
            disable_thinking: if True, set thinking={"type": "disabled"} (4.4x speedup).
            grayscale: if True, convert image to grayscale before sending.
            max_image_size: max pixel size (longest edge). Smaller = faster API response.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        M = len(images)
        results: List[Optional[dict]] = [None] * M

        use_variant = disable_thinking or grayscale
        variant = None
        if use_variant:
            variant = VLMVariant(
                key="batch_custom",
                label=f"batch({'no_thinking' if disable_thinking else ''}{'+gray' if grayscale else ''})",
                strategy="batch",
                max_image_size=max_image_size,
                grayscale=grayscale,
                disable_thinking=disable_thinking,
            )

        def _extract_one(idx: int) -> tuple:
            try:
                if use_variant:
                    struct, _elapsed = self.extract_structure_variant(
                        images[idx], schema, variant, original_prompt=original_prompt)
                    return idx, struct, None
                else:
                    struct = self.extract_structure(
                        images[idx], schema, original_prompt=original_prompt,
                        max_image_size=max_image_size,
                    )
                    return idx, struct, None
            except Exception as e:
                return idx, {"objects": [], "_error": str(e)}, str(e)

        with ThreadPoolExecutor(max_workers=min(max_workers, M)) as ex:
            futures = []
            for i in range(M):
                futures.append(ex.submit(_extract_one, i))
                if stagger_delay > 0 and i < M - 1:
                    time.sleep(stagger_delay)
            for future in as_completed(futures):
                idx, struct, err = future.result()
                results[idx] = struct
                if err:
                    print(f"  WARNING chain {idx}: {err}")

        return results  # type: ignore[return-value]

    def clear_cache(self):
        self._schema_cache.clear()


class VLMClientNoun(VLMClient):
    """VLM 客户端 — 预提取名词模式，无 spaCy。

    继承 VLMClient 的全部 API 调用、结构提取、并行批处理能力，
    仅覆盖 extract_schema() 使用预提取名词替代 spaCy POS tagging。
    """

    def __init__(
        self,
        prompt_objects: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_objects = prompt_objects or {}

    def extract_schema(self, prompt: str) -> dict:
        """用预提取名词构建 schema，不调 spaCy 也不调 VLM API。

        查找: self.prompt_objects[prompt] → 名词列表。
        无匹配时回退到父类 extract_schema (spaCy)。
        """
        if prompt in self._schema_cache:
            return self._schema_cache[prompt]

        nouns = self.prompt_objects.get(prompt)
        if nouns:
            result = _build_schema_from_nouns(prompt, nouns)
            result["notes"]["extraction_method"] = "pre_extracted"
            self._schema_cache[prompt] = result
            return result

        return super().extract_schema(prompt)


# ============================================================
# Bbox validation (post-extraction quality gate)
# ============================================================

def validate_structure_bboxes(structure: dict, max_bad_ratio: float = 0.5) -> bool:
    """Check whether a VLM structure dict has usable bbox data.

    Rejects chains where > max_bad_ratio of instances have non-numeric or
    malformed bboxes, which would otherwise crash normalize_bbox or produce
    meaningless spatial features (coverage intersection / relation direction).

    Called after VLM extraction, before phi computation / bbox drawing.
    """
    objects = structure.get("objects", [])
    if not isinstance(objects, list) or len(objects) == 0:
        return False

    total_instances = 0
    bad_instances = 0

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for inst in obj.get("instances", []):
            if not isinstance(inst, dict):
                bad_instances += 1
                continue
            bbox = inst.get("bbox")
            total_instances += 1
            if bbox is None:
                bad_instances += 1
            elif not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bad_instances += 1
            else:
                try:
                    _ = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    bad_instances += 1

    if total_instances == 0:
        return False
    return (bad_instances / total_instances) <= max_bad_ratio
