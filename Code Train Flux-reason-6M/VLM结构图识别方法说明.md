# VLM 结构图识别方法说明

我是这样让 VLM 识别图像结构的：

先做 Phase 1——用 LLM 预先把所有 prompt 里的名词提取出来，存成 `_gt.jsonl`。比如 prompt "a red cube on top of a blue sphere" 对应的条目是 `{"prompt": "...", "objects": {"cube": 1, "sphere": 1}}`。训练时直接查表得到 `["cube", "sphere"]` 当 schema。这一步零 API 调用。

然后 Phase 2——把生成的图片 + schema 标签列表一起发给 VLM（Doubao），让它只检测 schema 里有的物体，输出 count 和 bbox。

发给 VLM 的 prompt：

```
Detect each object from "cube", "sphere" in this image. For each object, record:
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
{
  "objects": [
    {"label": "cube", "count": 1, "instances": [{"bbox": [0.2, 0.6, 0.4, 0.8]}]},
    {"label": "frosting", "count": 1, "instances": [{"bbox": [0.1, 0.3, 0.9, 0.7]}]},
    {"label": "sphere", "count": 0, "instances": []}
  ]
}
```

VLM 返回的 JSON 就是每个物体有 label、count、以及每个实例的归一化 bbox。物体不在图中时 count=0, instances=[]。

Phase 3 纯数学：从 count/bbox 算出 φ_count（数量）、φ_coverage（重叠率）、φ_relation（方位关系），然后和 mode-based prototype 对比得到 r_SSR 奖励。

**核心设计：LLM 预提取名词（离线，一次完成）→ VLM 只负责 count+bbox（在线，按 schema 约束）→ 打分纯数学（不调 VLM）。**
