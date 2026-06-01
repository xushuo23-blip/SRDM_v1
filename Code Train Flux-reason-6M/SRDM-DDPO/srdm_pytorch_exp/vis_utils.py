"""Visualization utilities — bbox/centroid annotation drawing.

Training does NOT need this module. It exists purely for WandB logging
and experiment analysis. Keep vlm_client.py free of drawing code.
"""

from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

# Pre-defined color palette for bounding boxes (bright, distinguishable)
_BBOX_COLORS = [
    (255, 50, 50),     # red
    (50, 180, 50),     # green
    (50, 100, 255),    # blue
    (255, 160, 30),    # orange
    (180, 50, 255),    # purple
    (50, 200, 200),    # cyan
    (255, 60, 180),    # magenta
    (220, 220, 30),    # yellow
    (100, 160, 255),   # light blue
    (255, 120, 120),   # salmon
    (80, 220, 80),     # lime
    (200, 140, 60),    # brown
    (180, 180, 180),   # grey
    (255, 100, 200),   # pink
    (60, 200, 160),    # teal
]


def draw_structure_annotations(
    image: Image.Image,
    structure: dict,
    line_width: int = 2,
    font_size: int = 14,
) -> Image.Image:
    """Draw VLM structure on image: per-instance bboxes + aggregate class centroids.

    Two layers:
        1. Per-instance bounding boxes with labels (thin outline).
        2. Aggregate class centroid markers (filled circles) — mean of all
           instances of that class.

    Args:
        image: PIL Image.
        structure: VLM structure JSON with objects[].instances[].bbox.
        line_width: bounding box stroke width.
        font_size: label text size.

    Returns:
        Annotated PIL Image (copy).
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    objects = structure.get("objects", [])
    if not objects:
        return img

    # ---- Assign color per label ----
    label_colors: Dict[str, tuple] = {}
    for idx, obj in enumerate(objects):
        label = obj.get("label", f"obj_{idx}")
        if label not in label_colors:
            label_colors[label] = _BBOX_COLORS[len(label_colors) % len(_BBOX_COLORS)]

    # ---- Font ----
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
    # ============================================================
    # Layer 1: Per-instance bounding boxes
    # ============================================================
    for obj in objects:
        label = obj.get("label", "?")
        color = label_colors[label]
        for inst in obj.get("instances", []):
            bbox = inst.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                bbox = [float(v) for v in bbox]
            except (ValueError, TypeError):
                continue
            x1, y1, x2, y2 = bbox
            x1, x2 = sorted([max(0.0, min(x1, 1.0)), max(0.0, min(x2, 1.0))])
            y1, y2 = sorted([max(0.0, min(y1, 1.0)), max(0.0, min(y2, 1.0))])
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)

            draw.rectangle([px1, py1, px2, py2], outline=color, width=line_width)

            text = f"{label}"
            text_y = max(0, py1 - font_size - 4)
            try:
                bbox_text = draw.textbbox((px1, text_y), text, font=font)
            except (AttributeError, TypeError):
                bbox_text = None
            if bbox_text:
                tx1, ty1, tx2, ty2 = bbox_text
                draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
                draw.text((tx1, ty1), text, fill=(255, 255, 255), font=font)
            else:
                draw.text((px1, text_y), text, fill=color, font=font)

    # ============================================================
    # Layer 2: Aggregate class centroids (per-class mean centroid)
    # ============================================================
    agg_centroids: Dict[str, tuple] = {}
    for obj in objects:
        label = obj.get("label", "")
        instances = obj.get("instances", [])
        if not instances:
            continue
        cxs, cys = [], []
        for inst in instances:
            cent = inst.get("centroid")
            if isinstance(cent, (list, tuple)) and len(cent) == 2:
                cxs.append(cent[0])
                cys.append(cent[1])
        if cxs:
            agg_centroids[label] = (sum(cxs) / len(cxs), sum(cys) / len(cys))

    centroid_radius = max(8, int(min(w, h) * 0.018))
    for label, (cx, cy) in agg_centroids.items():
        px, py = int(cx * w), int(cy * h)
        color = label_colors.get(label, (255, 255, 255))
        draw.ellipse(
            [px - centroid_radius, py - centroid_radius,
             px + centroid_radius, py + centroid_radius],
            fill=color, outline=(255, 255, 255), width=3,
        )
        cr = centroid_radius // 2
        draw.line([px - cr, py, px + cr, py], fill=(255, 255, 255), width=2)
        draw.line([px, py - cr, px, py + cr], fill=(255, 255, 255), width=2)
        try:
            tbbox = draw.textbbox((px + centroid_radius + 6, py - font_size // 2), label, font=font)
        except (AttributeError, TypeError):
            tbbox = None
        if tbbox:
            draw.text((px + centroid_radius + 6, py - font_size // 2), label, fill=color, font=font)
        else:
            draw.text((px + centroid_radius + 4, py - 6), label, fill=color, font=font)

    return img
