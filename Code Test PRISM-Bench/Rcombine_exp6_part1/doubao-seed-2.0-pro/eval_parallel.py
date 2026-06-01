"""Parallel PRISM-Bench API evaluation — replaces eval_gpt41.py with ThreadPool.

Usage:
    python baseline_eval/eval_parallel.py \
        --image_path baseline_eval/images/sd3_baseline \
        --api_key sk-xxx --base_url https://ark.cn-beijing.volces.com/api/v3 \
        --api_model doubao-seed-2-0-pro-260215 --workers 12
"""

import argparse
import base64
import io
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from PIL import Image

# ============================================================
# JSON parse (same as eval_gpt41)
# ============================================================

def clean_and_parse_json(json_str: str) -> Dict[str, Any]:
    json_str = json_str.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()
    json_str = re.sub(r",\s*(?=[}\]])", "", json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            import demjson3
            return demjson3.decode(json_str)
        except Exception:
            return {}


def encode_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format=image.format or "PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# Message templates (same as eval_gpt41)
# ============================================================

def _get_message_templates():
    """Returns {alignment|aesthetic: {category: messages}}."""
    m1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against its text prompt using a strict, two-step process. You will provide a one-sentence justification and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
Core Principle: The primary criterion is always Text-Image Alignment. The image must first be a faithful depiction of the literal content described in the prompt. The evaluation of the emotional aspect is a secondary, but important, step.
9-10 (Exceptional): Flawless. The image perfectly depicts all literal content from the prompt AND masterfully visualizes the specified emotion with depth and creativity.
7-8 (Good): The image depicts all literal content correctly, AND the emotional visualization is strong and accurate.
5-6 (Average): A competent attempt. The image depicts the literal content correctly, but the emotional visualization is weak, superficial, or relies heavily on cliches.
3-4 (Poor): Major failure in content alignment. Key subjects, objects, or settings from the prompt are missing or wrong. The emotional evaluation is largely irrelevant because the core content is incorrect.
0-2 (Failure): The image shows no significant resemblance to the literal content of the prompt.

Track-Specific Instructions: A Two-Step Evaluation
You must follow this sequence. Start at 10 and deduct points for each failure.
Step 1: Verify Content Alignment (Primary Criterion)
First, ignore the emotional component and check only the physical description. Does the image contain the correct subjects, objects, setting, and actions?
Content Mismatch (-6 to -8 points): This is the most severe failure. The image is missing a key subject, setting, or object described in the prompt. If the core content is wrong, the score cannot be high.
Attribute Error (-3 to -5 points): The content is generally right, but key attributes are wrong.
Step 2: Evaluate Emotional Visualization (Secondary Criterion)
Only after confirming the content alignment, evaluate the emotional layer.
Emotional Dissonance (-3 to -5 points): The image content is correct, but the mood is completely wrong. The lighting, colors, and composition fail to evoke the requested emotion.
Missing Nuance / Cliched Symbolism (-2 to -4 points): The content is correct, but the emotion is handled superficially. The image uses an obvious cliche without any depth, or it captures a generic version of the emotion.
Literal Interpretation of Emotion (-2 to -4 points): The content is correct, but the emotion is interpreted in a clumsy, literal way.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    m2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against its text prompt, focusing on object count and spatial relationships. You will provide a one-sentence justification and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. Every object, count, attribute, and spatial relationship is rendered with perfect accuracy and logical consistency.
7-8 (Good): The main objects and their primary relationships are correct. There might be a single, minor error in a secondary object's attribute or position.
5-6 (Average): A competent attempt. The image contains the correct primary objects, but there are significant errors in their count, spatial relationships, or interactions.
3-4 (Poor): Major errors in object count or the relationships between primary objects. The scene is fundamentally incorrect.
0-2 (Failure): The wrong objects are depicted, or the image is completely unrelated to the prompt.

Track-Specific Instructions: Object Layout and Relationships
Start at 10 and deduct points for each failure. Be systematic.
Incorrect Object Count (-3 to -5 points): The number of a key object is wrong.
Incorrect Spatial Relationship (-3 to -5 points): The relative position of key objects is wrong.
Incorrect Object Attributes (-2 to -4 points): A key object has the wrong color, size, or other specified attribute.
Incorrect Interactions (-2 to -4 points): A described interaction between objects or subjects is missing or wrong.
Minor Positional/Attribute Errors (-1 to -2 points): A secondary object is slightly misplaced or has a minor incorrect attribute.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    m3 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a text prompt naming a specific entity. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The entity is rendered with photographic accuracy, and the surrounding scene perfectly matches all details in the prompt.
7-8 (Good): The entity is highly recognizable and accurate, and the overall scene is a good match for the prompt with only minor deviations.
5-6 (Average): A competent attempt. The entity is recognizable but has clear flaws, OR the entity is perfect but the surrounding scene described in the prompt is incorrect. An accurate entity in a wrong context is not a success.
3-4 (Poor): The entity is barely recognizable or is a generic substitute. The scene is also likely incorrect.
0-2 (Failure): The entity is wrong or absent, and the image is unrelated to the prompt.

Track-Specific Instructions: Specific Entity Generation
Start at 10 and deduct points for each failure. Prioritize overall alignment, then entity accuracy.
Incorrect Scene/Context (-4 to -6 points): The entity is correct, but the background, style, or action described in the prompt is completely wrong. This is a major failure.
Unrecognizable or Flawed Entity (-3 to -5 points): The entity is poorly rendered, has significant anatomical or structural errors, or looks like a generic version.
Missing Scene Details (-2 to -4 points): The scene is generally correct, but key descriptive elements are missing.
Minor Entity Inaccuracies (-1 to -3 points): The entity is recognizable but has small, specific inaccuracies.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    m4 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a text prompt describing an imaginative object. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. All described features are seamlessly and creatively integrated into a coherent, believable whole. The object feels truly unique and masterfully executed.
7-8 (Good): The object is well-designed and incorporates almost all key features from the prompt with good coherence.
5-6 (Average): A competent attempt. The object includes the main features described, but they appear "stitched together" or incoherent. Key details are missing or misinterpreted. The result is a recognizable but flawed collage of ideas.
3-4 (Poor): The object is a confusing mess, missing most of the core features described in the prompt.
0-2 (Failure): The object is completely wrong or the image is unrelated to the prompt.

Track-Specific Instructions: Imaginative Object Generation
Start at 10 and deduct points for each failure. Focus on coherence.
Missing Core Features (-4 to -6 points): Fails to include a defining feature of the object.
Lack of Coherence (-3 to -5 points): The described parts are present but look like a poorly assembled collage rather than a single, integrated object.
Misinterpreted Attributes (-2 to -4 points): A key material or quality is rendered incorrectly.
Incorrect Context (-1 to -3 points): The object is rendered well, but the surrounding environment described in the prompt is wrong.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    m5 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a text prompt requesting a specific style. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The image perfectly captures the content and executes the requested style with deep, nuanced understanding of its aesthetics, techniques, and historical context.
7-8 (Good): The content is correct, and the style is clearly recognizable and well-executed, with only minor deviations from the style's core principles.
5-6 (Average): A competent but superficial attempt. The content is correct, but the style is applied like a simple filter. It captures the most obvious stylistic cliches but misses the nuance of the art form.
3-4 (Poor): The content is correct but the style is wrong, OR the style is vaguely correct but the content is wrong.
0-2 (Failure): Both content and style are wrong.

Track-Specific Instructions: Specific Style Application
Start at 10 and deduct points for each failure. Penalize superficiality.
Incorrect Content (-5 to -7 points): The image shows the wrong subject matter, even if the style is correct. This is a major failure.
Superficial Style Application (-4 to -6 points): The image uses only the most obvious cliches of a style without understanding its underlying principles.
Missing Stylistic Elements (-2 to -4 points): The image misses key technical identifiers of the style.
Inconsistent Style (-1 to -3 points): Parts of the image are in the correct style while other parts are not.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    m6 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image that should contain rendered text. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The text is perfectly spelled, legible, and seamlessly integrated into the scene with correct perspective, lighting, and texture.
7-8 (Good): The text is perfectly spelled and legible, with only very minor issues in its integration.
5-6 (Average): A competent attempt. The text is spelled correctly but is poorly integrated into the scene. It may look flat, have unnatural lighting, or be placed awkwardly.
3-4 (Poor): The text contains significant spelling errors or is partially illegible, even if the placement is roughly correct.
0-2 (Failure): The text is nonsensical, completely wrong, or absent.

Track-Specific Instructions: In-Image Text Generation
Start at 10 and deduct points for each failure. Text accuracy is paramount.
Spelling or Wording Errors (-6 to -8 points): Any deviation from the requested text string. This is the most severe failure.
Poor Integration (-3 to -5 points): The text looks pasted on, with incorrect perspective, lighting, or shadows for the scene.
Illegibility (-3 to -5 points): The characters are garbled, distorted, or difficult to read.
Incorrect Placement/Font (-2 to -4 points): The text is on the wrong object or in the wrong location, or the requested font style is ignored.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    m7 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a highly critical AI evaluator for a text-to-image generation benchmark. Your task is to meticulously analyze a generated image against a long, detailed text prompt. You will provide one-sentence justification for point deductions and a score from 0 to 10 in JSON format. Your evaluation must be stringent.

Scoring Philosophy (Apply this strictly):
9-10 (Exceptional): Flawless. The image comprehensively and coherently visualizes virtually every detail from the prompt, from major elements to minor attributes.
7-8 (Good): The image captures all major elements and a clear majority of the secondary details and attributes. The omissions are minor.
5-6 (Average): A competent attempt. The image correctly depicts the main subject and setting but omits a significant number of secondary details and attributes. The core is there, but the richness is lost.
3-4 (Poor): The image captures only one of the major elements and misses almost all descriptive details.
0-2 (Failure): The image fails to capture any of the major elements described in the prompt.

Track-Specific Instructions: Long Text Comprehension
Start at 10 and deduct points for each failure. Be a detail-oriented critic.
First, identify the Major Elements (primary subject, setting, main action).
Second, list all Secondary Details (other objects, characters, specific attributes).
Deduct points for each omission or error.
Missing a Major Element (-5 to -7 points): Fails to include the primary subject, setting, or action.
Missing a Majority of Secondary Details (-3 to -5 points): The image feels generic because it ignored most of the specific descriptors that gave the prompt its character.
Incorrectly Rendered Detail (-2 to -4 points): A detail is included but rendered incorrectly.
Each Minor Omission (-1 point): For every small, specific detail that is missing, deduct a point.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    m8 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": """
You are a hyper-critical quality assurance inspector for a text-to-image generation benchmark. Your task is to evaluate images with forensic, microscopic scrutiny. Your primary directive is to penalize any deviation from physical, anatomical, and logical coherence, unless such deviations are explicitly requested by the text prompt. Assume all subjects and environments must be perfectly sound and plausible by default.

Scoring System: You will start with a perfect score of 10 and deduct points for any flaws you identify. A single significant flaw should prevent a high score.

Flaw Categories (Deduct points for each instance):
Critical Failures (-7 to -9 points):
Any violation of the fundamental anatomical or structural integrity of the main subjects. This includes inconsistencies in form, function, or natural appearance.
A breakdown in logical or physical plausibility within the scene, when not specified by the prompt.
Prominent, distracting digital artifacts, watermarks, or signatures that ruin immersion.
The central subject is rendered as grotesque or nonsensical, when not specified by the prompt.
Significant Flaws (-4 to -6 points):
Noticeable warping, distortion, or a lack of convincing texture on key objects or surfaces.
Unnatural blending, texture repetition, or other clear indicators of AI synthesis that break realism.
Lack of sharpness or resolution in the primary subject, making crucial details indistinct.
Incoherent or illogical features on secondary elements.
Minor Imperfections (-1 to -3 points):
Slight compositional awkwardness or minor issues with lighting and shadow that don't break realism.
Minimal blurriness or noise in secondary, non-focal areas of the image.
Faint, non-distracting artifacts that are only visible upon close inspection.

Required Output Format:
Your response must be a single JSON object containing a one-sentence " justification " for point deductions and a "score":
{
"justification": ...,
"score": ...
}

text prompt: {text_prompt}
"""}]}]

    return {
        "alignment": {
            "affection": m1, "composition": m2, "entity": m3,
            "imagination": m4, "style": m5, "text_rendering": m6, "long_text": m7,
        },
        "aesthetic": m8,
    }


# ============================================================
# Single-image scoring task (runs in thread)
# ============================================================

def _score_one_image(
    client: OpenAI,
    model: str,
    messages_template: list,
    text_prompt: str,
    img_path: Path,
    save_path: Path,
    task_id: str,
) -> dict:
    """Score a single image. Returns {"ok": bool, "score": float, "time": float}."""
    t0 = time.time()
    result = {"ok": False, "score": 0.0, "time": 0.0, "task_id": task_id}

    if save_path.exists():
        try:
            with open(save_path) as f:
                data = json.load(f)
            result["ok"] = True
            result["score"] = float(data.get("score", 0))
            result["time"] = 0.0
            result["skipped"] = True
            return result
        except Exception:
            pass  # corrupted, re-score

    try:
        image = Image.open(img_path)
        b64 = encode_image_to_base64(image)
    except Exception as e:
        print(f"  [{task_id}] image load error: {e}")
        result["time"] = time.time() - t0
        return result

    msgs = deepcopy(messages_template)
    img_msg = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    msgs[1]["content"].append(img_msg)
    msgs[1]["content"][0]["text"] = msgs[1]["content"][0]["text"].replace("{text_prompt}", text_prompt)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=4096,
            temperature=0.0,
            top_p=1.0,
        )
        output = completion.choices[0].message.content
    except Exception as e:
        print(f"  [{task_id}] API error: {type(e).__name__}: {e}")
        result["time"] = time.time() - t0
        return result

    if not output:
        result["time"] = time.time() - t0
        return result

    data = clean_and_parse_json(output)
    if not data:
        result["time"] = time.time() - t0
        return result

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    result["ok"] = True
    result["score"] = float(data.get("score", 0))
    result["time"] = time.time() - t0
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel PRISM-Bench API evaluator")
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--api_model", type=str, default="doubao-seed-2-0-pro-260215")
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of concurrent API workers (default: 12)")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Score output dir (default: [image_path]/score)")
    args = parser.parse_args()

    EVAL_POOLS = ["imagination", "entity", "text_rendering", "style", "affection", "composition", "long_text"]
    CAPTIONS_DIR = Path(__file__).resolve().parent.parent.parent / "prism-bench-main" / "captions" / "en"

    score_root = args.output_dir or (args.image_path / "score")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    templates = _get_message_templates()

    # Collect all tasks: (eval_type, category, img_idx, img_path, save_path, prompt)
    all_tasks = []

    for eval_type in ["alignment", "aesthetic"]:
        tmpl = templates[eval_type]
        for cat in EVAL_POOLS:
            img_dir = args.image_path / cat
            cap_file = CAPTIONS_DIR / f"{cat}.jsonl"
            prompts = []
            with open(cap_file, "r") as f:
                for line in f:
                    item = json.loads(line.strip())
                    prompts.append(item.get("prompt", ""))

            for idx in range(min(100, len(prompts))):
                img_path = img_dir / f"{idx}.png"
                save_path = score_root / eval_type / cat / f"{idx}.jsonl"
                if not img_path.exists():
                    continue
                msgs = tmpl if eval_type == "aesthetic" else tmpl[cat]
                all_tasks.append((eval_type, cat, idx, img_path, save_path, prompts[idx], msgs))

    total = len(all_tasks)
    done_before = sum(1 for _, _, _, _, sp, _, _ in all_tasks if sp.exists())
    print(f"Total tasks: {total}  |  Already done: {done_before}  |  To run: {total - done_before}")
    print(f"Workers: {args.workers}  |  Model: {args.api_model}\n")

    if total - done_before == 0:
        print("All tasks already completed.")
        return

    completed = 0
    errors = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for task in all_tasks:
            eval_type, cat, idx, img_path, save_path, prompt, msgs = task
            tid = f"{eval_type[:4]}/{cat[:6]}/{idx}"
            fut = executor.submit(
                _score_one_image, client, args.api_model, msgs, prompt, img_path, save_path, tid
            )
            futures[fut] = tid

        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  [{tid}] thread exception: {e}")
                errors += 1
                continue

            if r["ok"]:
                completed += 1
                skipped = " (cached)" if r.get("skipped") else ""
                if not r.get("skipped"):
                    remaining = total - done_before - completed
                    elapsed = time.time() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = remaining / rate if rate > 0 else 0
                    print(f"  [{tid}] score={r['score']:.0f}  {r['time']:.1f}s  |  "
                          f"done={completed}/{total - done_before}  rate={rate:.1f}/s  ETA={eta/60:.0f}min{skipped}")
            else:
                errors += 1
                print(f"  [{tid}] FAILED  {r['time']:.1f}s")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done!  completed={completed}  errors={errors}  elapsed={elapsed/60:.1f}min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
