"""PRISM-Bench image generation for Exp6 Part1 — SD3.5 r_in+r_SSR Epoch 300.

Variant naming: sd35_exp6_epoch300 → auto-compare against SD3.5 Baseline.

Usage (from Code Test PRISM-Bench/):
    python Rcombine_exp6_part1/gen_images.py
"""

import json
import os

import torch
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..", "hf_cache")

SD35_MODEL_PATH = os.path.abspath(os.path.join(
    CACHE_ROOT, "models--stabilityai--stable-diffusion-3.5-medium",
    "snapshots", "b940f670f0eda2d07fbb75229e779da1ad11eb80",
))

CAPTIONS_DIR = os.path.join(SCRIPT_DIR, "..", "prism-bench-main", "captions", "en")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "logs_checkpoints")
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "images")

HEIGHT = 512
WIDTH = 512
NUM_STEPS = 20
GUIDANCE_SCALE = 5.0
SEED = 42
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]

TRACKS = [
    "imagination", "entity", "text_rendering",
    "style", "affection", "composition", "long_text",
]

VARIANTS = [
    ("sd35_exp6_epoch300", SD35_MODEL_PATH,
     os.path.join(CHECKPOINT_DIR, "sd35_r_in_rssr", "epoch_300.pt")),
]


def load_prompts():
    all_prompts = {}
    for track in TRACKS:
        path = os.path.join(CAPTIONS_DIR, f"{track}.jsonl")
        prompts = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                prompts.append(data["prompt"])
        all_prompts[track] = prompts
        print(f"  {track}: {len(prompts)} prompts")
    return all_prompts


def make_pipeline(model_path, device, checkpoint_path=None):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16,
    ).to(device)

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["transformer_state_dict"]
        n_params = sum(v.numel() for v in state_dict.values())
        print(f"  Checkpoint epoch={ckpt.get('epoch')} model_id={ckpt.get('model_id')} "
              f"reward={ckpt.get('reward_mode')}")
        print(f"  LoRA params: {n_params:,}")

        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0, bias="none",
        )
        pipe.transformer = get_peft_model(pipe.transformer, lora_config)
        pipe.transformer.load_state_dict(state_dict, strict=False)
        pipe.transformer = pipe.transformer.merge_and_unload()

    pipe.transformer.eval()
    return pipe


@torch.no_grad()
def generate_images(pipe, prompts, output_dir, track_name, device):
    os.makedirs(output_dir, exist_ok=True)
    for idx, prompt in enumerate(tqdm(prompts, desc=f"    {track_name}", leave=False)):
        img_path = os.path.join(output_dir, f"{idx}.png")
        if os.path.exists(img_path):
            continue
        generator = torch.Generator(device=device).manual_seed(SEED)
        image = pipe(
            prompt=prompt, height=HEIGHT, width=WIDTH,
            num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE,
            generator=generator, output_type="pil",
        ).images[0]
        image.save(img_path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Output:  {OUTPUT_BASE}")

    print("\nLoading PRISM prompts...")
    all_prompts = load_prompts()

    for variant_name, model_path, ckpt_path in VARIANTS:
        model_short = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name} | model={model_short}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  SKIP: checkpoint not found at {ckpt_path}")
            continue

        print(f"  Base: {model_path}")
        print(f"  Exists: {os.path.isdir(model_path)}")
        print(f"  LoRA: {ckpt_path}")
        pipe = make_pipeline(model_path, device, checkpoint_path=ckpt_path)
        out_base = os.path.join(OUTPUT_BASE, variant_name)

        for track in TRACKS:
            generate_images(pipe, all_prompts[track],
                           os.path.join(out_base, track), track, device)

        del pipe
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All images generated! (1 variant x 7 tracks x 100 images = 700 total)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
