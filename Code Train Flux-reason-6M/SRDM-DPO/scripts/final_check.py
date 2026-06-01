"""Final import verification — simulates actual sys.path per experiment."""
import sys
import os

PROJECT_ROOT = "/Users/shuoxu/Desktop/科研自奖励扩散模型/第6期：初期SRDM代码/SRDM训练实验/Code Train Flux-reason-6M/SRDM-DDPO"
os.chdir(PROJECT_ROOT)

# Simulate what each experiment does: add its dir + project root
def setup_path(exp_name):
    exp_dir = os.path.join(PROJECT_ROOT, "experiments", exp_name)
    sys.path.insert(0, exp_dir)
    sys.path.insert(0, PROJECT_ROOT)

tests = {
    "exp1": [
        ("from prompts import load_prompts_from_file", "experiments.exp1.prompts"),
        ("from srdm_pytorch_exp.diffusers_patch.flow_match_sde import StochasticFlowMatchScheduler", None),
        ("from srdm_pytorch_exp.diffusers_patch.pipeline_sd3_logprob import pipeline_sd3_with_logprob", None),
    ],
    "exp2_part1": [
        ("from prompts import load_prompts_from_file", "experiments.exp2.prompts"),
        ("from srdm_pytorch_exp.reward_rin import compute_reward_rin", None),
    ],
    "exp2_part2": [
        ("from prompts import load_prompts_from_file", "experiments.exp2.prompts"),
        ("from srdm_pytorch_exp.sde_sampling import pipeline_sd3_train_sample, encode_prompt, make_chain_generators, total_log_prob_from_list", None),
        ("from srdm_pytorch_exp.reward_rin import compute_reward_rin", None),
    ],
    "exp2_part3": [
        ("from prompts import load_prompts_from_file", "experiments.exp2.prompts"),
        ("from srdm_pytorch_exp.sde_sampling import pipeline_sd3_train_sample, encode_prompt, make_chain_generators, total_log_prob_from_list", None),
    ],
    "exp3_part1": [
        ("from vlm_client import VLMClient, VLMVariant, compare_structures, draw_structure_annotations", "experiments.exp3.vlm_client"),
        ("from srdm_pytorch_exp.sde_sampling import encode_prompt, make_chain_generators, pipeline_sd3_train_sample, total_log_prob_from_list", None),
    ],
    "exp3_part2": [
        ("from vlm_client import VLMClient, VLMVariant, compare_structures, draw_structure_annotations", "experiments.exp3.vlm_client"),
        ("from srdm_pytorch_exp.sde_sampling import encode_prompt, make_chain_generators, pipeline_sd3_train_sample, total_log_prob_from_list", None),
    ],
    "exp3_part3": [
        ("from vlm_client import draw_structure_annotations, validate_structure_bboxes", "experiments.exp3.vlm_client"),
        ("from srdm_pytorch_exp.structure_features import phi_dicts_simplified", None),
        ("from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch, make_distance_plot", None),
    ],
    "exp4_part1": [
        ("from vlm_client import VLMClient, draw_structure_annotations, validate_structure_bboxes", "experiments.exp4.vlm_client"),
        ("from prompts import load_prompts_from_file", "experiments.exp4.prompts"),
        ("from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch", None),
        ("from srdm_pytorch_exp.structure_features import phi_dicts_simplified", None),
        ("from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch, make_distance_plot", None),
    ],
    "exp4_part2": [
        ("from vlm_client import VLMClient, draw_structure_annotations, validate_structure_bboxes", "experiments.exp4.vlm_client"),
        ("from prompts import load_prompts_from_file", "experiments.exp4.prompts"),
        ("from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch", None),
        ("from srdm_pytorch_exp.structure_features import phi_dicts_simplified", None),
        ("from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch, make_distance_plot", None),
    ],
    "exp4_part3": [
        ("from vlm_client import VLMClient, draw_structure_annotations, validate_structure_bboxes", "experiments.exp4.vlm_client"),
        ("from prompts import load_prompts_from_file", "experiments.exp4.prompts"),
        ("from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch", None),
        ("from srdm_pytorch_exp.structure_features import phi_dicts_simplified", None),
        ("from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch", None),
    ],
    "exp5": [
        ("from srdm_pytorch_exp.vlm_client_noun import VLMClientNoun, draw_structure_annotations, validate_structure_bboxes", None),
        ("from srdm_pytorch_exp.prompts_noun import load_prompts_from_file", None),
        ("from experiments.exp5.gt_utils import load_prompt_gt, extract_gt_phi_star", None),
        ("from experiments.exp5.reward_gt import compute_r_gt_single", None),
        ("from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch", None),
        ("from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch", None),
        ("from srdm_pytorch_exp.structure_features import phi_dicts_simplified", None),
    ],
    "exp6": [
        ("from srdm_pytorch_exp.vlm_client_noun import VLMClientNoun, draw_structure_annotations, validate_structure_bboxes", None),
        ("from srdm_pytorch_exp.prompts_noun import load_prompts_from_file, load_prompt_objects", None),
        ("from srdm_pytorch_exp.ppo_trainer import TrainingAlerter, ppo_update_mini_batch", None),
        ("from srdm_pytorch_exp.reward_ssr import compute_r_ssr_batch", None),
        ("from srdm_pytorch_exp.structure_features import phi_dicts_simplified", None),
        ("from experiments.exp6.reward_exp6 import compute_reward_exp6", None),
    ],
}

all_pass = True
for test_name, imports in tests.items():
    exp_name = test_name.split("_")[0]  # "exp1", "exp2", etc.
    setup_path(exp_name)

    for imp_stmt, alt_module in imports:
        try:
            exec(imp_stmt)
            print(f"  OK  [{test_name}] {imp_stmt}")
        except ImportError as e:
            # Try alternative (for local copies that don't import as srdm modules)
            if alt_module:
                try:
                    exec(f"import {alt_module}")
                    print(f"  OK  [{test_name}] {imp_stmt} (via {alt_module})")
                except ImportError:
                    print(f"  FAIL [{test_name}] {imp_stmt} -> {e}")
                    all_pass = False
            else:
                # Check if it's a known 3rd-party package
                top = imp_stmt.split()[1].split(".")[0]
                known_3rd = {"torch", "diffusers", "transformers", "PIL", "numpy", "tqdm",
                             "wandb", "peft", "absl", "ml_collections", "matplotlib", "cv2",
                             "safetensors", "requests", "einops", "scipy", "sklearn", "spacy"}
                if top in known_3rd:
                    print(f"  OK  [{test_name}] {imp_stmt} (3rd-party, not installed locally)")
                else:
                    print(f"  FAIL [{test_name}] {imp_stmt} -> {e}")
                    all_pass = False

# Data file check
print(f"\n{'='*60}")
print(f"  Data File References")
print(f"{'='*60}")

data_refs = {
    "exp1": "data/train_prompts/flux_reason_structured_5k.txt",
    "exp2 (all)": "data/train_prompts/flux_reason_structured_5k.txt",
    "exp4_part1": "data/train_prompts/flux_reason_spatial_short_1k.txt",
    "exp4_part2": "data/train_prompts/flux_reason_spatial_short_1k.txt",
    "exp4_part3": "data/train_prompts/flux_reason_spatial_short_1k.txt",
    "exp5": "data/train_prompts/flux_reason_spatial_short_1k.txt",
    "exp5 (gt)": "data/train_prompts/flux_reason_spatial_short_1k_gt.jsonl",
    "exp6": "data/train_prompts/flux_reason_spatial_20k.txt",
    "exp6 (objects)": "data/train_prompts/flux_reason_spatial_20k_gt.jsonl",
}

for label, path in data_refs.items():
    exists = os.path.exists(os.path.join(PROJECT_ROOT, path))
    status = "OK" if exists else "FAIL"
    if not exists:
        all_pass = False
    print(f"  [{status}] {label}: {path}")

# Exp dir completeness
print(f"\n{'='*60}")
print(f"  Experiment Directory Completeness")
print(f"{'='*60}")

for exp_name in ["exp1", "exp2", "exp3", "exp4"]:
    exp_dir = os.path.join(PROJECT_ROOT, "experiments", exp_name)
    for fname in ["prompts.py", "vlm_client.py"]:
        exists = os.path.exists(os.path.join(exp_dir, fname))
        status = "OK" if exists else "FAIL"
        if not exists:
            all_pass = False
        print(f"  [{status}] experiments/{exp_name}/{fname}")

print(f"\n{'='*60}")
if all_pass:
    print(f"  ALL CHECKS PASSED")
else:
    print(f"  SOME CHECKS FAILED (see above)")
print(f"{'='*60}")
