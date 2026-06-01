"""Deep check: verify all experiment scripts' imports and config path references."""
import sys
import os
import ast
import importlib
from pathlib import Path

PROJECT_ROOT = Path("/Users/shuoxu/Desktop/科研自奖励扩散模型/第6期：初期SRDM代码/SRDM训练实验/Code Train Flux-reason-6M/SRDM-DDPO")
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent.parent))  # 3 levels up for absolute imports

RESULTS = []

def check(msg, ok=True):
    status = "OK" if ok else "FAIL"
    RESULTS.append(f"  [{status}] {msg}")
    if not ok:
        print(f"  [FAIL] {msg}")
    return ok

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ============================================================
# 1. Check all srdm_pytorch_exp modules can be imported
# ============================================================
section("1. srdm_pytorch_exp module integrity")

srdm_modules = {
    "prompts_noun": ["load_prompts_from_file", "load_prompt_objects"],
    "vlm_client_noun": ["VLMClient", "VLMClientNoun", "validate_structure_bboxes", "draw_structure_annotations"],
    "reward_rin": ["zscore_normalize", "compute_reward_rin"],
    "reward_ssr": ["compute_r_ssr_batch", "compute_component_distance_l1", "make_distance_plot"],
    "sde_sampling": ["pipeline_sd3_train_sample", "encode_prompt", "make_chain_generators", "total_log_prob_from_list"],
    "structure_features": ["phi_count", "phi_coverage", "phi_relation", "phi_full", "phi_to_dict", "phi_dicts_simplified"],
    "ppo_trainer": ["TrainingAlerter", "compute_log_prob_at_step", "ppo_update_mini_batch"],
    "vis_utils": ["draw_structure_annotations"],
}
srdm_modules["diffusers_patch.flow_match_sde"] = ["StochasticFlowMatchScheduler"]
srdm_modules["diffusers_patch.pipeline_sd3_logprob"] = ["pipeline_sd3_with_logprob"]

for mod_name, expected_attrs in srdm_modules.items():
    try:
        mod = importlib.import_module(f"srdm_pytorch_exp.{mod_name}")
        for attr in expected_attrs:
            ok = hasattr(mod, attr)
            check(f"import srdm_pytorch_exp.{mod_name} -> {attr}", ok)
    except Exception as e:
        check(f"import srdm_pytorch_exp.{mod_name}", False)
        print(f"    Error: {e}")

# ============================================================
# 2. Extract imports from every experiment script
# ============================================================
section("2. Experiment script import analysis")

def extract_imports(filepath):
    """Extract all import statements from a Python file."""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(("import", alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(("from", module, alias.name))
    return imports

exp_scripts = []
for exp_dir in sorted(PROJECT_ROOT.glob("experiments/exp*")):
    for py_file in sorted(exp_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        exp_scripts.append(py_file)

for script in exp_scripts:
    rel = script.relative_to(PROJECT_ROOT)
    print(f"\n  --- {rel} ---")
    imports = extract_imports(script)
    for imp_type, module, name in imports:
        if name is None:
            continue  # bare import
        target = name if imp_type == "import" else f"{module}.{name}" if module else name
        if target.startswith("."):
            continue  # relative imports
        try:
            if imp_type == "import":
                importlib.import_module(name)
            else:
                mod = importlib.import_module(module)
                if not hasattr(mod, name):
                    check(f"{imp_type} {module} import {name}", False)
                else:
                    pass  # OK, don't spam
        except ModuleNotFoundError as e:
            # Check if it's a known third-party package
            known_3rd = {"torch", "diffusers", "transformers", "PIL", "numpy", "tqdm",
                         "wandb", "yaml", "requests", "einops", "peft", "accelerate",
                         "matplotlib", "scipy", "sklearn", "spacy", "cv2", "safetensors"}
            top_level = name if imp_type == "import" else module.split(".")[0]
            if top_level in known_3rd:
                pass  # expected missing in this env
            else:
                check(f"{imp_type} {module} import {name} (module not found: {e})", False)
        except Exception as e:
            check(f"{imp_type} {module} import {name}", False)
            print(f"      Error: {e}")

# ============================================================
# 3. Check data files referenced in configs
# ============================================================
section("3. Config data file references")

config_files = sorted(PROJECT_ROOT.glob("config/*_config.py"))
for cfg_file in config_files:
    rel = cfg_file.relative_to(PROJECT_ROOT)
    print(f"\n  --- {rel} ---")
    with open(cfg_file) as f:
        content = f.read()

    # Find string literals that look like file paths
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets if isinstance(node.targets, list) else [node.targets]:
                if isinstance(target, ast.Attribute) and isinstance(node.value, ast.Constant):
                    val = node.value.value
                    if isinstance(val, str) and (".json" in val or ".txt" in val or ".safetensors" in val or "checkpoint" in val.lower() or ".pt" in val or ".pth" in val or "model" in val.lower() or "data/" in val):
                        attr_name = target.attr
                        # Check if it's a path
                        if val.startswith("data/") or val.startswith("checkpoints/") or val.startswith("scripts/"):
                            full_path = PROJECT_ROOT / val
                            exists = full_path.exists()
                            check(f"{attr_name} = '{val}' -> exists={exists}", exists)

# ============================================================
# 4. Check for missing files in exp dirs
# ============================================================
section("4. Experiment directory completeness")

required_local = {
    "exp1": ["prompts.py", "vlm_client.py"],
    "exp2": ["prompts.py", "vlm_client.py"],
    "exp3": ["prompts.py", "vlm_client.py"],
    "exp4": ["prompts.py", "vlm_client.py"],
}

for exp_name, required_files in required_local.items():
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    for fname in required_files:
        fpath = exp_dir / fname
        check(f"{exp_name}/{fname} exists", fpath.exists())

# ============================================================
# 5. Check exp5-specific files
# ============================================================
section("5. Exp5 special files")

exp5_dir = PROJECT_ROOT / "experiments" / "exp5"
exp5_required = ["gt_utils.py", "reward_gt.py"]
for fname in exp5_required:
    check(f"experiments/exp5/{fname} exists", (exp5_dir / fname).exists())

# Check exp5 imports from gt_utils
if (exp5_dir / "gt_utils.py").exists():
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from experiments.exp5.gt_utils import load_prompt_gt, extract_gt_phi_star
        check("from experiments.exp5.gt_utils import load_prompt_gt", True)
        check("from experiments.exp5.gt_utils import extract_gt_phi_star", True)
    except Exception as e:
        check(f"from experiments.exp5.gt_utils import ... -> {e}", False)

# Check exp5 imports from reward_gt
if (exp5_dir / "reward_gt.py").exists():
    try:
        from experiments.exp5.reward_gt import compute_r_gt_single
        check("from experiments.exp5.reward_gt import compute_r_gt_single", True)
    except Exception as e:
        check(f"from experiments.exp5.reward_gt import ... -> {e}", False)

# ============================================================
# 6. Verify prompt_noun.py functions
# ============================================================
section("6. prompts_noun.py function verification")

from srdm_pytorch_exp.prompts_noun import load_prompts_from_file, load_prompt_objects

# Test load_prompts_from_file with a real file
test_txt = PROJECT_ROOT / "data/train_prompts/flux_reason_spatial_short_1k.txt"
if test_txt.exists():
    prompts = load_prompts_from_file(str(test_txt))
    check(f"load_prompts_from_file('{test_txt.name}') -> {len(prompts)} prompts", len(prompts) > 0)

# Test load_prompt_objects with a real file
test_jsonl = PROJECT_ROOT / "data/train_prompts/flux_reason_spatial_short_1k_gt.jsonl"
if test_jsonl.exists():
    po = load_prompt_objects(str(test_jsonl))
    check(f"load_prompt_objects('{test_jsonl.name}') -> {len(po)} entries", len(po) > 0)
    # Verify structure
    first_key = next(iter(po))
    first_val = po[first_key]
    check(f"  values are List[str] (len={len(first_val)})", isinstance(first_val, list) and isinstance(first_val[0], str))

# ============================================================
# Summary
# ============================================================
section("SUMMARY")

fails = [r for r in RESULTS if "FAIL" in r]
oks = [r for r in RESULTS if "OK" in r]
print(f"\n  Total checks: {len(RESULTS)}")
print(f"  Passed: {len(oks)}")
print(f"  Failed: {len(fails)}")
if fails:
    print(f"\n  FAILURES:")
    for f in fails:
        print(f"  {f}")
else:
    print(f"\n  All checks passed!")
