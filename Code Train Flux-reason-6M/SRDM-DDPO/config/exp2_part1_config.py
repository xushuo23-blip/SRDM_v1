"""实验二 Part1 配置：r_in 内生奖励验证.

设计:
    - 1 个 prompt, 6 条扩散链条, a=0.7 (实验一选定的最优值)
    - 计算每条链的 total_log_p, r_in = z-score normalize 跨 6 条链
    - 种子方案与实验一完全一致
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp2_part1"
    config.seed = 42
    config.mixed_precision = "fp16"

    ###### Model Path ######
    config.pretrained_model_path = (
        "../../../hf_cache/models--stabilityai--stable-diffusion-3-medium-diffusers/"
        "snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
    )

    ###### SDE Sampler (实验一选定的最优值) ######
    config.a = 0.7

    ###### Experiment 2 Part1: r_in Verification ######
    config.num_chains_per_prompt = 6
    config.num_inference_steps = 30
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### Test Prompt ######
    config.prompt_file = "data/train_prompts/flux_reason_structured_5k.txt"
    config.num_test_prompts = 1

    return config
