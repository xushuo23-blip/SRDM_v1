"""实验一 Part1 配置：随机性流匹配采样器验证.

设计:
    - 1 个 prompt，6 条扩散链条
    - 种子分配: seed(chain_j, step_i) = base_seed + j * num_steps + i
      - step_i=0: 初始噪声 x_T 的种子
      - step_i=1..N: 各步随机噪声 ε_t 的种子
    - x_T 预缓存 + 每 a 值重新创建 step generators → 不同 a 值之间种子相同
    - 不同 a 值控制每步噪声幅度 σ_t = a * √(t / (1-t))
    - a=0 时 σ_t=0，退化为确定性采样
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp1_part1"
    config.seed = 42
    config.mixed_precision = "fp16"

    ###### Model Path ######
    config.pretrained_model_path = (
        "../../../hf_cache/models--stabilityai--stable-diffusion-3-medium-diffusers/"
        "snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
    )

    ###### Experiment 1 Part1: Sampler Test ######
    config.a_values = [0, 0.3, 0.5, 0.7, 1.0]
    config.num_chains_per_prompt = 6
    config.num_inference_steps = 30
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### Test Prompts ######
    config.prompt_file = "data/train_prompts/flux_reason_structured_5k.txt"
    config.num_test_prompts = 1

    return config
