"""实验三 Part1 配置：VLM 结构提取验证 + 速度对比 Benchmark.

目的:
    - 验证 Doubao VLM 从生成图像中提取 count + bbox 的准确性
    - 对比 5 种加速策略的速度与质量 (vs baseline_512)
    - 不做训练，仅采样 + VLM 提取 + 分析
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp3_part1_vlm_benchmark"
    config.seed = 42
    config.mixed_precision = "fp16"

    ###### Model Path ######
    config.pretrained_model_path = (
        "../../../hf_cache/models--stabilityai--stable-diffusion-3-medium-diffusers/"
        "snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
    )

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Generation ######
    config.prompt = (
        "A red cube on the left side, two blue spheres on the right side, "
        "the spheres are above the cube. High quality 3D render."
    )
    config.num_chains_per_prompt = 6
    config.num_inference_steps = 30
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### VLM ######
    config.vlm_model = "doubao-seed-2-0-pro-260215"
    config.vlm_base_url = "https://ark.cn-beijing.volces.com/api/v3/responses"

    ###### Output ######
    config.output_dir = "experiments/exp3/exp3_part1_output"

    ###### Visualization ######
    config.visualize = True  # 设为 False 可跳过 bbox 绘图，加速训练

    ###### Logging ######
    config.wandb_project = "SRDM-DDPO"

    return config
