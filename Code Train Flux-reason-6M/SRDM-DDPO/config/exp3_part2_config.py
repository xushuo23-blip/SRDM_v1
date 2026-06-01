"""实验三 Part2 配置：VLM 结构提取 — 三种策略对比.

策略:
    - baseline_512:       512px, RGB, thinking=enabled (reference)
    - no_thinking:        512px, RGB, thinking=disabled
    - compress_grayscale: 256px + Grayscale (压缩 + 灰度 结合)

目的:
    - 对比 no_thinking 的加速效果是否稳定 (换了新的 prompt)
    - 测试 compress + grayscale 结合是否能比单独用更好
    - VLM prompt 已加入背景忽略提醒
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp3_part2_vlm_benchmark"
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
    config.num_chains_per_prompt = 6
    config.num_inference_steps = 30
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### VLM ######
    config.vlm_model = "doubao-seed-2-0-pro-260215"
    config.vlm_base_url = "https://ark.cn-beijing.volces.com/api/v3/responses"

    ###### Output ######
    config.output_dir = "experiments/exp3/exp3_part2_output"

    ###### Visualization ######
    config.visualize = True  # 设为 False 可跳过 bbox 绘图，加速训练

    ###### Logging ######
    config.wandb_project = "SRDM-DDPO"

    return config
