"""实验四 Part 2 配置：DDPO 训练 (r_SSR) — fp32 + VLM 压缩模式.

与 Part 1 的关键区别:
    - fp32 精度: ratio 严格 = 1.0, 无 clip 误杀, 训练稳定但慢 ~2x
    - VLM 模式: thinking=enabled + 图片压缩 (max 256px)
      - 不关闭 thinking → 解析精度更高
      - 压缩图片 → 适度加速 (减少 vision token 数量)
      - fp32 生成慢 → VLM 有充足时间完成 thinking, 不需要 no_thinking 极端加速

训练流程:
    - 100 epochs (2400 samples / model)
    - 每 epoch 随机选 3 个 prompt, 各 8 条链 = 24 samples
    - 每 GPU 一个完整模型: GPU 0 = SD3, GPU 1 = SD3.5 (并行采样+训练)
    - 奖励模式: r_SSR alone

用法:
    python experiments/exp4/exp4_part2.py --config config/exp4_part2_config.py
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp4_part2"
    config.seed = 42
    config.mixed_precision = "fp32"
    config.work_dir = "experiments/exp4/logs_checkpoints"

    ###### Model Selection ######
    config.model_ids = ["sd3", "sd35"]

    config.pretrained_model_paths = {
        "sd3": (
            "../../../hf_cache/models--stabilityai--stable-diffusion-3-medium-diffusers/"
            "snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
        ),
        "sd35": (
            "../../../hf_cache/models--stabilityai--stable-diffusion-3.5-medium/"
            "snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
        ),
    }

    ###### Reward Mode ######
    config.reward_mode = "r_ssr"

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Training ######
    config.num_epochs = 100
    config.num_prompts_per_epoch = 3
    config.num_chains_per_prompt = 8
    config.num_inference_steps = 30
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### Per-Model GPU Assignment ######
    config.gpu_sd3 = 0
    config.gpu_sd35 = 1

    ###### VLM (Doubao API, thinking=enabled + 图片压缩) ######
    config.vlm_backend = "doubao"
    config.vlm_model = "doubao-seed-2-0-pro-260215"
    config.vlm_max_image_size = 256          # 压缩: 256px (vs part1 的 512px + no_thinking)
    config.vlm_disable_thinking = False      # thinking=enabled → 解析精度更高
    # 注意: Doubao thinking 仅支持二态 — 默认(enabled) 或 {"type":"disabled"}
    config.vlm_max_workers = 6
    config.vlm_stagger_delay = 2.0
    config.vlm_max_retries = 3

    ###### r_SSR 参数 ######
    config.lambda_count = 0.5
    config.lambda_coverage = 0.25
    config.lambda_relation = 0.25
    config.phi_uniform_weights = True
    config.top_k = 2

    ###### PPO (fp32 稳定: clip_range 可以收紧) ######
    config.ppo_mini_batch_size = 4
    config.ppo_clip_range = 0.2              # fp32 ratio=1.0000, 收紧 clip 提高信号利用率
    config.learning_rate = 3e-5
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 5.0

    ###### TrainingAlerter ######
    config.alert_window = 10
    config.alert_threshold = 3
    config.alert_ratio_bad_pct = 0.5          # fp32: ratio 报警真实有效

    ###### LoRA ######
    config.lora_rank = 8
    config.lora_alpha = 16

    ###### Prompt ######
    config.prompt_file = "data/train_prompts/flux_reason_spatial_short_1k.txt"

    ###### Logging ######
    config.log_interval = 10
    config.save_interval = 10

    return config
