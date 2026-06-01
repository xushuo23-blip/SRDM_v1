"""exp5 Part 1 配置：r_gt + r_in DDPO 训练 — 仅 SD3.5 + fp32 + Doubao thinking + GT 监督信号.

与 exp4_part3 的关键区别:
    - GT 监督信号: DeepSeek 三级提取 (objects count + Top-2 + spatial direction)
    - 奖励: r_gt (count+direction 两阶段门控) + r_in (组内 z-score, 仅 r_gt=0 时生效)
    - VLM: Doubao-lite + thinking 开启 (深度思考)
    - M=6 chains, B=3 mini_batch, N=3 prompts/epoch
    - 50 epochs, lr=1e-5

用法:
    python experiments/exp5/exp5_part1.py --config config/exp5_part1_config.py
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp5_part1"
    config.seed = 42
    config.mixed_precision = "fp32"
    config.work_dir = "experiments/exp5/logs_checkpoints"

    ###### Model Selection (仅 SD3.5) ######
    config.model_ids = ["sd35"]

    config.pretrained_model_paths = {
        "sd35": (
            "../../../hf_cache/models--stabilityai--stable-diffusion-3.5-medium/"
            "snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
        ),
    }

    ###### Reward Mode ######
    config.reward_mode = "r_gt_rin"

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Training ######
    config.num_epochs = 50
    config.num_prompts_per_epoch = 3
    config.num_chains_per_prompt = 6     # M=6: 监督信号下 6 链提供足够正负对比
    config.num_inference_steps = 20
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### GPU Assignment (单 GPU) ######
    config.gpu_sd35 = 0

    ###### VLM (Doubao-pro, thinking 开启, 512px) ######
    config.vlm_backend = "doubao"
    config.vlm_model = "doubao-seed-2-0-pro-260215"
    config.vlm_max_image_size = 512
    config.vlm_disable_thinking = False    # 开启深度思考
    config.vlm_max_workers = 6
    config.vlm_stagger_delay = 2.0
    config.vlm_max_retries = 3

    ###### r_gt 对齐奖励 ######
    config.lambda_gt = 2.0                 # r_gt 的 ±λ 硬信号
    config.direction_threshold = 0.15      # 质心方向判定阈值

    ###### PPO (fp32: clip_range=0.2) ######
    config.ppo_mini_batch_size = 3          # M=6 / B=3 = 2 mini-batches/prompt
    config.ppo_clip_range = 0.2
    config.learning_rate = 1e-5
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 5.0

    ###### TrainingAlerter ######
    config.alert_window = 10
    config.alert_threshold = 3
    config.alert_ratio_bad_pct = 0.5

    ###### LoRA ######
    config.lora_rank = 8
    config.lora_alpha = 16

    ###### Prompt (GT 三级提取版本) ######
    config.prompt_file = "data/train_prompts/flux_reason_spatial_short_1k.txt"
    config.prompt_gt_file = "data/train_prompts/flux_reason_spatial_short_1k_gt.jsonl"

    ###### Logging ######
    config.log_interval = 10
    config.save_interval = 10

    return config
