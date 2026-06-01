"""实验四 Part 3 配置：DDPO 训练 (r_SSR) — 仅 SD3.5 + fp32 + no_thinking + spaCy.

与 Part 1/2 的关键区别:
    - 仅 SD3.5 单模型 (vs 双模型对比)
    - fp32 精度: ratio 严格 = 1.0, 训练稳定但慢 ~2x
    - VLM: no_thinking 模式 (4.4x 加速) + 512px 原图
    - spaCy 实时名词提取 → 秒级 schema 构建，零费用
    - 1K prompt 数据集 (flux_reason_spatial_short_1k)

用法:
    python experiments/exp4/exp4_part3.py --config config/exp4_part3_config.py
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp4_part3"
    config.seed = 42
    config.mixed_precision = "fp32"
    config.work_dir = "experiments/exp4/logs_checkpoints"

    ###### Model Selection (仅 SD3.5) ######
    config.model_ids = ["sd35"]

    config.pretrained_model_paths = {
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
    config.num_inference_steps = 20
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### GPU Assignment (单 GPU) ######
    config.gpu_sd35 = 0

    ###### VLM (Doubao API, no_thinking mode + 512px 原图) ######
    config.vlm_backend = "doubao"
    config.vlm_model = "doubao-seed-2-0-pro-260215"
    config.vlm_max_image_size = 512
    config.vlm_disable_thinking = True     # no_thinking → 4.4x 加速
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

    ###### Prompt (1K, spaCy 实时提取) ######
    config.prompt_file = "data/train_prompts/flux_reason_spatial_short_1k.txt"

    ###### Logging ######
    config.log_interval = 10
    config.save_interval = 10

    return config
