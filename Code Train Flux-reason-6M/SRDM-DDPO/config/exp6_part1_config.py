"""Exp6 Part 1 配置: r_in + r_SSR DDPO 训练 — SD3.5 + fp32 + Doubao thinking.

与 exp5 的关键区别:
    - 纯内生奖励: r_total = 0.5*r_in + 1.0*r_SSR (无 GT 监督)
    - φ* = 均匀平均 (M 条链的 φ 向量取均值)
    - 无 TrainingAlerter (alerter=None)
    - 20k 提示词, 300 epochs, lr=5e-5
    - save_interval=20, log_interval=10

用法:
    python experiments/exp6/exp6_part1.py --config config/exp6_part1_config.py
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp6_part1"
    config.seed = 42
    config.mixed_precision = "fp32"
    config.work_dir = "experiments/exp6/logs_checkpoints"

    ###### Model Selection (仅 SD3.5) ######
    config.model_ids = ["sd35"]

    config.pretrained_model_paths = {
        "sd35": (
            "../../../hf_cache/models--stabilityai--stable-diffusion-3.5-medium/"
            "snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
        ),
    }

    ###### Reward Mode ######
    config.reward_mode = "r_in_rssr"

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Training ######
    config.num_epochs = 300
    config.num_prompts_per_epoch = 3
    config.num_chains_per_prompt = 6
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
    config.vlm_disable_thinking = False
    config.vlm_max_workers = 6
    config.vlm_stagger_delay = 2.0
    config.vlm_max_retries = 3

    ###### r_SSR 参数 ######
    config.lambda_count = 0.5
    config.lambda_coverage = 0.25
    config.lambda_relation = 0.25
    config.phi_uniform_weights = True

    ###### 奖励权重 ######
    config.r_in_weight = 0.5
    config.r_ssr_weight = 1.0

    ###### PPO ######
    config.ppo_mini_batch_size = 3
    config.ppo_clip_range = 0.2
    config.learning_rate = 5e-5
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 5.0

    ###### LoRA ######
    config.lora_rank = 8
    config.lora_alpha = 16

    ###### Prompt (20k, objects 仅用于 VLM schema 提取) ######
    config.prompt_file = "data/train_prompts/flux_reason_spatial_20k.txt"
    config.prompt_objects_file = "data/train_prompts/flux_reason_spatial_20k_gt.jsonl"

    ###### Logging ######
    config.log_interval = 10
    config.save_interval = 20

    return config
