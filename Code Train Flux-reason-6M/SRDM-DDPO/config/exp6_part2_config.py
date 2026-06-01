"""Exp6 Part2 配置: r_in + r_SSR V2 DDPO 训练 — SD3.5 + fp32 + Doubao deep-think.

与 Part1 的关键区别:
    - r_SSR 升级为 V2: mode-based φ*, deviation ratio, existence penalty
    - VLM deep-think 模式 (深度思考，更高精度)
    - 新数据: ir4_reasprompt.jsonl (16k 条, dict-format objects)
    - 从 Part1 epoch 300 checkpoint 继续训练
    - 日志目录: logs_checkpoints_exp6_part2

用法:
    python experiments/exp6/exp6_part2.py --config config/exp6_part2_config.py --resume 300
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp6_part2"
    config.seed = 42
    config.mixed_precision = "fp32"
    config.work_dir = "experiments/exp6/logs_checkpoints_exp6_part2"

    ###### Resume from Part1 checkpoint ######
    config.resume_checkpoint_dir = "experiments/exp6/logs_checkpoints/sd35_r_in_rssr"

    ###### Model Selection (仅 SD3.5) ######
    config.model_ids = ["sd35"]

    config.pretrained_model_paths = {
        "sd35": (
            "../../../hf_cache/models--stabilityai--stable-diffusion-3.5-medium/"
            "snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
        ),
    }

    ###### Reward Mode ######
    config.reward_mode = "r_in_rssr_v2"

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Training ######
    config.num_epochs = 600  # 300 (Part1) + 300 (Part2)
    config.num_prompts_per_epoch = 3
    config.num_chains_per_prompt = 6
    config.num_inference_steps = 20
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### GPU Assignment (单 GPU) ######
    config.gpu_sd35 = 0

    ###### VLM (Doubao-pro, deep-think 深度思考) ######
    config.vlm_backend = "doubao"
    config.vlm_model = "doubao-seed-2-0-pro-260215"
    config.vlm_max_image_size = 512
    config.vlm_disable_thinking = False
    config.vlm_max_workers = 6
    config.vlm_stagger_delay = 2.0
    config.vlm_max_retries = 3

    ###### r_SSR V2 参数 ######
    config.lambda_exist = 2.0
    config.lambda_count = 0.5      # 1/2
    config.lambda_relation = 1/3   # 1/3
    config.lambda_coverage = 1/6   # 1/6
    config.r_ssr_temperature = 1.0
    config.phi_uniform_weights = False  # V2: mode-based φ*

    ###### 奖励权重 (r_in : r_SSR = 1 : 2) ######
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

    ###### Prompt (新数据集, objects 仅用于 VLM schema 提取) ######
    config.prompt_file = "data/train_prompts_self_created/ir4_reasprompt.txt"
    config.prompt_objects_file = "data/train_prompts_self_created/ir4_reasprompt.jsonl"

    ###### Logging ######
    config.log_interval = 10
    config.save_interval = 20

    return config
