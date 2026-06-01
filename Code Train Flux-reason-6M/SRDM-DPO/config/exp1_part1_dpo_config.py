"""Exp1 Part1 DPO 配置: r_in only + Pairwise SDE-DPO — SD3.5 + fp32.

与 PPO 的关键区别:
    - DPO loss 替代 PPO (pairwise preference optimization)
    - 仅用 r_in (log_probs_base z-score)，无 VLM，无 r_SSR
    - SD3.5 base 模型从零开始，不加载已有 checkpoint
    - 数据: flux_reason_spatial_20k.txt (20k 条 spatial prompts)
    - WandB project: SRDM-DPO
    - 每 prompt 6 条链 → 按 r_in 排序 → 3 对 (1st,6th), (2nd,5th), (3rd,4th)

用法:
    python experiments/exp1/exp1_part1_dpo.py --config config/exp1_part1_dpo_config.py
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp1_part1_dpo"
    config.seed = 42
    config.mixed_precision = "bf16"  # SD3 native half-precision
    config.work_dir = "experiments/exp1/logs_checkpoints_exp1_part1_dpo"

    ###### Model Selection (仅 SD3.5) ######
    config.model_ids = ["sd35"]

    config.pretrained_model_paths = {
        "sd35": (
            "../../../hf_cache/models--stabilityai--stable-diffusion-3.5-medium/"
            "snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
        ),
    }

    ###### Reward Mode (r_in only) ######
    config.reward_mode = "rin_only"

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Training ######
    config.num_epochs = 1000
    config.num_prompts_per_epoch = 3
    config.num_chains_per_prompt = 6
    config.num_inference_steps = 20
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### GPU Assignment (单 GPU) ######
    config.gpu_sd35 = 0

    ###### DPO 参数 ######
    config.dpo_beta = 1.0  # velocity-loss scale (β·T·Δℓ), single-timestep
    config.dpo_num_pairs = 3  # 6 chains → 3 pairs

    ###### PPO 残留 (仅 lr/grad/clip, 不参与 loss 计算) ######
    config.ppo_mini_batch_size = 2  # not used, kept for compatibility
    config.ppo_clip_range = 0.2    # not used
    config.learning_rate = 1e-5  # lower LR for velocity-matching DPO stability
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 5.0

    ###### LoRA ######
    config.lora_rank = 8
    config.lora_alpha = 16

    ###### Prompt (20k spatial prompts, 仅 .txt) ######
    config.prompt_file = "data/train_prompts/flux_reason_spatial_20k.txt"

    ###### Logging ######
    config.log_interval = 10
    config.save_interval = 20

    return config
