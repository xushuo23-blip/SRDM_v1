"""Exp1 Part2 DPO 配置: r_in only + Pairwise SDE-DPO — SD3.5 + bf16.

与 Part1 的关键区别:
    - 6 prompts/epoch (vs 3)，每个 prompt 6 条链
    - 每个 prompt 只用 (1st, 6th) 极值对 (vs 3 对)
    - 3 prompts 组成 1 个 batch → 1 次 DPO update → 2 steps/epoch
    - 更稳定的训练信号：只用最极端的好/坏样本

用法:
    python experiments/exp1/exp1_part2_dpo.py --config config/exp1_part2_dpo_config.py
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp1_part2_dpo"
    config.seed = 42
    config.mixed_precision = "bf16"
    config.work_dir = "experiments/exp1/logs_checkpoints_exp1_part2_dpo"

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
    config.num_prompts_per_epoch = 6   # 2 batches x 3 prompts
    config.num_chains_per_prompt = 6
    config.dpo_num_pairs = 1           # only (1st, 6th) per prompt
    config.dpo_batch_size = 3          # 3 prompts → 1 DPO backward
    config.num_inference_steps = 20
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### GPU Assignment (单 GPU) ######
    config.gpu_sd35 = 0

    ###### DPO 参数 ######
    config.dpo_beta = 1.0

    ###### Optimizer ######
    config.learning_rate = 1e-5
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 5.0

    ###### LoRA ######
    config.lora_rank = 8
    config.lora_alpha = 16

    ###### Prompt ######
    config.prompt_file = "data/train_prompts/flux_reason_spatial_20k.txt"

    ###### Logging ######
    config.log_interval = 10
    config.save_interval = 20

    return config
