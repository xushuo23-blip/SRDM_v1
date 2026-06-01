"""实验二 Part3 配置：DDPO 训练 (r_in = tanh 锚定映射).

与 Part2 的区别:
    - Part2: r_in = zscore(total_log_p_base) 组内归一化
    - Part3: r_in = tanh((total_log_p_base - center) / temperature)
            center=300k 锚定 base model, temperature=2000 控制组内区分度

训练流程:
    - 200 epochs, 每 epoch 随机选 2 个 prompt, 各 6 条链 = 12 chains
    - 用 frozen base model 计算 total_log_p_base → tanh 映射到 [-1, 1]
    - PPO: per-step ratio = exp(log_p_new - log_p_old), clip only
    - LoRA fine-tuning (rank=8, alpha=16)
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp2_part3"
    config.seed = 42
    config.mixed_precision = "fp16"
    config.work_dir = "experiments/exp2/logs checkpoints/part3"

    ###### Model Path ######
    config.pretrained_model_path = (
        "../../../hf_cache/models--stabilityai--stable-diffusion-3-medium-diffusers/"
        "snapshots/ea42f8cef0f178587cf766dc8129abd379c90671"
    )

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Training ######
    config.num_epochs = 200
    config.num_prompts_per_epoch = 2
    config.num_chains_per_prompt = 6
    config.num_inference_steps = 30
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### PPO ######
    config.ppo_mini_batch_size = 3
    config.ppo_clip_range = 0.2
    config.r_in_center = 310000.0
    config.r_in_temperature = 2000.0
    config.adv_clip_max = 2.0
    config.learning_rate = 3e-4
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 1.0

    ###### LoRA ######
    config.lora_rank = 8
    config.lora_alpha = 16

    ###### Prompt ######
    config.prompt_file = "data/train_prompts/flux_reason_structured_5k.txt"

    return config
