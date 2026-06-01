"""实验四 Part 1 配置：DDPO 训练 (r_SSR alone 结构奖励).

训练流程:
    - 小实验 100 epochs (2400 samples) / 大实验 500--1000 epochs
    - 每 epoch 随机选 3 个 prompt, 各 8 条链 = 24 samples
    - PPO mini-batch 按 prompt 分组, batch_size=4, 每 prompt 2 步, 共 6 步/epoch
    - 每 GPU 一个完整模型: GPU 0 = SD3, GPU 1 = SD3.5 (并行采样+训练)
    - VLM: Doubao API, no_thinking 模式 (4.4x 加速, 98% 一致性)
    - 奖励模式: r_SSR alone 或 r_SSR + r_in combined
    - PPO: standard min-clip + gradient norm clip + TrainingAlerter
    - 双模型对比: SD3 + SD3.5 同时训练, 同 prompt, WandB 同图叠加曲线

用法:
    python experiments/exp4/exp4_part1.py --config config/exp4_part1_config.py

对比维度:
    - 模型: SD3 Medium vs SD3.5 Medium
    - 奖励: r_SSR alone vs r_SSR + r_in combined
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp4_part1"
    config.seed = 42
    config.mixed_precision = "fp16"
    config.work_dir = "experiments/exp4/logs_checkpoints"

    ###### Model Selection ######
    # 对比模式: 同时加载 SD3 + SD3.5, 同 prompt 训练, WandB 叠加曲线
    # 单模型调试: 只保留一个即可 (e.g. ["sd3"])
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
    # "r_ssr":     纯结构奖励 (VLM φ 距离 → r_SSR)
    # "r_ssr_rin": r_SSR + r_in 组合 (r_in 作为辅助信号)
    config.reward_mode = "r_ssr"

    ###### SDE Sampler ######
    config.a = 0.7

    ###### Training ######
    # 小实验: 100--200 epochs (2400--4800 samples)
    # 正式训练: 500+ epochs (12000+ samples), 具体看速度
    config.num_epochs = 100
    config.num_prompts_per_epoch = 3
    config.num_chains_per_prompt = 8
    config.num_inference_steps = 30
    config.guidance_scale = 5.0
    config.height = 512
    config.width = 512

    ###### Per-Model GPU Assignment (一个 GPU 跑一个完整模型, 并行) ######
    config.gpu_sd3 = 0
    config.gpu_sd35 = 1

    ###### VLM (Doubao API, no_thinking mode) ######
    config.vlm_backend = "doubao"
    config.vlm_model = "doubao-seed-2-0-pro-260215"
    config.vlm_max_image_size = 512
    config.vlm_disable_thinking = True     # no_thinking → 4.4x 加速
    config.vlm_max_workers = 6             # 并行 HTTP 调用
    config.vlm_stagger_delay = 2.0         # 避免 429
    config.vlm_max_retries = 3

    ###### r_SSR 参数 ######
    config.lambda_count = 0.5
    config.lambda_coverage = 0.25
    config.lambda_relation = 0.25
    config.phi_uniform_weights = True       # φ* = uniform 1/M (不按 r_in 加权)
    config.top_k = 2

    ###### r_in 权重 (仅 reward_mode == "r_ssr_rin" 时生效) ######
    config.alpha_ssr = 0.7                  # r_SSR 权重
    config.alpha_in = 0.3                   # r_in 权重

    ###### PPO ######
    config.ppo_mini_batch_size = 4          # 8 chains / 4 = 2 updates per prompt; 3×2=6 updates/epoch
    config.ppo_clip_range = 0.3             # ratio ∈ [0.7, 1.3]; fp16 放宽以降低误杀
    config.learning_rate = 3e-5
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    config.max_grad_norm = 5.0

    ###### TrainingAlerter ######
    config.alert_window = 10
    config.alert_threshold = 3
    config.alert_ratio_bad_pct = 1.0           # fp16 下 ratio 报警无效, 关闭; 仅保留 grad 报警

    ###### LoRA ######
    config.lora_rank = 8
    config.lora_alpha = 16

    ###### Prompt ######
    config.prompt_file = "data/train_prompts/flux_reason_spatial_short_1k.txt"

    ###### Logging ######
    config.log_interval = 10                # 每 N epoch 打印 + 采样图像
    config.save_interval = 10               # 每 N epoch 保存 checkpoint

    return config
