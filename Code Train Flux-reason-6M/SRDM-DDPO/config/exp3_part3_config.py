"""实验三 Part 3 配置: φ 结构特征计算 + 距离可视化.

读取 Part 2 的 VLM JSON 数据，计算:
  - φ_count / φ_coverage / φ_relation (structure_features)
  - φ* 均匀平均原型
  - 各分量 z-score 归一化 → 加权合并距离
  - λ_count=0.5, λ_coverage=0.25, λ_relation=0.25

Wandb 输出 (per prompt):
  - 每条链的结构 bbox 图 + 特征值文本 (3 variants × 6 chains)
  - 距离散点图: φ* 为中心, 各链距离标记
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = "exp3_part3_phi_features"
    config.seed = 42

    ###### Input ######
    config.input_json = "experiments/exp3/exp3_part2_output/all_results.json"
    config.image_dir = "experiments/exp3/exp3_part2_output"

    ###### phi weights ######
    config.lambda_count = 0.5
    config.lambda_coverage = 0.25
    config.lambda_relation = 0.25

    ###### Visualization ######
    config.visualize = True
    config.bbox_width = 256
    config.panel_width = 320

    ###### Logging ######
    config.wandb_project = "SRDM-DDPO"

    return config
