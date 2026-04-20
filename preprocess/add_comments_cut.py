import os
import glob

replacements = {
    "run_prune_pruned50.sh": {
        "#!/usr/bin/env bash": "#!/usr/bin/env bash\n# 执行任务感知分层结构化剪枝并计算剪枝前后模型复杂度",
        "set -euo pipefail": "set -euo pipefail # 如果有任何命令失败或使用了未定义变量，立即退出脚本",
        "mkdir -p experiments/CUT/logs": "# 创建日志目录",
        "Run the pruning script": "# 运行剪枝脚本",
        "It loads the base weights, applies the structural pruning": "# 该脚本会加载基线权重，应用结构化剪枝",
        "and saves the initialized pruned weights": "# 并保存初始化好的剪枝后权重",
        "Compute FLOPs and Params": "# 计算并对比剪枝前后的计算量 (FLOPs) 和参数量 (Params)"
    },
    "run_finetune_pruned50_coco.sh": {
        "#!/usr/bin/env bash": "#!/usr/bin/env bash\n# 在 COCO 2017 数据集上对剪枝后的模型进行微调 (Finetuning)",
        "set -euo pipefail": "set -euo pipefail # 如果有任何命令失败或使用了未定义变量，立即退出脚本",
        "Create log directory": "# 创建日志目录",
        "Run distributed training": "# 运行分布式训练脚本 (使用 1 个 GPU)"
    },
    "run_eval_coco_base_pruned50.sh": {
        "#!/usr/bin/env bash": "#!/usr/bin/env bash\n# 评估基线模型和剪枝微调后模型在 COCO 数据集上的表现，并生成对比报告",
        "Evaluate baseline model": "# 评估基线模型",
        "Evaluate pruned model": "# 评估剪枝微调后的模型",
        "Summarize results": "# 总结评估结果并生成 JSON 格式的精度报告"
    },
    "run_generate_heatmaps.sh": {
        "#!/usr/bin/env bash": "#!/usr/bin/env bash\n# 生成基线模型与剪枝模型预测热力图的对比覆盖图 (Overlay)",
        "Generate heatmaps for a specific image": "# 为特定图像生成预测热力图"
    },
    "run_generate_layerwise_mse_psnr.sh": {
        "#!/usr/bin/env bash": "#!/usr/bin/env bash\n# 生成基线模型与剪枝模型逐层特征的 MSE (均方误差) 和 PSNR (峰值信噪比) 热力图表格",
        "Generate layerwise mse/psnr heatmaps": "# 生成逐层 MSE/PSNR 热力图"
    },
    "run_generate_layerwise_heatmaps.sh": {
        "#!/usr/bin/env bash": "#!/usr/bin/env bash\n# 生成并可视化模型在不同深度的层级特征响应热力图",
        "Generate layerwise heatmaps": "# 生成层级特征热力图"
    },
    "get_flops_pair.py": {
        "A script to compute and compare FLOPs and Params": "一个用于计算并对比基线模型和剪枝模型计算量 (FLOPs) 与参数量 (Params) 的脚本",
        "Get complexity for base model": "获取基线模型的复杂度",
        "Get complexity for pruned model": "获取剪枝模型的复杂度",
        "Save report": "保存对比报告到 JSON 文件"
    },
    "summarize_eval.py": {
        "A script to summarize AP metrics from base and pruned logs": "一个用于从基线模型和剪枝模型的评估日志中提取并总结 AP 指标的脚本",
        "Parse AP from log": "从日志中解析 AP 指标",
        "Extract metric": "使用正则表达式提取指标",
        "Generate report": "生成对比报告"
    }
}

for filename, reps in replacements.items():
    file_path = os.path.join("experiments/CUT", filename)
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for en, zh in reps.items():
        content = content.replace(en, zh)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("CUT shell script comments applied successfully.")
