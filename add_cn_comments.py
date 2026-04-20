import os
import re

file_comments = {
    "experiments/DIST/run_eval_distill.sh": "评估蒸馏后的模型精度。知识蒸馏训练完成后，调用此脚本在 COCO 2017 数据集上评估最终的精度指标。",
    "experiments/DIST/run_distill.sh": "启动原型对齐知识蒸馏 (Prototype-Aligned KD) 的主训练脚本。调用 mmpose 分布式训练，将大模型知识蒸馏给剪枝后的小模型。",
    "experiments/DIST/pruned50_distill_coco.py": "知识蒸馏训练配置文件。配置了教师模型 (原始 PoseBH) 和学生模型 (剪枝 PoseBH) 的结构、数据集路径及优化器超参数。",
    "experiments/DIST/kd_model.py": "知识蒸馏模型封装器。包含了 TopDownMoEProtoKD 类，用于前向传播截获中间层特征，计算基于热力图和原型特征的蒸馏损失 (MSE)。",
    "experiments/preprocess/preprocess_coco2017_keypoints.py": "COCO 2017 Keypoints 数据集预处理脚本。处理原始标注文件，准备模型训练/评估所需的格式。",
    "experiments/preprocess/preprocess_ap10k.py": "AP-10K 动物姿态估计数据集预处理脚本。将标注文件转换为通用的评估格式。",
    "experiments/preprocess/preprocess_ochuman.py": "OCHuman 数据集预处理脚本。用于处理遮挡严重的人体姿态估计数据集，以供后续测试评估。",
    "experiments/preprocess/preprocess_mpii.py": "MPII 人体姿态估计数据集预处理脚本。处理 .mat 标注文件，准备 MPII 的 GT 测试集格式。",
    "experiments/preprocess/preprocess_coco_wholebody.py": "COCO-WholeBody 数据集预处理脚本。处理全身关键点标注，生成用于评估面部、手部及全身关键点的标注文件。",
    "experiments/stage1/baseline/summarize_baseline.py": "汇总并统计基线模型的评估结果。读取各数据集的评估日志文件并提取 AP/AR 等关键指标，最终生成汇总 JSON 报告。",
    "experiments/stage1/baseline/run_baseline_eval.sh": "执行基线模型在所有数据集 (COCO, MPII, OCHuman, AP10K, WholeBody) 上的精度评估的主脚本。",
    "experiments/CUT/run_generate_heatmaps.sh": "生成基础模型与剪枝模型最终热力图对比可视化的执行脚本。",
    "experiments/CUT/summarize_eval.py": "汇总剪枝后模型在 COCO 数据集上的评估精度，并与基线模型进行精度对比的脚本。",
    "experiments/CUT/pruned50_coco_finetune.py": "剪枝后模型在 COCO 2017 数据集上进行基础微调 (Finetune) 的配置文件。",
    "experiments/CUT/run_generate_layerwise_heatmaps.sh": "生成逐层激活热力图可视化的执行脚本，用于观察不同架构深度的特征响应情况。",
    "experiments/CUT/generate_heatmaps.py": "读取基线模型和剪枝模型，在单张图像上生成姿态热力图并进行叠加可视化的 Python 脚本。",
    "experiments/CUT/vitmoe_prunable.py": "支持结构化剪枝的 ViT-MoE (Mixture of Experts) 主干网络定义脚本。允许配置可变的深度 (Depth) 和 MLP 隐藏层维度。",
    "experiments/CUT/run_eval_coco_base_pruned50.sh": "在 COCO 验证集上对比评估基线模型与剪枝后模型精度的执行脚本。",
    "experiments/CUT/run_generate_layerwise_mse_psnr.sh": "生成逐层特征均方误差 (MSE) 和峰值信噪比 (PSNR) 统计表格的执行脚本。",
    "experiments/CUT/generate_layerwise_heatmaps.py": "利用 PyTorch Hook 机制提取模型各层特征，并生成各层激活强度 (热力图) 可视化的 Python 脚本。",
    "experiments/CUT/get_flops_pair.py": "计算并对比基线模型和剪枝模型的计算复杂度 (FLOPs) 与参数量 (Params) 的脚本。",
    "experiments/CUT/generate_layerwise_mse_psnr.py": "通过前向传播提取特征图，计算剪枝模型相对基线模型的逐层 MSE 和 PSNR，并绘制彩色表格热力图的脚本。",
    "experiments/CUT/run_finetune_pruned50_coco.sh": "启动剪枝后模型在 COCO 数据集上微调训练的执行脚本。",
    "experiments/CUT/run_prune_pruned50.sh": "执行任务感知分层结构化剪枝 (Task-Aware Hierarchical Structured Pruning) 主流程的 Bash 脚本。",
    "experiments/CUT/vitb_posebh_pruned50.py": "剪枝 50% 后的 PoseBH 基础模型架构的注册与配置文件。",
    "experiments/CUT/vit_prunable.py": "基础的支持剪枝操作的 ViT (Vision Transformer) 主干网络定义脚本。",
    "experiments/CUT/vitpose_base_coco_256x192_taskaware_pruned.py": "任务感知剪枝后的 ViTPose 模型在 COCO 256x192 分辨率下的完整配置文件。",
    "experiments/CUT/task_aware_prune.py": "任务感知剪枝算法的核心实现脚本。执行具体的权重截断：保留底层，中层裁剪 50% MLP 维度，高层整体丢弃，并保存新权重。",
    "experiments/CUT/run_finetune_pruned50.sh": "启动模型剪枝后恢复性微调训练的基础执行脚本。"
}

def prepend_comment(filepath, comment_text):
    if not os.path.exists(filepath):
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Check if comment is already there
    if comment_text in content:
        return
        
    # Format the comment block
    if filepath.endswith('.py'):
        comment_block = f'"""\n{comment_text}\n"""\n'
    elif filepath.endswith('.sh'):
        comment_block = f'# ==========================================\n# {comment_text}\n# ==========================================\n'
    else:
        return

    # Insert comment block at the top, preserving shebang
    if content.startswith('#!'):
        lines = content.split('\n', 1)
        if len(lines) > 1:
            new_content = lines[0] + '\n' + comment_block + lines[1]
        else:
            new_content = lines[0] + '\n' + comment_block
    else:
        new_content = comment_block + content
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

for filepath, comment_text in file_comments.items():
    prepend_comment(filepath, comment_text)

print("All detailed Chinese comments prepended.")
