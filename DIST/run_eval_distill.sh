#!/usr/bin/env bash
# ==========================================
# 评估蒸馏后的模型精度。知识蒸馏训练完成后，调用此脚本在 COCO 2017 数据集上评估最终的精度指标。
# ==========================================
# 评估蒸馏后的模型精度
# 知识蒸馏训练完成后，调用此脚本评估剪枝并在 COCO 2017 数据集上蒸馏后的 PoseBH 模型的最终精度。

set -euo pipefail

# 1. 激活 Conda 虚拟环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PoseBH

# 2. 定义配置文件路径和最终的最佳权重路径
CFG="experiments/DIST/pruned50_distill_coco.py"
# 默认使用 epoch_40.pth 或 best_AP_epoch_*.pth
WEIGHT="experiments/DIST/work_dirs/pruned50_distill_coco/latest.pth"

# 3. 创建日志目录
mkdir -p experiments/DIST/logs

# 4. 执行分布式评估 (使用 1 个 GPU)
# 在 COCO 数据集上计算 AP/AR 指标
echo "开始评估蒸馏后模型精度..."
# 为避免与后台正在运行的蒸馏训练产生 DDP 端口冲突 (Address already in use)，指定一个随机端口 (如 29501)
PORT=29501 bash tools/dist_test.sh "${CFG}" "${WEIGHT}" 1 \
    |& tee experiments/DIST/logs/eval_distill_coco.log

# 5. 总结评估结果并生成 JSON 格式的精度报告
# 使用 summarize_eval.py 提取并对比 baseline 模型的日志和蒸馏后模型的日志，最终生成 accuracy_report_distill_coco.json
python experiments/CUT/summarize_eval.py \
  --base-log experiments/stage1/baseline/logs/06_coco_test.log \
  --pruned-log experiments/DIST/logs/eval_distill_coco.log \
  --out experiments/DIST/accuracy_report_distill_coco.json
