#!/bin/bash
# Stage 3: 跨架构 CNN 剪枝+蒸馏并行优化
# 教师模型: PoseBH-B (ViT-MoE)
# 学生模型: HRNet-W32 (CNN)

# 工作目录
cd /root/rivermind-data/PoseBH

# 激活虚拟环境（如果需要）
# source activate PoseBH

# 训练命令
python tools/dist_train.sh \
    experiments/DIST/hrnet_distill_coco.py \
    1 \
    --seed 0

echo "Stage 3 训练完成！"
