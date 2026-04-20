# 阶段1（Day 1）环境配置与基线对齐日志

日期：2026-04-13

## 完成项

- 已放置 PoseBH-B 官方预训练权重：`/home/lin/PoseBH/weights/posebh/base.pth`
- 已按配置期望的 `./data/...` 目录结构完成软链接对齐（不修改官方 dataloader/config）
- 已补齐 5 个数据集的引用文献信息（BibTeX）
- 已完成基线评测与对齐报告生成（COCO/MPII/OCHuman/COCO-WholeBody/AP-10K）

## 数据集路径对齐

配置期望的数据根目录：`data/`

- COCO 2017 Keypoint：`data/coco`
- MPII：`data/mpii`
- OCHuman：`data/ochuman`
- COCO-WholeBody：复用 `data/coco`，WholeBody 标注 JSON 软链到 `data/coco/annotations/`
- AP-10K：`data/ap10k`

数据集清单与路径映射文件：

- `experiments/stage1/datasets/dataset_manifest.json`

## 数据集引用文献

BibTeX 文件：

- `experiments/stage1/datasets/dataset_citations.bib`

## 评测必需文件补齐

- MPII PCKh 评测必需文件已补齐：
  - `preprocess/mpii/annotations/mpii_gt_val.mat`
- COCO/WholeBody 检测框协议所需文件已存在：
  - `data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json`

## 基线评测与对齐

一键评测脚本：

- `experiments/stage1/baseline/run_baseline_eval.sh`

日志目录：

- `experiments/stage1/baseline/logs/`

对齐报告（含 official / measured / delta）：

- `experiments/stage1/baseline/baseline_alignment_report.csv`
- `experiments/stage1/baseline/baseline_alignment_report.json`

对齐结果摘要（以 CSV 为准）：

- COCO val2017：AP 77.2799（official 77.3，delta -0.0201），AR 82.4355（official 82.4，delta +0.0355）
- MPII val：PCKh 93.1798（official 93.2，delta -0.0202），PCKh@0.1 39.3052（official 39.3，delta +0.0052）
- OCHuman test：AP 83.1433（official 83.1，delta +0.0433），AR 85.0668（official 85.1，delta -0.0332）
- AP-10K test(split1)：AP 75.0152（official 75.0，delta +0.0152），AR 78.2743（official 78.3，delta -0.0257）
- COCO-WholeBody val：AP 57.9036（official 57.9，delta +0.0036），AR 69.5025（official 69.5，delta +0.0025）

官方指标记录文件（PoseBH-B）：

- `experiments/stage1/baseline/official_posebh_b_metrics.json`

## 环境记录

环境信息输出目录：

- `experiments/stage1/env/`

包含：

- `collect_env.json`（MMPose collect_env）
- `pip_freeze_posebh.txt` / `conda_list_posebh.txt`（依赖冻结）
- `nvidia_smi.txt`（GPU 信息）

## 额外修复

- 修复 `tools/model_split.py`：拆分权重时移除 `proto_head.*`、`mlp.experts.*`、`associate_keypoint_heads.*`，避免评测阶段出现大量 unexpected key 输出
- 修复 `summarize_baseline.py`：自动将 AP/AR 从 0.xx 转换为百分比再写入对齐报告
