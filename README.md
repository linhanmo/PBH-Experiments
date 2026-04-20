# PoseBH Experiments (PBH-Experiments)

本项目包含了针对 **PoseBH** 模型进行的多阶段压缩与优化实验代码、配置、日志及可视化图表。主要目的是通过结构化剪枝和原型对齐知识蒸馏，在保持较高精度的前提下，大幅降低人体姿态估计模型的计算复杂度和参数量。

## 目录

- [系统配置](#系统配置)
- [实验阶段规划](#实验阶段规划)
  - [Stage 1: 基线对齐与环境准备](#stage-1-基线对齐与环境准备-stage1)
  - [Stage 2: 任务感知分层结构化剪枝探索](#stage-2-任务感知分层结构化剪枝探索-cut---vit-剪枝鲁棒性验证)
  - [Stage 3: 跨架构 CNN 剪枝+蒸馏并行优化](#stage-3-跨架构-cnn-剪枝蒸馏并行优化-dist)
- [评价口径与达标标准](#评价口径与达标标准务必先统一)
- [目录结构说明](#目录结构说明)
- [使用方法](#使用方法)
- [详细实验说明](#详细实验说明)
  - [Stage2 ViT 剪枝实验细节](#stage2-vit-剪枝实验细节)
  - [Stage3 跨架构 CNN 剪枝+蒸馏并行优化](#stage3-跨架构-cnn-剪枝蒸馏并行优化-1)
- [FPS 基准测试](#fps-基准测试指定-config--权重)

***

## 系统配置

- **操作系统**：Ubuntu 22.04
- **CPU**：Intel i9-14900hx
- **GPU**：NVIDIA RTX 3090 24G
- **Python**：3.8
- **CUDA**：11.3

***

## 实验阶段规划

### Stage 1: 基线对齐与环境准备 (`stage1/`)

- 准备包括 COCO, MPII, OCHuman, AP-10K, COCO-WholeBody 在内的多个人体/动物姿态估计数据集。
- 使用官方提供的 PoseBH-B 预训练权重进行评估，完成全数据集的精度基线 (Baseline) 对齐测试。
- 提取并清理模型权重，解决旧版本产生的键名不匹配问题。

### Stage 2: 任务感知分层结构化剪枝探索 (`CUT/`) - ViT 剪枝鲁棒性验证

为了实现人体姿态估计模型的轻量化，我们首先针对 CVPR 2025 SOTA 模型 PoseBH（ViT-MoE 架构）开展了任务感知分层结构化剪枝探索。实验设定**双目标硬约束**：**相对基线精度损失≤2%**，同时**压缩优先路线参数量下降≥50%**。

- 实现了针对 ViT-MoE (Mixture of Experts) 架构的结构化剪枝算法。
- **剪枝策略**：基于任务感知的分层结构化剪枝，保留底层特征提取能力，对中层和高层进行不同程度的 MLP Hidden Dim 裁剪。
- 在 COCO 数据集上对剪枝模型进行了基础微调 (Finetuning)，验证其精度损失。

#### 剪枝实验结果与分析

| 剪枝率          | 参数量下降  | 中层剪枝比例 | 高层剪枝比例 | 最佳验证 AP | 相对基线精度损失 | FPS (images/s, batch=1) |
| ------------ | ------ | ------ | ------ | ------- | -------- | ----------------------- |
| **Baseline** | 0%     | 0%     | 0%     | 77.28%  | 0%       | 35.31                   |
| **Pruned20** | 22.45% | 10%    | 20%    | 76.88%  | 0.52%    | 64.43                   |
| **Pruned30** | 25.84% | 15%    | 30%    | 76.66%  | 0.80%    | 65.63                   |
| **Pruned40** | 29.65% | 20%    | 40%    | 44.21%  | 33.07%   | 67.01                   |

#### 核心实验结论与分析

实验结果表明，PoseBH 的 ViT-MoE 架构对结构化剪枝的鲁棒性存在明显上限：

1. **剪枝率 ≤ 30% 时**：模型仅出现轻微精度损失（Pruned30 剪枝率 25.81%，AP 76.66%，相对基线损失 0.8%），满足精度约束，但最高仅实现 25.81% 的参数量压缩，远达不到 ≥50% 的压缩目标。
2. **剪枝率提升至 40% 时**：模型出现**断崖式精度崩塌**，AP 从 76.66% 骤降至 44.21%，相对基线损失达 33.07%，完全丧失人体姿态拓扑建模能力。我们尝试通过延长微调周期、调整学习率策略恢复精度，但最终 AP 仍无法突破 50%，证明该剪枝率下模型的核心特征表达能力已被不可逆地破坏。

#### 本质原因分析

这一现象的本质原因在于：PoseBH 的高层 Transformer 层与 MoE 专家模块是建模人体关节全局拓扑依赖、关键点原型特征的核心，其特征冗余呈现**集中式分布**，而非 CNN 架构的分布式冗余；当剪枝率超过 30%，MLP 隐藏维度的大幅裁剪会直接破坏全局注意力的特征空间与 MoE 专家的表达能力，导致人体原型建模完全失效，最终出现非线形的精度崩塌。

#### 讨论：ViT 与 CNN 的剪枝鲁棒性差异

本文的剪枝实验揭示了 ViT 与 CNN 架构在人体姿态估计任务中截然不同的剪枝鲁棒性：ViT 架构的特征冗余集中于少数注意力头与 MLP 层，且高层模块直接决定人体全局拓扑的建模能力，高剪枝率下极易出现精度断崖；而 CNN 架构的特征冗余呈现分布式分布，即使 70% 以上的通道被裁剪，剩余卷积核仍能有效提取关节局部特征，仅出现线性的精度下降。

这一结论也解释了为什么工业界姿态估计落地仍以 CNN 骨干为主：在边缘部署场景的高压缩率要求下，CNN 架构的容错空间与稳定性远优于 ViT 架构。

#### 最终决策

基于上述实验结论，ViT 骨干的同架构剪枝无法同时满足我们设定的精度与压缩率双目标，因此我们提出基于 PoseBH SOTA 教师的**跨架构原型对齐蒸馏方案**：以全局建模能力优异的 PoseBH 为教师，以剪枝鲁棒性强、部署生态完善的 HRNet 为轻量化学生骨干，既保留 ViT 的全局原型建模能力，又充分发挥 CNN 的高压缩潜力与推理效率。

#### 计算逻辑说明

- 剪枝率基于整个模型的参数量减少比例计算
- 采用分层剪枝策略：底层（0-3层）不剪枝，中层（4-7层）和高层（8-11层）应用不同剪枝比例
- 所有剪枝脚本使用相同的计算逻辑，仅参数设置不同
- 剪枝过程中会将 MLP Hidden Dim 四舍五入到64的倍数以保持计算效率

#### 剪枝参数设置

- **Pruned20**：`--prune-mid 0.1 --prune-late 0.2`
- **Pruned30**：`--prune-mid 0.15 --prune-late 0.3`
- **Pruned40**：`--prune-mid 0.2 --prune-late 0.4`

#### 产物

- 剪枝权重：`experiments/CUT/weights/pruned{20,30,40}_base_init.pth`
- 剪枝日志：`experiments/CUT/logs/prune_pruned{20,30,40}.log`
- 微调权重：`experiments/CUT/work_dirs/pruned{20,30,40}_coco_finetune/best_AP_epoch_*.pth`

### Stage 3: 跨架构 CNN 剪枝+蒸馏并行优化 (`DIST/`)

基于 Stage2 的实验结论，ViT 骨干的同架构剪枝无法同时满足精度与压缩率双目标，因此我们采用**直接引入 CNN 骨干、剪枝+蒸馏并行优化**的方案：

- **教师模型**：使用全局建模能力优异的 PoseBH（ViT-MoE）作为教师
- **学生模型**：选择剪枝鲁棒性强、部署生态完善的 HRNet 作为轻量化 CNN 骨干
- **核心策略**：
  - 不再使用 ViT 剪枝权重作为学生初始值
  - 对 HRNet 骨干同时进行结构化剪枝和原型对齐蒸馏
  - 剪枝与蒸馏并行进行，实现端到端的轻量化
- **知识蒸馏机制**：
  - 使用 PyTorch Hook 机制截取教师与学生的中间特征
  - 通过特征热力图 MSE 和原型 (Prototype) 输出 MSE 进行联合对齐
- **目标**：同时满足「相对基线精度损失≤2%」和「参数量下降≥50%」双目标

***

## 评价口径与达标标准（务必先统一）

本项目所有“是否达标”的判断，都基于 **Stage 1 的 COCO val2017 基线 mAP(AP)**。

- 记基线为 `AP_base`（示例：`AP_base=0.7728`）
- 论文/交付口径：**核心精度损失 ≤ 2%（相对基线）**
- 等价为：`AP_target = AP_base * 0.98`
  - 示例：`0.7728 * 0.98 = 0.7573`

### 为什么必须以 Stage1 基线作为参照

- Stage2 的剪枝后微调模型（student）本身不是最终基准，它仅是 Stage3 蒸馏的起点；
- 以 Stage2 作为参照容易出现“蒸馏相对 student 有提升，但最终仍远低于基线”的误判；
- 最终交付的硬目标必须对齐 Stage1 的 baseline。

***

## 目录结构说明

```
experiments/
├── .gitignore          # 过滤权重、数据集等大文件
├── preprocess/         # 各个数据集标注格式的预处理转换脚本
├── stage1/             # 第一阶段：基线评估、环境信息收集与模型权重拆分
├── CUT/                # 第二阶段：分层结构化剪枝、微调配置及可视化脚本
├── DIST/               # 第三阶段：基于特征与原型对齐的知识蒸馏代码与配置
└── docs/               # 实验进展日志 (Markdown 格式)
```

***

## 使用方法

所有的训练、评估和可视化脚本均使用 Bash 编写并支持相对路径，直接在 `experiments` 根目录下或项目根目录调用对应模块的 `.sh` 脚本即可。例如：

```bash
# 运行基线评估
bash experiments/stage1/baseline/run_baseline_eval.sh

# Pruned20：剪枝 -> 微调
bash experiments/CUT/run_prune_pruned20.sh
bash experiments/CUT/run_finetune_pruned20_coco.sh

# 可选：探索压缩-精度折中（Pruned30 / Pruned40），用于探测性能极限与消融对比
bash experiments/CUT/run_prune_pruned30.sh
bash experiments/CUT/run_finetune_pruned30_coco.sh

bash experiments/CUT/run_prune_pruned40.sh
bash experiments/CUT/run_finetune_pruned40_coco.sh

# Stage3：跨架构 CNN 蒸馏
bash experiments/DIST/run_distill.sh
```

> **注意**：脚本中默认激活名为 `PoseBH` 的 Conda 虚拟环境。

***

## 详细实验说明

### Stage2 ViT 剪枝实验细节

#### Pruned20 / Pruned30 / Pruned40 的探索目的

Stage2 的主要目的是**验证 ViT-MoE 架构的剪枝鲁棒性上限**，而不是为 Stage3 提供学生模型。通过系统性的剪枝探索，我们得出了关键结论：ViT 骨干的同架构剪枝无法同时满足精度与压缩率双目标。

#### 1）剪枝：生成 `pruned{20,30,40}_base_init.pth`

- 脚本：`experiments/CUT/run_prune_pruned{20,30,40}.sh`
- 产物：
  - 权重：`experiments/CUT/weights/pruned{20,30,40}_base_init.pth`
  - 过程日志：`experiments/CUT/logs/prune_pruned{20,30,40}.log`

**技术细节**：

- 剪枝采用对 ViT Backbone 的 MLP Hidden Dim 进行结构化裁剪（并同步裁剪 MoE Experts 相关权重形状），以保持网络形状一致性；
- 日志中可能出现 `unexpected key in source state_dict: proto_head.* / associate_keypoint_heads.*`，这是因为源 checkpoint 包含更多分支参数，而配置仅构建 backbone+keypoint head；属于 strict=False 的正常提示，不影响剪枝权重生成。

#### 2）微调：验证剪枝后的精度损失

- 脚本：`experiments/CUT/run_finetune_pruned{20,30,40}_coco.sh`
- 产物目录：`experiments/CUT/work_dirs/pruned{20,30,40}_coco_finetune/`
  - `best_AP_epoch_*.pth`：用于验证剪枝后的最佳精度
  - `latest.pth`：用于断点续训
  - `*.log` / `*.log.json`：曲线绘图

**技术细节**：

- 微调阶段的目标是**验证剪枝后的精度损失**，而非为 Stage3 准备学生模型；
- RTX 3090 上推荐提高 batch size 并按线性缩放法则调学习率（例如 `samples_per_gpu=64`、`lr=2e-5`）；
- 微调支持断点续训：
  - 从头训练：`bash experiments/CUT/run_finetune_pruned{20,30,40}_coco.sh`
  - 断点续训：`RESUME=1 bash experiments/CUT/run_finetune_pruned{20,30,40}_coco.sh`
- **早停机制**：
  - 配置了 `EarlyStopByAPHook` 钩子，当模型性能不再提升时自动停止训练
  - 早停条件：如果连续 2 个验证周期（每 10 轮验证一次）AP 提升不超过 0.001，则停止训练
  - 这就是为什么训练在第 30 轮就停止的原因：在第 20 轮和第 30 轮的验证中，AP 没有显著提升，触发了早停机制
  - 早停机制可以避免过拟合，节省计算资源，同时确保模型达到稳定状态

### Stage3 跨架构 CNN 剪枝+蒸馏并行优化

#### 方案设计

基于 Stage2 的实验结论，我们转向**跨架构 CNN 轻量化方案**：

- **教师模型**：PoseBH-B（ViT-MoE），提供全局原型建模能力
- **学生模型**：HRNet（CNN），具备强剪枝鲁棒性和部署优势
- **核心策略**：
  - 不再使用 ViT 剪枝权重作为学生初始值
  - 对 HRNet 骨干同时进行结构化剪枝和原型对齐蒸馏
  - 剪枝与蒸馏并行优化，实现端到端的轻量化

#### 技术细节

- 脚本：`experiments/DIST/run_distill.sh`
- 配置：`experiments/DIST/hrnet_distill_coco.py`
- 产物目录：`experiments/DIST/work_dirs/hrnet_distill_coco/`
- 日志：
  - 全量：`experiments/DIST/logs/distill_hrnet_coco.log`
  - 摘要：`experiments/DIST/logs/distill_hrnet_coco.summary.log`

**技术细节与为什么这么做**：

- 蒸馏损失包含热图对齐（hm）与原型对齐（proto）；为避免训练初期 KD 过强压制监督学习，蒸馏实现了 KD warmup 与更稳定的对齐目标；
- 日志会打印 `BASELINE AP` 与自动计算得到的 `TARGET AP`，并在训练结束后输出“是否达标”；
- **早停机制**：
  - 配置了 `EarlyStopByAPHook` 钩子，当模型性能不再提升时自动停止训练
  - 早停条件：如果连续 2 个验证周期（每 10 轮验证一次）AP 提升不超过 0.001，则停止训练
  - 早停机制可以避免过拟合，节省计算资源，同时确保模型达到稳定状态

#### 达标判定

- 若 `Best AP >= AP_base * 0.98`（示例：`>= 0.7573`）且参数量下降≥50%，则满足双目标硬约束。

***

## FPS 基准测试（指定 config + 权重）

用于在相同硬件（例如 RTX 3090）下，对不同剪枝率/不同权重的推理吞吐进行横向对比。

- 脚本位置：`experiments/benchmark_fps.py`
- 输入：默认仅需 `checkpoint`（会从 checkpoint meta 或路径自动推断 config；必要时可用 `--config` 覆盖）
- 输出：打印 FPS（images/s）及测试参数，并默认保存日志到 `experiments/fps_logs/`

### 标准化测试参数（学术对比）

- **Batch Size**：1
- **输入分辨率**：256×192
- **预热轮数**：150
- **统计迭代次数**：1000
- **计时方式**：CUDA Event 同步计时
- **耗时统计范围**：仅模型前向传播

### 使用方法

使用真实 val 数据（更贴近真实推理链路）：

```bash
/opt/conda/envs/PoseBH/bin/python experiments/benchmark_fps.py \
  experiments/CUT/work_dirs/pruned20_coco_finetune/best_AP_epoch_10.pth \
  --use-val-data
```

使用随机输入（不依赖数据集，纯算子吞吐）：

```bash
/opt/conda/envs/PoseBH/bin/python experiments/benchmark_fps.py \
  experiments/CUT/work_dirs/pruned20_coco_finetune/best_AP_epoch_10.pth
```

如果自动推断失败，可以显式指定 config：

```bash
/opt/conda/envs/PoseBH/bin/python experiments/benchmark_fps.py \
  experiments/CUT/work_dirs/pruned20_coco_finetune/best_AP_epoch_10.pth \
  --config experiments/CUT/pruned20_coco_finetune.py \
  --use-val-data
```

### 日志保存位置

默认输出到：`experiments/fps_logs/`（脚本会打印最终日志文件路径）。也可以自定义：

- `--log-dir <dir>`
- `--log-file <path>`

