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
  - 热图蒸馏：KL（空间 softmax，对齐热图分布）
  - 原型蒸馏：MSE（教师 proto 与学生特征经 1×1 对齐层后的 proto 对齐）
- **剪枝策略**：
  - 仅剪中高层：`prune_stages=(3,4)`，Stage1/2 完全不剪
  - 结构化通道剪枝：基于重要性（默认 BN gamma，可切换 conv L1）选择通道并做一致性裁剪（transition / fuse_layers）
  - 触发时机：从 epoch30 开始，每 10 epoch 一次；剪枝点执行 `val(pre-prune) -> prune -> val(post-prune)`
- **目标**：最终剪枝率 40%~50%，COCO val2017 关键点检测 AP ≥ AP_base * 0.98（示例 0.7573）

#### 当前实现状态（2026-04）

本阶段已实现并验证跑通的模块：

- **可复现训练入口**：
  - 单卡训练入口：`experiments/DIST/run_distill_prune_hrnet_w32_coco.sh`
  - 配置：`experiments/DIST/hrnet_w32_distill_prune_coco_256x192.py`
  - 训练过程中支持从 `epoch_*.pth` / `prune_step_*.pth` 断点续训（见下文“断点续训”）
- **教师模型（PoseBH-B / ViT-MoE）加载约束**：
  - 17 关键点时使用 strict=True 加载（避免 silent mismatch）
  - 教师权重绝不手动改 shape（避免破坏教师泛化）；若 keypoints 数不一致，应该在学生侧/蒸馏侧做适配层而不是改教师
  - 对于 ViT-MoE 教师，forward 需要携带 `dataset_source`（MoE gating indices），否则会出现 `NoneType has no attribute view` 的崩溃
- **学生模型（HRNet-W32）初始化**：
  - 使用 OpenMMLab 的 COCO top-down checkpoint：`weights/hrnet/hrnet_w32_coco_256x192.pth`
- **蒸馏损失与训练稳定性策略**：
  - 监督热图损失仍为主导，KD 仅做弱辅助
  - 训练初期前 `sup_ratio_enforce_iters` 步强制 `loss_sup/total >= min_sup_ratio`，避免 KD 压制监督收敛
- **剪枝实现与关键约束**：
  - 仅剪 HRNet 的 stage3/stage4 通道数（stage1/stage2 完全不剪）
  - 重要性准则支持：
    - `bn_gamma`：对 BN gamma 的 |γ| 聚合
    - `conv_l1`：对 conv weight 的输出通道 L1 聚合
  - 一致性裁剪覆盖 transition / fuse_layers，避免跨分支融合产生 shape 不一致
  - 通道数对齐采用 `round_to`（可配置，推荐 32 以平滑剪枝；64 容易产生大步长跳变）
  - 剪枝比例 schedule 支持 `prune_ramp_steps`，用于控制达到 mid_final/high_final 的速度
- **剪枝点验证与保护**：
  - 剪枝点执行顺序：`val(pre-prune) -> prune -> val(post-prune)`
  - 日志会额外打印：
    - `Pre-prune eval epoch ...: AP=...`
    - `Prune step ... stage3 (...) -> (...), stage4 (...) -> (...)`
    - `Post-prune eval epoch ...: AP=...`
  - 掉点保护：剪枝点若 AP 下跌超过阈值会触发停止（用于快速回滚与调参）
- **检查点与日志策略**：
  - 常规 checkpoint：`epoch_*.pth`
  - AP best checkpoint：`best_AP_epoch_*.pth`
  - 剪枝专用 checkpoint：`prune_step_{k}_epoch_{e}.pth`（剪枝完成后立即保存，便于回滚）
  - 总日志：`experiments/DIST/work_dirs/DIST_all.log`（聚合所有运行日志 + 未捕获异常堆栈；非 resume 时覆写，resume 时追加）

#### 断点续训（Resume）

- **推荐续训方式**：从 `epoch_*.pth` 续训，保留 optimizer state（Adam 动量/方差缓存），曲线更稳定
- **从剪枝点续训**：
  - `prune_step_*.pth` 也可 resume，但该文件可能不包含完整 optimizer state（视保存时机而定），更接近“从该点重新开始优化器”
- **结构恢复关键点**：
  - 由于剪枝会改变 HRNet 结构（重建 backbone extra），resume 时会从 checkpoint meta 中恢复 `student_backbone_extra`，避免 shape mismatch

#### 常见问题（Troubleshooting）

- **Epoch30/40 没剪到通道**：
  - 通常是因为剪枝比例太小且 `round_to` 量化门槛导致通道数四舍五入回原值
  - 解决方式：增大 `prune_ramp_steps`（让剪枝更快发生）或把 `round_to=64` 放宽为 `round_to=32`（更细粒度更平滑）
- **剪枝后 AP 出现断崖式下跌**：
  - 常见原因：通道对齐步长过大（例如 256->192 一次跳变太大）或早期剪枝过激
  - 解决方式：放宽 `round_to`（32）、增大 `prune_ramp_steps`、降低 high_final/mid_final 或提高蒸馏弱辅助权重
- **DDP 崩溃（reduction / unused params）**：
  - 剪枝阶段会重建 student 并替换参数集合，DDP reducer 不支持训练中途参数集合变化
  - 解决方式：剪枝训练使用单进程（本阶段默认脚本已切换为单卡非 DDP）

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
bash experiments/DIST/run_distill_prune_hrnet_w32_coco.sh
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

- 启动脚本：`experiments/DIST/run_distill_prune_hrnet_w32_coco.sh`
  - 默认使用单卡非 DDP（剪枝阶段会重建 student 结构，DDP 在训练中途变更参数集合会导致 reducer 报错）
- 训练配置：`experiments/DIST/hrnet_w32_distill_prune_coco_256x192.py`
  - 学生初始化权重：`weights/hrnet/hrnet_w32_coco_256x192.pth`
  - 教师加载：`weights/posebh/base.pth`（17 关键点时 strict=True 加载；教师 proto head 结构按多数据集总 keypoints 构建以匹配权重）
- 核心实现：
  - 蒸馏 + 剪枝封装模型：`experiments/DIST/distill_prune.py`
  - 自定义 hooks（剪枝/掉点保护/早停）：`experiments/DIST/custom_hooks.py`
- 产物目录：`experiments/DIST/work_dirs/hrnet_w32_distill_prune_coco_256x192/`
  - `epoch_*.pth`：按 runtime 的 checkpoint interval 保存（包含 optimizer state）
  - `best_AP_epoch_*.pth`：按 AP 保存 best
  - `prune_step_{k}_epoch_{e}.pth`：每次剪枝后立即额外保存的专用 checkpoint
- 日志：
  - 单次运行日志：`experiments/DIST/work_dirs/hrnet_w32_distill_prune_coco_256x192/*.log`
  - 总日志（汇总 + 异常堆栈）：`experiments/DIST/work_dirs/DIST_all.log`

**技术细节与为什么这么做**：

- 蒸馏损失与权重（可在 config 中调整）：
  - 监督（学生对 GT 热图）：`heatmap_loss_weight`
  - 蒸馏（热图 KL）：`kd_hm_weight`
  - 蒸馏（proto MSE）：`kd_proto_weight`
  - 训练初期为避免 KD 压制监督，前 `sup_ratio_enforce_iters` 步强制 `loss_sup/total >= min_sup_ratio`
- 剪枝与评估：
  - epoch30 起每 10 轮剪一次，并在剪枝点先跑一次 `val(pre-prune)`，再剪枝并立刻跑 `val(post-prune)` 用于掉点监控
  - 当单次剪枝导致 AP 下跌超过阈值（默认 0.3%）时会触发停止/保护（便于快速回滚与调参）
- **早停机制**：
  - 每 10 轮验证一次，若连续 2 次验证 AP 提升 < 0.001 则早停

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
