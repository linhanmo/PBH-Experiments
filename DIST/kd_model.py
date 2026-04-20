"""
知识蒸馏模型封装器。包含了 TopDownMoEProtoKD 类，用于前向传播截获中间层特征，计算基于热力图和原型特征的蒸馏损失 (MSE)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models.builder import build_posenet, POSENETS
from mmpose.models.detectors.base import BasePose
from mmcv.runner import load_checkpoint
import numpy as np

@POSENETS.register_module()
class TopDownMoEProtoKD(BasePose):
    """
    原型对齐知识蒸馏模型封装 (Prototype-Aligned Knowledge Distillation Wrapper)
    该类将教师模型 (Teacher) 和学生模型 (Student) 封装在一起，在训练过程中提取中间特征并计算蒸馏损失。
    """
    def __init__(
        self,
        student,
        teacher,
        teacher_ckpt=None,
        kd_hm_weight=0.5,
        kd_proto_weight=0.3,
        kd_warmup_iters=2000,
        kd_hm_conf_power=1.5,
        kd_proto_cos_weight=0.5,
        **kwargs
    ):
        super().__init__()
        # 1. 构建学生模型 (剪枝后的模型) 和教师模型 (原始未剪枝模型)
        self.student = build_posenet(student)
        self.teacher = build_posenet(teacher)
        
        # 2. 如果提供了教师模型的权重路径，则加载教师模型权重
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
            
        # 3. 冻结教师模型的所有参数，使其在蒸馏过程中不更新
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # 4. 设置蒸馏损失的权重
        self.kd_hm_weight = kd_hm_weight         # 热力图 (Heatmap) 蒸馏损失权重
        self.kd_proto_weight = kd_proto_weight   # 原型 (Prototype) 蒸馏损失权重
        self.kd_warmup_iters = max(int(kd_warmup_iters), 1)
        self.kd_hm_conf_power = float(kd_hm_conf_power)
        self.kd_proto_cos_weight = float(kd_proto_cos_weight)
        self._num_updates = 0
        
        # 用于存储学生模型的前向传播中间特征
        self.s_hm = None
        self.s_proto_out = None
        
        # 5. 注册钩子 (Hook) 以截获学生模型的热力图输出
        def hm_hook(module, inputs, outputs):
            self.s_hm = outputs
        self.student.keypoint_head.register_forward_hook(hm_hook)
        
        # 6. 注册钩子 (Hook) 以截获学生模型的原型输出 (如果存在原型头)
        def proto_hook(module, inputs, outputs):
            self.s_proto_out = outputs
        if getattr(self.student, 'proto_head', None) is not None:
            self.student.proto_head.register_forward_hook(proto_hook)

    @staticmethod
    def _spatial_standardize(x, eps=1e-6):
        """按每个样本-关节在空间维度做标准化，降低幅值差异带来的蒸馏不稳定。"""
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mean) / std

    @staticmethod
    def _safe_weighted_mean(loss_map, weight_map, eps=1e-6):
        denom = weight_map.sum().clamp_min(eps)
        return (loss_map * weight_map).sum() / denom

    def _kd_scale(self):
        """迭代级 warmup，避免训练初期 KD 过强压制学生监督学习。"""
        self._num_updates += 1
        return min(1.0, float(self._num_updates) / float(self.kd_warmup_iters))

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """训练前向传播，包含标准训练损失和蒸馏损失"""
        # 1. 运行学生模型 forward_train 获取初始损失 (基础的监督损失)
        losses = self.student.forward_train(img, target, target_weight, img_metas, **kwargs)
        
        if target is None:
            return losses
        
        # 2. 运行教师模型 forward，提取教师的特征
        with torch.no_grad():
            self.teacher.eval() # 确保教师模型处于推理模式
            # 获取图像对应的数据集来源，用于 MoE 路由
            img_sources = torch.from_numpy(np.array([ele['dataset_idx'] for ele in img_metas])).to(img.device)
            # 提取主干网络特征
            # 检查教师模型的 backbone 是否需要 img_sources
            if hasattr(self.teacher.backbone, 'forward') and callable(getattr(self.teacher.backbone, 'forward')):
                import inspect
                sig = inspect.signature(self.teacher.backbone.forward)
                if 'img_sources' in sig.parameters or len(sig.parameters) > 1:
                    t_output = self.teacher.backbone(img, img_sources)
                else:
                    t_output = self.teacher.backbone(img)
            else:
                t_output = self.teacher.backbone(img)
                
            if hasattr(self.teacher, 'neck') and self.teacher.with_neck:
                t_output = self.teacher.neck(t_output)
            # 提取教师模型的热力图特征
            t_hm = self.teacher.keypoint_head(t_output)
            t_proto_out = None
            # 提取教师模型的原型特征
            if getattr(self.teacher, 'proto_head', None) is not None:
                try:
                    t_proto_out = self.teacher.proto_head(t_output, [t_hm], img_sources)
                except Exception as e:
                    try:
                        t_proto_out = self.teacher.proto_head(t_output)
                    except:
                        pass
            
        kd_scale = self._kd_scale()

        # 3. 计算蒸馏损失 (KD losses)
        # 3.1 热力图蒸馏：空间标准化 + 置信度加权 SmoothL1，提升稳定性
        if self.s_hm is not None and t_hm is not None:
            with torch.cuda.amp.autocast(enabled=False):
                s_hm = self._spatial_standardize(self.s_hm.float())
                t_hm = self._spatial_standardize(t_hm.float())

                # 教师热图的归一化幅值作为软权重，强化高置信区域蒸馏
                conf = t_hm.abs()
                conf = conf / conf.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                conf = conf.pow(self.kd_hm_conf_power)

                hm_loss_map = F.smooth_l1_loss(s_hm, t_hm, reduction='none')
                hm_loss = self._safe_weighted_mean(hm_loss_map, conf)
                losses['kd_hm_loss'] = (self.kd_hm_weight * kd_scale) * hm_loss
        
        # 3.2 原型蒸馏：L2 归一化 + (MSE + Cosine) 混合目标
        if self.s_proto_out is not None and t_proto_out is not None and self.kd_proto_weight > 0:
            with torch.cuda.amp.autocast(enabled=False):
                s_proto_norm = F.normalize(self.s_proto_out.float(), p=2, dim=1)
                t_proto_norm = F.normalize(t_proto_out.float(), p=2, dim=1)
                proto_mse = F.mse_loss(s_proto_norm, t_proto_norm)
                proto_cos = 1.0 - F.cosine_similarity(
                    s_proto_norm.flatten(1), t_proto_norm.flatten(1), dim=1).mean()
                proto_loss = (1.0 - self.kd_proto_cos_weight) * proto_mse + self.kd_proto_cos_weight * proto_cos
                losses['kd_proto_loss'] = (self.kd_proto_weight * kd_scale) * proto_loss
                
        # 将所有的 float 或 int 类型 loss 转换为 Tensor
        for k, v in losses.items():
            if isinstance(v, (float, int)):
                losses[k] = torch.tensor(v, dtype=torch.float32, device=img.device)
            
        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """测试前向传播，仅使用学生模型"""
        return self.student.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward(self, img, target=None, target_weight=None, img_metas=None, return_loss=True, return_heatmap=False, **kwargs):
        """主前向传播函数，根据 return_loss 决定训练或测试"""
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_dummy(self, img):
        """用于 FLOPs 计算的虚拟前向传播"""
        return self.student.forward_dummy(img)

    def show_result(self, img, result, **kwargs):
        """可视化结果"""
        return self.student.show_result(img, result, **kwargs)
