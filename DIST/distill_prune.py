import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import POSENETS, build_posenet
from mmpose.models.detectors.base import BasePose


def _copy_tensor_slices(dst, src):
    if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
        return None
    if dst.dtype != src.dtype:
        src = src.to(dtype=dst.dtype)
    if dst.device != src.device:
        src = src.to(device=dst.device)
    if dst.dim() != src.dim():
        return None
    slices = tuple(slice(0, min(dst.size(i), src.size(i))) for i in range(dst.dim()))
    out = dst.clone()
    out[slices] = src[slices].clone()
    return out


def _load_state_dict_flexible(module, state_dict):
    new_sd = module.state_dict()
    loaded = {}
    for k, v in new_sd.items():
        if k not in state_dict:
            continue
        src = state_dict[k]
        copied = _copy_tensor_slices(v, src)
        if copied is None:
            continue
        loaded[k] = copied
    module.load_state_dict(loaded, strict=False)


def _select_channels_1d(x, idx):
    if x is None:
        return None
    if x.dim() != 1:
        return None
    if isinstance(idx, torch.Tensor) and idx.device != x.device:
        idx = idx.to(device=x.device)
    return x[idx].clone()


def _select_channels_conv_weight(w, out_idx=None, in_idx=None):
    if w is None or w.dim() != 4:
        return None
    out = w
    if out_idx is not None:
        if isinstance(out_idx, torch.Tensor) and out_idx.device != out.device:
            out_idx = out_idx.to(device=out.device)
        out = out.index_select(0, out_idx)
    if in_idx is not None:
        if isinstance(in_idx, torch.Tensor) and in_idx.device != out.device:
            in_idx = in_idx.to(device=out.device)
        out = out.index_select(1, in_idx)
    return out.clone()


def _parse_int_after(token, key_parts):
    if token not in key_parts:
        return None
    i = key_parts.index(token)
    if i + 1 >= len(key_parts):
        return None
    try:
        return int(key_parts[i + 1])
    except Exception:
        return None


def _build_hrnet_channel_index_map(student_state_dict, new_extra, criterion='bn_gamma'):
    stage3_new = list(new_extra.get('stage3', {}).get('num_channels', []))
    stage4_new = list(new_extra.get('stage4', {}).get('num_channels', []))
    stage3_old = None
    stage4_old = None

    for k, v in student_state_dict.items():
        if not k.startswith('backbone.stage3') or not k.endswith('.bn1.weight'):
            continue
        parts = k.split('.')
        b = _parse_int_after('branches', parts)
        if b is None:
            continue
        if stage3_old is None:
            stage3_old = {}
        if b not in stage3_old:
            stage3_old[b] = int(v.numel())

    for k, v in student_state_dict.items():
        if not k.startswith('backbone.stage4') or not k.endswith('.bn1.weight'):
            continue
        parts = k.split('.')
        b = _parse_int_after('branches', parts)
        if b is None:
            continue
        if stage4_old is None:
            stage4_old = {}
        if b not in stage4_old:
            stage4_old[b] = int(v.numel())

    def _init_scores(old_ch):
        return torch.zeros((old_ch,), dtype=torch.float32)

    scores = {'stage3': {}, 'stage4': {}}
    if stage3_old is not None:
        for b, ch in stage3_old.items():
            scores['stage3'][b] = _init_scores(ch)
    if stage4_old is not None:
        for b, ch in stage4_old.items():
            scores['stage4'][b] = _init_scores(ch)

    use_bn = criterion in ['bn', 'bn_gamma', 'gamma']
    use_l1 = criterion in ['l1', 'conv_l1', 'weight_l1']

    for k, v in student_state_dict.items():
        if not k.startswith('backbone.'):
            continue
        parts = k.split('.')
        stage = None
        if len(parts) > 2 and parts[1] in ['stage3', 'stage4']:
            stage = parts[1]
        if stage is None:
            continue
        b = _parse_int_after('branches', parts)
        if b is None or b not in scores[stage]:
            continue

        if use_bn and (k.endswith('.weight') or k.endswith('.bias')) and '.bn' in k:
            if v.dim() == 1 and v.numel() == scores[stage][b].numel() and k.endswith('.weight'):
                scores[stage][b] += v.detach().abs().float().cpu()
        if use_l1 and k.endswith('.weight') and '.conv' in k and v.dim() == 4:
            if v.size(0) == scores[stage][b].numel():
                scores[stage][b] += v.detach().abs().sum(dim=(1, 2, 3)).float().cpu()

    idx_map = {'stage3': {}, 'stage4': {}}
    for b, sc in scores['stage3'].items():
        keep = int(stage3_new[b]) if b < len(stage3_new) else sc.numel()
        keep = max(1, min(int(sc.numel()), keep))
        top = torch.topk(sc, k=keep, largest=True, sorted=True).indices
        idx_map['stage3'][b] = top.sort().values
    for b, sc in scores['stage4'].items():
        keep = int(stage4_new[b]) if b < len(stage4_new) else sc.numel()
        keep = max(1, min(int(sc.numel()), keep))
        top = torch.topk(sc, k=keep, largest=True, sorted=True).indices
        idx_map['stage4'][b] = top.sort().values

    return idx_map


def _remap_hrnet_pruned_weights(old_sd, new_sd, idx_map):
    def _pick_by_size(size, a, b):
        if a is not None and int(size) == int(a.numel()):
            return a
        if b is not None and int(size) == int(b.numel()):
            return b
        return a

    def _idx_for_stage_branch(stage, b):
        return idx_map.get(stage, {}).get(b, None)

    def _is_bn_param(suffix):
        return suffix in ['weight', 'bias', 'running_mean', 'running_var']

    for k, new_v in new_sd.items():
        if k not in old_sd:
            continue
        old_v = old_sd[k]
        if old_v.shape == new_v.shape:
            new_sd[k] = old_v.clone()
            continue

        if not k.startswith('backbone.'):
            copied = _copy_tensor_slices(new_v, old_v)
            if copied is not None:
                new_sd[k] = copied
            continue

        parts = k.split('.')
        if len(parts) < 2:
            continue

        stage = None
        if parts[1] in ['stage3', 'stage4']:
            stage = parts[1]

        if parts[1] in ['transition2', 'transition3']:
            trans = parts[1]
            b = _parse_int_after(trans, parts)
            if b is None:
                try:
                    b = int(parts[2])
                except Exception:
                    b = None
            if b is None:
                continue
            out_stage = 'stage3' if trans == 'transition2' else 'stage4'
            out_idx = _idx_for_stage_branch(out_stage, b)
            in_idx = None
            if trans == 'transition3':
                stage3_last = max(idx_map.get('stage3', {}).keys()) if idx_map.get('stage3', {}) else None
                if stage3_last is not None:
                    src_b = b if b in idx_map.get('stage3', {}) else stage3_last
                    in_idx = _idx_for_stage_branch('stage3', src_b)

            if k.endswith('.weight') and old_v.dim() == 4 and new_v.dim() == 4:
                mapped = _select_channels_conv_weight(old_v, out_idx=out_idx, in_idx=in_idx)
                if mapped is not None and mapped.shape == new_v.shape:
                    new_sd[k] = mapped
                continue
            if _is_bn_param(parts[-1]) and old_v.dim() == 1 and new_v.dim() == 1 and out_idx is not None:
                mapped = _select_channels_1d(old_v, out_idx)
                if mapped is not None and mapped.shape == new_v.shape:
                    new_sd[k] = mapped
                continue

        b = _parse_int_after('branches', parts)
        if stage is not None and b is not None:
            idx = _idx_for_stage_branch(stage, b)
            if idx is None:
                continue
            if k.endswith('.weight') and old_v.dim() == 4 and new_v.dim() == 4:
                mapped = _select_channels_conv_weight(old_v, out_idx=idx, in_idx=idx)
                if mapped is not None and mapped.shape == new_v.shape:
                    new_sd[k] = mapped
                continue
            if _is_bn_param(parts[-1]) and old_v.dim() == 1 and new_v.dim() == 1:
                mapped = _select_channels_1d(old_v, idx)
                if mapped is not None and mapped.shape == new_v.shape:
                    new_sd[k] = mapped
                continue

        if stage is not None and 'fuse_layers' in parts:
            i = _parse_int_after('fuse_layers', parts)
            if i is None:
                continue
            j = None
            try:
                j = int(parts[parts.index('fuse_layers') + 2])
            except Exception:
                j = None
            if j is None:
                continue
            out_idx_base = _idx_for_stage_branch(stage, i)
            in_idx_base = _idx_for_stage_branch(stage, j)
            if out_idx_base is None or in_idx_base is None:
                continue
            out_idx = _pick_by_size(new_v.size(0) if hasattr(new_v, 'size') else new_v.numel(), out_idx_base, in_idx_base)
            if new_v.dim() == 4:
                in_idx = _pick_by_size(new_v.size(1), in_idx_base, out_idx_base)
            else:
                in_idx = in_idx_base
            if k.endswith('.weight') and old_v.dim() == 4 and new_v.dim() == 4:
                mapped = _select_channels_conv_weight(old_v, out_idx=out_idx, in_idx=in_idx)
                if mapped is not None and mapped.shape == new_v.shape:
                    new_sd[k] = mapped
                continue
            if _is_bn_param(parts[-1]) and old_v.dim() == 1 and new_v.dim() == 1:
                bn_idx = _pick_by_size(new_v.numel(), out_idx_base, in_idx_base)
                mapped = _select_channels_1d(old_v, bn_idx)
                if mapped is not None and mapped.shape == new_v.shape:
                    new_sd[k] = mapped
                continue

        copied = _copy_tensor_slices(new_v, old_v)
        if copied is not None:
            new_sd[k] = copied

    return new_sd

def _spatial_kl(student_hm, teacher_hm, temperature=1.0):
    t = float(temperature)
    s = student_hm / t
    q = teacher_hm / t
    n, k, h, w = s.shape
    s = s.reshape(n, k, h * w)
    q = q.reshape(n, k, h * w)
    log_p = F.log_softmax(s, dim=-1)
    p_t = F.softmax(q, dim=-1)
    kl = F.kl_div(log_p, p_t, reduction='batchmean') * (t * t)
    return kl


def _infer_student_feat(student_backbone_out):
    if isinstance(student_backbone_out, (list, tuple)):
        return student_backbone_out[0]
    return student_backbone_out


def _infer_teacher_proto(teacher_proto_out):
    if isinstance(teacher_proto_out, (list, tuple)):
        return teacher_proto_out[0]
    return teacher_proto_out


@POSENETS.register_module()
class TopDownDistillPrune(BasePose):
    def __init__(
        self,
        teacher,
        student,
        distill_cfg=None,
        prune_cfg=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.fp16_enabled = False

        self.teacher_cfg = copy.deepcopy(teacher)
        self.student_cfg = copy.deepcopy(student)
        self.distill_cfg = {} if distill_cfg is None else copy.deepcopy(distill_cfg)
        self.prune_cfg = {} if prune_cfg is None else copy.deepcopy(prune_cfg)

        self.teacher = build_posenet(self.teacher_cfg)
        self.student = build_posenet(self.student_cfg)

        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self._proto_adaptor = None
        adaptor_out = self.distill_cfg.get('proto_adaptor_out_channels', None)
        if adaptor_out is not None:
            in_ch = self._infer_student_feat_channels()
            self._proto_adaptor = nn.Conv2d(
                in_ch,
                int(adaptor_out),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            nn.init.kaiming_normal_(self._proto_adaptor.weight, mode='fan_out', nonlinearity='relu')

        self._prune_state = dict(step=0)
        self._train_iter = 0

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _infer_student_feat_channels(self):
        extra = (
            self.student_cfg.get('backbone', {})
            .get('extra', {})
            .get('stage4', {})
            .get('num_channels', None)
        )
        if extra is None:
            return 32
        if isinstance(extra, (list, tuple)) and len(extra) > 0:
            return int(extra[0])
        return 32

    def prune_student_backbone_extra(self, new_extra, channel_map=None, importance_criterion='bn_gamma'):
        old_sd = self.student.state_dict()
        new_cfg = copy.deepcopy(self.student_cfg)
        new_cfg['pretrained'] = None
        new_cfg.setdefault('backbone', {})
        new_cfg['backbone']['extra'] = copy.deepcopy(new_extra)
        new_student = build_posenet(new_cfg)
        device = next(self.student.parameters()).device
        new_student = new_student.to(device)
        if channel_map is None:
            channel_map = _build_hrnet_channel_index_map(old_sd, new_extra, criterion=importance_criterion)
        new_sd = new_student.state_dict()
        new_sd = _remap_hrnet_pruned_weights(old_sd, new_sd, channel_map)
        new_student.load_state_dict(new_sd, strict=False)
        self.student = new_student
        self.student_cfg = new_cfg

    def set_prune_state(self, **kwargs):
        self._prune_state.update(kwargs)

    def get_prune_state(self):
        return dict(self._prune_state)

    def forward(
        self,
        img,
        target=None,
        target_weight=None,
        img_metas=None,
        return_loss=True,
        return_heatmap=False,
        **kwargs,
    ):
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas, **kwargs)
        return self.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        self._train_iter += 1
        self.teacher.eval()
        img_sources = None
        if img_metas is not None:
            try:
                img_sources = torch.tensor(
                    [int(m.get('dataset_idx', 0)) for m in img_metas],
                    device=img.device,
                    dtype=torch.long,
                )
            except Exception:
                img_sources = None
        with torch.no_grad():
            try:
                if img_sources is not None:
                    t_feat = self.teacher.backbone(img, img_sources)
                else:
                    t_feat = self.teacher.backbone(img)
            except TypeError:
                t_feat = self.teacher.backbone(img)
            if hasattr(self.teacher, 'neck'):
                t_feat = self.teacher.neck(t_feat)
            t_hm = self.teacher.keypoint_head(t_feat)
            t_proto = None
            if hasattr(self.teacher, 'proto_head'):
                t_proto = self.teacher.proto_head(t_feat)
                t_proto = _infer_teacher_proto(t_proto)

        s_feat = self.student.backbone(img)
        if hasattr(self.student, 'neck'):
            s_feat = self.student.neck(s_feat)
        s_hm = self.student.keypoint_head(s_feat)

        losses = {}
        sup_losses = self.student.keypoint_head.get_loss(s_hm, target, target_weight)
        heatmap_loss_weight = float(self.distill_cfg.get('heatmap_loss_weight', 1.0))
        if 'heatmap_loss' in sup_losses:
            losses['loss_sup'] = sup_losses['heatmap_loss'] * heatmap_loss_weight
        else:
            for k, v in sup_losses.items():
                losses[f'loss_sup_{k}'] = v * heatmap_loss_weight

        hm_w = float(self.distill_cfg.get('kd_hm_weight', self.distill_cfg.get('hm_weight', 0.0)))
        if hm_w > 0:
            t = float(self.distill_cfg.get('temperature', 1.0))
            losses['loss_kd_hm'] = _spatial_kl(s_hm, t_hm, temperature=t) * hm_w

        proto_w = float(self.distill_cfg.get('kd_proto_weight', self.distill_cfg.get('proto_weight', 0.0)))
        if proto_w > 0 and t_proto is not None and self._proto_adaptor is not None:
            s_map = _infer_student_feat(s_feat)
            s_map = self._proto_adaptor(s_map)
            if s_map.shape[-2:] != t_proto.shape[-2:]:
                s_map = F.interpolate(s_map, size=t_proto.shape[-2:], mode='bilinear', align_corners=False)
            losses['loss_kd_proto'] = F.mse_loss(s_map, t_proto) * proto_w

        warmup_iters = int(self.distill_cfg.get('sup_ratio_enforce_iters', 0))
        min_sup_ratio = float(self.distill_cfg.get('min_sup_ratio', 0.0))
        if warmup_iters > 0 and min_sup_ratio > 0 and self._train_iter <= warmup_iters:
            sup = losses.get('loss_sup', None)
            if isinstance(sup, torch.Tensor):
                kd_sum = 0.0
                kd_keys = [k for k in losses.keys() if k.startswith('loss_kd_')]
                for k in kd_keys:
                    v = losses.get(k, None)
                    if isinstance(v, torch.Tensor):
                        kd_sum = kd_sum + v
                if isinstance(kd_sum, torch.Tensor):
                    max_kd = sup * (1.0 - min_sup_ratio) / max(min_sup_ratio, 1.0e-12)
                    scale = (max_kd / (kd_sum + 1.0e-12)).clamp(max=1.0)
                    if kd_keys and float(scale.item()) < 1.0:
                        for k in kd_keys:
                            v = losses.get(k, None)
                            if isinstance(v, torch.Tensor):
                                losses[k] = v * scale
                        losses['sup_ratio'] = sup / (sup + kd_sum * scale + 1.0e-12)

        return losses

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        return self.student.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def show_result(self, **kwargs):
        return self.student.show_result(**kwargs)
