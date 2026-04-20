import argparse
import os
from datetime import datetime

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import torch
from mmcv import Config

from mmpose.apis import init_pose_model
from mmpose.datasets import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark FPS for a pose model checkpoint.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument('--config', default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup-iters', type=int, default=150)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--use-val-data', action='store_true')
    parser.add_argument('--use-amp', action='store_true', help='Enable automatic mixed precision (AMP) for inference')
    parser.add_argument('--log-dir', default='experiments/fps_logs')
    parser.add_argument('--log-file', default=None)
    return parser.parse_args()


def unwrap_data_container(x):
    if torch.is_tensor(x):
        return x
    return x.data[0] if hasattr(x, 'data') else x


def normalize_img(img, device, use_cuda):
    if torch.is_tensor(img):
        x = img
    elif hasattr(img, 'data'):
        d = img.data
        if isinstance(d, list) and len(d) > 1 and torch.is_tensor(d[0]):
            x = d
        else:
            x = d[0]
    else:
        x = img
    if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
        x = torch.stack(list(x), dim=0)
    if torch.is_tensor(x) and x.dim() == 3:
        x = x.unsqueeze(0)
    if torch.is_tensor(x):
        return x.to(device, non_blocking=use_cuda)
    raise TypeError('Unsupported img format for FPS benchmark.')


def normalize_img_metas(img_metas, batch_size):
    m = unwrap_data_container(img_metas)
    if isinstance(m, dict):
        return [m for _ in range(batch_size)]
    if isinstance(m, (list, tuple)):
        m = list(m)
        while len(m) == 1 and isinstance(m[0], (list, tuple)):
            m = list(m[0])
        if len(m) == 1 and isinstance(m[0], dict) and batch_size > 1:
            return [m[0] for _ in range(batch_size)]
        if len(m) == batch_size:
            return m
        if len(m) > 0 and all(isinstance(x, (list, tuple)) for x in m):
            flat = []
            for x in m:
                flat.extend(list(x))
            m = flat
            if len(m) >= batch_size and all(isinstance(x, dict) for x in m[:batch_size]):
                return m[:batch_size]
    return m


def ensure_img_metas_length(img_metas, batch_size):
    if isinstance(img_metas, dict):
        return [img_metas for _ in range(batch_size)]
    if isinstance(img_metas, list):
        if len(img_metas) == batch_size:
            return img_metas
        if len(img_metas) > 0 and isinstance(img_metas[0], dict):
            if len(img_metas) > batch_size:
                return img_metas[:batch_size]
            return img_metas + [img_metas[0] for _ in range(batch_size - len(img_metas))]
    return img_metas


def safe_basename(p):
    base = os.path.basename(p)
    base = base.replace('.py', '').replace('.pth', '')
    return ''.join(c if (c.isalnum() or c in ['-', '_', '.']) else '_' for c in base)


def infer_config_path_from_checkpoint_path(ckpt_path):
    p = ckpt_path.replace('\\', '/')
    candidates = [
        ('/pruned20_coco_finetune/', 'experiments/CUT/pruned20_coco_finetune.py'),
        ('/pruned50_coco_finetune/', 'experiments/CUT/pruned50_coco_finetune.py'),
        ('/pruned30_coco_finetune/', 'experiments/CUT/pruned30_coco_finetune.py'),
        ('/pruned40_coco_finetune/', 'experiments/CUT/pruned40_coco_finetune.py'),
        ('/pruned50_distill_coco/', 'experiments/DIST/pruned50_distill_coco.py'),
        ('/pruned20_distill_coco/', 'experiments/DIST/pruned20_distill_coco.py'),
    ]
    for key, cfg in candidates:
        if key in p:
            return cfg
    return None


def load_config_from_checkpoint_meta(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    meta = ckpt.get('meta', None) if isinstance(ckpt, dict) else None
    if not isinstance(meta, dict):
        return None
    cfg_text = meta.get('config', None)
    if not isinstance(cfg_text, str) or len(cfg_text.strip()) == 0:
        return None
    return Config.fromstring(cfg_text, file_format='.py')


def patch_cfg_compat(cfg):
    def _rewrite_path(p):
        if not isinstance(p, str) or len(p) == 0:
            return p
        s = p.replace('\\', '/')
        s = s.replace('/home/uyoung/human/ViTPose/data/coco', 'data/coco')
        s = s.replace('/home/uyoung/human/ViTPose/data', 'data')
        return s

    def _patch_dataset_paths(data_cfg):
        if not isinstance(data_cfg, dict):
            return
        if 'ann_file' in data_cfg and isinstance(data_cfg['ann_file'], str):
            data_cfg['ann_file'] = _rewrite_path(data_cfg['ann_file'])
        if 'img_prefix' in data_cfg and isinstance(data_cfg['img_prefix'], str):
            data_cfg['img_prefix'] = _rewrite_path(data_cfg['img_prefix'])
        if 'data_cfg' in data_cfg and isinstance(data_cfg['data_cfg'], dict):
            dc = data_cfg['data_cfg']
            if 'bbox_file' in dc and isinstance(dc['bbox_file'], str):
                dc['bbox_file'] = _rewrite_path(dc['bbox_file'])

    def _patch_multihead_pretrained(model_cfg):
        if not isinstance(model_cfg, dict):
            return
        p = model_cfg.get('multihead_pretrained', None)
        if isinstance(p, str) and len(p) > 0 and not os.path.exists(p):
            model_cfg['multihead_pretrained'] = None

    def _patch_proto_head(model_cfg):
        if not isinstance(model_cfg, dict):
            return
        proto_head = model_cfg.get('proto_head', None)
        if not isinstance(proto_head, dict):
            return
        extra = proto_head.get('extra', None)
        if not isinstance(extra, dict):
            return
        extra.setdefault('css_weight', 0.0)
        extra.setdefault('css_conf_thr', 0.05)
        extra.setdefault('css_update_weight', 0.01)

    if hasattr(cfg, 'model'):
        if isinstance(cfg.model, dict):
            _patch_multihead_pretrained(cfg.model)
        _patch_proto_head(cfg.model)
        if isinstance(cfg.model, dict):
            for k in ['teacher', 'student']:
                if k in cfg.model and isinstance(cfg.model[k], dict):
                    _patch_multihead_pretrained(cfg.model[k])
                    _patch_proto_head(cfg.model[k])

    if hasattr(cfg, 'data') and isinstance(cfg.data, dict):
        for split in ['val', 'test', 'train']:
            if split in cfg.data and isinstance(cfg.data[split], dict):
                _patch_dataset_paths(cfg.data[split])



def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = f'cuda:{args.gpu_id}' if use_cuda else 'cpu'
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)

    cfg_source = None
    cfg_path = None
    if args.config is not None:
        cfg_path = args.config
        cfg = Config.fromfile(cfg_path)
        cfg_source = cfg_path
    else:
        cfg = load_config_from_checkpoint_meta(args.checkpoint)
        if cfg is not None:
            cfg_source = 'checkpoint_meta'
        else:
            cfg_path = infer_config_path_from_checkpoint_path(args.checkpoint)
            if cfg_path is None:
                raise RuntimeError('Cannot infer config. Provide --config explicitly.')
            cfg = Config.fromfile(cfg_path)
            cfg_source = cfg_path

    patch_cfg_compat(cfg)

    cfg.model.pretrained = None
    cfg.data.val.test_mode = True

    model = init_pose_model(cfg, args.checkpoint, device=device)
    model.eval()

    if args.use_val_data:
        dataset = build_dataset(cfg.data.val)
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=0,
            dist=False,
            shuffle=False,
            drop_last=True,
        )
        batch = next(iter(dataloader))
        img = normalize_img(batch.get('img'), device, use_cuda)
        bsz = int(img.shape[0])
        img_metas = ensure_img_metas_length(normalize_img_metas(batch.get('img_metas'), bsz), bsz)
    else:
        img = torch.randn(args.batch_size, 3, 256, 192, device=device)
        img_metas = [
            dict(
                image_file='dummy.jpg',
                center=[96.0, 128.0],
                scale=[0.6, 0.8],
                rotation=0,
                bbox_score=1.0,
                bbox_id=0,
                flip_pairs=[],
                dataset_idx=0,
            )
            for _ in range(args.batch_size)
        ]

    os.makedirs(args.log_dir, exist_ok=True)
    if args.log_file is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        cfg_tag = safe_basename(cfg_path) if cfg_path is not None else safe_basename(cfg_source)
        ckpt_tag = safe_basename(args.checkpoint)
        mode = 'val' if args.use_val_data else 'rand'
        args.log_file = os.path.join(args.log_dir, f'fps_{cfg_tag}__{ckpt_tag}__bs{args.batch_size}__{mode}__{ts}.log')

    starter = torch.cuda.Event(enable_timing=True) if use_cuda else None
    ender = torch.cuda.Event(enable_timing=True) if use_cuda else None

    use_amp = args.use_amp and use_cuda and hasattr(torch.cuda.amp, 'autocast')
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(img, img_metas=img_metas, return_loss=False)
            else:
                _ = model(img, img_metas=img_metas, return_loss=False)
        if use_cuda:
            torch.cuda.synchronize()

        if use_cuda:
            starter.record()
            for _ in range(args.iters):
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = model(img, img_metas=img_metas, return_loss=False)
                else:
                    _ = model(img, img_metas=img_metas, return_loss=False)
            ender.record()
            torch.cuda.synchronize()
            elapsed_s = starter.elapsed_time(ender) / 1000.0
        else:
            t0 = datetime.now()
            for _ in range(args.iters):
                _ = model(img, img_metas=img_metas, return_loss=False)
            t1 = datetime.now()
            elapsed_s = (t1 - t0).total_seconds()

    total_images = args.iters * args.batch_size
    fps = total_images / (elapsed_s + 1e-12)

    lines = [
        '==========================================',
        f'Config src  : {cfg_source}',
        f'Checkpoint  : {args.checkpoint}',
        f'Batch size  : {args.batch_size}',
        f'Warmup iters: {args.warmup_iters}',
        f'Iters       : {args.iters}',
        f'GPU         : {args.gpu_id}',
        f'Device      : {device}',
        f'Mode        : {"val_data" if args.use_val_data else "random"}',
        f'Mixed prec. : {"enabled" if use_amp else "disabled"}',
        f'Elapsed     : {elapsed_s:.3f}s',
        f'FPS         : {fps:.2f}',
        f'Log file    : {args.log_file}',
        '==========================================',
    ]
    text = '\n'.join(lines)
    print(text)
    with open(args.log_file, 'w', encoding='utf-8') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
