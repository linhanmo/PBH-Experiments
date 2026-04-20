"""
知识蒸馏训练配置文件（Pruned20）。目标：相对 Stage1 COCO 基线精度损失 <= 2%。
"""
_base_ = [
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/datasets/coco.py'
]
import sys
sys.path.insert(0, './experiments/CUT')
sys.path.insert(0, './experiments/DIST')

custom_imports = dict(
    imports=[
        'experiments.CUT.vitmoe_prunable',
        'experiments.DIST.kd_model',
        'experiments.CUT.custom_hooks'
    ],
    allow_failed_imports=False
)

evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

proto_extra = dict(
    channels=64,
    activation='silu',
    neck_type='res',
    head_kernel=3,
    final_conv_kernel=1,
    num_keypoints=[17, 17, 17, 17, 17],
    dim_k=64,
    enc_depth=0,
    kpt_cond=dict(
        enabled=True,
        fusion_method='sum'
    ),
    phm_loss_weight=3.33e-6,
    num_in_class_proto=3,
    gamma=0.999,
    loss_proto=dict(
        type='PixelPrototypeCELoss',
        cfg=dict(
            weight=1.25e-5,
            ppc_weight=0.01,
            ppd_weight=0.001,
            num_joints=17,
            num_in_class_proto=3,
        )),
    css_weight=0.0,
    css_conf_thr=0.05,
    css_update_weight=0.01,
)

teacher_cfg = dict(
    type='TopDownMoEProto',
    pretrained=None,
    backbone=dict(
        type='ViTMoE',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        num_expert=6,
        part_features=192,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    associate_keypoint_head=[
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))
        for _ in range(4)
    ],
    proto_head=dict(
        type='KptProtoHead',
        in_channels=768,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=proto_extra,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate=False,
        use_udp=True))

mlp_hidden_dims = [3072, 3072, 3072, 3072, 2816, 2816, 2816, 2816, 2496, 2496, 2496, 2496]

student_cfg = dict(
    type='TopDownMoEProto',
    pretrained=None,
    multihead_pretrained='experiments/CUT/work_dirs/pruned20_coco_finetune/best_AP_epoch_10.pth',
    backbone=dict(
        type='ViTMoEPrunable',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        num_expert=6,
        part_features=192,
        mlp_hidden_dims=mlp_hidden_dims,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    proto_head=dict(
        type='KptProtoHead',
        in_channels=768,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=proto_extra,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type='GaussianHeatmap',
        modulate=False,
        use_udp=True))

model = dict(
    type='TopDownMoEProtoKD',
    student=student_cfg,
    teacher=teacher_cfg,
    teacher_ckpt='weights/posebh/base.pth',
    kd_hm_weight=0.5,
    kd_proto_weight=0.3,
    kd_warmup_iters=2000,
    kd_hm_conf_power=1.5,
    kd_proto_cos_weight=0.5,
)

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    num_joints_half_body=8,
    prob_half_body=0.3,
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
    dataset_idx=0,
    inference_channel=channel_cfg['inference_channel']
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2, encoding='UDP', target_type='GaussianHeatmap'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'dataset_idx'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'dataset_idx'
        ]),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    train=dict(
        type='TopDownCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_train2017.json',
        img_prefix='data/coco/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

work_dir = 'experiments/DIST/work_dirs/pruned20_distill_coco'
fp16 = dict(loss_scale=512.)

checkpoint_config = dict(interval=1)
custom_hooks = [
    dict(type='KeepMultiplesOfTenCheckpointHook', keep_interval=10),
    dict(type='FPSBenchmarkHook', interval=10, num_warmup=2, num_iters=10, log_key='fps_val'),
    dict(type='EarlyStopByAPHook', baseline_ap=0.7728, target_loss_pct=2.0, min_delta=0.001, patience=2, interval=10, monitor='AP')
]

