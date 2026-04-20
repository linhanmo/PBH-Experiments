"""
Pruned20 剪枝配置文件。
目标：生成 pruned20_base_init.pth
"""
_base_ = [
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/datasets/coco.py'
]
import sys
sys.path.insert(0, './experiments/CUT')

custom_imports = dict(
    imports=['experiments.CUT.vitmoe_prunable'],
    allow_failed_imports=False
)

prune_mode = 'vit_mlp_hidden'
prune_ratio = 0.2
mlp_hidden_dims = [3072, 3072, 3072, 3072, 1408, 1408, 1408, 1408, 1344, 1344, 1344, 1344]

model = dict(
    type='TopDownMoEProto',
    pretrained='../../weights/posebh/base.pth',
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
        mlp_hidden_dims=[768*4]*12,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    proto_head=dict(
        type='KptProtoHead',
        in_channels=768,
        out_channels=17,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            channels=64,
            activation='silu',
            neck_type='res',
            head_kernel=3,
            final_conv_kernel=1,
            num_keypoints=[17, 17, 17, 17, 17],
            dim_k=64,
            enc_depth=0,
            kpt_cond=dict(enabled=True, fusion_method='sum'),
            phm_loss_weight=3.33e-6,
            num_in_class_proto=3,
            gamma=0.999,
            loss_proto=dict(type='PixelPrototypeCELoss', cfg=dict(weight=1.25e-5, ppc_weight=0.01, ppd_weight=0.001, num_joints=17, num_in_class_proto=3)),
            css_weight=0.0,
            css_conf_thr=0.05,
            css_update_weight=0.01,
        ),
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

work_dir = 'experiments/CUT/work_dirs/pruned20_coco'