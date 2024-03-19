# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    "../_base_/models/mask_rcnn_r50_fpn.py",
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../runtime.py"
]
default_scope = "mmdet"

pretrained = (
    "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth"
)
# pretrained = 'mask_rcnn_internimage_t_fpn_1x_coco.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type="InternImage",
        core_op="DCNv3",
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        norm_layer="LN",
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(
        type="FPN", in_channels=[64, 128, 256, 512], out_channels=256, num_outs=5
    ),
)
# By default, models are trained on 8 GPUs with 2 images per GPU
# data = dict(samples_per_gpu=2)
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    constructor="CustomLayerDecayOptimizerConstructor",
    optimizer=dict(
        type="AdamW",
        lr=0.0001,
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0, depths=[4, 4, 18, 4]),
)
optimizer_config = dict(grad_clip=None)
fp16 = dict(loss_scale=dict(init_scale=512))
evaluation = dict(save_best="auto")
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)
