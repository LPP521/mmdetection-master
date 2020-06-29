# model settings
model = dict(
    type='MaskRCNN',
    # pretrained='torchvision://resnet50',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe',
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=30 + 1,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=30 + 1,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.02,
        nms=dict(type='nms', iou_thr=0.01),
        max_per_img=100,
        mask_thr_binary=0.5))

# dataset settings
dataset_type = 'ProdDataset'
# dataset_type = 'CompDataset'
data_root = '/data1/liujingyu/DR/'
json_root = '/home/lianjie/mmdetection-master/jsons/'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0, # 0.0625,
        scale_limit=0.0,
        rotate_limit=8,
        interpolation=1,
        p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.25),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=10,
    #             g_shift_limit=10,
    #             b_shift_limit=10,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0)
    #     ],
    #     p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(1600, 800), (1600, 1400)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Pad', size_divisor=32),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_bboxes': 'bboxes',
    #         'gt_masks': 'masks'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1600, 800), (1600, 1400)], # [(1600, 800), (1600, 1400)], (1600, 1200)
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=json_root + 'train_annos.json',
        img_prefix=data_root + 'fold_all/',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        ann_file=json_root + 'test_annos.json',
        img_prefix=data_root + 'fold_all/',
        pipeline=test_pipeline))

# evaluation = dict(interval=1, metric=['bbox', 'segm'])
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 26])
checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 26
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir =  './work_dirs/product/mask_rcnn_dconv_c3-c5_r50_fpn_1x'
load_from = None
resume_from = None # './work_dirs/product/mask_rcnn_dconv_c3-c5_r50_fpn_1x/epoch_14.pth'
workflow = [('train', 1)]

# base+rotate  avgrecall 67.98 ap1 48.73
# base+rotate+multi_scale_test  avgrecall 67.95 ap1 49.97
# base+rotate+cls_weight  avgrecall 66.57 ap1 48.23
# base+rotate+cls_weight+multi_scale_test  avgrecall 67.97 ap1 49.34
# base+rotate+neg  avgrecall 66.86 ap1 49.86
# base+rotate+neg+multi_scale_test  avgrecall 67.99 ap1 50.25
# retinanet avgrecall 62.94 ap1 43.10
# retinanet+flip avgrecall 64.18 ap1 43.64
# base+flip+neg  avgrecall 69.02 ap1 50.50
# base+flip+neg+multi_scale_test  avgrecall 70.32 ap1 50.84
# base+neg+multi_scale_test  avgrecall 66.91 ap1 49.41
# faster+atten avgrecall 67.60 ap1 50.39