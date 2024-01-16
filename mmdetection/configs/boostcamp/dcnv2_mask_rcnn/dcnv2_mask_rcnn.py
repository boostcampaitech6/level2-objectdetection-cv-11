_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

data_root = '/data/ephemeral/home/data/dataset/'

metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
}

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_333_fold_1.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,  
        ann_file='val_333_fold_1.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_333_fold_1.json',
    metric='bbox',
    format_only=False,
    classwise=True)

auto_scale_lr = dict(enable=False, base_batch_size=16)