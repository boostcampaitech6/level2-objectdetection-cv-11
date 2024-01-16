_base_ = '../../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

data_root = '/data/ephemeral/home/data/dataset/'

metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
}


test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,  
        ann_file='test.json',
        data_prefix=dict(img='')))

# Modify metric related settings
test_evaluator = dict( 
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    classwise=True)

