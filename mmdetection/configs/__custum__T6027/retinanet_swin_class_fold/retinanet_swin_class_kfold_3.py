_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py',
    '../retinanet_tta.py'
]


data_root='/data/ephemeral/home/mmdetection/dataset/'

metainfo={
    'classes':("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}



train_dataloader=dict(
    batch_size=10,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_580_fold_3.json',
        data_prefix=dict(img='')
    )
)

val_dataloader=dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_580_fold_3.json',
        data_prefix=dict(img='')
    )
)

test_dataloader=val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_580_fold_3.json',
    metric='bbox',
    format_only=False,
    classwise=True)
test_evaluator=val_evaluator
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))