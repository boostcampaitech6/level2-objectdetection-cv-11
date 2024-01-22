# python train.py \
#     --cfg ./models/yolov5x.yaml \
#     --weights '' \
#     --data trash.yaml \
#     --epochs 100 \
#     --batch-size 64 \
#     --imgsz 512 \
#     --optimizer AdamW \
#     --name yolov5x_AdamW_100epoch_high2_from_scratch \

# python train.py \
#     --weights yolov5x.pt \
#     --data trash.yaml \
#     --epochs 100 \
#     --batch-size 64 \
#     --imgsz 512 \
#     --optimizer AdamW \
#     --name yolov5x_AdamW_100epoch_high2 \

python detect.py \
    --weights ./runs/train/yolov5x_AdamW_100epoch_high2_from_scratch2/weights/best.pt \
    --source ../dataset/images/test/ \
    --imgsz 1024 \
    --save-txt \
    --save-csv \
    --save-conf \
    --name yolov5x_AdamW_100epoch_high_test \