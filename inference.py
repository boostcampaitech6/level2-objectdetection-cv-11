from ultralytics import YOLO
import json
import pandas as pd
import os

nums = [1, 2, 3]
for n in nums:
    root = '/data/ephemeral/baseline/ultralytics-main/runs/detect/divide'
    name = f'stratified_boxSize_rgt_{n}_x_BCE'

    model = YOLO(os.path.join(root, name, 'weights/best.pt'))  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    dataset_path = '/data/ephemeral/dataset'
    with open('/data/ephemeral/dataset/test.json') as f:
        data = json.load(f)
        img_paths = [os.path.join(dataset_path, img['file_name']) for img in data['images']]

    pascal_voc_data = pd.DataFrame()
    predictionString = []
    batchSize = 50
    print('Calculating...')
    for j in range(0, len(img_paths), batchSize):
        results = model(img_paths[j:min(j+batchSize, len(img_paths))], conf=0.001, imgsz=[1024,1024], stream=True)  

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            pred = ''
            for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
                pred += ' '.join([str(int(cls.item())), str(conf.item()), ' '.join(str(i) for i in xyxy.tolist())]) + ' '
            predictionString.append(pred)
        print(f'{min((j+batchSize)/len(img_paths)*100, 100)}% complete...')

    pascal_voc_data['PredictionString'] = predictionString 
    pascal_voc_data['image_id'] = [img['file_name'] for img in data['images']]
    pascal_voc_data.to_csv(os.path.join(root, name, f'submission_{name}.csv'), index=None)
    pascal_voc_data.head()