from mmdet.apis import DetInferencer
from mmengine.config import Config
from pycocotools.coco import COCO
import os
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument(
        '--config',
        help='test config file path'
    )
    parser.add_argument(
        '--model_path',
        help='model path'
    )
    parser.add_argument(
        '--model_name',
        help='model name'
    )
    parser.add_argument(
        '--output_path',
        help='output path'
    )
    parser.add_argument(
        '--image_path',
        help='test image folder path'
    )
    parser.add_argument(
        '--vis',
        help='set visualization',
        default=True
    )
    parser.add_argument(
        '--save_pred',
        help='whether save pred json file',
        default=True
    )
    
    args = parser.parse_args()
    return args

def main():
    class_num = 10
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.roi_head.bbox_head.num_classes = class_num
    cfg.test_pipeline[1]['scale'] = (512,512)
    inferencer = DetInferencer(model=cfg, weights=args.model_path)
    output = inferencer(inputs=args.image_path, out_dir=os.path.join(args.output_path, args.model_name), return_vis=args.vis, no_save_vis=not args.vis, no_save_pred=not args.save_pred)
    
    coco = COCO(os.path.join(cfg.data_root, cfg.test_dataloader['dataset']['ann_file']))
    prediction_strings = []
    file_names = []
    
    for i, out in enumerate(output['predictions']):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        prediction_string = ''
        for j in range(len(out['labels'])):
            prediction_string += str(out['labels'][j]) + ' ' + str(out['scores'][j]) + ' ' + str(out['bboxes'][j][0]) + ' ' + str(out['bboxes'][j][1]) + ' ' + str(out['bboxes'][j][2]) + ' ' + str(out['bboxes'][j][3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(args.output_path, f'submission_{args.model_name}.csv'), index=None)
        
if __name__ == '__main__':
    main()
