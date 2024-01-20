from mmdet.apis import DetInferencer
from mmengine.config import Config
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
import os
import argparse
import pandas as pd

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')

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
        default=False
    )
    parser.add_argument(
        '--save_pred',
        help='whether save pred json file',
        default=False
    )
    
    args = parser.parse_args()
    return args

def main():
    class_num = 10
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    cfg = Config.fromfile(args.config)
    if 'roi_head' in cfg.model:
        cfg.model.roi_head.bbox_head.num_classes = class_num
    else:
        cfg.model.bbox_head.num_classes = class_num
    cfg.test_pipeline[1]['scale'] = (512,512)
    inferencer = DetInferencer(model=cfg, weights=args.model_path)
    output = inferencer(inputs=args.image_path, return_datasamples=True)
                
    prediction_strings = []
    file_names = []
    
    for out in output['predictions']:
        file_name = os.path.basename(out.img_path)
        file_name = os.path.join('test', file_name)
        prediction_string = ''
        for i in range(len(out.pred_instances.labels)):
            prediction_string += str(out.pred_instances.labels.tolist()[i]) + ' ' + str(out.pred_instances.scores.tolist()[i]) + ' ' + str(out.pred_instances.bboxes.tolist()[i][0]) + ' ' + str(out.pred_instances.bboxes.tolist()[i][1]) + ' ' + str(out.pred_instances.bboxes.tolist()[i][2]) + ' ' + str(out.pred_instances.bboxes.tolist()[i][3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(file_name)
    
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(args.output_path, f'submission_retinanet_swin_tta.csv'), index=None)
    
    if args.vis:
        file_list = list_dir_or_file(args.image_path, list_dir=False, suffix=IMG_EXTENSIONS)
        inputs = [ join_path(args.image_path , filename) for filename in file_list ]
        inferencer.visualize(inputs = inputs, preds = output['predictions'], img_out_dir = args.output_path)
    
    if args.save_pred:
        for out in output['predictions']:
            inferencer.pred2dict(out, args.output_path)
        
if __name__ == '__main__':
    main()