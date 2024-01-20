import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold 
from pprint import pprint
import os

from collections import Counter
import pandas as pd

# # load json
# annotation = '/data/ephemeral/home/dataset/train.json'

# with open(annotation) as f:
#     data = json.load(f)

# var_dict = dict()

# for ann in data['annotations']:
#     if ann['image_id'] in var_dict:
#         var_dict[ann['image_id']] += 1
#     else:
#         var_dict[ann['image_id']] = 1

# var = dict()
# for key, value in var_dict.items():
#     if value <= 3:
#         var[key] = 's'
#     elif value <= 7:
#         var[key] = 'm'
#     else:
#         var[key] = 'l'



# X = np.ones((4883,1))
# y = np.array([v[1] for v in var.items()])
# groups = np.array([v[0] for v in var.items()])

# cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)




def main():
    annotation = '/data/ephemeral/dataset/train.json'

    with open(annotation) as f: 
        data = json.load(f)
    
    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]


    X = np.ones((len(data['annotations']),1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    SEED = 333
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


    save_dir = '/data/ephemeral/dataset'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):

        train_img_ids = set([idx for idx in groups[train_idx]])
        train_imgs = [data['images'][idx] for idx in train_img_ids]

        train_data = {
            'images': train_imgs,
            'categories': data['categories'],
            'annotations': [ann for ann in data['annotations'] if ann['image_id'] in train_img_ids]
        }
        
        val_img_ids = set([idx for idx in groups[val_idx]])
        val_imgs = [data['images'][idx] for idx in val_img_ids]
        
        val_data = {
            'images': val_imgs,
            'categories' : data['categories'],
            'annotations': [ann for ann in data['annotations'] if ann['image_id'] in val_img_ids]
        }
        
        train_path = os.path.join(save_dir, f'train_{SEED}_fold_{fold}.json')
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=4)
            
        val_path = os.path.join(save_dir, f'val_{SEED}_fold_{fold}.json')
        with open(val_path, 'w') as f:
            json.dump(val_data, f, indent=4)

if __name__ == '__main__':
    main()