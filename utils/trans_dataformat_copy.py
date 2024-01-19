import os
import json
from tqdm import tqdm
import shutil

def convert_coco_to_yolo_format(root_dir: str, json_file_train: str, json_file_val: str, save_dir: str):
    # Check directory
    try:
        assert os.path.exists(os.path.join(root_dir, save_dir, "images", "train")) == True
        assert os.path.exists(os.path.join(root_dir, save_dir, "images", "val")) == True
        assert os.path.exists(os.path.join(root_dir, save_dir, "labels", "train")) == True
        assert os.path.exists(os.path.join(root_dir, save_dir, "labels", "val")) == True
    except:
        os.makedirs(os.path.join(root_dir, save_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, save_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, save_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, save_dir, "labels", "val"), exist_ok=True)
    finally:
        print("Finish make directory")
   
    # Load json
    with open(os.path.join(root_dir, json_file_train), 'r') as f:
        coco_json_t = json.load(f)
    with open(os.path.join(root_dir, json_file_val), 'r') as f:
        coco_json_v = json.load(f)
    anoots_t = coco_json_t["annotations"]
    anoots_v = coco_json_v["annotations"]
   
    print("Start converting...")
    for image in tqdm(sorted(coco_json_t["images"], key=lambda x: x["id"])):
        
        w, h, file_name, image_id = image["width"], image["height"], image["file_name"], image["id"]
        file_name = file_name.split("/")[1]
        # filtering annotations
        obj_candits = list(filter(lambda x: x["image_id"] == image_id, anoots_t))
        # Save txt format to train yolo
        with open(os.path.join(root_dir, save_dir, "labels", "train", f"{file_name[:-4]}.txt"), "w") as f:
            for obj_candit in obj_candits:
                # x1 y1 w h -> cx cy w h                
                cat_id = obj_candit["category_id"]

                x1, y1, width, height = obj_candit["bbox"]
                scaled_cx, scaled_cy = (x1 + width/2) / w, (y1 + height/2) / h
                scaled_width, scaled_height = width / w, height / h
                f.write("%s %.3f %.3f %.3f %.3f\n" % (cat_id, scaled_cx, scaled_cy, scaled_width, scaled_height))     
        # Copy image to new directory
        shutil.copy(os.path.join(root_dir, "train", file_name), os.path.join(root_dir, save_dir, "images", "train", file_name))    

    for image in tqdm(sorted(coco_json_v["images"], key=lambda x: x["id"])):
        
        w, h, file_name, image_id = image["width"], image["height"], image["file_name"], image["id"]
        file_name = file_name.split("/")[1]
        # filtering annotations
        obj_candits = list(filter(lambda x: x["image_id"] == image_id, anoots_v))
        # Save txt format to train yolo
        with open(os.path.join(root_dir, save_dir, "labels", "val", f"{file_name[:-4]}.txt"), "w") as f:
            for obj_candit in obj_candits:
                # x1 y1 w h -> cx cy w h                
                cat_id = obj_candit["category_id"]

                x1, y1, width, height = obj_candit["bbox"]
                scaled_cx, scaled_cy = (x1 + width/2) / w, (y1 + height/2) / h
                scaled_width, scaled_height = width / w, height / h
                f.write("%s %.3f %.3f %.3f %.3f\n" % (cat_id, scaled_cx, scaled_cy, scaled_width, scaled_height))     
        # Copy image to new directory
        shutil.copy(os.path.join(root_dir, "train", file_name), os.path.join(root_dir, save_dir, "images", "val", file_name))    
    print("Finish converting...")

for i in range(1, 6):
    convert_coco_to_yolo_format("dataset", "train_rgt_441_fold_" + str(i) + ".json","val_rgt_441_fold_" + str(i) + ".json", "yolo_rgt_441_fold_" + str(i))