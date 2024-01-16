# MMDetection
v3.3.0

설치 : <a href = "mmdetection/docs/en/get_started.md"> get_strated.md </a>

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Note:** In MMCV-v2.x, `mmcv-full` is rename to `mmcv`, if you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc1"` to install the lite version.

**Step 1.** Install MMDetection.

Case a: If you develop and run mmdet directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmdet as a dependency or third-party package, install it with MIM:

```shell
mim install mmdet
```

# Train

<a href="train.py"> train.py </a> 를 통해 실행

-- config : config 파일 경로, kfold 이용할 경우 입력 안함

-- work-dir : 모델과 로그가 저장될 경로

-- auto-scale-lr : auto scale 사용 유무. 사용하지 않을 시, 실행 argument에서 입력 안하면 됨

-- resume : checkpoint에서 모델을 이어받아 계속 학습할 경우. 이때, checkpoint 모델의 epoch 이 10 이고 새로 설정한 epoch이 15인 경우, 5 epoch 만큼만 추가 학습함

-- kfold : kfold 사용할지 유무, 현재 문제가 있어서 사용x

-- kfold_config : kfold 사용할 경우 config 파일들이 있는 폴더 경로

-- epoch : 최대 epoch. 현재 EarlyStopping이 기본으로 사용되도록 설정되어 있어 총 epoch과는 다를 수 있음

** model 및 dataset, json path는 config 파일에서 지정함

### Example
```shell
python3 train.py \
        --config /path/to/config \
        --work-dir /path/to/work/dir \
        --epoch 20 \        
        --auto-scale-lr (사용하는 경우)
```

# Inference

<a href= "mmdetection/inference.py"> inference.py </a> 파일을 통해 실행

-- config : test용 config 파일 경로

-- model_path : 학습된 .pth 모델 파일 경로

-- model_name : 모델 이름. 결과 csv 파일 저장할 때 파일 이름에 사용됨

-- output_path : csv파일 및 json, visualization 결과가 저장될 경로

-- image_path : test 폴더 경로

-- vis : visualization 한 결과가 저장될지 유무. jpg 파일 형태로 저장됨. 사용할 경우 시간이 오래 걸릴 수 있음 (Default = Fasle)

-- save_pred : 결과가 각 파일에 대해 json 파일로 저장될 지 유무 (Default = False)

### Example
```shell
python3 inference.py \
        --config /path/to/config \
        --model_path /path/to/model \
        --model_name model_name \
        --output_path /path/to/output \
        --image_path /path/to/test/image/folder
```


# Config
각 모델에 적용한 config 파일

<a href = "mmdetection/configs/boostcamp/dcnv2_faster_rcnn/dcnv2_faster_rcnn.py"> DCNv2 faster rcnn (train) </a>

<a href = "mmdetection/configs/boostcamp/test/dcnv2_faster_rcnn_test.py"> DCNv2 faster rcnn (test) </a>

<a href = "mmdetection/configs/boostcamp/dcnv2_mask_rcnn/dcnv2_faster_rcnn.py"> DCNv2 mask rcnn (train) </a>

<a href = "mmdetection/configs/boostcamp/test/dcnv2_mask_rcnn_test.py"> DCNv2 mask rcnn (test) </a>

<a href = "mmdetection/configs/boostcamp/deformable_detr/deformable_detr.py"> Deformable DETR (train) </a>

<a href = "mmdetection/configs/boostcamp/test/deformable_detr_test.py"> Deformable DETR (test) </a>
