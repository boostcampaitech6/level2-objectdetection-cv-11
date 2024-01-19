from ultralytics import YOLO

# 모델을 로드하세요.
# sizes = ['n', 's', 'm', 'l', 'x']
size = 'm' 
mosaics = [1]
nums = [3, 4, 5]
confs = [0.01]


for num in nums:
	# 모델을 훈련합니다.
	model = YOLO(f'yolov8{size}.pt')  # 사전 훈련된 모델 로드 (훈련을 위해 권장됨)

	# results = model.train(cfg=f'divide_{size}.yaml', data=f'stratified_boxSize_{num}.yaml', name=f'stratified_boxSize_{num}_{size}', imgsz=640, epochs=70)
	results = model.train(cfg=f'divide_{size}.yaml', data=f'stratified_boxSize_rgt_{num}.yaml', name=f'stratified_boxSize_rgt_{num}_{size}_BCE', imgsz=640, epochs=80, conf=0.01, mosaic=1, lrf=0.1, batch=16, close_mosaic=5, hsv_s=0.3)