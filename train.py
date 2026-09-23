from ultralytics import YOLO

# YOLO 기본 모델 불러오기
model = YOLO("yolo11n.pt")

# 학습 시작
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)

print("학습 완료!")