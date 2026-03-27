from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="config.yaml",
    epochs=10,
    imgsz=416,
    batch=8
)