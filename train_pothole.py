from ultralytics import YOLO

# load base YOLO model
model = YOLO("yolov8n.pt")  # nano version (fast training)

# train model
model.train(
    data="PotHole-1\data.yaml",
    epochs=20,
    imgsz=640,
    batch=8
)