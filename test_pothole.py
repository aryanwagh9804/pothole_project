from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("PotHole-1/valid/images")

results[0].show()