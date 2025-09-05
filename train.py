from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

results = model.train(
    data = "dataset/data.yaml",
    epochs = 50,
    imgsz = 640,
    batch = 16,
)