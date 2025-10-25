from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

results = model.train(
    data="data/Chess Pieces.yolov8-obb/data.yaml",  
    epochs=10,
    imgsz=640,
    batch=16,
    name="chess_yolo_model",
    project="results"  # Creates 'results' folder in current directory
)