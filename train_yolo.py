from ultralytics import YOLO
import torch
import os

def train_model():
    print("--- Initializing YOLOv8-Large-Seg (Unified Dataset) ---")
    
    # Load model (Pre-trained)
    model = YOLO('yolov8l-seg.pt')

    # Hyperparameters (PDF Section 5.4.1)
    hyperparams = {
        "epochs": 100,              # Start with 100 epochs
        "imgsz": 640,               # Standard resolution for speed
        "batch": 16,                # 8 per GPU (Verified Safe on 8GB VRAM)
        "device": [0, 1],           # Dual GPU
        "optimizer": "SGD",         # PDF Recommendation
        "lr0": 0.01,
        "momentum": 0.937,
        "cos_lr": True,
        "mosaic": 1.0,              # Critical for small trash (TACO)
        "box": 7.5,
        "cls": 0.5,
        "patience": 50,
        "workers": 4,
        "project": "logs",
        "name": "yolov8l_unified_run",
        "amp": True                 # Mixed Precision ON
    }

    # Verify YAML exists
    yaml_path = 'dataset.yaml'
    if not os.path.exists(yaml_path):
        print("CRITICAL ERROR: No dataset.yaml found!")
        return

    print(f"--- Starting Training on {yaml_path} ---")
    
    # Start Training
    model.train(data=yaml_path, **hyperparams)

if __name__ == "__main__":
    train_model()