from ultralytics import YOLO

def train_binary():
    print("--- Training YOLOv8-Large (Binary Mode) ---")
    
    # Load base model (reset weights to avoid confusion from previous classes)
    model = YOLO('yolov8l-seg.pt')

    hyperparams = {
        "epochs": 50,               # 50 epochs is usually enough for binary
        "imgsz": 640,               # Start at 640 for stability
        "batch": 16,                # 8 per GPU
        "device": [0, 1],           
        "optimizer": "SGD",
        "lr0": 0.01,
        "momentum": 0.937,
        "cos_lr": True,
        "mosaic": 1.0,
        "box": 7.5,
        "cls": 0.5,
        "patience": 20,
        "project": "logs",
        "name": "yolov8l_binary_run",
        "amp": True,
        "single_cls": True          # EXPLICITLY TELL YOLO IT'S SINGLE CLASS
    }

    print(f"--- Starting Binary Training ---")
    model.train(data='binary_dataset.yaml', **hyperparams)

if __name__ == "__main__":
    train_binary()