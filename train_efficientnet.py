import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Configuration ---
DATA_ROOT = "datasets/semantic_processed"
CHECKPOINT_PATH = "deeplab_best.pth" # Load this
ENCODER = "efficientnet-b4" 
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["trash"]
ACTIVATION = "sigmoid" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 
EPOCHS = 20
LR = 0.0001
IMG_SIZE = 512

class WasteDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "masks")
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        
    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.images_dir, img_name + ".jpg")
        mask_path = os.path.join(self.masks_dir, img_name + ".png")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        return image, mask
        
    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.0, p=0.3),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)

def train_model():
    print(f"--- Training DeepLabV3+ ({ENCODER}) ---")
    
    train_dataset = WasteDataset(DATA_ROOT, split="train", transform=get_training_augmentation())
    valid_dataset = WasteDataset(DATA_ROOT, split="val", transform=get_validation_augmentation())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)
    
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)
    
    # Tversky Loss (High Recall)
    tversky_loss = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, alpha=0.3, beta=0.7, from_logits=False)
    optimizer = optim.Adam([dict(params=model.parameters(), lr=LR)])
    best_iou = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_logs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img, mask in pbar:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(img)
            loss = tversky_loss(pred, mask)
            loss.backward()
            optimizer.step()
            train_logs.append(loss.item())
            pbar.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        intersect_sum = 0
        union_sum = 0
        
        with torch.no_grad():
            for img, mask in valid_loader:
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                pred = model(img)
                pred_bin = (pred > 0.5).float()
                intersect_sum += (pred_bin * mask).sum().item()
                union_sum += (pred_bin + mask).sum().item() - (pred_bin * mask).sum().item()
        
        epoch_iou = intersect_sum / (union_sum + 1e-7)
        print(f"   Val IoU: {epoch_iou:.4f}")
        
        if epoch_iou > best_iou:
            best_iou = epoch_iou
            # Handle DataParallel vs Standard
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, "deeplab_best.pth")
            print("   ✅ Saved Best Model")

def continue_training():
    print(f"--- Resuming Training DeepLabV3+ ({ENCODER}) ---")
    
    train_dataset = WasteDataset(DATA_ROOT, split="train", transform=get_training_augmentation())
    valid_dataset = WasteDataset(DATA_ROOT, split="val", transform=get_validation_augmentation())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)
    
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )
    
    # LOAD PREVIOUS WEIGHTS
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Handle DataParallel prefix wrap if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    else:
        print("❌ CRITICAL: Checkpoint not found. Starting from scratch!")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(DEVICE)
    
    # Tversky Loss
    tversky_loss = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, alpha=0.3, beta=0.7, from_logits=False)
    optimizer = optim.Adam([dict(params=model.parameters(), lr=LR)])
    
    # We set best_iou to current performance so we only save if we IMPROVE
    best_iou = 0.70 
    
    for epoch in range(EPOCHS):
        model.train()
        train_logs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Resumed]")
        for img, mask in pbar:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(img)
            loss = tversky_loss(pred, mask)
            loss.backward()
            optimizer.step()
            train_logs.append(loss.item())
            pbar.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        intersect_sum = 0
        union_sum = 0
        
        with torch.no_grad():
            for img, mask in valid_loader:
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                pred = model(img)
                pred_bin = (pred > 0.5).float()
                intersect_sum += (pred_bin * mask).sum().item()
                union_sum += (pred_bin + mask).sum().item() - (pred_bin * mask).sum().item()
        
        epoch_iou = intersect_sum / (union_sum + 1e-7)
        print(f"   Val IoU: {epoch_iou:.4f}")
        
        if epoch_iou > best_iou:
            best_iou = epoch_iou
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, "deeplab_best.pth")
            print(f"   ✅ Improved! Saved new best model (IoU: {best_iou:.4f})")

if __name__ == "__main__":
    if os.path.exists(CHECKPOINT_PATH):
        continue_training()
    else:
        train_model()