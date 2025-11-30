import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from segment_anything import sam_model_registry
from tqdm import tqdm
import torch.nn.functional as F

# --- Configuration ---
FEATURE_DIR = "datasets/sam_features" 
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Effective Batch Strategy
# We process 1 image at a time to avoid SAM shape bugs, but accumulate 32 times.
PHYSICAL_BATCH_SIZE = 1 
GRAD_ACCUMULATION_STEPS = 32
LR = 8e-4       
EPOCHS = 20
TARGET_SIZE = 1024 

# --- Custom Dataset ---
class SAMFeatureDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.embed_dir = os.path.join(root_dir, split, "embeddings")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.files = [f.replace(".npy", "") for f in os.listdir(self.embed_dir) if f.endswith(".npy")]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        name = self.files[idx]
        
        # 1. Load Embedding
        embedding = np.load(os.path.join(self.embed_dir, name + ".npy"))
        embedding = torch.as_tensor(embedding, dtype=torch.float32).squeeze(0) 
        
        # 2. Load Mask
        mask_path = os.path.join(self.mask_dir, name + ".png")
        mask = cv2.imread(mask_path, 0)
        
        if mask.shape != (TARGET_SIZE, TARGET_SIZE):
            mask = cv2.resize(mask, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

        mask = torch.as_tensor(mask, dtype=torch.float32) / 255.0
        mask = (mask > 0.5).float()
        
        # 3. Generate Box Prompt
        y_indices, x_indices = torch.where(mask > 0)
        if len(y_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # Perturbation
            noise = torch.randint(-5, 5, (4,))
            box = torch.tensor([x_min, y_min, x_max, y_max]) + noise
            box = torch.clamp(box, min=0, max=TARGET_SIZE) 
        else:
            box = torch.tensor([0, 0, 10, 10]) 
            
        return embedding, mask, box

# --- Loss Functions ---
def dice_loss(pred, target):
    smooth = 1.0
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def focal_loss(pred, target, alpha=0.25, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt)**gamma * bce
    return loss.mean()

# --- Training Loop ---
def train():
    print(f"--- SAM Decoder Training (Grad Accumulation: {GRAD_ACCUMULATION_STEPS}) ---")
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device)
    
    # Freeze Encoders
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(sam.mask_decoder.parameters(), lr=LR, weight_decay=1e-4)
    
    train_ds = SAMFeatureDataset(FEATURE_DIR, "train")
    # Reduced num_workers to 0 to prevent multiprocessing glitches with simple loaders
    train_loader = DataLoader(train_ds, batch_size=PHYSICAL_BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Training on {len(train_ds)} samples.")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        sam.mask_decoder.train()
        total_loss = 0
        optimizer.zero_grad() # Initialize gradient
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, (embeddings, gt_masks, boxes) in enumerate(pbar):
            embeddings = embeddings.to(device)
            gt_masks = gt_masks.to(device).unsqueeze(1) 
            boxes = boxes.to(device)
            
            # Encode Prompts
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,
                boxes=boxes[:, None, :], # (1, 1, 4)
                masks=None,
            )
            
            # Decode Masks
            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=embeddings, # (1, 256, 64, 64)
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            upscaled_masks = F.interpolate(low_res_masks, (TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)
            
            loss = focal_loss(upscaled_masks, gt_masks) + dice_loss(upscaled_masks, gt_masks)
            
            # Scale Loss
            loss = loss / GRAD_ACCUMULATION_STEPS
            loss.backward()
            
            # Update Weights every N steps
            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRAD_ACCUMULATION_STEPS # Scale back for display
            pbar.set_postfix(loss=loss.item() * GRAD_ACCUMULATION_STEPS)
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(sam.mask_decoder.state_dict(), "sam_waste_decoder.pth")
            print("Saved Best Model.")

    print("--- Training Complete ---")

if __name__ == "__main__":
    train()