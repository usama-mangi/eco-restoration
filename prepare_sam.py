import torch
import cv2
import os
import numpy as np
import urllib.request
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from glob import glob

# --- Configuration ---
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
DATA_ROOT = "datasets/processed" # Pointing to the NEW dataset
OUTPUT_DIR = "datasets/sam_features"
IMG_SIZE = 1024

def download_weights():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"--- Downloading SAM Weights ({CHECKPOINT_PATH}) ---")
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
        print("Download complete.")
    else:
        print("SAM weights already present.")

def yolo_to_mask(yolo_path, img_w, img_h):
    # Converts YOLO polygon text file back to a binary mask for SAM training
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    if not os.path.exists(yolo_path):
        return mask 
        
    with open(yolo_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        # parts[0] is class_id. We treat ALL waste classes as "Foreground" (1) for SAM.
        coords = [float(x) for x in parts[1:]]
        
        # Denormalize
        poly_points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_w)
            y = int(coords[i+1] * img_h)
            poly_points.append([x, y])
            
        if len(poly_points) > 2:
            pts = np.array(poly_points, np.int32)
            cv2.fillPoly(mask, [pts], 1)
            
    return mask

def precompute_features():
    print("--- Initializing SAM Feature Pre-Computation () ---")
    
    # 1. Setup Model (Use GPU 0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)
    
    # 2. Prepare Output Dirs
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'embeddings'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'masks'), exist_ok=True)

    # 3. Process Images
    # Look for both .jpg and .JPG
    image_files = glob(os.path.join(DATA_ROOT, "images", "*", "*.jpg")) + \
                  glob(os.path.join(DATA_ROOT, "images", "*", "*.JPG")) + \
                  glob(os.path.join(DATA_ROOT, "images", "*", "*.png"))
    
    print(f"Found {len(image_files)} images to process.")
    
    for img_path in tqdm(image_files):
        # Determine split (train/val)
        split = "train" if "/train/" in img_path.replace("\\", "/") else "val"
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Generate SAM Embedding
        try:
            predictor.set_image(image)
            image_embedding = predictor.get_image_embedding().cpu().numpy()
            
            # Generate Ground Truth Mask from YOLO Label
            # We must find the matching label file
            label_path = img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
            
            gt_mask = yolo_to_mask(label_path, w, h)
            
            # Save ONLY if there is a valid mask (waste present)
            if gt_mask.max() > 0:
                np.save(os.path.join(OUTPUT_DIR, split, 'embeddings', name_no_ext + '.npy'), image_embedding)
                cv2.imwrite(os.path.join(OUTPUT_DIR, split, 'masks', name_no_ext + '.png'), gt_mask * 255)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n--- Pre-Computation Complete ---")
    print(f"Features saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_weights()
    precompute_features()