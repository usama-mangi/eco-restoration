import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# --- Configuration ---
SOURCE_IMAGES = "datasets/processed/images"
SOURCE_LABELS = "datasets/processed/labels"
DEST_ROOT = "datasets/semantic_processed"

def yolo_to_mask(img_shape, label_path):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not os.path.exists(label_path):
        return mask
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        # parts[0] is class, we ignore it (Binary Trash)
        coords = [float(x) for x in parts[1:]]
        
        # Denormalize polygon
        poly_points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            poly_points.append([x, y])
            
        if len(poly_points) > 2:
            pts = np.array(poly_points, np.int32)
            cv2.fillPoly(mask, [pts], 255) # White
            
    return mask

def convert_dataset():
    print("--- Converting YOLO Polygons to Semantic Masks ---")
    
    if os.path.exists(DEST_ROOT):
        import shutil
        shutil.rmtree(DEST_ROOT)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DEST_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DEST_ROOT, split, "masks"), exist_ok=True)
        
        img_dir = os.path.join(SOURCE_IMAGES, split)
        lbl_dir = os.path.join(SOURCE_LABELS, split)
        
        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob( os.path.join(img_dir, "*.JPG"))
        
        print(f"Processing {split} ({len(img_files)} images)...")
        
        for img_path in tqdm(img_files):
            # Read Image to get shape
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Find Label
            basename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(basename)[0]
            lbl_path = os.path.join(lbl_dir, name_no_ext + ".txt")
            
            # Create Mask
            mask = yolo_to_mask(img.shape, lbl_path)
            
            # Save Image (Copy)
            dst_img = os.path.join(DEST_ROOT, split, "images", basename)
            cv2.imwrite(dst_img, img)
            
            # Save Mask (PNG)
            dst_mask = os.path.join(DEST_ROOT, split, "masks", name_no_ext + ".png")
            cv2.imwrite(dst_mask, mask)
            
    print(f"✅ Conversion Complete. Data ready at {DEST_ROOT}")

if __name__ == "__main__":
    convert_dataset()