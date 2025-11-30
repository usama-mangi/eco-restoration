import json
import os
import shutil
import random
from tqdm import tqdm
import glob

# --- Configuration ---
DEST_DIR = os.path.join("datasets", "processed")

# Source Paths
TACO_DIR = os.path.join("datasets", "raw", "TACO")
ZW_DIR = os.path.join("datasets", "raw", "ZeroWaste")
UAV_DIR = os.path.join("datasets", "raw", "UAVVaste/repo")

# [cite_start]Target Taxonomy [cite: 61]
TARGET_CLASSES = {
    0: "Rigid Container",
    1: "Soft Plastic",
    2: "Paper/Cardboard",
    3: "Micro-Litter",
    4: "Waste Aggregate"
}

def get_target_id(dataset_name, cat_name):
    name = cat_name.lower()
    
    # --- TACO Logic ---
    if dataset_name == "TACO":
        if any(x in name for x in ['bottle', 'can', 'cup', 'lid', 'tub', 'jar', 'blister']): return 0
        if any(x in name for x in ['bag', 'wrapper', 'film', 'foil', 'sachet', 'pouch']): return 1
        if any(x in name for x in ['cardboard', 'paper', 'box', 'carton', 'magazine']): return 2
        if any(x in name for x in ['cigarette', 'straw', 'fragment', 'tab', 'rope']): return 3
        if any(x in name for x in ['trash', 'rubbish', 'heap', 'pile']): return 4

    # [cite_start]--- ZeroWaste Logic [cite: 46] ---
    elif dataset_name == "ZeroWaste":
        # ZeroWaste specific classes usually include 'rigid_plastic', 'soft_plastic', 'cardboard'
        if "rigid" in name or "metal" in name: return 0
        if "soft" in name or "plastic" in name: return 1 # Default plastic to soft if unspecified
        if "cardboard" in name or "paper" in name: return 2

    # [cite_start]--- UAVVaste Logic [cite: 54] ---
    elif dataset_name == "UAVVaste":
        # Aerial view heaps -> Waste Aggregate
        return 4 

    return -1

def process_dataset(name, json_path, img_root_dir, dest_root):
    print(f"--- Processing {name} ---")
    
    if not os.path.exists(json_path):
        print(f"Skipping {name}: JSON not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create ID map
    cat_map = {c['id']: c['name'] for c in data['categories']}
    
    # Map Annotations to Images
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns: img_anns[img_id] = []
        img_anns[img_id].append(ann)

    images = data['images']
    # Use 50% of ZeroWaste to avoid overwhelming TACO (imbalance handling)
    if name == "ZeroWaste": 
        random.shuffle(images)
        images = images[:int(len(images)*0.5)]

    processed_count = 0
    
    for img_info in tqdm(images, desc=f"Converting {name}"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Locate Image
        src_path = os.path.join(img_root_dir, file_name)
        if not os.path.exists(src_path):
            # Try flat search if subdirectory structure mismatches
            basename = os.path.basename(file_name)
            potential_path = os.path.join(img_root_dir, basename)
            if os.path.exists(potential_path):
                src_path = potential_path
            else:
                continue # Image missing

        # Determine Split (80/20)
        split = "train" if random.random() < 0.8 else "val"
        
        yolo_lines = []
        if img_id in img_anns:
            img_w = img_info['width']
            img_h = img_info['height']
            
            for ann in img_anns[img_id]:
                cat_name = cat_map.get(ann['category_id'], "unknown")
                target_id = get_target_id(name, cat_name)
                
                if target_id != -1 and 'segmentation' in ann:
                    # Handle Polygons
                    for seg in ann['segmentation']:
                        if len(seg) < 6: continue
                        
                        # Normalize
                        norm_coords = []
                        for i in range(0, len(seg), 2):
                            x = max(0, min(1, seg[i] / img_w))
                            y = max(0, min(1, seg[i+1] / img_h))
                            norm_coords.extend([f"{x:.6f}", f"{y:.6f}"])
                        
                        yolo_lines.append(f"{target_id} " + " ".join(norm_coords))

        if yolo_lines:
            # Save Image (Rename to avoid collisions)
            new_name = f"{name}_{img_id}_{os.path.basename(src_path)}"
            dst_img = os.path.join(dest_root, "images", split, new_name)
            shutil.copy2(src_path, dst_img)
            
            # Save Label
            dst_lbl = os.path.join(dest_root, "labels", split, os.path.splitext(new_name)[0] + ".txt")
            with open(dst_lbl, 'w') as f:
                f.write("\n".join(yolo_lines))
            processed_count += 1
            
    print(f"Finished {name}: {processed_count} images.")

def main():
    # Setup Directories
    if os.path.exists(DEST_DIR): shutil.rmtree(DEST_DIR)
    for s in ['train', 'val']:
        os.makedirs(os.path.join(DEST_DIR, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(DEST_DIR, 'labels', s), exist_ok=True)

    # 1. Process TACO
    taco_json = os.path.join(TACO_DIR, "data", "annotations.json")
    process_dataset("TACO", taco_json, os.path.join(TACO_DIR, "data"), DEST_DIR)

    # 2. Process ZeroWaste
    # Find the JSON file (usually labeled labels.json or train.json)
    zw_jsons = glob.glob(os.path.join(ZW_DIR, "train", "*.json"))
    if not zw_jsons: zw_jsons = glob.glob(os.path.join(ZW_DIR, "*.json"))
    
    if zw_jsons:
        # Assuming the first JSON found is the main annotation file
        process_dataset("ZeroWaste", zw_jsons[0], os.path.join(ZW_DIR, "train", "data"), DEST_DIR)
    else:
        print("CRITICAL: ZeroWaste JSON not found.")

    # 3. Process UAVVaste
    uav_json = os.path.join(UAV_DIR, "annotations/annotations.json")
    # UAVVaste images might be in 'images' or flat
    process_dataset("UAVVaste", uav_json, os.path.join(UAV_DIR, "images"), DEST_DIR)

    # 4. Generate YAML
    yaml_content = f"""
path: {os.path.abspath(DEST_DIR)}
train: images/train
val: images/val
names:
  0: Rigid Container
  1: Soft Plastic
  2: Paper/Cardboard
  3: Micro-Litter
  4: Waste Aggregate
"""
    with open("dataset.yaml", "w") as f:
        f.write(yaml_content)
    print("\n--- Dataset Ready ---")
    print("Created dataset.yaml")

if __name__ == "__main__":
    main()