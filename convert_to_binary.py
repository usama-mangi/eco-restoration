import utils.transport
import utils.session
import os
import shutil
from tqdm import tqdm
import glob

# --- Configuration ---
SOURCE_DIR = "datasets/processed"
DEST_DIR = "datasets/binary_processed"

def convert_to_binary():
    print("--- Converting Dataset to Binary (Trash vs Background) ---")
    
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    
    # 1. Copy Images (Symlink would be faster, but copy is safer for Windows/VMs)
    print("Copying images...")
    for split in ['train', 'val']:
        src_img_dir = os.path.join(SOURCE_DIR, 'images', split)
        dst_img_dir = os.path.join(DEST_DIR, 'images', split)
        os.makedirs(dst_img_dir, exist_ok=True)
        
        # Copy files
        files = os.listdir(src_img_dir)
        for f in tqdm(files, desc=f"Copying {split} images"):
            shutil.copy2(os.path.join(src_img_dir, f), os.path.join(dst_img_dir, f))

    # 2. Process Labels
    print("\nProcessing labels (Merging classes to ID 0)...")
    for split in ['train', 'val']:
        src_lbl_dir = os.path.join(SOURCE_DIR, 'labels', split)
        dst_lbl_dir = os.path.join(DEST_DIR, 'labels', split)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        files = glob.glob(os.path.join(src_lbl_dir, "*.txt"))
        for f in tqdm(files, desc=f"Converting {split} labels"):
            with open(f, 'r') as file:
                lines = file.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 1:
                    # FORCE CLASS ID TO 0
                    parts[0] = "0" 
                    new_lines.append(" ".join(parts))
            
            # Write to new destination
            out_path = os.path.join(dst_lbl_dir, os.path.basename(f))
            with open(out_path, 'w') as file:
                file.write("\n".join(new_lines))

    # 3. Create YAML
    yaml_content = f"""
path: {os.path.abspath(DEST_DIR)}
train: images/train
val: images/val
names:
  0: Trash
"""
    with open("binary_dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print("\n✅ Success! Binary dataset created at 'datasets/binary_processed'")
    print("   Created 'binary_dataset.yaml'")

if __name__ == "__main__":
    convert_to_binary()