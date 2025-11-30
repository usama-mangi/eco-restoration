import gradio as gr
import cv2
import os
import glob
import numpy as np

# --- CONFIGURATION ---
# Point this to the dataset folder you want to inspect
DATASET_ROOT = "datasets/processed" 
SPLIT = "train"  # 'train' or 'val'

IMG_DIR = os.path.join(DATASET_ROOT, "images", SPLIT)
LBL_DIR = os.path.join(DATASET_ROOT, "labels", SPLIT)

# Get list of images
image_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
                     glob.glob(os.path.join(IMG_DIR, "*.JPG")) +
                     glob.glob(os.path.join(IMG_DIR, "*.png")) + 
                     glob.glob(os.path.join(IMG_DIR, "*.jpeg")))

def get_overlay_color(num_points):
    """
    Visual Helper:
    - Red (0, 0, 255) for Box-Converted Polygons (4 points)
    - Green (0, 255, 0) for True Polygons (>4 points)
    """
    if num_points == 4:
        return (255, 0, 0) # Red (RGB) -> Box
    return (0, 255, 0)     # Green (RGB) -> Polygon

def visualize_image(index):
    if not image_files:
        return None, "No images found in directory."
    
    # Safe Indexing
    index = index % len(image_files)
    img_path = image_files[index]
    filename = os.path.basename(img_path)
    
    # 1. Load Image
    image = cv2.imread(img_path)
    if image is None:
        return None, f"Error loading {filename}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    # 2. Load Label
    lbl_name = os.path.splitext(filename)[0] + ".txt"
    lbl_path = os.path.join(LBL_DIR, lbl_name)
    
    info_text = f"Image {index+1}/{len(image_files)}: {filename}\n"
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        overlay = image.copy()
        
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5: continue
            
            # parts[0] is class_id (should be 0)
            coords = np.array(parts[1:])
            
            # Reshape to (N, 2)
            points = coords.reshape(-1, 2)
            
            # Denormalize
            points[:, 0] *= w
            points[:, 1] *= h
            points = points.astype(np.int32)
            
            # Determine type
            num_pts = len(points)
            color = get_overlay_color(num_pts)
            type_str = "BOX-POLY" if num_pts == 4 else "TRUE-POLY"
            
            # Draw Filled Polygon (Semi-transparent)
            cv2.fillPoly(overlay, [points], color)
            
            # Draw Outline (Solid)
            cv2.polylines(image, [points], True, color, 2)
            
            info_text += f"- {type_str} ({num_pts} pts)\n"

        # Blend Overlay
        alpha = 0.4
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
    else:
        info_text += "No Label File Found (Background Image?)"

    return image, info_text

# --- GRADIO UI ---
with gr.Blocks(title="Dataset Validator") as demo:
    gr.Markdown("# 🕵️ Dataset Ground Truth Validator")
    gr.Markdown("**Red** = Converted Box | **Green** = Original Polygon")
    
    with gr.Row():
        with gr.Column(scale=3):
            display_img = gr.Image(label="Ground Truth Overlay", type="numpy")
        with gr.Column(scale=1):
            status_box = gr.Textbox(label="Label Info", lines=10)
            
            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous")
                next_btn = gr.Button("Next ➡️")
                
            idx_state = gr.State(value=0)

    # Event Handlers
    def on_next(idx):
        new_idx = idx + 1
        img, txt = visualize_image(new_idx)
        return new_idx, img, txt

    def on_prev(idx):
        new_idx = idx - 1
        img, txt = visualize_image(new_idx)
        return new_idx, img, txt

    # Wire buttons
    next_btn.click(on_next, inputs=idx_state, outputs=[idx_state, display_img, status_box])
    prev_btn.click(on_prev, inputs=idx_state, outputs=[idx_state, display_img, status_box])
    
    # Load first image on launch
    demo.load(on_next, inputs=gr.State(-1), outputs=[idx_state, display_img, status_box])

if __name__ == "__main__":
    print(f"Scanning {len(image_files)} images...")
    demo.launch(share=True)