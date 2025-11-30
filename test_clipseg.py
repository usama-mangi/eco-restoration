import torch
import cv2
import numpy as np
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_ID = "CIDAS/clipseg-rd64-refined"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = ["trash", "garbage", "rubbish pile", "plastic waste"]
# Target Image (The one YOLO failed on)
IMAGE_PATH = "image_2abc78.jpg" 

def run_clipseg():
    print(f"--- Running CLIPSeg (Zero-Shot) on {IMAGE_PATH} ---")
    
    # 1. Load Model
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).to(DEVICE)
    print("✅ CLIPSeg Loaded")

    # 2. Load Image
    if not os.path.exists(IMAGE_PATH):
        # Fallback to validation folder
        import glob
        files = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
        if not files: return
        image_path = files[6]
    else:
        image_path = IMAGE_PATH
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    # 3. Inference
    print(f"   Querying prompts: {PROMPTS}")
    inputs = processor(
        text=PROMPTS, 
        images=[image] * len(PROMPTS), 
        padding="max_length", 
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        
    # 4. Process Output
    # preds is (N_prompts, 352, 352) - CLIPSeg works at low res
    preds = outputs.logits.unsqueeze(1)
    
    # Resize back to original
    preds = torch.nn.functional.interpolate(preds, size=(H, W), mode="bilinear", align_corners=False)
    preds = torch.sigmoid(preds) # Convert to 0-1 probability
    
    # Aggregate prompts (Max or Mean)
    # We take the MAX probability across all synonyms
    combined_mask = torch.max(preds, dim=0)[0][0].cpu().numpy()
    
    # Threshold
    binary_mask = (combined_mask > 0.4).astype(np.uint8) # 0.4 is a good starting point for CLIPSeg
    
    # 5. Visualize
    print("   Saving visualization...")
    
    # Heatmap
    heatmap = cv2.applyColorMap((combined_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay
    vis = image.copy()
    vis[binary_mask == 1] = vis[binary_mask == 1] * 0.5 + np.array([0, 0, 255]) * 0.5 # Red tint
    
    # Stack: Original | Heatmap | Binary Mask
    # Resize heatmap/mask to match image for stacking if needed (already resized above)
    
    # If image is huge, downscale for display
    display_h = 600
    scale = display_h / H
    d_w = int(W * scale)
    
    vis_s = cv2.resize(vis, (d_w, display_h))
    heat_s = cv2.resize(heatmap, (d_w, display_h))
    mask_s = cv2.resize(binary_mask * 255, (d_w, display_h))
    mask_s = cv2.cvtColor(mask_s, cv2.COLOR_GRAY2BGR)
    
    combined = np.hstack([vis_s, heat_s, mask_s])
    
    cv2.imwrite("clipseg_test.jpg", combined)
    print("✅ Saved 'clipseg_test.jpg'")

if __name__ == "__main__":
    import os
    run_clipseg()