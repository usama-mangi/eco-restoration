import torch
import cv2
import numpy as np
import os
import glob
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch.nn.functional as F

# --- Configuration ---
DEEPLAB_PATH = "deeplab_best_efficientnet.pth"
LAMA_PATH = "big-lama.pt"
CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
ENCODER = "efficientnet-b4" 

# Tiling Settings
TILE_SIZE = 1024   # LaMa Input Size
STRIDE = 512       # Overlap (Keep center 512)

DEVICE_0 = "cuda:0" # Segmentation
DEVICE_1 = "cuda:1" # Inpainting

PROMPTS = ["trash", "garbage", "rubbish pile", "plastic waste", "litter"]

def load_models():
    print(f"--- Loading Pro Engines (Sharpened) ---")
    
    # 1. DeepLabV3+
    deeplab = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=None, 
        classes=1, 
        activation="sigmoid",
    )
    if os.path.exists(DEEPLAB_PATH):
        state_dict = torch.load(DEEPLAB_PATH, map_location=DEVICE_0)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        deeplab.load_state_dict(new_state_dict)
    deeplab.to(DEVICE_0)
    deeplab.eval()
    print("✅ DeepLabV3+ Loaded")
    
    # 2. CLIPSeg
    processor = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
    clipseg = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).to(DEVICE_0)
    print("✅ CLIPSeg Loaded")
    
    # 3. LaMa
    lama = torch.jit.load(LAMA_PATH, map_location=DEVICE_1)
    lama.eval()
    print("✅ LaMa Loaded")
    
    return deeplab, (processor, clipseg), lama

def get_hybrid_mask(deeplab, clipseg_engine, image_rgb):
    # DeepLab
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE_0)
    with torch.no_grad():
        mask_pred = deeplab(input_tensor)
    mask_512 = mask_pred[0, 0].cpu().numpy()
    mask_dl = cv2.resize(mask_512, (image_rgb.shape[1], image_rgb.shape[0])) > 0.5

    # CLIPSeg
    processor, model = clipseg_engine
    inputs = processor(
        text=PROMPTS, 
        images=[image_rgb] * len(PROMPTS), 
        padding="max_length", 
        return_tensors="pt"
    ).to(DEVICE_0)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)
    preds = F.interpolate(preds, size=(image_rgb.shape[0], image_rgb.shape[1]), mode="bilinear", align_corners=False)
    preds = torch.sigmoid(preds)
    combined = torch.max(preds, dim=0)[0][0].cpu().numpy()
    mask_cs = combined > 0.4

    final_mask = np.logical_or(mask_dl, mask_cs).astype(np.uint8)
    return final_mask

def expand_mask_dynamically(mask, scale_percent=0.02):
    """
    Expands mask relative to image size.
    Fixed pixel values fail on high-res images.
    """
    h, w = mask.shape[:2]
    # Calculate expansion size (e.g. 2% of width)
    kernel_size = int(max(h, w) * scale_percent)
    
    # Ensure odd number for kernel
    if kernel_size % 2 == 0: kernel_size += 1
    
    # Cap minimum to avoid doing nothing
    kernel_size = max(15, kernel_size)
    
    print(f"   Dynamic Dilation: Kernel Size {kernel_size}px")
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    return dilated

def tiled_inpainting(image, mask, model, device):
    H, W = image.shape[:2]
    
    # Pad image to handle edges
    pad_h = (STRIDE - H % STRIDE) % STRIDE
    pad_w = (STRIDE - W % STRIDE) % STRIDE
    margin = (TILE_SIZE - STRIDE) // 2
    
    img_padded = cv2.copyMakeBorder(image, margin, margin + pad_h, margin, margin + pad_w, cv2.BORDER_REFLECT)
    mask_padded = cv2.copyMakeBorder(mask, margin, margin + pad_h, margin, margin + pad_w, cv2.BORDER_CONSTANT, value=0)
    
    out_padded = img_padded.copy()
    
    print(f"   Tiled Inpainting: {img_padded.shape[1]}x{img_padded.shape[0]} grid...")
    
    for y in range(0, H + pad_h, STRIDE):
        for x in range(0, W + pad_w, STRIDE):
            ty1, ty2 = y, y + TILE_SIZE
            tx1, tx2 = x, x + TILE_SIZE
            
            tile_img = img_padded[ty1:ty2, tx1:tx2]
            tile_mask = mask_padded[ty1:ty2, tx1:tx2]
            
            if tile_mask.sum() == 0: continue
                
            t_img = torch.from_numpy(tile_img).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            t_mask = torch.from_numpy(tile_mask).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                inpainted = model(t_img, t_mask)
                if isinstance(inpainted, (tuple, list)): inpainted = inpainted[0]
            
            out_tensor = inpainted[0].permute(1, 2, 0).cpu().numpy()
            out_tile = np.clip(out_tensor * 255, 0, 255).astype(np.uint8)
            
            # Extract Center
            cy1, cy2 = margin, margin + STRIDE
            cx1, cx2 = margin, margin + STRIDE
            center_patch = out_tile[cy1:cy2, cx1:cx2]
            
            # Place
            oy1, oy2 = y + margin, y + margin + STRIDE
            ox1, ox2 = x + margin, x + margin + STRIDE
            out_padded[oy1:oy2, ox1:ox2] = center_patch

    final_result = out_padded[margin:margin+H, margin:margin+W]
    return final_result

def run_pro_pipeline(image_path, output_name="final_result_pro.jpg"):
    deeplab, clipseg, lama = load_models()
    if not deeplab: return
    
    print(f"\nProcessing: {image_path}")
    original_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Hybrid Masking
    print("   Generating Semantic Mask...")
    mask = get_hybrid_mask(deeplab, clipseg, image_rgb)
    
    # 2. Dynamic Expansion (The Fix)
    # 2% expansion (e.g., 80px for a 4000px image)
    dilated_mask = expand_mask_dynamically(mask, scale_percent=0.02)
    
    print(f"   Pollution Coverage: {dilated_mask.mean()*100:.2f}%")
    
    # 3. Tiled Inpainting
    print("   Running High-Res Inpainting (Tiled)...")
    result_rgb = tiled_inpainting(image_rgb, dilated_mask, lama, DEVICE_1)
    
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    # Visualize: Show Original | Red Mask Overlay | Result
    overlay = original_bgr.copy()
    overlay[dilated_mask == 1] = [0, 0, 255]
    vis = cv2.addWeighted(original_bgr, 0.7, overlay, 0.3, 0)
    
    H, W = original_bgr.shape[:2]
    # Resize for display convenience only
    combined = np.hstack([
        cv2.resize(vis, (1024, int(1024*H/W))), 
        cv2.resize(result_bgr, (1024, int(1024*H/W)))
    ])
    
    cv2.imwrite(output_name, combined)
    print(f"✅ Success! Pro result saved to {output_name}")

if __name__ == "__main__":
    target = "image_ce26f8.jpg"
    if os.path.exists(target):
        run_pro_pipeline(target)
    else:
        val_images = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
        if val_images: run_pro_pipeline(val_images[3])
        else: print("No images found.")