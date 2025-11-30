import torch
import cv2
import numpy as np
import os
import glob
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# --- Configuration ---
DEEPLAB_PATH = "deeplab_best_efficientnet.pth"
LAMA_PATH = "big-lama.pt"
# CRITICAL: Must match the training script exactly
ENCODER = "efficientnet-b4" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INP = "cuda:1" # LaMa on GPU 1 if possible

def load_models():
    print(f"--- Loading Semantic Engines ({ENCODER}) ---")
    
    # 1. DeepLabV3+ (EfficientNet-B4)
    model_seg = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=None, # Loading custom weights, not ImageNet
        classes=1, 
        activation="sigmoid",
    )
    
    if os.path.exists(DEEPLAB_PATH):
        # Load weights
        state_dict = torch.load(DEEPLAB_PATH, map_location=DEVICE)
        
        # Strip 'module.' prefix if trained with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model_seg.load_state_dict(new_state_dict)
        print(f"✅ DeepLabV3+ ({ENCODER}) Loaded")
    else:
        print(f"❌ Error: Weights not found at {DEEPLAB_PATH}")
        return None, None

    model_seg.to(DEVICE)
    model_seg.eval()
    
    # 2. LaMa Inpainting
    try:
        lama = torch.jit.load(LAMA_PATH, map_location=DEVICE_INP)
        lama.eval()
        print("✅ LaMa Loaded")
    except Exception as e:
        print(f"❌ Error loading LaMa: {e}")
        return None, None
    
    return model_seg, lama

def preprocess_image(image):
    # Standardize to 512x512 for the model
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform(image=image)['image'].unsqueeze(0)

def run_inference(image_path, output_name="final_result_efficient.jpg"):
    model_seg, lama = load_models()
    if not model_seg: return
    
    print(f"\nProcessing: {image_path}")
    original_bgr = cv2.imread(image_path)
    if original_bgr is None: return
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    
    # 1. Segmentation (Global Context)
    input_tensor = preprocess_image(image_rgb).to(DEVICE)
    
    with torch.no_grad():
        mask_pred = model_seg(input_tensor) # Output: (1, 1, 512, 512)
        
    # Resize mask back to full resolution
    mask_512 = mask_pred[0, 0].cpu().numpy()
    mask_resized = cv2.resize(mask_512, (W, H))
    
    # Thresholding
    # Tversky loss pushes values high, so 0.5 is safe.
    # Lower to 0.3 if you want to be extremely aggressive.
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        print("ℹ️ No pollution detected.")
        return
        
    print(f"   Pollution Coverage: {binary_mask.mean()*100:.2f}%")

    # 2. Dilation (Margin)
    # Expand slightly to cover "fuzzy" edges of heaps
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    
    # 3. LaMa Inpainting
    print("   Running Inpainting...")
    
    # Resize for LaMa if image is massive (OOM protection)
    max_dim = 2048
    scale = 1.0
    if max(H, W) > max_dim:
        scale = max_dim / max(H, W)
        iH, iW = int(H * scale), int(W * scale)
        img_in = cv2.resize(image_rgb, (iW, iH))
        mask_in = cv2.resize(dilated_mask, (iW, iH), interpolation=cv2.INTER_NEAREST)
    else:
        img_in = image_rgb
        mask_in = dilated_mask

    img_t = torch.from_numpy(img_in).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE_INP)
    mask_t = torch.from_numpy(mask_in).float().unsqueeze(0).unsqueeze(0).to(DEVICE_INP)
    
    # Pad to divisible by 8
    pad_h = (8 - img_t.shape[2] % 8) % 8
    pad_w = (8 - img_t.shape[3] % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
        mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out_pad = lama(img_t, mask_t)
        if isinstance(out_pad, (tuple, list)): out_pad = out_pad[0]

    h_end = img_t.shape[2] - pad_h
    w_end = img_t.shape[3] - pad_w
    out_crop = out_pad[:, :, :h_end, :w_end]
    
    out_np = out_crop[0].permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    
    if scale != 1.0:
        out_np = cv2.resize(out_np, (W, H))

    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    # Visualization
    # Red overlay for mask
    overlay = original_bgr.copy()
    overlay[dilated_mask == 1] = [0, 0, 255]
    vis = cv2.addWeighted(original_bgr, 0.7, overlay, 0.3, 0)
    
    combined = np.hstack([
        cv2.resize(vis, (1024, int(1024*H/W))), 
        cv2.resize(out_bgr, (1024, int(1024*H/W)))
    ])
    
    cv2.imwrite(output_name, combined)
    print(f"✅ Success! Result saved to {output_name}")

if __name__ == "__main__":
    # Prioritize your uploaded heap image
    heap_img = "image_0424cc.jpg" 
    if os.path.exists(heap_img):
        run_inference(heap_img)
    else:
        # Fallback
        val_images = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
        if val_images: run_inference(val_images[6])
        else: print("No images found.")