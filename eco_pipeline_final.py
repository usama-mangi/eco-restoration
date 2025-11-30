import torch
import cv2
import numpy as np
import os
import glob
import random
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

# --- Configuration ---
# POINT TO THE NEW BINARY MODEL
YOLO_PATH = "logs/yolov8l_binary_run/weights/best.pt"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth" 
LAMA_PATH = "big-lama.pt"

# Device Split
DEVICE_0 = "cuda:0"
DEVICE_1 = "cuda:1"

def load_models():
    print(f"--- Loading Engines (Binary Mode) ---")
    
    # 1. Load YOLO (Binary)
    if not os.path.exists(YOLO_PATH):
        print(f"❌ Error: Binary weights not found at {YOLO_PATH}")
        return None, None, None
        
    yolo = YOLO(YOLO_PATH)
    print("✅ YOLOv8 (Binary) Loaded")
    
    # 2. Load BASE SAM (Robust)
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE_0)
    predictor = SamPredictor(sam)
    print("✅ Base SAM Loaded")
    
    # 3. Load LaMa
    lama = torch.jit.load(LAMA_PATH, map_location=DEVICE_1)
    lama.eval()
    print("✅ LaMa Loaded")
    
    return yolo, predictor, lama

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0
    return intersection / union

def pad_to_divisible(img_tensor, mask_tensor, divisor=8):
    b, c, h, w = img_tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return img_tensor, mask_tensor, h, w

def run_inference(image_path, output_name="final_result_binary.jpg"):
    yolo, sam_predictor, lama = load_models()
    if not yolo: return
    
    print(f"\nProcessing: {image_path}")
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print("❌ Error: Could not read image.")
        return
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. YOLO Detection (Binary Class 0)
    # Using retina_masks=True for better initial boundaries
    results = yolo(image_rgb, conf=0.15, imgsz=640, retina_masks=True, verbose=False, device=0)[0]
    
    if not results.boxes:
        print("ℹ️ No pollution detected.")
        return

    boxes = results.boxes.xyxy.cpu().numpy()
    # If masks are missing (rare with retina_masks), we skip
    if results.masks is None:
        print("ℹ️ Boxes found but no masks generated. Skipping.")
        return
    yolo_masks = results.masks.data.cpu().numpy() # (N, H, W)
    
    print(f"   Detected {len(boxes)} trash objects.")
    
    sam_predictor.set_image(image_rgb)
    final_composite_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    
    for i, box in enumerate(boxes):
        # Resize YOLO mask to image size
        y_mask_raw = yolo_masks[i]
        if y_mask_raw.shape != image_rgb.shape[:2]:
             y_mask_raw = cv2.resize(y_mask_raw, (image_rgb.shape[1], image_rgb.shape[0]))
        
        yolo_binary_mask = (y_mask_raw > 0.5).astype(np.uint8)
        
        # --- SAM Refinement (Mask Guided) ---
        box_tensor = torch.tensor(box[None, :], device=DEVICE_0)
        transformed_box = sam_predictor.transform.apply_boxes_torch(box_tensor, image_rgb.shape[:2])
        
        # Downsample YOLO mask for SAM hint (256x256)
        y_mask_tensor = torch.tensor(y_mask_raw, device=DEVICE_0).float().unsqueeze(0).unsqueeze(0)
        y_mask_lowres = F.interpolate(y_mask_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        
        masks, scores, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_box,
            mask_input=y_mask_lowres, # Hint
            multimask_output=False,
        )
        
        sam_mask = (masks[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        
        # --- IoU Safety Check ---
        # If SAM agrees with YOLO (IoU > 0.5), use SAM (it has better edges).
        # If SAM disagrees (hallucinating), fallback to YOLO.
        iou = calculate_iou(yolo_binary_mask, sam_mask)
        
        if iou > 0.5:
            final_composite_mask = np.logical_or(final_composite_mask, sam_mask)
        else:
            print(f"   [Obj {i}] SAM unstable (IoU={iou:.2f}). Fallback to YOLO.")
            final_composite_mask = np.logical_or(final_composite_mask, yolo_binary_mask)

    final_composite_mask = final_composite_mask.astype(np.uint8)
    
    # 4. Dilate Mask (Critical for Inpainting Margin)
    kernel = np.ones((8, 8), np.uint8)
    dilated_mask = cv2.dilate(final_composite_mask, kernel, iterations=1)
    
    # 5. LaMa Inpainting
    img_t = torch.from_numpy(image_rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE_1)
    mask_t = torch.from_numpy(dilated_mask).float().unsqueeze(0).unsqueeze(0).to(DEVICE_1)
    img_pad, mask_pad, orig_h, orig_w = pad_to_divisible(img_t, mask_t)
    
    print("   Running Inpainting...")
    with torch.no_grad():
        inpainted_pad = lama(img_pad, mask_pad)
        if isinstance(inpainted_pad, (tuple, list)): inpainted_pad = inpainted_pad[0]
        
    inpainted = inpainted_pad[:, :, :orig_h, :orig_w]
    
    # 6. Save
    out_np = inpainted[0].permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    
    # Visualization
    vis_boxes = original_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    mask_vis = cv2.cvtColor(dilated_mask * 255, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([vis_boxes, mask_vis, out_bgr])
    
    cv2.imwrite(output_name, combined)
    print(f"✅ Success! Saved final binary result to {output_name}")

if __name__ == "__main__":
    val_images = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
    if val_images:
        target_image = random.choice(val_images)
        run_inference(target_image)
    else:
        print("No validation images found.")