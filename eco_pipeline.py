import torch
import cv2
import numpy as np
import random
import glob
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

# --- Configuration ---
YOLO_PATH = "logs/yolov8l_1280_finetune/weights/best.pt"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth" 
LAMA_PATH = "big-lama.pt"

# Device Split
DEVICE_0 = "cuda:0"
DEVICE_1 = "cuda:1"

def load_models():
    print(f"--- Loading Production Engines (Smart-Guided) ---")
    
    # 1. Load YOLO
    yolo = YOLO(YOLO_PATH)
    print("✅ YOLOv8 (1280px) Loaded")
    
    # 2. Load BASE SAM (More robust for general edges)
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

def run_inference(image_path, output_name="final_result_v3.jpg"):
    yolo, sam_predictor, lama = load_models()
    
    print(f"\nProcessing: {image_path}")
    original_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. YOLO Detection + Coarse Mask
    # retinamask=True gives better quality masks from YOLO
    results = yolo(image_rgb, conf=0.25, imgsz=1280, retina_masks=True, verbose=False, device=0)[0]
    
    if not results.boxes or not results.masks:
        print("ℹ️ No pollution detected.")
        return

    boxes = results.boxes.xyxy.cpu().numpy()
    yolo_masks = results.masks.data.cpu().numpy() # (N, H, W) float
    
    print(f"   Detected {len(boxes)} objects. Refining...")
    
    sam_predictor.set_image(image_rgb)
    final_composite_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    
    for i, box in enumerate(boxes):
        # Resize YOLO mask to match image if needed (retina_masks handles this usually but let's be safe)
        y_mask_raw = yolo_masks[i]
        if y_mask_raw.shape != image_rgb.shape[:2]:
             y_mask_raw = cv2.resize(y_mask_raw, (image_rgb.shape[1], image_rgb.shape[0]))
        
        yolo_binary_mask = (y_mask_raw > 0.5).astype(np.uint8)
        
        # --- STRATEGY: MASK-GUIDED SAM ---
        # 1. Box Prompt
        box_tensor = torch.tensor(box[None, :], device=DEVICE_0)
        transformed_box = sam_predictor.transform.apply_boxes_torch(box_tensor, image_rgb.shape[:2])
        
        # 2. Mask Prompt (Low Res Logits)
        # SAM expects 256x256 logits. We must downsample YOLO mask.
        y_mask_tensor = torch.tensor(y_mask_raw, device=DEVICE_0).float().unsqueeze(0).unsqueeze(0) # 1,1,H,W
        y_mask_lowres = F.interpolate(y_mask_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        
        masks, scores, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_box,
            mask_input=y_mask_lowres, # GUIDE SAM
            multimask_output=False,
        )
        
        sam_mask = (masks[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        
        # --- SAFETY CHECK ---
        # If SAM diverges too much from YOLO (IoU < 0.5), it likely failed.
        iou = calculate_iou(yolo_binary_mask, sam_mask)
        
        if iou > 0.5:
            # SAM is reasonable, use it (it likely improved edges)
            final_composite_mask = np.logical_or(final_composite_mask, sam_mask)
        else:
            # SAM failed, fallback to YOLO
            print(f"   [Object {i}] SAM Diverged (IoU={iou:.2f}). Fallback to YOLO.")
            final_composite_mask = np.logical_or(final_composite_mask, yolo_binary_mask)

    final_composite_mask = final_composite_mask.astype(np.uint8)
    
    # 4. Dilate (Margin)
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
    print(f"✅ Success! Saved robust result to {output_name}")

if __name__ == "__main__":
    val_images = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
    if val_images:
        target_image = random.choice(val_images)
        run_inference(target_image)
    else:
        print("No validation images found.")