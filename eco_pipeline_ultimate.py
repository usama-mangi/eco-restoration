import torch
import cv2
import numpy as np
import glob
import torchvision
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

# --- Configuration ---
YOLO_PATH = "logs/yolov8l_binary_run/weights/best.pt"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
LAMA_PATH = "big-lama.pt"

# HEAP HUNTER SETTINGS
# 0.5 = See the mountain (Context)
# 1.0 = See the bottle (Detail)
SCALES = [0.5, 1.0]     
TILE_SIZE = 1280
OVERLAP = 0.25
CONF_THRESH = 0.05      # Keep low to catch amorphous heaps
IOU_THRESH = 0.4        # Merge duplicates

DEVICE_0 = "cuda:0"
DEVICE_1 = "cuda:1"

def load_models():
    print("--- Loading Engines (Heap Hunter Mode) ---")
    yolo = YOLO(YOLO_PATH)
    print("✅ YOLOv8 (Binary) Loaded")
    
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE_0)
    predictor = SamPredictor(sam)
    print("✅ Base SAM Loaded")
    
    lama = torch.jit.load(LAMA_PATH, map_location=DEVICE_1)
    lama.eval()
    print("✅ LaMa Loaded")
    return yolo, predictor, lama

def get_tiles(image_h, image_w, tile_size, overlap):
    # Generates coordinates for sliding window
    stride = int(tile_size * (1 - overlap))
    tiles = []
    y = 0
    while y < image_h:
        x = 0
        while x < image_w:
            h = min(tile_size, image_h - y)
            w = min(tile_size, image_w - x)
            tiles.append((x, y, w, h))
            if x + w >= image_w: break
            x += stride
        if y + h >= image_h: break
        y += stride
    return tiles

def run_heap_hunter(image_path, output_name="final_result_heaps.jpg"):
    yolo, sam_predictor, lama = load_models()
    
    print(f"\nProcessing: {image_path}")
    original_bgr = cv2.imread(image_path)
    if original_bgr is None: return
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    all_boxes = []
    all_scores = []

    # --- 1. Multi-Scale Pyramid ---
    for scale in SCALES:
        print(f"   Scanning Scale {scale}x...")
        
        # Resize image
        sH, sW = int(H * scale), int(W * scale)
        img_scaled = cv2.resize(image_rgb, (sW, sH))
        
        # Tile Generation
        # For 0.5x, the image might be smaller than TILE_SIZE. 
        # get_tiles handles this (returns 1 tile).
        tiles = get_tiles(sH, sW, TILE_SIZE, OVERLAP)
        
        for (x, y, w, h) in tiles:
            crop = img_scaled[y:y+h, x:x+w]
            
            # Detect
            # Force imgsz to match crop to avoid padding artifacts
            results = yolo(crop, conf=CONF_THRESH, imgsz=max(32, min(w, h)), verbose=False, device=0)[0]
            
            if results.boxes:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    bx1, by1, bx2, by2 = box
                    
                    # Offset to Scaled Global
                    gx1 = bx1 + x
                    gy1 = by1 + y
                    gx2 = bx2 + x
                    gy2 = by2 + y
                    
                    # Scale back to Original 1.0x
                    gx1 /= scale
                    gy1 /= scale
                    gx2 /= scale
                    gy2 /= scale
                    
                    # Clip
                    gx1, gy1 = max(0, gx1), max(0, gy1)
                    gx2, gy2 = min(W, gx2), min(H, gy2)
                    
                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_scores.append(scores[i])

    if not all_boxes:
        print("ℹ️ No pollution detected.")
        return

    # --- 2. Global NMS ---
    print(f"   Merging {len(all_boxes)} detections from all scales...")
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    
    # Standard NMS
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, IOU_THRESH)
    final_boxes = boxes_tensor[keep_indices].numpy()
    print(f"   Final Object Count: {len(final_boxes)}")

    # --- 3. SAM Refinement (Box Expansion + Mask Guiding) ---
    print("   Refining Masks (SAM)...")
    sam_predictor.set_image(image_rgb)
    final_composite_mask = np.zeros((H, W), dtype=np.uint8)
    
    batch_size = 32
    for i in range(0, len(final_boxes), batch_size):
        batch_boxes = final_boxes[i:i+batch_size]
        
        # 10% Expansion (Heaps usually need context)
        expanded_batch = []
        for b in batch_boxes:
            bx1, by1, bx2, by2 = b
            bw, bh = bx2-bx1, by2-by1
            expanded_batch.append([
                max(0, bx1 - bw*0.10),
                max(0, by1 - bh*0.10),
                min(W, bx2 + bw*0.10),
                min(H, by2 + bh*0.10)
            ])
        expanded_batch = torch.tensor(expanded_batch, device=DEVICE_0)

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(expanded_batch, image_rgb.shape[:2])
        
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        batch_mask = torch.any(masks, dim=0).squeeze().cpu().numpy().astype(np.uint8)
        final_composite_mask = np.logical_or(final_composite_mask, batch_mask)

    final_composite_mask = final_composite_mask.astype(np.uint8)

    # --- 4. LaMa Inpainting ---
    print("   Running Inpainting (GPU 1)...")
    
    # Heavy Dilation (Heaps have messy edges)
    # Expand mask by 20 pixels to ensure clean removal
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(final_composite_mask, kernel, iterations=1)

    # Smart Downsample (Max 2048)
    max_dim = 2048
    scale_factor = 1.0
    if max(H, W) > max_dim:
        scale_factor = max_dim / max(H, W)
        iH, iW = int(H * scale_factor), int(W * scale_factor)
        img_in = cv2.resize(image_rgb, (iW, iH))
        mask_in = cv2.resize(dilated_mask, (iW, iH), interpolation=cv2.INTER_NEAREST)
    else:
        img_in = image_rgb
        mask_in = dilated_mask

    img_t = torch.from_numpy(img_in).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE_1)
    mask_t = torch.from_numpy(mask_in).float().unsqueeze(0).unsqueeze(0).to(DEVICE_1)
    
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
    
    if scale_factor != 1.0:
        out_np = cv2.resize(out_np, (W, H))

    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    # Visualization
    vis = original_bgr.copy()
    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    combined = np.hstack([
        cv2.resize(vis, (1024, int(1024*H/W))), 
        cv2.resize(out_bgr, (1024, int(1024*H/W)))
    ])
    
    cv2.imwrite(output_name, combined)
    print(f"✅ Success! Saved Heap Hunter result to {output_name}")

if __name__ == "__main__":
    val_images = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
    if val_images:
        run_heap_hunter(val_images[0])
    else:
        print("No validation images found.")