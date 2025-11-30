import torch
import cv2
import numpy as np
import random
import glob
import torchvision
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch.nn.functional as F

# --- Configuration ---
YOLO_PATH = "logs/yolov8l_binary_run/weights/best.pt"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
LAMA_PATH = "big-lama.pt"

# AGGRESSIVE SETTINGS
TILE_SIZE = 1280
OVERLAP = 0.50       # 50% Overlap (See everything twice)
CONF_THRESH = 0.05   # Very low threshold (Catch everything)
IOU_THRESH = 0.4     # Merge overlapping duplicates aggressively

DEVICE_0 = "cuda:0"
DEVICE_1 = "cuda:1"

def load_models():
    print("--- Loading Engines (Aggressive Mode) ---")
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

def run_tiled_inference(image_path, output_name="final_result_aggressive.jpg"):
    yolo, sam_predictor, lama = load_models()
    
    print(f"\nProcessing: {image_path}")
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print("❌ Error: Could not read image.")
        return
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    # 1. Tiled Detection (Aggressive)
    print(f"   Slicing image ({W}x{H}) with 50% overlap...")
    tiles = get_tiles(H, W, TILE_SIZE, OVERLAP)
    print(f"   Scanning {len(tiles)} tiles on GPU 0...")

    all_boxes = []
    all_scores = []
    
    for (x, y, w, h) in tiles:
        crop = image_rgb[y:y+h, x:x+w]
        
        # AGGRESSIVE INFERENCE: augment=True enables TTA
        # This flips the image 3 times and averages predictions.
        # It is slower but finds MUCH more trash.
        results = yolo(crop, conf=CONF_THRESH, imgsz=TILE_SIZE, augment=True, verbose=False, device=0)[0]
        
        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            
            for i, box in enumerate(boxes):
                gx1 = box[0] + x
                gy1 = box[1] + y
                gx2 = box[2] + x
                gy2 = box[3] + y
                
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(W, gx2), min(H, gy2)
                
                all_boxes.append([gx1, gy1, gx2, gy2])
                all_scores.append(scores[i])

    if not all_boxes:
        print("ℹ️ No pollution detected.")
        return

    # 2. Global NMS
    print(f"   Merging {len(all_boxes)} detections...")
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, IOU_THRESH)
    final_boxes = boxes_tensor[keep_indices].numpy()
    
    print(f"   Final Object Count: {len(final_boxes)}")

    # 3. SAM Refinement (Batched)
    print("   Refining Masks (SAM)...")
    sam_predictor.set_image(image_rgb)
    final_composite_mask = np.zeros((H, W), dtype=np.uint8)
    
    batch_size = 32
    for i in range(0, len(final_boxes), batch_size):
        batch_boxes = final_boxes[i:i+batch_size]
        
        # Expand boxes slightly (5%) to give SAM context
        expanded_batch = []
        for b in batch_boxes:
            bx1, by1, bx2, by2 = b
            bw, bh = bx2-bx1, by2-by1
            expanded_batch.append([
                max(0, bx1 - bw*0.05),
                max(0, by1 - bh*0.05),
                min(W, bx2 + bw*0.05),
                min(H, by2 + bh*0.05)
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

    # 4. Dilate
    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(final_composite_mask, kernel, iterations=1)

    # 5. LaMa Inpainting
    print("   Running Inpainting (GPU 1)...")
    
    # Downsample if image is huge (>2048) to avoid OOM
    scale = 1.0
    if max(H, W) > 2048:
        scale = 2048 / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        img_resized = cv2.resize(image_rgb, (new_w, new_h))
        mask_resized = cv2.resize(dilated_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        img_resized = image_rgb
        mask_resized = dilated_mask

    img_t = torch.from_numpy(img_resized).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE_1)
    mask_t = torch.from_numpy(mask_resized).float().unsqueeze(0).unsqueeze(0).to(DEVICE_1)
    
    pad_h = (8 - img_t.shape[2] % 8) % 8
    pad_w = (8 - img_t.shape[3] % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
        mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        inpainted_pad = lama(img_t, mask_t)
        if isinstance(inpainted_pad, (tuple, list)): inpainted_pad = inpainted_pad[0]

    h_end = img_t.shape[2] - pad_h
    w_end = img_t.shape[3] - pad_w
    inpainted = inpainted_pad[:, :, :h_end, :w_end]
    
    out_np = inpainted[0].permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    
    if scale != 1.0:
        out_np = cv2.resize(out_np, (W, H))

    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    # Visualization
    vis = original_bgr.copy()
    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Save Side-by-Side
    combined = np.hstack([
        cv2.resize(vis, (1024, int(1024*H/W))), 
        cv2.resize(out_bgr, (1024, int(1024*H/W)))
    ])
    
    cv2.imwrite(output_name, combined)
    print(f"✅ Success! Saved Aggressive result to {output_name}")

if __name__ == "__main__":
    val_images = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
    if val_images:
        run_tiled_inference(random.choice(val_images))
    else:
        print("No validation images found.")