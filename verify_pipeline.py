import torch
import cv2
import random
import os
import glob
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# --- Configuration ---
YOLO_PATH = "logs/yolov8l_unified_run/weights/best.pt"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_DECODER_PATH = "sam_waste_decoder.pth"
TEST_IMG_DIR = "datasets/processed/images/val"
OUTPUT_PATH = "pipeline_verification.jpg"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_models():
    print("--- Loading Models ---")
    
    # 1. Load YOLO
    if not os.path.exists(YOLO_PATH):
        raise FileNotFoundError(f"YOLO model not found at {YOLO_PATH}")
    yolo = YOLO(YOLO_PATH)
    print("✅ YOLOv8 Loaded")
    
    # 2. Load SAM (Base + Fine-Tuned Decoder)
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    
    # Load Fine-Tuned Weights
    if os.path.exists(SAM_DECODER_PATH):
        print(f"Loading fine-tuned decoder from {SAM_DECODER_PATH}...")
        decoder_state = torch.load(SAM_DECODER_PATH, map_location="cpu")
        
        # We need to map the state dict keys correctly
        # The training script saved 'sam.mask_decoder.state_dict()', so keys are likely "transformer.layers..."
        # We need to prefix them with "mask_decoder." to load into the main SAM model
        
        sam_dict = sam.state_dict()
        
        # Check if keys already have prefix (depending on how it was saved)
        # Our training script did: torch.save(sam.mask_decoder.state_dict(), ...)
        # So keys are roughly "iou_prediction_head..." WITHOUT "mask_decoder" prefix.
        
        new_state_dict = {}
        for k, v in decoder_state.items():
            new_key = "mask_decoder." + k
            new_state_dict[new_key] = v
            
        # Update SAM state
        sam_dict.update(new_state_dict)
        sam.load_state_dict(sam_dict)
        print("✅ SAM Fine-Tuned Weights Applied")
    else:
        print("⚠️ WARNING: Fine-tuned weights not found. Using Base SAM.")
        
    sam.to(DEVICE)
    predictor = SamPredictor(sam)
    return yolo, predictor

def run_pipeline():
    yolo, sam_predictor = load_models()
    
    # Pick a random test image
    img_files = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    if not img_files:
        print("No validation images found.")
        return
    
    # test_img_path = img_files[0] # Pick first one for consistency or random.choice(img_files)
    test_img_path = random.choice(img_files)
    print(f"Testing on: {test_img_path}")
    
    # 1. Read Image
    image = cv2.imread(test_img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. YOLO Inference (Coarse)
    print("Running YOLO Detection...")
    yolo_results = yolo(image_rgb, conf=0.25)[0]
    boxes = yolo_results.boxes.xyxy.cpu().numpy()
    classes = yolo_results.boxes.cls.cpu().numpy()
    
    if len(boxes) == 0:
        print("No waste detected by YOLO.")
        return

    print(f"YOLO detected {len(boxes)} objects.")
    
    # 3. SAM Refinement (Fine)
    print("Running SAM Refinement...")
    sam_predictor.set_image(image_rgb)
    
    # Transform boxes for SAM
    boxes_tensor = torch.tensor(boxes, device=DEVICE)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_tensor, image_rgb.shape[:2])
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # masks: (N, 1, H, W)
    
    # 4. Visualization
    final_mask = torch.any(masks, dim=0).squeeze().cpu().numpy() # Combine all masks
    
    plt.figure(figsize=(15, 5))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    # YOLO Boxes
    plt.subplot(1, 3, 2)
    plt.title("YOLO Detections")
    res_plotted = yolo_results.plot()
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # SAM Mask
    plt.subplot(1, 3, 3)
    plt.title("SAM Fine-Tuned Mask")
    plt.imshow(image_rgb)
    plt.imshow(final_mask, cmap='jet', alpha=0.5) # Overlay mask
    plt.axis('off')
    
    plt.savefig(OUTPUT_PATH)
    print(f"Pipeline Result saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_pipeline()