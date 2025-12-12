import torch
import cv2
import numpy as np
import os
import glob
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F

# --- Configuration ---
DEEPLAB_PATH = "deeplab_best.pth"
CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
# MANUAL PATH
SD_CHECKPOINT_PATH = "sd-v1-5-inpainting.safetensors"
ENCODER = "efficientnet-b4" 

DEVICE_SEG = "cuda:0" 
DEVICE_GEN = "cuda:1"

PROMPTS_SEG = ["trash", "garbage", "rubbish pile", "plastic waste", "litter", "heaps of trash", ""]
PROMPT_INPAINT = "high resolution, 8k, realistic image with the segmented parts inpainted with consistent background that makes the overall image consistent and coherent"
NEGATIVE_PROMPT = "trash, rubbish, pollution, blur, artifacts, low resolution"

def load_models():
    print(f"--- Loading Generative Engines (Manual Mode) ---")
    
    # 1. DeepLabV3+
    deeplab = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=None, 
        classes=1, 
        activation="sigmoid",
    )
    if os.path.exists(DEEPLAB_PATH):
        state_dict = torch.load(DEEPLAB_PATH, map_location=DEVICE_SEG)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        deeplab.load_state_dict(new_state_dict)
    deeplab.to(DEVICE_SEG)
    deeplab.eval()
    print("✅ DeepLabV3+ Loaded")
    
    # 2. CLIPSeg
    processor = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
    clipseg = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).to(DEVICE_SEG)
    print("✅ CLIPSeg Loaded")
    
    # 3. Stable Diffusion 2 (Manual Load)
    if not os.path.exists(SD_CHECKPOINT_PATH):
        print(f"❌ Error: {SD_CHECKPOINT_PATH} not found. Please download it manually.")
        return None, None, None
        
    print(f"   Loading SD2 from {SD_CHECKPOINT_PATH}...")
    pipe = StableDiffusionInpaintPipeline.from_single_file(
        SD_CHECKPOINT_PATH, 
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to(DEVICE_GEN)
    pipe.enable_attention_slicing()
    print("✅ Stable Diffusion 2 Loaded")
    
    return deeplab, (processor, clipseg), pipe

def get_hybrid_mask(deeplab, clipseg_engine, image_rgb):
    # DeepLab
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE_SEG)
    with torch.no_grad():
        mask_pred = deeplab(input_tensor)
    mask_512 = mask_pred[0, 0].cpu().numpy()
    mask_dl = cv2.resize(mask_512, (image_rgb.shape[1], image_rgb.shape[0])) > 0.5

    # CLIPSeg
    processor, model = clipseg_engine
    inputs = processor(
        text=PROMPTS_SEG, 
        images=[image_rgb] * len(PROMPTS_SEG), 
        padding="max_length", 
        return_tensors="pt"
    ).to(DEVICE_SEG)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)
    preds = F.interpolate(preds, size=(image_rgb.shape[0], image_rgb.shape[1]), mode="bilinear", align_corners=False)
    preds = torch.sigmoid(preds)
    combined = torch.max(preds, dim=0)[0][0].cpu().numpy()
    mask_cs = combined > 0.4

    final_mask = np.logical_or(mask_dl, mask_cs).astype(np.uint8)
    return final_mask

def smart_sd_inpainting(image_rgb, mask, pipe):
    H, W = image_rgb.shape[:2]
    
    # Resize if massive
    max_dim = 1536 
    scale = 1.0
    if max(H, W) > max_dim:
        scale = max_dim / max(H, W)
        nH, nW = int(H * scale), int(W * scale)
        nH = (nH // 8) * 8
        nW = (nW // 8) * 8
        
        img_in = cv2.resize(image_rgb, (nW, nH))
        mask_in = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
    else:
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        img_in = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask_in = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        nH, nW = img_in.shape[:2]

    from PIL import Image
    pil_img = Image.fromarray(img_in)
    pil_mask = Image.fromarray(mask_in * 255)
    
    print("   Generating Texture...")
    with torch.no_grad():
        output = pipe(
            prompt=PROMPT_INPAINT, 
            negative_prompt=NEGATIVE_PROMPT,
            image=pil_img, 
            mask_image=pil_mask, 
            num_inference_steps=20,
            strength=0.99,
            guidance_scale=7.5
        ).images[0]
        
    out_np = np.array(output)
    
    if scale != 1.0:
        out_np = cv2.resize(out_np, (W, H))
    else:
        out_np = out_np[:H, :W, :]
        
    return out_np

def run_sd_pipeline(image_path, output_name="final_result_sd.jpg"):
    deeplab, clipseg, pipe = load_models()
    if not deeplab: return
    
    print(f"\nProcessing: {image_path}")
    original_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    print("   Generating Semantic Mask...")
    mask = get_hybrid_mask(deeplab, clipseg, image_rgb)
    
    # Dynamic Dilation
    h, w = mask.shape[:2]
    k_size = int(max(h, w) * 0.02)
    if k_size % 2 == 0: k_size += 1
    k_size = max(15, k_size)
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    print(f"   Pollution Coverage: {dilated_mask.mean()*100:.2f}%")
    
    print("   Running Stable Diffusion Inpainting...")
    result_rgb = smart_sd_inpainting(image_rgb, dilated_mask, pipe)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    
    overlay = original_bgr.copy()
    overlay[dilated_mask == 1] = [0, 0, 255]
    vis = cv2.addWeighted(original_bgr, 0.7, overlay, 0.3, 0)
    
    combined = np.hstack([
        cv2.resize(vis, (1024, int(1024*image_rgb.shape[0]/image_rgb.shape[1]))), 
        cv2.resize(result_bgr, (1024, int(1024*image_rgb.shape[0]/image_rgb.shape[1])))
    ])
    
    cv2.imwrite(output_name, combined)
    print(f"✅ Success! Generative result saved to {output_name}")

if __name__ == "__main__":
    target = "image_ce26f8.jpg"
    if os.path.exists(target):
        run_sd_pipeline(target)
    else:
        val_images = glob.glob("datasets/FreeInternetLitterTest/*.jpg")
        if val_images: run_sd_pipeline(val_images[4])
        else: print("No images found.")
