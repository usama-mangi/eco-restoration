import torch
import cv2
import numpy as np
import os
import base64
import asyncio
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionInpaintPipeline
import torch.nn.functional as F

# --- Configuration ---
# Robust weight finding logic
if os.path.exists("deeplab_best.pth"):
    DEEPLAB_PATH = "deeplab_best.pth"
elif os.path.exists("weights/deeplab_production.pth"):
    DEEPLAB_PATH = "weights/deeplab_production.pth"
else:
    DEEPLAB_PATH = None

SD_PATH = "sd-v1-5-inpainting.safetensors"
if not os.path.exists(SD_PATH) and os.path.exists("weights/sd-v1-5-inpainting.safetensors"):
    SD_PATH = "weights/sd-v1-5-inpainting.safetensors"

CLIPSEG_ID = "CIDAS/clipseg-rd64-refined"
CLIP_ID = "openai/clip-vit-base-patch32"
ENCODER = "efficientnet-b4"

DEVICE_SEG = "cuda:0"
DEVICE_GEN = "cuda:1"

SCENES = ["asphalt road", "concrete pavement", "green grass", "dirt ground",
          "sandy beach", "forest floor", "water surface", "gravel"]

def np_to_b64(img_np):
    """Converts OpenCV image to Base64 string for API response."""
    _, buffer = cv2.imencode('.jpg', img_np)
    return base64.b64encode(buffer).decode('utf-8')

class EcoEngineAPI:
    def __init__(self):
        print("--- 🔌 Initializing API Engine ---")
        self.lock = asyncio.Lock()
        self._load_models()

    def _load_models(self):
        try:
            # 1. DeepLab (Visual Segmenter)
            print("   Loading Segmentation...")
            self.deeplab = smp.DeepLabV3Plus(encoder_name=ENCODER, encoder_weights=None, classes=1, activation="sigmoid")
            if DEEPLAB_PATH and os.path.exists(DEEPLAB_PATH):
                state = torch.load(DEEPLAB_PATH, map_location=DEVICE_SEG)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                self.deeplab.load_state_dict(state)
                self.deeplab.to(DEVICE_SEG).eval()
            else:
                print("❌ Warning: DeepLab weights not found.")

            # 2. CLIPSeg & Scene Classifier
            print("   Loading Semantic Engines...")
            self.clipseg_proc = CLIPSegProcessor.from_pretrained(CLIPSEG_ID)
            self.clipseg = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_ID).to(DEVICE_SEG)
            
            self.clip_proc = CLIPProcessor.from_pretrained(CLIP_ID)
            self.clip = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE_SEG)

            # 3. Inpainter (SD2)
            print("   Loading Generative Engine...")
            if os.path.exists(SD_PATH):
                self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                    SD_PATH, torch_dtype=torch.float16, variant="fp16"
                )
                self.pipe.to(DEVICE_GEN)
                self.pipe.enable_attention_slicing()
            else:
                print(f"❌ Warning: SD2 weights not found at {SD_PATH}")

            print("✅ Engine Ready.")
        except Exception as e:
            print(f"❌ Critical Load Error: {e}")

    def resize_with_aspect_ratio(self, img, mask, max_dim=1536):
        """
        Resizes image while maintaining aspect ratio, then pads to nearest multiple of 8.
        This prevents distortion/stretching artifacts.
        """
        h, w = img.shape[:2]
        
        # 1. Calculate Scale
        scale = min(max_dim / h, max_dim / w)
        if scale >= 1.0:
            scale = 1.0
            new_h, new_w = h, w
        else:
            new_h, new_w = int(h * scale), int(w * scale)
            
        # 2. Resize Content
        if scale != 1.0:
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            img_resized = img
            mask_resized = mask

        # 3. Calculate Padding (Divisible by 8)
        pad_h = (8 - new_h % 8) % 8
        pad_w = (8 - new_w % 8) % 8
        
        # 4. Apply Reflection Padding (Better context for inpainting)
        img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        # Constant padding for mask (0 = protect)
        mask_padded = np.pad(mask_resized, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        return img_padded, mask_padded, (new_h, new_w)

    def get_mask(self, img_rgb):
        H, W = img_rgb.shape[:2]
        
        # DeepLab
        tf = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
        x = tf(image=img_rgb)['image'].unsqueeze(0).to(DEVICE_SEG)
        with torch.no_grad():
            m1 = self.deeplab(x)
        m1 = cv2.resize(m1[0,0].cpu().numpy(), (W, H)) > 0.5

        # CLIPSeg
        prompts = ["trash", "garbage", "rubbish pile", "litter"]
        inps = self.clipseg_proc(text=prompts, images=[img_rgb]*len(prompts), return_tensors="pt", padding=True).to(DEVICE_SEG)
        with torch.no_grad():
            out = self.clipseg(**inps)
        preds = torch.sigmoid(out.logits.unsqueeze(1))
        preds = F.interpolate(preds, size=(H, W), mode="bilinear")
        m2 = torch.max(preds, dim=0)[0][0].cpu().numpy() > 0.4

        return np.logical_or(m1, m2).astype(np.uint8)

    async def process_stream(self, image_bytes):
        async with self.lock:
            try:
                # 1. Decode
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None: raise ValueError("Invalid image")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H_orig, W_orig = rgb.shape[:2]
                
                yield {"event": "status", "data": "Analyzing Context..."}
                
                # 2. Classify
                inputs = self.clip_proc(text=SCENES, images=rgb, return_tensors="pt", padding=True).to(DEVICE_SEG)
                with torch.no_grad():
                    probs = self.clip(**inputs).logits_per_image.softmax(dim=1)
                scene = SCENES[probs.argmax().item()]
                
                yield {"event": "status", "data": f"Detected: {scene}"}
                
                # 3. Masking
                yield {"event": "status", "data": "Segmenting Pollution..."}
                mask = self.get_mask(rgb)
                
                # Dynamic Dilation
                k = max(15, int(max(H_orig, W_orig) * 0.02))
                if k%2==0: k+=1
                dilated = cv2.dilate(mask, np.ones((k,k), np.uint8), iterations=1)
                
                # Preview
                vis = img.copy()
                vis[dilated==1] = [0, 0, 255]
                preview = cv2.addWeighted(img, 0.7, vis, 0.3, 0)
                
                yield {
                    "event": "preview", 
                    "data": {
                        "stage": "mask", 
                        "image": np_to_b64(preview),
                        "coverage": f"{dilated.mean()*100:.1f}%"
                    }
                }

                if dilated.sum() == 0:
                    yield {"event": "complete", "data": {"message": "Clean site detected."}}
                    return

                # 4. Inpainting
                yield {"event": "status", "data": "Generating Restoration..."}
                
                # Correct Resizing (No Distortion)
                img_in, mask_in, valid_dims = self.resize_with_aspect_ratio(rgb, dilated)
                
                pil_img = Image.fromarray(img_in)
                pil_mask = Image.fromarray(mask_in * 255)
                
                prompt = f"clean {scene}, high resolution, 8k, photorealistic, seamless texture"
                n_prompt = "trash, rubbish, pollution, blur, artifacts, objects"
                
                with torch.no_grad():
                    out = self.pipe(
                        prompt=prompt, negative_prompt=n_prompt, 
                        image=pil_img, mask_image=pil_mask, 
                        num_inference_steps=25, strength=0.99, guidance_scale=8.0
                    ).images[0]
                
                out_np = np.array(out)
                
                # Crop Padding
                valid_h, valid_w = valid_dims
                out_cropped = out_np[:valid_h, :valid_w, :]
                
                # Resize back to original
                final_bgr = cv2.resize(out_cropped, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
                final_bgr = cv2.cvtColor(final_bgr, cv2.COLOR_RGB2BGR)
                
                yield {
                    "event": "result", 
                    "data": {
                        "image": np_to_b64(final_bgr),
                        "message": "Restoration Complete"
                    }
                }

            except Exception as e:
                yield {"event": "error", "data": str(e)}