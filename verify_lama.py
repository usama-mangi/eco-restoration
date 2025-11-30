import torch
import cv2
import numpy as np
import os

def run_test():
    print("--- Testing Clean LaMa (JIT) ---")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "big-lama.pt"
    
    if not os.path.exists(model_path):
        print("❌ Error: big-lama.pt not found.")
        return

    # 1. Load Model (One Line, No Configs!)
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        model.to(device)
        print("✅ LaMa Model Loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print("Generating test input...")
    # Image: 512x512 RGB
    img = np.zeros((512, 512, 3), dtype=np.uint8) + 100 # Grey
    cv2.circle(img, (256, 256), 50, (0, 0, 255), -1)    # Red dot
    
    # Mask: 512x512 Binary (0 or 1)
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (256, 256), 55, (255), -1)         # White circle on black
    
    # 3. Preprocess & Shape
    # Image -> (1, 3, 512, 512)
    img_t = img.astype(np.float32) / 255.0
    img_t = np.transpose(img_t, (2, 0, 1)) # HWC -> CHW
    img_t = torch.from_numpy(img_t).unsqueeze(0).to(device)
    
    # Mask -> (1, 1, 512, 512)
    # Note: LaMa expects mask as float, where 1.0 = hole to fill
    mask_t = mask.astype(np.float32) / 255.0
    mask_t = torch.from_numpy(mask_t).unsqueeze(0).unsqueeze(0).to(device)
    
    # 4. Inference
    print("Running Inference...")
    with torch.no_grad():
        # FIX: Pass as two separate arguments
        output = model(img_t, mask_t)
        
        # JIT model usually returns the raw tensor
        if isinstance(output, (tuple, list)):
            output = output[0]
    
    # 5. Save
    output_np = output[0].permute(1, 2, 0).cpu().numpy()
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("clean_lama_result.jpg", output_bgr)
    print("✅ Test Successful. Saved 'clean_lama_result.jpg'")

if __name__ == "__main__":
    run_test()