import torch
import ultralytics
import segment_anything
import diffusers
import cv2
import xformers

def verify_system():
    print("--- EcoRestoration System Verification ---")
    
    # 1. Check CUDA & GPUs
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            # Convert bytes to GB
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {name} ({mem:.2f} GB VRAM)")
            
        if gpu_count < 2:
            print("WARNING: You mentioned having two GPUs, but PyTorch only sees one.")
    else:
        print("CRITICAL ERROR: CUDA is not available. Training will not work.")
        return

    # 2. Check Xformers (Critical for 8GB VRAM)
    try:
        import xformers.ops
        print(f"Xformers Available: Yes (Version: {xformers.__version__})")
    except ImportError:
        print("WARNING: Xformers is installed but not loading correctly. Memory usage will be higher.")

    # 3. Check Library Versions
    print(f"\n--- Library Versions ---")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics (YOLO): {ultralytics.__version__}")
    print(f"Diffusers: {diffusers.__version__}")
    
    print("\nSUCCESS: Environment is ready.")

if __name__ == "__main__":
    verify_system()