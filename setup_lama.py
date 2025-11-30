import os
import shutil
import subprocess

def setup():
    print("--- Setting up Clean LaMa (TorchScript) ---")
    
    # 1. Nuke the old repo (It caused all the pain)
    if os.path.exists("lama"):
        try:
            shutil.rmtree("lama")
            print("Deleted broken 'lama' repository.")
        except PermissionError:
            print("Warning: Could not delete 'lama' folder (files in use?). Ignoring it.")

    # 2. Download the Standalone Model
    # Source: Official LaMa JIT export hosted on GitHub/HuggingFace mirrors
    # We use a reliable mirror for the JIT file.
    url = "https://huggingface.co/fashn-ai/LaMa/resolve/main/big-lama.pt"
    target_path = "big-lama.pt"
    
    if not os.path.exists(target_path):
        print(f"Downloading {target_path}...")
        subprocess.run(["curl", "-L", "-o", target_path, url], check=True)
        print("✅ Download Complete.")
    else:
        print("✅ big-lama.pt already exists.")

if __name__ == "__main__":
    setup()