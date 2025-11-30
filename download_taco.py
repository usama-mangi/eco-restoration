import os
import subprocess
import sys
import time

def download_taco():
    base_dir = os.path.join("datasets", "raw", "TACO")
    
    # 1. Install requirements using uv (safer/faster)
    req_path = os.path.join(base_dir, "requirements.txt")
    if os.path.exists(req_path):
        print("Ensuring TACO requirements are installed via uv...")
        subprocess.run(["uv", "pip", "install", "-r", req_path], check=False)

    # 2. Loop the download script to catch timeouts
    max_retries = 5
    previous_count = 0

    if previous_count == 0:
        print(f"--- Starting TACO download in {base_dir} ---")
    else:
        print(f"--- Retrying TACO Download in {base_dir} ---")
    
    for i in range(max_retries):
        print(f"\n[Attempt {i+1}/{max_retries}] Running download script...")
        
        try:
            # We use the internal python of the venv
            subprocess.run([sys.executable, "download.py"], cwd=base_dir, check=False)
        except Exception as e:
            print(f"Script crash: {e}")

        # Count images
        img_dir = os.path.join(base_dir, "data")
        current_count = 0
        if os.path.exists(img_dir):
            current_count = sum([len(files) for r, d, files in os.walk(img_dir) 
                               if any(f.endswith(('.jpg', '.png')) for f in files)])
        
        print(f"Current Image Count: {current_count}")
        
        if current_count == previous_count and current_count > 1000:
            print("Download seems stable. Stopping retries.")
            break
            
        previous_count = current_count
        time.sleep(2) # Cooldown for API

    # 3. Final Verification
    print(f"\n--- Final Verification ---")
    print(f"Total Images: {current_count}")
    
    if current_count >= 1400:
        print("SUCCESS: TACO dataset is ready.")
    else:
        print("WARNING: Still missing images. Some Flickr links may be permanently dead.")
        print("We will proceed with what we have if count is > 1000.")

if __name__ == "__main__":
    download_taco()