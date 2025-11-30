import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_zerowaste():
    base_dir = os.path.join("datasets", "raw", "ZeroWaste")
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"--- Setting up ZeroWaste in {base_dir} ---")
    
    # URL derived from official ZeroWaste website (http://ai.bu.edu/zerowaste/)
    # We specifically need "ZeroWaste-f" (foreground) for segmentation
    zip_url = "http://ai.bu.edu/zerowaste/data/zerowaste-f.zip"
    zip_path = os.path.join(base_dir, "zerowaste-f.zip")
    
    # 1. Download
    if not os.path.exists(zip_path):
        print("Downloading ZeroWaste-f (approx 2GB)...")
        try:
            download_file(zip_url, zip_path)
        except Exception as e:
            print(f"Download failed: {e}")
            return
    
    # 2. Extract
    extract_path = os.path.join(base_dir, "data")
    if not os.path.exists(extract_path):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        # Rename the extracted folder to 'data' for consistency
        # ZeroWaste zip usually extracts to a folder named 'zerowaste-f'
        if os.path.exists(os.path.join(base_dir, "zerowaste-f")):
             os.rename(os.path.join(base_dir, "zerowaste-f"), extract_path)
    
    print("SUCCESS: ZeroWaste downloaded.")

if __name__ == "__main__":
    download_zerowaste()