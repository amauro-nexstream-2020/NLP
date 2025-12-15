import os
import time
from huggingface_hub import snapshot_download
from tqdm import tqdm

DATA_DIR = "/data/fineweb"
TARGET_SIZE_GB = 4000
REPO_ID = "HuggingFaceFW/fineweb"

def get_dir_size_gb(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)

def main():
    print(f"Starting download of {REPO_ID} to {DATA_DIR}...")
    print(f"Target size: {TARGET_SIZE_GB} GB")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Selecting specific subsets (snapshots) to reach ~4TB
    # FineWeb is organized by dumps. Each dump is massive.
    # We will download specific folders corresponding to recent high-quality dumps.
    
    # Approximate sizes:
    # CC-MAIN-2024-10: ~45TB total -> way too big.
    # We need to download specific 'data' chunks.
    
    # Strategy: Download the 'sample-100BT' first (high quality, small)
    # Then download specific data shards from the main dataset until capacity.
    
    print("Step 1: Downloading sample-100BT (High Quality, ~50GB)...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns="sample/100BT/*",
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=16
    )
    
    current_size = get_dir_size_gb(DATA_DIR)
    print(f"Current size: {current_size:.2f} GB")
    
    # Now download chunk by chunk from the main data folder
    # We'll target the 'data' folder which contains parquet files.
    # Since we can't easily list and pick files with snapshot_download partial,
    # we'll use a specific allow_pattern for subsets.
    
    # CC-MAIN-2023-50 is a good candidate.
    print("Step 2: Downloading CC-MAIN-2023-50 subset...")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="data/CC-MAIN-2023-50/*",
            local_dir=DATA_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=16
        )
    except Exception as e:
        print(f"Stopped or error: {e}")

    current_size = get_dir_size_gb(DATA_DIR)
    print(f"Current size: {current_size:.2f} GB")
    
    if current_size < TARGET_SIZE_GB:
        print("Step 3: Downloading CC-MAIN-2024-10 subset (partial)...")
        # download another dump
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="data/CC-MAIN-2024-10/*",
            local_dir=DATA_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=16
        )

    final_size = get_dir_size_gb(DATA_DIR)
    print(f"Download complete! Final size: {final_size:.2f} GB")

if __name__ == "__main__":
    main()
