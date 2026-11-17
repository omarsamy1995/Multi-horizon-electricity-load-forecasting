"""
PJM Hourly Energy Consumption Dataset Downloader
Dataset source: Kaggle - Rob Mulla (robikscube/hourly-energy-consumption)
Target file: AEP_hourly.csv (American Electric Power hourly load in MW)
"""

import os
import sys
import argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATASET_NAME = "robikscube/hourly-energy-consumption"
DEFAULT_TARGET_FILE = "AEP_hourly.csv"
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")

def verify_dataset(file_name: str = DEFAULT_TARGET_FILE) -> bool:
    target_path = os.path.join(RAW_DATA_DIR, file_name)
    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[FOUND] {file_name} already exists in data/raw/ ({size_mb:.2f} MB).")
        return True
    return False

def download_via_kaggle(file_name: str = DEFAULT_TARGET_FILE):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    target_path = os.path.join(RAW_DATA_DIR, file_name)

    if verify_dataset(file_name):
        print("[INFO] Dataset is ready. Skipping download.")
        return

    print(f"[INFO] Attempting to fetch '{file_name}' from Kaggle dataset '{DATASET_NAME}'...")
    try:
        import kagglehub
        path = kagglehub.dataset_download(DATASET_NAME)
        src_file = os.path.join(path, file_name)
        if os.path.exists(src_file):
            import shutil
            shutil.copy(src_file, target_path)
            print(f"[SUCCESS] Downloaded and copied {file_name} to data/raw/")
            return
    except Exception as e:
        print(f"[WARNING] kagglehub download failed: {e}")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(DATASET_NAME, file_name, path=RAW_DATA_DIR)
        print(f"[SUCCESS] Downloaded {file_name} via Kaggle API to data/raw/")
        return
    except Exception as e:
        print(f"[WARNING] Kaggle API download failed: {e}")

    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption")
    print(f"2. Download '{file_name}' (or 'DUQ_hourly.csv' for out-of-distribution evaluation).")
    print(f"3. Place the CSV inside the 'data/raw/' directory: data/raw/{file_name}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or verify PJM hourly dataset.")
    parser.add_argument("--file", type=str, default=DEFAULT_TARGET_FILE, help="CSV file name to download/verify")
    args = parser.parse_args()
    download_via_kaggle(args.file)
