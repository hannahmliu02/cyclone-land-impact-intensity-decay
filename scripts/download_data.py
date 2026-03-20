"""
Download WP, NA, EP cyclone basin data from Google Drive.

Usage:
    pip install gdown
    python scripts/download_data.py
"""

import os
import zipfile
import gdown

# Google Drive folder ID for the dataset
FOLDER_ID = "1CIVnMCakIoDF7_2xEMWsclKRz8ICAnXq"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

# Only download these three basins from subfolder 1D
TARGET_BASINS = ["WP", "NA", "EP"]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def download_folder():
    print(f"Downloading folder contents from Google Drive...")
    print(f"URL: {FOLDER_URL}\n")

    tmp_dir = os.path.join(RAW_DIR, "_tmp_download")
    os.makedirs(tmp_dir, exist_ok=True)

    gdown.download_folder(FOLDER_URL, output=tmp_dir, quiet=False, remaining_ok=True)

    # Move only WP, NA, EP zips to data/raw/
    moved = []
    for basin in TARGET_BASINS:
        for root, _, files in os.walk(tmp_dir):
            for fname in files:
                if fname.upper().startswith(basin) and fname.endswith(".zip"):
                    src = os.path.join(root, fname)
                    dst = os.path.join(RAW_DIR, fname)
                    os.replace(src, dst)
                    moved.append(dst)
                    print(f"  Saved: {dst}")

    # Clean up temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not moved:
        print("\nNo matching basin files found. Check that the folder contains")
        print(f"zip files named WP.zip, NA.zip, EP.zip (or similar) in subfolder 1D.")
    else:
        print(f"\nDownloaded {len(moved)} file(s).")

    return moved


def extract_all():
    print("\nExtracting zip files...")
    for basin in TARGET_BASINS:
        pattern = os.path.join(RAW_DIR, f"{basin}.zip")
        matches = [f for f in os.listdir(RAW_DIR) if f.upper().startswith(basin) and f.endswith(".zip")]
        for fname in matches:
            zip_path = os.path.join(RAW_DIR, fname)
            extract_dir = os.path.join(RAW_DIR, basin)
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
            print(f"  Extracted {fname} -> data/raw/{basin}/")


if __name__ == "__main__":
    files = download_folder()
    if files:
        extract_all()
        print("\nDone. Data is ready in data/raw/")
