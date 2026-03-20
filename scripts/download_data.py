"""
Download TCND Data_1d basin files (WP, NA, EP) from Google Drive.

The TCND dataset is structured into three pillars:
  - Data_1d : IBTrACS tabular records (Location, Pressure, Wind) — used here
  - Data_3d : 20°x20° gridded spatial patches at 0.25°, 6-hour intervals
  - Env-Data: Pre-calculated physical shortcuts and historical momentum

We only download Data_1d for the WP, NA, and EP basins.

Usage:
    pip install -r requirements.txt
    python scripts/download_data.py
"""

import os
import shutil
import zipfile

import gdown

# Root Google Drive folder shared by the dataset
FOLDER_URL = "https://drive.google.com/drive/folders/1CIVnMCakIoDF7_2xEMWsclKRz8ICAnXq"

# Data_1d subfolder — tabular IBTrACS records
DATA_1D_SUBFOLDER = "Data_1d"

# Only these three basins are needed
TARGET_BASINS = ["WP", "NA", "EP"]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TMP_DIR = os.path.join(RAW_DIR, "_tmp")


def download():
    os.makedirs(TMP_DIR, exist_ok=True)
    print(f"Downloading TCND folder from Google Drive...")
    print(f"URL: {FOLDER_URL}\n")

    gdown.download_folder(FOLDER_URL, output=TMP_DIR, quiet=False, remaining_ok=True)


def extract_basins():
    """Walk the downloaded folder, find Data_1d basin zips, extract WP/NA/EP only."""
    moved = []

    for root, dirs, files in os.walk(TMP_DIR):
        # Only look inside Data_1d subfolder
        if DATA_1D_SUBFOLDER not in root:
            continue
        for fname in files:
            basin = fname.replace(".zip", "").upper()
            if basin in TARGET_BASINS:
                src = os.path.join(root, fname)
                dst_zip = os.path.join(RAW_DIR, fname)
                dst_dir = os.path.join(RAW_DIR, basin)

                os.replace(src, dst_zip)
                os.makedirs(dst_dir, exist_ok=True)
                with zipfile.ZipFile(dst_zip, "r") as z:
                    z.extractall(dst_dir)
                os.remove(dst_zip)

                print(f"  Extracted {fname} -> data/raw/{basin}/")
                moved.append(dst_dir)

    return moved


def cleanup():
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    download()
    results = extract_basins()
    cleanup()

    if not results:
        print("\nNo WP/NA/EP files found in Data_1d. Check the folder structure.")
    else:
        print(f"\nDone. {len(results)} basin(s) ready in data/raw/")
        print("Next step: python scripts/analyze.py")
