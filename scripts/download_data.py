"""
Download all three TCND pillars (Data_1d, Data_3d, Env-Data) for WP, NA, EP.

TCND dataset structure:
  Data_1d/  — IBTrACS tabular records (Location, Pressure, Wind)
  Data_3d/  — 20°x20° gridded spatial patches @ 0.25°, 6-hour intervals
  Env-Data/ — Pre-calculated physical shortcuts and historical momentum

Usage:
    pip install -r requirements.txt
    python scripts/download_data.py
"""

import os
import shutil
import zipfile

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1CIVnMCakIoDF7_2xEMWsclKRz8ICAnXq"

TARGET_BASINS   = ["WP", "NA", "EP"]
TARGET_PILLARS  = ["Data_1d", "Data_3d", "Env-Data"]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TMP_DIR = os.path.join(RAW_DIR, "_tmp")


def download():
    os.makedirs(TMP_DIR, exist_ok=True)
    
    # Check if the directory already has files in it
    if os.listdir(TMP_DIR): 
        print(f"Files already exist in {TMP_DIR}. Skipping download.")
        return
    elif os.listdir(RAW_DIR): 
        print(f"Files already exist in {RAW_DIR}. Skipping download.")
        return

    print(f"Downloading TCND from Google Drive...\n{FOLDER_URL}\n")
    gdown.download_folder(FOLDER_URL, output=TMP_DIR, quiet=False, remaining_ok=True)


def extract():
    extracted = []
    # Walk through the download directory
    for root, _, files in os.walk(TMP_DIR):
        for fname in files:
            # 1. FIND THE PILLAR (Check if pillar string exists in filename)
            # We use .lower() and replace('_', '') to handle 'Data_1d' vs 'Data1D'
            pillar = None
            for p in TARGET_PILLARS:
                clean_p = p.lower().replace('_', '')
                clean_f = fname.lower().replace('_', '')
                if clean_p in clean_f:
                    pillar = p
                    break
            
            if not pillar:
                continue # Not a file we care about (like TCND_test.zip)

            # 2. FIND THE BASIN (Check if 'NA', 'EP', or 'WP' is in the filename)
            basin = None
            fname_upper = fname.upper()
            for b in TARGET_BASINS:
                if b in fname_upper:
                    basin = b
                    break
            
            # Special case: If it's Env-Data or Data1D, it might not have a basin in the name
            # You might want to default these to 'GLOBAL' or skip them
            if not basin:
                basin = "GLOBAL" 

            # 3. CONSTRUCT PATHS AND EXTRACT
            src = os.path.join(root, fname)
            out_dir = os.path.join(RAW_DIR, pillar, basin)
            os.makedirs(out_dir, exist_ok=True)

            if fname.endswith(".zip"):
                print(f"📦 Extracting {fname} to {pillar}/{basin}...")
                with zipfile.ZipFile(src, "r") as z:
                    z.extractall(out_dir)
            else:
                print(f"📄 Copying {fname} to {pillar}/{basin}...")
                shutil.copy2(src, os.path.join(out_dir, fname))

            extracted.append(out_dir)

    return extracted


def cleanup():
    shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pillars", nargs="+",
                        default=TARGET_PILLARS,
                        choices=TARGET_PILLARS,
                        help="Which data pillars to download (default: all)")
    parser.add_argument("--basins", nargs="+",
                        default=TARGET_BASINS,
                        choices=TARGET_BASINS,
                        help="Which basins to keep (default: all)")
    cli = parser.parse_args()

    # Apply CLI filters
    TARGET_PILLARS[:] = cli.pillars
    TARGET_BASINS[:]  = cli.basins

    download()
    results = extract()
    cleanup()

    if not results:
        print("\nNo matching files found. Verify the Drive folder structure.")
    else:
        print(f"\nDone — {len(results)} dataset(s) ready.")
        print("Next step: python scripts/features.py")
