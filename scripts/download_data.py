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
    print(f"Downloading TCND from Google Drive...\n{FOLDER_URL}\n")
    gdown.download_folder(FOLDER_URL, output=TMP_DIR, quiet=False, remaining_ok=True)


def extract():
    """
    Walk tmp download, find files matching target pillars and basins,
    extract to data/raw/<pillar>/<basin>/.
    """
    extracted = []
    for root, _, files in os.walk(TMP_DIR):
        # Determine which pillar this directory belongs to
        pillar = None
        for p in TARGET_PILLARS:
            if p in root:
                pillar = p
                break
        if pillar is None:
            continue

        for fname in files:
            basin = os.path.splitext(fname)[0].upper()
            if basin not in TARGET_BASINS:
                continue

            src     = os.path.join(root, fname)
            out_dir = os.path.join(RAW_DIR, pillar, basin)
            os.makedirs(out_dir, exist_ok=True)

            if fname.endswith(".zip"):
                with zipfile.ZipFile(src, "r") as z:
                    z.extractall(out_dir)
                print(f"  Extracted  {pillar}/{fname} -> data/raw/{pillar}/{basin}/")
            else:
                dst = os.path.join(out_dir, fname)
                shutil.copy2(src, dst)
                print(f"  Copied     {pillar}/{fname} -> data/raw/{pillar}/{basin}/")

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
