"""
One-time script to build global grids at 0.25 degree resolution:
  data/processed/land_mask_025.npy      — (720, 1440) bool, True = land
  data/processed/dist_to_coast_025.npy  — (720, 1440) float32
                                          ocean cells: distance to nearest land
                                          land cells:  0.0
  data/processed/dist_inland_025.npy    — (720, 1440) float32
                                          land cells:  distance to nearest ocean
                                          ocean cells: 0.0

Together, dist_to_coast and dist_inland give a unified signed distance:
  over ocean -> dist_to_coast tells how close to land (approaching landfall)
  over land  -> dist_inland tells how far from the coast (post-landfall decay)

Uses global-land-mask to generate the binary land mask, then
scipy.ndimage.distance_transform_edt for both distance grids.

Run once before features.py:
  python scripts/build_land_grids.py

Takes ~2-3 minutes total.
"""

import os
import sys
import numpy as np

try:
    from global_land_mask import globe
except ImportError:
    sys.exit("global-land-mask not installed -- run: pip install global-land-mask")

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    sys.exit("scipy not installed -- run: pip install scipy")

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

MASK_PATH   = os.path.join(PROC_DIR, "land_mask_025.npy")
DIST_PATH   = os.path.join(PROC_DIR, "dist_to_coast_025.npy")
INLAND_PATH = os.path.join(PROC_DIR, "dist_inland_025.npy")

RESOLUTION = 0.25   # degrees


def build_grids():
    # ── 1. Land mask ──────────────────────────────────────────────────────────
    print("Building land mask (720 x 1440 at 0.25 deg) ...")
    lats = np.arange(-90,  90, RESOLUTION)    # 720 points
    lons = np.arange(-180, 180, RESOLUTION)   # 1440 points

    lon_grid, lat_grid = np.meshgrid(lons, lats)   # (720, 1440)
    print("  Querying global_land_mask ... (~30-60 seconds)")
    land_mask = globe.is_land(lat_grid, lon_grid).astype(bool)
    np.save(MASK_PATH, land_mask)
    print(f"  Saved: {MASK_PATH}  shape={land_mask.shape}  "
          f"land fraction={land_mask.mean()*100:.1f}%")

    # ── 2. Ocean-to-land distance (dist_to_coast) ─────────────────────────────
    # EDT on ~land_mask: each ocean cell gets its distance to the nearest
    # land cell in grid units. Multiply by RESOLUTION for degrees.
    # Land cells get 0.0.
    # Use: how close is an ocean storm to the coastline?
    print("Computing ocean-to-land distance grid (dist_to_coast) ...")
    dist_to_coast = distance_transform_edt(~land_mask) * RESOLUTION
    dist_to_coast = dist_to_coast.astype(np.float32)
    np.save(DIST_PATH, dist_to_coast)
    print(f"  Saved: {DIST_PATH}  "
          f"max={dist_to_coast.max():.1f} deg  "
          f"(ocean cells only; land=0)")

    # ── 3. Land-to-ocean distance (dist_inland) ───────────────────────────────
    # EDT on land_mask: each land cell gets its distance to the nearest
    # ocean cell in grid units. Multiply by RESOLUTION for degrees.
    # Ocean cells get 0.0.
    # Use: how far inland has a storm penetrated post-landfall?
    print("Computing land-to-ocean distance grid (dist_inland) ...")
    dist_inland = distance_transform_edt(land_mask) * RESOLUTION
    dist_inland = dist_inland.astype(np.float32)
    np.save(INLAND_PATH, dist_inland)
    print(f"  Saved: {INLAND_PATH}  "
          f"max={dist_inland.max():.1f} deg  "
          f"(land cells only; ocean=0)")

    print("\nDone. All grids ready for features.py.")
    print(f"  Land mask     : {MASK_PATH}")
    print(f"  Dist to coast : {DIST_PATH}")
    print(f"  Dist inland   : {INLAND_PATH}")
    print(
        "\nInterpretation in features.py:\n"
        "  over ocean -> dist_to_coast > 0 (degrees to nearest land)\n"
        "  over land  -> dist_inland   > 0 (degrees from nearest ocean)\n"
        "  signed_dist = dist_inland - dist_to_coast\n"
        "                negative = ocean side, positive = land side"
    )


if __name__ == "__main__":
    build_grids()