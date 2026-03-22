"""
Extract 925 hPa boundary-layer patches from TCND Data_3d NetCDF files.

For each storm in the decay feature matrix, reads T=8 NetCDF files ending at
the reference time, extracts the 925 hPa level for u, v, z and the 2-D SST
field, stacks them into a (T, 4, 81, 81) tensor, normalises per channel, and
saves to data/processed/3d/{storm_id}.npy.

The saved patches are consumed by SpatialDecayDataset in train_ufno.py.

Directory layout expected
─────────────────────────
data/raw/_tmp/Data3D/<BASIN>/<YEAR>/<STORM_NAME>/
    TCND_<NAME>_<YYYYMMDDH0>_sst_z_u_v.nc

Usage
─────
  python scripts/preprocess.py [--basins WP NA EP] [--lookback 8]

Outputs
───────
  data/processed/3d/<storm_id>.npy   — (T, 4, H, W) float32
  data/processed/3d_stats.json       — per-channel global mean/std
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    sys.exit("xarray not installed — run: pip install xarray netCDF4")

# Allow importing load_tcnd from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from load_tcnd import load_basin as _tcnd_load_basin

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")

# 925 hPa is index 3 in [200, 500, 850, 925]
PRESSURE_IDX = 3
LOOKBACK_DEFAULT = 8   # number of 6-hour steps

os.makedirs(os.path.join(PROC_DIR, "3d"), exist_ok=True)


# ── Filename → datetime ────────────────────────────────────────────────────────
def _fname_to_dt(fname: str) -> datetime | None:
    """
    Parse the datetime stamp embedded in a TCND filename.
    e.g. 'TCND_ALETTA_1988061618_sst_z_u_v.nc'  →  datetime(1988, 6, 16, 18)
    """
    base = os.path.basename(fname)
    parts = base.split("_")
    # Find the part that looks like a 10-digit timestamp
    for p in parts:
        if len(p) == 10 and p.isdigit():
            try:
                return datetime.strptime(p, "%Y%m%d%H")
            except ValueError:
                pass
    return None


# ── Build a storm → Data_3d directory mapping for one basin ───────────────────
def _build_storm_dir_map(basin: str) -> dict[str, str]:
    """
    Returns {storm_id: abs_path_to_storm_dir} for every storm found under
    data/raw/_tmp/Data3D/<BASIN>/
    """
    base = os.path.join(RAW_DIR, "_tmp", "Data3D", basin)
    if not os.path.isdir(base):
        return {}

    mapping = {}
    # Load Data_1d to get (year, name) → storm_id correspondence
    try:
        df = _tcnd_load_basin(basin)
    except FileNotFoundError:
        return {}

    # Extract year from storm_id (e.g. 'EP1988BSTGILMA' → '1988')
    df = df.drop_duplicates("storm_id").copy()
    df["year"] = df["storm_id"].str[len(basin):len(basin) + 4]
    year_name_to_id: dict[tuple[str, str], str] = {
        (row["year"], row["name"]): row["storm_id"]
        for _, row in df.iterrows()
    }

    for year in os.listdir(base):
        year_dir = os.path.join(base, year)
        if not os.path.isdir(year_dir):
            continue
        for storm_name in os.listdir(year_dir):
            storm_dir = os.path.join(year_dir, storm_name)
            if not os.path.isdir(storm_dir):
                continue
            sid = year_name_to_id.get((year, storm_name))
            if sid is not None:
                mapping[sid] = storm_dir

    return mapping


# ── Load one 925 hPa snapshot from a single NetCDF file ───────────────────────
def _load_snapshot(nc_path: str) -> np.ndarray | None:
    """
    Returns (4, 81, 81) float32: [u925, v925, z925, sst].
    Returns None on failure.
    """
    try:
        ds = xr.open_dataset(nc_path)
        u   = ds["u"].values[0, PRESSURE_IDX, :, :]    # (81, 81)
        v   = ds["v"].values[0, PRESSURE_IDX, :, :]
        z   = ds["z"].values[0, PRESSURE_IDX, :, :]
        sst = ds["sst"].values                          # (81, 81)
        ds.close()
        out = np.stack([u, v, z, sst], axis=0).astype(np.float32)  # (4, 81, 81)
        return out
    except Exception as e:
        print(f"    [warn] failed to read {nc_path}: {e}")
        return None


# ── Build (T, 4, 81, 81) patch for one storm ──────────────────────────────────
def _build_patch(storm_dir: str,
                 ref_time: pd.Timestamp,
                 lookback: int) -> np.ndarray | None:
    """
    Collect the `lookback` 6-hourly snapshots ending at ref_time.
    Returns (T, 4, 81, 81) float32 or None if insufficient files found.
    """
    # Index all files in the storm directory by their datetime
    nc_files = sorted(f for f in os.listdir(storm_dir) if f.endswith(".nc"))
    dt_map: dict[pd.Timestamp, str] = {}
    for f in nc_files:
        dt = _fname_to_dt(f)
        if dt is not None:
            ts = pd.Timestamp(dt)
            dt_map[ts] = os.path.join(storm_dir, f)

    if not dt_map:
        return None

    # Select up to `lookback` timesteps ending at ref_time
    available = sorted(t for t in dt_map if t <= ref_time)
    selected  = available[-lookback:]
    if not selected:
        return None

    snapshots = []
    for ts in selected:
        snap = _load_snapshot(dt_map[ts])
        if snap is None:
            return None
        snapshots.append(snap)

    patch = np.stack(snapshots, axis=0)   # (T, 4, 81, 81)
    # Pad with earliest snapshot if fewer than lookback steps available
    if len(snapshots) < lookback:
        pad_count = lookback - len(snapshots)
        pad = np.repeat(snapshots[0:1], pad_count, axis=0)
        patch = np.concatenate([pad, patch], axis=0)

    return patch


# ── Main ───────────────────────────────────────────────────────────────────────
def run(basins: list[str], lookback: int):
    # Load decay feature matrix to know which storms/ref_times are needed
    decay_path = os.path.join(FEAT_DIR, "feature_matrix_decay.csv")
    if not os.path.exists(decay_path):
        sys.exit(f"Feature matrix not found: {decay_path}\n"
                 "Run scripts/features.py first.")

    df = pd.read_csv(decay_path, parse_dates=["ref_time"])
    df = df[df["basin"].isin(basins)]
    print(f"Decay samples: {len(df)} across basins {basins}")

    all_patches: list[np.ndarray] = []   # for global normalisation
    patch_map:   dict[str, str]   = {}   # storm_id → saved .npy path

    skipped = 0
    for basin in basins:
        print(f"\n── {basin} ─────────────────────────────────────")
        dir_map = _build_storm_dir_map(basin)
        print(f"  Data_3d storms found: {len(dir_map)}")

        basin_df = df[df["basin"] == basin].drop_duplicates("storm_id")

        for _, row in basin_df.iterrows():
            sid      = row["storm_id"]
            ref_time = pd.Timestamp(row["ref_time"])

            storm_dir = dir_map.get(sid)
            if storm_dir is None:
                skipped += 1
                continue

            patch = _build_patch(storm_dir, ref_time, lookback)
            if patch is None:
                print(f"  [skip] {sid} — patch build failed")
                skipped += 1
                continue

            out_path = os.path.join(PROC_DIR, "3d", f"{sid}.npy")
            np.save(out_path, patch)
            all_patches.append(patch)
            patch_map[sid] = out_path

        print(f"  Saved: {len(patch_map)}  Skipped: {skipped}")

    if not all_patches:
        print("\nNo patches extracted. Check Data_3d path and feature matrix.")
        return

    # ── Global per-channel normalisation ──────────────────────────────────────
    print("\nComputing global channel statistics …")
    stack = np.concatenate(all_patches, axis=0)  # (N*T, 4, H, W)
    stats = {}
    for c in range(stack.shape[1]):
        ch = stack[:, c, :, :].reshape(-1).astype(np.float64)  # float64 avoids overflow
        ch = ch[np.isfinite(ch)]
        mean, std = float(np.mean(ch)), float(np.std(ch))
        std = max(std, 1e-6)
        stats[f"ch{c}_mean"] = mean
        stats[f"ch{c}_std"]  = std

    # Apply normalisation and re-save
    for sid, out_path in patch_map.items():
        patch = np.load(out_path)
        for c in range(patch.shape[1]):
            patch[:, c, :, :] = (
                (patch[:, c, :, :] - stats[f"ch{c}_mean"]) / stats[f"ch{c}_std"]
            )
        np.save(out_path, patch)

    stats_path = os.path.join(PROC_DIR, "3d_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nExtracted : {len(patch_map)} storm patches")
    print(f"Skipped   : {skipped}")
    print(f"Shape     : (T={lookback}, C=4 [u925 v925 z925 sst], H=81, W=81)")
    print(f"Stats     : {stats_path}")
    print(f"Patches   : {os.path.join(PROC_DIR, '3d/')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basins",   nargs="+", default=["WP", "NA", "EP"])
    parser.add_argument("--lookback", type=int,  default=LOOKBACK_DEFAULT)
    args = parser.parse_args()
    run(args.basins, args.lookback)
