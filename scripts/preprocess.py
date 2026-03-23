"""
Extract 925 hPa boundary-layer patches from TCND Data_3d NetCDF files.

For each storm in the decay/landfall feature matrix, reads T=8 NetCDF files
ending at the reference time, extracts the 925 hPa level for u, v, z and the
2-D SST field, stacks them into a (T, 4, 81, 81) tensor, normalises per
channel, and saves to data/processed/3d[_landfall]/{storm_id[_ref]}.npy.

The saved patches are consumed by SpatialDecayDataset / SpatialLandfallDataset
in train_ufno.py.

Directory layout expected
─────────────────────────
data/raw/_tmp/Data3D/<BASIN>/<YEAR>/<STORM_NAME>/
    TCND_<NAME>_<YYYYMMDDH0>_sst_z_u_v.nc

Usage
─────
  python scripts/preprocess.py [--basins WP NA EP] [--lookback 8]
                                [--task decay|landfall|all]

Outputs
───────
  data/processed/3d/<storm_id>.npy                      — decay patches
  data/processed/3d_stats.json                          — decay channel stats
  data/processed/3d_landfall/<storm_id>_<YYYYMMDDHH>.npy — landfall patches
  data/processed/3d_landfall_stats.json                 — landfall channel stats
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

os.makedirs(os.path.join(PROC_DIR, "3d"),          exist_ok=True)
os.makedirs(os.path.join(PROC_DIR, "3d_landfall"), exist_ok=True)


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

    # Some zips extract with an extra basin-named subdirectory (e.g. NA/NA/);
    # find the actual root by descending into any same-named subdirectory.
    search_base = base
    if os.path.isdir(os.path.join(base, basin)):
        search_base = os.path.join(base, basin)

    for year in os.listdir(search_base):
        year_dir = os.path.join(search_base, year)
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

    df = pd.read_csv(decay_path, parse_dates=["ref_time"], keep_default_na=False)
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


# ── Landfall patch extraction ─────────────────────────────────────────────────
def run_landfall(basins: list[str], lookback: int):
    """
    Build one (T, 4, 81, 81) patch per (storm, ref_time) row in
    feature_matrix_landfall.csv and save to data/processed/3d_landfall/.

    Filename: {storm_id}_{YYYYMMDDHH}.npy  (one file per observation)
    """
    lf_path = os.path.join(FEAT_DIR, "feature_matrix_landfall.csv")
    if not os.path.exists(lf_path):
        sys.exit(f"Feature matrix not found: {lf_path}\nRun scripts/features.py first.")

    df = pd.read_csv(lf_path, parse_dates=["ref_time"], keep_default_na=False)
    df = df[df["basin"].isin(basins)]
    df = df[df["hours_to_landfall"] > 0].reset_index(drop=True)
    print(f"Landfall samples: {len(df)} across basins {basins}")

    out_dir = os.path.join(PROC_DIR, "3d_landfall")
    all_patches: list[np.ndarray] = []
    patch_map:   dict[str, str]   = {}   # key → saved path
    skipped = 0

    for basin in basins:
        print(f"\n── {basin} ─────────────────────────────────────")
        dir_map  = _build_storm_dir_map(basin)
        print(f"  Data_3d storms found: {len(dir_map)}")

        basin_df = df[df["basin"] == basin]

        for _, row in basin_df.iterrows():
            sid      = row["storm_id"]
            ref_time = pd.Timestamp(row["ref_time"])
            ref_str  = ref_time.strftime("%Y%m%d%H")
            key      = f"{sid}_{ref_str}"
            out_path = os.path.join(out_dir, f"{key}.npy")

            # Skip if already extracted
            if os.path.exists(out_path):
                patch_map[key] = out_path
                continue

            storm_dir = dir_map.get(sid)
            if storm_dir is None:
                skipped += 1
                continue

            patch = _build_patch(storm_dir, ref_time, lookback)
            if patch is None:
                skipped += 1
                continue

            np.save(out_path, patch)
            all_patches.append(patch)
            patch_map[key] = out_path

        print(f"  Saved: {len(patch_map)}  Skipped: {skipped}")

    if not all_patches and not patch_map:
        print("\nNo patches extracted. Check Data_3d path and feature matrix.")
        return

    # ── Global per-channel normalisation ──────────────────────────────────────
    # Load all patches (including previously saved ones) for global stats
    all_keys = [f[:-4] for f in os.listdir(out_dir) if f.endswith(".npy")]
    print(f"\nComputing global channel statistics from {len(all_keys)} patches …")
    all_for_stats = [np.load(os.path.join(out_dir, f"{k}.npy")) for k in all_keys]
    stack = np.concatenate(all_for_stats, axis=0)
    stats = {}
    for c in range(stack.shape[1]):
        ch = stack[:, c, :, :].reshape(-1).astype(np.float64)
        ch = ch[np.isfinite(ch)]
        stats[f"ch{c}_mean"] = float(np.mean(ch))
        stats[f"ch{c}_std"]  = float(max(np.std(ch), 1e-6))

    for k in all_keys:
        p = np.load(os.path.join(out_dir, f"{k}.npy"))
        for c in range(p.shape[1]):
            p[:, c, :, :] = (p[:, c, :, :] - stats[f"ch{c}_mean"]) / stats[f"ch{c}_std"]
        np.save(os.path.join(out_dir, f"{k}.npy"), p)

    stats_path = os.path.join(PROC_DIR, "3d_landfall_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nExtracted : {len(patch_map)} landfall patches")
    print(f"Skipped   : {skipped}")
    print(f"Stats     : {stats_path}")
    print(f"Patches   : {out_dir}/")


# ── Zip-streaming mode (avoids full extraction for large zips) ─────────────────
def run_from_zip(zip_path: str, basin: str, lookback: int):
    """
    Process one basin directly from a zip file, extracting only T=lookback
    nc files per storm to a temp directory.  Deletes temp files after each
    storm.  Never has more than ~3 MB of temp files on disk at once.

    Usage:
      python scripts/preprocess.py --zip /path/to/TCND_Data3D_WP.zip --basins WP
    """
    import tempfile, zipfile as _zipfile

    decay_path = os.path.join(FEAT_DIR, "feature_matrix_decay.csv")
    df = pd.read_csv(decay_path, parse_dates=["ref_time"], keep_default_na=False)
    df = df[df["basin"] == basin].drop_duplicates("storm_id")
    print(f"Decay samples ({basin}): {len(df)}")

    # Load Data_1d mapping: (year, name) → storm_id
    try:
        df1d = _tcnd_load_basin(basin).drop_duplicates("storm_id").copy()
    except FileNotFoundError:
        sys.exit(f"Data_1d for {basin} not found.")
    df1d["year"] = df1d["storm_id"].str[len(basin):len(basin) + 4]
    id_to_meta = {
        row["storm_id"]: (row["year"], row["name"])
        for _, row in df1d.iterrows()
    }

    print(f"Opening zip: {zip_path}")
    with _zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Build (year, storm_name) → sorted list of zip entry names
        entry_map: dict[tuple[str, str], list[str]] = {}
        for name in all_names:
            if not name.endswith(".nc"):
                continue
            parts = name.replace("\\", "/").split("/")
            # Expected: <anything>/<YEAR>/<STORM_NAME>/TCND_NAME_YYYYMMDDH0_sst_z_u_v.nc
            nc_idx = next((i for i, p in enumerate(parts) if p.endswith(".nc")), None)
            if nc_idx is None or nc_idx < 2:
                continue
            storm_name = parts[nc_idx - 1]
            year       = parts[nc_idx - 2]
            entry_map.setdefault((year, storm_name), []).append(name)

        for k in entry_map:
            entry_map[k].sort()

        all_patches: list[np.ndarray] = []
        patch_map:   dict[str, str]   = {}
        skipped = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            for _, row in df.iterrows():
                sid      = row["storm_id"]
                ref_time = pd.Timestamp(row["ref_time"])
                meta     = id_to_meta.get(sid)
                if meta is None:
                    skipped += 1
                    continue
                year, storm_name = meta
                entries = entry_map.get((year, storm_name), [])
                if not entries:
                    skipped += 1
                    continue

                # Parse timestamps and select T=lookback steps ending at ref_time
                dt_entries: dict[pd.Timestamp, str] = {}
                for e in entries:
                    dt = _fname_to_dt(os.path.basename(e))
                    if dt is not None:
                        dt_entries[pd.Timestamp(dt)] = e

                available = sorted(t for t in dt_entries if t <= ref_time)
                selected  = available[-lookback:]
                if not selected:
                    skipped += 1
                    continue

                # Extract selected files to tmpdir, load, delete
                snapshots = []
                ok = True
                for ts in selected:
                    entry = dt_entries[ts]
                    tmp_nc = os.path.join(tmpdir, os.path.basename(entry))
                    try:
                        zf.extract(entry, tmpdir)
                        # The extract preserves directory structure; find actual file
                        extracted = os.path.join(tmpdir, entry.replace("\\", "/"))
                        snap = _load_snapshot(extracted)
                        if snap is None:
                            ok = False
                            break
                        snapshots.append(snap)
                        os.remove(extracted)
                    except Exception as ex:
                        print(f"  [warn] {sid} {ts}: {ex}")
                        ok = False
                        break

                if not ok or not snapshots:
                    skipped += 1
                    continue

                patch = np.stack(snapshots, axis=0)
                if len(snapshots) < lookback:
                    pad = np.repeat(snapshots[0:1], lookback - len(snapshots), axis=0)
                    patch = np.concatenate([pad, patch], axis=0)

                out_path = os.path.join(PROC_DIR, "3d", f"{sid}.npy")
                np.save(out_path, patch)
                all_patches.append(patch)
                patch_map[sid] = out_path

        print(f"  Saved: {len(patch_map)}  Skipped: {skipped}")

    if not all_patches:
        print("No patches extracted.")
        return

    # Re-normalise using all patches (including previously saved EP/NA)
    # Load existing patches to compute global stats
    existing_ids = [f[:-4] for f in os.listdir(os.path.join(PROC_DIR, "3d"))
                    if f.endswith(".npy")]
    print(f"\nRecomputing global stats from {len(existing_ids)} total patches …")
    all_for_stats = []
    for sid in existing_ids:
        p = np.load(os.path.join(PROC_DIR, "3d", f"{sid}.npy"))
        all_for_stats.append(p)

    stack = np.concatenate(all_for_stats, axis=0)
    stats = {}
    for c in range(stack.shape[1]):
        ch = stack[:, c, :, :].reshape(-1).astype(np.float64)
        ch = ch[np.isfinite(ch)]
        mean, std = float(np.mean(ch)), float(np.std(ch))
        stats[f"ch{c}_mean"] = mean
        stats[f"ch{c}_std"]  = max(std, 1e-6)

    for sid in existing_ids:
        p = np.load(os.path.join(PROC_DIR, "3d", f"{sid}.npy"))
        for c in range(p.shape[1]):
            p[:, c, :, :] = (p[:, c, :, :] - stats[f"ch{c}_mean"]) / stats[f"ch{c}_std"]
        np.save(os.path.join(PROC_DIR, "3d", f"{sid}.npy"), p)

    stats_path = os.path.join(PROC_DIR, "3d_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Extracted : {len(patch_map)} new patches for {basin}")
    print(f"Total     : {len(existing_ids)} patches normalised")
    print(f"Stats     : {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basins",   nargs="+", default=["WP", "NA", "EP"])
    parser.add_argument("--lookback", type=int,  default=LOOKBACK_DEFAULT)
    parser.add_argument("--task",     choices=["decay", "landfall", "all"],
                        default="all",
                        help="Which task's patches to build (default: all)")
    parser.add_argument("--zip",      default=None,
                        help="Path to a single-basin Data_3d zip file.  "
                             "Streams nc files without full extraction.  "
                             "Use with --basins specifying the single basin.")
    args = parser.parse_args()
    if args.zip:
        if len(args.basins) != 1:
            sys.exit("--zip requires exactly one basin via --basins")
        run_from_zip(args.zip, args.basins[0], args.lookback)
    else:
        if args.task in ("decay", "all"):
            run(args.basins, args.lookback)
        if args.task in ("landfall", "all"):
            run_landfall(args.basins, args.lookback)
