"""
TCND Data_1d loader.

Reads the tab-separated .txt files distributed in the TropiCycloneNet dataset
(https://github.com/xiaochengfuhuo/TropiCycloneNet-Dataset) and returns a
standard DataFrame used by features.py, analyze.py, and preprocess.py.

File format (8 tab-separated columns, no header)
─────────────────────────────────────────────────
  0  step        — 6-hour time-step index (float)
  1  flag        — intensity category flag (always 1.0 in practice)
  2  lat         — actual latitude in degrees  (-90 to 90)
  3  lon_norm    — normalised longitude  (approx. -14 to 14)
  4  wind_norm   — z-score normalised wind speed
  5  pres_norm   — z-score normalised min sea-level pressure
  6  datetime    — YYYYMMDDHH  (UTC)
  7  name        — storm name string

Note: only lat is in physical degrees; lon, wind, and pressure are normalised.
Features derived from these columns remain valid for ML (deltas, trends, stats)
even in normalised space.  Land-mask features that rely on real lon coordinates
are disabled automatically when loading through this module.
"""

import glob
import os

import numpy as np
import pandas as pd

# Root of the extracted TCND Data_1d tree
_DATA1D_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "Data_1d", "GLOBAL", "Data1D"
)

# Sub-splits available in the dataset
SPLITS = ("train", "val", "test")


def load_basin(basin: str,
               splits: tuple = SPLITS,
               data1d_root = None) -> pd.DataFrame:
    """
    Load all Data_1d tracks for one basin across the requested splits.

    Parameters
    ----------
    basin       : TCND basin code  ("WP", "NA", "EP", "NI", "SI", "SP")
    splits      : which sub-splits to include  (default: all three)
    data1d_root : override path to the Data1D root directory

    Returns
    -------
    pd.DataFrame with columns:
        storm_id, basin, split, time, lat, lon_norm,
        wind_norm, pres_norm, step, name
    """
    root = data1d_root or _DATA1D_ROOT
    frames = []

    for split in splits:
        pattern = os.path.join(root, basin, split, "*.txt")
        files   = sorted(glob.glob(pattern))
        if not files:
            continue

        for fpath in files:
            sid = os.path.splitext(os.path.basename(fpath))[0]  # e.g. WP1988BSTDOYLE
            try:
                df = _read_txt(fpath)
            except Exception:
                continue
            df["storm_id"] = sid
            df["basin"]    = basin
            df["split"]    = split
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No Data_1d files found for basin '{basin}' in {root}. "
            f"Run download_data.py first."
        )

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["storm_id", "time"]).reset_index(drop=True)
    return data


def load_all_basins(basins=("WP", "NA", "EP"),
                    splits: tuple = SPLITS,
                    data1d_root = None) -> pd.DataFrame:
    """Convenience wrapper: load multiple basins into one DataFrame."""
    frames = []
    for b in basins:
        try:
            frames.append(load_basin(b, splits, data1d_root))
        except FileNotFoundError as e:
            print(f"  SKIP {b}: {e}")
    if not frames:
        raise FileNotFoundError("No basin data could be loaded.")
    return pd.concat(frames, ignore_index=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_txt(fpath: str) -> pd.DataFrame:
    """Parse a single TCND .txt track file."""
    rows = []
    with open(fpath) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            rows.append({
                "step":      float(parts[0]),
                "lat":       float(parts[2]),
                "lon_norm":  float(parts[3]),
                "wind_norm": float(parts[4]),
                "pres_norm": float(parts[5]),
                "dt_raw":    parts[6],
                "name":      parts[7] if len(parts) > 7 else "",
            })

    if not rows:
        raise ValueError(f"Empty file: {fpath}")

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["dt_raw"], format="%Y%m%d%H", errors="coerce")
    df = df.drop(columns=["dt_raw"])
    df = df.dropna(subset=["time"])
    return df


def storm_count(basin: str, splits: tuple = SPLITS,
                data1d_root = None) -> dict:
    """Return {split: n_storms} for a basin."""
    root = data1d_root or _DATA1D_ROOT
    return {
        s: len(glob.glob(os.path.join(root, basin, s, "*.txt")))
        for s in splits
    }


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    for basin in ("WP", "NA", "EP"):
        try:
            df = load_basin(basin)
            print(f"{basin}: {df['storm_id'].nunique()} storms, "
                  f"{len(df):,} rows  "
                  f"lat=[{df['lat'].min():.1f},{df['lat'].max():.1f}]  "
                  f"wind_norm=[{df['wind_norm'].min():.2f},{df['wind_norm'].max():.2f}]")
        except FileNotFoundError as e:
            print(f"{basin}: {e}")
