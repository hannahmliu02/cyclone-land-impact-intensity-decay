"""
Generate synthetic TCND-like data for pipeline testing.
Produces realistic IBTrACS-format CSVs, Data_3d patches, and Env-Data vectors.
"""

import os
import numpy as np
import pandas as pd

SEED   = 42
rng    = np.random.default_rng(SEED)
RAW    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

BASINS = {
    "WP": {"lat_range": (5, 35),  "lon_range": (110, 170), "n_storms": 80},
    "NA": {"lat_range": (10, 40), "lon_range": (-90, -20), "n_storms": 60},
    "EP": {"lat_range": (8, 30),  "lon_range": (-140, -80),"n_storms": 50},
}

N_CHANNELS = 5   # e.g. u-wind, v-wind, SST, humidity, geopotential
ENV_DIM    = 32  # pre-calculated env features


def _simulate_track(basin_cfg, sid, season):
    """Simulate a realistic tropical cyclone track."""
    la0 = rng.uniform(*basin_cfg["lat_range"])
    lo0 = rng.uniform(*basin_cfg["lon_range"])
    n_steps = rng.integers(20, 60)

    lats, lons, winds, pres = [la0], [lo0], [], []
    wind = rng.uniform(25, 45)
    prs  = 1005 - wind * 0.8

    # Intensification then weakening
    peak  = rng.integers(n_steps // 3, 2 * n_steps // 3)
    for i in range(n_steps):
        if i < peak:
            wind += rng.uniform(0, 4)
        else:
            wind -= rng.uniform(0, 3)
        wind = max(15, min(wind, 165))
        prs  = max(880, min(1010, 1013 - wind * 0.88 + rng.normal(0, 2)))
        winds.append(wind); pres.append(prs)

        # Poleward + westward (WP) or northward/eastward (NA/EP) drift
        dlat = rng.uniform(0.2, 0.6)
        dlon = rng.uniform(-0.5, 0.2) if basin_cfg["lon_range"][0] > 0 \
               else rng.uniform(-0.3, 0.5)
        lats.append(lats[-1] + dlat)
        lons.append(lons[-1] + dlon)

    t0 = pd.Timestamp(f"{season}-06-01") + pd.Timedelta(
        hours=int(rng.integers(0, 120 * 24)))
    times = [t0 + pd.Timedelta(hours=6 * i) for i in range(n_steps)]

    return pd.DataFrame({
        "SID":      sid,
        "ISO_TIME": times,
        "LAT":      lats[:n_steps],
        "LON":      lons[:n_steps],
        "WMO_WIND": winds,
        "WMO_PRES": pres,
    })


def make_1d(basin, cfg):
    out_dir = os.path.join(RAW, "Data_1d", basin)
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(cfg["n_storms"]):
        sid    = f"{basin}2000{i:04d}"
        season = rng.integers(2000, 2023)
        frames.append(_simulate_track(cfg, sid, season))
    df = pd.concat(frames, ignore_index=True)
    path = os.path.join(out_dir, f"{basin}_tracks.csv")
    df.to_csv(path, index=False)
    print(f"  Data_1d/{basin}: {len(df):5d} rows  ({cfg['n_storms']} storms) -> {path}")


def make_3d(basin, cfg):
    """One .npy patch per storm, shape (T=8, C=5, H=80, W=80)."""
    out_dir = os.path.join(RAW, "Data_3d", basin)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(cfg["n_storms"]):
        sid = f"{basin}2000{i:04d}"
        T   = rng.integers(4, 9)
        arr = rng.normal(0, 1, (T, N_CHANNELS, 80, 80)).astype(np.float32)
        # Channel 0 ~ wind field: stronger near centre
        cx, cy = 40, 40
        for t in range(T):
            Y, X = np.ogrid[:80, :80]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            arr[t, 0] += 3 * np.exp(-dist / 15)
        np.save(os.path.join(out_dir, f"{sid}.npy"), arr)
    print(f"  Data_3d/{basin}: {cfg['n_storms']} patches (T,5,80,80)")


def make_env(basin, cfg):
    """One .npy env vector per storm, shape (ENV_DIM,)."""
    out_dir = os.path.join(RAW, "Env-Data", basin)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(cfg["n_storms"]):
        sid = f"{basin}2000{i:04d}"
        arr = rng.normal(0, 1, ENV_DIM).astype(np.float32)
        np.save(os.path.join(out_dir, f"{sid}.npy"), arr)
    print(f"  Env-Data/{basin}: {cfg['n_storms']} vectors (dim={ENV_DIM})")


if __name__ == "__main__":
    print("Generating synthetic TCND data...\n")
    for basin, cfg in BASINS.items():
        print(f"Basin: {basin}")
        make_1d(basin, cfg)
        make_3d(basin, cfg)
        make_env(basin, cfg)
    print("\nDone. Run: python scripts/features.py")
