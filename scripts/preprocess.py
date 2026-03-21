"""
Multimodal preprocessing for TCND — Data_1d, Data_3d, Env-Data.

Produces aligned, normalised tensors for each storm sample, ready for
the multimodal model in model.py.

Output layout  (data/processed/):
    samples.csv          — one row per sample (storm_id, basin, landfall_time,
                           landfall_wind, label_wind_24h, label_wind_48h)
    1d/<id>.npy          — (T, F1) tabular sequence, T=lookback steps, F1=features
    3d/<id>.npy          — (T, C, H, W) spatial patch sequence
    env/<id>.npy         — (E,)  env feature vector
    scaler_1d.pkl        — StandardScaler for Data_1d features
    scaler_env.pkl       — StandardScaler for Env-Data features
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
BASINS   = ["WP", "NA", "EP"]

# Number of 6-hour steps to look back before landfall
LOOKBACK_STEPS = 8   # 48 hours

# ── Column aliases (IBTrACS varies across versions) ───────────────────────────
COL_MAP = {
    "storm_id": ["SID", "storm_id", "ID"],
    "time":     ["ISO_TIME", "time", "TIME", "datetime"],
    "lat":      ["LAT", "lat", "CLAT", "USA_LAT"],
    "lon":      ["LON", "lon", "CLON", "USA_LON"],
    "wind":     ["WMO_WIND", "wind", "USA_WIND", "VMAX"],
    "pressure": ["WMO_PRES", "pressure", "USA_PRES", "MSLP"],
}

def _col(df, key):
    for c in COL_MAP[key]:
        if c in df.columns:
            return c
    raise KeyError(f"Column not found for '{key}' among {list(df.columns)}")


# ── Land mask (lightweight) ───────────────────────────────────────────────────
_LAND_BOXES = [
    (-35,  75,  -25,  60),
    ( 10,  75, -170, -50),
    (-55,  15,  -85, -30),
    (-10,  55,   60, 150),
    (-45, -10,  110, 155),
    (  5,  30,   70, 105),
    ( 30,  75,   10,  60),
]

def _over_land(lat, lon):
    lon = ((lon + 180) % 360) - 180
    return any(r[0] <= lat <= r[1] and r[2] <= lon <= r[3] for r in _LAND_BOXES)


# ── 1. Load Data_1d ───────────────────────────────────────────────────────────
def load_1d(basin: str) -> pd.DataFrame:
    pattern = os.path.join(RAW_DIR, "Data_1d", basin, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No Data_1d CSVs for {basin}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, skiprows=lambda i: i == 1)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    data = data.rename(columns={
        _col(data, "storm_id"): "storm_id",
        _col(data, "time"):     "time",
        _col(data, "lat"):      "lat",
        _col(data, "lon"):      "lon",
        _col(data, "wind"):     "wind",
        _col(data, "pressure"): "pressure",
    })
    data["time"]     = pd.to_datetime(data["time"], errors="coerce")
    data["lat"]      = pd.to_numeric(data["lat"],      errors="coerce")
    data["lon"]      = pd.to_numeric(data["lon"],      errors="coerce")
    data["wind"]     = pd.to_numeric(data["wind"],     errors="coerce")
    data["pressure"] = pd.to_numeric(data["pressure"], errors="coerce")
    data["basin"]    = basin
    return data.dropna(subset=["storm_id","time","lat","lon"]).sort_values(
        ["storm_id","time"]).reset_index(drop=True)


# ── 2. Detect landfall events ─────────────────────────────────────────────────
def detect_landfalls(data: pd.DataFrame) -> pd.DataFrame:
    data["over_land"] = data.apply(lambda r: _over_land(r["lat"], r["lon"]), axis=1)
    events = []
    for sid, grp in data.groupby("storm_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        for i in range(1, len(grp)):
            if not grp.loc[i-1, "over_land"] and grp.loc[i, "over_land"]:
                events.append({
                    "storm_id":      sid,
                    "basin":         grp.loc[i, "basin"],
                    "landfall_time": grp.loc[i, "time"],
                    "landfall_wind": grp.loc[i, "wind"],
                    "landfall_pres": grp.loc[i, "pressure"],
                    "landfall_lat":  grp.loc[i, "lat"],
                    "landfall_lon":  grp.loc[i, "lon"],
                })
                break
    return pd.DataFrame(events)


# ── 3. Build labelled samples ─────────────────────────────────────────────────
def build_samples(data: pd.DataFrame, landfalls: pd.DataFrame) -> pd.DataFrame:
    """Add regression targets: wind 24 h and 48 h post-landfall."""
    rows = []
    for _, lf in landfalls.iterrows():
        sid = lf["storm_id"]
        t0  = lf["landfall_time"]
        track = data[data["storm_id"] == sid].set_index("time")["wind"]

        def _wind_at(hours):
            t = t0 + pd.Timedelta(hours=hours)
            nearest = track.index[abs(track.index - t).argmin()]
            if abs(nearest - t) <= pd.Timedelta(hours=4):
                return track[nearest]
            return np.nan

        rows.append({**lf.to_dict(),
                     "label_wind_24h": _wind_at(24),
                     "label_wind_48h": _wind_at(48)})
    return pd.DataFrame(rows).dropna(subset=["label_wind_24h","label_wind_48h"])


# ── 4. Extract Data_1d lookback sequence ─────────────────────────────────────
_1D_FEATURES = ["lat", "lon", "wind", "pressure"]

def extract_1d_sequence(data: pd.DataFrame, sid: str, t_landfall: pd.Timestamp) -> np.ndarray:
    """Return (LOOKBACK_STEPS, F1) array of tabular features before landfall."""
    track = data[data["storm_id"] == sid].sort_values("time")
    track = track[track["time"] <= t_landfall].tail(LOOKBACK_STEPS)
    arr = track[_1D_FEATURES].values.astype(np.float32)
    # Pad with the earliest row if not enough history
    if len(arr) < LOOKBACK_STEPS:
        pad = np.repeat(arr[:1], LOOKBACK_STEPS - len(arr), axis=0)
        arr = np.vstack([pad, arr])
    return arr   # (LOOKBACK_STEPS, 4)


# ── 5. Load Data_3d patch ─────────────────────────────────────────────────────
def load_3d_patch(basin: str, sid: str, t_landfall: pd.Timestamp) -> np.ndarray | None:
    """
    Load the 3D spatial patch closest to landfall time.
    Expected file layout: Data_3d/<BASIN>/<storm_id>/<timestamp>.npy  OR
                          Data_3d/<BASIN>/<storm_id>.npy  (single array)
    Returns (T, C, H, W) or None if not found.
    """
    base = os.path.join(RAW_DIR, "Data_3d", basin)

    # Try per-storm directory
    storm_dir = os.path.join(base, sid)
    if os.path.isdir(storm_dir):
        files = sorted(glob.glob(os.path.join(storm_dir, "*.npy")))
        if files:
            # Pick the LOOKBACK_STEPS files ending at landfall
            patches = [np.load(f) for f in files[-LOOKBACK_STEPS:]]
            arr = np.stack(patches, axis=0).astype(np.float32)  # (T, C, H, W)
            return arr

    # Try single .npy per storm
    npy_file = os.path.join(base, f"{sid}.npy")
    if os.path.exists(npy_file):
        arr = np.load(npy_file).astype(np.float32)
        if arr.ndim == 3:          # (C, H, W) — single snapshot
            arr = arr[np.newaxis]  # -> (1, C, H, W)
        return arr[-LOOKBACK_STEPS:]

    return None


# ── 6. Load Env-Data vector ───────────────────────────────────────────────────
def load_env_features(basin: str, sid: str) -> np.ndarray | None:
    """
    Load pre-calculated environmental feature vector for a storm.
    Expected layout: Env-Data/<BASIN>/<storm_id>.npy  or a master CSV.
    """
    npy_file = os.path.join(RAW_DIR, "Env-Data", basin, f"{sid}.npy")
    if os.path.exists(npy_file):
        return np.load(npy_file).astype(np.float32)

    # Fallback: look for a CSV where rows are storms
    csv_files = glob.glob(os.path.join(RAW_DIR, "Env-Data", basin, "**", "*.csv"), recursive=True)
    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        if sid in df.index:
            return df.loc[sid].values.astype(np.float32)

    return None


# ── 7. Fit scalers & serialise ────────────────────────────────────────────────
def fit_and_save_scalers(all_1d: list[np.ndarray], all_env: list[np.ndarray]):
    sc1d = StandardScaler()
    sc1d.fit(np.vstack(all_1d).reshape(-1, len(_1D_FEATURES)))
    with open(os.path.join(PROC_DIR, "scaler_1d.pkl"), "wb") as f:
        pickle.dump(sc1d, f)

    if all_env:
        scenv = StandardScaler()
        scenv.fit(np.vstack(all_env))
        with open(os.path.join(PROC_DIR, "scaler_env.pkl"), "wb") as f:
            pickle.dump(scenv, f)
        return sc1d, scenv
    return sc1d, None


def apply_scaler_1d(sc, arr: np.ndarray) -> np.ndarray:
    T, F = arr.shape
    return sc.transform(arr.reshape(-1, F)).reshape(T, F)


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    for d in ["1d", "3d", "env"]:
        os.makedirs(os.path.join(PROC_DIR, d), exist_ok=True)

    all_samples = []
    raw_1d_seqs, raw_env_vecs = [], []

    # Pass 1: load data and collect raw arrays for scaler fitting
    basin_data, basin_lf = {}, {}
    for basin in BASINS:
        print(f"\nLoading {basin} Data_1d...")
        try:
            data = load_1d(basin)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        lf = detect_landfalls(data)
        samples = build_samples(data, lf)
        basin_data[basin] = data
        basin_lf[basin]   = samples
        print(f"  {len(samples)} labelled samples")

        for _, row in samples.iterrows():
            seq = extract_1d_sequence(data, row["storm_id"], row["landfall_time"])
            raw_1d_seqs.append(seq)

            env = load_env_features(basin, row["storm_id"])
            if env is not None:
                raw_env_vecs.append(env)

    if not raw_1d_seqs:
        print("\nNo samples found. Run download_data.py first.")
        return

    print("\nFitting scalers...")
    sc1d, scenv = fit_and_save_scalers(raw_1d_seqs, raw_env_vecs)

    # Pass 2: normalise and serialise each sample
    idx = 0
    for basin in BASINS:
        if basin not in basin_lf:
            continue
        data    = basin_data[basin]
        samples = basin_lf[basin]

        for _, row in samples.iterrows():
            sid = row["storm_id"]
            t0  = row["landfall_time"]
            sample_id = f"{basin}_{sid}".replace(" ", "_")

            # Data_1d
            seq_raw = raw_1d_seqs[idx]
            seq_norm = apply_scaler_1d(sc1d, seq_raw)
            np.save(os.path.join(PROC_DIR, "1d", f"{sample_id}.npy"), seq_norm)

            # Data_3d
            patch = load_3d_patch(basin, sid, t0)
            if patch is not None:
                np.save(os.path.join(PROC_DIR, "3d", f"{sample_id}.npy"), patch)

            # Env-Data
            env = load_env_features(basin, sid)
            if env is not None and scenv is not None:
                env_norm = scenv.transform(env.reshape(1, -1)).flatten()
                np.save(os.path.join(PROC_DIR, "env", f"{sample_id}.npy"), env_norm)

            all_samples.append({**row.to_dict(), "sample_id": sample_id,
                                 "has_3d": patch is not None,
                                 "has_env": env is not None})
            idx += 1

    samples_df = pd.DataFrame(all_samples)
    samples_df.to_csv(os.path.join(PROC_DIR, "samples.csv"), index=False)
    print(f"\nTotal samples : {len(samples_df)}")
    print(f"With Data_3d  : {samples_df['has_3d'].sum()}")
    print(f"With Env-Data : {samples_df['has_env'].sum()}")
    print(f"Saved to      : {PROC_DIR}")


if __name__ == "__main__":
    run()
