"""
Full feature extraction for both analysis tasks.

Two tasks are handled separately throughout the pipeline:
  1. landfall_prediction  — does/when does a storm make landfall?
                            label: binary (made_landfall) + hours_to_landfall
  2. intensity_decay      — how fast does wind decay after landfall?
                            label: wind_24h, wind_48h (knots)

Feature groups extracted
────────────────────────
Group              Source       Features
─────────────────────────────────────────────────────────────────────────────
wind               Data_1d      wind speed + 6h/12h/24h deltas + rolling stats
pressure           Data_1d      MSLP + 6h/12h/24h deltas + rolling stats
position           Data_1d      lat, lon, storm motion speed + direction
spatial            Data_3d      per-channel mean/std/max/p90 (summary stats)
env                Env-Data     all pre-computed environmental features
land_sea           derived      distance-to-coast proxy + land fraction flag
─────────────────────────────────────────────────────────────────────────────

Output (data/features/):
    feature_matrix_landfall.csv   — one row per pre-landfall sample
    feature_matrix_decay.csv      — one row per post-landfall sample
    feature_groups.json           — maps group name -> list of column names
"""

import glob
import json
import math
import os

import numpy as np
import pandas as pd

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
BASINS   = ["WP", "NA", "EP"]

os.makedirs(FEAT_DIR, exist_ok=True)

# Lookback window (number of 6-hour steps) used for feature windows
LOOKBACK = 8   # 48 h

# ── Column aliases ─────────────────────────────────────────────────────────────
_COL_MAP = {
    "storm_id": ["SID", "storm_id", "ID"],
    "time":     ["ISO_TIME", "time", "TIME", "datetime"],
    "lat":      ["LAT", "lat", "CLAT", "USA_LAT"],
    "lon":      ["LON", "lon", "CLON", "USA_LON"],
    "wind":     ["WMO_WIND", "wind", "USA_WIND", "VMAX"],
    "pressure": ["WMO_PRES", "pressure", "USA_PRES", "MSLP"],
}

def _col(df, key):
    for c in _COL_MAP[key]:
        if c in df.columns:
            return c
    raise KeyError(f"Column for '{key}' not found in {list(df.columns)}")


# ── Land mask ──────────────────────────────────────────────────────────────────
_LAND_BOXES = [
    (-35, 75, -25, 60), (10, 75, -170, -50), (-55, 15, -85, -30),
    (-10, 55, 60, 150), (-45, -10, 110, 155), (5, 30, 70, 105),
    (30, 75, 10, 60),
]

def _over_land(lat, lon):
    lon = ((lon + 180) % 360) - 180
    return any(r[0] <= lat <= r[1] and r[2] <= lon <= r[3] for r in _LAND_BOXES)

def _dist_to_coast_proxy(lat, lon):
    """
    Rough distance-to-coast estimate (degrees) as the min distance to any
    land-box boundary.  Replace with a proper coastline shapefile if available.
    """
    lon = ((lon + 180) % 360) - 180
    min_d = float("inf")
    for la0, la1, lo0, lo1 in _LAND_BOXES:
        d = min(abs(lat - la0), abs(lat - la1),
                abs(lon - lo0), abs(lon - lo1))
        min_d = min(min_d, d)
    return min_d


# ── Load Data_1d ───────────────────────────────────────────────────────────────
def _load_1d(basin: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(RAW_DIR, "Data_1d", basin, "**", "*.csv"),
                      recursive=True)
    if not files:
        raise FileNotFoundError(f"No Data_1d CSVs for basin {basin}")
    frames = [pd.read_csv(f, low_memory=False, skiprows=lambda i: i == 1)
              for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        _col(df, "storm_id"): "storm_id",
        _col(df, "time"):     "time",
        _col(df, "lat"):      "lat",
        _col(df, "lon"):      "lon",
        _col(df, "wind"):     "wind",
        _col(df, "pressure"): "pressure",
    })
    for col in ["lat", "lon", "wind", "pressure"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"]  = pd.to_datetime(df["time"], errors="coerce")
    df["basin"] = basin
    return df.dropna(subset=["storm_id","time","lat","lon"]).sort_values(
        ["storm_id","time"]).reset_index(drop=True)


# ── Derive tabular features for a single track window ─────────────────────────
def _tabular_features(window: pd.DataFrame) -> dict:
    """
    Given a DataFrame slice (up to LOOKBACK rows) ending at the reference time,
    compute all tabular features.  Returns a flat dict.
    """
    w = window.copy().reset_index(drop=True)
    feats = {}

    # ── wind group ──────────────────────────────────────────────────
    w_wind = w["wind"].values
    feats["wind_last"]      = w_wind[-1]
    feats["wind_max"]       = float(np.nanmax(w_wind))
    feats["wind_mean"]      = float(np.nanmean(w_wind))
    feats["wind_std"]       = float(np.nanstd(w_wind))
    feats["wind_delta_6h"]  = float(w_wind[-1] - w_wind[-2]) if len(w_wind) >= 2 else 0.0
    feats["wind_delta_12h"] = float(w_wind[-1] - w_wind[-3]) if len(w_wind) >= 3 else 0.0
    feats["wind_delta_24h"] = float(w_wind[-1] - w_wind[-5]) if len(w_wind) >= 5 else 0.0
    feats["wind_trend"]     = float(np.polyfit(range(len(w_wind)), w_wind, 1)[0]) \
                              if len(w_wind) >= 2 else 0.0

    # ── pressure group ──────────────────────────────────────────────
    w_pres = w["pressure"].values
    feats["pres_last"]      = float(np.nanmin(w_pres))   # min MSLP = intensity
    feats["pres_min"]       = float(np.nanmin(w_pres))
    feats["pres_mean"]      = float(np.nanmean(w_pres))
    feats["pres_std"]       = float(np.nanstd(w_pres))
    feats["pres_delta_6h"]  = float(w_pres[-1] - w_pres[-2]) if len(w_pres) >= 2 else 0.0
    feats["pres_delta_12h"] = float(w_pres[-1] - w_pres[-3]) if len(w_pres) >= 3 else 0.0
    feats["pres_delta_24h"] = float(w_pres[-1] - w_pres[-5]) if len(w_pres) >= 5 else 0.0
    feats["pres_trend"]     = float(np.polyfit(range(len(w_pres)), w_pres, 1)[0]) \
                              if len(w_pres) >= 2 else 0.0

    # ── wind-pressure coupling ───────────────────────────────────────
    # Deviation from the empirical Dvorak W-P relationship
    if not np.isnan(feats["wind_last"]) and not np.isnan(feats["pres_last"]):
        expected_wind = 2.3 * (1013 - feats["pres_last"]) ** 0.5
        feats["wp_residual"] = feats["wind_last"] - expected_wind
    else:
        feats["wp_residual"] = np.nan

    # ── position group ──────────────────────────────────────────────
    feats["lat_last"] = float(w["lat"].iloc[-1])
    feats["lon_last"] = float(w["lon"].iloc[-1])

    if len(w) >= 2:
        dlat = w["lat"].iloc[-1] - w["lat"].iloc[-2]
        dlon = w["lon"].iloc[-1] - w["lon"].iloc[-2]
        # approximate km per degree
        feats["motion_speed_kph"] = math.hypot(dlat * 111, dlon * 111 *
                                    math.cos(math.radians(w["lat"].iloc[-1]))) / 6
        feats["motion_dir_deg"]   = math.degrees(math.atan2(dlon, dlat)) % 360
    else:
        feats["motion_speed_kph"] = 0.0
        feats["motion_dir_deg"]   = 0.0

    # ── land-sea group ───────────────────────────────────────────────
    lat0, lon0 = w["lat"].iloc[-1], w["lon"].iloc[-1]
    feats["over_land"]       = int(_over_land(lat0, lon0))
    feats["dist_to_coast"]   = _dist_to_coast_proxy(lat0, lon0)
    # Count how many of the lookback steps were over land
    feats["land_frac_window"] = float(
        sum(_over_land(r["lat"], r["lon"]) for _, r in w.iterrows()) / len(w))

    return feats


# ── Spatial features from Data_3d patch ───────────────────────────────────────
def _spatial_features(basin: str, sid: str) -> dict:
    """
    Summarise the Data_3d patch (T, C, H, W) into scalar statistics.
    Returns {} if no patch is found.
    """
    base = os.path.join(RAW_DIR, "Data_3d", basin)
    arr = None

    storm_dir = os.path.join(base, sid)
    if os.path.isdir(storm_dir):
        files = sorted(glob.glob(os.path.join(storm_dir, "*.npy")))
        if files:
            patches = [np.load(f) for f in files[-LOOKBACK:]]
            arr = np.stack(patches, axis=0)   # (T, C, H, W)

    if arr is None:
        npy = os.path.join(base, f"{sid}.npy")
        if os.path.exists(npy):
            arr = np.load(npy)
            if arr.ndim == 3:
                arr = arr[np.newaxis]
            arr = arr[-LOOKBACK:]

    if arr is None:
        return {}

    feats = {}
    n_channels = arr.shape[1]
    for c in range(n_channels):
        patch = arr[:, c, :, :]   # (T, H, W)
        tag   = f"sp_ch{c}"
        feats[f"{tag}_mean"]     = float(np.nanmean(patch))
        feats[f"{tag}_std"]      = float(np.nanstd(patch))
        feats[f"{tag}_max"]      = float(np.nanmax(patch))
        feats[f"{tag}_p90"]      = float(np.nanpercentile(patch, 90))
        feats[f"{tag}_asymmetry"]= float(  # left-right asymmetry across storm
            np.nanmean(patch[:, :, patch.shape[2]//2:]) -
            np.nanmean(patch[:, :, :patch.shape[2]//2]))
    return feats


# ── Environmental features from Env-Data ──────────────────────────────────────
def _env_features(basin: str, sid: str) -> dict:
    npy = os.path.join(RAW_DIR, "Env-Data", basin, f"{sid}.npy")
    if os.path.exists(npy):
        arr = np.load(npy).flatten()
        return {f"env_{i}": float(v) for i, v in enumerate(arr)}

    csv_files = glob.glob(os.path.join(RAW_DIR, "Env-Data", basin, "**", "*.csv"),
                          recursive=True)
    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        if sid in df.index:
            return {f"env_{k}": float(v) for k, v in df.loc[sid].items()}
    return {}


# ── Detect landfalls ───────────────────────────────────────────────────────────
def _detect_landfalls(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["over_land"] = data.apply(
        lambda r: _over_land(r["lat"], r["lon"]), axis=1)
    events = []
    for sid, grp in data.groupby("storm_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        made_lf = False
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
                    "made_landfall": 1,
                })
                made_lf = True
                break
        if not made_lf:
            # No landfall — use peak intensity point as reference
            peak_i = grp["wind"].idxmax()
            events.append({
                "storm_id":      sid,
                "basin":         grp.loc[peak_i, "basin"] if "basin" in grp.columns else grp.iloc[0]["basin"],
                "landfall_time": grp.loc[peak_i, "time"],
                "landfall_wind": grp.loc[peak_i, "wind"],
                "landfall_pres": grp.loc[peak_i, "pressure"],
                "landfall_lat":  grp.loc[peak_i, "lat"],
                "landfall_lon":  grp.loc[peak_i, "lon"],
                "made_landfall": 0,
            })
    return pd.DataFrame(events)


# ── Build feature matrices ─────────────────────────────────────────────────────
def build_feature_matrices():
    landfall_rows, decay_rows = [], []

    for basin in BASINS:
        print(f"\n{'─'*50}\nBasin: {basin}")
        try:
            data = _load_1d(basin)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        events = _detect_landfalls(data)
        print(f"  Storms: {data['storm_id'].nunique()}  "
              f"Landfalls: {events['made_landfall'].sum()}")

        for _, ev in events.iterrows():
            sid = ev["storm_id"]
            t0  = ev["landfall_time"]

            # Lookback window ending at reference time
            track = data[data["storm_id"] == sid].sort_values("time")
            window = track[track["time"] <= t0].tail(LOOKBACK)
            if window.empty:
                continue

            tab  = _tabular_features(window)
            sp   = _spatial_features(basin, sid)
            env  = _env_features(basin, sid)

            base = {
                "storm_id":      sid,
                "basin":         basin,
                "ref_time":      t0,
                "made_landfall": ev["made_landfall"],
            }

            # ── Landfall matrix ──────────────────────────────────────
            row_lf = {**base, **tab, **sp, **env}
            # Time-to-landfall label (hours) — 0 for no-landfall storms
            row_lf["hours_to_landfall"] = 0.0 if ev["made_landfall"] == 0 else 0.0
            landfall_rows.append(row_lf)

            # ── Decay matrix (landfall storms only) ──────────────────
            if ev["made_landfall"] == 1:
                v0 = ev["landfall_wind"]
                if pd.isna(v0) or v0 == 0:
                    continue

                def _wind_at(h):
                    t = t0 + pd.Timedelta(hours=h)
                    near = track.set_index("time")["wind"]
                    idx  = abs(near.index - t).argmin()
                    if abs(near.index[idx] - t) <= pd.Timedelta(hours=4):
                        return float(near.iloc[idx])
                    return np.nan

                w24 = _wind_at(24)
                w48 = _wind_at(48)
                if np.isnan(w24) or np.isnan(w48):
                    continue

                row_dc = {
                    **base,
                    **tab, **sp, **env,
                    "landfall_wind":  v0,
                    "landfall_pres":  ev["landfall_pres"],
                    "wind_24h":       w24,
                    "wind_48h":       w48,
                    "wind_frac_24h":  w24 / v0,
                    "wind_frac_48h":  w48 / v0,
                }
                decay_rows.append(row_dc)

    # Save
    lf_df = pd.DataFrame(landfall_rows)
    dc_df = pd.DataFrame(decay_rows)

    lf_path = os.path.join(FEAT_DIR, "feature_matrix_landfall.csv")
    dc_path = os.path.join(FEAT_DIR, "feature_matrix_decay.csv")
    lf_df.to_csv(lf_path, index=False)
    dc_df.to_csv(dc_path, index=False)
    print(f"\nLandfall matrix : {lf_df.shape}  -> {lf_path}")
    print(f"Decay matrix    : {dc_df.shape}  -> {dc_path}")

    # Build and save feature group map
    all_cols = list(lf_df.columns)
    groups = {
        "wind":      [c for c in all_cols if c.startswith("wind") or "wind" in c.split("_")],
        "pressure":  [c for c in all_cols if c.startswith("pres")],
        "wp_couple": [c for c in all_cols if "wp_" in c],
        "position":  [c for c in all_cols if c in
                      ("lat_last","lon_last","motion_speed_kph","motion_dir_deg")],
        "spatial":   [c for c in all_cols if c.startswith("sp_")],
        "env":       [c for c in all_cols if c.startswith("env_")],
        "land_sea":  [c for c in all_cols if c in
                      ("over_land","dist_to_coast","land_frac_window")],
    }
    # Remove meta columns from all groups
    _meta = {"storm_id","basin","ref_time","made_landfall","hours_to_landfall",
             "landfall_wind","landfall_pres","wind_24h","wind_48h",
             "wind_frac_24h","wind_frac_48h"}
    groups = {k: [c for c in v if c not in _meta] for k, v in groups.items()}

    grp_path = os.path.join(FEAT_DIR, "feature_groups.json")
    with open(grp_path, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"Feature groups  : {grp_path}")

    for g, cols in groups.items():
        print(f"  {g:<12}: {len(cols)} features")

    return lf_df, dc_df, groups


if __name__ == "__main__":
    build_feature_matrices()
