"""
scripts/features_improved.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Added some changes from scripts/features.py.

Changes from the original:
  1. Data_3d features — actually loads .nc files and extracts:
       • Δp  (central pressure drop — Hadrian's top predictor)
       • 925 hPa wind speed at storm center (boundary layer)
       • Wind shear 200–850 hPa  (inhibits intensification)
       • SST at storm center
       • Spatial asymmetry of 925 hPa wind field
  2. Env-Data features — loads .npy files from the TCND Env-Data folder
       • movement velocity, intensity history, subtropical high, etc.
  3. Improved land-sea feature — dist_to_coast now uses real latitude so
       it is meaningful even with normalised longitude.
  4. Feature groups updated so ablation.py and train_ufno.py see the
       new groups automatically.

Run order (unchanged):
  python scripts/features_improved.py   # produces data/features/*.csv + *.json
  python scripts/ablation.py            # ranks groups, writes selected_feature_groups.json
  python scripts/train_ufno.py          # trains UFNO using selected features
"""

import glob
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from load_tcnd import load_basin as _tcnd_load_basin

# RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "_tmp") # Added _tmp following path in load_tcnd.py
FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
# BASINS   = ["WP", "NA", "EP"]
BASINS   = ["NA"]

os.makedirs(FEAT_DIR, exist_ok=True)

LOOKBACK = 8   # 48 h of 6-hour steps

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


# ── Load Data_1d ───────────────────────────────────────────────────────────────
def _load_1d(basin: str) -> pd.DataFrame:
    df = _tcnd_load_basin(basin)
    df = df.rename(columns={"wind_norm": "wind", "pres_norm": "pressure"})
    return df


# ── Tabular features (wind/pressure deltas, wp_couple, position) ──────────────
def _tabular_features(window: pd.DataFrame) -> dict:
    w = window["wind"].values.astype(float)
    p = window["pressure"].values.astype(float)

    def _delta(arr, steps):
        return float(arr[-1] - arr[-1 - steps]) if len(arr) > steps else 0.0

    def _trend(arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        return float(np.polyfit(x, arr, 1)[0])

    feats = {
        # Wind stats
        "wind_last":      float(w[-1]),
        "wind_max":       float(w.max()),
        "wind_mean":      float(w.mean()),
        "wind_std":       float(w.std()),
        "wind_delta_6h":  _delta(w, 1),
        "wind_delta_12h": _delta(w, 2),
        "wind_delta_24h": _delta(w, 4),
        "wind_trend":     _trend(w),
        # Pressure stats
        "pres_last":      float(p[-1]),
        "pres_min":       float(p.min()),
        "pres_mean":      float(p.mean()),
        "pres_std":       float(p.std()),
        "pres_delta_6h":  _delta(p, 1),
        "pres_delta_12h": _delta(p, 2),
        "pres_delta_24h": _delta(p, 4),
        "pres_trend":     _trend(p),
        # Wind-pressure coupling (residual from linear fit)
        "wp_residual":    float(w[-1] - (-0.5 * p[-1])),
        # Position / motion
        "lat_last":       float(window["lat"].iloc[-1]),
        "lon_norm_last":  float(window["lon_norm"].iloc[-1]) if "lon_norm" in window else 0.0,
    }

    # Motion speed and direction from last two rows
    if len(window) >= 2:
        dlat = window["lat"].iloc[-1] - window["lat"].iloc[-2]
        dlon = (window["lon_norm"].iloc[-1] - window["lon_norm"].iloc[-2]
                if "lon_norm" in window.columns else 0.0)
        feats["motion_speed_kph"] = float(math.hypot(dlat, dlon) * 111.0)
        feats["motion_dir_deg"]   = float(math.degrees(math.atan2(dlon, dlat)) % 360)
    else:
        feats["motion_speed_kph"] = 0.0
        feats["motion_dir_deg"]   = 0.0

    return feats


# ── Land-sea features ──────────────────────────────────────────────────────────
_LAND_BOXES = [
    (-35, 75, -25, 60), (10, 75, -170, -50), (-55, 15, -85, -30),
    (-10, 55, 60, 150), (-45, -10, 110, 155), (5, 30, 70, 105),
    (30, 75, 10, 60),
]

def _over_land(lat, lon_norm):
    # lon_norm is normalised — use lat for distance proxy only
    return any(r[0] <= lat <= r[1] for r in _LAND_BOXES)

def _dist_to_coast_lat(lat):
    """Distance from nearest land-box edge in latitude degrees."""
    return min(min(abs(lat - r[0]), abs(lat - r[1])) for r in _LAND_BOXES)

def _land_sea_features(window: pd.DataFrame) -> dict:
    lats = window["lat"].values
    land_flags = [_over_land(la, 0) for la in lats]
    return {
        "over_land":        float(land_flags[-1]),
        "dist_to_coast":    float(_dist_to_coast_lat(lats[-1])),
        "land_frac_window": float(np.mean(land_flags)),
    }


# ── NEW: Data_3d features ──────────────────────────────────────────────────────
# Pressure level index in the TCND .nc files: 0=200hPa 1=500hPa 2=850hPa 3=925hPa
_PLEV_200  = 0
_PLEV_850  = 2
_PLEV_925  = 3   # boundary layer — Hadrian's recommended focus

def _load_nc(nc_path: str):
    """Load a single TCND Data_3d .nc file. Returns xarray Dataset or None."""
    try:
        import xarray as xr
        if os.path.getsize(nc_path) < 100:
            print(f"DEBUG: file too small: {os.path.getsize(nc_path)} bytes", flush=True)
            return None
        return xr.open_dataset(nc_path, engine="netcdf4")
    except Exception as e:
        print(f"DEBUG: _load_nc error: {e}", flush=True)
        return None

def _center_slice(arr, half=5):
    """Extract a (2*half+1) x (2*half+1) patch from the center of arr (H, W)."""
    H, W = arr.shape
    ch, cw = H // 2, W // 2
    return arr[max(0, ch - half):ch + half + 1,
               max(0, cw - half):cw + half + 1]

def _spatial_features_3d(basin: str, sid: str) -> dict:
    """
    Extract physically meaningful features from Data_3d .nc files.

    Features extracted:
        dp_central      — central pressure drop at 925 hPa (Δp, Hadrian's #1)
        wind925_center  — mean 925 hPa wind speed at storm center
        wind_shear      — 200–850 hPa wind shear magnitude (inhibits intensification)
        sst_center      — mean SST at storm center (thermodynamic fuel)
        asym_925        — left-right wind asymmetry at 925 hPa
        dp_trend        — change in central Δp over available timesteps
    """
    storm_name_raw = sid.split("BST")[-1]

    # Try title case first (WP: Billie), then upper case (NA/EP: BRENDA/ALETTA)
    nc_files = []
    for storm_name in [storm_name_raw.title(), storm_name_raw.upper()]:
        pattern = os.path.join(RAW_DIR, "Data3D", basin, "**", storm_name, "*.nc")
        nc_files = sorted(glob.glob(pattern, recursive=True))
        if nc_files:
            break

    if not nc_files:
        return {}   # Data_3d not available for this storm — skip silently

    feats = {}
    dp_vals = []

    for f in nc_files[-LOOKBACK:]:   # use most recent LOOKBACK files
        ds = _load_nc(f)
        if ds is None:
            print(f"DEBUG: _load_nc returned None for {os.path.basename(f)}", flush=True)
            continue

        try:
            # ── 925 hPa wind speed at center ──────────────────────────────
            u925 = ds["u"].values.squeeze(0)[_PLEV_925]   # (H, W)
            v925 = ds["v"].values.squeeze(0)[_PLEV_925]
            ws925 = np.sqrt(u925**2 + v925**2)
            center_ws = float(_center_slice(ws925).mean())

            # ── SST at center ──────────────────────────────────────────────
            sst = ds["sst"].values                          # (H, W)
            center_sst = float(_center_slice(sst).mean())

            # ── Wind shear 200–850 hPa ─────────────────────────────────────
            u200 = ds["u"].values.squeeze(0)[_PLEV_200]
            v200 = ds["v"].values.squeeze(0)[_PLEV_200]
            u850 = ds["u"].values.squeeze(0)[_PLEV_850]
            v850 = ds["v"].values.squeeze(0)[_PLEV_850]
            shear = float(np.sqrt((u200 - u850)**2 + (v200 - v850)**2).mean())

            # ── Δp: geopotential height anomaly at center (925 hPa) ────────
            z925 = ds["z"].values.squeeze(0)[_PLEV_925]    # (H, W)
            z_center = float(_center_slice(z925).mean())
            z_env    = float(z925.mean())
            dp = z_env - z_center    # positive = lower pressure at center = stronger storm
            dp_vals.append(dp)

            # ── Asymmetry ──────────────────────────────────────────────────
            H, W = ws925.shape
            left  = ws925[:, :W//2].mean()
            right = ws925[:, W//2:].mean()
            asym  = float(right - left)

        except Exception:
            continue
        finally:
            ds.close()

    if not dp_vals:
        return {}

    # Summarise over the lookback window
    feats["dp_central"]     = float(np.mean(dp_vals))
    feats["wind925_center"] = center_ws          # last timestep
    feats["wind_shear"]     = shear              # last timestep
    feats["sst_center"]     = center_sst         # last timestep
    feats["asym_925"]       = asym               # last timestep
    feats["dp_trend"]       = float(dp_vals[-1] - dp_vals[0]) if len(dp_vals) > 1 else 0.0

    return feats

def _env_features(basin: str, sid: str) -> dict:
    """
    Load Env-Data .npy files for a storm.
    Tries multiple path layouts used by TCND.
    Returns a flat dict: env_0, env_1, ... env_N

    Key env features (from the TCND slides):
      move_velocity, month, location, history_direction12/24,
      history_intensity_change24, subtropical_high
    """
    storm_name_raw = sid.split("BST")[-1]

    # Try title case first (WP: Billie), then upper case (NA/EP: BRENDA/ALETTA)
    npy_files = []
    for storm_name in [storm_name_raw.title(), storm_name_raw.upper()]:
        pattern = os.path.join(RAW_DIR, "Env-Data", basin, "**", storm_name, "*.npy")
        npy_files = sorted(glob.glob(pattern, recursive=True))
        if npy_files:
            break

    if not npy_files:
        for storm_name in [storm_name_raw.title(), storm_name_raw.upper()]:
            single = os.path.join(RAW_DIR, "Env-Data", basin, f"{storm_name}.npy")
            if os.path.exists(single):
                npy_files = [single]
                break

    if not npy_files:
        return {}

    # Use the last LOOKBACK timesteps, take the most recent one for scalar features
    # and aggregate over the window for history features
    last_file = npy_files[-1]
    if os.path.getsize(last_file) < 10:
        return {}

    try:
        data = np.load(last_file, allow_pickle=True).item()
    except Exception:
        return {}

    feats = {}

    # Scalar features
    feats["env_move_velocity"] = float(data.get("move_velocity", 0.0))
    feats["env_wind"]          = float(data.get("wind", 0.0))

    # One-hot arrays → argmax for a compact scalar
    month_arr = data.get("month", None)
    if month_arr is not None:
        arr = np.asarray(month_arr).flatten()
        feats["env_month"] = float(np.argmax(arr)) if arr.sum() > 0 else 0.0

    area_arr = data.get("area", None)
    if area_arr is not None:
        arr = np.asarray(area_arr).flatten()
        feats["env_area"] = float(np.argmax(arr)) if arr.sum() > 0 else 0.0

    intensity_arr = data.get("intensity_class", None)
    if intensity_arr is not None:
        arr = np.asarray(intensity_arr).flatten()
        feats["env_intensity_class"] = float(np.argmax(arr)) if arr.sum() > 0 else 0.0

    # Historical direction and intensity change (key signals per Hadrian)
    hd12 = data.get("history_direction12", -1)
    hd24 = data.get("history_direction24", -1)
    hi24 = data.get("history_inte_change24", -1)

    def _to_scalar(v):
        try:
            return float(np.asarray(v).item())
        except Exception:
            return -1.0

    feats["env_history_dir12"]   = _to_scalar(hd12)
    feats["env_history_dir24"]   = _to_scalar(hd24)
    feats["env_history_int24"]   = _to_scalar(hi24)

    return feats

# ── Detect reference events (peak intensity) ───────────────────────────────────
def _detect_landfalls(data: pd.DataFrame) -> pd.DataFrame:
    events = []
    for sid, grp in data.groupby("storm_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        if len(grp) < 4:
            continue
        peak_i = int(grp["wind"].idxmax())
        peak_r = grp.loc[peak_i]
        steps_after = len(grp) - 1 - peak_i
        if steps_after < 1:
            continue
        post_wind = grp.loc[peak_i + 1:, "wind"].values
        made_lf   = int(len(post_wind) >= 4 and post_wind[0] < peak_r["wind"])
        events.append({
            "storm_id":      sid,
            "basin":         peak_r.get("basin", grp.iloc[0]["basin"]),
            "landfall_time": peak_r["time"],
            "landfall_wind": peak_r["wind"],
            "landfall_pres": peak_r["pressure"],
            "landfall_lat":  peak_r["lat"],
            "landfall_lon":  peak_r.get("lon_norm", 0.0),
            "made_landfall": made_lf,
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
              f"Events: {len(events)}  "
              f"Landfalls: {events['made_landfall'].sum()}")

        n_with_3d  = 0
        n_with_env = 0

        for i, (_, ev) in enumerate(events.iterrows()):
            sid = ev["storm_id"]
            t0  = ev["landfall_time"]

            # Progress every 50 storms
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(events)} storms processed...", flush=True)

            track  = data[data["storm_id"] == sid].sort_values("time")
            window = track[track["time"] <= t0].tail(LOOKBACK)
            if window.empty:
                continue

            # ── Feature groups ──────────────────────────────────────────
            tab     = _tabular_features(window)
            ls      = _land_sea_features(window)
            sp3d    = _spatial_features_3d(basin, sid)
            env     = _env_features(basin, sid)

            if sp3d:
                n_with_3d += 1
            if env:
                n_with_env += 1

            tcnd_split = (data[data["storm_id"] == sid]["split"].iloc[0]
                          if "split" in data.columns else "train")

            base = {
                "storm_id":      sid,
                "basin":         basin,
                "split":         tcnd_split,
                "ref_time":      t0,
                "made_landfall": ev["made_landfall"],
            }

            # Landfall row
            row_lf = {**base, **tab, **ls, **sp3d, **env,
                      "hours_to_landfall": 0.0}
            landfall_rows.append(row_lf)

            # Decay row (landfall events only, need targets at +24h/+48h)
            if ev["made_landfall"] == 1:
                v0 = ev["landfall_wind"]
                if pd.isna(v0) or v0 == 0:
                    continue

                near = track.set_index("time")["wind"]

                def _wind_at(h):
                    t = t0 + pd.Timedelta(hours=h)
                    idx = abs(near.index - t).argmin()
                    if abs(near.index[idx] - t) <= pd.Timedelta(hours=4):
                        return float(near.iloc[idx])
                    return np.nan

                w24, w48 = _wind_at(24), _wind_at(48)
                if np.isnan(w24) or np.isnan(w48):
                    continue

                decay_rows.append({
                    **base, **tab, **ls, **sp3d, **env,
                    "landfall_wind":  v0,
                    "landfall_pres":  ev["landfall_pres"],
                    "wind_24h":       w24,
                    "wind_48h":       w48,
                    "wind_frac_24h":  w24 / v0,
                    "wind_frac_48h":  w48 / v0,
                })

        print(f"  With Data_3d features : {n_with_3d}")
        print(f"  With Env-Data features: {n_with_env}")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    lf_df = pd.DataFrame(landfall_rows)
    dc_df = pd.DataFrame(decay_rows)

    lf_df.to_csv(os.path.join(FEAT_DIR, "feature_matrix_landfall.csv"), index=False)
    dc_df.to_csv(os.path.join(FEAT_DIR, "feature_matrix_decay.csv"),    index=False)
    print(f"\nLandfall matrix : {lf_df.shape}")
    print(f"Decay matrix    : {dc_df.shape}")

    # ── Update feature_groups.json (read by ablation.py + train_ufno.py) ──────
    all_cols = list(lf_df.columns)
    groups = {
        "wind":      [c for c in all_cols if c.startswith("wind") and not c.startswith("wind_frac")],
        "pressure":  [c for c in all_cols if c.startswith("pres")],
        "wp_couple": [c for c in all_cols if "wp_" in c],
        "position":  [c for c in all_cols if c in
                      ("lat_last", "lon_norm_last", "motion_speed_kph", "motion_dir_deg")],
        "land_sea":  [c for c in all_cols if c in
                      ("over_land", "dist_to_coast", "land_frac_window")],
        # NEW groups
        "spatial_3d": [c for c in all_cols if c in
                       ("dp_central", "wind925_center", "wind_shear",
                        "sst_center", "asym_925", "dp_trend")],
        "env_data":  [c for c in all_cols if c.startswith("env_")],
    }

    _meta = {"storm_id", "basin", "split", "ref_time", "made_landfall",
             "hours_to_landfall", "landfall_wind", "landfall_pres",
             "wind_24h", "wind_48h", "wind_frac_24h", "wind_frac_48h"}
    groups = {k: [c for c in v if c not in _meta] for k, v in groups.items()}

    grp_path = os.path.join(FEAT_DIR, "feature_groups.json")
    with open(grp_path, "w") as f:
        json.dump(groups, f, indent=2)

    print(f"\nFeature groups:")
    for g, cols in groups.items():
        print(f"  {g:<14}: {len(cols)} features  {cols[:3]}{'...' if len(cols)>3 else ''}")

    return lf_df, dc_df, groups


if __name__ == "__main__":
    build_feature_matrices()