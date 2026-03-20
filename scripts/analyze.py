"""
Cyclone Landfall Impact & Intensity Decay Analysis
===================================================
Uses TCND Data_1d (IBTrACS tabular records) for WP, NA, and EP basins.

Each basin CSV is expected to have at minimum:
    - storm_id / SID       : unique storm identifier
    - time / ISO_TIME      : timestamp (6-hour intervals)
    - lat / LAT            : latitude
    - lon / LON            : longitude
    - wind / WMO_WIND      : max sustained wind (knots)
    - pressure / WMO_PRES  : min sea-level pressure (hPa)

Outputs:
    - data/processed/<BASIN>_landfall_events.csv   : one row per landfall
    - data/processed/<BASIN>_decay_curves.csv      : post-landfall decay timeseries
    - figures/<BASIN>_intensity_decay.png          : decay curve plots
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BASINS = ["WP", "NA", "EP"]
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Column name aliases (IBTrACS uses different names across versions) ─────────
COL_MAP = {
    "storm_id": ["SID", "storm_id", "ID", "STORMID"],
    "time":     ["ISO_TIME", "time", "TIME", "datetime"],
    "lat":      ["LAT", "lat", "CLAT", "USA_LAT"],
    "lon":      ["LON", "lon", "CLON", "USA_LON"],
    "wind":     ["WMO_WIND", "wind", "USA_WIND", "VMAX"],
    "pressure": ["WMO_PRES", "pressure", "USA_PRES", "MSLP"],
}


def resolve_col(df, key):
    """Return the first matching column name for a logical key."""
    for candidate in COL_MAP[key]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not find a column for '{key}'. Available: {list(df.columns)}")


def load_basin(basin: str) -> pd.DataFrame:
    basin_dir = os.path.join(RAW_DIR, basin)
    csv_files = glob.glob(os.path.join(basin_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {basin_dir}. Run download_data.py first.")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, skiprows=lambda i: i == 1)  # IBTrACS has a unit row
        df["_source_file"] = os.path.basename(f)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Normalise column names
    col_time = resolve_col(data, "time")
    col_lat  = resolve_col(data, "lat")
    col_lon  = resolve_col(data, "lon")
    col_wind = resolve_col(data, "wind")
    col_pres = resolve_col(data, "pressure")
    col_sid  = resolve_col(data, "storm_id")

    data = data.rename(columns={
        col_sid:  "storm_id",
        col_time: "time",
        col_lat:  "lat",
        col_lon:  "lon",
        col_wind: "wind",
        col_pres: "pressure",
    })

    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data["lat"]  = pd.to_numeric(data["lat"],  errors="coerce")
    data["lon"]  = pd.to_numeric(data["lon"],  errors="coerce")
    data["wind"] = pd.to_numeric(data["wind"], errors="coerce")
    data["pressure"] = pd.to_numeric(data["pressure"], errors="coerce")

    data = data.dropna(subset=["storm_id", "time", "lat", "lon"])
    data = data.sort_values(["storm_id", "time"]).reset_index(drop=True)
    data["basin"] = basin
    return data


# ── Simple land mask via lat/lon bounding boxes ────────────────────────────────
# A lightweight approximation; replace with shapely/cartopy for production use.
def is_over_land(lat: float, lon: float) -> bool:
    """
    Rough land detection using coarse bounding boxes for major landmasses.
    Returns True when the point is likely over land.
    """
    lon = ((lon + 180) % 360) - 180  # normalise to [-180, 180]
    boxes = [
        # (lat_min, lat_max, lon_min, lon_max)
        (-35,  75,  -25,  60),   # Africa + Europe
        ( 10,  75,  -170, -50),  # North America
        (-55,  15,  -85,  -30),  # South America
        (-10,  55,   60,  150),  # Asia
        (-45,  -10, 110,  155),  # Australia
        (  5,  30,   70,  105),  # Indian subcontinent
        ( 30,  75,   10,   60),  # Middle East / Eurasia
    ]
    return any(lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
               for lat_min, lat_max, lon_min, lon_max in boxes)


def detect_landfalls(data: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the first point where each storm transitions from ocean to land.
    Returns a DataFrame of landfall events.
    """
    data["over_land"] = data.apply(lambda r: is_over_land(r["lat"], r["lon"]), axis=1)

    events = []
    for sid, grp in data.groupby("storm_id"):
        grp = grp.sort_values("time").reset_index(drop=True)
        for i in range(1, len(grp)):
            if not grp.loc[i - 1, "over_land"] and grp.loc[i, "over_land"]:
                row = grp.loc[i]
                events.append({
                    "storm_id":        sid,
                    "basin":           grp.loc[i, "basin"],
                    "landfall_time":   row["time"],
                    "landfall_lat":    row["lat"],
                    "landfall_lon":    row["lon"],
                    "landfall_wind":   row["wind"],
                    "landfall_pres":   row["pressure"],
                })
                break  # first landfall only

    return pd.DataFrame(events)


def build_decay_curves(data: pd.DataFrame, landfalls: pd.DataFrame,
                       hours: int = 72) -> pd.DataFrame:
    """
    For each landfall event, extract up to `hours` of post-landfall wind records.
    Returns a tidy DataFrame with columns: storm_id, hours_since_lf, wind, wind_frac.
    """
    records = []
    delta = pd.Timedelta(hours=hours)

    for _, lf in landfalls.iterrows():
        sid = lf["storm_id"]
        t0  = lf["landfall_time"]
        v0  = lf["landfall_wind"]
        if pd.isna(v0) or v0 == 0:
            continue

        track = data[(data["storm_id"] == sid) & (data["time"] >= t0) &
                     (data["time"] <= t0 + delta)].copy()
        track["hours_since_lf"] = (track["time"] - t0).dt.total_seconds() / 3600
        track["wind_frac"] = track["wind"] / v0

        for _, r in track.iterrows():
            records.append({
                "storm_id":      sid,
                "basin":         lf["basin"],
                "hours_since_lf": r["hours_since_lf"],
                "wind":          r["wind"],
                "wind_frac":     r["wind_frac"],
            })

    return pd.DataFrame(records)


# ── Decay model: exponential  V(t) = V0 * exp(-alpha * t) ─────────────────────
def exp_decay(t, alpha):
    return np.exp(-alpha * t)


def fit_decay(decay_df: pd.DataFrame) -> dict:
    """Fit an exponential decay rate to the normalised wind fraction."""
    df = decay_df.dropna(subset=["hours_since_lf", "wind_frac"])
    df = df[(df["wind_frac"] > 0) & (df["hours_since_lf"] >= 0)]
    if len(df) < 5:
        return {"alpha": np.nan, "half_life_hrs": np.nan}

    try:
        popt, _ = curve_fit(exp_decay, df["hours_since_lf"], df["wind_frac"],
                            p0=[0.02], bounds=(0, 1))
        alpha = popt[0]
        half_life = np.log(2) / alpha if alpha > 0 else np.nan
        return {"alpha": alpha, "half_life_hrs": half_life}
    except RuntimeError:
        return {"alpha": np.nan, "half_life_hrs": np.nan}


def plot_decay(decay_df: pd.DataFrame, fit: dict, basin: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Individual storm traces (faint)
    for sid, grp in decay_df.groupby("storm_id"):
        ax.plot(grp["hours_since_lf"], grp["wind_frac"],
                color="steelblue", alpha=0.15, linewidth=0.8)

    # Bin-median
    bins = np.arange(0, decay_df["hours_since_lf"].max() + 6, 6)
    decay_df["bin"] = pd.cut(decay_df["hours_since_lf"], bins=bins, labels=bins[:-1])
    medians = decay_df.groupby("bin")["wind_frac"].median()
    ax.plot(medians.index.astype(float), medians.values,
            color="navy", linewidth=2, label="Median")

    # Fitted curve
    if not np.isnan(fit["alpha"]):
        t_fit = np.linspace(0, bins[-1], 300)
        ax.plot(t_fit, exp_decay(t_fit, fit["alpha"]),
                color="crimson", linestyle="--", linewidth=2,
                label=f"Exp fit  α={fit['alpha']:.4f}  t½={fit['half_life_hrs']:.1f} h")

    ax.set_xlabel("Hours since landfall")
    ax.set_ylabel("Wind speed / landfall wind  (V/V₀)")
    ax.set_title(f"{basin} basin — Post-landfall intensity decay")
    ax.legend()
    ax.set_ylim(0, 1.3)
    ax.grid(alpha=0.3)

    out = os.path.join(FIG_DIR, f"{basin}_intensity_decay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    summary_rows = []

    for basin in BASINS:
        print(f"\n{'='*50}")
        print(f"Basin: {basin}")
        print(f"{'='*50}")

        try:
            data = load_basin(basin)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        print(f"  Records loaded : {len(data):,}")
        print(f"  Unique storms  : {data['storm_id'].nunique():,}")

        landfalls = detect_landfalls(data)
        print(f"  Landfall events: {len(landfalls):,}")

        lf_out = os.path.join(PROC_DIR, f"{basin}_landfall_events.csv")
        landfalls.to_csv(lf_out, index=False)
        print(f"  Saved: {lf_out}")

        if landfalls.empty:
            print("  No landfall events detected — skipping decay analysis.")
            continue

        decay = build_decay_curves(data, landfalls, hours=72)
        decay_out = os.path.join(PROC_DIR, f"{basin}_decay_curves.csv")
        decay.to_csv(decay_out, index=False)
        print(f"  Saved: {decay_out}")

        fit = fit_decay(decay)
        print(f"  Decay rate α   : {fit['alpha']:.5f} /hr" if not np.isnan(fit['alpha']) else "  Decay fit: insufficient data")
        print(f"  Wind half-life : {fit['half_life_hrs']:.1f} hrs" if not np.isnan(fit['half_life_hrs']) else "")

        plot_decay(decay, fit, basin)

        summary_rows.append({
            "basin":           basin,
            "n_storms":        data["storm_id"].nunique(),
            "n_landfalls":     len(landfalls),
            "decay_alpha":     fit["alpha"],
            "wind_half_life_hrs": fit["half_life_hrs"],
        })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(PROC_DIR, "basin_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
