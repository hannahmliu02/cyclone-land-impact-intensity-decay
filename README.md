# Cyclone Land-Impact Intensity Decay

Predicting tropical cyclone **landfall timing** and **post-landfall intensity decay** using the **TropiCycloneNet Dataset (TCND)** and a **U-shaped Fourier Neural Operator (UFNO)** with 925 hPa boundary-layer spatial fields.

---

## Overview

Two prediction tasks:

1. **Landfall timing** — regression: hours remaining until the storm dissipates, estimated from the current point in the decay track (`feature_matrix_landfall.csv`, target: `hours_to_landfall`)
2. **Intensity decay** — regression: wind speed 24 h and 48 h after reference time (`feature_matrix_decay.csv`, targets: `wind_24h`, `wind_48h`)

Each task has its own model. The decay model optionally ingests a learned embedding from the frozen landfall model, injected via FiLM conditioning at UFNO blocks 3–5 — so landfall timing information shapes the spatial field rather than just being another input feature.

**Primary spatial input:** 925 hPa boundary-layer fields (u-wind, v-wind, geopotential z, SST) extracted from TCND Data_3d NetCDF files and preprocessed into `(T=8, C=4, H=81, W=81)` tensors per storm.

---

## Dataset

**TropiCycloneNet Dataset (TCND)** — Data_1d + Data_3d

| Property | Value |
|---|---|
| Basins available | WP, NA, EP, NI, SI, SP |
| Basins used | WP, NA, EP |
| Total storms (decay task) | 933 |
| Storms with 925 hPa patches | 932 (WP: 604, NA: 214, EP: 114) |
| Landfall matrix samples | 7,167 (3 reference times per storm) |
| Train / Val / Test | 651 / 188 / 94 (TCND original splits) |
| Data_1d format | Tab-separated `.txt`, 8 columns per timestep |
| Data_3d format | NetCDF, 81×81 grid, 4 pressure levels (200/500/850/925 hPa) |

TCND tracks start at or near peak intensity — all track data represents the post-peak decay phase. TCND original train/val/test splits are respected throughout to prevent temporal leakage.

Raw Data_1d: `data/raw/_tmp/Data1D/<basin>/<split>/`
Processed patches: `data/processed/3d/<storm_id>.npy`

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/load_tcnd.py` | Shared TCND Data_1d loader |
| `scripts/preprocess.py` | Extract 925 hPa patches from Data_3d NetCDF files; supports `--zip` streaming mode for large basins |
| `scripts/features.py` | Feature engineering — tabular (wind, pressure, position) + spatial scalar summaries from 925 hPa patches |
| `scripts/ablation.py` | Ablation study — ranks feature groups by R², writes `selected_feature_groups.json` |
| `scripts/explain.py` | SHAP + LIME feature importance (optional) |
| `scripts/ufno.py` | CycloneUFNO model architecture |
| `scripts/train_ufno.py` | Training — landfall timing and decay tasks |
| `scripts/view_results.py` | 8-panel results dashboard |
| `scripts/log_experiment.py` | Log experiment results to `experiments/` |
| `scripts/cross_basin.py` | Cross-basin generalization experiments |
| `scripts/download_data.py` | TCND data downloader |

---

## Model: CycloneUFNO

Based on **Gege Wen et al. 2022** ([github.com/gegewen/ufno](https://github.com/gegewen/ufno)).

| Component | Description |
|---|---|
| SpectralConv2d | 2D FFT over H×W; 4-quadrant complex weights |
| UNet2d | 3-level encoder-decoder with skip connections |
| FiLM (tabular) | Tabular features → scale + shift at every block |
| FiLM (landfall) | Landfall embedding → scale + shift at blocks 3–5 only |
| CycloneUFNOStack | 6 UFNO blocks; UNet at blocks 3, 4, 5 |
| Parameters | ~11 M |

**Task outputs:**
- Landfall: `(B, 1)` → `hours_to_landfall` (hours remaining in storm track)
- Decay: `(B, 2)` → `[wind_24h, wind_48h]`

**Input modes (auto-detected):**
- Spatial — 925 hPa patch tensors `(T, 4, H, W)` + tabular (requires `data/processed/3d/`)
- Tabular-only — tabular features projected to 16×16 pseudo-grid (fallback when no patches)

Device priority: CUDA → MPS (Apple Silicon) → CPU.

---

## Pipeline

### Step 0. Download and preprocess spatial data

```bash
# EP (extract from zip, standard mode)
python scripts/preprocess.py --basins EP

# NA (same)
python scripts/preprocess.py --basins NA

# WP (13 GB zip — stream directly without full extraction)
python scripts/preprocess.py --zip /path/to/TCND_Data3D_WP.zip --basins WP
```

Saves `(8, 4, 81, 81)` normalised 925 hPa patches to `data/processed/3d/`.

### Step 1. Build feature matrices

```bash
python scripts/features.py
```

Builds `feature_matrix_landfall.csv` and `feature_matrix_decay.csv`, including 20 spatial scalar summary features derived from the 925 hPa patches.

### Step 2. Run ablation (auto-updates feature selection)

```bash
python scripts/ablation.py
```

### Step 3. Train — two-stage

**Stage 1: landfall timing model**
```bash
python scripts/train_ufno.py --task landfall --epochs 100
```

**Stage 2: decay model with landfall embedding**
```bash
python scripts/train_ufno.py --task decay --epochs 150 \
    --landfall-ckpt models/best_ufno_landfall.pt
```

Or without landfall embedding:
```bash
python scripts/train_ufno.py --task decay --epochs 150
```

Options:
```
--task          {landfall,decay}  Task to train             (default decay)
--landfall-ckpt PATH              Landfall checkpoint for embedding injection
--epochs        int               Training epochs           (default 100)
--batch         int               Batch size                (default 8)
--lr            float             Initial LR                (default 1e-3)
--modes         int               Fourier modes             (default 12)
--width         int               UFNO hidden width         (default 32)
--unet-dropout  float             UNet dropout              (default 0.2)
--seed          int               Random seed               (default 42)
```

### Step 4. View results

```bash
python scripts/view_results.py --show
```

Saves an 8-panel dashboard to `figures/ufno_results/dashboard.png`.

### Step 5. Log the experiment

```bash
python scripts/log_experiment.py \
  --name "Two-stage: landfall embedding in decay model" \
  --notes "What you observed"
```

### Step 6. Feature importance (optional)

```bash
python scripts/explain.py
```

### Step 7. Cross-basin generalization (optional)

```bash
python scripts/cross_basin.py --epochs 60
```

---

## Feature Engineering

| Group | Source | Features | Ablation R² (decay 24h) |
|---|---|---|---|
| wp_couple | Data_1d | wind-pressure residual | 0.272 |
| wind | Data_1d | last, max, mean, std, delta_6h/12h/24h, trend | 0.269 |
| pressure | Data_1d | last, min, mean, std, delta_6h/12h/24h, trend | 0.216 |
| spatial | Data_3d 925 hPa | mean, std, max, p90, asymmetry × 4 channels | 0.093 |
| position | Data_1d | lat, lon_norm, motion_speed, motion_dir | negative |
| land_sea | derived | over_land, dist_to_coast, land_frac_window | negative |

**Selected groups (ablation-driven):** `wp_couple`, `wind`, `pressure`, `spatial` — 37 features total.

Spatial scalar summaries (R²=0.093) understate the value of 925 hPa patches for the UFNO, which ingests the full `(T, 4, 81, 81)` field directly.

`ablation.py` automatically updates `data/features/selected_feature_groups.json`, which `train_ufno.py` reads at startup.

---

## Outputs

| Path | Contents |
|---|---|
| `data/processed/3d/<storm_id>.npy` | (8, 4, 81, 81) normalised 925 hPa patch per storm |
| `data/processed/3d_stats.json` | Per-channel global mean/std used for normalisation |
| `data/features/feature_matrix_decay.csv` | 933 × 55 feature matrix — decay task |
| `data/features/feature_matrix_landfall.csv` | 7,167 × 50 feature matrix — landfall timing task |
| `data/features/feature_groups.json` | Feature group definitions |
| `data/features/selected_feature_groups.json` | Ablation-selected groups (auto-updated) |
| `data/features/ablation_*.csv` | Ablation results per task |
| `data/features/cross_basin_results.csv` | Cross-basin experiment RMSE table |
| `models/best_ufno_landfall.pt` | Best landfall timing checkpoint |
| `models/best_ufno_decay.pt` | Best decay checkpoint |
| `models/ufno_history.json` | Per-epoch loss and MAE log |
| `figures/ufno_results/dashboard.png` | 8-panel results dashboard |
| `figures/ufno_results/predictions.csv` | Test-set predictions with errors |
| `figures/cross_basin_*.png` | Cross-basin experiment figures |
| `experiments/exp_NNN_*.json` | Per-experiment logs |

---

## Dependencies

```
torch
numpy
pandas
scikit-learn
xgboost
xarray
netCDF4
shap
tqdm
matplotlib
lime          # optional — LIME explanations skipped if not installed
```

```bash
pip install torch numpy pandas scikit-learn xgboost xarray netCDF4 shap tqdm matplotlib
```
