# Cyclone Land-Impact Intensity Decay

Predicting tropical cyclone **landfall timing** and **post-landfall intensity decay** using the **TropiCycloneNet Dataset (TCND)** and a **U-shaped Fourier Neural Operator (UFNO)** with 925 hPa boundary-layer spatial fields and environmental features.

---

## Overview

Two prediction tasks:

1. **Landfall timing** — regression: hours remaining until the storm dissipates, estimated from the current point in the decay track (`feature_matrix_landfall.csv`, target: `hours_to_landfall`)
2. **Intensity decay** — regression: wind speed 24 h and 48 h after reference time (`feature_matrix_decay.csv`, targets: `wind_24h`, `wind_48h`)

Each task has its own model checkpoint. Both tasks support two input modes — tabular-only and multimodal (tabular + 3D spatial image patches).

**Primary spatial input:** 925 hPa boundary-layer fields (u-wind, v-wind, geopotential z, SST) extracted from TCND Data_3d NetCDF files and preprocessed into `(T=8, C=4, H=81, W=81)` tensors per storm.

**Environmental input:** Pre-computed per-timestep environmental features from TCND Env-Data, aggregated over the lookback window into `_last`, `_mean`, `_trend` scalar summaries (~243 features per sample).

---

## Dataset

**TropiCycloneNet Dataset (TCND)** — Data_1d + Data_3d + Env-Data

| Property | Value |
|---|---|
| Basins available | WP, NA, EP, NI, SI, SP |
| Basins used | WP, NA, EP |
| Total storms (decay task) | 933 |
| Storms with 925 hPa patches | 932 (WP: 604, NA: 214, EP: 114) |
| Landfall matrix samples | 7,167 (4 reference times per storm) |
| Env-Data coverage | WP (1950–2023), NA (1960–2023), EP (1988–2023) |
| Train / Val / Test | 651 / 188 / 94 (TCND original splits) |
| Data_1d format | Tab-separated `.txt`, 8 columns per timestep |
| Data_3d format | NetCDF, 81×81 grid, 4 pressure levels (200/500/850/925 hPa) |
| Env-Data format | Per-timestep `.npy` dicts — 13 keys including area, motion, intensity class, position |

TCND tracks start at or near peak intensity — all track data represents the post-peak decay phase. TCND original train/val/test splits are respected throughout to prevent temporal leakage.

Raw Data_1d: `data/raw/_tmp/Data1D/<basin>/`
Raw Data_3d: `data/raw/_tmp/Data3D/<basin>/`
Raw Env-Data: `data/raw/Env-Data/<basin>/<year>/<storm_name>/<YYYYMMDDHH>.npy`
Processed decay patches: `data/processed/3d/<storm_id>.npy`
Processed landfall patches: `data/processed/3d_landfall/<storm_id>_<YYYYMMDDHH>.npy`

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/load_tcnd.py` | Shared TCND Data_1d loader |
| `scripts/download_data.py` | TCND data downloader (`--pillars`, `--basins` flags) |
| `scripts/preprocess.py` | Extract 925 hPa patches from Data_3d; `--task {decay,landfall,all}`; `--zip` streaming mode |
| `scripts/features.py` | Feature engineering — tabular (wind, pressure, position) + spatial scalar summaries + Env-Data aggregation |
| `scripts/ablation.py` | XGBoost feature group ablation + optional spatial modality ablation (`--spatial-modality`) |
| `scripts/explain.py` | SHAP + LIME feature importance (optional) |
| `scripts/ufno.py` | CycloneUFNO model architecture |
| `scripts/train_ufno.py` | Training — landfall and decay tasks; `--no-spatial` to force tabular-only mode |
| `scripts/view_results.py` | Results dashboard; `--task {decay,landfall}` |
| `scripts/cross_basin.py` | Cross-basin generalization experiments; `--task {decay,landfall}` |
| `scripts/log_experiment.py` | Log experiment results to `experiments/` |

---

## Model: CycloneUFNO

Based on **Gege Wen et al. 2022** ([github.com/gegewen/ufno](https://github.com/gegewen/ufno)).

| Component | Description |
|---|---|
| SpectralConv2d | 2D FFT over H×W; 4-quadrant complex weights |
| UNet2d | 3-level encoder-decoder with skip connections |
| FiLM (tabular) | Tabular features → scale + shift at every block |
| CycloneUFNOStack | 6 UFNO blocks; UNet at blocks 3, 4, 5 |
| Parameters (decay) | ~11 M (width=32, modes=12) |
| Parameters (landfall) | ~1 M (width=16, modes=8) |

**Task outputs:**
- Landfall: `(B, 1)` → `hours_to_landfall`
- Decay: `(B, 2)` → `[wind_24h, wind_48h]`

**Input modes (auto-detected):**
- Spatial — 925 hPa patch tensors `(T, 4, H, W)` + tabular features (requires processed patches)
- Tabular-only — tabular features projected to 16×16 pseudo-grid (fallback or `--no-spatial`)

**Loss functions:**
- Landfall: `MSELoss` (normalised scalar regression)
- Decay: `LpLoss` (p=2, relative norm)

**Optimizer:** AdamW with `ReduceLROnPlateau` (patience=5 landfall, 10 decay).

Device priority: CUDA → MPS (Apple Silicon) → CPU.

---

## Pipeline

### Step 0. Download data

```bash
# Download all pillars (Data_1d, Data_3d, Env-Data) for WP, NA, EP
python scripts/download_data.py

# Or selectively:
python scripts/download_data.py --pillars Env-Data --basins WP NA EP
```

For large Data_3d zips, stream directly without full extraction:
```bash
python scripts/preprocess.py --zip /path/to/TCND_Data3D_WP.zip --basins WP
python scripts/preprocess.py --zip /path/to/TCND_Data3D_NA.zip --basins NA
```

Place Env-Data at `data/raw/Env-Data/<basin>/` (structure: `<year>/<STORM_NAME>/<YYYYMMDDHH>.npy`).

### Step 1. Build spatial patches

```bash
# Build decay patches (one per storm, at peak intensity ref_time)
python scripts/preprocess.py --task decay

# Build landfall patches (one per observation row in feature_matrix_landfall.csv)
python scripts/preprocess.py --task landfall

# Or both at once
python scripts/preprocess.py --task all
```

Saves normalised `(8, 4, 81, 81)` tensors to `data/processed/3d/` and `data/processed/3d_landfall/`.

### Step 2. Build feature matrices

```bash
python scripts/features.py
```

Builds `feature_matrix_landfall.csv` and `feature_matrix_decay.csv` with:
- Tabular features (wind, pressure, position, wp_couple)
- Spatial scalar summaries from 925 hPa patches (`sp_*`)
- Env-Data aggregations — `_last`, `_mean`, `_trend` over LOOKBACK window (`env_*`)

### Step 3. Run ablation

```bash
# XGBoost feature group ablation (fast, ~5 min)
python scripts/ablation.py --task all

# Also run spatial modality ablation — trains UFNO with vs without 3D patches
python scripts/ablation.py --task all --spatial-modality --spatial-epochs 20
```

Auto-updates `selected_feature_groups_{task}.json` which `train_ufno.py` reads at startup.

### Step 4. Train

```bash
# Landfall timing model
python scripts/train_ufno.py --task landfall --epochs 100 --lr 3e-4 --batch 32

# Decay model
python scripts/train_ufno.py --task decay --epochs 150 --lr 3e-4 --batch 32

# Force tabular-only (no spatial patches)
python scripts/train_ufno.py --task decay --no-spatial --epochs 150
```

Key options:
```
--task          {landfall,decay}    Task to train              (default decay)
--epochs        int                 Training epochs            (default 100)
--batch         int                 Batch size                 (default 8)
--lr            float               Initial LR                 (default 1e-3)
--modes         int                 Fourier modes              (default 12; landfall: 8)
--width         int                 UFNO hidden width          (default 32; landfall: 16)
--no-spatial    flag                Force tabular-only mode
--early-stop    int                 Early stopping patience    (default 20)
--seed          int                 Random seed                (default 42)
```

### Step 5. View results

```bash
python scripts/view_results.py --task landfall
python scripts/view_results.py --task decay
```

Saves a 6-panel (landfall) or 8-panel (decay) dashboard to `figures/`.

### Step 6. Cross-basin generalization (optional)

```bash
python scripts/cross_basin.py --task landfall --epochs 60
python scripts/cross_basin.py --task decay   --epochs 60
```

Trains on one basin, tests on all others. Saves heatmap, gradual addition curves, and summary bar chart to `figures/`.

### Step 7. Feature importance (optional)

```bash
python scripts/explain.py
```

---

## Feature Engineering

| Group | Source | Key Features | Ablation R² (decay 24h) |
|---|---|---|---|
| wp_couple | Data_1d | wind-pressure residual | 0.272 |
| wind | Data_1d | last, max, mean, std, delta_6h/12h/24h, trend | 0.269 |
| pressure | Data_1d | last, min, mean, std, delta_6h/12h/24h, trend | 0.216 |
| spatial | Data_3d 925 hPa | mean, std, max, p90, asymmetry × 4 channels | 0.093 |
| env | Env-Data | area, intensity class, motion, seasonality, position (last/mean/trend) | TBD after re-run |
| position | Data_1d | lat, lon_norm, motion_speed, motion_dir_sin/cos | negative |

**Selected groups:**
- Decay: `wp_couple`, `wind`, `pressure`, `spatial`
- Landfall: `spatial`, `position`, `wp_couple`, `pressure`, `wind`

`motion_dir_deg` is encoded as `motion_dir_sin` / `motion_dir_cos` to avoid circular discontinuity.

The `sp_*` scalar summaries and `env_*` aggregations are used by the XGBoost ablation. The actual `(8, 4, 81, 81)` image tensors are only consumed by the UFNO spatial encoder — their contribution is measured separately by `--spatial-modality`.

---

## Cross-Basin Results

Training on one basin and testing on others reveals large distribution shifts. Key findings:

- **WP ↔ EP**: good transfer (~70h RMSE), both Pacific basins share similar storm dynamics
- **NA ↔ anything**: poor transfer (exploding RMSE), distinct latitude/SST/environmental regime
- **Top distribution shift features**: `lat_last` (2.34σ, real), `sp_z925_*` (2.09σ, fill-value artifact when Data_3d missing), `sp_u/v925_std` (real WP monsoon trough effect)

See `figures/cross_basin_distribution_shift.png` for full analysis.

---

## Outputs

| Path | Contents |
|---|---|
| `data/processed/3d/<storm_id>.npy` | (8, 4, 81, 81) normalised decay patch per storm |
| `data/processed/3d_landfall/<storm_id>_<ref>.npy` | (8, 4, 81, 81) normalised landfall patch per observation |
| `data/processed/3d_stats.json` | Per-channel global mean/std for decay patches |
| `data/processed/3d_landfall_stats.json` | Per-channel global mean/std for landfall patches |
| `data/features/feature_matrix_decay.csv` | Decay feature matrix |
| `data/features/feature_matrix_landfall.csv` | Landfall feature matrix |
| `data/features/feature_groups.json` | Feature group definitions |
| `data/features/selected_feature_groups_decay.json` | Ablation-selected groups for decay |
| `data/features/selected_feature_groups_landfall.json` | Ablation-selected groups for landfall |
| `data/features/ablation_*.csv` | Ablation results per task |
| `data/features/ablation_spatial_modality.csv` | Spatial modality ablation results |
| `data/features/cross_basin_results_{task}.csv` | Cross-basin experiment RMSE table |
| `models/best_ufno_landfall.pt` | Best landfall checkpoint (includes tab_mean/tab_scale) |
| `models/best_ufno_decay.pt` | Best decay checkpoint |
| `models/ufno_history_landfall.json` | Per-epoch loss log — landfall |
| `models/ufno_history_decay.json` | Per-epoch loss log — decay |
| `figures/cross_basin_distribution_shift.png` | Basin distribution shift analysis |
| `figures/cross_basin_heatmap_{task}.png` | Cross-basin transfer RMSE heatmap |
| `figures/cross_basin_gradual_{task}.png` | Gradual basin addition curves |
| `figures/cross_basin_summary_{task}.png` | In-domain vs cross-basin comparison |
| `figures/ablation_{task}_{metric}.png` | Ablation bar charts |

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
gdown
```

```bash
pip install torch numpy pandas scikit-learn xgboost xarray netCDF4 shap tqdm matplotlib gdown
```
