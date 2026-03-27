# Cyclone Land-Impact Intensity Decay: Experiment Branch

> **Branch:** `reducing-overfitting-experiment`
> This branch contains experimental improvements over the main branch. See [Changes from main](#changes-from-main) for a summary. See [Key findings](#key-findings) for a comparison.

Predicting tropical cyclone **landfall impact** and **post-landfall intensity decay** using the **TropiCycloneNet Dataset (TCND)** and a **U-shaped Fourier Neural Operator (UFNO)**.

---

## Overview

Two prediction tasks:

1. **Landfall timing** — regression: hours remaining until landfall (`feature_matrix_landfall.csv`)
2. **Intensity decay** — regression: wind speed 24 h and 48 h after reference time (`feature_matrix_decay.csv`)

Each task has its own model. The decay model optionally ingests a learned embedding from the frozen landfall model, injected via FiLM conditioning at UFNO blocks 3–5 — so landfall information shapes the spatial field rather than just being another input feature.

---

## Changes from main

### Feature engineering (`features.py`)

| Change | Detail |
|---|---|
| Real land-sea mask | `global-land-mask` (0.1°) + precomputed EDT grids (0.25°), instead of coarse bounding boxes |
| Signed dist-to-coast | Negative = approaching coast, positive = inland penetration depth |
| Env-data encoding | One-hot arrays collapsed to argmax scalar (~26 features vs ~300) |
| Future key stripping | `future_direction24`, `future_inte_change24` removed (were leaking ground-truth labels) |
| Landfall detection | Consecutive coastline crossing (2 steps over land) replaces peak-intensity proxy |
matching |

### Spatial input (`train_ufno.py`)

| Change | Detail |
|---|---|
| 20×20 centre crop | Applied at dataset load time — reduces spatial input from 81×81 to 20×20 |
| Gaussian mask disabled | `spatial_sigma=0`, redundant given hard centre crop |

---

## Key findings

| Metric | Main branch | This branch |
|---|---|---|
| Decay 24h RMSE | 0.1 kt | 0.2 kt |
| Decay 48h RMSE | 0.3 kt | 0.3–0.4 kt |
| Landfall RMSE | 74 h | 72.2 h |
| Decay train/val gap | Large (severe overfitting) | Reduced (better generalisation) |

The trade-off is explicit: this branch reduces overfitting at the cost of slightly worse raw performance on intensity decay. The smaller train/val gap is a more honest estimate of generalisation on unseen data. The right fix remains scaling to the full 70-year TCND dataset.

---

## Dataset

**TropiCycloneNet Dataset (TCND)**

| Property | Value |
|---|---|
| Basins available | WP, NA, EP, NI, SI, SP |
| Basins used | WP, NA, EP |
| Total storms (decay task) | 933 |
| Train / Val / Test | 651 / 188 / 94 (TCND original splits) |
| Format | Tab-separated `.txt`, 8 columns per timestep |

TCND original train/val/test splits are respected throughout, no random re-splitting,to prevent temporal leakage.

Raw data: `data/raw/Data_1d/GLOBAL/Data1D/<basin>/<split>/`

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/load_tcnd.py` | Shared TCND Data_1d loader |
| `scripts/download_data.py` | TCND data downloader (`--pillars`, `--basins` flags) |
| `scripts/features.py` | Improved feature engineering — real land-sea mask, env argmax, pres_delta_from_landfall |
| `scripts/build_land_grids.py` | Precompute EDT distance grids (run once, ~2–3 min) |
| `scripts/preprocess.py` | Extract 925 hPa patches from Data_3d; `--task {decay,landfall,all}` |
| `scripts/ablation.py` | XGBoost feature group ablation; writes `selected_feature_groups_{task}.json` |
| `scripts/ufno.py` | CycloneUFNO model architecture |
| `scripts/train_ufno.py` | Training — landfall and decay tasks; 20×20 centre crop applied here |
| `scripts/evaluate.py` | Standalone evaluation of a saved checkpoint |
| `scripts/view_results.py` | Results dashboard; `--task {decay,landfall}` |
| `scripts/cross_basin.py` | Cross-basin generalization experiments |
| `scripts/model_size_sweep.py` | Model capacity sweep across widths [8, 12, 16, 24, 32] |
| `scripts/explain.py` | SHAP + LIME feature importance (optional) |
| `scripts/log_experiment.py` | Log experiment results to `experiments/` |
| `run_pipeline.py` | End-to-end pipeline orchestrator for this branch |

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
| Parameters | ~700K (width=16) — reduced from ~11M in main branch |

**Task outputs:**
- Landfall: `(B, 1)` — hours to landfall (regression)
- Decay: `(B, 2)` → `[wind_24h, wind_48h]`

**Spatial input:**
- Patches cropped to 20×20 centre region at load time (storm centre at pixel 40,40 of 81×81 patch)
- Tabular-only fallback if no patches available (auto-detected)

Device priority: CUDA → MPS (Apple Silicon) → CPU.

---

## Feature Engineering

Features computed from the 48-hour track window before the reference time.

| Group | Features | Notes |
|---|---|---|
| wind | last, max, mean, std, delta_6h/12h/24h, trend | From Data_1d |
| pressure | last, min, max, mean, std, delta_6h/12h/24h, trend, range, delta_from_landfall | `delta_from_landfall` decay-only |
| wp_couple | wp_residual | Wind-pressure deviation from Dvorak relationship |
| position | lat_last, lon_norm_last, motion_speed, motion_dir_sin/cos | From Data_1d |
| spatial | per-channel mean/std/max/p90/asymmetry (925 hPa) | From Data_3d patches |
| env | argmax-collapsed scalars; ~26 features; future keys stripped | From Env-Data |
| land_sea | over_land, signed dist_to_coast, land_frac_window | Derived; real lon reconstructed from lon_norm |

**Selected groups (ablation-driven):**
- Decay: `env`, `wind`, `pressure`, `wp_couple`, `spatial`
- Landfall: `env`, `spatial`, `position`, `wind`, `wp_couple`, `pressure`

---

## Pipeline

> **Prerequisites:** Install `global-land-mask` before running: `pip install global-land-mask`
>
> **Important:** `features.py` must run twice — once before preprocessing (to create CSVs that `preprocess.py` needs), and once after (to add `sp_*` spatial scalar summaries).

### Quickstart — run everything
```bash
python run_experiment.py
```

### Step-by-step

#### Step 1. Download data
```bash
python scripts/download_data.py
```

#### Step 2. Build land grids (one-time, ~2–3 min)
```bash
python scripts/build_land_grids.py
```
Generates `data/processed/dist_to_coast_025.npy` and `data/processed/dist_inland_025.npy`. Skipped automatically if both files already exist.

#### Step 3. First feature pass
```bash
python scripts/features.py
```
Creates `feature_matrix_landfall.csv` and `feature_matrix_decay.csv`. `sp_*` columns empty at this stage — expected.

#### Step 4. Extract spatial patches
```bash
python scripts/preprocess.py --task all
```
Saves `(8, 4, 81, 81)` tensors to `data/processed/3d/` and `data/processed/3d_landfall/`.

#### Step 5. Second feature pass (adds sp_* scalars)
```bash
python scripts/features.py
```

#### Step 6. Feature selection
```bash
python scripts/ablation.py --task all
```

#### Step 7. Train — two-stage
**Stage 1: landfall timing model**
```bash
python scripts/train_ufno.py --task landfall --epochs 100 --lr 3e-4 --batch 32 \
    --modes 6 --width 12 --early-stop 12 --spatial-sigma 0
```

**Stage 2: intensity decay model with landfall embedding**
```bash
python scripts/train_ufno.py --task decay --epochs 150 --lr 3e-4 --batch 32 \
    --modes 8 --width 16 --early-stop 12 --spatial-sigma 0 \
    --landfall-ckpt models/best_ufno_landfall.pt
```

#### Step 8. View results
```bash
python scripts/view_results.py --task landfall
python scripts/view_results.py --task decay
```

#### Step 9. Cross-basin generalization (optional)
```bash
python scripts/cross_basin.py --task landfall --epochs 60
python scripts/cross_basin.py --task decay   --epochs 60
```

#### Step 10. Model size sweep (optional)
```bash
python scripts/model_size_sweep.py --epochs 50 --early-stop 12 --batch 32 --lr 3e-4
```

#### Step 11. Log experiment
```bash
python scripts/log_experiment.py \
  --name "v8: 20x20 crop, land-sea fix, delta-p, env argmax" \
  --notes "See branch experiment/crop-landsea-envfix for full change log"
```

---

## Outputs

| Path | Contents |
|---|---|
| `data/features/feature_matrix_decay.csv` | 933 × ~35 feature matrix — decay task |
| `data/features/feature_matrix_landfall.csv` | Feature matrix — landfall task |
| `data/features/feature_groups.json` | Feature group definitions |
| `data/features/selected_feature_groups_decay.json` | Ablation-selected groups for decay |
| `data/features/selected_feature_groups_landfall.json` | Ablation-selected groups for landfall |
| `data/features/ablation_*.csv` | Ablation results per task |
| `data/features/model_size_sweep.json` | Model size sweep results |
| `data/features/cross_basin_results_{task}.csv` | Cross-basin RMSE table per task |
| `data/processed/dist_to_coast_025.npy` | EDT distance-to-coast grid (0.25°) |
| `data/processed/dist_inland_025.npy` | EDT distance-inland grid (0.25°) |
| `models/best_ufno_landfall.pt` | Best landfall model checkpoint |
| `models/best_ufno_decay.pt` | Best decay model checkpoint |
| `figures/ufno_results/dashboard.png` | Results dashboard |
| `figures/model_size_sweep.png` | Model size sweep figure |
| `figures/cross_basin_*.png` | Cross-basin figures |
| `experiments/exp_NNN_*.json` | Per-experiment logs |

---

## Dependencies

```
torch
numpy
pandas
scikit-learn
xgboost
shap
tqdm
matplotlib
global-land-mask   # required for land-sea features
lime               # optional — LIME explanations skipped if not installed
```

```bash
pip install torch numpy pandas scikit-learn xgboost shap tqdm matplotlib global-land-mask
```