# Cyclone Land-Impact Intensity Decay

Predicting tropical cyclone intensity decay after landfall using the **TropiCycloneNet Dataset (TCND)** and a **U-shaped Fourier Neural Operator (UFNO)**.

---

## Overview

This project addresses two prediction tasks:

1. **Landfall prediction** — binary classification: will a storm make landfall?
2. **Intensity decay** — regression: what is the storm's wind speed 24 h and 48 h after the reference time?

The pipeline runs from raw TCND tabular tracks through feature engineering, ablation-guided feature selection, and neural operator training.

---

## Dataset

**TropiCycloneNet Dataset (TCND)** — Data_1d format

| Property | Value |
|---|---|
| Basins available | WP, NA, EP, NI, SI, SP |
| Basins used | WP, NA, EP |
| Total storms (decay task) | 933 |
| Train / Val / Test | 651 / 188 / 94 (TCND original splits) |
| Format | Tab-separated `.txt`, 8 columns per timestep |

TCND original train/val/test splits are respected throughout — no random re-splitting — to prevent temporal leakage.

Raw data lives in `data/raw/_tmp/Data1D/<basin>/<split>/`.

---

## Pipeline

```
scripts/load_tcnd.py      # TCND Data_1d loader (shared utility)
       ↓
scripts/features.py       # Feature engineering → data/features/feature_matrix_*.csv
       ↓
scripts/ablation.py       # Ablation study → ranks feature groups by R²
                          # writes data/features/selected_feature_groups.json
       ↓
scripts/explain.py        # SHAP + LIME feature importance (optional)
       ↓
scripts/train_ufno.py     # Train CycloneUFNO; reads TAB_COLS from ablation output
       ↓
scripts/view_results.py   # 8-panel results dashboard
```

---

## Model: CycloneUFNO

Based on **Gege Wen et al. 2022** ([github.com/gegewen/ufno](https://github.com/gegewen/ufno)).

| Component | Description |
|---|---|
| SpectralConv2d | 2D FFT over H×W; 4-quadrant complex weights |
| UNet2d | 3-level encoder-decoder with skip connections |
| FiLM | Tabular features → per-channel scale + shift at every block |
| CycloneUFNOStack | 6 UFNO blocks; UNet applied at blocks 3, 4, 5 |
| Parameters | ~11 M |
| Output | (B, 2) → [wind_24h, wind_48h] |

The model supports two input modes:
- **Spatial** — 3D patch tensors + tabular features (requires `data/processed/`)
- **Tabular-only** — tabular features projected to a 16×16 pseudo-grid (auto-detected fallback)

Device priority: CUDA → MPS (Apple Silicon) → CPU.

---

## Feature Engineering

Features are computed from the 48-hour track window before the reference time.

| Group | Features | Ablation R² (24h) |
|---|---|---|
| wind | last, max, mean, std, delta_6h/12h/24h, trend | 0.269 |
| pressure | last, min, mean, std, delta_6h/12h/24h, trend | 0.260 |
| wp_couple | wp_residual | 0.272 |
| position | lat_last, lon_norm_last, motion_speed, motion_dir | negative |
| land_sea | over_land, dist_to_coast, land_frac_window | negative |

**Selected groups (ablation-driven):** `wp_couple`, `wind`, `pressure` — 17 features total.

Running `ablation.py` automatically updates `data/features/selected_feature_groups.json`, which `train_ufno.py` reads at startup to set `TAB_COLS`.

---

## Usage

### 1. Build features
```bash
python scripts/features.py
```

### 2. Run ablation (updates feature selection automatically)
```bash
python scripts/ablation.py
```

### 3. Train the model
```bash
python scripts/train_ufno.py --epochs 100 --batch 8 --lr 1e-3
```

Options:
```
--epochs   int    Training epochs            (default 100)
--batch    int    Batch size                 (default 8)
--lr       float  Initial learning rate      (default 1e-3)
--modes    int    Fourier modes              (default 12)
--width    int    UFNO hidden width          (default 32)
--no-tab         Disable tabular FiLM conditioning
--seed     int    Random seed                (default 42)
```

### 4. View results
```bash
python scripts/view_results.py
```
Saves an 8-panel dashboard to `figures/ufno_results/dashboard.png`.

### 5. Feature importance (optional)
```bash
python scripts/explain.py
```
Produces SHAP beeswarm/bar plots and LIME local explanations in `figures/`.

---

## Outputs

| Path | Contents |
|---|---|
| `data/features/feature_matrix_decay.csv` | 933 × 35 feature matrix for decay task |
| `data/features/feature_matrix_landfall.csv` | 2389 × 30 feature matrix for landfall task |
| `data/features/feature_groups.json` | Feature group definitions |
| `data/features/selected_feature_groups.json` | Ablation-selected groups (auto-updated) |
| `data/features/ablation_decay_24h.csv` | Ablation results for 24h decay |
| `data/features/ablation_decay_48h.csv` | Ablation results for 48h decay |
| `data/features/ablation_landfall.csv` | Ablation results for landfall prediction |
| `models/best_ufno.pt` | Best validation checkpoint |
| `models/last_ufno.pt` | Final epoch checkpoint |
| `models/ufno_history.json` | Per-epoch loss and MAE log |
| `figures/ufno_results/dashboard.png` | 8-panel results dashboard |
| `figures/ufno_results/predictions.csv` | Test-set predictions with errors |

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
lime          # optional — LIME explanations skipped if not installed
```

Install:
```bash
pip install torch numpy pandas scikit-learn xgboost shap tqdm matplotlib
```
