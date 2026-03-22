# Cyclone Land-Impact Intensity Decay

Predicting tropical cyclone **landfall impact** and **post-landfall intensity decay** using the **TropiCycloneNet Dataset (TCND)** and a **U-shaped Fourier Neural Operator (UFNO)**.

---

## Overview

Two prediction tasks:

1. **Landfall impact** — binary classification: will this storm make landfall? (`feature_matrix_landfall.csv`)
2. **Intensity decay** — regression: wind speed 24 h and 48 h after reference time (`feature_matrix_decay.csv`)

Each task has its own model. The decay model optionally ingests a learned embedding from the frozen landfall model, injected via FiLM conditioning at UFNO blocks 3–5 — so landfall information shapes the spatial field rather than just being another input feature.

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

Raw data: `data/raw/_tmp/Data1D/<basin>/<split>/`

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/load_tcnd.py` | Shared TCND Data_1d loader |
| `scripts/features.py` | Feature engineering from Data_1d |
| `scripts/features_improved.py` | Extended features: Data_3d (pressure drop, wind shear, SST), Env-Data, improved land-sea |
| `scripts/ablation.py` | Ablation study — ranks feature groups by R², writes `selected_feature_groups.json` |
| `scripts/explain.py` | SHAP + LIME feature importance (optional) |
| `scripts/ufno.py` | CycloneUFNO model architecture |
| `scripts/train_ufno.py` | Training — landfall and decay tasks |
| `scripts/view_results.py` | 8-panel results dashboard |
| `scripts/log_experiment.py` | Log experiment results to `experiments/` |
| `scripts/cross_basin.py` | Cross-basin generalization experiments |
| `scripts/download_data.py` | TCND data downloader |
| `scripts/preprocess.py` | Preprocessing for spatial (Data_3d) mode |

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
- Landfall: `(B, 1)` logit — apply sigmoid for P(landfall)
- Decay: `(B, 2)` → `[wind_24h, wind_48h]`

**Input modes:**
- Spatial — 3D patch tensors + tabular (requires `data/processed/`)
- Tabular-only — tabular features projected to 16×16 pseudo-grid (auto-detected fallback)

Device priority: CUDA → MPS (Apple Silicon) → CPU.

---

## Feature Engineering

Features computed from the 48-hour track window before the reference time.

| Group | Features | Ablation R² (24h) |
|---|---|---|
| wind | last, max, mean, std, delta_6h/12h/24h, trend | 0.269 |
| pressure | last, min, mean, std, delta_6h/12h/24h, trend | 0.260 |
| wp_couple | wp_residual | 0.272 |
| position | lat_last, lon_norm_last, motion_speed, motion_dir | negative |
| land_sea | over_land, dist_to_coast, land_frac_window | negative |

**Selected groups (ablation-driven):** `wp_couple`, `wind`, `pressure` — 17 features total.

`ablation.py` automatically updates `data/features/selected_feature_groups.json`, which `train_ufno.py` reads at startup.

---

## Usage

### 1. Build features
```bash
python scripts/features.py
# or, for extended 3D/env features:
python scripts/features_improved.py
```

### 2. Run ablation (auto-updates feature selection)
```bash
python scripts/ablation.py
```

### 3. Train — two-stage

**Step 1: landfall model**
```bash
python scripts/train_ufno.py --task landfall --epochs 100
```

**Step 2: decay model with landfall embedding**
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
--lr            float             Initial learning rate     (default 1e-3)
--modes         int               Fourier modes             (default 12)
--width         int               UFNO hidden width         (default 32)
--unet-dropout  float             UNet dropout              (default 0.2)
--seed          int               Random seed               (default 42)
```

### 4. View results
```bash
python scripts/view_results.py --show
```
Saves an 8-panel dashboard to `figures/ufno_results/dashboard.png`.

### 5. Log the experiment
```bash
python scripts/log_experiment.py \
  --name "Two-stage: landfall embedding in decay model" \
  --notes "What you observed"
```

### 6. Feature importance (optional)
```bash
python scripts/explain.py
```

### 7. Cross-basin generalization (optional)
```bash
python scripts/cross_basin.py --epochs 60
```
Produces transfer RMSE heatmap, gradual basin addition curves, and in-domain vs transfer comparison.

---

## Outputs

| Path | Contents |
|---|---|
| `data/features/feature_matrix_decay.csv` | 933 × 35 feature matrix — decay task |
| `data/features/feature_matrix_landfall.csv` | 2389 × 30 feature matrix — landfall task |
| `data/features/feature_groups.json` | Feature group definitions |
| `data/features/selected_feature_groups.json` | Ablation-selected groups (auto-updated) |
| `data/features/ablation_*.csv` | Ablation results per task |
| `data/features/cross_basin_results.csv` | Cross-basin experiment RMSE table |
| `models/best_ufno_landfall.pt` | Best landfall model checkpoint |
| `models/best_ufno_decay.pt` | Best decay model checkpoint |
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
shap
tqdm
matplotlib
lime          # optional — LIME explanations skipped if not installed
```

```bash
pip install torch numpy pandas scikit-learn xgboost shap tqdm matplotlib
```
