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

Raw data: `data/raw/Data_1d/GLOBAL/Data1D/<basin>/<split>/`

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/load_tcnd.py` | Shared TCND Data_1d loader |
| `scripts/download_data.py` | TCND data downloader (`--pillars`, `--basins` flags) |
| `scripts/features.py` | Feature engineering — tabular (wind, pressure, position) + spatial scalar summaries + Env-Data aggregation |
| `scripts/preprocess.py` | Extract 925 hPa patches from Data_3d; `--task {decay,landfall,all}`; `--zip` streaming mode |
| `scripts/ablation.py` | XGBoost feature group ablation; writes `selected_feature_groups_{task}.json` |
| `scripts/ufno.py` | CycloneUFNO model architecture |
| `scripts/train_ufno.py` | Training — landfall and decay tasks; `--basin`, `--no-spatial`, `--landfall-ckpt` flags |
| `scripts/evaluate.py` | Standalone evaluation of a saved checkpoint |
| `scripts/view_results.py` | Results dashboard; `--task {decay,landfall}` |
| `scripts/cross_basin.py` | Cross-basin generalization experiments |
| `scripts/explain.py` | SHAP + LIME feature importance (optional) |
| `scripts/log_experiment.py` | Log experiment results to `experiments/` |
| `run_pipeline.py` | End-to-end pipeline orchestrator |

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

`ablation.py` automatically updates `data/features/selected_feature_groups_{task}.json` (one per task), which `train_ufno.py` reads at startup.

---

## Pipeline

> **Important:** `features.py` must run twice — once before preprocessing (to create the CSVs that `preprocess.py` needs for landfall patch indices), and once after (to add `sp_*` spatial scalar summaries from the extracted patches).

### Step 1. Download data
```bash
python scripts/download_data.py
```

### Step 2. First feature pass (no spatial scalars yet)
```bash
python scripts/features.py
```
Creates `feature_matrix_landfall.csv` and `feature_matrix_decay.csv`. The `sp_*` columns will be empty at this stage — that is expected.

### Step 3. Extract spatial patches
```bash
python scripts/preprocess.py --task all
```
Reads `feature_matrix_landfall.csv` to determine which (storm, ref_time) pairs to extract. Saves `(8, 4, 81, 81)` tensors to `data/processed/3d/` (decay) and `data/processed/3d_landfall/` (landfall).

### Step 4. Second feature pass (adds sp_* scalars)
```bash
python scripts/features.py
```
Re-runs with patches now available. Adds `sp_*` mean/std/max/p90/asymmetry summaries per channel.

### Step 5. Feature selection
```bash
python scripts/ablation.py --task all
```
XGBoost ablation over feature groups. Writes `selected_feature_groups_landfall.json` and `selected_feature_groups_decay.json`, which `train_ufno.py` reads at startup.

**Optional — spatial modality ablation** (not included in the end-to-end pipeline): trains the model twice per task (tabular-only vs. with spatial patches) to measure the contribution of 3D input:
```bash
python scripts/ablation.py --task all --spatial-modality --spatial-epochs 20
```

### Step 6. Train — two-stage
**Stage 1: landfall timing model**
```bash
python scripts/train_ufno.py --task landfall --epochs 100 --lr 3e-4 --batch 32
```

**Stage 2: intensity decay model with landfall embedding**
```bash
python scripts/train_ufno.py --task decay --epochs 150 --lr 3e-4 --batch 32 \
    --landfall-ckpt models/best_ufno_landfall.pt
```

Or without landfall embedding:
```bash
python scripts/train_ufno.py --task decay --epochs 150 --lr 3e-4 --batch 32
```

Key options:
```
--task          {landfall,decay}  Task to train                    (default decay)
--landfall-ckpt PATH              Landfall checkpoint for FiLM injection
--no-spatial    flag              Force tabular-only mode (no image patches)
--epochs        int               Training epochs                  (default 100)
--batch         int               Batch size                       (default 8)
--lr            float             Initial learning rate            (default 1e-3)
--modes         int               Fourier modes                    (default 12)
--width         int               UFNO hidden width                (default 32)
--early-stop    int               Early stopping patience          (default 20)
--seed          int               Random seed                      (default 42)
```

### Step 7. View results
```bash
python scripts/view_results.py --task landfall
python scripts/view_results.py --task decay
```
Saves dashboards to `figures/ufno_results/`.

### Step 8. Cross-basin generalization (optional)
```bash
python scripts/cross_basin.py --task landfall --epochs 60
python scripts/cross_basin.py --task decay   --epochs 60
```

### Step 9. Feature importance (optional)
```bash
python scripts/explain.py
```

### Step 10. Log experiment
```bash
python scripts/log_experiment.py \
  --name "Two-stage: landfall + decay with FiLM embedding" \
  --notes "What you observed"
```

---

## Outputs

| Path | Contents |
|---|---|
| `data/features/feature_matrix_decay.csv` | 933 × 35 feature matrix — decay task |
| `data/features/feature_matrix_landfall.csv` | 2389 × 30 feature matrix — landfall task |
| `data/features/feature_groups.json` | Feature group definitions |
| `data/features/selected_feature_groups_decay.json` | Ablation-selected groups for decay (auto-updated) |
| `data/features/selected_feature_groups_landfall.json` | Ablation-selected groups for landfall (auto-updated) |
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
