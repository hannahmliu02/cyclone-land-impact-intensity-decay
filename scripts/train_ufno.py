"""
Train CycloneUFNO — landfall impact or intensity decay.

Two tasks
─────────
  landfall  — regression: hours until landfall (hours_to_landfall)
               Uses feature_matrix_landfall.csv, MSELoss
               Non-landfall storms have hours_to_landfall = -1 (sentinel)
               Saves: best_ufno_landfall[_BASIN].pt

  decay     — regression: wind speed 24h and 48h after reference time
               Uses feature_matrix_decay.csv, LpLoss
               Saves: best_ufno_decay[_BASIN].pt
               Optionally accepts --landfall-ckpt to inject a learned
               landfall embedding into the model via FiLM at blocks 3-5

Two data modes (auto-detected):
  spatial   — processed tensors from preprocess.py exist in data/processed/
  tabular   — only feature_matrix_*.csv is available (default fallback)

Usage
─────
  # Global training
  python scripts/train_ufno.py --task landfall --epochs 100
  python scripts/train_ufno.py --task decay --epochs 150 \
      --landfall-ckpt models/best_ufno_landfall.pt

  # Per-basin training
  python scripts/train_ufno.py --task landfall --basin EP --epochs 100
  python scripts/train_ufno.py --task decay --basin EP --epochs 150 \
      --landfall-ckpt models/best_ufno_landfall_EP.pt

Options
  --task          {landfall,decay}   Which task to train      (default decay)
  --basin         {WP,NA,EP}         Train on one basin only  (default all)
  --landfall-ckpt PATH               Landfall checkpoint for embedding injection
  --epochs        int                Training epochs          (default 100)
  --batch         int                Batch size               (default 8)
  --lr            float              Initial LR               (default 1e-3)
  --modes         int                Fourier modes            (default 12)
  --width         int                UFNO hidden width        (default 32)
  --unet-dropout  float              UNet dropout             (default 0.2)
  --no-tab                           Disable tabular FiLM conditioning
  --no-spatial                       Force tabular-only mode
  --early-stop    int                Patience epochs          (default 20)
  --seed          int                Random seed              (default 42)
  --save-name     str                Override checkpoint stem
  --noise-std     float              Gaussian noise std on tabular features
                                     during training (default 0.0, try 0.02)
  --spatial-sigma float              Gaussian centre mask sigma in pixels
                                     (default 0.0 = off, try 10 for tight focus)

Outputs (models/)
  best_ufno_{task}[_{basin}].pt    — best checkpoint (saved on every improvement)
  last_ufno_{task}[_{basin}].pt    — final epoch checkpoint
  ufno_history_{task}[_{basin}].json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(__file__))
from ufno import CycloneUFNO, LpLoss

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEAT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

_FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")

VALID_BASINS = {"WP", "NA", "EP"}

_FALLBACK_COLS = {
    "landfall": [
        "wp_residual",
        "wind_last", "wind_max", "wind_mean", "wind_std",
        "wind_delta_6h", "wind_delta_12h", "wind_delta_24h", "wind_trend",
        "pres_last", "pres_min", "pres_mean", "pres_std",
        "pres_delta_6h", "pres_delta_12h", "pres_delta_24h", "pres_trend",
    ],
    "decay": [
        "wind_last", "wind_max", "wind_mean", "wind_std",
        "wind_delta_6h", "wind_delta_12h", "wind_delta_24h", "wind_trend",
        "pres_last", "pres_min", "pres_mean", "pres_std",
        "pres_delta_6h", "pres_delta_12h", "pres_delta_24h", "pres_trend",
        "wp_residual",
    ],
}


def _load_tab_cols(task: str) -> list:
    """Load tabular feature columns from ablation output, with fallback."""
    grp_path  = os.path.join(_FEAT_DIR, "feature_groups.json")
    fallback  = _FALLBACK_COLS[task]
    candidates = [
        os.path.join(_FEAT_DIR, f"selected_feature_groups_{task}.json"),
        os.path.join(_FEAT_DIR, "selected_feature_groups.json"),
    ]
    sel_path = next((p for p in candidates if os.path.exists(p)), None)

    if sel_path is None or not os.path.exists(grp_path):
        print(f"[train_ufno] No ablation selection for '{task}' — using fallback.")
        return fallback

    with open(sel_path) as f:
        sel = json.load(f)
    with open(grp_path) as f:
        groups = json.load(f)

    selected_groups = sel.get("selected_groups", [])
    cols = []
    for g in selected_groups:
        cols.extend(groups.get(g, []))
    cols = list(dict.fromkeys(cols))

    if not cols:
        print(f"[train_ufno] Ablation for '{task}' produced no columns — using fallback.")
        return fallback

    print(f"[train_ufno] TAB_COLS ({task}): {len(cols)} features "
          f"from groups {selected_groups}  [{os.path.basename(sel_path)}]")
    return cols


TARGET_COLS     = ["wind_24h", "wind_48h"]
LANDFALL_TARGET = "hours_to_landfall"


# ── Datasets ──────────────────────────────────────────────────────────────────

class LandfallDataset(Dataset):
    def __init__(self, df, tab_scaler, tgt_scaler, fit=False, tab_cols=None):
        df = df[df[LANDFALL_TARGET] > 0].reset_index(drop=True)
        tab_cols = [c for c in (tab_cols or []) if c in df.columns]
        X = df[tab_cols].replace("", np.nan).apply(
            pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        y = df[[LANDFALL_TARGET]].values.astype(np.float32)
        X = tab_scaler.fit_transform(X) if fit else tab_scaler.transform(X)
        y = tgt_scaler.fit_transform(y) if fit else tgt_scaler.transform(y)
        self.X    = torch.tensor(X, dtype=torch.float32)
        self.y    = torch.tensor(y, dtype=torch.float32)
        self.meta = df[["storm_id", "basin"]].reset_index(drop=True)

    def __len__(self):         return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i]


class SpatialLandfallDataset(Dataset):
    LF_PATCH_DIR = os.path.join(PROC_DIR, "3d_landfall")

    def __init__(self, df, tab_scaler, tgt_scaler, fit=False, tab_cols=None):
        df = df[df[LANDFALL_TARGET] > 0].reset_index(drop=True)
        tab_cols = [c for c in (tab_cols or []) if c in df.columns]
        X = df[tab_cols].replace("", np.nan).apply(
            pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        y = df[[LANDFALL_TARGET]].values.astype(np.float32)
        X = tab_scaler.fit_transform(X) if fit else tab_scaler.transform(X)
        y = tgt_scaler.fit_transform(y) if fit else tgt_scaler.transform(y)
        self.X_tab = torch.tensor(X, dtype=torch.float32)
        self.y     = torch.tensor(y, dtype=torch.float32)
        self.meta  = df[["storm_id", "basin"]].reset_index(drop=True)
        self.keys  = [
            f"{row['storm_id']}_{pd.Timestamp(row['ref_time']).strftime('%Y%m%d%H')}"
            for _, row in df.iterrows()
        ]

    def __len__(self): return len(self.X_tab)

    def __getitem__(self, idx):
        npy = os.path.join(self.LF_PATCH_DIR, f"{self.keys[idx]}.npy")
        if os.path.exists(npy):
            patch = np.load(npy)
            patch = patch[:, :, 30:50, 30:50]
            x3d = torch.tensor(patch, dtype=torch.float32)
        else:
            x3d = None
        return x3d, self.X_tab[idx], self.y[idx]


class TabularDecayDataset(Dataset):
    def __init__(self, df, tab_scaler, tgt_scaler, fit=False, tab_cols=None):
        tab_cols = [c for c in (tab_cols or []) if c in df.columns]
        # X = df[tab_cols].values.astype(np.float32)
        X = df[tab_cols].replace("", np.nan).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        y = df[TARGET_COLS].values.astype(np.float32)
        X = tab_scaler.fit_transform(X) if fit else tab_scaler.transform(X)
        y = tgt_scaler.fit_transform(y) if fit else tgt_scaler.transform(y)
        self.X    = torch.tensor(X, dtype=torch.float32)
        self.y    = torch.tensor(y, dtype=torch.float32)
        self.meta = df[["storm_id", "basin"]].reset_index(drop=True)

    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SpatialDecayDataset(Dataset):
    def __init__(self, df, tab_scaler, tgt_scaler, fit=False, tab_cols=None):
        tab_cols = [c for c in (tab_cols or []) if c in df.columns]
        # X_tab = df[tab_cols].values.astype(np.float32)
        X_tab = df[tab_cols].replace("", np.nan).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        y     = df[TARGET_COLS].values.astype(np.float32)
        X_tab = tab_scaler.fit_transform(X_tab) if fit else tab_scaler.transform(X_tab)
        y     = tgt_scaler.fit_transform(y)     if fit else tgt_scaler.transform(y)
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)
        self.ids   = df["storm_id"].tolist()
        self.meta  = df[["storm_id", "basin"]].reset_index(drop=True)

    def __len__(self): return len(self.X_tab)

    def __getitem__(self, idx):
        npy = os.path.join(PROC_DIR, "3d", f"{self.ids[idx]}.npy")
        if os.path.exists(npy):
            patch = np.load(npy)              # (T, C, 81, 81)
            patch = patch[:, :, 30:50, 30:50] # centre 20x20 crop
            x3d = torch.tensor(patch, dtype=torch.float32)
        else:
            x3d = None
        return x3d, self.X_tab[idx], self.y[idx]


def _collate_spatial(batch):
    x3ds, x_tabs, ys = zip(*batch)
    x3d_out = None if any(x is None for x in x3ds) else torch.stack(x3ds)
    return x3d_out, torch.stack(x_tabs), torch.stack(ys)


# ── Data loading ──────────────────────────────────────────────────────────────

def _split_df(df, seed):
    """TCND original splits if available, else random 70/15/15."""
    if "split" in df.columns and df["split"].isin(["train", "val", "test"]).any():
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df   = df[df["split"] == "val"].reset_index(drop=True)
        test_df  = df[df["split"] == "test"].reset_index(drop=True)
        print(f"Split      : TCND original  "
              f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
    else:
        train_df, tmp = train_test_split(df, test_size=0.3, random_state=seed)
        val_df, test_df = train_test_split(tmp, test_size=0.5, random_state=seed)
        print(f"Split      : random 70/15/15  "
              f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
    return train_df, val_df, test_df


def _filter_basin(df: pd.DataFrame, basin: str) -> pd.DataFrame:
    """Filter dataframe to a single basin, with a helpful error if empty."""
    if basin is None:
        return df
    out = df[df["basin"] == basin].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(
            f"No rows found for basin '{basin}'. "
            f"Available: {sorted(df['basin'].unique())}")
    return out


def _load_csv(task: str, basin: str) -> pd.DataFrame:
    """
    Load feature matrix for a task.
    Looks for feature_matrix_{task}_{basin}.csv first (pre-filtered),
    then falls back to filtering the full matrix on the fly.
    """
    # Try basin-specific file first
    if basin is not None:
        specific = os.path.join(FEAT_DIR, f"feature_matrix_{task}_{basin}.csv")
        if os.path.exists(specific):
            print(f"Loading basin-specific file: {os.path.basename(specific)}")
            return pd.read_csv(specific, keep_default_na=False)

    # Fall back to full matrix + filter
    full = os.path.join(FEAT_DIR, f"feature_matrix_{task}.csv")
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"Feature matrix not found: {full}\nRun features.py first.")
    df = pd.read_csv(full, keep_default_na=False)
    if basin is not None:
        print(f"Filtering full matrix to basin={basin} (no pre-filtered file found)")
        df = _filter_basin(df, basin)
    return df


def load_data(seed: int, task: str = "decay", batch: int = 8,
              no_spatial: bool = False, basin: str = None):
    """Return (train_loader, val_loader, test_loader, meta, tgt_scaler, tab_scaler)."""
    tab_cols_for_task = _load_tab_cols(task)

    if task == "landfall":
        df = _load_csv("landfall", basin)
        df = df.dropna(subset=[LANDFALL_TARGET]).reset_index(drop=True)
        df_lf = df[pd.to_numeric(df[LANDFALL_TARGET], errors="coerce") > 0
                   ].reset_index(drop=True)

        lf_patch_dir = os.path.join(PROC_DIR, "3d_landfall")
        has_patches  = (not no_spatial
                        and os.path.isdir(lf_patch_dir)
                        and any(f.endswith(".npy") for f in os.listdir(lf_patch_dir)))
        mode = "spatial" if has_patches else "tabular"

        print(f"Task       : landfall")
        print(f"Basin      : {basin or 'all'}")
        print(f"Data mode  : {mode}")
        print(f"Rows       : {len(df_lf)}  |  basins: "
              f"{df_lf['basin'].value_counts().to_dict()}")
        print(f"hrs range  : [{pd.to_numeric(df_lf[LANDFALL_TARGET]).min():.0f}, "
              f"{pd.to_numeric(df_lf[LANDFALL_TARGET]).max():.0f}] h")

        tab_scaler = StandardScaler()
        tgt_scaler = StandardScaler()
        train_df, val_df, test_df = _split_df(df_lf, seed)

        DS, collate = (SpatialLandfallDataset, _collate_spatial) \
                      if mode == "spatial" else (LandfallDataset, None)

        train_ds = DS(train_df, tab_scaler, tgt_scaler, fit=True,  tab_cols=tab_cols_for_task)
        val_ds   = DS(val_df,   tab_scaler, tgt_scaler, fit=False, tab_cols=tab_cols_for_task)
        test_ds  = DS(test_df,  tab_scaler, tgt_scaler, fit=False, tab_cols=tab_cols_for_task)

        basin_counts   = train_df["basin"].value_counts().to_dict()
        sample_weights = train_df["basin"].map(
            lambda b: 1.0 / basin_counts.get(b, 1)).values
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                        replacement=True)

        tab_cols = [c for c in tab_cols_for_task if c in df.columns]
        meta = {
            "task": "landfall", "mode": mode, "basin": basin,
            "tab_dim": len(tab_cols), "tab_cols": tab_cols,
            "tgt_mean": tgt_scaler.mean_.tolist(),
            "tgt_scale": tgt_scaler.scale_.tolist(),
            "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        }
        kw = dict(batch_size=batch, num_workers=0, collate_fn=collate)
        return (DataLoader(train_ds, sampler=sampler, **kw),
                DataLoader(val_ds,   shuffle=False,   **kw),
                DataLoader(test_ds,  shuffle=False,   **kw),
                meta, tgt_scaler, tab_scaler)

    # ── decay ──────────────────────────────────────────────────────────────
    df = _load_csv("decay", basin)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)

    patch_dir   = os.path.join(PROC_DIR, "3d")
    has_patches = (not no_spatial
                   and os.path.isdir(patch_dir)
                   and any(f.endswith(".npy") for f in os.listdir(patch_dir)))
    mode = "spatial" if has_patches else "tabular"

    print(f"Task       : decay")
    print(f"Basin      : {basin or 'all'}")
    print(f"Data mode  : {mode}")
    print(f"Samples    : {len(df)}  |  basins: {df['basin'].value_counts().to_dict()}")

    tab_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    train_df, val_df, test_df = _split_df(df, seed)

    DS, collate = (SpatialDecayDataset, _collate_spatial) \
                  if mode == "spatial" else (TabularDecayDataset, None)

    train_ds = DS(train_df, tab_scaler, tgt_scaler, fit=True,  tab_cols=tab_cols_for_task)
    val_ds   = DS(val_df,   tab_scaler, tgt_scaler, fit=False, tab_cols=tab_cols_for_task)
    test_ds  = DS(test_df,  tab_scaler, tgt_scaler, fit=False, tab_cols=tab_cols_for_task)

    basin_counts   = train_df["basin"].value_counts().to_dict()
    sample_weights = train_df["basin"].map(
        lambda b: 1.0 / basin_counts.get(b, 1)).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)

    tab_cols = [c for c in tab_cols_for_task if c in df.columns]
    meta = {
        "task": "decay", "mode": mode, "basin": basin,
        "tab_dim": len(tab_cols), "tab_cols": tab_cols,
        "tgt_mean": tgt_scaler.mean_.tolist(),
        "tgt_scale": tgt_scaler.scale_.tolist(),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
    }
    kw = dict(batch_size=batch, num_workers=0, collate_fn=collate)
    return (DataLoader(train_ds, sampler=sampler, **kw),
            DataLoader(val_ds,   shuffle=False,   **kw),
            DataLoader(test_ds,  shuffle=False,   **kw),
            meta, tgt_scaler, tab_scaler)


# ── Forward pass ──────────────────────────────────────────────────────────────

def _forward(model, batch, mode, device, lf_model=None, noise_std=0.0):
    if mode == "tabular":
        x_tab, y = batch
        x_tab = x_tab.to(device); y = y.to(device)
        if noise_std > 0.0:
            x_tab = x_tab + torch.randn_like(x_tab) * noise_std
        lf_embed = None
        if lf_model is not None:
            with torch.no_grad():
                # tabular-only mode: no spatial patches available,
                # so we cannot extract a spatial embedding from lf_model.
                # Skip lf_embed entirely to avoid tab_dim mismatch.
                lf_embed = None
        return model(x_tab=x_tab, lf_embed=lf_embed), y
    else:
        x3d, x_tab, y = batch
        x3d   = x3d.to(device) if x3d is not None else None
        x_tab = x_tab.to(device); y = y.to(device)
        if noise_std > 0.0:
            x_tab = x_tab + torch.randn_like(x_tab) * noise_std
        lf_embed = None
        if lf_model is not None:
            with torch.no_grad():
                # Use x_3d only — avoids tab_dim mismatch between landfall
                # (345 features) and decay (340 features) feature sets.
                # The spatial field is the same shape for both tasks.
                lf_embed = lf_model.extract_embedding(x_3d=x3d)                            if x3d is not None else None
        return model(x_3d=x3d, x_tab=x_tab, lf_embed=lf_embed), y


def run_epoch(model, loader, mode, device, optimizer=None, criterion=None,
              desc="", show_bar=True, lf_model=None, noise_std=0.0):
    from tqdm import tqdm
    if criterion is None:
        criterion = LpLoss(p=2)
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = total_mae = n = 0
    bar = tqdm(loader, desc=desc, leave=False, ncols=72,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining} loss={postfix}]") \
          if show_bar else loader

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in bar:
            pred, y = _forward(model, batch, mode, device, lf_model=lf_model,
                               noise_std=noise_std if training else 0.0)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_mae  += (pred - y).abs().mean().item() * bs
            n += bs
            if show_bar:
                bar.set_postfix_str(f"{total_loss/n:.4f}")

    return total_loss / n, total_mae / n


# ── ASCII sparkline ───────────────────────────────────────────────────────────

_SPARKS = " ▁▂▃▄▅▆▇█"

def _sparkline(values, width=30):
    if not values: return ""
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    rng = hi - lo or 1e-9
    return "".join(_SPARKS[min(8, int((v - lo) / rng * 8))] for v in tail)

def _loss_bar(train_vals, val_vals, width=38):
    if len(train_vals) < 2: return
    print(f"\n  Train {_sparkline(train_vals, width)}  lo={min(train_vals):.4f}")
    print(f"  Val   {_sparkline(val_vals,   width)}  lo={min(val_vals):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="decay", choices=["landfall", "decay"])
    parser.add_argument("--basin",  default=None,
                        help="Train on one basin only: WP, NA, or EP")
    parser.add_argument("--landfall-ckpt", default=None)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch",       type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--modes",       type=int,   default=12)
    parser.add_argument("--width",       type=int,   default=32)
    parser.add_argument("--unet-dropout",type=float, default=0.2)
    parser.add_argument("--no-tab",      action="store_true")
    parser.add_argument("--no-spatial",  action="store_true")
    parser.add_argument("--early-stop",  type=int,   default=20)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--save-name",   type=str,   default=None)
    parser.add_argument("--noise-std",   type=float, default=0.0,
                        help="Std of Gaussian noise added to tabular features "
                             "during training only (0 = disabled). "
                             "Features are StandardScaler-normalised so "
                             "0.02 means ~2%% perturbation. Recommended: 0.01-0.05.")
    parser.add_argument("--spatial-sigma", type=float, default=0.0,
                        help="Std of Gaussian centre mask applied to spatial "
                             "field (pixels, 0 = disabled). sigma=10 focuses "
                             "on the inner ~2.5 deg storm core. No effect in "
                             "tabular-only mode.")
    args = parser.parse_args()

    # Validate basin
    if args.basin is not None:
        args.basin = args.basin.upper()
        if args.basin not in VALID_BASINS:
            print(f"Error: --basin must be one of {sorted(VALID_BASINS)}")
            sys.exit(1)

    # Landfall defaults — smaller still since it's a simpler task
    if args.task == "landfall":
        if args.modes == 12: args.modes = 8
        if args.width == 32: args.width = 16

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, test_loader, meta, tgt_scaler, tab_scaler = \
        load_data(args.seed, task=args.task, batch=args.batch,
                  no_spatial=args.no_spatial, basin=args.basin)

    mode = meta["mode"]
    task = args.task

    # ── Checkpoint naming — includes basin suffix if per-basin ─────────────
    basin_tag = f"_{args.basin}" if args.basin else ""
    if args.save_name:
        ckpt_name = f"{args.save_name}.pt"
    else:
        ckpt_name = f"best_ufno_{task}{basin_tag}.pt"
    hist_name = f"ufno_history_{task}{basin_tag}.json"

    # ── Frozen landfall model ───────────────────────────────────────────────
    lf_model     = None
    lf_embed_dim = 0
    if task == "decay" and args.landfall_ckpt:
        if not os.path.exists(args.landfall_ckpt):
            raise FileNotFoundError(
                f"Landfall checkpoint not found: {args.landfall_ckpt}\n"
                f"Train landfall first: python scripts/train_ufno.py "
                f"--task landfall{' --basin ' + args.basin if args.basin else ''}")
        lf_ckpt  = torch.load(args.landfall_ckpt, map_location=DEVICE)
        lf_args  = lf_ckpt.get("args", {})
        lf_meta  = lf_ckpt.get("meta", {})
        lf_model = CycloneUFNO(
            sp_channels   = 4,
            T             = 8,
            tab_features  = lf_meta.get("tab_dim", meta["tab_dim"]),
            modes1        = lf_args.get("modes", 8),
            modes2        = lf_args.get("modes", 8),
            width         = lf_args.get("width", 16),
            unet_dropout  = lf_args.get("unet_dropout", 0.0),
            n_outputs     = 1,
            spatial_sigma = lf_args.get("spatial_sigma", 0.0),
        ).to(DEVICE)
        lf_model.load_state_dict(lf_ckpt["state"])
        lf_model.eval()
        for p in lf_model.parameters():
            p.requires_grad = False
        lf_embed_dim = lf_args.get("width", 16)
        print(f"Landfall ckpt  : {args.landfall_ckpt}")
        print(f"LF embed dim   : {lf_embed_dim}")

    # ── Build model ─────────────────────────────────────────────────────────
    n_outputs = 1 if task == "landfall" else 2
    model = CycloneUFNO(
        sp_channels  = 4,
        T            = 8,
        tab_features = meta["tab_dim"],
        modes1       = args.modes,
        modes2       = args.modes,
        width        = args.width,
        unet_dropout = args.unet_dropout,
        n_outputs    = n_outputs,
        lf_embed_dim = lf_embed_dim,
        spatial_sigma = args.spatial_sigma,
    ).to(DEVICE)

    print(f"\nTask       : {task}  |  Basin: {args.basin or 'all'}")
    print(f"Model      : CycloneUFNO  ({model.count_params():,} params)")
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {args.epochs}  |  Early stop: {args.early_stop}")
    print(f"Checkpoint : {ckpt_name}")

    criterion = nn.MSELoss() if task == "landfall" else LpLoss(p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    patience  = 5 if task == "landfall" else 10
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience, min_lr=1e-5)

    history       = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val      = float("inf")
    epochs_no_imp = 0

    header = (f"{'Epoch':>6}  {'Train':>8}  {'Val':>8}  "
              f"{'ValMAE':>8}  {'LR':>8}  {'Time':>6}  {'':>4}")
    print(f"\n{header}")
    print("─" * len(header))
    sys.stdout.flush()

    for epoch in range(1, args.epochs + 1):
        t0     = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        tr_loss, tr_mae = run_epoch(
            model, train_loader, mode, DEVICE, optimizer, criterion,
            desc=f"Ep {epoch:>3} train", show_bar=True, lf_model=lf_model,
            noise_std=args.noise_std)
        vl_loss, vl_mae = run_epoch(
            model, val_loader, mode, DEVICE, criterion=criterion,
            desc=f"Ep {epoch:>3} val  ", show_bar=True, lf_model=lf_model)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)

        is_best = vl_loss < best_val
        if is_best:
            best_val      = vl_loss
            epochs_no_imp = 0
            # ── Save best checkpoint immediately on improvement ──────────
            ckpt = {
                "epoch": epoch, "state": model.state_dict(),
                "meta": meta, "args": vars(args),
                "tgt_mean":  tgt_scaler.mean_.tolist(),
                "tgt_scale": tgt_scaler.scale_.tolist(),
                "tab_mean":  tab_scaler.mean_.tolist(),
                "tab_scale": tab_scaler.scale_.tolist(),
            }
            torch.save(ckpt, os.path.join(MODELS_DIR, ckpt_name))
        else:
            epochs_no_imp += 1

        elapsed = time.time() - t0
        star    = "★" if is_best else " "
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
              f"{vl_mae:>8.4f}  {lr_now:>8.2e}  {elapsed:>5.1f}s  {star}")

        if epoch % 5 == 0 or epoch == args.epochs:
            _loss_bar(history["train_loss"], history["val_loss"])
            print()

        sys.stdout.flush()

        if args.early_stop > 0 and epochs_no_imp >= args.early_stop:
            print(f"\nEarly stopping — no val improvement for {args.early_stop} epochs.")
            break

    # ── Save last checkpoint + history ──────────────────────────────────────
    last_ckpt = {
        "epoch": epoch, "state": model.state_dict(),
        "meta": meta, "args": vars(args),
        "tgt_mean":  tgt_scaler.mean_.tolist(),
        "tgt_scale": tgt_scaler.scale_.tolist(),
        "tab_mean":  tab_scaler.mean_.tolist(),
        "tab_scale": tab_scaler.scale_.tolist(),
    }
    torch.save(last_ckpt, os.path.join(MODELS_DIR, f"last_ufno_{task}{basin_tag}.pt"))

    hist_path = os.path.join(MODELS_DIR, hist_name)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ──────────────────────────────────────────────────────
    best_ckpt_path = os.path.join(MODELS_DIR, ckpt_name)
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state"])
        print(f"\nLoaded best checkpoint (epoch {ckpt['epoch']}) for test eval")
    else:
        print("\nNo best checkpoint found — evaluating final epoch weights")

    test_loss, test_mae = run_epoch(
        model, test_loader, mode, DEVICE,
        criterion=criterion, lf_model=lf_model, show_bar=False)

    scale = np.array(tgt_scaler.scale_)
    print(f"\n{'═'*50}")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test MAE      : {test_mae:.4f}", end="")
    if task == "decay":
        print(f"  (wind_24h ~{test_mae*scale[0]:.2f}  wind_48h ~{test_mae*scale[1]:.2f}  norm units)")
    elif task == "landfall":
        print(f"  (~{test_mae*scale[0]:.1f} h)")
    print(f"{'═'*50}")
    print(f"\nCheckpoint : models/{ckpt_name}")
    print(f"History    : {hist_path}")


if __name__ == "__main__":
    main()