"""
Train CycloneUFNO — landfall impact or intensity decay.

Two tasks
─────────
  landfall  — binary classification: will this storm make landfall?
               Uses feature_matrix_landfall.csv, BCEWithLogitsLoss
               Saves: best_ufno_landfall.pt

  decay     — regression: wind speed 24h and 48h after reference time
               Uses feature_matrix_decay.csv, LpLoss
               Saves: best_ufno_decay.pt
               Optionally accepts --landfall-ckpt to inject a learned
               landfall embedding into the model via FiLM at blocks 3–5

Two data modes (auto-detected for decay task):
  spatial   — processed tensors from preprocess.py exist in data/processed/
  tabular   — only feature_matrix_decay.csv is available (default fallback)

Usage
─────
  # Step 1 — train landfall model
  python scripts/train_ufno.py --task landfall --epochs 100

  # Step 2 — train decay model with landfall embedding
  python scripts/train_ufno.py --task decay --epochs 150 \\
      --landfall-ckpt models/best_ufno_landfall.pt

Options
  --task          {landfall,decay}  Which task to train  (default decay)
  --landfall-ckpt PATH              Landfall checkpoint for embedding injection
  --epochs        int               Training epochs      (default 100)
  --batch         int               Batch size           (default 8)
  --lr            float             Initial LR           (default 1e-3)
  --modes         int               Fourier modes        (default 12)
  --width         int               UFNO hidden width    (default 32)
  --unet-dropout  float             UNet dropout         (default 0.2)
  --no-tab                          Disable tabular FiLM conditioning
  --seed          int               Random seed          (default 42)

Outputs (models/)
  best_ufno_landfall.pt   — best landfall checkpoint
  best_ufno_decay.pt      — best decay checkpoint
  ufno_history.json       — per-epoch loss / metric log
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

# Allow importing from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from ufno import CycloneUFNO, LpLoss

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEAT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

# ── Feature columns — loaded dynamically from ablation output ─────────────────
def _load_tab_cols() -> list:
    """
    Load tabular feature columns from ablation-selected groups.

    Priority:
      1. data/features/selected_feature_groups.json  (written by ablation.py)
      2. Hardcoded fallback (wind + pressure + wp_couple)
    """
    feat_dir = os.path.join(os.path.dirname(__file__), "..", "data", "features")
    sel_path = os.path.join(feat_dir, "selected_feature_groups.json")
    grp_path = os.path.join(feat_dir, "feature_groups.json")

    fallback = [
        "wind_last", "wind_max", "wind_mean", "wind_std",
        "wind_delta_6h", "wind_delta_12h", "wind_delta_24h", "wind_trend",
        "pres_last", "pres_min", "pres_mean", "pres_std",
        "pres_delta_6h", "pres_delta_12h", "pres_delta_24h", "pres_trend",
        "wp_residual",
    ]

    if not os.path.exists(sel_path) or not os.path.exists(grp_path):
        print("[train_ufno] selected_feature_groups.json not found — using fallback TAB_COLS.")
        return fallback

    with open(sel_path) as f:
        sel = json.load(f)
    with open(grp_path) as f:
        groups = json.load(f)

    selected_groups = sel.get("selected_groups", [])
    cols = []
    for g in selected_groups:
        cols.extend(groups.get(g, []))
    cols = list(dict.fromkeys(cols))   # dedup, preserve order

    if not cols:
        print("[train_ufno] selected_feature_groups.json produced no columns — using fallback.")
        return fallback

    print(f"[train_ufno] TAB_COLS loaded from ablation: {len(cols)} features "
          f"from groups {selected_groups}")
    return cols


TAB_COLS    = _load_tab_cols()
TARGET_COLS = ["wind_24h", "wind_48h"]
LANDFALL_TARGET = "made_landfall"


# ── Datasets ──────────────────────────────────────────────────────────────────
class LandfallDataset(Dataset):
    """Tabular dataset for landfall binary classification."""

    def __init__(self, df: pd.DataFrame,
                 tab_scaler: StandardScaler,
                 fit: bool = False):
        tab_cols = [c for c in TAB_COLS if c in df.columns]
        X = df[tab_cols].fillna(0).values.astype(np.float32)
        y = df[LANDFALL_TARGET].values.astype(np.float32)
        self.X = torch.tensor(tab_scaler.fit_transform(X) if fit
                              else tab_scaler.transform(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,1)
        self.meta = df[["storm_id", "basin"]].reset_index(drop=True)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TabularDecayDataset(Dataset):
    """Tabular-only dataset — no spatial patches required."""

    def __init__(self, df: pd.DataFrame,
                 tab_scaler: StandardScaler,
                 tgt_scaler: StandardScaler,
                 fit: bool = False):
        tab_cols = [c for c in TAB_COLS if c in df.columns]
        X = df[tab_cols].values.astype(np.float32)
        y = df[TARGET_COLS].values.astype(np.float32)

        if fit:
            X = tab_scaler.fit_transform(X)
            y = tgt_scaler.fit_transform(y)
        else:
            X = tab_scaler.transform(X)
            y = tgt_scaler.transform(y)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # Store metadata for post-hoc analysis
        self.meta = df[["storm_id", "basin"]].reset_index(drop=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SpatialDecayDataset(Dataset):
    """Full multimodal dataset — requires data/processed/ from preprocess.py."""

    def __init__(self, df: pd.DataFrame,
                 tab_scaler: StandardScaler,
                 tgt_scaler: StandardScaler,
                 fit: bool = False):
        tab_cols = [c for c in TAB_COLS if c in df.columns]
        X_tab = df[tab_cols].values.astype(np.float32)
        y     = df[TARGET_COLS].values.astype(np.float32)

        if fit:
            X_tab = tab_scaler.fit_transform(X_tab)
            y     = tgt_scaler.fit_transform(y)
        else:
            X_tab = tab_scaler.transform(X_tab)
            y     = tgt_scaler.transform(y)

        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)
        self.ids   = df["storm_id"].tolist()
        self.meta  = df[["storm_id", "basin"]].reset_index(drop=True)

    def __len__(self):
        return len(self.X_tab)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        npy = os.path.join(PROC_DIR, "3d", f"{sid}.npy")
        if os.path.exists(npy):
            x3d = torch.tensor(np.load(npy), dtype=torch.float32)  # (T,C,H,W)
        else:
            x3d = None
        return x3d, self.X_tab[idx], self.y[idx]


def _collate_spatial(batch):
    """Custom collate: keeps x3d as None if any sample lacks a patch."""
    x3ds, x_tabs, ys = zip(*batch)
    if any(x is None for x in x3ds):
        x3d_out = None
    else:
        x3d_out = torch.stack(x3ds)
    return x3d_out, torch.stack(x_tabs), torch.stack(ys)


# ── Data loading ──────────────────────────────────────────────────────────────
def _split_df(df, seed):
    """Apply TCND original splits if available, else random 70/15/15."""
    if "split" in df.columns and df["split"].isin(["train","val","test"]).any():
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df   = df[df["split"] == "val"].reset_index(drop=True)
        test_df  = df[df["split"] == "test"].reset_index(drop=True)
        print(f"Split      : TCND original  "
              f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
    else:
        train_df, tmp_df = train_test_split(df, test_size=0.3, random_state=seed)
        val_df,  test_df = train_test_split(tmp_df, test_size=0.5, random_state=seed)
        print(f"Split      : random 70/15/15  "
              f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
    return train_df, val_df, test_df


def load_data(seed: int, task: str = "decay", batch: int = 8):
    """Return (train_loader, val_loader, test_loader, meta, tgt_scaler)."""

    if task == "landfall":
        csv = os.path.join(FEAT_DIR, "feature_matrix_landfall.csv")
        if not os.path.exists(csv):
            raise FileNotFoundError("Run features.py first — feature_matrix_landfall.csv not found.")
        df = pd.read_csv(csv, keep_default_na=False)
        df = df.dropna(subset=[LANDFALL_TARGET]).reset_index(drop=True)
        print(f"Task       : landfall (binary classification)")
        print(f"Samples    : {len(df)}  |  basins: {df['basin'].value_counts().to_dict()}")
        print(f"Landfall % : {df[LANDFALL_TARGET].mean()*100:.1f}%")

        tab_scaler = StandardScaler()
        train_df, val_df, test_df = _split_df(df, seed)

        train_ds = LandfallDataset(train_df, tab_scaler, fit=True)
        val_ds   = LandfallDataset(val_df,   tab_scaler, fit=False)
        test_ds  = LandfallDataset(test_df,  tab_scaler, fit=False)

        # Class-balanced sampler for imbalanced landfall labels
        pos_weight = (train_df[LANDFALL_TARGET] == 0).sum() / \
                     max((train_df[LANDFALL_TARGET] == 1).sum(), 1)
        weights = train_df[LANDFALL_TARGET].map(
            lambda y: float(pos_weight) if y == 1 else 1.0).values
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        tab_cols = [c for c in TAB_COLS if c in df.columns]
        meta = {
            "task":    "landfall",
            "mode":    "tabular",
            "tab_dim": len(tab_cols),
            "tab_cols": tab_cols,
            "n_train": len(train_ds),
            "n_val":   len(val_ds),
            "n_test":  len(test_ds),
        }
        kw = dict(batch_size=batch, num_workers=0)
        return (DataLoader(train_ds, sampler=sampler, **kw),
                DataLoader(val_ds,   shuffle=False,   **kw),
                DataLoader(test_ds,  shuffle=False,   **kw),
                meta, None)

    # ── decay task ────────────────────────────────────────────────────────────
    samples_csv = os.path.join(PROC_DIR, "samples.csv")
    decay_csv   = os.path.join(FEAT_DIR,  "feature_matrix_decay.csv")

    if os.path.exists(samples_csv):
        mode = "spatial"
        df   = pd.read_csv(samples_csv, keep_default_na=False)
        df   = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    elif os.path.exists(decay_csv):
        mode = "tabular"
        df   = pd.read_csv(decay_csv, keep_default_na=False)
        df   = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    else:
        raise FileNotFoundError(
            "No data found. Run features.py (tabular) or preprocess.py (spatial).")

    print(f"Task       : decay (regression)")
    print(f"Data mode  : {mode}")
    print(f"Samples    : {len(df)}  |  basins: {df['basin'].value_counts().to_dict()}")

    tab_scaler = StandardScaler()
    tgt_scaler = StandardScaler()
    train_df, val_df, test_df = _split_df(df, seed)

    if mode == "tabular":
        DS, collate = TabularDecayDataset, None
    else:
        DS, collate = SpatialDecayDataset, _collate_spatial

    train_ds = DS(train_df, tab_scaler, tgt_scaler, fit=True)
    val_ds   = DS(val_df,   tab_scaler, tgt_scaler, fit=False)
    test_ds  = DS(test_df,  tab_scaler, tgt_scaler, fit=False)

    # Basin-weighted sampler: oversample EP
    basin_counts   = train_df["basin"].value_counts().to_dict()
    sample_weights = train_df["basin"].map(
        lambda b: 1.0 / basin_counts.get(b, 1)).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)

    kw = dict(batch_size=batch, num_workers=0, collate_fn=collate)
    tab_cols = [c for c in TAB_COLS if c in df.columns]
    meta = {
        "task":      "decay",
        "mode":      mode,
        "tab_dim":   len(tab_cols),
        "tab_cols":  tab_cols,
        "tgt_mean":  tgt_scaler.mean_.tolist(),
        "tgt_scale": tgt_scaler.scale_.tolist(),
        "n_train":   len(train_ds),
        "n_val":     len(val_ds),
        "n_test":    len(test_ds),
    }
    return (DataLoader(train_ds, sampler=sampler, **kw),
            DataLoader(val_ds,   shuffle=False,   **kw),
            DataLoader(test_ds,  shuffle=False,   **kw),
            meta, tgt_scaler)


# ── Training / evaluation loops ───────────────────────────────────────────────
def _forward(model, batch, mode: str, device, lf_model=None):
    """
    Run one forward pass.  lf_model: frozen landfall model for embedding injection.
    """
    if mode == "tabular":
        x_tab, y = batch
        x_tab = x_tab.to(device)
        y     = y.to(device)
        lf_embed = None
        if lf_model is not None:
            with torch.no_grad():
                lf_embed = lf_model.extract_embedding(x_tab=x_tab)
        pred = model(x_tab=x_tab, lf_embed=lf_embed)
    else:
        x3d, x_tab, y = batch
        x3d   = x3d.to(device)   if x3d is not None else None
        x_tab = x_tab.to(device)
        y     = y.to(device)
        lf_embed = None
        if lf_model is not None:
            with torch.no_grad():
                lf_embed = lf_model.extract_embedding(x_tab=x_tab, x_3d=x3d)
        pred = model(x_3d=x3d, x_tab=x_tab, lf_embed=lf_embed)
    return pred, y


def run_epoch(model, loader, mode, device, optimizer=None, criterion=None,
              desc="", show_bar=True, lf_model=None):
    from tqdm import tqdm
    if criterion is None:
        criterion = LpLoss(p=2)
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_mae  = 0.0
    n = 0

    bar = tqdm(loader, desc=desc, leave=False, ncols=72,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining} loss={postfix}]") \
          if show_bar else loader

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in bar:
            pred, y = _forward(model, batch, mode, device, lf_model=lf_model)
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


# ── ASCII sparkline of loss history ───────────────────────────────────────────
_SPARKS = " ▁▂▃▄▅▆▇█"

def _sparkline(values, width=30):
    if not values:
        return ""
    tail   = values[-width:]
    lo, hi = min(tail), max(tail)
    rng    = hi - lo or 1e-9
    return "".join(_SPARKS[min(8, int((v - lo) / rng * 8))] for v in tail)


def _loss_bar(train_vals, val_vals, width=38):
    """Print a two-row ASCII chart of recent train/val loss."""
    if len(train_vals) < 2:
        return
    print(f"\n  Train ▏{_sparkline(train_vals, width)}▏  "
          f"lo={min(train_vals):.4f}")
    print(f"  Val   ▏{_sparkline(val_vals,   width)}▏  "
          f"lo={min(val_vals):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="decay",
                        choices=["landfall", "decay"],
                        help="Which task to train (default: decay)")
    parser.add_argument("--landfall-ckpt", default=None,
                        help="Path to landfall checkpoint for embedding injection "
                             "(decay task only)")
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--batch",  type=int,   default=8)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--modes",  type=int,   default=12)
    parser.add_argument("--width",  type=int,   default=32)
    parser.add_argument("--unet-dropout", type=float, default=0.2,
                        help="Dropout rate inside UNet2d branches")
    parser.add_argument("--no-tab", action="store_true",
                        help="Disable tabular FiLM conditioning")
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, test_loader, meta, tgt_scaler = \
        load_data(args.seed, task=args.task, batch=args.batch)

    mode = meta["mode"]
    task = args.task

    # ── Load frozen landfall model for embedding injection ─────────────────
    lf_model     = None
    lf_embed_dim = 0
    if task == "decay" and args.landfall_ckpt:
        lf_ckpt_path = args.landfall_ckpt
        if not os.path.exists(lf_ckpt_path):
            raise FileNotFoundError(
                f"Landfall checkpoint not found: {lf_ckpt_path}\n"
                f"Train landfall model first: "
                f"python scripts/train_ufno.py --task landfall")
        lf_ckpt = torch.load(lf_ckpt_path, map_location=DEVICE)
        lf_args = lf_ckpt.get("args", {})
        lf_meta = lf_ckpt.get("meta", {})
        lf_model = CycloneUFNO(
            sp_channels  = 5,
            T            = 8,
            tab_features = lf_meta.get("tab_dim", meta["tab_dim"]),
            modes1       = lf_args.get("modes", 12),
            modes2       = lf_args.get("modes", 12),
            width        = lf_args.get("width", args.width),
            unet_dropout = lf_args.get("unet_dropout", 0.0),
            n_outputs    = 1,
        ).to(DEVICE)
        lf_model.load_state_dict(lf_ckpt["state"])
        lf_model.eval()
        for p in lf_model.parameters():
            p.requires_grad = False
        lf_embed_dim = lf_args.get("width", args.width)
        print(f"Landfall model loaded from {lf_ckpt_path}")
        print(f"Landfall embedding dim: {lf_embed_dim} (injected at blocks 3–5)")

    # ── Build model ────────────────────────────────────────────────────────
    n_outputs = 1 if task == "landfall" else 2
    model = CycloneUFNO(
        sp_channels  = 5,
        T            = 8,
        tab_features = meta["tab_dim"],
        modes1       = args.modes,
        modes2       = args.modes,
        width        = args.width,
        unet_dropout = args.unet_dropout,
        n_outputs    = n_outputs,
        lf_embed_dim = lf_embed_dim,
    ).to(DEVICE)

    print(f"\nTask       : {task}")
    print(f"Model      : CycloneUFNO  ({model.count_params():,} params)")
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {args.epochs}")

    # ── Loss ───────────────────────────────────────────────────────────────
    if task == "landfall":
        # Class imbalance: weight positive class
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = LpLoss(p=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)

    history  = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val = float("inf")
    ckpt_name = f"best_ufno_{task}.pt"

    header = (f"{'Epoch':>6}  {'Train':>8}  {'Val':>8}  "
              f"{'ValMAE':>8}  {'LR':>8}  {'Time':>6}  {'':>4}")
    print(f"\n{header}")
    print("─" * len(header))
    sys.stdout.flush()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        tr_loss, tr_mae = run_epoch(
            model, train_loader, mode, DEVICE, optimizer, criterion,
            desc=f"Ep {epoch:>3} train", show_bar=True, lf_model=lf_model)
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
            best_val = vl_loss
            ckpt = {
                "epoch":      epoch,
                "state":      model.state_dict(),
                "meta":       meta,
                "args":       vars(args),
            }
            if tgt_scaler is not None:
                ckpt["tgt_mean"]  = tgt_scaler.mean_.tolist()
                ckpt["tgt_scale"] = tgt_scaler.scale_.tolist()
            torch.save(ckpt, os.path.join(MODELS_DIR, ckpt_name))

        elapsed = time.time() - t0
        star    = "★" if is_best else " "
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
              f"{vl_mae:>8.4f}  {lr_now:>8.2e}  {elapsed:>5.1f}s  {star}")

        if epoch % 5 == 0 or epoch == args.epochs:
            _loss_bar(history["train_loss"], history["val_loss"])
            print()

        sys.stdout.flush()

    # ── Save last checkpoint + history ────────────────────────────────────
    last_ckpt = {
        "epoch": args.epochs, "state": model.state_dict(),
        "meta": meta, "args": vars(args),
    }
    if tgt_scaler is not None:
        last_ckpt["tgt_mean"]  = tgt_scaler.mean_.tolist()
        last_ckpt["tgt_scale"] = tgt_scaler.scale_.tolist()
    torch.save(last_ckpt, os.path.join(MODELS_DIR, f"last_ufno_{task}.pt"))

    hist_path = os.path.join(MODELS_DIR, "ufno_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ───────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(MODELS_DIR, ckpt_name), map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    test_loss, test_mae = run_epoch(model, test_loader, mode, DEVICE,
                                    criterion=criterion, lf_model=lf_model)
    print(f"\nTest loss  : {test_loss:.4f}")
    print(f"Test MAE   : {test_mae:.4f}")
    if task == "decay" and tgt_scaler is not None:
        scale = np.array(tgt_scaler.scale_)
        print(f"Test MAE   : wind_24h ≈ {test_mae * scale[0]:.2f}  "
              f"| wind_48h ≈ {test_mae * scale[1]:.2f}  (normalised units)")
    print(f"\nCheckpoint : {MODELS_DIR}/{ckpt_name}")
    print(f"History    : {hist_path}")


if __name__ == "__main__":
    main()
