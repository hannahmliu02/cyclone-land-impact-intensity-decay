"""
Train CycloneUFNO for post-landfall intensity decay prediction.

Two data modes (auto-detected):
  spatial   — processed tensors from preprocess.py exist in data/processed/
               Uses Data_3d patches + tabular features + env features
  tabular   — only feature_matrix_decay.csv from features.py is available
               Uses tabular features only (no 3-D patches)

Usage
─────
  python scripts/train_ufno.py [options]

Options
  --epochs   int    Training epochs                    (default 100)
  --batch    int    Batch size                         (default 8)
  --lr       float  Initial learning rate              (default 1e-3)
  --modes    int    Fourier modes (modes1 = modes2)    (default 12)
  --width    int    UFNO hidden width                  (default 32)
  --no-tab         Disable tabular FiLM conditioning
  --seed     int    Random seed                        (default 42)

Outputs (models/)
  best_ufno.pt          — best validation checkpoint
  last_ufno.pt          — final epoch checkpoint
  ufno_history.json     — per-epoch loss / metric log
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


TAB_COLS = _load_tab_cols()
TARGET_COLS = ["wind_24h", "wind_48h"]


# ── Datasets ──────────────────────────────────────────────────────────────────
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
def load_data(seed: int):
    """Return (train_loader, val_loader, test_loader, meta, mode, tab_dim)."""
    samples_csv = os.path.join(PROC_DIR, "samples.csv")
    decay_csv   = os.path.join(FEAT_DIR,  "feature_matrix_decay.csv")

    # Prefer processed tensors; fall back to feature matrix
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
            "No data found. Run features.py (tabular mode) or "
            "preprocess.py (spatial mode) first.")

    print(f"Data mode  : {mode}")
    print(f"Samples    : {len(df)}  |  basins: {df['basin'].value_counts().to_dict()}")

    tab_scaler = StandardScaler()
    tgt_scaler = StandardScaler()

    # Use TCND original splits if available, else fall back to random 70/15/15
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

    if mode == "tabular":
        DS = TabularDecayDataset
        collate = None
    else:
        DS = SpatialDecayDataset
        collate = _collate_spatial

    train_ds = DS(train_df, tab_scaler, tgt_scaler, fit=True)
    val_ds   = DS(val_df,   tab_scaler, tgt_scaler, fit=False)
    test_ds  = DS(test_df,  tab_scaler, tgt_scaler, fit=False)

    # Basin-weighted sampler: inverse-frequency weighting so EP is oversampled
    basin_counts = train_df["basin"].value_counts().to_dict()
    sample_weights = train_df["basin"].map(
        lambda b: 1.0 / basin_counts.get(b, 1)).values
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    kw = dict(batch_size=8, num_workers=0, collate_fn=collate)
    train_loader = DataLoader(train_ds, sampler=sampler, **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False,   **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False,   **kw)

    tab_cols = [c for c in TAB_COLS if c in df.columns]
    meta = {
        "mode":        mode,
        "tab_dim":     len(tab_cols),
        "tab_cols":    tab_cols,
        "tgt_mean":    tgt_scaler.mean_.tolist(),
        "tgt_scale":   tgt_scaler.scale_.tolist(),
        "n_train":     len(train_ds),
        "n_val":       len(val_ds),
        "n_test":      len(test_ds),
    }
    return train_loader, val_loader, test_loader, meta, tgt_scaler


# ── Training / evaluation loops ───────────────────────────────────────────────
def _forward(model, batch, mode: str, device):
    if mode == "tabular":
        x_tab, y = batch
        x_tab = x_tab.to(device)
        y     = y.to(device)
        pred  = model(x_tab=x_tab)
    else:
        x3d, x_tab, y = batch
        x3d   = x3d.to(device)   if x3d is not None else None
        x_tab = x_tab.to(device)
        y     = y.to(device)
        pred  = model(x_3d=x3d, x_tab=x_tab)
    return pred, y


def run_epoch(model, loader, mode, device, optimizer=None, criterion=None,
              desc="", show_bar=True):
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
            pred, y = _forward(model, batch, mode, device)
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
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--batch",  type=int,   default=8)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--modes",  type=int,   default=12)
    parser.add_argument("--width",  type=int,   default=32)
    parser.add_argument("--no-tab", action="store_true",
                        help="Disable tabular FiLM conditioning")
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, test_loader, meta, tgt_scaler = \
        load_data(args.seed)

    mode  = meta["mode"]

    model = CycloneUFNO(
        sp_channels  = 5,
        T            = 8,
        tab_features = meta["tab_dim"],
        modes1       = args.modes,
        modes2       = args.modes,
        width        = args.width,
        unet_dropout = 0.2,
    ).to(DEVICE)

    print(f"\nModel      : CycloneUFNO  ({model.count_params():,} params)")
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {args.epochs}")

    criterion = LpLoss(p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)

    history = {"train_loss": [], "val_loss": [],
               "train_mae": [],  "val_mae": []}
    best_val = float("inf")

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
            desc=f"Ep {epoch:>3} train", show_bar=True)
        vl_loss, vl_mae = run_epoch(
            model, val_loader, mode, DEVICE,
            desc=f"Ep {epoch:>3} val  ", show_bar=True)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(vl_mae)

        is_best = vl_loss < best_val
        if is_best:
            best_val = vl_loss
            torch.save({
                "epoch":      epoch,
                "state":      model.state_dict(),
                "meta":       meta,
                "args":       vars(args),
                "tgt_mean":   tgt_scaler.mean_.tolist(),
                "tgt_scale":  tgt_scaler.scale_.tolist(),
            }, os.path.join(MODELS_DIR, "best_ufno.pt"))

        elapsed = time.time() - t0
        star    = "★" if is_best else " "
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
              f"{vl_mae:>8.4f}  {lr_now:>8.2e}  {elapsed:>5.1f}s  {star}")

        # Print ASCII loss sparkline every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            _loss_bar(history["train_loss"], history["val_loss"])
            print()

        sys.stdout.flush()

    # Save last checkpoint and history
    torch.save({
        "epoch":     args.epochs,
        "state":     model.state_dict(),
        "meta":      meta,
        "args":      vars(args),
        "tgt_mean":  tgt_scaler.mean_.tolist(),
        "tgt_scale": tgt_scaler.scale_.tolist(),
    }, os.path.join(MODELS_DIR, "last_ufno.pt"))

    hist_path = os.path.join(MODELS_DIR, "ufno_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ───────────────────────────────────────────────
    ckpt = torch.load(os.path.join(MODELS_DIR, "best_ufno.pt"),
                      map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    test_loss, test_mae = run_epoch(model, test_loader, mode, DEVICE)
    print(f"\nTest loss  : {test_loss:.4f}")
    print(f"Test MAE   : {test_mae:.4f}  (normalised)")

    # Denormalise MAE back to knots
    scale = np.array(tgt_scaler.scale_)
    print(f"Test MAE   : wind_24h ≈ {test_mae * scale[0]:.2f} kt  "
          f"| wind_48h ≈ {test_mae * scale[1]:.2f} kt  (approx.)")
    print(f"\nCheckpoints saved in  {MODELS_DIR}/")
    print(f"History saved to      {hist_path}")


if __name__ == "__main__":
    main()
