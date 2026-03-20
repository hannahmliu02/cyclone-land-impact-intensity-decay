"""
Training script for CycloneDecayModel.

Usage:
    python scripts/train.py [--epochs 50] [--batch 32] [--no-3d] [--no-env]

Reads preprocessed tensors from data/processed/ (run preprocess.py first).
Saves best model checkpoint to models/best_model.pt.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import CycloneDecayModel

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class TCNDDataset(Dataset):
    def __init__(self, samples: pd.DataFrame, use_3d: bool, use_env: bool):
        self.samples = samples.reset_index(drop=True)
        self.use_3d  = use_3d
        self.use_env = use_env

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        sid = row["sample_id"]

        # Data_1d  — always present
        x1d = torch.tensor(
            np.load(os.path.join(PROC_DIR, "1d", f"{sid}.npy")),
            dtype=torch.float32
        )

        # Data_3d  — optional
        x3d = None
        if self.use_3d and row.get("has_3d", False):
            p = os.path.join(PROC_DIR, "3d", f"{sid}.npy")
            if os.path.exists(p):
                x3d = torch.tensor(np.load(p), dtype=torch.float32)

        # Env-Data — optional
        xenv = None
        if self.use_env and row.get("has_env", False):
            p = os.path.join(PROC_DIR, "env", f"{sid}.npy")
            if os.path.exists(p):
                xenv = torch.tensor(np.load(p), dtype=torch.float32)

        # Targets
        y = torch.tensor([row["label_wind_24h"], row["label_wind_48h"]],
                         dtype=torch.float32)
        return x1d, x3d, xenv, y


def collate_fn(batch):
    """Handle None patches/env tensors by zero-padding absent modalities."""
    x1d_list, x3d_list, xenv_list, y_list = zip(*batch)

    x1d = torch.stack(x1d_list)
    y   = torch.stack(y_list)

    x3d = None
    if any(t is not None for t in x3d_list):
        ref = next(t for t in x3d_list if t is not None)
        x3d = torch.stack([t if t is not None else torch.zeros_like(ref)
                           for t in x3d_list])

    xenv = None
    if any(t is not None for t in xenv_list):
        ref = next(t for t in xenv_list if t is not None)
        xenv = torch.stack([t if t is not None else torch.zeros_like(ref)
                            for t in xenv_list])

    return x1d, x3d, xenv, y


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args):
    samples = pd.read_csv(os.path.join(PROC_DIR, "samples.csv"))
    if samples.empty:
        raise RuntimeError("No samples found. Run preprocess.py first.")

    # Infer feature dimensions from first sample
    sample_id = samples.iloc[0]["sample_id"]
    arr_1d  = np.load(os.path.join(PROC_DIR, "1d", f"{sample_id}.npy"))
    tab_features = arr_1d.shape[1]  # F1

    cnn_channels, env_features = 5, 32   # defaults; update after inspecting data
    if args.use_3d:
        p3d = os.path.join(PROC_DIR, "3d", f"{sample_id}.npy")
        if os.path.exists(p3d):
            cnn_channels = np.load(p3d).shape[1]
    if args.use_env:
        penv = os.path.join(PROC_DIR, "env", f"{sample_id}.npy")
        if os.path.exists(penv):
            env_features = np.load(penv).shape[0]

    dataset = TCNDDataset(samples, use_3d=args.use_3d, use_env=args.use_env)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_test  = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, _ = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = CycloneDecayModel(
        tabular_features=tab_features,
        cnn_channels=cnn_channels,
        env_features=env_features,
        use_3d=args.use_3d,
        use_env=args.use_env,
    ).to(DEVICE)

    print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {n_train}  Val: {n_val}  Device: {DEVICE}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for x1d, x3d, xenv, y in train_loader:
            x1d  = x1d.to(DEVICE)
            x3d  = x3d.to(DEVICE)  if x3d  is not None else None
            xenv = xenv.to(DEVICE) if xenv is not None else None
            y    = y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x1d, x3d, xenv)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1d, x3d, xenv, y in val_loader:
                x1d  = x1d.to(DEVICE)
                x3d  = x3d.to(DEVICE)  if x3d  is not None else None
                xenv = xenv.to(DEVICE) if xenv is not None else None
                y    = y.to(DEVICE)
                pred = model(x1d, x3d, xenv)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(MODELS_DIR, "best_model.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(MODELS_DIR, 'best_model.pt')}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=50)
    parser.add_argument("--batch",  type=int,   default=32)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--no-3d",  dest="use_3d",  action="store_false")
    parser.add_argument("--no-env", dest="use_env", action="store_false")
    parser.set_defaults(use_3d=True, use_env=True)
    args = parser.parse_args()
    train(args)
