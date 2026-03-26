"""
Cross-basin generalization experiments.

Experiment plan
───────────────
  1. Single-basin training  — train on one basin, test on same basin
  2. Cross-basin transfer   — train on one basin, test on each other basin
  3. Gradual addition       — train on WP → WP+NA → WP+NA+EP, test on all

For each configuration, trains a fresh CycloneUFNO and records RMSE/MAE
on every basin's test split.

Outputs
───────
  data/features/cross_basin_results_{task}.csv     — full results table
  figures/cross_basin_heatmap_{task}.png           — transfer RMSE heatmap
  figures/cross_basin_gradual_{task}.png           — gradual addition curves
  figures/cross_basin_summary_{task}.png           — bar chart comparison

Usage
─────
  python scripts/cross_basin.py --task decay   [--epochs 60] [--seed 42]
  python scripts/cross_basin.py --task landfall [--epochs 60] [--seed 42]
  python scripts/cross_basin.py --task all      [--epochs 60] [--seed 42]
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(__file__))
from ufno import CycloneUFNO, LpLoss

FEAT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "features")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "..", "figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "cross_basin")
os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = (torch.device("cuda")  if torch.cuda.is_available()  else
          torch.device("mps")   if torch.backends.mps.is_available() else
          torch.device("cpu"))

ALL_BASINS = ["WP", "NA", "EP"]

TASK_CONFIG = {
    "decay": {
        "csv":         "feature_matrix_decay.csv",
        "target_cols": ["wind_24h", "wind_48h"],
        "sel_json":    "selected_feature_groups_decay.json",
        "metric_cols": ["rmse_24h", "rmse_48h"],
        "metric_labels": ["RMSE 24h (normalised wind)", "RMSE 48h (normalised wind)"],
        "heatmap_col": "rmse_24h",
        "heatmap_label": "RMSE (normalised wind, 24h)",
        "summary_col": "rmse_24h",
        "summary_label": "RMSE (normalised wind, 24h)",
    },
    "landfall": {
        "csv":         "feature_matrix_landfall.csv",
        "target_cols": ["hours_to_landfall"],
        "sel_json":    "selected_feature_groups_landfall.json",
        "metric_cols": ["rmse_hours"],
        "metric_labels": ["RMSE (hours to landfall)"],
        "heatmap_col": "rmse_hours",
        "heatmap_label": "RMSE (hours to landfall)",
        "summary_col": "rmse_hours",
        "summary_label": "RMSE (hours to landfall)",
    },
}

# ── Per-task globals (set in run_task) ────────────────────────────────────────
TARGET_COLS = None
TAB_COLS    = None
CFG         = None


def _load_tab_cols(task):
    candidates = [
        os.path.join(FEAT_DIR, TASK_CONFIG[task]["sel_json"]),
        os.path.join(FEAT_DIR, "selected_feature_groups.json"),
    ]
    grp_path = os.path.join(FEAT_DIR, "feature_groups.json")
    fallback = [
        "wind_last","wind_max","wind_mean","wind_std",
        "wind_delta_6h","wind_delta_12h","wind_delta_24h","wind_trend",
        "pres_last","pres_min","pres_mean","pres_std",
        "pres_delta_6h","pres_delta_12h","pres_delta_24h","pres_trend",
        "wp_residual",
    ]
    sel_path = next((p for p in candidates if os.path.exists(p)), None)
    if sel_path is None or not os.path.exists(grp_path):
        return fallback
    with open(sel_path) as f:
        sel = json.load(f)
    with open(grp_path) as f:
        groups = json.load(f)
    cols = []
    for g in sel.get("selected_groups", []):
        cols.extend(groups.get(g, []))
    return list(dict.fromkeys(cols)) or fallback


# ── Dataset ────────────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, df, tab_scaler, tgt_scaler, fit=False):
        tab_cols = [c for c in TAB_COLS if c in df.columns]
        X_df = df[tab_cols].apply(pd.to_numeric, errors="coerce")
        X = X_df.fillna(X_df.median()).fillna(0.0).values.astype(np.float32)
        y = df[TARGET_COLS].values.astype(np.float32)
        self.X = torch.tensor(tab_scaler.fit_transform(X) if fit
                              else tab_scaler.transform(X), dtype=torch.float32)
        self.y = torch.tensor(tgt_scaler.fit_transform(y) if fit
                              else tgt_scaler.transform(y), dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Build loaders for a given train/val/test basin split ──────────────────────
def make_loaders(df, train_basins, test_basins, seed, batch=8):
    global TAB_COLS
    train_df = df[df["basin"].isin(train_basins) & (df["split"] == "train")
                  ].dropna(subset=TARGET_COLS).reset_index(drop=True)
    val_df   = df[df["basin"].isin(train_basins) & (df["split"] == "val")
                  ].dropna(subset=TARGET_COLS).reset_index(drop=True)

    test_dfs = {
        b: df[(df["basin"] == b) & (df["split"] == "test")]
               .dropna(subset=TARGET_COLS).reset_index(drop=True)
        for b in test_basins
    }

    if len(train_df) < 10:
        raise ValueError(f"Too few training samples ({len(train_df)}) "
                         f"for basins {train_basins}")

    # Drop zero-variance columns computed on training data — they carry no
    # information and cause unscaled large values on out-of-distribution test data.
    tab_cols_active = [c for c in TAB_COLS if c in train_df.columns]
    X_check = train_df[tab_cols_active].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    nonzero_var = X_check.columns[X_check.var() > 1e-8].tolist()
    _orig_tab_cols = TAB_COLS
    TAB_COLS = nonzero_var

    tab_scaler = StandardScaler()
    tgt_scaler = StandardScaler()

    train_ds = TabularDataset(train_df, tab_scaler, tgt_scaler, fit=True)
    val_ds   = TabularDataset(val_df,   tab_scaler, tgt_scaler, fit=False)

    counts  = train_df["basin"].value_counts().to_dict()
    weights = train_df["basin"].map(lambda b: 1.0 / counts.get(b, 1)).values
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False)
    test_loaders = {
        b: DataLoader(TabularDataset(tdf, tab_scaler, tgt_scaler, fit=False),
                      batch_size=batch, shuffle=False)
        for b, tdf in test_dfs.items() if len(tdf) > 0
    }

    tab_cols = [c for c in TAB_COLS if c in train_df.columns]
    meta = {"tab_dim": len(tab_cols), "tgt_scaler": tgt_scaler}
    TAB_COLS = _orig_tab_cols  # restore for next call
    return train_loader, val_loader, test_loaders, meta


# ── Train one model ────────────────────────────────────────────────────────────
def train_model(train_loader, val_loader, tab_dim, epochs, seed):
    torch.manual_seed(seed)
    model = CycloneUFNO(
        sp_channels=4, T=8, tab_features=tab_dim,
        modes1=12, modes2=12, width=32, unet_dropout=0.2,
        n_outputs=len(TARGET_COLS),
    ).to(DEVICE)

    criterion = LpLoss(p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)

    best_val   = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x_tab=x), y)
            optimizer.zero_grad()
            loss.clamp(max=100.0).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0; n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x_tab=x), y).item() * y.size(0)
                n += y.size(0)
        val_loss /= max(n, 1)
        if not np.isfinite(val_loss):
            continue
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


# ── Evaluate on a loader ───────────────────────────────────────────────────────
def evaluate(model, loader, tgt_scaler):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x_tab=x.to(DEVICE)).cpu().numpy())
            targets.append(y.numpy())
    preds   = np.vstack(preds)
    targets = np.vstack(targets)

    preds_p   = tgt_scaler.inverse_transform(preds)
    targets_p = tgt_scaler.inverse_transform(targets)

    rmse = np.sqrt(((preds_p - targets_p) ** 2).mean(axis=0))
    mae  = np.abs(preds_p - targets_p).mean(axis=0)

    result = {"n": len(preds)}
    for i, col in enumerate(CFG["metric_cols"]):
        result[col]                      = rmse[i]
        result[col.replace("rmse", "mae")] = mae[i]
    return result


# ── Experiment runners ─────────────────────────────────────────────────────────
def run_single_and_cross(df, epochs, seed, task):
    rows = []
    for train_b in ALL_BASINS:
        print(f"\n  Training on [{train_b}]...")
        try:
            tl, vl, test_loaders, meta = make_loaders(df, [train_b], ALL_BASINS, seed)
        except ValueError as e:
            print(f"    Skipped: {e}"); continue

        model, best_val = train_model(tl, vl, meta["tab_dim"], epochs, seed)
        tgt_scaler = meta["tgt_scaler"]
        print(f"    Best val loss: {best_val:.4f}")

        for test_b, loader in test_loaders.items():
            m = evaluate(model, loader, tgt_scaler)
            rows.append({"train_basins": train_b, "test_basin": test_b, **m})
            metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n")
            print(f"    → test [{test_b}]  {metrics_str}  n={m['n']}")

        ckpt_path = os.path.join(MODELS_DIR, f"train_{train_b}_{task}.pt")
        torch.save(model.state_dict(), ckpt_path)

    return pd.DataFrame(rows)


def run_gradual(df, epochs, seed):
    configs = [["WP"], ["WP", "NA"], ["WP", "NA", "EP"]]
    rows = []
    for train_basins in configs:
        label = "+".join(train_basins)
        print(f"\n  Gradual — training on [{label}]...")
        try:
            tl, vl, test_loaders, meta = make_loaders(df, train_basins, ALL_BASINS, seed)
        except ValueError as e:
            print(f"    Skipped: {e}"); continue

        model, best_val = train_model(tl, vl, meta["tab_dim"], epochs, seed)
        tgt_scaler = meta["tgt_scaler"]
        print(f"    Best val loss: {best_val:.4f}")

        for test_b, loader in test_loaders.items():
            m = evaluate(model, loader, tgt_scaler)
            rows.append({"train_basins": label, "test_basin": test_b, **m})
            metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n")
            print(f"    → test [{test_b}]  {metrics_str}  n={m['n']}")

    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_heatmap(cross_df, task):
    single = cross_df[~cross_df["train_basins"].str.contains(r"\+")]
    col    = CFG["heatmap_col"]
    pivot  = single.pivot(index="train_basins", columns="test_basin",
                          values=col).reindex(index=ALL_BASINS, columns=ALL_BASINS)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label=CFG["heatmap_label"])
    ax.set_xticks(range(len(ALL_BASINS))); ax.set_xticklabels(ALL_BASINS)
    ax.set_yticks(range(len(ALL_BASINS))); ax.set_yticklabels(ALL_BASINS)
    ax.set_xlabel("Test basin"); ax.set_ylabel("Train basin")
    ax.set_title(f"Cross-basin Transfer — {task.capitalize()}\n"
                 "Diagonal = in-domain, off-diagonal = transfer")

    for i in range(len(ALL_BASINS)):
        for j in range(len(ALL_BASINS)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > pivot.values.max() * 0.6 else "black",
                        fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"cross_basin_heatmap_{task}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Saved: {path}")


def plot_gradual(gradual_df, task):
    configs = gradual_df["train_basins"].unique().tolist()
    metrics = CFG["metric_cols"]
    labels  = CFG["metric_labels"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    colors = {"WP": "#1f78b4", "NA": "#33a02c", "EP": "#e31a1c"}

    for ax, metric, label in zip(axes, metrics, labels):
        for test_b in ALL_BASINS:
            sub  = gradual_df[gradual_df["test_basin"] == test_b]
            vals = [sub[sub["train_basins"] == c][metric].values for c in configs]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.plot(configs, vals, marker="o",
                    label=f"Test: {test_b}", color=colors.get(test_b, "#888"))
        ax.set_title(f"Gradual Basin Addition — {label}")
        ax.set_xlabel("Training basins")
        ax.set_ylabel(label)
        ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"cross_basin_gradual_{task}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary(cross_df, gradual_df, task):
    col = CFG["summary_col"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ALL_BASINS))
    w = 0.25

    in_domain = []
    for b in ALL_BASINS:
        row = cross_df[(cross_df["train_basins"] == b) & (cross_df["test_basin"] == b)]
        in_domain.append(row[col].values[0] if len(row) else np.nan)

    best_transfer = []
    for b in ALL_BASINS:
        others = cross_df[(cross_df["train_basins"] != b) &
                          (~cross_df["train_basins"].str.contains(r"\+")) &
                          (cross_df["test_basin"] == b)]
        best_transfer.append(others[col].min() if len(others) else np.nan)

    all_basin = []
    for b in ALL_BASINS:
        row = gradual_df[(gradual_df["train_basins"] == "WP+NA+EP") &
                         (gradual_df["test_basin"] == b)]
        all_basin.append(row[col].values[0] if len(row) else np.nan)

    ax.bar(x - w, in_domain,     w, label="In-domain",            color="#457B9D")
    ax.bar(x,     best_transfer, w, label="Best transfer",         color="#E63946")
    ax.bar(x + w, all_basin,     w, label="All-basin (WP+NA+EP)", color="#2A9D8F")

    ax.set_xticks(x); ax.set_xticklabels(ALL_BASINS)
    ax.set_ylabel(CFG["summary_label"])
    ax.set_title(f"In-domain vs Cross-basin vs All-basin — {task.capitalize()}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"cross_basin_summary_{task}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── Per-task runner ────────────────────────────────────────────────────────────
def run_task(task, epochs, seed):
    global TARGET_COLS, TAB_COLS, CFG
    CFG         = TASK_CONFIG[task]
    TARGET_COLS = CFG["target_cols"]
    TAB_COLS    = _load_tab_cols(task)

    print(f"\n{'='*60}")
    print(f"Task: {task.upper()}")
    print(f"{'='*60}")

    csv_path = os.path.join(FEAT_DIR, CFG["csv"])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run features.py first — missing {CFG['csv']}")

    df = pd.read_csv(csv_path, keep_default_na=False)
    df = df[df["basin"].isin(ALL_BASINS)].dropna(subset=TARGET_COLS)
    for col in TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=TARGET_COLS)

    print(f"Basin counts: {df['basin'].value_counts().to_dict()}")
    print(f"Split counts: {df['split'].value_counts().to_dict()}")

    print("\nExperiment 1 & 2: Single-basin training + cross-basin transfer")
    cross_df = run_single_and_cross(df, epochs, seed, task)

    print("\nExperiment 3: Gradual basin addition")
    gradual_df = run_gradual(df, epochs, seed)

    all_results = pd.concat([cross_df, gradual_df], ignore_index=True)
    out_path = os.path.join(FEAT_DIR, f"cross_basin_results_{task}.csv")
    all_results.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    print("\nGenerating plots...")
    plot_heatmap(cross_df, task)
    plot_gradual(gradual_df, task)
    plot_summary(cross_df, gradual_df, task)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   choices=["decay", "landfall", "all"],
                        default="decay",
                        help="Which task to run (default: decay)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Epochs per experiment (default 60)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    print(f"Device: {DEVICE}  |  Epochs: {args.epochs}  |  Seed: {args.seed}")

    tasks = ["decay", "landfall"] if args.task == "all" else [args.task]
    for task in tasks:
        run_task(task, args.epochs, args.seed)

    print("\nCross-basin experiments complete.")
    print(f"Figures: {FIG_DIR}/cross_basin_*.png")


if __name__ == "__main__":
    main()
