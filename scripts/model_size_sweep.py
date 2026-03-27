"""
Model size vs performance sweep for decay and landfall tasks.

Trains CycloneUFNO at multiple capacity levels for both tasks and plots
a combined 2x2 figure:
  Row 1 (decay):    val loss vs param count | train/val gap vs param count
  Row 2 (landfall): val loss vs param count | train/val gap vs param count

Width configurations swept: [8, 12, 16, 24, 32]
Modes scaled proportionally: max(4, width // 2)

Each run uses the same data split, LR schedule, and early stopping so
results are directly comparable.  Checkpoints are saved to models/sweep/
and do not overwrite production checkpoints.

Usage
─────
  python scripts/model_size_sweep.py
  python scripts/model_size_sweep.py --epochs 50 --noise-std 0.02
  python scripts/model_size_sweep.py --widths 8 12 16 24 32 --epochs 50

Options
  --widths        list of int   Width values to sweep    (default: 8 12 16 24 32)
  --epochs        int           Max epochs per run       (default: 50)
  --early-stop    int           Early stopping patience  (default: 7)
  --batch         int           Batch size               (default: 8)
  --lr            float         Learning rate            (default: 3e-4)
  --noise-std     float         Tabular noise std        (default: 0.0)
  --unet-dropout  float         UNet dropout             (default: 0.2)
  --seed          int           Random seed              (default: 42)
  --no-spatial                  Force tabular-only mode
  --spatial-sigma float         Gaussian centre mask sigma (pixels, 0=off)

Outputs
  models/sweep/sweep_{task}_w{width}.pt  — best checkpoint per task/width
  figures/model_size_sweep.png           — combined 2x2 plot
  data/features/model_size_sweep.json   — raw results for both tasks
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
import torch

sys.path.insert(0, os.path.dirname(__file__))
from ufno import CycloneUFNO, LpLoss
from train_ufno import load_data, run_epoch

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SWEEP_DIR  = os.path.join(MODELS_DIR, "sweep")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "..", "figures")
FEAT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "features")
os.makedirs(SWEEP_DIR, exist_ok=True)
os.makedirs(FIG_DIR,   exist_ok=True)

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

# Task-specific settings
_TASK_CFG = {
    "decay":    {"n_outputs": 2, "criterion": LpLoss(p=2)},
    "landfall": {"n_outputs": 1, "criterion": torch.nn.MSELoss()},
}


def _modes_for_width(width: int, grid_size: int = 16, pad_size: int = 4) -> int:
    """Keep modes proportional to width, capped at half the padded pseudo-grid size.
    The tabular-only path uses a grid_size x grid_size field; after replication
    padding it becomes (grid_size + 2*pad_size) wide, giving (padded//2 + 1)
    rfft frequency bins.  modes must not exceed this or einsum will crash.
    For spatial patches (81x81 -> 89x89) the limit is 45, so only tabular-only
    mode is affected in practice.
    """
    padded_grid = grid_size + 2 * pad_size  # 24 with defaults
    max_modes   = padded_grid // 2          # 12 with defaults
    return min(max_modes, max(4, width // 2))


def train_one(task: str, width: int, args,
              train_loader, val_loader, meta: dict) -> dict:
    """
    Train a single CycloneUFNO configuration for one task and return results.
    Does not modify any production checkpoint.
    """
    modes     = _modes_for_width(width)
    n_outputs = _TASK_CFG[task]["n_outputs"]
    criterion = _TASK_CFG[task]["criterion"]
    ckpt_path = os.path.join(SWEEP_DIR, f"sweep_{task}_w{width}.pt")

    model = CycloneUFNO(
        sp_channels   = 4,
        T             = 8,
        tab_features  = meta["tab_dim"],
        modes1        = modes,
        modes2        = modes,
        width         = width,
        unet_dropout  = args.unet_dropout,
        n_outputs     = n_outputs,
    ).to(DEVICE)

    n_params  = model.count_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    best_val      = float("inf")
    best_train    = float("inf")
    epochs_no_imp = 0
    history       = {"train": [], "val": []}

    print(f"\n  [{task}] width={width:>2}  modes={modes}  params={n_params:,}")
    print(f"  {'Ep':>4}  {'Train':>8}  {'Val':>8}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(
            model, train_loader, meta["mode"], DEVICE,
            optimizer, criterion, show_bar=False,
)
        vl_loss, _ = run_epoch(
            model, val_loader, meta["mode"], DEVICE,
            criterion=criterion, show_bar=False)
        scheduler.step(vl_loss)

        history["train"].append(tr_loss)
        history["val"].append(vl_loss)

        if vl_loss < best_val:
            best_val      = vl_loss
            best_train    = tr_loss
            epochs_no_imp = 0
            torch.save({"state": model.state_dict(),
                        "task": task, "width": width, "modes": modes,
                        "n_params": n_params, "epoch": epoch,
                        "best_val": best_val},
                       ckpt_path)
        else:
            epochs_no_imp += 1

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  {epoch:>4}  {tr_loss:>8.4f}  {vl_loss:>8.4f}"
                  f"{'  ★' if epochs_no_imp == 0 else ''}")

        if args.early_stop > 0 and epochs_no_imp >= args.early_stop:
            print(f"  Early stop at epoch {epoch}")
            break

    gap = best_val - best_train

    print(f"  Best val={best_val:.4f}  train={best_train:.4f}  gap={gap:.4f}")

    return {
        "task":       task,
        "width":      width,
        "modes":      modes,
        "n_params":   n_params,
        "best_val":   round(best_val,   4),
        "best_train": round(best_train, 4),
        "gap":        round(gap,        4),
        "epochs_run": len(history["val"]),
        "history":    history,
    }


def plot_sweep(results_by_task: dict[str, list[dict]], out_path: str):
    """
    Combined 2x2 plot:
      Col 1: val loss vs parameter count
      Col 2: train/val gap vs parameter count (overfitting severity)
      Row 1: decay task
      Row 2: landfall task
    """
    tasks      = ["decay", "landfall"]
    task_label = {"decay": "Intensity decay", "landfall": "Landfall timing"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("CycloneUFNO: model size vs performance",
                 fontsize=14, fontweight="bold")

    for row, task in enumerate(tasks):
        results = sorted(results_by_task.get(task, []),
                         key=lambda r: r["n_params"])
        if not results:
            continue

        params     = [r["n_params"] for r in results]
        val_losses = [r["best_val"] for r in results]
        gaps       = [r["gap"]      for r in results]
        widths     = [r["width"]    for r in results]

        # ── Left panel: val loss ──────────────────────────────────────────────
        ax = axes[row, 0]
        ax.plot(params, val_losses, "o-", color="#2c7bb6", linewidth=2,
                markersize=8, markerfacecolor="white", markeredgewidth=2)
        for x, y, w in zip(params, val_losses, widths):
            ax.annotate(f"w={w}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color="#2c7bb6")

        best_idx = int(np.argmin(val_losses))
        ax.plot(params[best_idx], val_losses[best_idx], "*",
                color="#d7191c", markersize=14,
                label=f"Best (w={widths[best_idx]})", zorder=5)

        ax.set_xlabel("Parameter count", fontsize=10)
        ax.set_ylabel("Best val loss", fontsize=10)
        ax.set_title(f"{task_label[task]} — val loss vs model size", fontsize=10)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # ── Right panel: train/val gap ────────────────────────────────────────
        ax = axes[row, 1]
        colors = ["#1a9641" if g < 0.05 else
                  "#fdae61" if g < 0.15 else
                  "#d7191c" for g in gaps]
        bars = ax.bar([str(w) for w in widths], gaps, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.axhline(0.05, color="#1a9641", linestyle="--", linewidth=1,
                   alpha=0.7, label="Low overfitting")
        ax.axhline(0.15, color="#d7191c", linestyle="--", linewidth=1,
                   alpha=0.7, label="High overfitting")
        for bar, g in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{g:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Width", fontsize=10)
        ax.set_ylabel("Val loss - Train loss", fontsize=10)
        ax.set_title(f"{task_label[task]} — overfitting vs model size",
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


def _print_summary(task: str, results: list[dict]):
    print(f"\n{'─'*58}")
    print(f"  {task.upper()} SUMMARY")
    print(f"{'Width':>6}  {'Modes':>5}  {'Params':>10}  "
          f"{'Val loss':>9}  {'Train':>9}  {'Gap':>7}")
    print("─" * 58)
    best_val = min(r["best_val"] for r in results)
    for r in sorted(results, key=lambda x: x["n_params"]):
        star = " ★" if r["best_val"] == best_val else ""
        print(f"{r['width']:>6}  {r['modes']:>5}  {r['n_params']:>10,}  "
              f"{r['best_val']:>9.4f}  {r['best_train']:>9.4f}  "
              f"{r['gap']:>7.4f}{star}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths",       type=int, nargs="+",
                        default=[8, 12, 16, 24, 32])
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--early-stop",   type=int,   default=7)
    parser.add_argument("--batch",        type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--unet-dropout", type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-spatial",   action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device       : {DEVICE}")
    print(f"Widths       : {args.widths}")
    print(f"Modes        : {[_modes_for_width(w) for w in args.widths]}")
    print(f"Epochs       : {args.epochs}  |  Early stop: {args.early_stop}")

    results_by_task: dict[str, list[dict]] = {}
    t_start = time.time()

    for task in ["decay", "landfall"]:
        print(f"\n{'='*60}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*60}")

        print(f"\nLoading {task} data...")
        train_loader, val_loader, _, meta, _, _ = load_data(
            args.seed, task=task, batch=args.batch,
            no_spatial=args.no_spatial)
        print(f"Mode: {meta['mode']}  |  Tab features: {meta['tab_dim']}")

        task_results = []
        for width in args.widths:
            result = train_one(task, width, args,
                               train_loader, val_loader, meta)
            task_results.append(result)

        results_by_task[task] = task_results
        _print_summary(task, task_results)

    total_time = time.time() - t_start
    print(f"\nSweep complete in {total_time/60:.1f} min")

    # ── Save results JSON ─────────────────────────────────────────────────────
    out_results = {}
    for task, results in results_by_task.items():
        out_results[task] = []
        for r in results:
            row = {k: v for k, v in r.items() if k != "history"}
            row["val_curve"]   = r["history"]["val"]
            row["train_curve"] = r["history"]["train"]
            out_results[task].append(row)

    json_path = os.path.join(FEAT_DIR, "model_size_sweep.json")
    with open(json_path, "w") as f:
        json.dump({"args": vars(args), "results": out_results}, f, indent=2)
    print(f"Results      : {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(FIG_DIR, "model_size_sweep.png")
    plot_sweep(results_by_task, plot_path)


if __name__ == "__main__":
    main()
