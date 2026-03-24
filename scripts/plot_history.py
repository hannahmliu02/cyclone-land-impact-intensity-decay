"""
Plot training and validation loss curves from saved history files.

Usage
─────
  python scripts/plot_history.py --task landfall --basin EP
  python scripts/plot_history.py --task decay --basin EP
  python scripts/plot_history.py --task decay  # global model
"""

import argparse
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

matplotlib.rcParams.update({
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
})


def plot_history(history: dict, task: str, basin: str, save_path: str):
    epochs     = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    train_mae  = history["train_mae"]
    val_mae    = history["val_mae"]

    best_epoch = int(val_loss.index(min(val_loss))) + 1
    best_val   = min(val_loss)

    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(
        f"Training history — {task}  |  basin: {basin or 'all'}  "
        f"|  best epoch: {best_epoch}  (val={best_val:.4f})",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, label="Train", color="#457B9D", lw=1.5)
    ax1.plot(epochs, val_loss,   label="Val",   color="#E63946", lw=1.5)
    ax1.axvline(best_epoch, color="#2A9D8F", lw=1, ls="--",
                label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train / Val Loss")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # ── MAE ───────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_mae, label="Train", color="#457B9D", lw=1.5)
    ax2.plot(epochs, val_mae,   label="Val",   color="#E63946", lw=1.5)
    ax2.axvline(best_epoch, color="#2A9D8F", lw=1, ls="--",
                label=f"Best epoch {best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (normalised)")
    ax2.set_title("Train / Val MAE")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default="decay",
                        choices=["decay", "landfall"])
    parser.add_argument("--basin", default=None,
                        help="Basin suffix e.g. EP — leave blank for global model")
    parser.add_argument("--show",  action="store_true",
                        help="Open interactive window instead of saving")
    args = parser.parse_args()

    basin_tag = f"_{args.basin.upper()}" if args.basin else ""
    hist_path = os.path.join(MODELS_DIR,
                             f"ufno_history_{args.task}{basin_tag}.json")

    if not os.path.exists(hist_path):
        print(f"History file not found: {hist_path}")
        print(f"Run training first: python scripts/train_ufno_v2.py "
              f"--task {args.task}"
              + (f" --basin {args.basin}" if args.basin else ""))
        sys.exit(1)

    with open(hist_path) as f:
        history = json.load(f)

    print(f"Loaded : {hist_path}")
    print(f"Epochs : {len(history['train_loss'])}")
    print(f"Best val loss : {min(history['val_loss']):.4f} "
          f"at epoch {history['val_loss'].index(min(history['val_loss'])) + 1}")

    save_path = os.path.join(
        FIG_DIR, f"training_history_{args.task}{basin_tag}.png")

    if args.show:
        matplotlib.use("TkAgg")

    plot_history(history, args.task, args.basin, save_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()