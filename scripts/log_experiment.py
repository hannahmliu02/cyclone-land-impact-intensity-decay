"""
Log a completed experiment to experiments/.

Usage (run after training):
  python scripts/log_experiment.py \
    --name "My experiment description" \
    --notes "What I observed"

Automatically reads:
  - models/best_ufno.pt        → hyperparameters, data meta
  - models/ufno_history.json   → train/val loss curves
  - figures/ufno_results/predictions.csv → per-basin RMSE

Writes:
  experiments/exp_NNN_<slug>.json
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import torch

EXPR_DIR   = os.path.join(os.path.dirname(__file__), "..", "experiments")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "..", "figures", "ufno_results")
os.makedirs(EXPR_DIR, exist_ok=True)


def _next_id():
    existing = [f for f in os.listdir(EXPR_DIR) if f.startswith("exp_") and f.endswith(".json")]
    nums = [int(re.search(r"exp_(\d+)", f).group(1)) for f in existing
            if re.search(r"exp_(\d+)", f)]
    return max(nums, default=0) + 1


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower())[:40].strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",  required=True, help="Short experiment name")
    parser.add_argument("--notes", default="",    help="Observations / notes")
    parser.add_argument("--ckpt",  default=os.path.join(MODELS_DIR, "best_ufno.pt"))
    args = parser.parse_args()

    from datetime import date
    exp_id   = _next_id()
    exp_name = f"exp_{exp_id:03d}"
    today    = date.today().isoformat()

    record = {
        "id":   exp_name,
        "name": args.name,
        "date": today,
        "script": "scripts/train_ufno.py",
        "description": args.name,
        "hyperparameters": {},
        "data": {},
        "results": {},
        "observations": args.notes,
        "checkpoint": f"models/best_ufno.pt (at time of logging)",
    }

    # ── Read checkpoint ────────────────────────────────────────────────────────
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        record["hyperparameters"] = ckpt.get("args", {})
        meta = ckpt.get("meta", {})
        record["data"] = {
            "basins": ["WP", "NA", "EP"],
            "split":  "TCND original",
            "n_train": meta.get("n_train"),
            "n_val":   meta.get("n_val"),
            "n_test":  meta.get("n_test"),
            "tab_cols": meta.get("tab_dim"),
            "tab_col_names": meta.get("tab_cols", []),
        }
        record["results"]["best_epoch"] = ckpt.get("epoch")
    else:
        print(f"  [warn] Checkpoint not found: {args.ckpt}")

    # ── Read training history ──────────────────────────────────────────────────
    hist_path = os.path.join(MODELS_DIR, "ufno_history.json")
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        vl = h["val_loss"]
        tl = h["train_loss"]
        record["results"].update({
            "epochs_trained":    len(tl),
            "best_val_loss":     round(min(vl), 4),
            "best_val_epoch":    vl.index(min(vl)) + 1,
            "final_train_loss":  round(tl[-1], 4),
            "final_val_loss":    round(vl[-1], 4),
        })

    # ── Read predictions ───────────────────────────────────────────────────────
    pred_path = os.path.join(FIG_DIR, "predictions.csv")
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path, keep_default_na=False)
        per_basin = {}
        for b, g in df.groupby("basin"):
            per_basin[b] = {
                "n":        int(len(g)),
                "rmse_24h": round(float(np.sqrt((g["err_24h"]**2).mean())), 4),
                "rmse_48h": round(float(np.sqrt((g["err_48h"]**2).mean())), 4),
                "mae_24h":  round(float(g["mae_24h"].mean()), 4),
                "mae_48h":  round(float(g["mae_48h"].mean()), 4),
                "bias_24h": round(float(g["err_24h"].mean()), 4),
                "bias_48h": round(float(g["err_48h"].mean()), 4),
            }
        record["results"]["per_basin"] = per_basin
    else:
        print(f"  [warn] Predictions not found: {pred_path}")
        print(f"         Run view_results.py first to generate predictions.")

    # ── Write record ───────────────────────────────────────────────────────────
    fname = f"{exp_name}_{_slug(args.name)}.json"
    out   = os.path.join(EXPR_DIR, fname)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nExperiment logged: {out}")
    print(f"  Best val loss : {record['results'].get('best_val_loss')} "
          f"(epoch {record['results'].get('best_val_epoch')})")
    if "per_basin" in record["results"]:
        for b, m in record["results"]["per_basin"].items():
            print(f"  {b}: RMSE_24h={m['rmse_24h']}  RMSE_48h={m['rmse_48h']}")


if __name__ == "__main__":
    main()
