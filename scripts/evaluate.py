"""
Evaluate a trained CycloneUFNO checkpoint.

Loads the best checkpoint, runs inference on the test split (or all splits),
prints a detailed results table, and saves a results JSON.

Usage
─────
  # Evaluate global decay model on test split
  python scripts/evaluate.py --task decay

  # Evaluate EP-only landfall model
  python scripts/evaluate.py --task landfall --basin EP

  # Evaluate a specific checkpoint on all splits
  python scripts/evaluate.py --task decay --ckpt models/last_ufno_decay.pt --all-splits

  # Compare global vs per-basin
  python scripts/evaluate.py --task decay
  python scripts/evaluate.py --task decay --basin EP
  python scripts/evaluate.py --task decay --basin WP
  python scripts/evaluate.py --task decay --basin NA

Options
  --task        {landfall,decay}    Task to evaluate        (default decay)
  --basin       {WP,NA,EP}          Evaluate on one basin   (default all)
  --ckpt        PATH                Override checkpoint path
  --all-splits                      Evaluate on all splits, not just test
  --save-json   PATH                Save results JSON (default: models/results_{task}[_{basin}].json)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from ufno import CycloneUFNO

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEAT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

DEVICE = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

TARGET_COLS     = ["wind_24h", "wind_48h"]
LANDFALL_TARGET = "hours_to_landfall"
VALID_BASINS    = {"WP", "NA", "EP"}


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    meta = ckpt["meta"]
    args = ckpt.get("args", {})

    n_outputs    = 1 if meta.get("task") == "landfall" else 2
    # Reconstruct lf_embed_dim by inspecting the saved weights directly.
    # If lf_film layers exist in the state dict, read their input dimension
    # from the weight shape — always correct regardless of what was logged in args.
    state = ckpt["state"]
    lf_key = "stack.lf_film.0.net.0.weight"
    if lf_key in state:
        lf_embed_dim = int(state[lf_key].shape[1])
    else:
        lf_embed_dim = 0
    model = CycloneUFNO(
        sp_channels  = 4,
        T            = 8,
        tab_features = meta["tab_dim"],
        modes1       = args.get("modes", 8 if meta.get("task") == "landfall" else 12),
        modes2       = args.get("modes", 8 if meta.get("task") == "landfall" else 12),
        width        = args.get("width", 16 if meta.get("task") == "landfall" else 32),
        unet_dropout = args.get("unet_dropout", 0.0),
        n_outputs    = n_outputs,
        lf_embed_dim = lf_embed_dim,
    ).to(DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, meta, ckpt


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(model, df: pd.DataFrame, meta: dict, ckpt: dict,
                   task: str) -> np.ndarray:
    """
    Run inference on df. Returns raw normalised predictions (N, n_outputs).
    Uses scaler params from checkpoint so no refit is needed.
    """
    tab_cols = [c for c in meta.get("tab_cols", []) if c in df.columns]
    if not tab_cols:
        raise ValueError("No matching tab_cols found in dataframe.")

    X = df[tab_cols].replace("", np.nan).apply(
        pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    tab_mean  = np.array(ckpt.get("tab_mean",  [0.0] * X.shape[1]))
    tab_scale = np.array(ckpt.get("tab_scale", [1.0] * X.shape[1]))
    # Clip scale to avoid division by zero
    tab_scale = np.where(tab_scale == 0, 1.0, tab_scale)
    X_scaled  = (X - tab_mean) / tab_scale

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), 64):
            batch = torch.tensor(X_scaled[i:i+64], dtype=torch.float32).to(DEVICE)
            out   = model(x_tab=batch).cpu().numpy()
            preds.append(out)

    return np.vstack(preds)   # (N, n_outputs)


def _denorm(preds_norm: np.ndarray, ckpt: dict) -> np.ndarray:
    """Denormalise predictions using checkpoint scaler params."""
    tgt_mean  = np.array(ckpt["tgt_mean"])
    tgt_scale = np.array(ckpt["tgt_scale"])
    return preds_norm * tgt_scale + tgt_mean


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, bias, R² for a single output dimension."""
    err  = predicted - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    r2   = float(r2_score(actual, predicted))
    return {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2}


# ── Per-basin breakdown ───────────────────────────────────────────────────────

def _per_basin(df: pd.DataFrame, pred_cols: list, actual_cols: list) -> dict:
    results = {}
    for basin, grp in df.groupby("basin"):
        results[basin] = {}
        for pred_col, actual_col in zip(pred_cols, actual_cols):
            act  = pd.to_numeric(grp[actual_col], errors="coerce").dropna().values
            pred = grp.loc[grp[actual_col].notna(), pred_col].values
            if len(act) == 0:
                continue
            results[basin][actual_col] = _metrics(act, pred)
            results[basin][actual_col]["n"] = len(act)
    return results


# ── Print tables ──────────────────────────────────────────────────────────────

def _print_decay(df: pd.DataFrame, overall: dict, by_basin: dict,
                 ckpt_path: str, split_label: str):
    print(f"\n{'═'*62}")
    print(f"  CycloneUFNO — Decay Evaluation  [{split_label}]")
    print(f"  Checkpoint : {os.path.basename(ckpt_path)}")
    print(f"  Samples    : {len(df)}")
    print(f"{'═'*62}")
    print(f"  {'Horizon':<10}  {'R²':>6}  {'RMSE (kt)':>10}  "
          f"{'MAE (kt)':>9}  {'Bias':>7}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*7}")
    for horizon, col in [("24 h", "wind_24h"), ("48 h", "wind_48h")]:
        m = overall[col]
        print(f"  {horizon:<10}  {m['r2']:>6.3f}  {m['rmse']:>10.3f}  "
              f"{m['mae']:>9.3f}  {m['bias']:>+7.3f}")
    print()
    print(f"  {'Basin':<6}  {'N':>4}  "
          f"{'R²-24h':>7}  {'MAE-24h':>8}  {'R²-48h':>7}  {'MAE-48h':>8}")
    print(f"  {'─'*6}  {'─'*4}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*8}")
    for basin in sorted(by_basin):
        b = by_basin[basin]
        n    = b.get("wind_24h", {}).get("n", 0)
        r24  = b.get("wind_24h", {}).get("r2",  float("nan"))
        m24  = b.get("wind_24h", {}).get("mae", float("nan"))
        r48  = b.get("wind_48h", {}).get("r2",  float("nan"))
        m48  = b.get("wind_48h", {}).get("mae", float("nan"))
        print(f"  {basin:<6}  {n:>4}  "
              f"{r24:>7.3f}  {m24:>8.3f}  {r48:>7.3f}  {m48:>8.3f}")
    print(f"{'═'*62}")


def _print_landfall(df: pd.DataFrame, overall: dict, by_basin: dict,
                    ckpt_path: str, split_label: str):
    print(f"\n{'═'*55}")
    print(f"  CycloneUFNO — Landfall Timing  [{split_label}]")
    print(f"  Checkpoint : {os.path.basename(ckpt_path)}")
    print(f"  Samples    : {len(df)}")
    print(f"{'═'*55}")
    m = overall[LANDFALL_TARGET]
    print(f"  R²   : {m['r2']:>7.3f}")
    print(f"  RMSE : {m['rmse']:>7.2f} h")
    print(f"  MAE  : {m['mae']:>7.2f} h")
    print(f"  Bias : {m['bias']:>+7.2f} h")
    print()
    print(f"  {'Basin':<6}  {'N':>4}  {'R²':>7}  {'RMSE (h)':>9}  {'MAE (h)':>8}")
    print(f"  {'─'*6}  {'─'*4}  {'─'*7}  {'─'*9}  {'─'*8}")
    for basin in sorted(by_basin):
        b = by_basin[basin]
        n    = b.get(LANDFALL_TARGET, {}).get("n",    0)
        r2   = b.get(LANDFALL_TARGET, {}).get("r2",   float("nan"))
        rmse = b.get(LANDFALL_TARGET, {}).get("rmse", float("nan"))
        mae  = b.get(LANDFALL_TARGET, {}).get("mae",  float("nan"))
        print(f"  {basin:<6}  {n:>4}  {r2:>7.3f}  {rmse:>9.2f}  {mae:>8.2f}")
    print(f"{'═'*55}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="decay", choices=["decay", "landfall"])
    parser.add_argument("--basin",  default=None,
                        help="Evaluate on one basin only: WP, NA, EP")
    parser.add_argument("--ckpt",   default=None,
                        help="Override checkpoint path")
    parser.add_argument("--all-splits", action="store_true",
                        help="Evaluate on all splits, not just test")
    parser.add_argument("--save-json", default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    if args.basin is not None:
        args.basin = args.basin.upper()
        if args.basin not in VALID_BASINS:
            print(f"Error: --basin must be one of {sorted(VALID_BASINS)}")
            sys.exit(1)

    # ── Resolve checkpoint path ─────────────────────────────────────────────
    basin_tag = f"_{args.basin}" if args.basin else ""
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(MODELS_DIR,
                                 f"best_ufno_{args.task}{basin_tag}.pt")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"Run: python scripts/train_ufno.py --task {args.task}"
              + (f" --basin {args.basin}" if args.basin else ""))
        sys.exit(1)

    print(f"Loading checkpoint : {ckpt_path}")
    model, meta, ckpt = load_checkpoint(ckpt_path)
    trained_basin = meta.get("basin")
    print(f"Task               : {args.task}")
    print(f"Trained on basin   : {trained_basin or 'all'}")
    print(f"Tab features       : {meta['tab_dim']}")
    print(f"Epoch              : {ckpt.get('epoch', '?')}")

    # ── Load feature matrix ─────────────────────────────────────────────────
    csv_path = os.path.join(FEAT_DIR, f"feature_matrix_{args.task}.csv")
    if not os.path.exists(csv_path):
        print(f"Feature matrix not found: {csv_path}\nRun features.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, keep_default_na=False)

    # Apply split filter
    if args.all_splits:
        split_label = "all splits"
    else:
        if "split" in df.columns and df["split"].isin(["train", "val", "test"]).any():
            df = df[df["split"] == "test"].reset_index(drop=True)
            split_label = "test split"
        else:
            split_label = "all splits (no split column found)"

    # Apply basin filter for evaluation scope
    if args.basin:
        df = df[df["basin"] == args.basin].reset_index(drop=True)

    # Task-specific filtering
    if args.task == "landfall":
        df[LANDFALL_TARGET] = pd.to_numeric(df[LANDFALL_TARGET], errors="coerce")
        df = df[df[LANDFALL_TARGET] > 0].reset_index(drop=True)
    else:
        df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)

    if len(df) == 0:
        print("No samples found after filtering. Check --basin and --task.")
        sys.exit(1)

    print(f"Evaluating on      : {len(df)} samples  [{split_label}]"
          + (f"  basin={args.basin}" if args.basin else ""))

    # ── Run inference ───────────────────────────────────────────────────────
    preds_norm = _run_inference(model, df, meta, ckpt, args.task)
    preds      = _denorm(preds_norm, ckpt)

    # ── Attach predictions to dataframe ─────────────────────────────────────
    if args.task == "decay":
        df["pred_24h"] = preds[:, 0]
        df["pred_48h"] = preds[:, 1]
        actual_cols = ["wind_24h", "wind_48h"]
        pred_cols   = ["pred_24h", "pred_48h"]
    else:
        df["pred_htl"] = preds[:, 0]
        actual_cols = [LANDFALL_TARGET]
        pred_cols   = ["pred_htl"]

    # ── Compute metrics ─────────────────────────────────────────────────────
    overall  = {}
    for pred_col, actual_col in zip(pred_cols, actual_cols):
        act  = pd.to_numeric(df[actual_col], errors="coerce").values
        pred = df[pred_col].values
        mask = ~np.isnan(act)
        overall[actual_col] = _metrics(act[mask], pred[mask])
        overall[actual_col]["n"] = int(mask.sum())

    by_basin = _per_basin(df, pred_cols, actual_cols)

    # ── Print results ────────────────────────────────────────────────────────
    if args.task == "decay":
        _print_decay(df, overall, by_basin, ckpt_path, split_label)
    else:
        _print_landfall(df, overall, by_basin, ckpt_path, split_label)

    # ── Save results JSON ────────────────────────────────────────────────────
    results = {
        "checkpoint":   ckpt_path,
        "task":         args.task,
        "eval_basin":   args.basin,
        "trained_basin":trained_basin,
        "split":        split_label,
        "n_samples":    len(df),
        "overall":      overall,
        "by_basin":     by_basin,
    }

    json_path = args.save_json or os.path.join(
        MODELS_DIR, f"results_{args.task}{basin_tag}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    main()