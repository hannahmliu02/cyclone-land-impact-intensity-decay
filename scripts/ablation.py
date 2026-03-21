"""
Ablation study — systematically test feature group contributions.

For each task × feature-group-subset, trains a gradient-boosted tree
(XGBoost) and reports held-out metrics.  Results are saved as:
  data/features/ablation_landfall.csv
  data/features/ablation_decay.csv
  figures/ablation_landfall.png
  figures/ablation_decay.png

Ablation strategies
───────────────────
  "full"               — all feature groups
  "leave_one_out"      — full minus one group (shows what each group adds)
  "single_group"       — only one group at a time
  "wind_vs_pressure"   — wind-only vs. pressure-only vs. both
  "spatial_vs_tabular" — spatial-only vs. tabular-only vs. both
  "no_land_sea"        — full minus land_sea mask (landfall task)
  "3d_focus"           — spatial + env (intensity decay task)
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_FOLDS = 5
SEED    = 42

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbosity=0,
    n_jobs=-1,
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load(task: str):
    path = os.path.join(FEAT_DIR, f"feature_matrix_{task}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run features.py first — {path} not found.")
    return pd.read_csv(path, keep_default_na=False)

def _load_groups():
    with open(os.path.join(FEAT_DIR, "feature_groups.json")) as f:
        return json.load(f)

_META = {"storm_id","basin","ref_time","made_landfall","hours_to_landfall",
         "landfall_wind","landfall_pres","wind_24h","wind_48h",
         "wind_frac_24h","wind_frac_48h"}

def _select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    return df[available].copy()

def _eval_classification(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auc"] = np.nan
    return metrics

def _eval_regression(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _cv_classification(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    accs, aucs = [], []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        sc = StandardScaler()
        Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns)
        Xte = pd.DataFrame(sc.transform(Xte),     columns=Xte.columns)

        model = xgb.XGBClassifier(**XGB_PARAMS, eval_metric="logloss",
                                   use_label_encoder=False)
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        accs.append(accuracy_score(yte, pred))
        try:
            aucs.append(roc_auc_score(yte, prob))
        except Exception:
            aucs.append(np.nan)

    return {"label": label,
            "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
            "auc_mean":      np.mean(aucs), "auc_std":      np.std(aucs)}


def _cv_regression(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    rmses, maes, r2s = [], [], []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        sc = StandardScaler()
        Xtr = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns)
        Xte = pd.DataFrame(sc.transform(Xte),     columns=Xte.columns)

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        pred = model.predict(Xte)
        m = _eval_regression(yte, pred)
        rmses.append(m["rmse"]); maes.append(m["mae"]); r2s.append(m["r2"])

    return {"label": label,
            "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
            "mae_mean":  np.mean(maes),  "mae_std":  np.std(maes),
            "r2_mean":   np.mean(r2s),   "r2_std":   np.std(r2s)}


# ── Define ablation configs ────────────────────────────────────────────────────
def _build_configs(groups: dict) -> dict:
    all_groups = list(groups.keys())

    configs = {}

    # 1. Full feature set
    configs["full"] = all_groups

    # 2. Leave-one-out
    for g in all_groups:
        configs[f"leave_out_{g}"] = [x for x in all_groups if x != g]

    # 3. Single group
    for g in all_groups:
        configs[f"only_{g}"] = [g]

    # 4. Wind vs. pressure comparisons
    configs["wind_only"]     = ["wind"]
    configs["pressure_only"] = ["pressure"]
    configs["wind+pressure"] = ["wind", "pressure", "wp_couple"]

    # 5. Spatial vs. tabular
    tabular = ["wind", "pressure", "wp_couple", "position"]
    spatial  = ["spatial"]
    configs["tabular_only"]         = tabular
    configs["spatial_only"]         = spatial
    configs["tabular+spatial"]      = tabular + spatial
    configs["tabular+spatial+env"]  = tabular + spatial + ["env"]

    # 6. Land-sea mask impact (landfall-specific)
    configs["no_land_sea"]   = [g for g in all_groups if g != "land_sea"]
    configs["land_sea_only"] = ["land_sea"]

    # 7. 3D-focus (intensity decay emphasis per advisor guidance)
    configs["3d_focus"]       = ["spatial", "env"]
    configs["3d+wind"]        = ["spatial", "env", "wind"]
    configs["3d+pressure"]    = ["spatial", "env", "pressure"]
    configs["3d+tabular"]     = ["spatial", "env"] + tabular

    return configs


def _get_feature_cols(df, groups, group_names):
    cols = []
    for g in group_names:
        cols += [c for c in groups.get(g, []) if c in df.columns]
    return list(dict.fromkeys(cols))   # dedup, preserve order


# ── Run ablation for one task ──────────────────────────────────────────────────
def _run_task(task: str, target: str, is_clf: bool,
              df: pd.DataFrame, groups: dict, configs: dict) -> pd.DataFrame:
    print(f"\n{'='*60}\nTask: {task}  |  Target: {target}\n{'='*60}")

    df = df.dropna(subset=[target]).copy()
    y  = df[target]
    results = []

    for name, group_names in configs.items():
        feat_cols = _get_feature_cols(df, groups, group_names)
        # Filter to groups that actually exist in this task's matrix
        feat_cols = [c for c in feat_cols if c in df.columns and c not in _META]
        if not feat_cols:
            continue
        X = df[feat_cols].fillna(df[feat_cols].median())

        print(f"  {name:<30}  {len(feat_cols):3d} features", end="  ")
        if is_clf:
            res = _cv_classification(X, y, name)
            print(f"AUC={res['auc_mean']:.3f}±{res['auc_std']:.3f}  "
                  f"Acc={res['accuracy_mean']:.3f}")
        else:
            res = _cv_regression(X, y, name)
            print(f"RMSE={res['rmse_mean']:.2f}±{res['rmse_std']:.2f}  "
                  f"R²={res['r2_mean']:.3f}")

        results.append({**res, "n_features": len(feat_cols),
                        "groups": ",".join(group_names)})

    return pd.DataFrame(results)


# ── Plotting ───────────────────────────────────────────────────────────────────
def _plot_ablation(results: pd.DataFrame, task: str, metric: str, title: str):
    # Focus on the comparison configs (not leave-one-out for clarity)
    highlight = [r for r in results["label"]
                 if not r.startswith("leave_out_") and not r.startswith("only_")]
    df = results[results["label"].isin(highlight)].copy()
    df = df.sort_values(f"{metric}_mean", ascending=(metric == "rmse"))

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 2))
    colors = ["#2c7bb6" if "full" in l else
              "#d7191c" if "3d" in l else
              "#1a9641" if "spatial" in l else
              "#fdae61" for l in df["label"]]
    bars = ax.barh(df["label"], df[f"{metric}_mean"], color=colors,
                   xerr=df.get(f"{metric}_std"), capsize=3)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(df[df["label"]=="full"][f"{metric}_mean"].values[0],
               color="black", linestyle="--", linewidth=1, label="full model")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"ablation_{task}_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    lf_df  = _load("landfall")
    dc_df  = _load("decay")
    groups = _load_groups()
    configs = _build_configs(groups)

    # ── Task 1: Landfall prediction (binary) ──────────────────────────────────
    res_lf = _run_task("landfall", "made_landfall", True, lf_df, groups, configs)
    lf_path = os.path.join(FEAT_DIR, "ablation_landfall.csv")
    res_lf.to_csv(lf_path, index=False)
    print(f"\nSaved: {lf_path}")
    _plot_ablation(res_lf, "landfall", "auc", "Landfall Prediction — AUC by Feature Group")

    # ── Task 2a: Intensity decay at 24 h ─────────────────────────────────────
    res_24 = _run_task("decay_24h", "wind_24h", False, dc_df, groups, configs)
    res_24.to_csv(os.path.join(FEAT_DIR, "ablation_decay_24h.csv"), index=False)
    _plot_ablation(res_24, "decay_24h", "rmse",
                   "Intensity Decay (24 h) — RMSE by Feature Group")

    # ── Task 2b: Intensity decay at 48 h ─────────────────────────────────────
    res_48 = _run_task("decay_48h", "wind_48h", False, dc_df, groups, configs)
    res_48.to_csv(os.path.join(FEAT_DIR, "ablation_decay_48h.csv"), index=False)
    _plot_ablation(res_48, "decay_48h", "rmse",
                   "Intensity Decay (48 h) — RMSE by Feature Group")

    print("\nAblation complete.")

    # ── Auto-update selected feature groups for train_ufno.py ────────────────
    _save_selected_groups(res_24, groups)


def _save_selected_groups(res_24: pd.DataFrame, groups: dict):
    """
    From decay_24h ablation results, pick the best-performing feature groups
    (by R²) from the single-group rows ('only_*'), then write the selected
    group names to data/features/selected_feature_groups.json so that
    train_ufno.py can load them dynamically.
    """
    # Only look at single-group runs
    single = res_24[res_24["label"].str.startswith("only_")].copy()
    if single.empty:
        print("  [warn] No single-group rows found — skipping auto-update.")
        return

    # Parse group name from label ("only_wind" → "wind")
    single = single.copy()
    single["group"] = single["label"].str.replace("only_", "", n=1)

    # Keep groups with positive R²
    positive = single[single["r2_mean"] > 0].sort_values("r2_mean", ascending=False)
    if positive.empty:
        print("  [warn] No groups with positive R² — keeping all groups.")
        selected = list(single["group"])
    else:
        selected = list(positive["group"])

    out_path = os.path.join(FEAT_DIR, "selected_feature_groups.json")
    with open(out_path, "w") as f:
        json.dump({"selected_groups": selected,
                   "ranked_by": "r2_mean_decay_24h"}, f, indent=2)

    print(f"\n  Selected feature groups (by R²): {selected}")
    print(f"  Written to: {out_path}")


if __name__ == "__main__":
    run()
