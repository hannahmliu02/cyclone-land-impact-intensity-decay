"""
SHAP + LIME feature importance and ranking.

Trains a full XGBoost model on all features for each task, then:
  1. SHAP — global importance, summary plots, group-level comparisons
  2. LIME — local explanation for the highest-impact individual samples
  3. Rankings — final feature and group rankings saved as CSVs + bar charts

Outputs (figures/ and data/features/):
  figures/shap_<task>_beeswarm.png
  figures/shap_<task>_bar.png
  figures/shap_<task>_group_comparison.png
  figures/shap_<task>_wind_vs_pressure.png
  figures/shap_<task>_spatial_vs_tabular.png
  figures/lime_<task>_top_samples.png
  data/features/feature_rankings_<task>.csv
  data/features/group_rankings_<task>.csv
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42

_META = {"storm_id","basin","ref_time","made_landfall","hours_to_landfall",
         "landfall_wind","landfall_pres","wind_24h","wind_48h",
         "wind_frac_24h","wind_frac_48h"}

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbosity=0, n_jobs=-1,
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load(task):
    p = os.path.join(FEAT_DIR, f"feature_matrix_{task}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Run features.py first — {p} not found.")
    return pd.read_csv(p)

def _load_groups():
    with open(os.path.join(FEAT_DIR, "feature_groups.json")) as f:
        return json.load(f)

def _feature_cols(df):
    return [c for c in df.columns if c not in _META]

def _prep(df, target, is_clf):
    df = df.dropna(subset=[target]).copy()
    feat_cols = _feature_cols(df)
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[target]
    return X, y, feat_cols


# ── Train full model ───────────────────────────────────────────────────────────
def _train_full(X, y, is_clf):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
        stratify=y if is_clf else None)
    sc = StandardScaler()
    Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns)
    Xte_s = pd.DataFrame(sc.transform(Xte),     columns=Xte.columns)

    if is_clf:
        model = xgb.XGBClassifier(**XGB_PARAMS, eval_metric="logloss",
                                   use_label_encoder=False)
    else:
        model = xgb.XGBRegressor(**XGB_PARAMS)

    model.fit(Xtr_s, ytr, eval_set=[(Xte_s, yte)], verbose=False)
    return model, sc, Xtr_s, Xte_s, ytr, yte


# ── SHAP analysis ──────────────────────────────────────────────────────────────
def _shap_analysis(model, X_train, X_test, groups, task, is_clf):
    print(f"  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Handle multi-output SHAP (classification: use positive class)
    sv = shap_values.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs_shap = np.abs(sv).mean(axis=0)
    feat_names    = list(X_test.columns)

    # ── 1. Beeswarm plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, len(feat_names) * 0.25)))
    shap.summary_plot(sv, X_test, show=False, max_display=30)
    plt.title(f"SHAP Beeswarm — {task}")
    plt.tight_layout()
    p = os.path.join(FIG_DIR, f"shap_{task}_beeswarm.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {p}")

    # ── 2. Bar plot (mean |SHAP|) ──────────────────────────────────
    ranking = pd.DataFrame({"feature": feat_names,
                            "mean_abs_shap": mean_abs_shap})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    fig, ax = plt.subplots(figsize=(10, max(5, len(ranking.head(30)) * 0.35)))
    top = ranking.head(30)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2c7bb6")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP) — {task}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, f"shap_{task}_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {p}")

    # ── 3. Group-level SHAP comparison ────────────────────────────
    group_shap = {}
    for g, cols in groups.items():
        idx = [i for i, f in enumerate(feat_names) if f in cols]
        if idx:
            group_shap[g] = float(np.abs(sv[:, idx]).sum(axis=1).mean())

    group_df = pd.DataFrame(list(group_shap.items()),
                            columns=["group", "group_mean_abs_shap"])
    group_df = group_df.sort_values("group_mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    palette = {"wind": "#d7191c", "pressure": "#fdae61", "wp_couple": "#e8601c",
               "position": "#abd9e9", "spatial": "#2c7bb6", "env": "#1a9641",
               "land_sea": "#7b3294"}
    colors = [palette.get(g, "#aaaaaa") for g in group_df["group"]]
    ax.bar(group_df["group"], group_df["group_mean_abs_shap"], color=colors)
    ax.set_ylabel("Mean |SHAP| contribution")
    ax.set_title(f"Feature Group Importance — {task}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, f"shap_{task}_group_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {p}")

    # ── 4. Wind vs. Pressure ───────────────────────────────────────
    _pair_bar(sv, feat_names, groups, task,
              ("wind", "pressure", "wp_couple"), "wind_vs_pressure",
              "Wind vs. Pressure SHAP Contributions")

    # ── 5. Spatial vs. Tabular ─────────────────────────────────────
    _pair_bar(sv, feat_names, groups, task,
              ("spatial",), "spatial_vs_tabular_3d",
              "Spatial (3D) vs. Tabular SHAP Contributions",
              complement_label="tabular",
              complement_groups=("wind","pressure","wp_couple","position"))

    return ranking, group_df


def _pair_bar(sv, feat_names, groups, task, group_keys, fname, title,
              complement_label=None, complement_groups=None):
    data = {}
    for g in group_keys:
        idx = [i for i, f in enumerate(feat_names) if f in groups.get(g, [])]
        if idx:
            data[g] = float(np.abs(sv[:, idx]).sum(axis=1).mean())
    if complement_label and complement_groups:
        idx = [i for i, f in enumerate(feat_names)
               if any(f in groups.get(g, []) for g in complement_groups)]
        if idx:
            data[complement_label] = float(np.abs(sv[:, idx]).sum(axis=1).mean())
    if not data:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"wind": "#d7191c", "pressure": "#fdae61", "wp_couple": "#e8601c",
               "spatial": "#2c7bb6", "tabular": "#abdda4", "env": "#1a9641"}
    ax.bar(data.keys(), data.values(),
           color=[palette.get(k, "#aaaaaa") for k in data])
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"{title}\n{task}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, f"shap_{task}_{fname}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {p}")


# ── LIME analysis ──────────────────────────────────────────────────────────────
def _lime_analysis(model, sc, X_train, X_test, y_test, task, is_clf, n_samples=5):
    print(f"  Computing LIME explanations for top {n_samples} samples...")

    predict_fn = (model.predict_proba if is_clf else
                  lambda x: model.predict(x).reshape(-1, 1))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=list(X_train.columns),
        mode="classification" if is_clf else "regression",
        random_state=SEED,
    )

    # Pick most impactful samples: highest predicted probability / prediction
    if is_clf:
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = np.abs(model.predict(X_test) - y_test.values)

    top_idx = np.argsort(scores)[-n_samples:][::-1]

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        exp = explainer.explain_instance(
            X_test.iloc[idx].values,
            predict_fn,
            num_features=12,
            labels=(1,) if is_clf else None,
        )
        lime_vals = exp.as_list(label=1) if is_clf else exp.as_list()
        feats  = [v[0] for v in lime_vals]
        weights = [v[1] for v in lime_vals]
        colors = ["#2c7bb6" if w > 0 else "#d7191c" for w in weights]
        ax.barh(feats[::-1], weights[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"LIME — sample {idx}  "
                     f"({'landfall' if is_clf else f'wind={y_test.iloc[idx]:.1f} kt'})")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"LIME Local Explanations — {task}", fontsize=13, y=1.01)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, f"lime_{task}_top_samples.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Saved: {p}")


# ── Final ranking summary ──────────────────────────────────────────────────────
def _save_rankings(feat_ranking, group_ranking, task):
    feat_path  = os.path.join(FEAT_DIR, f"feature_rankings_{task}.csv")
    group_path = os.path.join(FEAT_DIR, f"group_rankings_{task}.csv")
    feat_ranking.to_csv(feat_path,   index=False)
    group_ranking.to_csv(group_path, index=False)
    print(f"\n  Feature rankings saved: {feat_path}")
    print(f"  Group rankings saved  : {group_path}")

    print(f"\n  Top 10 features [{task}]:")
    print(feat_ranking.head(10)[["rank","feature","mean_abs_shap"]].to_string(index=False))

    print(f"\n  Group ranking [{task}]:")
    print(group_ranking.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────
def _run_task(task, target, is_clf):
    print(f"\n{'='*60}\nTask: {task}  |  Target: {target}")
    df     = _load("landfall" if task == "landfall" else "decay")
    groups = _load_groups()

    X, y, feat_cols = _prep(df, target, is_clf)
    if len(X) < 20:
        print(f"  Too few samples ({len(X)}) — skipping.")
        return

    print(f"  Samples: {len(X)}  Features: {len(feat_cols)}")
    model, sc, Xtr, Xte, ytr, yte = _train_full(X, y, is_clf)

    feat_rank, group_rank = _shap_analysis(model, Xtr, Xte, groups, task, is_clf)
    _lime_analysis(model, sc, Xtr, Xte, yte, task, is_clf)
    _save_rankings(feat_rank, group_rank, task)


def run():
    # Task 1 — Landfall prediction (binary classification)
    _run_task("landfall", "made_landfall", is_clf=True)

    # Task 2a — Intensity decay at 24 h (regression)
    _run_task("decay_24h", "wind_24h", is_clf=False)

    # Task 2b — Intensity decay at 48 h (regression)
    _run_task("decay_48h", "wind_48h", is_clf=False)

    print("\nExplainability analysis complete.")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Rankings saved to: {FEAT_DIR}")


if __name__ == "__main__":
    run()
