"""
Regenerate figures/cross_basin_distribution_shift.png

Shows the top 20 features by cross-basin distribution shift (max pairwise
mean difference normalised by pooled σ), then plots the actual per-basin
KDE distributions for the top 4 features.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BASINS = ["WP", "NA", "EP"]
COLORS = {"WP": "#1f78b4", "NA": "#33a02c", "EP": "#e31a1c"}

# Columns that are labels/metadata, not features
SKIP_COLS = {
    "storm_id", "basin", "split", "ref_time", "made_landfall",
    "hours_to_landfall", "wind_24h", "wind_48h", "wind_frac_24h",
    "wind_frac_48h", "landfall_wind", "landfall_pres",
}


TASK = "landfall"   # "landfall" or "decay"

def load_data():
    fname = f"feature_matrix_{TASK}.csv"
    path = os.path.join(FEAT_DIR, fname)
    if not os.path.exists(path):
        sys.exit("Run features.py first.")
    df = pd.read_csv(path, keep_default_na=False, low_memory=False)
    df = df[df["basin"].isin(BASINS)].reset_index(drop=True)
    return df


def compute_shift(df):
    """
    For each numeric feature, compute max pairwise normalised mean difference.
    Skips features where pooled std is near-zero (constant or degenerate).
    """
    feat_cols = [c for c in df.columns if c not in SKIP_COLS]
    results = []

    for col in feat_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().mean() > 0.5:
            continue

        basin_stats = {}
        skip = False
        for b in BASINS:
            v = vals[df["basin"] == b].dropna()
            if len(v) < 5:
                skip = True
                break
            basin_stats[b] = (v.mean(), v.std(ddof=1), len(v))
        if skip:
            continue

        # Pooled standard deviation
        total_n = sum(s[2] for s in basin_stats.values())
        pooled_var = (
            sum((s[2] - 1) * s[1] ** 2 for s in basin_stats.values())
            / (total_n - len(BASINS))
        )
        pooled_std = np.sqrt(pooled_var) if pooled_var > 1e-8 else None
        if pooled_std is None:
            continue  # constant feature — skip

        means = [basin_stats[b][0] for b in BASINS]
        max_diff = max(
            abs(means[i] - means[j])
            for i in range(len(BASINS))
            for j in range(i + 1, len(BASINS))
        )
        score = max_diff / pooled_std

        # Classify feature
        if col.startswith("sp_"):
            category = "Spatial scalar (sp_*)"
            color = "#e63946"
        elif col.startswith("env_"):
            category = "Env-Data feature"
            color = "#f4a261"
        else:
            category = "Physical / tabular"
            color = "#457b9d"

        results.append({
            "feature": col,
            "score": score,
            "category": category,
            "color": color,
        })

    results.sort(key=lambda x: -x["score"])
    return results


def plot_kde(ax, vals_by_basin, feature_name):
    """Overlay KDE curves per basin to show distribution shift.
    Falls back to histogram if the feature has near-zero variance."""
    use_hist = False
    for b in BASINS:
        v = vals_by_basin[b].dropna().values
        if len(v) >= 5 and v.std() < 1e-6:
            use_hist = True
            break

    all_vals = np.concatenate([vals_by_basin[b].dropna().values for b in BASINS])
    xmin, xmax = all_vals.min(), all_vals.max()
    pad = (xmax - xmin) * 0.15 or 1.0
    bins = np.linspace(xmin - pad, xmax + pad, 40)

    for b in BASINS:
        v = vals_by_basin[b].dropna().values
        if len(v) < 5:
            continue
        if use_hist:
            ax.hist(v, bins=bins, density=True, alpha=0.5,
                    color=COLORS[b], label=f"{b} (n={len(v)})")
        else:
            try:
                kde = gaussian_kde(v, bw_method="scott")
                xs = np.linspace(xmin - pad, xmax + pad, 300)
                ax.plot(xs, kde(xs), color=COLORS[b], lw=2, label=f"{b} (n={len(v)})")
                ax.fill_between(xs, kde(xs), alpha=0.15, color=COLORS[b])
            except Exception:
                ax.hist(v, bins=bins, density=True, alpha=0.5,
                        color=COLORS[b], label=f"{b} (n={len(v)})")
        ax.axvline(v.mean(), color=COLORS[b], lw=1, linestyle="--", alpha=0.7)

    ax.set_title(feature_name, fontsize=9, fontweight="bold")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    df = load_data()
    print(f"Loaded {len(df)} decay samples across {BASINS}")

    results = compute_shift(df)
    print(f"Scored {len(results)} features")
    print("Top 10:")
    for r in results[:10]:
        print(f"  {r['feature']}: {r['score']:.3f}  [{r['category']}]")

    top20 = results[:20]
    top3  = results[:3]

    # ── Layout: bar chart on top, 3 KDE plots below ───────────────────────────
    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.55, wspace=0.4)

    ax_bar = fig.add_subplot(gs[0, :])   # full-width bar chart
    axs    = [fig.add_subplot(gs[1, i]) for i in range(3)]   # 3 KDE plots

    # ── Bar chart ─────────────────────────────────────────────────────────────
    names  = [r["feature"] for r in top20]
    scores = [r["score"]   for r in top20]
    colors = [r["color"]   for r in top20]

    y_pos = np.arange(len(names))
    ax_bar.barh(y_pos, scores, color=colors, edgecolor="white", height=0.7)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Max pairwise mean difference (normalised by pooled σ)", fontsize=10)
    ax_bar.set_title(
        f"Cross-Basin Distribution Shift Analysis — {TASK.capitalize()} Task\n"
        "Top 20 Features Causing Large RMSE When Transferring Between Basins",
        fontsize=12, fontweight="bold",
    )
    ax_bar.grid(axis="x", alpha=0.3)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#457b9d", label="Physical / tabular"),
        Patch(facecolor="#e63946", label="Spatial scalar (sp_*)"),
        Patch(facecolor="#f4a261", label="Env-Data feature"),
    ]
    ax_bar.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Annotate scores on bars
    for i, (s, r) in enumerate(zip(scores, top20)):
        ax_bar.text(s + 0.02, i, f"{s:.2f}", va="center", fontsize=8)

    # ── KDE plots for top 3 ───────────────────────────────────────────────────
    feat_vals = {
        col: pd.to_numeric(df[col], errors="coerce")
        for col in [r["feature"] for r in top3]
    }

    for ax, r in zip(axs, top3):
        col = r["feature"]
        vals_by_basin = {b: feat_vals[col][df["basin"] == b] for b in BASINS}
        score_str = f"shift = {r['score']:.2f}σ"
        label = f"{col}\n({score_str}, {r['category']})"
        plot_kde(ax, vals_by_basin, label)

    fig.suptitle(
        "Per-Basin Feature Distributions for Top 3 Shifted Features",
        y=0.48, fontsize=11, fontweight="bold",
    )

    out = os.path.join(FIG_DIR, "cross_basin_distribution_shift.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
