"""
Generate figures/ablation_combined.png

4-panel figure showing feature group ablation results across all metrics:
  - Decay 24h RMSE
  - Decay 48h RMSE
  - Landfall AUC
  - Landfall RMSE (hours to landfall)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Only show leave_out and full rows — skip "only_*" rows
KEEP_PREFIX = ("full", "leave_out_")

GROUP_LABELS = {
    "full":              "All features",
    "leave_out_wind":    "− Wind",
    "leave_out_pressure":"− Pressure",
    "leave_out_wp_couple":"− WP couple",
    "leave_out_position":"− Position",
    "leave_out_spatial": "− Spatial",
    "leave_out_env":     "− Env-Data",
    "leave_out_land_sea":"− Land/sea",
}

# Color: red if removing this group hurts, grey for full baseline
def bar_colors(labels, baseline_val, values, higher_is_better=False):
    colors = []
    for lbl, val in zip(labels, values):
        if lbl == "full":
            colors.append("#457b9d")
        else:
            worse = val > baseline_val if not higher_is_better else val < baseline_val
            colors.append("#e63946" if worse else "#2a9d8f")
    return colors


def load_and_filter(fname):
    path = os.path.join(FEAT_DIR, fname)
    df = pd.read_csv(path)
    df = df[df["label"].apply(lambda x: any(x.startswith(p) for p in KEEP_PREFIX))]
    df["display"] = df["label"].map(GROUP_LABELS).fillna(df["label"])
    return df


def plot_panel(ax, df, metric_col, err_col, title, ylabel, higher_is_better=False):
    baseline = df.loc[df["label"] == "full", metric_col].values[0]

    # Plot delta from baseline (exclude the full row)
    leave_out = df[df["label"] != "full"].copy()
    # Delta: positive = removing hurt (for RMSE), or flipped for R²
    if higher_is_better:
        leave_out["delta"] = baseline - leave_out[metric_col]  # positive = hurts (R² dropped)
    else:
        leave_out["delta"] = leave_out[metric_col] - baseline  # positive = hurts (RMSE rose)

    # Green = KEEP (delta > 0, removing hurts), Red = DROP (delta <= 0)
    colors = ["#2a9d8f" if d > 0 else "#e63946" for d in leave_out["delta"]]

    y = np.arange(len(leave_out))
    ax.barh(y, leave_out["delta"], color=colors, edgecolor="white", height=0.65)

    ax.axvline(0, color="#457b9d", lw=1.2, linestyle="--", alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(leave_out["display"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ {ylabel} vs. all-features baseline\n(positive = removing this group hurts)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    x_range = leave_out["delta"].abs().max() or 0.01
    pad = x_range * 0.05
    for i, (delta, color) in enumerate(zip(leave_out["delta"], colors)):
        if delta >= 0:
            x_pos = delta + pad
            ha = "left"
        else:
            x_pos = delta - pad
            ha = "right"
        ax.text(x_pos, i, f"{delta:+.4g}",
                va="center", ha=ha, fontsize=8, fontweight="bold", color=color)

    # Expand x-axis so text labels have room and don't clip
    x_min = leave_out["delta"].min()
    x_max = leave_out["delta"].max()
    ax.set_xlim(x_min - x_range * 0.35, x_max + x_range * 0.35)


def main():
    decay_24 = load_and_filter("ablation_decay_24h.csv")
    decay_48 = load_and_filter("ablation_decay_48h.csv")
    landfall = load_and_filter("ablation_landfall.csv")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Feature Group Ablation — All Tasks & Metrics",
                 fontsize=14, fontweight="bold", y=0.98)

    plot_panel(axes[0, 0], decay_24, "rmse_mean", "rmse_std",
               "Decay — 24h RMSE", "RMSE (normalised wind)")

    plot_panel(axes[0, 1], decay_48, "rmse_mean", "rmse_std",
               "Decay — 48h RMSE", "RMSE (normalised wind)")

    plot_panel(axes[1, 0], landfall, "r2_mean", "r2_std",
               "Landfall — R²", "R² (higher = better)",
               higher_is_better=True)

    plot_panel(axes[1, 1], landfall, "rmse_mean", "rmse_std",
               "Landfall — Timing RMSE", "RMSE (hours to landfall)")

    # Shared legend
    legend_elements = [
        mpatches.Patch(color="#457b9d", label="Baseline (all features)"),
        mpatches.Patch(color="#2a9d8f", label="KEEP — removing this group hurts performance"),
        mpatches.Patch(color="#e63946", label="DROP — removing this group helps or is neutral"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = os.path.join(FIG_DIR, "ablation_combined.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
