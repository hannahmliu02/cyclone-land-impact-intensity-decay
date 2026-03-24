"""One-off script to generate figures/ablation_spatial_modality.png"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
FEAT_DIR = os.path.join(ROOT, "data", "features")
FIG_DIR  = os.path.join(ROOT, "figures")

df = pd.read_csv(os.path.join(FEAT_DIR, "ablation_spatial_modality.csv"))

tasks        = df["task"].tolist()
tab_losses   = df["val_loss_tabular"].tolist()
spa_losses   = df["val_loss_spatial"].tolist()
deltas       = df["delta"].tolist()
pcts         = df["improvement_pct"].tolist()

TASK_LABELS  = {"landfall": "Landfall Timing", "decay": "Intensity Decay"}
TAB_COLOR    = "#4C72B0"
SPA_COLOR    = "#DD8452"
EPOCHS       = 20

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f"Spatial Modality Ablation  —  UFNO trained for {EPOCHS} epochs\n"
    "Does adding 3D spatial patches (925 hPa fields) improve validation loss?",
    fontsize=12, fontweight="bold", y=1.02,
)

# ── Panel 1: grouped bar — val loss per task ─────────────────────────────────
ax = axes[0]
x     = np.arange(len(tasks))
width = 0.35
bars_tab = ax.bar(x - width/2, tab_losses, width, label="Tabular-only",
                  color=TAB_COLOR, edgecolor="white", linewidth=0.8)
bars_spa = ax.bar(x + width/2, spa_losses, width, label="Tabular + Spatial",
                  color=SPA_COLOR, edgecolor="white", linewidth=0.8)

for bars in (bars_tab, bars_spa):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
ax.set_ylabel("Best Validation Loss (lower = better)", fontsize=9)
ax.set_title("Validation Loss by Mode", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, max(tab_losses + spa_losses) * 1.15)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# ── Panel 2: delta bar (positive = spatial better) ───────────────────────────
ax = axes[1]
colors = [SPA_COLOR if d > 0 else "#C44E52" for d in deltas]
bars = ax.bar([TASK_LABELS.get(t, t) for t in tasks], deltas,
              color=colors, edgecolor="white", linewidth=0.8, width=0.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

for b, d in zip(bars, deltas):
    va   = "bottom" if d >= 0 else "top"
    yoff = 0.0005 if d >= 0 else -0.0005
    ax.text(b.get_x() + b.get_width()/2, d + yoff,
            f"{d:+.4f}", ha="center", va=va, fontsize=9, fontweight="bold")

ax.set_ylabel("Δ Val Loss  (tabular − spatial)\nPositive = spatial helps", fontsize=9)
ax.set_title("Improvement from Spatial Patches", fontsize=10, fontweight="bold")
pos_patch = mpatches.Patch(color=SPA_COLOR, label="Spatial helps (↓ loss)")
neg_patch = mpatches.Patch(color="#C44E52", label="Spatial hurts (↑ loss)")
ax.legend(handles=[pos_patch, neg_patch], fontsize=8)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# ── Panel 3: % improvement ────────────────────────────────────────────────────
ax = axes[2]
colors = [SPA_COLOR if p > 0 else "#C44E52" for p in pcts]
bars = ax.bar([TASK_LABELS.get(t, t) for t in tasks], pcts,
              color=colors, edgecolor="white", linewidth=0.8, width=0.5)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

for b, p in zip(bars, pcts):
    va   = "bottom" if p >= 0 else "top"
    yoff = 0.05 if p >= 0 else -0.05
    ax.text(b.get_x() + b.get_width()/2, p + yoff,
            f"{p:+.1f}%", ha="center", va=va, fontsize=10, fontweight="bold")

ax.set_ylabel("% Improvement  (positive = spatial better)", fontsize=9)
ax.set_title("Relative Improvement (%)", fontsize=10, fontweight="bold")
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# ── Annotation box ────────────────────────────────────────────────────────────
note = (
    "Note: Spatial patches (8×4×81×81) add 925 hPa u/v-wind,\n"
    "geopotential, and SST fields. Tabular features include\n"
    f"env, wind, pressure, wp_couple, spatial scalars (~340 features).\n"
    f"Training: {EPOCHS} epochs, AdamW lr=3e-4, batch=32, early-stop=10."
)
fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="grey",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgrey"))

fig.tight_layout()
out = os.path.join(FIG_DIR, "ablation_spatial_modality.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
