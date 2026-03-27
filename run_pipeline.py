"""
TCND Experiment Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━
Changes from run_pipeline.py:
  - 20x20 centre crop on spatial patches (spatial-sigma disabled)
  - Improved landfall detection via land-sea mask (features_improved.py)
  - env argmax encoding, future key stripping, pres_delta_from_landfall
  - Reduced model capacity (modes=6/8, width=12/16) + early-stop=7
  - No cross-basin, no model size sweep, no explain.py
"""

import subprocess
import sys
import os


def run_step(script_name, args=None, allow_failure=False):
    """Helper to run a sub-script and check for errors."""
    cmd = [sys.executable, os.path.join("scripts", script_name)]
    if args:
        cmd.extend(args)

    print(f"\nSTARTING: {script_name}")
    print(f"   cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        if allow_failure:
            print(f"WARNING: {script_name} exited with errors but pipeline continues.")
        else:
            print(f"ERROR: {script_name} failed. Halting pipeline.")
            sys.exit(1)
    else:
        print(f"COMPLETED: {script_name}")


def main():

    # --- PHASE 1: DATA ACQUISITION ---
    run_step("download_data.py", allow_failure=True)

    # --- PHASE 2: LAND GRID PRECOMPUTATION (one-time, ~2-3 min) ---
    proc_dir = os.path.join("data", "processed")
    grids_exist = (
        os.path.exists(os.path.join(proc_dir, "dist_to_coast_025.npy")) and
        os.path.exists(os.path.join(proc_dir, "dist_inland_025.npy"))
    )
    if grids_exist:
        print("\n SKIPPING: build_land_grids.py (grids already present)")
    else:
        run_step("build_land_grids.py")

    # --- PHASE 3: FEATURES (FIRST PASS) ---
    # features_improved.py includes:
    #   - Real land-sea mask via global-land-mask + EDT grids
    #   - env argmax encoding (~26 features vs ~300)
    #   - future_* keys stripped (no label leakage)
    #   - pres_delta_from_landfall for decay task
    #   - Improved landfall detection via coastline crossing
    run_step("features_improved.py")

    # --- PHASE 4: SPATIAL PREPROCESSING ---
    # Extracts 925 hPa patches (8, 4, 81, 81) to data/processed/3d/
    # and data/processed/3d_landfall/.
    # Note: patches are saved at full 81x81 resolution.
    # The 20x20 centre crop is applied at training time in the dataset
    # __getitem__ methods in train_ufno.py.
    run_step("preprocess.py", ["--task", "all"])

    # --- PHASE 5: FEATURES (SECOND PASS) ---
    # Re-run now patches exist to add sp_* scalar summaries.
    run_step("features_improved.py")

    # --- PHASE 6: FEATURE SELECTION ---
    run_step("ablation.py", ["--task", "all"])

    # --- PHASE 7: TRAINING ---

    # Stage 1: Landfall timing model.
    # spatial-sigma=0 — Gaussian mask disabled, using 20x20 crop instead.
    # modes=6, width=12 — small capacity appropriate for landfall task.
    run_step("train_ufno.py", [
        "--task",          "landfall",
        "--epochs",        "100",
        "--lr",            "3e-4",
        "--batch",         "32",
        "--modes",         "6",
        "--width",         "12",
        "--early-stop",    "7",
        "--spatial-sigma", "0",
    ])

    # Stage 2: Intensity decay model with landfall embedding via FiLM.
    # spatial-sigma=0 — Gaussian mask disabled, using 20x20 crop instead.
    # modes=8, width=16 — slightly larger capacity for harder regression task.
    run_step("train_ufno.py", [
        "--task",          "decay",
        "--epochs",        "150",
        "--lr",            "3e-4",
        "--batch",         "32",
        "--modes",         "8",
        "--width",         "16",
        "--early-stop",    "7",
        "--spatial-sigma", "0",
        "--landfall-ckpt", "models/best_ufno_landfall.pt",
    ])

    # --- PHASE 8: EVALUATION ---
    run_step("view_results.py", ["--task", "landfall"])
    run_step("view_results.py", ["--task", "decay"])

    # --- PHASE 9: EXPERIMENT LOGGING ---
    run_step("log_experiment.py", [
        "--name",  "v8: 20x20 crop, land-sea fix, delta-p, env argmax",
        "--notes", (
            "Spatial: 20x20 centre crop on 925 hPa patches (Gaussian mask disabled). "
            "Features: real land-sea mask (global-land-mask + EDT grids, signed dist). "
            "Features: pres_delta_from_landfall anchored to peak intensity. "
            "Features: env argmax encoding (~26 features, future keys stripped). "
            "Features: improved landfall detection via coastline crossing. "
            "Model: modes=6/8, width=12/16, early-stop=7, spatial-sigma=0."
        ),
    ])

    print("\nPIPELINE COMPLETE. Check 'figures/ufno_results/' for dashboards.")


if __name__ == "__main__":
    main()