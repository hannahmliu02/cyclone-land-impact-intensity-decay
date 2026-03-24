"""
TCND Master Pipeline Orchestrator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automates the full workflow:
1. Data Acquisition
2. Feature Engineering & Selection
3. Multimodal Preprocessing
4. Model Training (Landfall -> Intensity)
5. Evaluation & Logging
"""

import subprocess
import sys
import os

def run_step(script_name, args=None):
    """Helper to run a sub-script and check for errors."""
    cmd = [sys.executable, os.path.join("scripts", script_name)]
    if args:
        cmd.extend(args)
    
    print(f"\n🚀 STARTING: {script_name}")
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed. Halting pipeline.")
        sys.exit(1)
    print(f"✅ COMPLETED: {script_name}")

def main():
    # --- PHASE 1: DATA ACQUISITION ---

    # 1. Download raw TCND pillars (Data_1d, Data_3d, Env-Data)
    run_step("download_data.py")

    # --- PHASE 2: FEATURES (FIRST PASS) ---

    # 2. First pass: build feature matrices without sp_* spatial scalars.
    #    preprocess.py needs feature_matrix_landfall.csv to know which
    #    (storm, ref_time) pairs to extract patches for.
    run_step("features.py")

    # --- PHASE 3: SPATIAL PREPROCESSING ---

    # 3. Extract 925 hPa patches from Data_3d NetCDF files.
    #    Saves (8, 4, 81, 81) tensors to data/processed/3d/ and
    #    data/processed/3d_landfall/. Requires feature_matrix_landfall.csv
    #    (produced above) to know the landfall ref_times.
    run_step("preprocess.py", ["--task", "all"])

    # --- PHASE 4: FEATURES (SECOND PASS) ---

    # 4. Second pass: re-run features now that patches exist.
    #    This adds sp_* scalar summaries (mean/std/max/p90/asymmetry per
    #    channel) derived from the processed 3D patches.
    run_step("features.py")

    # --- PHASE 5: FEATURE SELECTION ---

    # 5. XGBoost ablation — ranks feature groups by R², writes
    #    selected_feature_groups_{task}.json which train_ufno.py reads.
    run_step("ablation.py", ["--task", "all"])

    # --- PHASE 6: TRAINING ---

    # 6. Train landfall timing model first (produces landfall embedding).
    run_step("train_ufno.py", ["--task", "landfall", "--epochs", "100",
                               "--lr", "3e-4", "--batch", "32"])

    # 7. Train intensity decay model with landfall embedding via FiLM.
    run_step("train_ufno.py", ["--task", "decay", "--epochs", "150",
                               "--lr", "3e-4", "--batch", "32",
                               "--landfall-ckpt", "models/best_ufno_landfall.pt"])

    # --- PHASE 7: EVALUATION ---

    # 8. Generate results dashboards for both tasks.
    run_step("view_results.py", ["--task", "landfall"])
    run_step("view_results.py", ["--task", "decay"])

    # 9. Cross-basin generalization (optional — comment out if not needed).
    # run_step("cross_basin.py", ["--task", "landfall", "--epochs", "60"])
    # run_step("cross_basin.py", ["--task", "decay",    "--epochs", "60"])

    # 10. Feature importance (optional).
    # run_step("explain.py")

    # 11. Log experiment.
    run_step("log_experiment.py", [
        "--name", "Full Pipeline Run",
        "--notes", "Two-stage training: landfall timing then decay with FiLM embedding."
    ])

    print("\nPIPELINE COMPLETE. Check 'figures/ufno_results/' for dashboards.")

if __name__ == "__main__":
    main()