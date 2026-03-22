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
    # --- PHASE 1: DATA & FEATURES ---
    
    # 1. Download raw TCND pillars (Data_1d, Data_3d, Env-Data)
    run_step("download_data.py")

    # 2. Extract engineered features (Deltas, Δp, Shear, SST)
    # This creates the CSV matrices used by all subsequent steps.
    run_step("features_improved.py")

    # 3. Perform Ablation Study
    # CRITICAL: This generates 'selected_feature_groups.json' which 
    # 'train_ufno.py' requires for dynamic feature loading.
    run_step("ablation.py")

    # 4. Generate interpretability rankings (SHAP/LIME)
    # Provides visual evidence of feature importance before deep learning.
    run_step("explain.py")

    # --- PHASE 2: DEEP LEARNING PREP & TRAINING ---

    # 5. Multimodal Preprocessing
    # Converts CSVs and .nc files into synchronized tensors (.npy).
    run_step("preprocess.py")

    # 6. Train Landfall Classification Model
    # We train this first to get the 'best_ufno_landfall.pt' checkpoint.
    #run_step("train_ufno.py", ["--task", "landfall", "--epochs", "50"])

    # 7. Train Intensity Decay Model with Landfall Embedding
    # Uses the landfall model as a frozen feature extractor via FiLM.
    # run_step("train_ufno.py", [
    #     "--task", "decay", 
    #     "--epochs", "60", 
    #     "--landfall-ckpt", "models/best_ufno_landfall.pt"
    # ])

    # --- PHASE 3: EVALUATION & GENERALIZATION ---

    # 8. Generate Results Dashboard
    # Produces scatter plots, decay curves, and MAE maps.
    run_step("view_results.py")

    # 9. Cross-Basin Generalization Experiment
    # Tests if the U-FNO model can generalize across WP, NA, and EP.
    #run_step("cross_basin.py", ["--epochs", "30"])

    # 10. Log Experiment
    # Records hyperparameters and RMSE to experiments/ for versioning.
    run_step("log_experiment.py", [
        "--name", "Full Pipeline Integration Run",
        "--notes", "Standardized run of all modules with FiLM landfall injection."
    ])

    print("\n🎉 PIPELINE COMPLETE. Check 'figures/ufno_results/' for the dashboard.")

if __name__ == "__main__":
    main()