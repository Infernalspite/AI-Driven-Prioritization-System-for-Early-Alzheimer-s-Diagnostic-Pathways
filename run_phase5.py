"""
run_phase5.py
--------------
Orchestrates Phase 5 end to end on the HARDENED SYNTHETIC v2 path (5B/5C),
then runs the validation gate (5D). This is the "don't block on ADNI/OASIS
access clearing" track described in the plan -- for the real-data
migration track (5A), run src/migrate_oasis.py separately once you have
network access and a Kaggle token (see that script's docstring), then
rerun this same sequence with --data_dir data/raw_oasis and
--checkpoint_dir checkpoints_oasis (OASIS-migrated data needs its own
checkpoints: the fusion model degrades to 2-modality, and Module C's
action space drops PET -- both are DIFFERENT model shapes than the
default 4-modality / 3-escalation-action checkpoints, so they cannot
share checkpoint files with the v1/v2 synthetic runs).

Steps:
  1. Generate hardened synthetic v2 data (src/generate_synthetic_v2.py).
  2. Retrain Module A (baseline + fusion) on v2 data.
  3. Retrain Module B (trajectory LSTM) on v2 data, with focal loss +
     subject-level oversampling active (5C) -- this is the module most
     starved for converter examples, so it's the one most worth turning
     these on for by default; pass --skip_focal_loss / --skip_oversampling
     to compare against plain weighted CE if you want that ablation.
  4. Retrain Module C (PPO pathway agent) on top of the freshly retrained
     v2 fusion checkpoint.
  5. Rerun evaluate.py / evaluate_trajectory.py / evaluate_pathway.py
     against v2 data and the newly-trained checkpoints.
  6. Run the validation gate (src/validation_gate.py) against those fresh
     reports and print/exit accordingly.

Usage:
    python run_phase5.py
    python run_phase5.py --n_subjects 800 --total_timesteps 100000
    python run_phase5.py --skip_generation   # reuse existing data/raw_v2
"""

import argparse
import subprocess
import sys


def run(cmd, allow_failure=False):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0 and not allow_failure:
        sys.exit(result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 (hardened synthetic v2 track) end to end")
    parser.add_argument("--n_subjects", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/raw_v2")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v2")
    parser.add_argument("--epochs_module_a", type=int, default=40)
    parser.add_argument("--epochs_module_b", type=int, default=60)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--total_timesteps_module_c", type=int, default=60000)
    parser.add_argument("--oversample_factor", type=int, default=4)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--skip_generation", action="store_true",
                         help="Reuse existing data at --data_dir instead of regenerating.")
    parser.add_argument("--skip_focal_loss", action="store_true",
                         help="Ablation: train Module B with plain weighted CE instead of focal loss.")
    parser.add_argument("--skip_oversampling", action="store_true",
                         help="Ablation: train Module B without subject-level oversampling.")
    args = parser.parse_args()

    import os
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- 1. Data ----
    if not args.skip_generation:
        run([sys.executable, "src/generate_synthetic_v2.py",
             "--n_subjects", str(args.n_subjects), "--seed", str(args.seed),
             "--out_dir", args.data_dir])
    else:
        print(f"\nSkipping generation, reusing existing data at {args.data_dir}/")

    # ---- 2. Module A: baseline + fusion ----
    common_a = ["--epochs", str(args.epochs_module_a), "--seed", str(args.seed),
                "--data_dir", args.data_dir, "--checkpoint_dir", args.checkpoint_dir]
    run([sys.executable, "src/train.py", "--model_type", "baseline"] + common_a)
    run([sys.executable, "src/train.py", "--model_type", "fusion",
         "--embed_dim", str(args.embed_dim)] + common_a)
    run([sys.executable, "src/evaluate.py",
         "--data_dir", args.data_dir, "--checkpoint_dir", args.checkpoint_dir,
         "--output", "outputs/phase1_comparison_report_v2.json"])

    # ---- 3. Module B: trajectory LSTM, with 5C improvements on by default ----
    module_b_cmd = [sys.executable, "src/train_trajectory.py",
                     "--epochs", str(args.epochs_module_b), "--seed", str(args.seed),
                     "--data_dir", args.data_dir, "--checkpoint_dir", args.checkpoint_dir]
    if not args.skip_focal_loss:
        module_b_cmd += ["--use_focal_loss", "--focal_gamma", str(args.focal_gamma)]
    if not args.skip_oversampling:
        module_b_cmd += ["--use_oversampling", "--oversample_factor", str(args.oversample_factor)]
    run(module_b_cmd)
    run([sys.executable, "src/evaluate_trajectory.py",
         "--data_dir", args.data_dir, "--checkpoint", f"{args.checkpoint_dir}/trajectory_best.pt",
         "--output", "outputs/phase2_comparison_report_v2.json"])

    # ---- 3.5. Phase 5C reference check: SMOTE vs subject-level oversampling ----
    # Not part of the validation gate (5D) and not blocking -- this is the
    # plan's own "reference/comparison" line for imbalanced-learn/SMOTE,
    # run here so the comparison numbers exist alongside the rest of
    # Phase 5's output rather than only in prose. Uses scikit-learn only
    # (see src/smote_baseline.py); no torch dependency, so it also runs
    # fine even in a torch-less environment.
    run([sys.executable, "src/smote_baseline.py",
         "--csv_path", f"{args.data_dir}/cleaned_dataset.csv",
         "--oversample_factor", str(args.oversample_factor),
         "--output", "outputs/phase5_smote_comparison_report.json"], allow_failure=True)

    # ---- 4. Module C: PPO pathway agent, on top of the FRESH v2 fusion checkpoint ----
    run([sys.executable, "src/train_pathway.py",
         "--total_timesteps", str(args.total_timesteps_module_c), "--seed", str(args.seed),
         "--data_dir", args.data_dir, "--checkpoint_dir", args.checkpoint_dir,
         "--fusion_checkpoint", f"{args.checkpoint_dir}/fusion_best.pt"])
    run([sys.executable, "src/evaluate_pathway.py",
         "--data_dir", args.data_dir,
         "--fusion_checkpoint", f"{args.checkpoint_dir}/fusion_best.pt",
         "--ppo_checkpoint", f"{args.checkpoint_dir}/pathway_ppo.zip"])
    # evaluate_pathway.py writes to a fixed outputs/phase3_comparison_report.json path
    # (unlike evaluate.py/evaluate_trajectory.py); copy it alongside the others so all
    # three v2 reports live together for the gate and for later comparison against v1.
    import shutil
    shutil.copy("outputs/phase3_comparison_report.json", "outputs/phase3_comparison_report_v2.json")

    # ---- 5. Validation gate ----
    gate_exit_code = run([
        sys.executable, "src/validation_gate.py",
        "--phase1_report", "outputs/phase1_comparison_report_v2.json",
        "--phase2_report", "outputs/phase2_comparison_report_v2.json",
        "--phase3_report", "outputs/phase3_comparison_report_v2.json",
        "--output", "outputs/phase5_validation_gate_report.json",
    ], allow_failure=True)

    print("\n" + "=" * 70)
    print("PHASE 5 (hardened synthetic v2 track) COMPLETE")
    print("=" * 70)
    print(f"Data       -> {args.data_dir}/ (+ ground_truth.csv, generator_config.json)")
    print(f"Checkpoints -> {args.checkpoint_dir}/")
    print("Reports    -> outputs/phase{1,2,3}_comparison_report_v2.json")
    print("             outputs/phase5_validation_gate_report.json")
    print("             outputs/phase5_smote_comparison_report.json (5C reference check)")
    if gate_exit_code == 0:
        print("\nGate PASSED on v2 data -- Phase 6/7 may proceed citing these numbers.")
    else:
        print("\nGate did NOT pass on v2 data -- see outputs/phase5_validation_gate_report.json ")
        print("for which criterion/criteria failed, and report that honestly rather than ")
        print("proceeding to Phase 6/7 on unvalidated numbers.")
    print("\nFor the real-data (OASIS) migration track:")
    print("  python src/migrate_oasis.py --input_csv path/to/oasis_longitudinal.csv")
    print("  # then rerun this script with --data_dir data/raw_oasis --checkpoint_dir checkpoints_oasis")
    print("  # (and see migrate_oasis.py's migration_report.json for the PET-action-space caveat)")

    sys.exit(gate_exit_code)


if __name__ == "__main__":
    main()
