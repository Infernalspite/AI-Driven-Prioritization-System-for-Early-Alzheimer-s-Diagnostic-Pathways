"""
run_phase4.py
-------------
Runs Phase 4 end to end: (1) MC-Dropout/calibration evaluation of the
frozen Phase 1 fusion model (no retraining needed -- dropout was already
there), and (2) the Phase 4 -> Phase 3 integration, training a second PPO
pathway agent whose state includes a live epistemic-uncertainty estimate,
then comparing it against the original Phase 3 agent.

Usage:
    python run_phase4.py
    python run_phase4.py --skip_uncertainty_agent   # just the calibration report
"""

import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full Phase 4 pipeline")
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--lambda_cost", type=float, default=0.5)
    parser.add_argument("--mu_invasive", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_uncertainty_agent", action="store_true",
                         help="Only run the calibration/uncertainty report, skip retraining the pathway agent.")
    args = parser.parse_args()

    run([sys.executable, "src/evaluate_uncertainty.py"])
    run([sys.executable, "src/evaluate_uncertainty_trajectory.py"])

    if not args.skip_uncertainty_agent:
        run([sys.executable, "src/train_pathway.py",
             "--total_timesteps", str(args.total_timesteps),
             "--lambda_cost", str(args.lambda_cost), "--mu_invasive", str(args.mu_invasive),
             "--seed", str(args.seed), "--uncertainty_aware",
             "--output_name", "pathway_ppo_uncertainty"])
        run([sys.executable, "src/evaluate_pathway.py",
             "--lambda_cost", str(args.lambda_cost), "--mu_invasive", str(args.mu_invasive)])

    print("\n=== Phase 4 complete ===")
    print("Outputs written to ./outputs/phase4_uncertainty_report.json  (Module A)")
    print("               and ./outputs/phase4_uncertainty_trajectory_report.json  (Module B)")
    print("               and ./outputs/phase3_comparison_report.json (updated with the uncertainty-aware agent)")
    print("Checkpoint -> ./checkpoints/pathway_ppo_uncertainty.zip")


if __name__ == "__main__":
    main()
