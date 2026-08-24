"""
run_phase3.py
-------------
Runs Phase 3 end to end: trains the PPO adaptive pathway agent (Module C)
on top of the frozen Phase 1 fusion model (Module A), then evaluates it
against three baselines (always-full-workup, cognitive-only, and the
plan's own greedy-confidence fallback heuristic) on the held-out test set.

Usage:
    python run_phase3.py
    python run_phase3.py --total_timesteps 100000
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
    parser = argparse.ArgumentParser(description="Run full Phase 3 pipeline")
    parser.add_argument("--total_timesteps", type=int, default=60000)
    parser.add_argument("--lambda_cost", type=float, default=0.05)
    parser.add_argument("--mu_invasive", type=float, default=0.05)
    parser.add_argument("--confidence_threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run([sys.executable, "src/train_pathway.py",
         "--total_timesteps", str(args.total_timesteps),
         "--lambda_cost", str(args.lambda_cost), "--mu_invasive", str(args.mu_invasive),
         "--seed", str(args.seed)])
    run([sys.executable, "src/evaluate_pathway.py",
         "--lambda_cost", str(args.lambda_cost), "--mu_invasive", str(args.mu_invasive),
         "--confidence_threshold", str(args.confidence_threshold)])

    print("\n=== Phase 3 complete ===")
    print("Outputs written to ./outputs/phase3_comparison_report.json")
    print("Checkpoint -> ./checkpoints/pathway_ppo.zip")


if __name__ == "__main__":
    main()
