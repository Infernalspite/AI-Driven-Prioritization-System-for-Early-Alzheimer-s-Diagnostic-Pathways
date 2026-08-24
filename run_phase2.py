"""
run_phase2.py
-------------
Runs Phase 2 end to end: trains TrajectoryLSTM on the subject-level visit
sequences, then evaluates on the held-out test split (next-visit
diagnosis, MCI->AD conversion vs. a cross-sectional RF baseline, CDR-SB
slope regression, and a few time-to-threshold projection examples with MC
Dropout uncertainty).

Usage:
    python run_phase2.py
    python run_phase2.py --epochs 80 --hidden_dim 96
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
    parser = argparse.ArgumentParser(description="Run full Phase 2 pipeline")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--slope_loss_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run([sys.executable, "src/train_trajectory.py",
         "--epochs", str(args.epochs), "--lr", str(args.lr), "--batch_size", str(args.batch_size),
         "--embed_dim", str(args.embed_dim), "--hidden_dim", str(args.hidden_dim),
         "--n_layers", str(args.n_layers), "--slope_loss_weight", str(args.slope_loss_weight),
         "--seed", str(args.seed)])
    run([sys.executable, "src/evaluate_trajectory.py"])

    print("\n=== Phase 2 complete ===")
    print("Outputs written to ./outputs/ :")
    print("  - trajectory_train_history.json")
    print("  - phase2_comparison_report.json  (next-visit metrics, MCI->AD conversion vs RF baseline,")
    print("                                      CDR-SB slope regression, time-to-threshold projections)")
    print("Checkpoint -> ./checkpoints/trajectory_best.pt")


if __name__ == "__main__":
    main()
