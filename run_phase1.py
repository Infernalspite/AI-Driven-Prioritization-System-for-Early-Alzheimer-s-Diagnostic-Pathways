"""
run_phase1.py
-------------
Runs Phase 1 end to end: trains the ConcatBaselineMLP, trains the
CrossAttentionFusion model, then evaluates both on the held-out test set
and writes a head-to-head comparison report.

Usage:
    python run_phase1.py
    python run_phase1.py --epochs 60 --embed_dim 48
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
    parser = argparse.ArgumentParser(description="Run full Phase 1 pipeline")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    common = ["--epochs", str(args.epochs), "--lr", str(args.lr),
              "--batch_size", str(args.batch_size), "--seed", str(args.seed)]

    run([sys.executable, "src/train.py", "--model_type", "baseline"] + common)
    run([sys.executable, "src/train.py", "--model_type", "fusion",
         "--embed_dim", str(args.embed_dim)] + common)
    run([sys.executable, "src/evaluate.py"])
    run([sys.executable, "src/inspect_attention.py"])

    print("\n=== Phase 1 complete ===")
    print("Outputs written to ./outputs/ :")
    print("  - baseline_train_history.json / fusion_train_history.json")
    print("  - phase1_comparison_report.json")
    print("  - attention_examples.json")
    print("Checkpoints -> ./checkpoints/{baseline,fusion}_best.pt")


if __name__ == "__main__":
    main()
