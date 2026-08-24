"""
run_phase7.py
--------------
Runs Phase 7 (Module F: population resource-constrained optimizer) end
to end. Explicitly out of this package's own build scope per the plan
("should-build after the gate clears") -- built anyway on request, with
the same gate-status honesty as Phase 6.

Steps:
  1. Gate check against outputs/phase5_validation_gate_report.json --
     same warning-not-hard-stop pattern as run_phase6.py.
  2. Solver check: is `pulp` importable? If not (this dev sandbox's
     situation -- no network), src/resource_optimizer.py's DP knapsack
     fallback is used automatically and the report says so explicitly.
  3. Compute per-patient expected diagnostic yield on the real v2 test
     split (src/resource_optimizer.py:compute_population_yield_surrogate)
     and solve the allocation at several capacity levels
     (src/resource_optimizer.py:capacity_sweep) -- the static equivalent
     of the plan's "live capacity-slider demo."

Usage:
    python run_phase7.py
    python run_phase7.py --data_dir data/raw_v2 --capacities 20 50 100 200
"""

import argparse
import json
import os
import subprocess
import sys


def run(cmd, allow_failure=False):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0 and not allow_failure:
        sys.exit(result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Phase 7 (population resource-constrained optimizer)")
    parser.add_argument("--data_dir", type=str, default="data/raw_v2")
    parser.add_argument("--capacities", type=float, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--gate_report", type=str, default="outputs/phase5_validation_gate_report.json")
    parser.add_argument("--output", type=str, default="outputs/phase7_resource_allocation_report.json")
    args = parser.parse_args()

    # ---- Gate check (the plan: Phase 7 "should-build after the gate clears") ----
    gate_passed = None
    if os.path.exists(args.gate_report):
        with open(args.gate_report) as f:
            gate_passed = json.load(f).get("all_passed")

    if gate_passed is not True:
        print("\n" + "!" * 70)
        print("WARNING: Phase 5's validation gate has not passed "
              f"(outputs/{os.path.basename(args.gate_report)} shows all_passed={gate_passed}).")
        print("Per the plan, Phase 6/7 should not cite numbers as validated until it does.")
        print("Proceeding anyway -- this is methodology development for the optimizer")
        print("itself (the knapsack solver's correctness doesn't depend on Module A/B/C's")
        print("numbers being final), not a claim that the underlying yield estimates are.")
        print("!" * 70)

    # ---- Solver check (best-effort, informational) ----
    pulp_rc = run([sys.executable, "-c", "import pulp; print('pulp available')"], allow_failure=True)
    if pulp_rc != 0:
        print("\npulp not available in this environment -- the exact dynamic-programming "
              "knapsack fallback in src/resource_optimizer.py will be used instead "
              "(proven exact against brute-force enumeration in its sanity_check(); "
              "same guarantee, no MILP library dependency).")

    # ---- Sanity checks first (cheap, catches a broken solver/yield computation early) ----
    run([sys.executable, "src/resource_optimizer.py", "--sanity_check", "--data_dir", args.data_dir])

    # ---- Full report ----
    run([
        sys.executable, "src/resource_optimizer.py",
        "--data_dir", args.data_dir,
        "--capacities", *[str(c) for c in args.capacities],
        "--output", args.output,
    ])

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE")
    print("=" * 70)
    print(f"Report -> {args.output}")
    if gate_passed is not True:
        print("\nReminder: Phase 5's validation gate has NOT passed -- treat this phase's")
        print("output as methodology development, not a validated result, until it has.")


if __name__ == "__main__":
    main()
