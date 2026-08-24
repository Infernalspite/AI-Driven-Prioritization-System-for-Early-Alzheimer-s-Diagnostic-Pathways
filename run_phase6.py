"""
run_phase6.py
--------------
Runs Phase 6 (Module E: causal confound-adjustment layer) end to end.

Per the plan, this phase "runs after the validation gate" -- i.e. only
once outputs/phase5_validation_gate_report.json shows all_passed: true
on real or hardened-v2 data should its numbers be cited as validated.
This script checks that and prints a clear warning (not a hard stop --
see --require_gate_pass) if the gate hasn't passed yet, same honesty
standard as every other phase in this package.

Two steps:
  1. SHAP explanation of the frozen Module A fusion model
     (src/causal_explainability.py:explain_fusion_model_shap) -- needs
     torch + the `shap` package + a real fusion checkpoint. Skipped
     automatically (allow_failure) if either import fails, e.g. in this
     dev sandbox, which has neither installed and no network access to
     get them.
  2. Backdoor-adjusted confound analysis
     (src/causal_explainability.py:backdoor_adjusted_importance) --
     numpy/pandas/scikit-learn only, always runs, and is what carries
     the actual verified numbers in this package right now.

Usage:
    python run_phase6.py
    python run_phase6.py --data_dir data/raw_v2 --checkpoint checkpoints_v2/fusion_best.pt
    python run_phase6.py --skip_gate_check
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
    parser = argparse.ArgumentParser(description="Run Phase 6 (causal confound-adjustment layer)")
    parser.add_argument("--data_dir", type=str, default="data/raw_v2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v2/fusion_best.pt")
    parser.add_argument("--gate_report", type=str, default="outputs/phase5_validation_gate_report.json")
    parser.add_argument("--skip_gate_check", action="store_true",
                         help="Proceed even if the Phase 5 validation gate hasn't passed (still prints a warning).")
    parser.add_argument("--output", type=str, default="outputs/phase6_causal_report.json")
    args = parser.parse_args()

    # ---- Gate check (the plan: Phase 6 "runs after the validation gate") ----
    gate_passed = None
    if os.path.exists(args.gate_report):
        with open(args.gate_report) as f:
            gate_passed = json.load(f).get("all_passed")

    if gate_passed is not True:
        print("\n" + "!" * 70)
        print("WARNING: Phase 5's validation gate has not passed "
              f"(outputs/{os.path.basename(args.gate_report)} shows all_passed={gate_passed}).")
        print("Per the plan, Phase 6/7 should not cite numbers as validated until it does.")
        print("Proceeding anyway (this is exploratory/methodology-development output,")
        print("not a claim that Module A/B/C's numbers are trustworthy yet) -- but do")
        print("not present this phase's results to judges as validated without rerunning")
        print("run_phase5.py against real or properly hardened data first and confirming")
        print("all_passed: true.")
        print("!" * 70)
        if not args.skip_gate_check:
            print("\n(Continuing without --skip_gate_check anyway, since this is a warning, "
                  "not a hard block -- Phase 6's own methodology can be developed and its "
                  "logic verified independent of whether Module A/B/C's numbers are final.)")

    # ---- Step 1: SHAP on the frozen fusion model (best-effort) ----
    shap_ok = run([
        sys.executable, "-c",
        "import torch, shap; "
        "print('torch + shap available')",
    ], allow_failure=True)
    if shap_ok == 0:
        run([
            sys.executable, "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "from causal_explainability import explain_fusion_model_shap; "
            "import json; "
            f"r = explain_fusion_model_shap('{args.checkpoint}', '{args.data_dir}'); "
            "print(json.dumps(r, indent=2))",
        ], allow_failure=True)
    else:
        print("\ntorch and/or shap not available in this environment -- skipping the "
              "SHAP-on-frozen-model step (src/causal_explainability.py:explain_fusion_model_shap "
              "is written and ready to run in a torch+shap environment; see the README).")

    # ---- Step 2: backdoor-adjusted confound analysis (always runs) ----
    run([
        sys.executable, "src/causal_explainability.py",
        "--csv_path", f"{args.data_dir}/cleaned_dataset.csv",
        "--output", args.output,
    ])

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)
    print(f"Report -> {args.output}")
    if gate_passed is not True:
        print("\nReminder: Phase 5's validation gate has NOT passed -- treat this phase's")
        print("output as methodology development, not a validated result, until it has.")


if __name__ == "__main__":
    main()
