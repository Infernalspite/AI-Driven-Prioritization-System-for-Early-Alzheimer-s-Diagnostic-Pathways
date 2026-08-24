"""
validation_gate.py
--------------------
Phase 5D: the gate. Nothing in Phase 6 (causal explainability) or Phase 7
(population resource optimizer) should build on numbers that haven't
cleared this check.

This script does NOT retrain anything itself -- it reads the standard
Phase 1/2/3 evaluation reports (outputs/phase1_comparison_report.json,
outputs/phase2_comparison_report.json, outputs/phase3_comparison_report.json)
and checks three criteria against them:

  1. Module A's fusion model beats the concatenation baseline by a
     non-trivial margin SPECIFICALLY on the missing-modality test subset
     (not just overall) -- reads evaluate.py's existing
     `missing_modality_subset` breakout.

  2. Module B's LSTM/TFT beats the persistence baseline SPECIFICALLY on
     the converter subset, with a converter sample large enough to trust
     (>=20-30 events, not 3) -- reads evaluate_trajectory.py's
     `converter_subset_vs_persistence_baseline` breakout (added in this
     phase; rerun evaluate_trajectory.py after retraining on v2/OASIS data
     to populate it).

  3. Module C's PPO agent finds at least one scenario where escalating to
     MRI (or PET) provides higher reward than stopping early -- reads
     evaluate_pathway.py's `all_episode_trajectories` (added in this
     phase) and checks whether any episode that escalated beat the
     average reward of episodes that stopped immediately, restricted to
     cases where escalating was the CORRECT call (avoids rewarding
     escalation that happened to still get the diagnosis right by luck).

USAGE
    # After retraining Phases 1-3 on data/raw_v2 (or data/raw_oasis) and
    # rerunning the corresponding evaluate_*.py scripts against that data
    # (which write into the SAME outputs/*.json paths by default -- pass
    # --output on each evaluate_*.py call if you want to keep multiple
    # dataset versions' reports side by side, then point this script at
    # them with --phase1_report / --phase2_report / --phase3_report):

    python src/validation_gate.py

    python src/validation_gate.py \\
        --phase1_report outputs/phase1_comparison_report_v2.json \\
        --phase2_report outputs/phase2_comparison_report_v2.json \\
        --phase3_report outputs/phase3_comparison_report_v2.json

Exit code is 0 if all three criteria pass, 1 otherwise -- so this can be
used as a CI-style gate (e.g. `run_phase5.py` treats a nonzero exit as
"do not proceed to Phase 6/7 yet") as well as a human-readable report.
"""

import argparse
import json
import sys

# "Non-trivial margin" thresholds -- documented, not tuned against any
# particular dataset's result, since tuning the bar to whatever the data
# happens to produce would defeat the point of having a gate at all.
FUSION_MISSING_SUBSET_MIN_F1_DELTA = 0.03  # fusion macro-F1 - baseline macro-F1, missing-modality subset
CONVERTER_MIN_N = 20  # minimum converter-subset sample size to trust any comparison on it
CONVERTER_MIN_F1_DELTA = 0.0  # model must at least match, not necessarily beat by a margin (see note in check fn)


def check_fusion_missing_modality_gain(phase1_report_path):
    with open(phase1_report_path) as f:
        report = json.load(f)

    result = {"criterion": "1. Fusion beats concatenation on missing-modality subset", "passed": False}
    try:
        baseline_sub = report["baseline"]["missing_modality_subset"]
        fusion_sub = report["fusion"]["missing_modality_subset"]
    except KeyError:
        result["detail"] = ("No missing_modality_subset in this report -- either the test set has "
                             "no visits with missing modalities (unlikely/misconfigured), or evaluate.py "
                             "wasn't rerun against the data in question.")
        return result

    delta = fusion_sub["macro_f1"] - baseline_sub["macro_f1"]
    result["fusion_macro_f1"] = fusion_sub["macro_f1"]
    result["baseline_macro_f1"] = baseline_sub["macro_f1"]
    result["delta"] = delta
    result["threshold"] = FUSION_MISSING_SUBSET_MIN_F1_DELTA
    result["passed"] = delta >= FUSION_MISSING_SUBSET_MIN_F1_DELTA
    result["detail"] = (f"fusion macro-F1 {fusion_sub['macro_f1']:.3f} vs baseline {baseline_sub['macro_f1']:.3f} "
                         f"(delta {delta:+.3f}, threshold {FUSION_MISSING_SUBSET_MIN_F1_DELTA:+.3f}) "
                         f"on the missing-modality test subset specifically.")
    return result


def check_trajectory_converter_subset(phase2_report_path):
    with open(phase2_report_path) as f:
        report = json.load(f)

    result = {"criterion": "2. Trajectory model beats persistence baseline on converter subset (n>=20)",
              "passed": False}
    sub = report.get("converter_subset_vs_persistence_baseline")
    if not sub or sub.get("n_converter_positions", 0) == 0:
        result["detail"] = ("No converter_subset_vs_persistence_baseline in this report, or zero converter "
                             "positions -- rerun evaluate_trajectory.py (post-Phase-5-update) against the "
                             "data in question.")
        return result

    n = sub["n_converter_positions"]
    model_f1 = sub["model"]["macro_f1"]
    persistence_f1 = sub["persistence_baseline"]["macro_f1"]
    delta = model_f1 - persistence_f1

    result["n_converter_positions"] = n
    result["model_macro_f1"] = model_f1
    result["persistence_baseline_macro_f1"] = persistence_f1
    result["delta"] = delta
    result["min_n_required"] = CONVERTER_MIN_N

    n_ok = n >= CONVERTER_MIN_N
    # Note on the F1 bar: persistence is *always wrong* on every converter position by
    # construction (see evaluate_trajectory.evaluate_converter_subset docstring), so its
    # macro-F1 here is already low/degenerate -- beyond a large-enough sample, ANY model
    # doing better than "always wrong" is the real bar, not a margin over an already-weak
    # baseline. Delta > 0 is the operative check once n is large enough to trust.
    f1_ok = delta > CONVERTER_MIN_F1_DELTA
    result["passed"] = n_ok and f1_ok
    result["detail"] = (f"n={n} converter positions (need >={CONVERTER_MIN_N}); "
                         f"model macro-F1 {model_f1:.3f} vs persistence-baseline {persistence_f1:.3f} "
                         f"(delta {delta:+.3f}). Sample size {'OK' if n_ok else 'TOO SMALL'}, "
                         f"model {'beats' if f1_ok else 'does NOT beat'} persistence.")
    return result


def check_pathway_escalation_value(phase3_report_path):
    with open(phase3_report_path) as f:
        report = json.load(f)

    result = {"criterion": "3. PPO agent finds >=1 scenario where escalating beats stopping early",
              "passed": False}
    ppo = report.get("ppo_agent", {})
    episodes = ppo.get("all_episode_trajectories")
    if not episodes:
        result["detail"] = ("No all_episode_trajectories in ppo_agent report -- rerun "
                             "evaluate_pathway.py (post-Phase-5-update) to populate it.")
        return result

    stop_immediately = [e for e in episodes if e["n_tests_ordered"] == 0]
    escalated_and_correct = [e for e in episodes if e["n_tests_ordered"] > 0 and e["correct"]]

    if not stop_immediately:
        baseline_reward = None
        baseline_note = "No stop-immediately episodes in this run to compare against; using 0.0 as the reward floor."
        baseline_reward_value = 0.0
    else:
        baseline_reward_value = sum(e["reward"] for e in stop_immediately) / len(stop_immediately)
        baseline_note = f"Mean reward of {len(stop_immediately)} stop-immediately episodes."

    qualifying = [e for e in escalated_and_correct if e["reward"] > baseline_reward_value]

    result["n_episodes_total"] = len(episodes)
    result["n_stop_immediately"] = len(stop_immediately)
    result["n_escalated_and_correct"] = len(escalated_and_correct)
    result["stop_immediately_mean_reward"] = baseline_reward_value
    result["n_escalations_beating_stop_immediately_reward"] = len(qualifying)
    result["passed"] = len(qualifying) >= 1
    result["example_qualifying_trajectory"] = qualifying[0]["trajectory"] if qualifying else None
    result["detail"] = (f"{baseline_note} {len(qualifying)} of {len(escalated_and_correct)} correct, "
                         f"escalated episodes exceeded that reward "
                         f"({'>=1 found -> escalation IS sometimes worth it' if qualifying else 'NONE found -> escalation never paid off in this run'}).")
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 5D: validation gate")
    parser.add_argument("--phase1_report", type=str, default="outputs/phase1_comparison_report.json")
    parser.add_argument("--phase2_report", type=str, default="outputs/phase2_comparison_report.json")
    parser.add_argument("--phase3_report", type=str, default="outputs/phase3_comparison_report.json")
    parser.add_argument("--output", type=str, default="outputs/phase5_validation_gate_report.json")
    args = parser.parse_args()

    checks = []
    for name, fn, path in [
        ("fusion_missing_modality", check_fusion_missing_modality_gain, args.phase1_report),
        ("trajectory_converter_subset", check_trajectory_converter_subset, args.phase2_report),
        ("pathway_escalation_value", check_pathway_escalation_value, args.phase3_report),
    ]:
        try:
            checks.append(fn(path))
        except FileNotFoundError:
            checks.append({"criterion": name, "passed": False,
                            "detail": f"Report not found: {path}. Rerun the corresponding evaluate_*.py first."})

    all_passed = all(c["passed"] for c in checks)

    print("=" * 70)
    print("PHASE 5D VALIDATION GATE")
    print("=" * 70)
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"\n[{status}] {c['criterion']}")
        print(f"       {c.get('detail', '(no detail)')}")

    print("\n" + "=" * 70)
    if all_passed:
        print("GATE: PASSED -- Phase 6/7 may proceed on this dataset's numbers.")
    else:
        print("GATE: NOT PASSED -- treat unmet criteria as a genuine negative result to")
        print("report honestly (per the plan's Phase 5D framing), not to omit or paper over.")
        print("Do not build Phase 6/7 claims on the failing criterion/criteria above yet.")
    print("=" * 70)

    report = {"all_passed": all_passed, "checks": checks}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
