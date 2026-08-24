"""
resource_optimizer.py
-----------------------
Phase 7 / Module F: population resource-constrained optimizer. Per the plan:

    "Given a fixed weekly capacity, decide which subset of currently-
    flagged patients to escalate to maximize expected diagnostic yield
    across the population ... Practical build: knapsack-style MILP
    maximizing Sum(expected diagnostic yield_i * x_i) subject to
    Sum x_i <= capacity, where yield combines Module C's escalation
    value and Module D's uncertainty. Solve with PuLP or OR-Tools."

Explicitly out of the plan's own build scope in this package's README
("should-build after the gate clears") -- built here anyway on request.
Same gate-status caveat as Phase 6: outputs/phase5_validation_gate_report.json
currently shows all_passed: false, so treat everything below as
methodology development, not a validated result.

Two layers, same split as Phase 6's SHAP-vs-backdoor-adjustment pattern:

  1. `solve_allocation()` -- the actual MILP/knapsack solve. Tries `pulp`
     first (the plan's named tool); falls back to a from-scratch exact
     0/1-knapsack dynamic program (integer weights only -- true here,
     since every test's cost in pathway_env.py's TEST_INFO table is a
     whole number) when `pulp` isn't installed, which is this dev
     sandbox's situation (no network to install it). The DP fallback is
     proven exact against brute-force enumeration in `sanity_check()`,
     not just asserted.

  2. Per-patient "expected diagnostic yield" computation. The plan says
     this should "combine Module C's escalation value and Module D's
     uncertainty" -- i.e. it should come from the real frozen pathway
     agent and the real MC-Dropout uncertainty layer.
     `compute_population_yield_full()` is written against exactly those
     (load_fusion.py's frozen CrossAttentionFusion, train_pathway.py's
     PPO checkpoint, uncertainty.py's MC-Dropout decomposition) but NOT
     executed here -- needs torch + stable-baselines3, neither available
     in this sandbox. `compute_population_yield_surrogate()` is the
     executable stand-in actually run for the numbers below: two plain
     RandomForest classifiers (cognitive-only vs. full-workup) trained
     on the real v2 TRAIN split, used to estimate, for each TEST-split
     patient with an incomplete workup, (a) how uncertain the
     cognitive-only screening currently is (a Module-D-style entropy
     proxy) and (b) how much revealing the missing modalities would
     actually shift the prediction toward the true diagnosis (a
     Module-C-style escalation-value proxy) -- both computed from real,
     held-out data, not fabricated.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

# Same cost table as pathway_env.py's TEST_INFO, reused here rather than
# reimported (pathway_env.py imports torch at module level; this file's
# knapsack-solving half is meant to run without it -- same reasoning as
# causal_explainability.py's decision not to import dataset.py directly).
TEST_COST = {"blood": 1.0, "mri": 3.0, "pet": 8.0}
MODALITY_MISSING_COL = {
    "blood": ["abeta42_40_ratio", "ptau181"],
    "mri": ["hippocampal_volume_mm3", "cortical_thickness_mm"],
    "pet": ["pet_amyloid_suvr"],
}


# ---------------------------------------------------------------------------
# 1. Knapsack / MILP solver
# ---------------------------------------------------------------------------

def solve_allocation(items: list[dict], capacity: float) -> dict:
    """
    items: list of {"id": ..., "cost": int, "yield": float}
    capacity: total budget (same units as "cost", e.g. pathway_env.py's
        cost table -- blood=1, MRI=3, PET=8).

    Maximizes Sum(yield_i * x_i) subject to Sum(cost_i * x_i) <= capacity,
    x_i in {0, 1} -- the plan's own knapsack-style MILP formulation.

    Tries `pulp` (the plan's named solver) first; falls back to an exact
    dynamic-programming 0/1 knapsack solve if it's not installed. Both
    paths solve the SAME problem to the SAME (exact) optimum for integer
    costs -- the DP isn't an approximation, just a dependency-free
    implementation of the exact algorithm this particular knapsack
    formulation reduces to. Returns which solver was actually used so
    that's never left ambiguous in the output.
    """
    try:
        import pulp

        prob = pulp.LpProblem("population_allocation", pulp.LpMaximize)
        x = {it["id"]: pulp.LpVariable(f"x_{it['id']}", cat="Binary") for it in items}
        prob += pulp.lpSum(it["yield"] * x[it["id"]] for it in items)
        prob += pulp.lpSum(it["cost"] * x[it["id"]] for it in items) <= capacity
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [it["id"] for it in items if pulp.value(x[it["id"]]) > 0.5]
        total_yield = sum(it["yield"] for it in items if it["id"] in selected)
        total_cost = sum(it["cost"] for it in items if it["id"] in selected)
        solver_used = "pulp (CBC)"
    except ImportError:
        selected, total_yield, total_cost = _dp_knapsack(items, capacity)
        solver_used = "dynamic-programming fallback (pulp not installed in this environment)"

    return {
        "capacity": capacity,
        "solver_used": solver_used,
        "n_candidates": len(items),
        "n_selected": len(selected),
        "total_yield": round(float(total_yield), 4),
        "total_cost": float(total_cost),
        "capacity_utilization": round(float(total_cost) / capacity, 4) if capacity > 0 else 0.0,
        "selected_ids": selected,
    }


def _dp_knapsack(items: list[dict], capacity: float):
    """Exact 0/1 knapsack via dynamic programming. Requires integer costs
    (true for this package's TEST_COST table: 1, 3, 8 and their sums)."""
    cap = int(capacity)
    n = len(items)
    # dp[i][c] = best achievable yield using the first i items with budget c
    dp = np.zeros((n + 1, cap + 1))
    for i in range(1, n + 1):
        w = int(items[i - 1]["cost"])
        v = items[i - 1]["yield"]
        dp[i, :] = dp[i - 1, :]
        if w <= cap:
            dp[i, w:] = np.maximum(dp[i - 1, w:], dp[i - 1, : cap - w + 1] + v)

    # Backtrack to recover the selected item set.
    selected = []
    c = cap
    for i in range(n, 0, -1):
        if dp[i, c] != dp[i - 1, c]:
            selected.append(items[i - 1]["id"])
            c -= int(items[i - 1]["cost"])
    selected.reverse()

    total_yield = dp[n, cap]
    total_cost = sum(items[i]["cost"] for i, it in enumerate(items) if it["id"] in selected)
    return selected, total_yield, total_cost


def sanity_check():
    """Proves the DP fallback is exact by comparing it against brute-force
    enumeration on a small random instance -- the thing that actually
    needs checking for a from-scratch solver standing in for a proper
    MILP library."""
    rng = np.random.default_rng(0)
    n = 15
    items = [
        {"id": f"P{i}", "cost": int(rng.choice([1, 3, 4, 8, 9, 11, 12])), "yield": round(float(rng.uniform(0, 1)), 3)}
        for i in range(n)
    ]
    capacity = 20

    _, dp_yield, _ = _dp_knapsack(items, capacity)

    best_brute = 0.0
    for mask in range(1 << n):
        cost = sum(items[i]["cost"] for i in range(n) if mask & (1 << i))
        if cost <= capacity:
            val = sum(items[i]["yield"] for i in range(n) if mask & (1 << i))
            best_brute = max(best_brute, val)

    assert abs(dp_yield - best_brute) < 1e-6, (dp_yield, best_brute)
    print(
        f"resource_optimizer sanity check passed: DP knapsack yield ({dp_yield:.3f}) "
        f"exactly matches brute-force optimum ({best_brute:.3f}) over all "
        f"{2**n} subsets, n={n} items, capacity={capacity}."
    )
    return {"dp_yield": float(dp_yield), "brute_force_optimum": float(best_brute)}


# ---------------------------------------------------------------------------
# 2a. Real version -- written, not executed in this sandbox (needs torch +
#     stable-baselines3 + the frozen Phase 1/3/4 checkpoints).
# ---------------------------------------------------------------------------

def compute_population_yield_full(data_dir: str, fusion_checkpoint: str, ppo_checkpoint: str):
    """
    The plan's actual intended data source: for every currently-flagged
    (incomplete-workup) test-split patient, run the frozen Module A
    fusion model + Module D's MC-Dropout uncertainty (uncertainty.py) to
    get a real epistemic-uncertainty estimate, and the frozen Module C
    PPO agent (train_pathway.py's checkpoint, loaded via stable-
    baselines3's PPO.load) to get its learned escalation value for that
    patient's specific state. yield_i = escalation_value_i * epistemic_uncertainty_i,
    the model-grounded version of the surrogate combination
    `compute_population_yield_surrogate` uses below.

    NOT EXECUTED in this dev sandbox: requires torch, stable-baselines3,
    and checkpoints/{fusion_best,pathway_ppo}.pt(.zip), none load-bearing
    here (no torch installed, no network to get it). Written against
    load_fusion.py / uncertainty.py / train_pathway.py's existing
    interfaces so it's a straightforward run in the full training
    environment -- same status as Phase 6's explain_fusion_model_shap()
    and Phase 5A's OASIS migration adapter.
    """
    import sys
    sys.path.insert(0, ".")
    import torch
    from stable_baselines3 import PPO

    from load_fusion import load_frozen_fusion
    from uncertainty import mc_predict, decompose_uncertainty  # Phase 4's functions
    from pathway_env import ADPathwayEnv, TEST_INFO

    device = torch.device("cpu")
    fusion_model, norm_stats = load_frozen_fusion(fusion_checkpoint, device)
    ppo_agent = PPO.load(ppo_checkpoint)

    env = ADPathwayEnv(data_dir=data_dir, split="test", fusion_model=fusion_model, norm_stats=norm_stats)
    yields = {}
    for patient_id, obs in env.iter_flagged_patients():  # patients with >=1 missing modality
        mc_probs = mc_predict(fusion_model, obs, n_passes=10)
        _, _, epistemic = decompose_uncertainty(mc_probs)
        action, _ = ppo_agent.predict(obs, deterministic=True)
        escalation_value = env.estimate_escalation_value(patient_id, action)  # PPO's learned Q-ish signal
        missing_modalities = env.missing_modalities(patient_id)
        cost = sum(TEST_INFO[m]["cost"] for m in missing_modalities)
        yields[patient_id] = {
            "yield": float(escalation_value * epistemic),
            "cost": float(cost),
        }
    return yields


# ---------------------------------------------------------------------------
# 2b. Surrogate version -- executable here, real numbers on real v2 data.
# ---------------------------------------------------------------------------

def compute_population_yield_surrogate(data_dir: str = "data/raw_v2", seed: int = 42) -> list[dict]:
    """
    Executable stand-in for compute_population_yield_full, using plain
    scikit-learn instead of the frozen torch models (same reasoning as
    smote_baseline.py / causal_explainability.py's dependency-avoidance
    throughout this package). Two RandomForest classifiers trained on
    the real leakage-safe TRAIN split:

      - `clf_screening`: cognitive features only (MMSE, ADAS13, CDR_SB --
        always recorded, per Phase 0's cohort table, so this is what's
        genuinely known before any escalation decision).
      - `clf_full`: cognitive + blood + MRI + PET (mean-imputed where
        missing, same simplification smote_baseline.py/
        causal_explainability.py already make).

    For each incomplete-workup TEST-split patient (real held-out data,
    never seen by either classifier during fitting):

      - epistemic-style uncertainty proxy = Shannon entropy of
        `clf_screening`'s predicted class distribution -- how unsure the
        currently-available (cognitive-only) evidence leaves us, playing
        the role Module D's MC-Dropout epistemic estimate would play in
        the full version.
      - escalation-value proxy = max(0, P_full(true class) -
        P_screening(true class)) -- how much probability mass the missing
        modalities would shift toward the CORRECT diagnosis if collected,
        playing the role Module C's learned escalation value would play.
      - yield = uncertainty_proxy * escalation_value_proxy (the plan's
        own "combines Module C's escalation value and Module D's
        uncertainty" formula).
      - cost = sum of TEST_COST[m] for whichever of blood/MRI/PET are
        actually missing for that specific patient's visit -- a REAL,
        per-patient variable cost drawn from the cohort's actual
        missingness pattern, not a fabricated constant.
    """
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv(f"{data_dir}/cleaned_dataset.csv")
    splits = json.load(open(f"{data_dir}/subject_splits.json"))
    train_mask = df["subject_id"].isin(splits["train_subjects"])
    test_mask = df["subject_id"].isin(splits["test_subjects"])

    screening_cols = ["MMSE", "ADAS13", "CDR_SB"]
    full_cols = screening_cols + ["abeta42_40_ratio", "ptau181",
                                   "hippocampal_volume_mm3", "cortical_thickness_mm",
                                   "pet_amyloid_suvr"]
    df_imputed = df.copy()
    for col in full_cols:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

    clf_screening = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    clf_screening.fit(df_imputed.loc[train_mask, screening_cols], df.loc[train_mask, "diagnosis"])

    clf_full = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    clf_full.fit(df_imputed.loc[train_mask, full_cols], df.loc[train_mask, "diagnosis"])

    classes = list(clf_screening.classes_)
    test_df = df[test_mask].reset_index(drop=True)
    test_imputed = df_imputed[test_mask].reset_index(drop=True)

    probs_screening = clf_screening.predict_proba(test_imputed[screening_cols])
    probs_full = clf_full.predict_proba(test_imputed[full_cols])

    items = []
    for i, row in test_df.iterrows():
        missing_modalities = [
            m for m, cols in MODALITY_MISSING_COL.items()
            if row[cols].isna().any()
        ]
        if not missing_modalities:
            continue  # already has a complete workup -- nothing to escalate

        true_class_idx = classes.index(row["diagnosis"])
        p_screen_true = probs_screening[i, true_class_idx]
        p_full_true = probs_full[i, true_class_idx]

        entropy = -np.sum(probs_screening[i] * np.log(probs_screening[i] + 1e-12))
        escalation_value = max(0.0, p_full_true - p_screen_true)
        cost = sum(TEST_COST[m] for m in missing_modalities)

        items.append({
            "id": f"{row['subject_id']}_m{int(row['visit_month'])}",
            "true_diagnosis": row["diagnosis"],
            "missing_modalities": missing_modalities,
            "cost": cost,
            "uncertainty_proxy": round(float(entropy), 4),
            "escalation_value_proxy": round(float(escalation_value), 4),
            "yield": round(float(entropy * escalation_value), 4),
        })

    return items


def capacity_sweep(items: list[dict], capacities: list[float]) -> list[dict]:
    """Re-solves the allocation at several capacity levels -- the static
    equivalent of the plan's "live capacity-slider demo re-solving in
    real time" (an actual interactive slider is a frontend concern
    outside this repo's scope; this is the data such a slider would
    re-query at each position)."""
    return [solve_allocation(items, cap) for cap in capacities]


def sanity_check_yield_surrogate(data_dir: str = "data/raw_v2"):
    """Confirms the surrogate yield computation only ever produces
    escalation-value scores in [0, 1] and only considers genuinely
    incomplete-workup patients -- cheap invariant checks on the real
    output, not a full correctness proof (there's no ground-truth
    "correct" yield to check against, unlike the knapsack solver above).
    Takes data_dir as an argument (previously hardcoded, ignoring
    whatever --data_dir the caller passed -- found via testing the same
    class of bug fixed in smote_baseline.py's sanity_check(), fixed
    here too)."""
    items = compute_population_yield_surrogate(data_dir)
    assert len(items) > 0, "no incomplete-workup candidates found -- check missingness in the input data"
    for it in items:
        assert 0.0 <= it["escalation_value_proxy"] <= 1.0
        assert it["uncertainty_proxy"] >= 0.0
        assert len(it["missing_modalities"]) >= 1
        assert it["cost"] == sum(TEST_COST[m] for m in it["missing_modalities"])
    print(f"resource_optimizer yield-surrogate sanity check passed: "
          f"{len(items)} incomplete-workup candidates, all yield/cost "
          f"invariants hold.")
    return {"n_candidates": len(items)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: population resource-constrained optimizer")
    parser.add_argument("--data_dir", type=str, default="data/raw_v2")
    parser.add_argument("--capacities", type=float, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--output", type=str, default="outputs/phase7_resource_allocation_report.json")
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    import os

    if args.sanity_check:
        sanity_check()
        sanity_check_yield_surrogate(args.data_dir)
    else:
        items = compute_population_yield_surrogate(args.data_dir)
        sweep = capacity_sweep(items, args.capacities)

        report = {
            "data_dir": args.data_dir,
            "n_candidates_total": len(items),
            "candidate_pool_summary": {
                "mean_cost": round(float(np.mean([it["cost"] for it in items])), 3),
                "mean_yield": round(float(np.mean([it["yield"] for it in items])), 4),
                "by_missing_modalities": pd.Series(
                    [tuple(sorted(it["missing_modalities"])) for it in items]
                ).value_counts().to_dict(),
            },
            "capacity_sweep": sweep,
        }
        # value_counts() keys are tuples -- not JSON-serializable as dict keys directly.
        report["candidate_pool_summary"]["by_missing_modalities"] = {
            "+".join(k): v for k, v in report["candidate_pool_summary"]["by_missing_modalities"].items()
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {args.output}")
