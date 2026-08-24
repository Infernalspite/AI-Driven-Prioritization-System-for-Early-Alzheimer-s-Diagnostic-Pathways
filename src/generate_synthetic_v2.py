"""
generate_synthetic_v2.py
-------------------------
Phase 5B: hardened synthetic cohort generator.

Every prior phase's evaluation converged on the same diagnosis: the v1
generator (whatever produced data/raw/cleaned_dataset.csv) derives every
modality from a single shared `decline_rate` latent per subject. That
means:

  - Module A (fusion) can't beat concatenation, because there's no
    genuine cross-modal interaction to exploit -- every modality is just
    a noisy linear readout of the same one number.
  - Module B (trajectory) sees almost-linear decline, not real
    disease-stage-dependent dynamics, and only ~3 conversions exist in
    the whole test split (v1's conversion rate is ~5.8%, vs. ~22% in the
    real 2-year MCI->AD conversion literature this plan cites).
  - Module C (pathway agent) never finds escalation to MRI/PET worth it,
    because no scenario in the data actually *requires* the extra
    modality to resolve ambiguity.

This generator fixes the structural cause, not the symptom:

  1. Two SEPARATE, only weakly-correlated latents drive decline --
     `neurodegeneration` (drives MRI atrophy + cognitive decline) and
     `amyloid_burden` (drives blood biomarkers + PET SUVR) -- so a
     subject can be high on one and low on the other. That's what makes
     fusion's cross-attention potentially worth more than concatenation:
     concatenation can add the two signals together, but it can't
     represent an AND-style interaction between them.
  2. Stage transition (conversion) probability is an explicit AND
     condition: conversion risk requires BOTH tracks to be past a
     threshold, not a linear sum of the two. A subject who is high on
     neurodegeneration alone, or amyloid alone, converts far more slowly
     than one who is high on both -- this is the literal thing a
     concatenation model cannot represent and a cross-attention model
     can.
  3. Per-visit transition probability is calibrated so the resulting
     2-year MCI->AD conversion rate lands in the ADNI-realistic 15-22%
     band (this script prints the realized rate so it can be checked,
     rather than assumed).
  4. The ground-truth interaction function and both latents are logged
     per subject in a separate `ground_truth.csv` / `generator_config.json`
     -- NOT fed to any model -- so Module A/B's learned representations
     can later be checked against a known target instead of just trusting
     a performance number with no ground truth to compare it to.

Output schema is IDENTICAL to data/raw/cleaned_dataset.csv +
subject_splits.json, so it's a drop-in replacement for every existing
Phase 1-4 script (dataset.py, longitudinal_dataset.py, tft_dataset.py all
read this schema unchanged).
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

LABEL_NAMES = ["CN", "MCI", "AD"]
VISIT_MONTHS = [0, 6, 12, 18, 24]  # matches the plan's cited LSTM precedent (5 timepoints)

# Missingness pattern preserved from v1's clinical-realism framing: PET is the
# rarest/most invasive test and is missing most often; blood is intermediate;
# MRI/cognitive are closest to always-ordered.
MISSINGNESS_RATE = {
    "cognitive": 0.02,
    "blood": 0.35,
    "mri": 0.12,
    "pet": 0.68,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_subject(rng, subject_id, config):
    """
    Draws one subject's static covariates, two decline latents, and full
    visit trajectory (stage path + all modality readouts), using the AND-
    interaction conversion rule described in the module docstring.
    """
    age0 = float(np.clip(rng.normal(74, 7.5), 55, 92))
    sex = rng.choice(["M", "F"])
    education_years = float(np.clip(rng.normal(15.5, 2.8), 8, 22))

    # Two only-weakly-correlated latents (rho ~ 0.25): NOT one shared decline_rate.
    mean = [0.0, 0.0]
    rho = config["latent_correlation"]
    cov = [[1.0, rho], [rho, 1.0]]
    neuro_z, amyloid_z = rng.multivariate_normal(mean, cov)

    # Baseline stage sampled with realistic class balance (skewed toward CN/MCI,
    # matching the v1 cohort's baseline_diagnosis_class_balance).
    stage_idx = rng.choice([0, 1, 2], p=[0.40, 0.44, 0.16])

    # Each latent maps to a 0-1 "severity" via sigmoid, offset by baseline stage
    # so more advanced subjects start with higher expected severity on both.
    neuro_severity = _sigmoid(neuro_z + 0.9 * stage_idx - 0.9)
    amyloid_severity = _sigmoid(amyloid_z + 0.9 * stage_idx - 0.9)

    visits = []
    ground_truth_rows = []
    current_stage = stage_idx
    n_visits = rng.choice([3, 4, 5], p=[0.25, 0.35, 0.40])  # matches v1's ~3.07 mean visits/subject

    for vi in range(n_visits):
        stage_before_this_visit = current_stage
        month = VISIT_MONTHS[vi]
        age = age0 + month / 12.0

        # Mild progression of each latent's severity over time (independent tracks).
        t = month / 24.0
        neuro_t = float(np.clip(neuro_severity + 0.08 * t + rng.normal(0, 0.03), 0, 1))
        amyloid_t = float(np.clip(amyloid_severity + 0.05 * t + rng.normal(0, 0.03), 0, 1))

        # --- THE interaction (the structural fix): AND, not a linear sum. ---
        # High conversion risk requires BOTH tracks elevated; being high on only
        # one track gives a much smaller, sub-additive risk bump. This is what a
        # concatenation/linear model structurally cannot represent, and what
        # cross-attention fusion is designed to pick up.
        and_signal = neuro_t * amyloid_t  # product, not sum -> genuine interaction
        additive_signal = 0.5 * (neuro_t + amyloid_t)  # what a linear model sees
        conversion_prob = config["transition_scale"] * and_signal + 0.15 * config["transition_scale"] * additive_signal
        conversion_prob = float(np.clip(conversion_prob, 0.0, 0.9))

        if vi > 0 and current_stage < 2 and rng.random() < conversion_prob:
            current_stage += 1

        stage = current_stage

        # --- Modality readouts, each keyed to the RIGHT latent track (this is what
        # makes the two tracks separately identifiable instead of collapsing back
        # into one number) ---
        mmse = float(np.clip(30 - 10 * neuro_t - 2 * stage + rng.normal(0, 1.1), 0, 30))
        adas13 = float(np.clip(5 + 35 * neuro_t + 4 * stage + rng.normal(0, 2.0), 0, 70))
        cdr_sb = float(np.clip(0.2 + 8.5 * neuro_t + 1.3 * stage + rng.normal(0, 0.4), 0, 18))

        abeta_ratio = float(np.clip(0.12 - 0.07 * amyloid_t + rng.normal(0, 0.006), 0.02, 0.16))
        ptau181 = float(np.clip(15 + 45 * amyloid_t + rng.normal(0, 3.0), 5, 90))

        hippo_vol = float(np.clip(7200 - 1700 * neuro_t + rng.normal(0, 180), 3200, 8200))
        cortical_thick = float(np.clip(2.7 - 0.55 * neuro_t + rng.normal(0, 0.06), 1.6, 3.1))

        pet_suvr = float(np.clip(1.05 + 0.85 * amyloid_t + rng.normal(0, 0.05), 0.9, 2.4))

        row = {
            "subject_id": subject_id,
            "visit_month": month,
            "age": round(age, 1),
            "sex": sex,
            "education_years": round(education_years),
            "diagnosis": LABEL_NAMES[stage],
            "MMSE": round(mmse, 1),
            "ADAS13": round(adas13, 1),
            "CDR_SB": round(cdr_sb, 2),
            "abeta42_40_ratio": round(abeta_ratio, 4),
            "ptau181": round(ptau181, 2),
            "hippocampal_volume_mm3": round(hippo_vol, 1),
            "cortical_thickness_mm": round(cortical_thick, 3),
            "pet_amyloid_suvr": round(pet_suvr, 3),
        }

        # Apply missingness (independent per modality, matches v1's clinical framing).
        if rng.random() < MISSINGNESS_RATE["blood"]:
            row["abeta42_40_ratio"] = np.nan
            row["ptau181"] = np.nan
        if rng.random() < MISSINGNESS_RATE["mri"]:
            row["hippocampal_volume_mm3"] = np.nan
            row["cortical_thickness_mm"] = np.nan
        if rng.random() < MISSINGNESS_RATE["pet"]:
            row["pet_amyloid_suvr"] = np.nan

        visits.append(row)
        ground_truth_rows.append({
            "subject_id": subject_id,
            "visit_month": month,
            "neuro_severity_true": round(neuro_t, 4),
            "amyloid_severity_true": round(amyloid_t, 4),
            "and_interaction_signal_true": round(and_signal, 4),
            "conversion_prob_true": round(conversion_prob, 4),
            "converted_this_visit": bool(current_stage > stage_before_this_visit),
        })

    return visits, ground_truth_rows, stage_idx, current_stage


def generate(n_subjects, seed, config):
    rng = np.random.default_rng(seed)
    all_visits, all_gt = [], []
    n_converted = 0

    for i in range(n_subjects):
        subject_id = f"SUBJ2-{i:04d}"
        visits, gt_rows, baseline_stage, final_stage = make_subject(rng, subject_id, config)
        all_visits.extend(visits)
        all_gt.extend(gt_rows)
        if final_stage > baseline_stage:
            n_converted += 1

    df = pd.DataFrame(all_visits)
    gt_df = pd.DataFrame(all_gt)
    conversion_rate = n_converted / n_subjects
    return df, gt_df, conversion_rate


def make_subject_splits(df, seed, train_frac=0.7, val_frac=0.15):
    rng = np.random.default_rng(seed + 1)
    subjects = df["subject_id"].unique().tolist()
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train_subjects": subjects[:n_train],
        "val_subjects": subjects[n_train:n_train + n_val],
        "test_subjects": subjects[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5B: generate hardened synthetic cohort v2")
    parser.add_argument("--n_subjects", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_correlation", type=float, default=0.25,
                         help="Correlation between the neurodegeneration and amyloid latents. "
                              "Low on purpose -- this is what makes them separately identifiable "
                              "instead of collapsing back into one v1-style decline_rate.")
    parser.add_argument("--transition_scale", type=float, default=0.3,
                         help="Scales conversion probability. Tuned by default to land the "
                              "realized conversion rate in the ADNI-realistic 15-22% band -- "
                              "rerun and check the printed rate if you change n_subjects/seed.")
    parser.add_argument("--out_dir", type=str, default="data/raw_v2")
    args = parser.parse_args()

    config = {
        "latent_correlation": args.latent_correlation,
        "transition_scale": args.transition_scale,
    }

    df, gt_df, conversion_rate = generate(args.n_subjects, args.seed, config)
    splits = make_subject_splits(df, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(f"{args.out_dir}/cleaned_dataset.csv", index=False)
    gt_df.to_csv(f"{args.out_dir}/ground_truth.csv", index=False)
    with open(f"{args.out_dir}/subject_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    n_visits = len(df)
    n_subjects = df["subject_id"].nunique()
    baseline = df.sort_values(["subject_id", "visit_month"]).groupby("subject_id").first()
    class_balance = baseline["diagnosis"].value_counts().to_dict()

    generator_report = {
        "n_subjects": int(n_subjects),
        "n_visits": int(n_visits),
        "mean_visits_per_subject": float(df.groupby("subject_id").size().mean()),
        "baseline_diagnosis_class_balance": {k: int(v) for k, v in class_balance.items()},
        "realized_conversion_rate": round(conversion_rate, 4),
        "target_conversion_band": [0.15, 0.22],
        "within_target_band": bool(0.15 <= conversion_rate <= 0.22),
        "config": config,
        "reference": "Target band cites the 22% (77/347 subject) 2-year MCI->AD conversion "
                      "rate from the longitudinal LSTM study in the plan's reference list.",
    }
    with open(f"{args.out_dir}/generator_config.json", "w") as f:
        json.dump(generator_report, f, indent=2)

    print(f"Generated {n_subjects} subjects / {n_visits} visits -> {args.out_dir}/")
    print(f"Baseline diagnosis class balance: {class_balance}")
    print(f"Realized subject-level conversion rate (baseline stage -> any later stage): "
          f"{conversion_rate:.1%}  (target band: 15-22%)")
    if not generator_report["within_target_band"]:
        print("WARNING: conversion rate outside target band -- adjust --transition_scale and rerun.")
    print(f"Ground truth (not fed to any model) written to {args.out_dir}/ground_truth.csv")


if __name__ == "__main__":
    main()
