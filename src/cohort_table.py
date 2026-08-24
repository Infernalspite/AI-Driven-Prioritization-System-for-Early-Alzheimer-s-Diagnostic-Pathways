"""
cohort_table.py
-----------------
Phase 0 deliverable, per the plan:

    "Deliverable: clean tabular + imaging-feature dataset, versioned, with
    a documented cohort table (like ADNI papers always include: n
    subjects, class balance, age/sex breakdown)."

This was the one Phase 0 artifact not yet written down anywhere in the
repo -- the cleaned dataset and leakage-safe subject splits existed, but
not the cohort table itself. Run this to (re)generate it from
data/raw/cleaned_dataset.csv + subject_splits.json; the numbers quoted in
the README's Phase 0 section come from this script's output.

Reports both VISIT-level counts (every row in cleaned_dataset.csv) and
SUBJECT-level counts (one row per subject, using each subject's FIRST
recorded visit -- i.e., their baseline diagnosis/age -- which is what
"n subjects" and "class balance" mean in an ADNI-style cohort table; a
subject who starts CN and later becomes AD is counted as CN at baseline,
not double-counted).
"""

import json
import argparse

import pandas as pd


def build_cohort_table(data_dir="data/raw"):
    df = pd.read_csv(f"{data_dir}/cleaned_dataset.csv")
    with open(f"{data_dir}/subject_splits.json") as f:
        splits = json.load(f)

    baseline = (df.sort_values(["subject_id", "visit_month"])
                  .groupby("subject_id").first().reset_index())

    def split_of(sid):
        for name, ids in splits.items():
            if sid in ids:
                return name.replace("_subjects", "")
        return "unassigned"
    baseline["split"] = baseline["subject_id"].map(split_of)

    report = {
        "n_visits_total": int(len(df)),
        "n_subjects_total": int(df["subject_id"].nunique()),
        "visits_per_subject": {
            "mean": float(df.groupby("subject_id").size().mean()),
            "distribution": df.groupby("subject_id").size().value_counts().sort_index().to_dict(),
        },
        "baseline_diagnosis_class_balance": {
            k: int(v) for k, v in baseline["diagnosis"].value_counts().to_dict().items()
        },
        "baseline_age": {
            "mean": float(baseline["age"].mean()), "std": float(baseline["age"].std()),
            "min": float(baseline["age"].min()), "max": float(baseline["age"].max()),
        },
        "baseline_sex": {k: int(v) for k, v in baseline["sex"].value_counts().to_dict().items()},
        "baseline_education_years": {
            "mean": float(baseline["education_years"].mean()), "std": float(baseline["education_years"].std()),
        },
        "by_split": {},
        "modality_missingness_visit_level": {
            col: float(df[col].isna().mean()) for col in
            ["abeta42_40_ratio", "ptau181", "hippocampal_volume_mm3", "cortical_thickness_mm", "pet_amyloid_suvr"]
        },
    }

    for split_name in ["train", "val", "test"]:
        sub = baseline[baseline["split"] == split_name]
        report["by_split"][split_name] = {
            "n_subjects": int(len(sub)),
            "diagnosis_class_balance": {k: int(v) for k, v in sub["diagnosis"].value_counts().to_dict().items()},
            "mean_age": float(sub["age"].mean()) if len(sub) else None,
        }

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate the Phase 0 cohort table")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="outputs/phase0_cohort_table.json")
    args = parser.parse_args()

    report = build_cohort_table(args.data_dir)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"n subjects: {report['n_subjects_total']}  |  n visits: {report['n_visits_total']}")
    print(f"Baseline diagnosis: {report['baseline_diagnosis_class_balance']}")
    print(f"Baseline age: {report['baseline_age']['mean']:.1f} +/- {report['baseline_age']['std']:.1f} "
          f"(range {report['baseline_age']['min']:.0f}-{report['baseline_age']['max']:.0f})")
    print(f"Baseline sex: {report['baseline_sex']}")
    for split_name, s in report["by_split"].items():
        print(f"  {split_name}: n={s['n_subjects']}  {s['diagnosis_class_balance']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
