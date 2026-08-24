"""
migrate_oasis.py
------------------
Phase 5A: real-data migration path -- OASIS Longitudinal (Kaggle,
jboysen/mri-and-alzheimers) adapted to this pipeline's schema.

WHY OASIS FIRST (not straight to ADNI)
ADNI (adni.loni.usc.edu) is the plan's full-fidelity target and is what
Phase 0's README already recommends registering for on day one -- but it
requires credentialed access approval that takes real time to clear. OASIS
Longitudinal is a faster-to-access, genuinely real (non-simulated)
longitudinal cohort that can validate the pipeline's architecture while
an ADNI application is pending. This script is the adapter for that
interim dataset; nothing here is a substitute for eventually validating
against ADNI once access clears.

WHAT OASIS HAS AND DOESN'T HAVE
OASIS longitudinal gives: MMSE, CDR, eTIV, nWBV, ASF, age, education, SES,
M/F, hand, visit number -- but NO blood biomarkers (Abeta42/40, p-tau) and
NO PET SUVR. Those two modalities are dropped for every subject, not
imputed. Because model.py's CrossAttentionFusion already substitutes a
learned "missing" embedding per modality when data is absent (Phase 1
design choice, made specifically so this kind of degradation wouldn't
require a rearchitecture), running this pipeline on OASIS-migrated data
means blood+PET are permanently masked missing -- the fusion model
degrades gracefully to a 2-modality (cognitive + MRI-derived) fusion,
using the exact same forward pass and checkpoint format as Phase 1.

COLUMN MAPPING (OASIS -> this pipeline's schema, see dataset.py):
    OASIS column         -> this pipeline               transform
    ---------------------------------------------------------------------
    Subject ID            -> subject_id                  as-is (string)
    MR Delay (days)        -> visit_month                 /30.44, rounded
    Age                     -> age                          as-is
    M/F                      -> sex                          as-is ('M'/'F')
    EDUC (years)              -> education_years              as-is
    CDR (0/0.5/1/2)            -> diagnosis                    CDR->{CN,MCI,AD}
                                                                 (see cdr_to_diagnosis)
    MMSE                        -> MMSE                          as-is
    (none)                       -> ADAS13                        NOT AVAILABLE -> NaN
    CDR (raw, *2 for CDR-SB proxy)-> CDR_SB                        see note below
    (none)                        -> abeta42_40_ratio, ptau181     NOT AVAILABLE -> NaN
    nWBV, eTIV                    -> hippocampal_volume_mm3 proxy  see note below
    ASF                            -> cortical_thickness_mm proxy  see note below
    (none)                          -> pet_amyloid_suvr             NOT AVAILABLE -> NaN

CDR_SB note: OASIS reports only the global CDR score (0, 0.5, 1, 2), not
the 18-point Clinical Dementia Rating Sum-of-Boxes this pipeline's
CDR_SB column expects. There is no principled way to reconstruct CDR-SB
from global CDR alone (that would need the six individual domain scores,
which OASIS longitudinal doesn't publish). This script does NOT invent
those digits: `CDR_SB` here is `global_cdr * 3.0` -- a documented, coarse
linear proxy (midpoint of the typical CDR-SB range reported for each
global CDR stage in the literature), clearly not clinically equivalent to
a real CDR-SB. Anyone using downstream CDR-SB slope regression (Phase 2's
Module B) on migrated OASIS data should treat that output as illustrative
of the methodology, not a literal CDR-SB forecast -- the same honesty
standard the rest of this plan already applies to its synthetic-data
caveats.

MRI feature proxies note: OASIS longitudinal publishes normalized whole-
brain volume (nWBV) and estimated total intracranial volume (eTIV), not
this pipeline's hippocampal_volume_mm3 / cortical_thickness_mm directly.
hippocampal_volume_mm3 is approximated as nWBV * eTIV (an absolute
whole-brain volume proxy, not literally hippocampal volume -- OASIS
longitudinal does not publish a hippocampal-specific measure).
cortical_thickness_mm has no OASIS equivalent at all and is left NaN
(masked missing) rather than proxied from ASF, which measures head-size
scaling, not cortical thickness, and would be a materially misleading
substitution.

USAGE
    # 1. Download OASIS longitudinal CSV (requires network + a Kaggle
    #    account/API token -- NOT available in this sandbox):
    #        pip install kagglehub
    #        python -c "import kagglehub; print(kagglehub.dataset_download('jboysen/mri-and-alzheimers'))"
    #    or download 'oasis_longitudinal.csv' manually from
    #    https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
    #
    # 2. python src/migrate_oasis.py --input_csv path/to/oasis_longitudinal.csv

This produces data/raw_oasis/cleaned_dataset.csv + subject_splits.json in
the exact schema dataset.py / longitudinal_dataset.py already expect --
no changes needed to any Phase 1-4 script to run against it.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# Modalities that OASIS longitudinal cannot supply at all -- permanently
# masked missing for every subject/visit, not imputed. See CrossAttentionFusion's
# learned-missing-embedding handling in model.py.
OASIS_UNAVAILABLE_COLUMNS = ["abeta42_40_ratio", "ptau181", "ADAS13", "cortical_thickness_mm", "pet_amyloid_suvr"]

# Coarse, documented linear proxy from global CDR -> a CDR-SB-shaped number.
# NOT a real CDR-SB; see module docstring.
GLOBAL_CDR_TO_CDR_SB_PROXY_SCALE = 3.0


def cdr_to_diagnosis(cdr):
    """Standard global-CDR staging used throughout the OASIS/ADNI literature:
    0 = CN (cognitively normal), 0.5 = MCI (very mild), >=1 = AD (dementia)."""
    if pd.isna(cdr):
        return np.nan
    if cdr == 0:
        return "CN"
    if cdr == 0.5:
        return "MCI"
    return "AD"


def load_oasis_csv(path):
    df = pd.read_csv(path)
    # OASIS longitudinal's published column names (jboysen/mri-and-alzheimers);
    # normalize whitespace/case defensively since Kaggle re-exports sometimes vary.
    df.columns = [c.strip() for c in df.columns]
    required = ["Subject ID", "MR Delay", "Age", "M/F", "EDUC", "CDR", "MMSE", "nWBV", "eTIV"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing expected OASIS longitudinal columns: {missing}. "
            f"Found columns: {df.columns.tolist()}. If this is a re-exported/renamed "
            f"version of the dataset, adjust the column-name constants at the top of "
            f"load_oasis_csv() rather than guessing -- do not silently proceed on a "
            f"column that might mean something different."
        )
    return df


def migrate(df: pd.DataFrame) -> pd.DataFrame:
    """OASIS longitudinal -> this pipeline's cleaned_dataset.csv schema."""
    out = pd.DataFrame()
    out["subject_id"] = df["Subject ID"].astype(str)
    out["visit_month"] = (df["MR Delay"].fillna(0) / 30.44).round().astype(int)
    out["age"] = df["Age"].astype(float)
    out["sex"] = df["M/F"].map({"M": "M", "F": "F"})
    out["education_years"] = df["EDUC"].astype(float)
    out["diagnosis"] = df["CDR"].apply(cdr_to_diagnosis)
    out["MMSE"] = df["MMSE"].astype(float)

    out["ADAS13"] = np.nan  # not available in OASIS
    out["CDR_SB"] = df["CDR"].astype(float) * GLOBAL_CDR_TO_CDR_SB_PROXY_SCALE  # documented proxy, see docstring

    out["abeta42_40_ratio"] = np.nan  # not available
    out["ptau181"] = np.nan  # not available

    out["hippocampal_volume_mm3"] = df["nWBV"].astype(float) * df["eTIV"].astype(float)  # whole-brain-volume proxy
    out["cortical_thickness_mm"] = np.nan  # not available, not proxied (see docstring)

    out["pet_amyloid_suvr"] = np.nan  # not available

    # Drop rows with no diagnosis (CDR missing) or no subject id -- can't be used
    # for either training or evaluation without a label.
    before = len(out)
    out = out.dropna(subset=["subject_id", "diagnosis"]).reset_index(drop=True)
    dropped = before - len(out)

    return out, dropped


def make_subject_splits(df, seed=42, train_frac=0.7, val_frac=0.15):
    """Subject-level split, same leakage-safe discipline as Phase 0's
    original load_splits: a subject's whole visit history stays on one
    side of the split."""
    rng = np.random.default_rng(seed)
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


def try_kagglehub_download():
    """Best-effort automatic download via kagglehub, if installed and a
    Kaggle API token is configured. Requires network access, which this
    development sandbox does not have -- provided for use in an
    environment that does have it. Returns the CSV path, or None if it
    can't be done automatically (caller should fall back to --input_csv)."""
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed (`pip install kagglehub`) -- "
              "falling back to manual --input_csv.")
        return None

    try:
        dataset_dir = kagglehub.dataset_download("jboysen/mri-and-alzheimers")
    except Exception as e:
        print(f"kagglehub download failed ({e}) -- this needs network access and a "
              f"configured Kaggle API token (~/.kaggle/kaggle.json). Falling back to "
              f"manual --input_csv.")
        return None

    for fname in os.listdir(dataset_dir):
        if fname.lower().endswith(".csv") and "longitudinal" in fname.lower():
            return os.path.join(dataset_dir, fname)
    # fall back to any csv in the download if the exact filename differs
    csvs = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    return os.path.join(dataset_dir, csvs[0]) if csvs else None


def main():
    parser = argparse.ArgumentParser(description="Phase 5A: migrate OASIS Longitudinal -> pipeline schema")
    parser.add_argument("--input_csv", type=str, default=None,
                         help="Path to a manually-downloaded oasis_longitudinal.csv. "
                              "If omitted, attempts kagglehub auto-download (requires network).")
    parser.add_argument("--out_dir", type=str, default="data/raw_oasis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = args.input_csv
    if csv_path is None:
        print("No --input_csv given, attempting kagglehub auto-download...")
        csv_path = try_kagglehub_download()
        if csv_path is None:
            raise SystemExit(
                "Could not obtain the OASIS longitudinal CSV automatically. Either:\n"
                "  1) pip install kagglehub, configure a Kaggle API token, and rerun, or\n"
                "  2) download 'oasis_longitudinal.csv' manually from "
                "https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers and pass "
                "--input_csv path/to/oasis_longitudinal.csv"
            )

    raw = load_oasis_csv(csv_path)
    migrated, n_dropped = migrate(raw)
    splits = make_subject_splits(migrated, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    migrated.to_csv(f"{args.out_dir}/cleaned_dataset.csv", index=False)
    with open(f"{args.out_dir}/subject_splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    baseline = migrated.sort_values(["subject_id", "visit_month"]).groupby("subject_id").first()
    migration_report = {
        "source": "OASIS Longitudinal (Kaggle, jboysen/mri-and-alzheimers)",
        "n_subjects": int(migrated["subject_id"].nunique()),
        "n_visits": int(len(migrated)),
        "n_rows_dropped_missing_label_or_id": int(n_dropped),
        "baseline_diagnosis_class_balance": {k: int(v) for k, v in baseline["diagnosis"].value_counts().to_dict().items()},
        "mean_visits_per_subject": float(migrated.groupby("subject_id").size().mean()),
        "modalities_permanently_unavailable": OASIS_UNAVAILABLE_COLUMNS,
        "caveats": [
            "CDR_SB is a documented linear proxy (global_CDR * 3.0), NOT a real "
            "Clinical Dementia Rating Sum-of-Boxes -- OASIS longitudinal only "
            "publishes the global CDR score. Treat any CDR-SB slope output on "
            "this data as illustrative of methodology, not a clinical forecast.",
            "hippocampal_volume_mm3 is a whole-brain-volume proxy (nWBV * eTIV), "
            "not a literal hippocampal volume measurement.",
            "cortical_thickness_mm, ADAS13, abeta42_40_ratio, ptau181, and "
            "pet_amyloid_suvr are not available in OASIS longitudinal and are "
            "masked missing (NaN) for every visit -- Module A's fusion model "
            "degrades to cognitive+MRI-only fusion on this data (see model.py's "
            "learned-missing-embedding design).",
            "Phase 3's pathway-agent action space must drop 'order_pet' entirely "
            "when running against this data -- there is no PET action to learn, "
            "since the modality doesn't exist here at all (not just missing at "
            "some visits). See pathway_env.py's ESCALATION_MODALITIES.",
        ],
    }
    with open(f"{args.out_dir}/migration_report.json", "w") as f:
        json.dump(migration_report, f, indent=2)

    print(f"Migrated {migration_report['n_subjects']} subjects / {migration_report['n_visits']} visits "
          f"-> {args.out_dir}/ ({n_dropped} rows dropped for missing label/id)")
    print(f"Baseline diagnosis class balance: {migration_report['baseline_diagnosis_class_balance']}")
    print(f"Permanently unavailable modalities (masked missing): {OASIS_UNAVAILABLE_COLUMNS}")
    print(f"Full migration report -> {args.out_dir}/migration_report.json")


if __name__ == "__main__":
    main()
