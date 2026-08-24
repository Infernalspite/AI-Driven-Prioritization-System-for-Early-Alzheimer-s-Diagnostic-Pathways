"""
smote_baseline.py
------------------
Phase 5C: SMOTE comparison baseline, named explicitly in the plan's tech
stack ("imbalanced-learn ... used only as a reference/comparison against
subject-level oversampling") and reference list (Chawla et al., 2002)
but not previously implemented anywhere in this package. This module
fills that gap.

oversampling.py's SubjectLevelOversampler duplicates whole converter
SUBJECT-SEQUENCES -- the model only ever sees real, recorded visits, and
a subject's temporal ordering is preserved. Standard SMOTE (Chawla et
al., 2002) does something structurally different: it interpolates
between k-nearest-neighbor FEATURE VECTORS at the row level, with no
notion of "subject" or "visit order" at all. Applied to this cohort's
one-item-per-subject trajectory modeling, that means:

  - A synthetic row can be an interpolation between two DIFFERENT
    subjects' visits (or two visits at different disease stages), which
    is not a real patient and cannot be assigned a real longitudinal
    identity.
  - It has no place inside ADTrajectoryDataset's per-subject sequence
    structure without fabricating a fictitious subject_id and visit
    history for it.

This is exactly why the plan calls SMOTE a reference/comparison only,
not the primary approach -- SubjectLevelOversampler stays the
recommended technique for Module B. This module makes that comparison
concrete and runnable (no torch dependency -- SMOTE and the comparison
classifier only need numpy/pandas/scikit-learn, all already in
requirements.txt via scikit-learn) rather than just asserted in prose.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

NUMERIC_FEATURE_COLS = [
    "age", "education_years", "MMSE", "ADAS13", "CDR_SB",
    "abeta42_40_ratio", "ptau181", "hippocampal_volume_mm3",
    "cortical_thickness_mm", "pet_amyloid_suvr",
]


def load_row_level_features(csv_path: str, feature_cols=None) -> pd.DataFrame:
    """Per-visit numeric feature matrix + diagnosis label + subject_id kept
    alongside (not used in interpolation) so synthetic rows can be flagged
    as not belonging to any real subject."""
    feature_cols = feature_cols or NUMERIC_FEATURE_COLS
    df = pd.read_csv(csv_path)
    # Missing modalities (blood/PET not collected at every visit, same
    # missingness Phase 0/1 already document) are mean-imputed here --
    # SMOTE's k-NN distance needs a complete numeric matrix, whereas the
    # actual fusion model instead keeps missingness explicit via masks
    # (dataset.py). That difference is itself part of why this is a
    # simplified reference baseline, not a replacement for the real
    # missing-modality handling used elsewhere in this pipeline.
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].mean())
    return df


class RowLevelSMOTE:
    """
    Minimal from-scratch implementation of Chawla et al. (2002)'s SMOTE:
    for each minority-class sample, find its k nearest minority-class
    neighbors and interpolate a synthetic point along the line to a
    randomly chosen one. Behaviorally equivalent to
    `imblearn.over_sampling.SMOTE` for the continuous-feature case; kept
    dependency-free (numpy/sklearn only) since this dev sandbox has no
    network access to install imbalanced-learn, and the point of this
    module is the *comparison*, not the library.

    Deliberately row-level and subject-unaware, per the class docstring
    above -- that's the property being compared against, not a bug.
    """

    def __init__(self, k_neighbors: int = 5, seed: int = 0):
        self.k_neighbors = k_neighbors
        self.seed = seed

    def fit_resample(self, X: np.ndarray, y: np.ndarray, target_class, n_synthetic: int | None = None):
        rng = np.random.default_rng(self.seed)
        minority_mask = y == target_class
        X_min = X[minority_mask]
        n_min = X_min.shape[0]
        if n_min <= self.k_neighbors:
            raise ValueError(
                f"Only {n_min} minority ('{target_class}') rows -- need more than "
                f"k_neighbors={self.k_neighbors} to interpolate. Lower k_neighbors "
                f"or check the target_class label."
            )
        n_synthetic = n_synthetic if n_synthetic is not None else (X.shape[0] - 2 * n_min)
        n_synthetic = max(n_synthetic, 0)

        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(X_min)
        _, neighbor_idx = nn.kneighbors(X_min)
        neighbor_idx = neighbor_idx[:, 1:]  # drop self-match (distance 0)

        synthetic = np.zeros((n_synthetic, X.shape[1]))
        for i in range(n_synthetic):
            base_idx = rng.integers(0, n_min)
            neighbor = neighbor_idx[base_idx, rng.integers(0, self.k_neighbors)]
            gap = rng.uniform(0.0, 1.0)
            synthetic[i] = X_min[base_idx] + gap * (X_min[neighbor] - X_min[base_idx])

        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.concatenate([y, np.full(n_synthetic, target_class, dtype=y.dtype)])
        return X_resampled, y_resampled, n_synthetic


def compare_oversampling_strategies(
    csv_path: str = "data/raw_v2/cleaned_dataset.csv",
    target_class: str = "AD",
    k_neighbors: int = 5,
    oversample_factor: int = 4,
    seed: int = 42,
) -> dict:
    """
    Runnable, honest comparison (not just prose) of three training-data
    strategies for the same rare-class problem this cohort has throughout
    (AD / converter visits), evaluated with a simple held-out classifier
    -- this is a Phase 5C reference check, not a replacement for Module
    B's own real evaluation harness (evaluate_trajectory.py), which is
    what Phase 5D's validation gate actually reads.

      1. Baseline: no oversampling, class-weighted only (matches
         Phases 1-2's existing default).
      2. Row-level SMOTE (this module): the reference/comparison method
         named in the plan.
      3. Subject-level duplication (the SubjectLevelOversampler logic
         from oversampling.py, applied here to this row-level frame by
         duplicating every row belonging to a target_class subject --
         the row-level analogue of what oversampling.py does at the
         sequence level).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    df = load_row_level_features(csv_path)
    X_all = df[NUMERIC_FEATURE_COLS].to_numpy()
    y_all = df["diagnosis"].to_numpy()
    subject_ids = df["subject_id"].to_numpy()

    # Subject-level train/test split (leakage-safe, same discipline as the
    # rest of this pipeline) so oversampling only ever touches the train side.
    unique_subjects = np.unique(subject_ids)
    train_subj, test_subj = train_test_split(unique_subjects, test_size=0.2, random_state=seed)
    train_mask = np.isin(subject_ids, train_subj)
    test_mask = np.isin(subject_ids, test_subj)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    results = {}

    def eval_strategy(name, X_tr, y_tr):
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_test)
        macro_f1 = f1_score(y_test, preds, average="macro")
        target_f1 = f1_score(y_test, preds, labels=[target_class], average="macro")
        results[name] = {
            "n_train_rows": int(len(y_tr)),
            "n_target_class_train_rows": int((y_tr == target_class).sum()),
            "test_macro_f1": round(float(macro_f1), 4),
            f"test_{target_class}_f1": round(float(target_f1), 4),
        }

    # 1. Baseline: class-weighted only, no resampling.
    eval_strategy("no_oversampling_class_weighted_only", X_train, y_train)

    # 2. Row-level SMOTE.
    smote = RowLevelSMOTE(k_neighbors=k_neighbors, seed=seed)
    n_min = int((y_train == target_class).sum())
    n_majority_avg = int(np.mean([np.sum(y_train == c) for c in np.unique(y_train) if c != target_class]))
    n_synth = max(n_majority_avg - n_min, 0)
    X_smote, y_smote, n_synth_actual = smote.fit_resample(X_train, y_train, target_class, n_synthetic=n_synth)
    eval_strategy("row_level_smote", X_smote, y_smote)
    results["row_level_smote"]["n_synthetic_rows_generated"] = n_synth_actual
    results["row_level_smote"]["note"] = (
        "Synthetic rows are feature-space interpolations between two "
        "possibly-different subjects' visits -- not real patients, not "
        "assignable a genuine subject_id or visit order. Fine for a flat "
        "row-level classifier like the one used for this comparison; not "
        "usable inside ADTrajectoryDataset's per-subject sequence "
        "structure without fabricating a fictitious longitudinal identity."
    )

    # 3. Subject-level duplication (row-level analogue of oversampling.py's
    #    SubjectLevelOversampler: duplicate every real row belonging to a
    #    target_class subject, oversample_factor times -- no interpolation,
    #    no synthetic feature values, every duplicated row is a real
    #    recorded visit).
    target_subject_mask = np.isin(
        subject_ids[train_mask],
        np.unique(subject_ids[train_mask][y_train == target_class]),
    )
    X_dup = np.vstack([X_train] + [X_train[target_subject_mask]] * (oversample_factor - 1))
    y_dup = np.concatenate([y_train] + [y_train[target_subject_mask]] * (oversample_factor - 1))
    eval_strategy("subject_level_duplication", X_dup, y_dup)
    results["subject_level_duplication"]["oversample_factor"] = oversample_factor
    results["subject_level_duplication"]["note"] = (
        "Every duplicated row is a real recorded visit from a real "
        "subject -- no interpolation, no fabricated identity. This is "
        "the row-level analogue of oversampling.py's "
        "SubjectLevelOversampler; the actual Module B training uses the "
        "sequence-level version (whole padded subject trajectories), not "
        "this flattened row-level form."
    )

    return {
        "csv_path": csv_path,
        "target_class": target_class,
        "n_train_subjects": int(len(train_subj)),
        "n_test_subjects": int(len(test_subj)),
        "strategies": results,
    }


def verify_smote_logic() -> dict:
    """
    Correctness check for RowLevelSMOTE's interpolation logic in
    isolation, on a small hand-built fixture with a known minority
    region -- always run (unlike the old version of this check, which
    only ran when data/raw_v2/cleaned_dataset.csv was absent, silently
    skipping its own assertions once real data existed; found via
    testing, fixed by splitting the always-run logic check from the
    real-data comparison below, see sanity_check()).
    """
    rng = np.random.default_rng(0)
    X_majority = rng.normal(loc=0.0, scale=1.0, size=(40, 4))
    X_minority = rng.normal(loc=3.0, scale=0.5, size=(8, 4))
    X = np.vstack([X_majority, X_minority])
    y = np.array(["CN"] * 40 + ["AD"] * 8)

    smote = RowLevelSMOTE(k_neighbors=3, seed=0)
    X_res, y_res, n_synth = smote.fit_resample(X, y, target_class="AD", n_synthetic=20)
    assert n_synth == 20
    assert (y_res == "AD").sum() == 8 + 20
    assert X_res.shape[0] == X.shape[0] + 20
    # Every synthetic point must lie within the minority class's convex
    # hull region (checked cheaply via per-feature min/max bounds), since
    # SMOTE interpolates between two minority points and can never
    # extrapolate beyond them.
    synth_points = X_res[X.shape[0]:]
    assert np.all(synth_points >= X_minority.min(axis=0) - 1e-6)
    assert np.all(synth_points <= X_minority.max(axis=0) + 1e-6)

    print(f"RowLevelSMOTE logic check passed: {n_synth} synthetic AD rows "
          f"generated, all within the minority feature bounds.")
    return {"fixture_check": "passed", "n_synthetic": int(n_synth)}


def sanity_check(csv_path: str = "data/raw_v2/cleaned_dataset.csv"):
    """Always runs verify_smote_logic()'s correctness assertions first,
    regardless of whether csv_path exists, then additionally runs the
    real comparison against csv_path if it's present (real numbers on
    the hardened synthetic cohort) -- so this function's own correctness
    guarantees never get silently skipped just because real data happens
    to be available (see verify_smote_logic()'s docstring for why that
    matters). Takes csv_path as an argument (previously hardcoded to the
    default path regardless of what the caller passed -- found via
    testing with a deliberately nonexistent --csv_path, fixed here)."""
    logic_result = verify_smote_logic()

    import os

    if os.path.exists(csv_path):
        report = compare_oversampling_strategies(csv_path)
        print(json.dumps(report, indent=2))
        return {"logic_check": logic_result, "real_data_report": report}

    print(f"(csv_path '{csv_path}' not found -- skipping the real-data comparison; "
          f"logic check above still ran and passed.)")
    return {"logic_check": logic_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5C: SMOTE vs subject-level oversampling comparison")
    parser.add_argument("--csv_path", type=str, default="data/raw_v2/cleaned_dataset.csv")
    parser.add_argument("--target_class", type=str, default="AD")
    parser.add_argument("--k_neighbors", type=int, default=5)
    parser.add_argument("--oversample_factor", type=int, default=4)
    parser.add_argument("--output", type=str, default="outputs/phase5_smote_comparison_report.json")
    parser.add_argument("--sanity_check", action="store_true",
                         help="Run verify_smote_logic()'s correctness assertions (and the real-data "
                              "comparison too, if --csv_path exists) instead of writing a report.")
    args = parser.parse_args()

    import os

    if args.sanity_check:
        sanity_check(args.csv_path)
    elif os.path.exists(args.csv_path):
        report = compare_oversampling_strategies(
            args.csv_path, args.target_class, args.k_neighbors, args.oversample_factor
        )
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        print(f"\nWrote {args.output}")
    else:
        sanity_check(args.csv_path)
