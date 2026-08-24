"""
tft_dataset.py
---------------
Phase 2's stretch goal: "swap in a small TFT (pytorch-forecasting) for
probabilistic multi-horizon output." This module builds the long-format
dataframe pytorch_forecasting.TimeSeriesDataSet expects, reusing Phase
0/1's leakage-safe subject splits and missing-modality convention
(explicit presence indicators, not silent imputation passed off as real
-- TFT can't consume NaN directly the way the LSTM's masked-embedding
approach can, so each modality's missingness becomes its own binary
time-varying feature instead of being hidden).

Design:
  - time_idx = visit_month // 6 (0..4). This cohort has zero visit gaps
    (every subject's recorded visits are back-to-back 6-month steps --
    verified against data/raw/cleaned_dataset.csv), so this is exact,
    not an approximation.
  - target = CDR_SB (raw, unnormalized -- pytorch_forecasting handles its
    own target normalization via GroupNormalizer, fit on train only).
  - static: age, education_years (reals), sex (categorical).
  - time-varying known real: time_idx itself (trivially "known" ahead of
    time, standard TFT convention).
  - time-varying unknown reals: MMSE, ADAS13 (always present in this
    cohort), plus blood/MRI/PET features train-mean-imputed with a
    paired *_present binary indicator for each, so missingness is still
    visible to the model as a feature exactly like Phase 1/2's missing-
    modality tokens -- just represented differently because TFT's
    encoder can't take a NaN.
  - group_ids = subject_id. Train/val/test subjects are disjoint (Phase
    0's leakage-safe split), which is fine for TimeSeriesDataSet: group
    id is only used to chunk rows into per-subject sequences, not fit as
    a categorical model feature, so unseen subject ids in val/test don't
    hit the "unseen category" issue that would occur if subject_id were
    added to static_categoricals.
"""

import json
import numpy as np
import pandas as pd

MODALITY_IMPUTE_COLS = [
    "abeta42_40_ratio", "ptau181", "hippocampal_volume_mm3",
    "cortical_thickness_mm", "pet_amyloid_suvr",
]
ALWAYS_PRESENT_COLS = ["MMSE", "ADAS13"]
STATIC_REAL_COLS = ["age", "education_years"]
STATIC_CAT_COLS = ["sex"]
TARGET = "CDR_SB"


def build_long_df(data_dir: str = "data/raw"):
    """Returns (train_df, val_df, test_df, impute_means) all in the long
    format TimeSeriesDataSet expects, with *_present indicator columns
    added and missing values train-mean-imputed (fit on train only)."""
    df = pd.read_csv(f"{data_dir}/cleaned_dataset.csv")
    with open(f"{data_dir}/subject_splits.json") as f:
        splits = json.load(f)

    df = df.sort_values(["subject_id", "visit_month"]).reset_index(drop=True)
    df["time_idx"] = (df["visit_month"] // 6).astype(int)
    df["subject_id"] = df["subject_id"].astype(str)

    train_mask = df["subject_id"].isin(splits["train_subjects"])
    impute_means = {c: float(df.loc[train_mask, c].dropna().mean()) for c in MODALITY_IMPUTE_COLS}

    for c in MODALITY_IMPUTE_COLS:
        df[f"{c}_present"] = (~df[c].isna()).astype(np.float32)
        df[c] = df[c].fillna(impute_means[c]).astype(np.float32)

    for c in ALWAYS_PRESENT_COLS + [TARGET] + STATIC_REAL_COLS:
        df[c] = df[c].astype(np.float32)

    train_df = df[df["subject_id"].isin(splits["train_subjects"])].copy().reset_index(drop=True)
    val_df = df[df["subject_id"].isin(splits["val_subjects"])].copy().reset_index(drop=True)
    test_df = df[df["subject_id"].isin(splits["test_subjects"])].copy().reset_index(drop=True)

    return train_df, val_df, test_df, impute_means


def unknown_real_cols():
    return ALWAYS_PRESENT_COLS + MODALITY_IMPUTE_COLS + [f"{c}_present" for c in MODALITY_IMPUTE_COLS]
