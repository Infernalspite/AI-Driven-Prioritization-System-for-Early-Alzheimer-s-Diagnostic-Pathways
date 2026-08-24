"""
longitudinal_dataset.py
------------------------
Phase 2: turns each subject's visit history into a padded SEQUENCE, instead
of Phase 1's cross-sectional one-visit-per-example framing (README Phase 1
note: "longitudinal sequencing is Phase 2" -- this is that).

Reuses Phase 0's leakage-safe subject splits and Phase 1's modality/static
column definitions and missing-modality convention (explicit presence
masks, never silent zero-imputation passed off as a real reading).

Targets computed per timestep t (0-indexed within a subject's own visit
sequence, chronological order by visit_month):
  - next_diagnosis[t] = diagnosis at visit t+1 (LABEL_MAP). Valid only if a
    visit t+1 exists for that subject (next_valid[t]). This is the
    TADPOLE-style "predict future clinical status" task.
  - cdr_slope[t] = (CDR_SB[t+1] - CDR_SB[t]) / months_between(t, t+1) * 6
    -> the plan's "MMSE/CDR trajectory slope" regression target, expressed
    as CDR-SB points per 6-month interval (this cohort's visit spacing) so
    the number is directly interpretable. Valid only when t+1 exists AND
    CDR_SB is present at both ends (always true in this cohort -- CDR_SB
    has 0% missingness here -- but the guard is kept because on real ADNI
    data cognitive scores can also go missing at a visit).

Sequences are padded to MAX_VISITS (5, the max visit count in this cohort)
with an explicit seq_mask (True = real visit, False = padding), and the
model packs/unpacks with pack_padded_sequence rather than ever feeding a
padded zero-visit to the LSTM as if it were real.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import MODALITY_COLUMNS, STATIC_COLUMNS, LABEL_MAP, NormalizationStats

MAX_VISITS = 5


class ADTrajectoryDataset(Dataset):
    """
    Each item is one SUBJECT's full visit history (padded to MAX_VISITS):
        modality_features: dict[str -> FloatTensor(MAX_VISITS, len(cols))]
        modality_mask:      dict[str -> BoolTensor(MAX_VISITS)]
        static_features:    FloatTensor(MAX_VISITS, len(STATIC_COLUMNS))
        seq_mask:            BoolTensor(MAX_VISITS)          -- real vs padding
        length:               int                              -- n real visits
        cur_diagnosis:        LongTensor(MAX_VISITS)
        next_diagnosis:       LongTensor(MAX_VISITS)
        next_valid:           BoolTensor(MAX_VISITS)
        cdr_slope:             FloatTensor(MAX_VISITS)
        slope_valid:           BoolTensor(MAX_VISITS)
        cdr_raw:               FloatTensor(MAX_VISITS)         -- unnormalized CDR-SB
        visit_month:           FloatTensor(MAX_VISITS)
    """

    def __init__(self, df: pd.DataFrame, norm_stats: NormalizationStats):
        self.norm_stats = norm_stats
        df = df.sort_values(["subject_id", "visit_month"]).reset_index(drop=True)

        all_numeric_cols = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS
        df_norm = norm_stats.transform(df, all_numeric_cols)
        df_norm["cdr_sb_raw"] = df["CDR_SB"].values  # keep RAW CDR-SB alongside for slope targets

        self.subjects = [(sid, g.sort_values("visit_month").reset_index(drop=True))
                          for sid, g in df_norm.groupby("subject_id")]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id, g = self.subjects[idx]
        T = min(len(g), MAX_VISITS)

        modality_features = {m: torch.zeros(MAX_VISITS, len(cols)) for m, cols in MODALITY_COLUMNS.items()}
        modality_mask = {m: torch.zeros(MAX_VISITS, dtype=torch.bool) for m in MODALITY_COLUMNS}
        static_features = torch.zeros(MAX_VISITS, len(STATIC_COLUMNS))
        seq_mask = torch.zeros(MAX_VISITS, dtype=torch.bool)
        cur_diagnosis = torch.zeros(MAX_VISITS, dtype=torch.long)
        next_diagnosis = torch.zeros(MAX_VISITS, dtype=torch.long)
        next_valid = torch.zeros(MAX_VISITS, dtype=torch.bool)
        cdr_slope = torch.zeros(MAX_VISITS, dtype=torch.float32)
        slope_valid = torch.zeros(MAX_VISITS, dtype=torch.bool)
        cdr_raw = torch.zeros(MAX_VISITS, dtype=torch.float32)
        visit_month = torch.zeros(MAX_VISITS, dtype=torch.float32)

        for t in range(T):
            row = g.iloc[t]
            seq_mask[t] = True
            cur_diagnosis[t] = LABEL_MAP[row["diagnosis"]]
            cdr_raw[t] = row["cdr_sb_raw"]
            visit_month[t] = row["visit_month"]

            for m, cols in MODALITY_COLUMNS.items():
                vals = row[cols].values.astype(np.float32)
                present = not np.isnan(vals).any()
                modality_mask[m][t] = present
                modality_features[m][t] = torch.tensor(np.nan_to_num(vals, nan=0.0))

            static_vals = row[STATIC_COLUMNS].values.astype(np.float32)
            static_features[t] = torch.tensor(np.nan_to_num(static_vals, nan=0.0))

            if t + 1 < T:
                next_row = g.iloc[t + 1]
                next_diagnosis[t] = LABEL_MAP[next_row["diagnosis"]]
                next_valid[t] = True
                dt_months = float(next_row["visit_month"] - row["visit_month"])
                if dt_months > 0 and not np.isnan(next_row["cdr_sb_raw"]) and not np.isnan(row["cdr_sb_raw"]):
                    cdr_slope[t] = (next_row["cdr_sb_raw"] - row["cdr_sb_raw"]) / dt_months * 6.0
                    slope_valid[t] = True

        return {
            "subject_id": subject_id, "length": T,
            "modality_features": modality_features, "modality_mask": modality_mask,
            "static_features": static_features, "seq_mask": seq_mask,
            "cur_diagnosis": cur_diagnosis, "next_diagnosis": next_diagnosis, "next_valid": next_valid,
            "cdr_slope": cdr_slope, "slope_valid": slope_valid,
            "cdr_raw": cdr_raw, "visit_month": visit_month,
        }


def collate_fn(batch):
    out = {"subject_id": [b["subject_id"] for b in batch],
           "length": torch.tensor([b["length"] for b in batch], dtype=torch.long)}
    out["modality_features"] = {m: torch.stack([b["modality_features"][m] for b in batch]) for m in MODALITY_COLUMNS}
    out["modality_mask"] = {m: torch.stack([b["modality_mask"][m] for b in batch]) for m in MODALITY_COLUMNS}
    for key in ["static_features", "seq_mask", "cur_diagnosis", "next_diagnosis",
                "next_valid", "cdr_slope", "slope_valid", "cdr_raw", "visit_month"]:
        out[key] = torch.stack([b[key] for b in batch])
    return out


def load_splits(data_dir: str = "data/raw"):
    """Same subject splits as Phase 1 (Phase 0's leakage-safe split), so
    Phase 1 (cross-sectional) and Phase 2 (longitudinal) are evaluated on
    exactly the same held-out subjects and are directly comparable."""
    df = pd.read_csv(f"{data_dir}/cleaned_dataset.csv")
    with open(f"{data_dir}/subject_splits.json") as f:
        splits = json.load(f)

    train_df = df[df["subject_id"].isin(splits["train_subjects"])].copy()
    val_df = df[df["subject_id"].isin(splits["val_subjects"])].copy()
    test_df = df[df["subject_id"].isin(splits["test_subjects"])].copy()

    all_numeric_cols = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS
    norm_stats = NormalizationStats().fit(train_df, all_numeric_cols)

    return train_df, val_df, test_df, norm_stats
