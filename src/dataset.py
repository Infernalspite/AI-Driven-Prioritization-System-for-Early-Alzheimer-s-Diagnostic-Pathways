"""
dataset.py
----------
Turns the Phase 0 cleaned dataset into per-modality tensors for the fusion
model. Each visit-row is one training example (cross-sectional framing, as
Phase 1 of the plan specifies — longitudinal sequencing is Phase 2).

Key design decision: missing modalities are NOT imputed with mean/zero and
silently fed to the model as if they were real. Instead each modality gets
an explicit presence mask, and the model (see model.py) uses that mask to
either (a) skip the modality in cross-attention entirely, or (b) substitute
a learned "missing" embedding. Silently mean-imputing a modality that's
missing 69% of the time (PET) would let the model learn spurious patterns
from imputed values — this is a common, quiet bug in medical ML pipelines.

Normalization stats (mean/std) are fit on the TRAIN split only and applied
to val/test, to avoid leakage through normalization statistics.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MODALITY_COLUMNS = {
    "cognitive": ["MMSE", "ADAS13", "CDR_SB"],
    "blood": ["abeta42_40_ratio", "ptau181"],
    "mri": ["hippocampal_volume_mm3", "cortical_thickness_mm"],
    "pet": ["pet_amyloid_suvr"],
}
STATIC_COLUMNS = ["age", "education_years"]  # always present, fed alongside every token
LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
LABEL_NAMES = ["CN", "MCI", "AD"]


class NormalizationStats:
    """Fit on train only; reused for val/test to avoid leakage."""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, df: pd.DataFrame, columns: list[str]):
        for col in columns:
            vals = df[col].dropna().values
            self.means[col] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            self.stds[col] = float(np.std(vals) + 1e-6) if len(vals) > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            df[col] = (df[col] - self.means[col]) / self.stds[col]
        return df

    def to_dict(self):
        return {"means": self.means, "stds": self.stds}


class ADFusionDataset(Dataset):
    """
    Each item:
        modality_features: dict[str -> FloatTensor(len(cols))]   (0.0 where missing, post-norm)
        modality_mask:      dict[str -> bool]                    (True if modality present this visit)
        static_features:    FloatTensor(len(STATIC_COLUMNS))
        label:              LongTensor scalar (0=CN,1=MCI,2=AD)
    """

    def __init__(self, df: pd.DataFrame, norm_stats: NormalizationStats):
        self.df = df.reset_index(drop=True)
        self.norm_stats = norm_stats

        all_numeric_cols = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS
        self.df = self.norm_stats.transform(self.df, all_numeric_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        modality_features = {}
        modality_mask = {}
        for modality, cols in MODALITY_COLUMNS.items():
            vals = row[cols].values.astype(np.float32)
            present = not np.isnan(vals).any()
            modality_mask[modality] = present
            vals = np.nan_to_num(vals, nan=0.0)  # zero only matters if mask says "absent"
            modality_features[modality] = torch.tensor(vals, dtype=torch.float32)

        static_vals = row[STATIC_COLUMNS].values.astype(np.float32)
        static_vals = np.nan_to_num(static_vals, nan=0.0)
        static_features = torch.tensor(static_vals, dtype=torch.float32)

        label = torch.tensor(LABEL_MAP[row["diagnosis"]], dtype=torch.long)

        return {
            "modality_features": modality_features,
            "modality_mask": modality_mask,
            "static_features": static_features,
            "label": label,
        }


def collate_fn(batch):
    """Stacks a list of dataset items into batched tensors, keyed by modality."""
    out = {"modality_features": {}, "modality_mask": {}}
    for modality in MODALITY_COLUMNS:
        out["modality_features"][modality] = torch.stack([b["modality_features"][modality] for b in batch])
        out["modality_mask"][modality] = torch.tensor([b["modality_mask"][modality] for b in batch], dtype=torch.bool)
    out["static_features"] = torch.stack([b["static_features"] for b in batch])
    out["label"] = torch.stack([b["label"] for b in batch])
    return out


def load_splits(data_dir: str = "data/raw"):
    """Loads cleaned_dataset.csv + subject_splits.json from Phase 0 and returns
    (train_df, val_df, test_df, norm_stats)."""
    df = pd.read_csv(f"{data_dir}/cleaned_dataset.csv")
    with open(f"{data_dir}/subject_splits.json") as f:
        splits = json.load(f)

    train_df = df[df["subject_id"].isin(splits["train_subjects"])].copy()
    val_df = df[df["subject_id"].isin(splits["val_subjects"])].copy()
    test_df = df[df["subject_id"].isin(splits["test_subjects"])].copy()

    all_numeric_cols = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS
    norm_stats = NormalizationStats().fit(train_df, all_numeric_cols)

    return train_df, val_df, test_df, norm_stats
