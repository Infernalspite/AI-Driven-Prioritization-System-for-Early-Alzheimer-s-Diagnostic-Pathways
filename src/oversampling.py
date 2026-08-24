"""
oversampling.py
----------------
Phase 5C: subject-level oversampling of converter trajectories.

Class-weighted loss (and FocalLoss, see losses.py) reweights the GRADIENT
once an example is seen, but doesn't change how often it's seen. With
only ~89 converter visits out of ~2,100 in the hardened v2 cohort (17.4%
subject-level conversion rate -- see generate_synthetic_v2.py), a
random minibatch can still go several steps without a converter subject
in it at all.

The naive fix -- oversample individual ROWS that show a transition -- is
wrong here specifically because Phase 2's dataset
(longitudinal_dataset.ADTrajectoryDataset) is one-item-per-SUBJECT: each
__getitem__ call returns a subject's whole padded visit sequence, not a
single row. Oversampling rows would either do nothing (there are no
row-level items to duplicate) or, if applied upstream of that dataset,
would let the same subject's OTHER (non-converting) visits appear
duplicated too, inflating apparent learnability of the stable visits
along with the rare converting one.

The correct unit of oversampling here is therefore the SUBJECT: duplicate
whole converter subject-sequences as extra dataset indices, so Module B's
LSTM/TFT sees more converting trajectories per epoch without splitting a
subject's own timeline across the train/val/test boundary (which would be
a Phase 0 leakage-safe-split violation) and without inventing any new
row-level data.
"""

from collections import Counter

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


def subject_is_converter(subject_df: pd.DataFrame) -> bool:
    """True if this subject's diagnosis changes at any point across their
    chronologically-sorted visit history (matches generate_synthetic_v2.py's
    own baseline-vs-final-stage definition of 'converted')."""
    diagnoses = subject_df.sort_values("visit_month")["diagnosis"].tolist()
    return len(set(diagnoses)) > 1


class SubjectLevelOversampler(Sampler):
    """
    Drop-in replacement for the default sampler on
    longitudinal_dataset.ADTrajectoryDataset (or any Dataset that is one
    item per subject, exposing `.subjects` as a list of (subject_id,
    subject_df) pairs -- ADTrajectoryDataset already does this).

    Each epoch, every converter subject-index is repeated
    `oversample_factor` times (duplicated indices, not synthetic rows --
    the model still only ever sees real recorded visits); non-converter
    subjects appear once. Indices are shuffled after duplication so
    repeated converter indices aren't clustered together in a batch.

    Usage:
        train_dataset = ADTrajectoryDataset(train_df, norm_stats)
        sampler = SubjectLevelOversampler(train_dataset, oversample_factor=4)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler,
                                   collate_fn=collate_fn)
    """

    def __init__(self, dataset, oversample_factor: int = 4, seed: int = 0):
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        self.seed = seed
        self._epoch = 0

        self.converter_indices = []
        self.stable_indices = []
        for idx, (subject_id, subject_df) in enumerate(dataset.subjects):
            if subject_is_converter(subject_df):
                self.converter_indices.append(idx)
            else:
                self.stable_indices.append(idx)

        if len(self.converter_indices) == 0:
            raise ValueError(
                "No converter subjects found in this dataset split -- oversampling "
                "has nothing to oversample. This itself is a signal worth surfacing "
                "(check generate_synthetic_v2.py's realized_conversion_rate report), "
                "not something to silently proceed past."
            )

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch so the shuffle differs run to run
        (mirrors DistributedSampler's convention)."""
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        indices = list(self.stable_indices) + list(self.converter_indices) * self.oversample_factor
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.stable_indices) + len(self.converter_indices) * self.oversample_factor

    def report(self) -> dict:
        """Diagnostic summary -- log this alongside training so the effective
        class balance the model actually trained on is auditable, not just
        the pre-oversampling split's balance."""
        n_stable = len(self.stable_indices)
        n_converter_effective = len(self.converter_indices) * self.oversample_factor
        total = n_stable + n_converter_effective
        return {
            "n_subjects_total": len(self.dataset.subjects),
            "n_converter_subjects_raw": len(self.converter_indices),
            "n_stable_subjects_raw": n_stable,
            "oversample_factor": self.oversample_factor,
            "effective_converter_share_per_epoch": round(n_converter_effective / total, 4),
            "raw_converter_share": round(len(self.converter_indices) / len(self.dataset.subjects), 4),
        }


def sanity_check():
    """Pure-Python/pandas check -- doesn't require torch, since the logic
    being verified (which subjects count as converters, and the resulting
    index multiset) doesn't depend on the tensor collation step."""

    class _FakeDataset:
        def __init__(self, subjects):
            self.subjects = subjects

    rows_stable = pd.DataFrame({"visit_month": [0, 6, 12], "diagnosis": ["CN", "CN", "CN"]})
    rows_converter = pd.DataFrame({"visit_month": [0, 6, 12], "diagnosis": ["MCI", "MCI", "AD"]})

    fake = _FakeDataset([
        ("S1", rows_stable), ("S2", rows_stable), ("S3", rows_stable),
        ("S4", rows_converter),
    ])

    sampler = SubjectLevelOversampler(fake, oversample_factor=3, seed=0)
    assert sampler.converter_indices == [3]
    assert sorted(sampler.stable_indices) == [0, 1, 2]
    assert len(sampler) == 3 + 1 * 3  # 3 stable subjects + 1 converter subject repeated 3x

    counts = Counter(list(iter(sampler)))
    assert counts[3] == 3  # converter subject appears exactly oversample_factor times
    for i in (0, 1, 2):
        assert counts[i] == 1

    report = sampler.report()
    assert report["n_converter_subjects_raw"] == 1
    assert report["oversample_factor"] == 3

    print("SubjectLevelOversampler sanity check passed:", report)


if __name__ == "__main__":
    sanity_check()
