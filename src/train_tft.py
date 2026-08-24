"""
train_tft.py
-------------
Phase 2 stretch goal: "swap in a small TFT (the pytorch-forecasting
library has a ready TFT implementation) for probabilistic multi-horizon
output." Trains a TemporalFusionTransformer to forecast next-visit
CDR-SB with quantile (probabilistic) output, on the same leakage-safe
subject splits as the LSTM (trajectory_model.py), so evaluate_tft.py can
put the two head-to-head on identical held-out subjects.

Single-step forecast (max_prediction_length=1) to match what the LSTM's
slope head is evaluated on in evaluate_trajectory.py (next-visit CDR-SB),
rather than showcasing multi-step forecasting the LSTM was never asked to
do -- an unequal comparison would defeat the point of a head-to-head.

Two real library-version bugs had to be worked around to get this
training at all, on pytorch-forecasting 1.8.0 + lightning 2.6.5 +
torchmetrics 1.9.0 (verified by isolating each one, not guessed):

1. "element 0 of tensors does not require grad and does not have a
   grad_fn" on the very first backward pass. Traced this to
   MultiHorizonMetric (the base class QuantileLoss inherits from):
   its forward() accumulates the batch loss into persistent torchmetrics
   state buffers (self.losses / self.lengths, registered via add_state)
   and reads the loss back out of that buffer via compute(). On this
   library/torchmetrics combination that accumulate-then-read round trip
   silently detaches the buffer from the autograd graph -- confirmed by
   patching update()/compute() with print statements and watching
   requires_grad go True -> False across that exact boundary, and ruled
   out lightning's inference_mode-poisoning-a-buffer explanation by
   setting Trainer(inference_mode=False) and reproducing anyway. The fix
   below (_fixed_multihorizon_forward) computes and reduces the loss
   directly -- the same arithmetic update()/compute() do internally --
   without ever writing into the buggy persistent buffer, which keeps
   the autograd graph intact. Monkeypatched onto MultiHorizonMetric
   (QuantileLoss's parent) rather than forked into a QuantileLoss
   subclass, since the bug is in the inherited accumulation path, not
   in QuantileLoss's own loss() method.
2. NaN loss after fix #1 (i.e. gradients flowed, but the loss value
   itself was NaN). Cause: target_normalizer=GroupNormalizer(groups=
   ["subject_id"]) fits a separate mean/std per subject, and this
   cohort's per-subject visit counts are small (2-5 visits) -- for
   several subjects the resulting per-group std was ~0, so normalizing
   CDR_SB by it produced inf/NaN. Switched to a global (ungrouped)
   GroupNormalizer, which fits one mean/std across all subjects --
   correct given CDR_SB's absolute scale is comparable across subjects
   in this cohort (it's a bounded clinical scale, not something that
   needs per-subject centering the way e.g. a subject-specific
   baseline-offset value would).
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import lightning.pytorch as pl
import torch
from torch.nn.utils import rnn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.metrics.base_metrics._base_metrics import MultiHorizonMetric
from pytorch_forecasting.utils import unsqueeze_like, unpack_sequence

from tft_dataset import build_long_df, STATIC_REAL_COLS, STATIC_CAT_COLS, TARGET, unknown_real_cols

MAX_ENCODER_LENGTH = 3
MAX_PREDICTION_LENGTH = 1


def _fixed_multihorizon_forward(self, y_pred, target, **kwargs):
    """Replacement for MultiHorizonMetric.forward (see module docstring,
    workaround #1). Computes+reduces the loss directly instead of round-
    tripping it through the buggy stateful update()/compute() torchmetrics
    accumulator, which silently drops the autograd graph on this
    pytorch-forecasting/torchmetrics/lightning version combination."""
    if isinstance(target, (list, tuple)) and not isinstance(target, rnn.PackedSequence):
        target_val, weight = target
    else:
        target_val, weight = target, None

    if isinstance(target_val, rnn.PackedSequence):
        target_val, lengths = unpack_sequence(target_val)
    else:
        lengths = torch.full((target_val.size(0),), fill_value=target_val.size(1),
                              dtype=torch.long, device=target_val.device)

    losses = self.loss(y_pred, target_val)
    if weight is not None:
        losses = losses * unsqueeze_like(weight, losses)
    losses = self.mask_losses(losses, lengths)
    return self.reduce_loss(losses, lengths)


MultiHorizonMetric.forward = _fixed_multihorizon_forward


def make_datasets(data_dir: str = "data/raw"):
    train_df, val_df, test_df, impute_means = build_long_df(data_dir)

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["subject_id"],
        min_encoder_length=1,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_CAT_COLS,
        static_reals=STATIC_REAL_COLS,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=unknown_real_cols() + [TARGET],
        # Global (ungrouped) normalizer -- see workaround #2 in the module
        # docstring. A per-subject GroupNormalizer blows up to NaN because
        # several subjects only have 2-3 visits, giving a near-zero
        # per-group std.
        target_normalizer=GroupNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
        # Train/val/test subjects are disjoint by design (Phase 0's leakage-safe
        # subject-level split). subject_id is only used to GROUP rows into
        # sequences, never fed to the model as a feature, so unseen ids in
        # val/test are harmless -- just let them fall into a shared bucket
        # instead of raising "unknown category".
        categorical_encoders={"subject_id": NaNLabelEncoder(add_nan=True)},
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True, predict=False)
    testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True, predict=False)

    return training, validation, testing, train_df, val_df, test_df, impute_means


def train_tft(max_epochs: int, batch_size: int, lr: float, hidden_size: int,
              attention_head_size: int, dropout: float, seed: int,
              data_dir: str, checkpoint_dir: str):
    pl.seed_everything(seed)

    training, validation, testing, *_ = make_datasets(data_dir)

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=max(4, hidden_size // 2),
        loss=QuantileLoss(),
        log_interval=0,
        optimizer="adam",
    )

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, mode="min")
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir, filename="tft_best", monitor="val_loss", mode="min", save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint_cb],
        enable_progress_bar=True,
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_cb.best_model_path or f"{checkpoint_dir}/tft_best.ckpt"
    print(f"\nBest TFT checkpoint: {best_path} (val_loss={checkpoint_cb.best_model_score})")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 stretch-goal TFT")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--attention_head_size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train_tft(args.max_epochs, args.batch_size, args.lr, args.hidden_size,
              args.attention_head_size, args.dropout, args.seed, args.data_dir, args.checkpoint_dir)


if __name__ == "__main__":
    main()
