"""
evaluate_tft.py
----------------
Head-to-head comparison of the Phase 2 stretch-goal TFT against the
canonical TrajectoryLSTM, on the SAME metric: next-visit CDR-SB MAE on
the held-out test split.

Why next-visit CDR-SB (not slope) is the common currency: the LSTM's own
head predicts a *slope* (points/6mo), not next-visit CDR-SB directly, so
it isn't natively comparable to the TFT's raw quantile forecast. We
convert both models onto the same target here: next_cdr_sb = current_cdr
+ predicted_slope * (dt/6) for the LSTM (dt is always 6 months in this
cohort -- verified in tft_dataset.py's docstring), and the TFT's own
rescaled point prediction (median quantile) directly, since TFT forecasts
next_cdr_sb natively. Evaluated on the identical held-out test subjects
(same leakage-safe Phase 0 split), though NOT on identical row counts --
TFT's TimeSeriesDataSet only requires >=1 encoder step so it scores every
valid consecutive-visit pair including the first, while the LSTM's
next-visit head is evaluated at every visit index with a following visit
(see evaluate_trajectory.py) -- both are the largest test set each model
natively supports, not an artificially matched subsample, and the row
counts are reported so this isn't hidden.

Also reports TFT's own probabilistic output (10/50/90th percentile
interval width) at a few example subjects, mirroring the "80% CI" framing
evaluate_trajectory.py's MC-Dropout projection uses for the LSTM -- the
plan's whole pitch for the TFT swap-in was "probabilistic multi-horizon
output" instead of a bare point estimate, so the comparison isn't
complete without showing that TFT gets this natively (no MC Dropout
looping required) while the LSTM needs the MC-Dropout workaround Phase 2
already built.
"""

import json
import numpy as np
import torch

from pytorch_forecasting import TemporalFusionTransformer

from train_tft import make_datasets  # noqa: F401 -- imported for its module-level MultiHorizonMetric monkeypatch
from trajectory_model import TrajectoryLSTM
from longitudinal_dataset import ADTrajectoryDataset, collate_fn, load_splits
from train_trajectory import to_device
from torch.utils.data import DataLoader


def evaluate_tft(checkpoint_path, data_dir="data/raw"):
    training, validation, testing, train_df, val_df, test_df, impute_means = make_datasets(data_dir)
    tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    tft.eval()

    test_loader = testing.to_dataloader(train=False, batch_size=64, num_workers=0)

    point = tft.predict(test_loader, mode="prediction", return_x=False, return_y=True,
                         trainer_kwargs={"enable_progress_bar": False, "logger": False})
    quantile_output = tft.predict(test_loader, mode="quantiles", return_x=False, return_y=False,
                                   trainer_kwargs={"enable_progress_bar": False, "logger": False})

    preds = point.output.squeeze(-1).numpy()
    targets = point.y[0].squeeze(-1).numpy()
    quantiles = quantile_output.squeeze(1).numpy()  # (n, n_quantiles) -- returned as a bare Tensor, not a Prediction, when return_x/return_y are both False

    mae = float(np.mean(np.abs(preds - targets)))

    # QuantileLoss's default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    q_levels = tft.loss.quantiles
    lo_idx, hi_idx = q_levels.index(0.1), q_levels.index(0.9)
    mean_interval_width = float(np.mean(quantiles[:, hi_idx] - quantiles[:, lo_idx]))
    coverage_80 = float(np.mean((targets >= quantiles[:, lo_idx]) & (targets <= quantiles[:, hi_idx])))

    examples = []
    for i in range(min(5, len(targets))):
        examples.append({
            "actual_next_cdr_sb": float(targets[i]),
            "predicted_next_cdr_sb_median": float(preds[i]),
            "predicted_80pct_interval": [float(quantiles[i, lo_idx]), float(quantiles[i, hi_idx])],
        })

    return {
        "n_test_positions": int(len(targets)),
        "mae": mae,
        "mean_80pct_interval_width": mean_interval_width,
        "empirical_80pct_coverage": coverage_80,
        "quantile_levels": q_levels,
        "examples": examples,
    }


def evaluate_lstm_next_cdr(checkpoint_path, data_dir="data/raw"):
    device = torch.device("cpu")
    train_df, val_df, test_df, norm_stats = load_splits(data_dir)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TrajectoryLSTM(embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_ds = ADTrajectoryDataset(test_df, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = to_device(batch, device)
            _, slope_pred = model(batch)
            valid = batch["slope_valid"].cpu().numpy().astype(bool)
            cdr_raw = batch["cdr_raw"].cpu().numpy()
            slope_pred_np = slope_pred.cpu().numpy()
            cdr_slope_actual = batch["cdr_slope"].cpu().numpy()

            # next_cdr = current_cdr + slope * (6/6) -- every visit gap in this
            # cohort is exactly 6 months (verified in tft_dataset.py), so the
            # slope (points/6mo) converts directly without a dt correction.
            pred_next = cdr_raw + slope_pred_np
            actual_next = cdr_raw + cdr_slope_actual

            preds.append(pred_next[valid])
            targets.append(actual_next[valid])

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = float(np.mean(np.abs(preds - targets)))
    return {"n_test_positions": int(len(targets)), "mae": mae}


def main():
    tft_results = evaluate_tft("checkpoints/tft_best.ckpt")
    lstm_results = evaluate_lstm_next_cdr("checkpoints/trajectory_best.pt")

    report = {
        "metric": "next-visit CDR-SB MAE (lower is better)",
        "tft": tft_results,
        "lstm": lstm_results,
        "delta_mae_tft_minus_lstm": tft_results["mae"] - lstm_results["mae"],
        "note": ("Row counts differ (TFT: {tft_n}, LSTM: {lstm_n}) because each model is evaluated on the "
                 "largest held-out set it natively supports on the same test subjects, not an artificially "
                 "matched subsample -- see module docstring. TFT's probabilistic interval (mean width {width:.3f} "
                 "CDR-SB points, empirical 80% coverage {cov:.2f}) is the plan's actual pitch for this stretch "
                 "goal ('probabilistic multi-horizon output' instead of a bare point estimate) -- it gets this "
                 "natively from QuantileLoss, without the MC-Dropout forward-pass loop the LSTM needs for the "
                 "same kind of interval (see evaluate_trajectory.py's project_time_to_threshold). Same running "
                 "caveat as every other phase: this cohort's shared decline_rate latent and short per-subject "
                 "visit histories (2-5 visits) cap how much any architecture, including a full TFT, can improve "
                 "on the LSTM here -- a small MAE delta either direction should not be read as a validated "
                 "'TFT beats/loses to LSTM' result.").format(
            tft_n=tft_results["n_test_positions"], lstm_n=lstm_results["n_test_positions"],
            width=tft_results["mean_80pct_interval_width"], cov=tft_results["empirical_80pct_coverage"],
        ),
    }

    with open("outputs/phase2_tft_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Phase 2 stretch goal: TFT vs LSTM head-to-head ===")
    print(f"TFT  next-visit CDR-SB MAE:  {tft_results['mae']:.3f}  (n={tft_results['n_test_positions']})")
    print(f"LSTM next-visit CDR-SB MAE:  {lstm_results['mae']:.3f}  (n={lstm_results['n_test_positions']})")
    print(f"TFT 80% interval: mean width {tft_results['mean_80pct_interval_width']:.3f}, "
          f"empirical coverage {tft_results['empirical_80pct_coverage']:.2f}")
    print("\nWrote outputs/phase2_tft_comparison_report.json")


if __name__ == "__main__":
    main()
