"""
evaluate_trajectory.py
------------------------
Head-to-head test-set evaluation for Phase 2, mirroring Phase 1's honest,
show-the-comparison style. Three things, per the plan's own citations:

1. Next-visit diagnosis (TADPOLE-style future-status prediction): overall
   accuracy/macro-F1, computed only at sequence positions where a next
   visit actually exists.

2. MCI-to-AD conversion detection, the exact comparison the plan cites --
   "an LSTM ... hit AUC 0.93 ... beating a cross-sectional random forest
   (AUC 0.90) on the same ADNI cohort." We reproduce that comparison here:
   the longitudinal LSTM's P(next=AD) at each MCI visit, vs. a Random
   Forest trained on that SAME visit's cross-sectional features alone
   (no history). Caveat stated up front, not buried: this synthetic
   cohort has very few conversion events (17 train / 2 val / 3 test), so
   treat the AUC gap as illustrative of the comparison methodology, not a
   validated result -- same spirit as Phase 1's honest calibration note.

3. CDR-SB trajectory-slope regression: model MAE vs. a naive
   "predict-no-change" (slope=0) baseline, on the same held-out positions.

4. Time-to-threshold projection demo: for a few test subjects, MC Dropout
   (Gal & Ghahramani, 2016 -- keep dropout active at inference, repeated
   stochastic passes) turns the point-estimate slope into a distribution,
   giving the plan's target output shape: "projected to cross MCI->AD
   threshold in ~X months, 80% CI: [lo, hi]" instead of a bare score. This
   is Phase 2 exposing the hook Phase 4's calibration work is meant to
   refine, not a substitute for it.
"""

import json
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score

from dataset import MODALITY_COLUMNS, STATIC_COLUMNS, LABEL_MAP, LABEL_NAMES
from longitudinal_dataset import ADTrajectoryDataset, collate_fn, load_splits, MAX_VISITS
from trajectory_model import TrajectoryLSTM
from train_trajectory import to_device
from torch.utils.data import DataLoader

# Empirically chosen from the TRAIN split only (midpoint of MCI's 75th
# percentile CDR-SB and AD's 25th percentile CDR-SB) -- see README for the
# derivation. Not a clinical cutoff, a cohort-specific escalation trigger.
CDR_SB_AD_THRESHOLD = 3.8

N_MC_PASSES = 30
CI_LOW_PCT, CI_HIGH_PCT = 10, 90


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TrajectoryLSTM(embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


# ---------------------------------------------------------------------
# 1 & 2: next-visit prediction + MCI->AD conversion subset
# ---------------------------------------------------------------------

def evaluate_next_visit(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    mci_ad_probs, mci_ad_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            next_logits, _ = model(batch)  # (B,T,3)
            probs = torch.softmax(next_logits, dim=-1)

            valid = batch["next_valid"]
            mask_np = valid.cpu().numpy().astype(bool)
            preds = next_logits.argmax(dim=-1).cpu().numpy()
            labels = batch["next_diagnosis"].cpu().numpy()
            all_preds.append(preds[mask_np])
            all_labels.append(labels[mask_np])

            is_mci = (batch["cur_diagnosis"].cpu().numpy() == LABEL_MAP["MCI"])
            mci_mask = mask_np & is_mci
            if mci_mask.any():
                ad_prob = probs[..., LABEL_MAP["AD"]].cpu().numpy()
                mci_ad_probs.append(ad_prob[mci_mask])
                mci_ad_labels.append((labels[mci_mask] == LABEL_MAP["AD"]).astype(int))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mci_ad_probs = np.concatenate(mci_ad_probs) if mci_ad_probs else np.array([])
    mci_ad_labels = np.concatenate(mci_ad_labels) if mci_ad_labels else np.array([])

    return {
        "overall": {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "macro_f1": float(f1_score(all_labels, all_preds, average="macro")),
            "n_positions": int(len(all_labels)),
        },
        "mci_to_ad_probs": mci_ad_probs,
        "mci_to_ad_labels": mci_ad_labels,
    }


def evaluate_converter_subset(model, loader, device):
    """
    Phase 5D validation gate criterion 2: does the LSTM/TFT beat the
    persistence baseline SPECIFICALLY on the converter subset (positions
    where the diagnosis actually changes at the next visit), not just on
    overall next-visit accuracy -- which is dominated by (and easy to win
    on via) the much larger stable-diagnosis majority.

    Persistence baseline = predict cur_diagnosis (no change). By
    construction it is *always wrong* on every true converter position, so
    the model's accuracy on this subset is compared against 0.0, and its
    macro-F1 against the persistence baseline's macro-F1 on the same
    (label-imbalanced-by-definition) subset.
    """
    model.eval()
    converter_preds, converter_labels, converter_persistence_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            next_logits, _ = model(batch)  # (B,T,3)

            valid = batch["next_valid"].cpu().numpy().astype(bool)
            cur = batch["cur_diagnosis"].cpu().numpy()
            nxt = batch["next_diagnosis"].cpu().numpy()
            preds = next_logits.argmax(dim=-1).cpu().numpy()

            is_converter = valid & (cur != nxt)
            if is_converter.any():
                converter_preds.append(preds[is_converter])
                converter_labels.append(nxt[is_converter])
                converter_persistence_preds.append(cur[is_converter])  # always wrong, by definition

    if not converter_labels:
        return {"n_converter_positions": 0, "note": "No converter positions in this split/loader."}

    converter_preds = np.concatenate(converter_preds)
    converter_labels = np.concatenate(converter_labels)
    converter_persistence_preds = np.concatenate(converter_persistence_preds)

    return {
        "n_converter_positions": int(len(converter_labels)),
        "model": {
            "accuracy": float(accuracy_score(converter_labels, converter_preds)),
            "macro_f1": float(f1_score(converter_labels, converter_preds, average="macro", zero_division=0)),
        },
        "persistence_baseline": {
            "accuracy": float(accuracy_score(converter_labels, converter_persistence_preds)),  # = 0.0 always
            "macro_f1": float(f1_score(converter_labels, converter_persistence_preds, average="macro", zero_division=0)),
        },
        "note": ("Persistence baseline predicts cur_diagnosis (no change), so by construction it scores "
                 "0 accuracy on every true converter position -- this subset isolates exactly the cases "
                 "where longitudinal modeling should have something to add over a naive baseline."),
    }


def cross_sectional_conversion_baseline(train_df, test_df):
    """Random Forest trained on ONE visit's raw (zero-imputed) cross-
    sectional features, no history -- the plan's cited comparison point
    for 'longitudinal beats cross-sectional'."""
    feat_cols = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS

    def build_xy(df):
        df = df.sort_values(["subject_id", "visit_month"])
        rows_X, rows_y = [], []
        for sid, g in df.groupby("subject_id"):
            g = g.reset_index(drop=True)
            for i in range(len(g) - 1):
                if g.loc[i, "diagnosis"] != "MCI":
                    continue
                x = g.loc[i, feat_cols].values.astype(np.float32)
                x = np.nan_to_num(x, nan=0.0)
                y = 1 if g.loc[i + 1, "diagnosis"] == "AD" else 0
                rows_X.append(x)
                rows_y.append(y)
        return np.array(rows_X), np.array(rows_y)

    X_train, y_train = build_xy(train_df)
    X_test, y_test = build_xy(test_df)

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "n_train_positions": int(len(y_train)), "n_test_positions": int(len(y_test)),
        "n_test_conversions": int(y_test.sum()),
        "accuracy": float(accuracy_score(y_test, preds)) if len(y_test) else None,
        "auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None,
        "probs": probs, "labels": y_test,
    }


# ---------------------------------------------------------------------
# 3: CDR-SB slope regression
# ---------------------------------------------------------------------

def evaluate_slope_regression(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            _, slope_pred = model(batch)
            valid = batch["slope_valid"].cpu().numpy().astype(bool)
            all_preds.append(slope_pred.cpu().numpy()[valid])
            all_targets.append(batch["cdr_slope"].cpu().numpy()[valid])
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    naive_zero_mae = float(mean_absolute_error(all_targets, np.zeros_like(all_targets)))
    model_mae = float(mean_absolute_error(all_targets, all_preds))
    return {"model_mae": model_mae, "naive_zero_change_baseline_mae": naive_zero_mae, "n_positions": int(len(all_targets))}


# ---------------------------------------------------------------------
# 4: time-to-threshold projection demo, with MC Dropout uncertainty
# ---------------------------------------------------------------------

def project_time_to_threshold(model, single_subject_batch, device, n_passes=N_MC_PASSES):
    """Runs N stochastic forward passes (dropout left ON) at a subject's
    last real visit, turning the point-estimate slope into a distribution
    of projected months-to-threshold-crossing."""
    model.train()  # keep dropout active -- MC Dropout, Gal & Ghahramani 2016
    batch = to_device(single_subject_batch, device)
    length = batch["length"].item()
    current_cdr = batch["cdr_raw"][0, length - 1].item()

    slopes = []
    ad_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            next_logits, slope_pred = model.project_last_visit(batch)
            slopes.append(slope_pred.item())
            ad_probs.append(torch.softmax(next_logits, dim=-1)[0, LABEL_MAP["AD"]].item())
    model.eval()

    slopes = np.array(slopes)
    ad_probs = np.array(ad_probs)

    if current_cdr >= CDR_SB_AD_THRESHOLD:
        return {
            "status": "already_at_or_above_threshold",
            "current_cdr_sb": current_cdr, "threshold": CDR_SB_AD_THRESHOLD,
            "next_visit_ad_prob_mean": float(ad_probs.mean()), "next_visit_ad_prob_std": float(ad_probs.std()),
        }

    # months to cross: (threshold - current) / (points per 6mo) * 6
    months = np.where(slopes > 1e-4, (CDR_SB_AD_THRESHOLD - current_cdr) / slopes * 6.0, np.inf)
    finite = months[np.isfinite(months)]
    frac_declining = float((slopes > 1e-4).mean())

    if len(finite) < n_passes * 0.5:
        return {
            "status": "no_clear_decline_on_current_trajectory",
            "current_cdr_sb": current_cdr, "threshold": CDR_SB_AD_THRESHOLD,
            "frac_mc_passes_declining": frac_declining,
            "next_visit_ad_prob_mean": float(ad_probs.mean()), "next_visit_ad_prob_std": float(ad_probs.std()),
        }

    return {
        "status": "projected",
        "current_cdr_sb": current_cdr, "threshold": CDR_SB_AD_THRESHOLD,
        "projected_months_to_threshold_mean": float(np.mean(finite)),
        "projected_months_ci": [float(np.percentile(finite, CI_LOW_PCT)), float(np.percentile(finite, CI_HIGH_PCT))],
        "frac_mc_passes_declining": frac_declining,
        "next_visit_ad_prob_mean": float(ad_probs.mean()), "next_visit_ad_prob_std": float(ad_probs.std()),
    }


def build_projection_examples(model, test_df, norm_stats, device, max_examples=6):
    """Picks a handful of test MCI subjects with >=2 visits for the demo."""
    examples = []
    ds = ADTrajectoryDataset(test_df, norm_stats)
    candidates = [(sid, g) for sid, g in ds.subjects
                  if g["cdr_sb_raw"].notna().any() and len(g) >= 2 and g.iloc[-1]["diagnosis"] in ("MCI", "AD")]
    candidates = candidates[:max_examples]

    for sid, g in candidates:
        idx = [i for i, (s, _) in enumerate(ds.subjects) if s == sid][0]
        item = ds[idx]
        batch = collate_fn([item])
        result = project_time_to_threshold(model, batch, device)
        result["subject_id"] = sid
        result["n_visits_observed"] = int(item["length"])
        examples.append(result)
    return examples


# ---------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: evaluate trajectory model on test set")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/trajectory_best.pt")
    parser.add_argument("--output", type=str, default="outputs/phase2_comparison_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, norm_stats = load_splits(args.data_dir)

    model = load_model(args.checkpoint, device)

    test_ds = ADTrajectoryDataset(test_df, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    next_visit_results = evaluate_next_visit(model, test_loader, device)
    lstm_probs, lstm_labels = next_visit_results["mci_to_ad_probs"], next_visit_results["mci_to_ad_labels"]
    lstm_conversion_auc = float(roc_auc_score(lstm_labels, lstm_probs)) if len(np.unique(lstm_labels)) > 1 else None
    lstm_conversion_acc = float(accuracy_score(lstm_labels, (lstm_probs >= 0.5).astype(int))) if len(lstm_labels) else None

    rf_results = cross_sectional_conversion_baseline(train_df, test_df)
    converter_subset_results = evaluate_converter_subset(model, test_loader, device)

    slope_results = evaluate_slope_regression(model, test_loader, device)

    projection_examples = build_projection_examples(model, test_df, norm_stats, device)

    report = {
        "next_visit_diagnosis": next_visit_results["overall"],
        "mci_to_ad_conversion": {
            "trajectory_lstm": {
                "auc": lstm_conversion_auc, "accuracy": lstm_conversion_acc,
                "n_test_positions": int(len(lstm_labels)), "n_test_conversions": int(lstm_labels.sum()) if len(lstm_labels) else 0,
            },
            "cross_sectional_rf_baseline": {
                "auc": rf_results["auc"], "accuracy": rf_results["accuracy"],
                "n_test_positions": rf_results["n_test_positions"], "n_test_conversions": rf_results["n_test_conversions"],
            },
            "caveat": ("Only 3 MCI->AD conversions in the test split (2 in val, 17 in train). "
                       "AUC here is a methodology demo reproducing the plan's cited LSTM-vs-cross-sectional-RF "
                       "comparison, not a validated result -- treat as illustrative, same spirit as Phase 1's "
                       "honest calibration note."),
        },
        "converter_subset_vs_persistence_baseline": converter_subset_results,
        "cdr_sb_slope_regression": slope_results,
        "cdr_sb_ad_threshold_used": CDR_SB_AD_THRESHOLD,
        "cdr_sb_ad_threshold_derivation": ("Midpoint of MCI's 75th percentile CDR-SB (3.15) and AD's 25th "
                                            "percentile CDR-SB (4.395) on the TRAIN split -- a cohort-specific "
                                            "escalation trigger, not a clinical cutoff."),
        "projection_examples": projection_examples,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Phase 2 test-set results ===")
    print(f"Next-visit diagnosis: acc {next_visit_results['overall']['accuracy']:.3f}, "
          f"macro-F1 {next_visit_results['overall']['macro_f1']:.3f} "
          f"(n={next_visit_results['overall']['n_positions']})")
    print(f"MCI->AD conversion AUC: LSTM {lstm_conversion_auc} vs cross-sectional RF {rf_results['auc']} "
          f"(n_test_conversions={rf_results['n_test_conversions']} -- small-sample caveat applies)")
    print(f"CDR-SB slope MAE: model {slope_results['model_mae']:.3f} vs naive-zero-change "
          f"{slope_results['naive_zero_change_baseline_mae']:.3f}")
    if converter_subset_results.get("n_converter_positions", 0) > 0:
        c = converter_subset_results
        print(f"Converter subset (n={c['n_converter_positions']}): model macro-F1 "
              f"{c['model']['macro_f1']:.3f} vs persistence-baseline macro-F1 {c['persistence_baseline']['macro_f1']:.3f}")
    print(f"\nWrote outputs/phase2_comparison_report.json with {len(projection_examples)} projection examples.")


if __name__ == "__main__":
    main()
