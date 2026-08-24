"""
evaluate_uncertainty.py
--------------------------
Phase 4 test-set evaluation, same honest-comparison style as Phases 1-3.
Runs on the frozen Phase 1 `CrossAttentionFusion` checkpoint (no retraining
needed -- dropout was already there; Module D is purely an *inference-time*
addition, per the plan's practical build).

Three things:

1. **Deterministic-pass calibration** (dropout off, one forward pass per
   example, the "bare risk number" Module D is meant to replace) -- ECE
   before and after fitting a single temperature-scaling scalar on val
   logits (Guo et al., 2017), applied to test.

2. **MC Dropout calibration** (30 stochastic passes, dropout left on,
   probabilities averaged) -- ECE on the averaged probabilities, for
   comparison against the deterministic pass. The plan's own caveat is
   tested directly here: raw MC Dropout is NOT automatically better-
   calibrated than a single deterministic pass; averaging can help or
   hurt confidence calibration independent of whether it captures useful
   epistemic signal. We report both honestly rather than assuming MC
   Dropout is a calibration fix on its own.

3. **Aleatoric/epistemic decomposition** on every test example (see
   uncertainty.py), and a worked demo: the test examples with the highest
   epistemic uncertainty (model doesn't know -- ordering another test is
   the plan's own prescription) vs. highest aleatoric uncertainty (data
   itself is ambiguous -- more tests may not resolve it) are pulled out
   with their predicted probabilities for the judge-facing walkthrough.
"""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import ADFusionDataset, collate_fn, load_splits, LABEL_NAMES
from model import CrossAttentionFusion
from uncertainty import mc_predict, decompose_uncertainty, expected_calibration_error, TemperatureScaler


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CrossAttentionFusion(embed_dim=ckpt["embed_dim"], n_classes=3, dropout=ckpt.get("dropout", 0.2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


def to_device(batch, device):
    batch["modality_features"] = {k: v.to(device) for k, v in batch["modality_features"].items()}
    batch["modality_mask"] = {k: v.to(device) for k, v in batch["modality_mask"].items()}
    batch["static_features"] = batch["static_features"].to(device)
    batch["label"] = batch["label"].to(device)
    return batch


def collect_deterministic_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def collect_mc_uncertainty(model, loader, device, n_passes=30):
    all_mean_probs, all_total, all_aleatoric, all_epistemic, all_labels = [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        probs_stack = mc_predict(model, batch, n_passes=n_passes)
        decomp = decompose_uncertainty(probs_stack)
        all_mean_probs.append(decomp["mean_probs"].cpu())
        all_total.append(decomp["total"].cpu())
        all_aleatoric.append(decomp["aleatoric"].cpu())
        all_epistemic.append(decomp["epistemic"].cpu())
        all_labels.append(batch["label"].cpu())
    return {
        "mean_probs": torch.cat(all_mean_probs), "total": torch.cat(all_total),
        "aleatoric": torch.cat(all_aleatoric), "epistemic": torch.cat(all_epistemic),
        "labels": torch.cat(all_labels),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, norm_stats = load_splits("data/raw")

    val_ds = ADFusionDataset(val_df, norm_stats)
    test_ds = ADFusionDataset(test_df, norm_stats)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = load_model("checkpoints/fusion_best.pt", device)

    # --- 1. Deterministic-pass calibration, before/after temperature scaling ---
    val_logits, val_labels = collect_deterministic_logits(model, val_loader, device)
    test_logits, test_labels = collect_deterministic_logits(model, test_loader, device)

    test_probs_raw = torch.softmax(test_logits, dim=-1)
    test_conf_raw, test_pred_raw = test_probs_raw.max(dim=-1)
    correct_raw = (test_pred_raw == test_labels).numpy().astype(np.float32)
    ece_raw, bins_raw = expected_calibration_error(test_conf_raw.numpy(), correct_raw)
    acc_raw = float(accuracy_score(test_labels.numpy(), test_pred_raw.numpy()))

    scaler = TemperatureScaler()
    fitted_T = scaler.fit(val_logits, val_labels)
    with torch.no_grad():
        test_probs_scaled = torch.softmax(scaler(test_logits), dim=-1)
    test_conf_scaled, test_pred_scaled = test_probs_scaled.max(dim=-1)
    correct_scaled = (test_pred_scaled == test_labels).numpy().astype(np.float32)
    ece_scaled, bins_scaled = expected_calibration_error(test_conf_scaled.numpy(), correct_scaled)

    # --- 2. MC Dropout calibration ---
    mc = collect_mc_uncertainty(model, test_loader, device, n_passes=30)
    mc_conf, mc_pred = mc["mean_probs"].max(dim=-1)
    correct_mc = (mc_pred == mc["labels"]).numpy().astype(np.float32)
    ece_mc, bins_mc = expected_calibration_error(mc_conf.numpy(), correct_mc)
    acc_mc = float(accuracy_score(mc["labels"].numpy(), mc_pred.numpy()))

    # --- 3. Aleatoric/epistemic worked examples ---
    epistemic_np = mc["epistemic"].numpy()
    aleatoric_np = mc["aleatoric"].numpy()
    order_epistemic = np.argsort(-epistemic_np)[:5]
    order_aleatoric = np.argsort(-aleatoric_np)[:5]

    def describe(idx):
        probs = mc["mean_probs"][idx].numpy()
        return {
            "predicted": LABEL_NAMES[int(mc_pred[idx])],
            "true": LABEL_NAMES[int(mc["labels"][idx])],
            "correct": bool(mc_pred[idx] == mc["labels"][idx]),
            "probs": {LABEL_NAMES[c]: float(probs[c]) for c in range(3)},
            "total_uncertainty": float(mc["total"][idx]),
            "aleatoric": float(aleatoric_np[idx]),
            "epistemic": float(epistemic_np[idx]),
        }

    high_epistemic_examples = [describe(i) for i in order_epistemic]
    high_aleatoric_examples = [describe(i) for i in order_aleatoric]

    report = {
        "deterministic_pass": {
            "accuracy": acc_raw, "ece": ece_raw, "reliability_bins": bins_raw,
        },
        "temperature_scaled": {
            "fitted_temperature": fitted_T, "ece": ece_scaled, "reliability_bins": bins_scaled,
            "note": "Temperature fit by NLL minimization on VAL logits only, then applied to test.",
        },
        "mc_dropout": {
            "n_passes": 30, "accuracy": acc_mc, "ece": ece_mc, "reliability_bins": bins_mc,
            "mean_total_uncertainty": float(mc["total"].mean()),
            "mean_aleatoric": float(mc["aleatoric"].mean()),
            "mean_epistemic": float(mc["epistemic"].mean()),
            "note": ("MC Dropout accuracy/ECE use the averaged softmax probability across 30 stochastic "
                     "passes. Compare its ECE to the deterministic-pass ECE above -- MC averaging is NOT "
                     "assumed to improve calibration on its own, per the plan's caveat; we report both so "
                     "the comparison is checkable rather than asserted."),
        },
        "high_epistemic_uncertainty_examples": high_epistemic_examples,
        "high_aleatoric_uncertainty_examples": high_aleatoric_examples,
        "clinical_framing": ("High EPISTEMIC uncertainty = the model itself is unsure (passes disagree) -- "
                              "the plan's prescription is to escalate testing for these cases (this is the "
                              "signal Phase 3's reward-shaping hook, tried below, is meant to consume). High "
                              "ALEATORIC uncertainty = passes agree but the prediction itself is a low-"
                              "confidence blend across classes -- more testing may not resolve genuinely "
                              "borderline pathology, and the plan flags this as its own clinically meaningful "
                              "case rather than a modeling failure."),
    }

    with open("outputs/phase4_uncertainty_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Phase 4 test-set results ===")
    print(f"Deterministic pass:  acc {acc_raw:.3f}  ECE {ece_raw:.3f}")
    print(f"Temperature-scaled:  T={fitted_T:.3f}  ECE {ece_scaled:.3f}  (val-fit, applied to test)")
    print(f"MC Dropout (30 pass): acc {acc_mc:.3f}  ECE {ece_mc:.3f}")
    print(f"Mean uncertainty -- total {mc['total'].mean():.3f} | aleatoric {mc['aleatoric'].mean():.3f} "
          f"| epistemic {mc['epistemic'].mean():.3f}")
    print("\nWrote outputs/phase4_uncertainty_report.json")


if __name__ == "__main__":
    main()
