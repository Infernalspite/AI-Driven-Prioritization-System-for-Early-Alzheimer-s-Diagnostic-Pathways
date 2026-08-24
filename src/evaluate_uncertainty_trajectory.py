"""
evaluate_uncertainty_trajectory.py
-------------------------------------
Phase 4's own text names both models: "Add dropout layers to Module A/B's
networks ... run 20-50 stochastic forward passes at inference." The first
Phase 4 pass only covered Module A (`evaluate_uncertainty.py`). This closes
that gap for Module B (`TrajectoryLSTM`, Phase 2): same MC Dropout / ECE /
aleatoric-epistemic machinery from `uncertainty.py`, applied to the
next-visit diagnosis head instead of Module A's single-visit classifier.

TrajectoryLSTM.forward returns per-TIMESTEP logits (B, T, n_classes), only
some of which are valid targets (a subject's own last visit has no
"next" to predict, and padded timesteps aren't real). This script's
`mc_predict_trajectory` mirrors `uncertainty.mc_predict` but flattens to
just the valid (next_valid==True) positions before computing entropy/ECE,
so the comparison to Module A's per-example numbers is apples-to-apples.
"""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataset import LABEL_NAMES
from longitudinal_dataset import ADTrajectoryDataset, collate_fn, load_splits
from trajectory_model import TrajectoryLSTM
from train_trajectory import to_device
from uncertainty import entropy, expected_calibration_error


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TrajectoryLSTM(embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


def mc_predict_trajectory(model, loader, device, n_passes=30):
    """Runs N stochastic passes (dropout active) over the whole test set,
    keeping only valid next-visit positions, and returns per-position
    probability stacks + labels for uncertainty decomposition."""
    all_valid_probs_per_pass = [[] for _ in range(n_passes)]
    all_labels = []
    det_logits_per_batch = []  # single deterministic pass, for comparison

    batches = list(loader)

    model.eval()
    with torch.no_grad():
        for batch in batches:
            batch = to_device(batch, device)
            next_logits, _ = model(batch)
            valid = batch["next_valid"]
            mask_np = valid.cpu().numpy().astype(bool)
            det_logits_per_batch.append(next_logits.cpu().numpy()[mask_np])
            all_labels.append(batch["next_diagnosis"].cpu().numpy()[mask_np])

    model.train()  # dropout ON for MC passes
    with torch.no_grad():
        for p in range(n_passes):
            for batch in batches:
                batch = to_device(batch, device)
                next_logits, _ = model(batch)
                probs = torch.softmax(next_logits, dim=-1)
                valid = batch["next_valid"]
                mask_np = valid.cpu().numpy().astype(bool)
                all_valid_probs_per_pass[p].append(probs.cpu().numpy()[mask_np])
    model.eval()

    det_logits = np.concatenate(det_logits_per_batch, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs_stack = np.stack([np.concatenate(pass_list, axis=0) for pass_list in all_valid_probs_per_pass], axis=0)
    # probs_stack: (n_passes, n_positions, n_classes)
    return det_logits, probs_stack, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_df, norm_stats = load_splits("data/raw")

    test_ds = ADTrajectoryDataset(test_df, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = load_model("checkpoints/trajectory_best.pt", device)

    det_logits, probs_stack, labels = mc_predict_trajectory(model, test_loader, device, n_passes=30)

    # deterministic pass
    det_probs = torch.softmax(torch.tensor(det_logits), dim=-1).numpy()
    det_conf = det_probs.max(axis=-1)
    det_pred = det_probs.argmax(axis=-1)
    det_correct = (det_pred == labels).astype(np.float32)
    det_acc = float(accuracy_score(labels, det_pred))
    ece_det, bins_det = expected_calibration_error(det_conf, det_correct)

    # MC dropout
    probs_t = torch.tensor(probs_stack)  # (n_passes, n_positions, n_classes)
    mean_probs = probs_t.mean(dim=0)
    mc_conf, mc_pred = mean_probs.max(dim=-1)
    mc_pred_np = mc_pred.numpy()
    mc_correct = (mc_pred_np == labels).astype(np.float32)
    mc_acc = float(accuracy_score(labels, mc_pred_np))
    ece_mc, bins_mc = expected_calibration_error(mc_conf.numpy(), mc_correct)

    total_unc = entropy(mean_probs)
    aleatoric_unc = entropy(probs_t).mean(dim=0)
    epistemic_unc = (total_unc - aleatoric_unc).clamp(min=0.0)

    report = {
        "module": "TrajectoryLSTM (Module B) next-visit diagnosis head",
        "n_test_positions": int(len(labels)),
        "deterministic_pass": {"accuracy": det_acc, "ece": ece_det, "reliability_bins": bins_det},
        "mc_dropout": {
            "n_passes": 30, "accuracy": mc_acc, "ece": ece_mc, "reliability_bins": bins_mc,
            "mean_total_uncertainty": float(total_unc.mean()),
            "mean_aleatoric": float(aleatoric_unc.mean()),
            "mean_epistemic": float(epistemic_unc.mean()),
        },
        "note": ("Closes the Phase 4 'Module A/B' gap -- the first Phase 4 pass only covered Module A "
                 "(evaluate_uncertainty.py). Same MC Dropout / ECE / aleatoric-epistemic machinery, "
                 "applied here to the longitudinal next-visit head instead of the single-visit classifier."),
    }

    with open("outputs/phase4_uncertainty_trajectory_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Phase 4 (Module B) test-set results ===")
    print(f"Deterministic pass: acc {det_acc:.3f}  ECE {ece_det:.3f}")
    print(f"MC Dropout (30 pass): acc {mc_acc:.3f}  ECE {ece_mc:.3f}")
    print(f"Mean uncertainty -- total {total_unc.mean():.3f} | aleatoric {aleatoric_unc.mean():.3f} "
          f"| epistemic {epistemic_unc.mean():.3f}")
    print("\nWrote outputs/phase4_uncertainty_trajectory_report.json")


if __name__ == "__main__":
    main()
