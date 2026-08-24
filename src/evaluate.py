"""
evaluate.py
-----------
Loads both trained checkpoints (baseline + fusion), evaluates on the
held-out TEST split (never seen during training or model selection),
and writes a head-to-head comparison report. Also breaks out performance
specifically on visits where at least one modality was missing, since
that's the population the fusion model's missing-token design is meant to
help with — the baseline is expected to degrade more on those visits.
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from dataset import ADFusionDataset, collate_fn, load_splits, LABEL_NAMES, NormalizationStats
from model import ConcatBaselineMLP, CrossAttentionFusion


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt["model_type"] == "baseline":
        model = ConcatBaselineMLP(n_classes=len(LABEL_NAMES))
    else:
        model = CrossAttentionFusion(embed_dim=ckpt["embed_dim"], n_classes=len(LABEL_NAMES),
                                      dropout=ckpt.get("dropout", 0.2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    norm_stats = NormalizationStats()
    norm_stats.means = ckpt["norm_stats"]["means"]
    norm_stats.stds = ckpt["norm_stats"]["stds"]
    return model, norm_stats


def get_predictions(model, loader, device):
    all_preds, all_labels, any_missing = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch["modality_features"] = {k: v.to(device) for k, v in batch["modality_features"].items()}
            mask_cpu = {k: v.clone() for k, v in batch["modality_mask"].items()}
            batch["modality_mask"] = {k: v.to(device) for k, v in batch["modality_mask"].items()}
            batch["static_features"] = batch["static_features"].to(device)

            logits = model(batch)
            preds = logits.argmax(dim=-1).cpu().numpy()

            missing_any = np.zeros(len(preds), dtype=bool)
            for k, v in mask_cpu.items():
                missing_any |= (~v.numpy())

            all_preds.append(preds)
            all_labels.append(batch["label"].numpy())
            any_missing.append(missing_any)

    return (np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(any_missing))


def report_for(preds, labels, name):
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_NAMES))))
    report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred), order CN/MCI/AD:")
    print(cm)
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist(),
            "per_class": report}


def main():
    parser = argparse.ArgumentParser(description="Phase 1: evaluate + compare baseline vs fusion model")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="outputs/phase1_comparison_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_df, _ = load_splits(args.data_dir)  # norm_stats loaded per-checkpoint instead

    results = {}
    for model_type in ["baseline", "fusion"]:
        model, norm_stats = load_model(f"{args.checkpoint_dir}/{model_type}_best.pt", device)
        test_ds = ADFusionDataset(test_df, norm_stats)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        preds, labels, any_missing = get_predictions(model, test_loader, device)

        overall = report_for(preds, labels, f"{model_type.upper()} — full test set")
        results[model_type] = {"overall": overall}

        if any_missing.sum() > 0:
            missing_report = report_for(preds[any_missing], labels[any_missing],
                                          f"{model_type.upper()} — visits with >=1 missing modality (n={any_missing.sum()})")
            results[model_type]["missing_modality_subset"] = missing_report

        if (~any_missing).sum() > 0:
            complete_report = report_for(preds[~any_missing], labels[~any_missing],
                                           f"{model_type.upper()} — visits with all modalities present (n={(~any_missing).sum()})")
            results[model_type]["complete_modality_subset"] = complete_report

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD SUMMARY (test set)")
    print("=" * 60)
    for model_type in ["baseline", "fusion"]:
        o = results[model_type]["overall"]
        print(f"{model_type:>10s}: accuracy={o['accuracy']:.3f}  macro-F1={o['macro_f1']:.3f}")

    if "missing_modality_subset" in results["baseline"] and "missing_modality_subset" in results["fusion"]:
        b = results["baseline"]["missing_modality_subset"]["macro_f1"]
        fu = results["fusion"]["missing_modality_subset"]["macro_f1"]
        print(f"\nOn visits with missing modalities specifically:")
        print(f"  baseline macro-F1: {b:.3f}")
        print(f"  fusion   macro-F1: {fu:.3f}  (delta: {fu - b:+.3f})")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report written -> {args.output}")


if __name__ == "__main__":
    main()
