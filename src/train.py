"""
train.py
--------
Trains either the ConcatBaselineMLP or the CrossAttentionFusion model on the
Phase 0 subject-level splits. Uses class-weighted cross-entropy since
diagnosis classes are imbalanced (AD is the minority class, ~16% baseline
in the synthetic cohort — realistic for ADNI too).
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from dataset import ADFusionDataset, collate_fn, load_splits, LABEL_MAP, LABEL_NAMES
from model import ConcatBaselineMLP, CrossAttentionFusion
from losses import FocalLoss


def compute_class_weights(df, device):
    counts = df["diagnosis"].map(LABEL_MAP).value_counts().sort_index()
    counts = counts.reindex(range(len(LABEL_NAMES)), fill_value=1)
    weights = (1.0 / counts.values.astype(np.float32))
    weights = weights / weights.sum() * len(LABEL_NAMES)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch["modality_features"] = {k: v.to(device) for k, v in batch["modality_features"].items()}
            batch["modality_mask"] = {k: v.to(device) for k, v in batch["modality_mask"].items()}
            batch["static_features"] = batch["static_features"].to(device)
            labels = batch["label"].to(device)

            logits = model(batch)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def train_model(model_type: str, epochs: int, lr: float, batch_size: int,
                 embed_dim: int, seed: int, data_dir: str, checkpoint_dir: str,
                 dropout: float = 0.2, weight_decay: float = 1e-5,
                 checkpoint_name: str = None, write_history: bool = True,
                 report_epoch_callback=None, quiet: bool = False,
                 use_focal_loss: bool = False, focal_gamma: float = 2.0):
    """checkpoint_name lets callers (e.g. optuna_sweep.py) write trial
    checkpoints to a scratch path instead of clobbering the canonical
    `{model_type}_best.pt` on every trial. report_epoch_callback(epoch,
    val_f1) is used by optuna_sweep.py to report intermediate values for
    pruning."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, norm_stats = load_splits(data_dir)
    train_ds = ADFusionDataset(train_df, norm_stats)
    val_ds = ADFusionDataset(val_df, norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if model_type == "baseline":
        model = ConcatBaselineMLP(n_classes=len(LABEL_NAMES))
    elif model_type == "fusion":
        model = CrossAttentionFusion(embed_dim=embed_dim, n_classes=len(LABEL_NAMES), dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    class_weights = compute_class_weights(train_df, device)
    # Phase 5C: FocalLoss with gamma=0 is mathematically identical to plain
    # weighted CE (see losses.sanity_check), so this branch only changes
    # behavior when use_focal_loss is explicitly requested.
    if use_focal_loss:
        criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    history = []
    ckpt_name = checkpoint_name or f"{model_type}_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })
        if not quiet:
            print(f"[{model_type}] epoch {epoch:02d} | train loss {train_loss:.3f} acc {train_acc:.3f} f1 {train_f1:.3f} "
                  f"| val loss {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}")

        if report_epoch_callback is not None:
            report_epoch_callback(epoch, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if model_type == "fusion":
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "norm_stats": norm_stats.to_dict(),
                }, f"{checkpoint_dir}/{ckpt_name}")
            else:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type,
                    "embed_dim": embed_dim,
                    "norm_stats": norm_stats.to_dict(),
                }, f"{checkpoint_dir}/{ckpt_name}")

    if write_history:
        with open(f"outputs/{model_type}_train_history.json", "w") as f:
            json.dump(history, f, indent=2)

    if not quiet:
        print(f"\nBest val macro-F1 for {model_type}: {best_val_f1:.3f}")
    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 fusion or baseline model")
    parser.add_argument("--model_type", choices=["baseline", "fusion"], required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--use_focal_loss", action="store_true",
                         help="Phase 5C: use FocalLoss instead of plain weighted cross-entropy.")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    args = parser.parse_args()

    train_model(args.model_type, args.epochs, args.lr, args.batch_size,
                args.embed_dim, args.seed, args.data_dir, args.checkpoint_dir,
                dropout=args.dropout, weight_decay=args.weight_decay,
                use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma)


if __name__ == "__main__":
    main()
