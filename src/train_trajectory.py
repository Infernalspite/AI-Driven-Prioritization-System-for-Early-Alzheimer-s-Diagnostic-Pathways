"""
train_trajectory.py
--------------------
Trains TrajectoryLSTM on the Phase 0 subject splits (same subjects as
Phase 1, so results are directly comparable). Two losses, each masked to
only the sequence positions where the target actually exists:

  - class-weighted cross-entropy on next-visit diagnosis, masked to
    (this is a real visit) AND (a next visit exists for this subject) --
    a subject's own last visit and all padded positions contribute 0.
  - Smooth-L1 regression on CDR-SB trajectory slope, same masking (plus
    requiring CDR-SB be present at both ends, always true in this cohort).

Total loss = cls_loss + slope_loss_weight * reg_loss. Model selection uses
val next-visit macro-F1 (matches Phase 1's selection criterion), the same
way the plan's cited LSTM study is evaluated primarily on conversion
prediction quality.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from dataset import LABEL_MAP, LABEL_NAMES
from longitudinal_dataset import ADTrajectoryDataset, collate_fn, load_splits
from trajectory_model import TrajectoryLSTM
from losses import FocalLoss
from oversampling import SubjectLevelOversampler


def compute_class_weights(df, device):
    counts = df["diagnosis"].map(LABEL_MAP).value_counts().sort_index()
    counts = counts.reindex(range(len(LABEL_NAMES)), fill_value=1)
    weights = 1.0 / counts.values.astype(np.float32)
    weights = weights / weights.sum() * len(LABEL_NAMES)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def to_device(batch, device):
    batch["modality_features"] = {k: v.to(device) for k, v in batch["modality_features"].items()}
    batch["modality_mask"] = {k: v.to(device) for k, v in batch["modality_mask"].items()}
    for key in ["static_features", "seq_mask", "cur_diagnosis", "next_diagnosis",
                "next_valid", "cdr_slope", "slope_valid", "cdr_raw", "visit_month", "length"]:
        batch[key] = batch[key].to(device)
    return batch


def run_epoch(model, loader, optimizer, class_weights, device, train: bool, slope_loss_weight: float,
              use_focal_loss: bool = False, focal_gamma: float = 2.0):
    model.train() if train else model.eval()
    # Phase 5C: FocalLoss(gamma=0) is mathematically identical to plain
    # weighted CE, so this only changes behavior when explicitly requested.
    if use_focal_loss:
        ce = FocalLoss(weight=class_weights, gamma=focal_gamma, reduction="none")
    else:
        ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    smooth_l1 = nn.SmoothL1Loss(reduction="none")

    total_loss = total_cls_loss = total_reg_loss = 0.0
    n_cls = n_reg = 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = to_device(batch, device)
            next_logits, slope_pred = model(batch)  # (B,T,3), (B,T)
            B, T, C = next_logits.shape

            valid = batch["next_valid"]
            cls_loss_full = ce(next_logits.reshape(B * T, C), batch["next_diagnosis"].reshape(B * T)).reshape(B, T)
            cls_loss = (cls_loss_full * valid.float()).sum() / valid.float().sum().clamp(min=1)

            slope_valid = batch["slope_valid"]
            reg_loss_full = smooth_l1(slope_pred, batch["cdr_slope"])
            reg_loss = (reg_loss_full * slope_valid.float()).sum() / slope_valid.float().sum().clamp(min=1)

            loss = cls_loss + slope_loss_weight * reg_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n_valid = int(valid.sum().item())
            n_slope_valid = int(slope_valid.sum().item())
            total_loss += loss.item() * max(n_valid, 1)
            total_cls_loss += cls_loss.item() * max(n_valid, 1)
            total_reg_loss += reg_loss.item() * max(n_slope_valid, 1)
            n_cls += n_valid
            n_reg += n_slope_valid

            mask_np = valid.detach().cpu().numpy().astype(bool)
            all_preds.append(next_logits.argmax(dim=-1).detach().cpu().numpy()[mask_np])
            all_labels.append(batch["next_diagnosis"].detach().cpu().numpy()[mask_np])

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro") if len(all_labels) else 0.0

    return {
        "loss": total_loss / max(n_cls, 1),
        "cls_loss": total_cls_loss / max(n_cls, 1),
        "reg_loss": total_reg_loss / max(n_reg, 1),
        "next_diag_acc": acc,
        "next_diag_f1": macro_f1,
        "n_cls_positions": n_cls,
        "n_reg_positions": n_reg,
    }


def train_model(epochs, lr, batch_size, embed_dim, hidden_dim, n_layers, seed,
                 data_dir, checkpoint_dir, slope_loss_weight,
                 use_focal_loss: bool = False, focal_gamma: float = 2.0,
                 use_oversampling: bool = False, oversample_factor: int = 4):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df, norm_stats = load_splits(data_dir)
    train_ds = ADTrajectoryDataset(train_df, norm_stats)
    val_ds = ADTrajectoryDataset(val_df, norm_stats)

    # Phase 5C: subject-level oversampling of converter trajectories. Only
    # touches the TRAIN loader -- val/test stay at their natural (unresampled)
    # distribution, since oversampling the evaluation set would inflate
    # apparent performance rather than measure it honestly.
    if use_oversampling:
        sampler = SubjectLevelOversampler(train_ds, oversample_factor=oversample_factor, seed=seed)
        print(f"Subject-level oversampling active: {sampler.report()}")
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        sampler = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = TrajectoryLSTM(embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    class_weights = compute_class_weights(train_df, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, optimizer, class_weights, device, True, slope_loss_weight,
                                   use_focal_loss=use_focal_loss, focal_gamma=focal_gamma)
        val_metrics = run_epoch(model, val_loader, optimizer, class_weights, device, False, slope_loss_weight,
                                 use_focal_loss=use_focal_loss, focal_gamma=focal_gamma)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"epoch {epoch:02d} | train f1 {train_metrics['next_diag_f1']:.3f} "
              f"reg_loss {train_metrics['reg_loss']:.3f} | "
              f"val f1 {val_metrics['next_diag_f1']:.3f} reg_loss {val_metrics['reg_loss']:.3f}")

        if val_metrics["next_diag_f1"] > best_val_f1:
            best_val_f1 = val_metrics["next_diag_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "embed_dim": embed_dim, "hidden_dim": hidden_dim, "n_layers": n_layers,
                "norm_stats": norm_stats.to_dict(),
            }, f"{checkpoint_dir}/trajectory_best.pt")

    with open("outputs/trajectory_train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val next-visit macro-F1: {best_val_f1:.3f}")
    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 trajectory LSTM")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--slope_loss_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--use_focal_loss", action="store_true",
                         help="Phase 5C: use FocalLoss instead of plain weighted cross-entropy.")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--use_oversampling", action="store_true",
                         help="Phase 5C: subject-level oversampling of converter trajectories.")
    parser.add_argument("--oversample_factor", type=int, default=4)
    args = parser.parse_args()

    train_model(args.epochs, args.lr, args.batch_size, args.embed_dim, args.hidden_dim,
                args.n_layers, args.seed, args.data_dir, args.checkpoint_dir, args.slope_loss_weight,
                use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                use_oversampling=args.use_oversampling, oversample_factor=args.oversample_factor)


if __name__ == "__main__":
    main()
