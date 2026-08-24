"""
optuna_sweep.py
----------------
Phase 1's own named stretch goal: "Optuna for quick hyperparameter sweeps
if time allows." Tunes the CrossAttentionFusion model (the novel model,
not the baseline -- the baseline exists purely as the strawman fusion is
measured against, so tuning it isn't the point).

Search space: embed_dim, learning rate, batch size, dropout, weight decay
-- the knobs that actually move a small cross-attention model's val
macro-F1 on a ~1.5k-row dataset. Each trial reuses train.py's own
train_model() (not a re-implementation of the training loop) so the
tuned model is trained by literally the same code path as every other
checkpoint in this project, just with different hyperparameters and a
scratch checkpoint path per trial so trials don't clobber each other or
the canonical checkpoint mid-sweep.

Model selection: median-pruning on val macro-F1 reported every epoch,
so trials that are clearly worse than the trial-so-far median get killed
early instead of running to completion -- this is what makes an Optuna
sweep meaningfully faster than a manual grid search, not just automated.

After the sweep: the best trial's hyperparameters are used to retrain a
FINAL model for the full epoch budget (more epochs than any individual
sweep trial gets, since pruned/short trials are for *comparing* configs,
not for producing the deployable model). That final model overwrites
checkpoints/fusion_best.pt -- the canonical Phase 1 fusion checkpoint --
and outputs/phase1_optuna_report.json records the untuned baseline's own
val/test macro-F1 (already on disk from the original run_phase1.py run)
alongside the tuned result, so the delta is checkable, not asserted.
"""

import argparse
import json
import shutil

import optuna
import torch
from torch.utils.data import DataLoader

from dataset import ADFusionDataset, collate_fn, load_splits, LABEL_NAMES
from model import CrossAttentionFusion
from train import train_model, run_epoch
import torch.nn as nn


def objective_factory(args, trial_epochs):
    def objective(trial: optuna.Trial):
        embed_dim = trial.suggest_categorical("embed_dim", [16, 24, 32, 48, 64])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        def report_cb(epoch, val_f1):
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        best_val_f1 = train_model(
            model_type="fusion", epochs=trial_epochs, lr=lr, batch_size=batch_size,
            embed_dim=embed_dim, seed=args.seed, data_dir=args.data_dir,
            checkpoint_dir="checkpoints", dropout=dropout, weight_decay=weight_decay,
            checkpoint_name=f"_optuna_trial_{trial.number}.pt", write_history=False,
            report_epoch_callback=report_cb, quiet=True,
        )
        return best_val_f1

    return objective


def evaluate_on_test(checkpoint_path, data_dir, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CrossAttentionFusion(embed_dim=ckpt["embed_dim"], n_classes=len(LABEL_NAMES),
                                  dropout=ckpt.get("dropout", 0.2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_df, norm_stats_train = load_splits(data_dir)
    from dataset import NormalizationStats
    norm_stats = NormalizationStats()
    norm_stats.means = ckpt["norm_stats"]["means"]
    norm_stats.stds = ckpt["norm_stats"]["stds"]

    test_ds = ADFusionDataset(test_df, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    _, acc, f1 = run_epoch(model, test_loader, optimizer=None, criterion=criterion, device=device, train=False)
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Optuna sweep for Phase 1 CrossAttentionFusion")
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--trial_epochs", type=int, default=25,
                         help="Epochs per sweep trial (short -- for COMPARING configs, not the final model).")
    parser.add_argument("--final_epochs", type=int, default=60,
                         help="Epochs for retraining the winning config into the deployable checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Snapshot the untuned fusion checkpoint's test performance BEFORE
    # overwriting it, so the report shows a real before/after, not a
    # number pulled from an old JSON file that might be stale.
    pretune_acc, pretune_f1 = evaluate_on_test("checkpoints/fusion_best.pt", args.data_dir, device)
    print(f"Pre-tuning fusion checkpoint: test acc {pretune_acc:.3f}, macro-F1 {pretune_f1:.3f}")
    shutil.copy("checkpoints/fusion_best.pt", "checkpoints/fusion_pretune_best.pt")

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=8)
    study = optuna.create_study(direction="maximize", pruner=pruner,
                                 sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective_factory(args, args.trial_epochs), n_trials=args.n_trials)

    print(f"\nBest trial: #{study.best_trial.number}  val macro-F1={study.best_value:.3f}")
    print(f"Best params: {study.best_params}")

    # Clean up scratch trial checkpoints.
    import glob, os
    for f in glob.glob("checkpoints/_optuna_trial_*.pt"):
        os.remove(f)

    # Retrain the winning config for the full epoch budget -> canonical checkpoint.
    best = study.best_params
    final_val_f1 = train_model(
        model_type="fusion", epochs=args.final_epochs, lr=best["lr"], batch_size=best["batch_size"],
        embed_dim=best["embed_dim"], seed=args.seed, data_dir=args.data_dir,
        checkpoint_dir="checkpoints", dropout=best["dropout"], weight_decay=best["weight_decay"],
        checkpoint_name="fusion_best.pt", write_history=True,
    )
    posttune_acc, posttune_f1 = evaluate_on_test("checkpoints/fusion_best.pt", args.data_dir, device)
    print(f"\nPost-tuning fusion checkpoint (final, {args.final_epochs} epochs): "
          f"val macro-F1 {final_val_f1:.3f} | test acc {posttune_acc:.3f}, macro-F1 {posttune_f1:.3f}")

    trials_summary = [
        {
            "number": t.number, "state": str(t.state), "value": t.value,
            "params": t.params,
        }
        for t in study.trials
    ]

    report = {
        "search_space": {
            "embed_dim": [16, 24, 32, 48, 64], "lr": "loguniform(1e-4, 5e-3)",
            "batch_size": [16, 32, 64], "dropout": "uniform(0.1, 0.4)",
            "weight_decay": "loguniform(1e-6, 1e-3)",
        },
        "n_trials_requested": args.n_trials, "n_trials_completed": len(study.trials),
        "n_trials_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "trial_epochs": args.trial_epochs, "final_epochs": args.final_epochs,
        "best_trial_number": study.best_trial.number,
        "best_params": study.best_params,
        "best_trial_val_macro_f1": study.best_value,
        "pretune": {
            "checkpoint": "checkpoints/fusion_pretune_best.pt (default hyperparams: embed_dim=32, lr=1e-3, "
                          "batch_size=32, dropout=0.2, weight_decay=1e-5, 40 epochs -- the run_phase1.py defaults)",
            "test_accuracy": pretune_acc, "test_macro_f1": pretune_f1,
        },
        "posttune": {
            "checkpoint": "checkpoints/fusion_best.pt (overwritten with tuned config, retrained for "
                           f"{args.final_epochs} epochs)",
            "val_macro_f1": final_val_f1,
            "test_accuracy": posttune_acc, "test_macro_f1": posttune_f1,
        },
        "delta_test_macro_f1": posttune_f1 - pretune_f1,
        "trials": trials_summary,
        "note": ("Optuna optimizes VAL macro-F1 only (never test, to avoid tuning-set leakage into the "
                 "number reported as generalization performance); test macro-F1 above is reported purely "
                 "for the before/after comparison, exactly like Phase 1's baseline-vs-fusion comparison. "
                 "On this ~1.5k-row synthetic cohort, sweep gains are expected to be modest -- the plan's "
                 "own honest-caveat theme (shared decline_rate latent capping how much any architecture "
                 "choice can move the needle) applies here too."),
    }

    with open("outputs/phase1_optuna_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote outputs/phase1_optuna_report.json")
    print(f"Delta test macro-F1 (tuned - untuned): {report['delta_test_macro_f1']:+.3f}")
    print("checkpoints/fusion_best.pt now holds the TUNED model.")
    print("checkpoints/fusion_pretune_best.pt holds the original (untuned) model, kept for the record.")


if __name__ == "__main__":
    main()
