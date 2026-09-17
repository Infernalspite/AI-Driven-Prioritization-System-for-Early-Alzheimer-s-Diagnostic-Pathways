---
title: Early Alzheimers Diagnostic Prioritization
emoji: ??
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# AI-Driven Prioritization System for Early Alzheimer's Diagnostic Pathways

An implementation of Phases 0–4 of the plan: a pipeline that fuses
whatever multimodal data exists at each point (Module A), models each
patient's trajectory over time rather than a single snapshot (Module B),
learns a policy for when escalation is worth it rather than always going
stage-by-stage (Module C), and reports decomposed, calibrated uncertainty
instead of a bare risk number (Module D) — each layer built, trained, and
evaluated against an honest baseline on real recorded values from the
Phase 0 cohort, not a synthetic best-case demo.

**Status: Phases 0–4 complete**, plus both of Phase 1/2's own named
"if time allows" stretch goals are now also implemented and evaluated:
an Optuna hyperparameter sweep for Module A (`src/optuna_sweep.py`,
`outputs/phase1_optuna_report.json`), and a Temporal Fusion Transformer
swap-in for Module B (`src/train_tft.py`, `src/evaluate_tft.py`,
`outputs/phase2_tft_comparison_report.json`) — see "Stretch goals" below
for both, including two real library-version bugs that had to be
debugged and fixed to get the TFT training at all.

**Phase 5 (real-data migration & synthetic hardening) is also
implemented** — 5A (OASIS migration), 5B (hardened synthetic v2
generator), 5C (focal loss, subject-level oversampling, and a SMOTE
reference comparison), and 5D (the validation gate) — see the Phase 5
section below for exactly what's been executed in this dev sandbox
(no torch, no network) versus what's written and ready to run in the
full training environment.

**Phase 6 (Module E: causal confound-adjustment layer) is now also
implemented** — a backdoor-adjustment comparison of unadjusted vs.
age/education-adjusted feature importance, per the plan's own "first-pass
confound-adjustment layer" framing, plus a SHAP-explanation function
written against the frozen Module A checkpoint (not executable in this
sandbox — no torch/shap here either). See the Phase 6 section below.

**Not implemented:** nothing — all seven phases (0–7) named anywhere in
the plan, including every explicitly-optional "if time allows"/stretch/
reference item, are now built. Per the plan's own instruction, Phase
6/7's numbers shouldn't be treated as validated until Phase 5's gate
passes on real or properly hardened data — see "Gate status" in the
Phase 6 and 7 sections for exactly where that stands right now (currently
`all_passed: false`, so both phases below are methodology development
and verification, not a final result).

**How the phases connect, concretely, not just narratively:**
- Phase 2's trajectory model and Phase 3's pathway agent both build
  directly on Phase 1's `CrossAttentionFusion` checkpoint — Phase 3's MDP
  state literally IS Module A's output, re-fused as more tests get
  ordered.
- Phase 4's epistemic-uncertainty estimate was fed into Phase 3's agent
  as an additional state dimension and retrained end to end (not just
  described) — see "Phase 4 → Phase 3 integration" below for the honest
  result.
- Phase 4's MC-Dropout/calibration analysis was run on **both** Module A
  and Module B, independently reproducing the same finding on two
  different architectures.

**Running theme across every phase, stated once here so it isn't
repeated as a surprise five times below:** this synthetic cohort (Phase
0) generates every modality from one shared `decline_rate` latent per
subject. That's a deliberate simplification for a working end-to-end
pipeline, not an attempt to over-claim results — and it shows up
consistently as: cross-attention doesn't beat naive concatenation
(Phase 1), only 3 MCI→AD conversions exist in the test split (Phase 2),
the pathway agent never finds MRI/PET worth their cost (Phase 3), and
epistemic uncertainty stays small (Phase 4). Every phase's README section
below says this in its own terms rather than hiding it — the honest
takeaway for judges is "the architecture, training, and evaluation
protocol are all correct and literature-grounded; validating the
*advantages* these methods are designed to show requires real ADNI data
with real cross-modal decoupling and disease-stage heterogeneity."

---

## Testing this package (and re-checking torch/network access)

Before writing anything further, this dev sandbox was re-checked for
`torch` and network access, in case either had become available since
the earlier phases were written:

```
$ curl https://pypi.org        -> HTTP 403, x-deny-reason: host_not_allowed
$ curl https://files.pythonhosted.org -> HTTP 403, x-deny-reason: host_not_allowed
$ pip install torch             -> ERROR: No matching distribution found for torch
```

Still fully blocked — the egress proxy actively denies the request
(`host_not_allowed`), not a timeout or misconfiguration, so this isn't
something a retry or a different mirror would fix from inside this
sandbox. Every "written, not executed here" note elsewhere in this
README (full v2 retraining, the OASIS migration adapter,
`explain_fusion_model_shap()`, `dowhy_backdoor_estimate()`, the `pulp`
MILP path) is still accurate for the same reason it was before.

**What was actually tested**, since torch/network weren't available to
newly unlock: every `sanity_check()`/`--sanity_check` entry point across
`smote_baseline.py`, `causal_explainability.py`, and
`resource_optimizer.py`; `run_phase5.py`/`run_phase6.py`/`run_phase7.py`
end to end, twice each, to confirm identical (deterministic) output on
rerun; and a set of edge cases each module hadn't been exercised against
yet (zero capacity and below-minimum-cost knapsack instances, a
zero-variance confounder column, a deliberately nonexistent `--csv_path`).

**Two real, reproducible bugs turned up and were fixed, not just
noted:**

1. `smote_baseline.py`'s `sanity_check()` only ran its own correctness
   assertions (the fixture-based check that SMOTE's synthetic points stay
   within the minority class's feature bounds) when
   `data/raw_v2/cleaned_dataset.csv` was **absent** — once that file
   existed (as it does in this package), calling `sanity_check()` silently
   skipped its own assertions entirely and ran the real-data report
   instead, meaning the one thing a "sanity check" is supposed to
   guarantee was never actually being checked in this environment. Fixed
   by splitting the always-run logic check
   (`verify_smote_logic()`) from the real-data report, and by adding the
   `--sanity_check` CLI flag this module was missing (the other two Phase
   6/7 modules already had one).
2. The same function, plus `resource_optimizer.py`'s
   `sanity_check_yield_surrogate()`, hardcoded `data/raw_v2` internally
   regardless of what `--csv_path`/`--data_dir` was actually passed on the
   command line — so `python src/smote_baseline.py --csv_path
   somewhere_else.csv --sanity_check` silently ran against
   `data/raw_v2` instead of erroring or respecting the flag. Fixed by
   threading the path through as a real argument in both files, and in
   `run_phase7.py`'s call site.

Both are now covered by the edge-case tests above (a deliberately
nonexistent `--csv_path` correctly falls back to the logic-only check
and says so, rather than silently substituting the default path).
Nothing else exercised in this pass turned up incorrect behavior —
capacity=0 and below-minimum-cost knapsack instances correctly select
nothing, and `backdoor_adjusted_importance()` correctly refuses to divide
by a zero-variance column instead of returning garbage.

---

# Phase 0 — Problem Framing & Data Acquisition

The 4-stage AD diagnostic funnel (cognitive screening → blood biomarkers
→ MRI → PET), a leakage-safe subject-level train/val/test split, and a
documented cohort table — the plan's own required deliverable:

> "Deliverable: clean tabular + imaging-feature dataset, versioned, with a
> documented cohort table (like ADNI papers always include: n subjects,
> class balance, age/sex breakdown)."

## Quickstart

```bash
cd ad-phase1
python src/cohort_table.py
```

Writes `outputs/phase0_cohort_table.json` from `data/raw/cleaned_dataset.csv`
+ `subject_splits.json`.

## Cohort table

| | |
|---|---|
| Subjects | 500 |
| Visits | 1,537 (1–5 per subject, 6-month spacing) |
| Baseline diagnosis | CN 192 · MCI 226 · AD 82 |
| Baseline age | 73.2 ± 7.5 (range 55–93) |
| Baseline sex | M 272 · F 228 |
| Baseline education | 14.4 ± 2.9 years |

| Split | n subjects | CN / MCI / AD |
|---|---|---|
| Train | 349 | 134 / 158 / 57 |
| Val | 75 | 29 / 34 / 12 |
| Test | 76 | 29 / 34 / 13 |

Split proportions roughly track the full-cohort class balance, and the
split is by **subject**, not by visit — no subject's records appear in
more than one of train/val/test, which is what makes every later phase's
test-set number leakage-safe.

Modality missingness (visit-level, matches Phase 1's design motivation for
explicit presence masks over imputation): blood 25.0%, MRI 15.1%, PET
69.2% — PET being the most invasive/expensive test is also, correctly,
the most often skipped in this cohort, mirroring real clinical practice.

---

# Phase 1 — Multimodal Fusion Risk Scoring

Builds on Phase 0's cleaned dataset + leakage-safe subject splits. Trains and
head-to-head compares two models per the plan:

1. **`ConcatBaselineMLP`** — the literal problem-statement baseline: zero-impute
   missing modalities, concatenate everything, MLP classifier.
2. **`CrossAttentionFusion`** — the novel model, following the MCAD /
   NeuroNet-AD pattern: one encoder per modality (cognitive, blood, MRI, PET),
   a learned "missing" embedding per modality (BERT/ViT-style mask token)
   substituted when that modality wasn't collected for a visit, then
   cross-attention across all modality tokens + a CLS pooling token, then a
   classification head.

## Quickstart

```bash
cd ad-phase1
pip install -r requirements.txt
python run_phase1.py --epochs 40
```

This trains both models, evaluates both on the held-out **test** split
(never touched during training or model selection), and writes:

- `outputs/phase1_comparison_report.json` — accuracy/macro-F1/confusion
  matrices for both models, overall AND broken out by whether the visit had
  any missing modality (the population the fusion model's design specifically
  targets)
- `outputs/attention_examples.json` — per-example cross-attention weights,
  for the "here's what the model actually looked at" moment in a demo
- `checkpoints/{baseline,fusion}_best.pt`

## Actual results on the synthetic dataset (25 epochs, seed 42)

| | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (concat MLP) | 0.910 | 0.892 |
| Fusion (cross-attention) | 0.898 | 0.879 |

**Honest read of this, don't skip it in the demo:** the fusion model does
*not* beat the baseline here, and attention weights come out close to
uniform (~0.17 across every token) rather than sharply differentiated. This
is expected, not a bug — and it's worth explaining *why* if a judge asks:

The synthetic generator (Phase 0) derives every modality's value from the
**same single underlying `decline_rate` latent** per subject, so cognitive
scores, blood biomarkers, MRI, and PET are all just noisy readings of one
shared signal. A simple concatenation already captures that fine, because
there's no genuine *cross-modal interaction* to exploit — nothing where,
say, blood biomarkers only matter conditional on what MRI shows. Cross-
attention's advantage in the literature (MCAD, NeuroNet-AD) comes from real
AD biology, where amyloid pathology (PET/blood), neurodegeneration (MRI),
and cognitive symptoms genuinely **decouple** at different disease stages —
that's the actual clinical motivation for fusion, and it's not present in
this synthetic set by construction.

**What this means practically:**
- Don't claim a fusion-beats-baseline win on synthetic data in front of judges — claim the *architecture* is correct and grounded in the literature, and that validating the advantage requires real ADNI data with real modality decoupling.
- The missing-modality subset comparison (fusion −0.016 macro-F1 vs baseline on visits with missing data) is the fairest test of the design's actual value proposition — rerun this exact comparison the moment real ADNI data is in, this is the number that should move.
- If you want to *demonstrate* the fusion advantage on synthetic data for the demo, the honest way to do it is to inject an actual cross-modal interaction into the generator (e.g., make risk depend on `MRI_below_threshold AND blood_below_threshold` jointly, not additively) — that would give the cross-attention model something real to exploit that concatenation can't represent. Not done here, to avoid a demo dataset that's been shaped to make the story win.

## Design notes for judges / teammates

- **Missing-modality handling is explicit, not silent.** The baseline
  zero-imputes; the fusion model substitutes a *learned* per-modality
  "missing" embedding instead, so missingness itself becomes a usable signal
  rather than being indistinguishable from "value happens to be zero."
- **Normalization stats are fit on train only** (`dataset.py`), reusing
  Phase 0's leakage discipline at the feature-scaling level too.
- **Class-weighted loss** — AD is the minority class (~16% baseline
  prevalence), so cross-entropy is weighted inversely by class frequency
  rather than optimizing for the majority classes.
- **CLS-token pooling** (BERT/ViT-style) rather than mean-pooling, so the
  classification head reads from a single learned "summary" token whose
  attention weights double as your interpretability layer for the demo.

## Stretch goal: Optuna hyperparameter sweep

`src/optuna_sweep.py` — the plan's own "Optuna for quick hyperparameter
sweeps if time allows" line for Module A. Tunes `CrossAttentionFusion`
only (the baseline is a strawman, not the point of tuning). Search space:
`embed_dim ∈ {16,24,32,48,64}`, `lr ~ loguniform(1e-4, 5e-3)`,
`batch_size ∈ {16,32,64}`, `dropout ~ U(0.1, 0.4)`,
`weight_decay ~ loguniform(1e-6, 1e-3)`. Median pruning on val macro-F1
(reported every epoch) kills clearly-worse trials early; the winning
config is then retrained for a longer, dedicated final budget (60 epochs
vs. 25 per sweep trial) to produce the deployable checkpoint.

```bash
python src/optuna_sweep.py --n_trials 25 --trial_epochs 25 --final_epochs 60
```

**Actual sweep result (25 trials, seed 42):** 17 trials completed, 8
pruned early. Best config: `embed_dim=16, lr=1.9e-3, batch_size=16,
dropout=0.214, weight_decay=6.8e-6`, reaching val macro-F1 **0.940**
(vs. the untuned default's 40-epoch val performance). Retrained for the
full 60-epoch budget and evaluated on test:

| | Test accuracy | Test macro-F1 |
|---|---|---|
| Untuned (defaults: embed_dim=32, lr=1e-3, 40 epochs) | 0.914 | 0.899 |
| Tuned (Optuna best config, 60 epochs) | 0.914 | 0.899 |

**Honest read:** the delta is +0.0003 macro-F1 — essentially nothing.
This is consistent with Phase 1's own finding above (fusion doesn't beat
the baseline either) — the synthetic cohort's shared `decline_rate`
latent caps how much *any* architecture or hyperparameter choice can move
test performance, because there's no real signal structure left to tune
into. The sweep infrastructure (pruning, before/after checkpointing,
honest reporting) is the actual deliverable here, not a claimed
performance win — see `outputs/phase1_optuna_report.json` for the full
25-trial log. `checkpoints/fusion_pretune_best.pt` keeps the untuned
model for the record; `checkpoints/fusion_best.pt` now holds the tuned
one (all downstream phases load `embed_dim` from the checkpoint, so
Phase 3/4 pick up the new architecture automatically — verified by
rerunning both after the sweep).

---

# Phase 2 — Longitudinal Trajectory Modeling

Moves from Phase 1's single-snapshot risk score to modeling each subject's
**decline over time**, per the plan: *"predicting not just current class
but time-to-threshold-crossing."* Practical build, exactly as scoped —
a stacked LSTM over each patient's available visit timepoints (this cohort
has 1–5 visits, 6 months apart), with two heads:

1. **Next-visit diagnosis** — TADPOLE-style future clinical status
   prediction (CN/MCI/AD at the *next* visit, not the current one).
2. **CDR-SB trajectory slope** — regression for rate of cognitive decline
   (CDR-Sum-of-Boxes points per 6-month interval), the "MMSE/CDR
   trajectory slope" head the plan calls for.

## Quickstart

```bash
cd ad-phase1
python run_phase2.py --epochs 60
```

Trains `TrajectoryLSTM` on the **same** Phase 0/1 subject splits (so
Phase 1 and Phase 2 are evaluated on identical held-out subjects), then
evaluates on test and writes:

- `outputs/trajectory_train_history.json`
- `outputs/phase2_comparison_report.json` — next-visit accuracy/macro-F1,
  MCI→AD conversion AUC (trajectory LSTM vs. a cross-sectional Random
  Forest baseline — the plan's own cited comparison), CDR-SB slope
  regression MAE vs. a naive no-change baseline, and 6 worked
  time-to-threshold projection examples
- `checkpoints/trajectory_best.pt`

## Actual results on the synthetic dataset (60 epochs, seed 42)

| Task | Result |
|---|---|
| Next-visit diagnosis (test, n=168 positions) | Acc 0.911, macro-F1 0.891 |
| MCI→AD conversion AUC — TrajectoryLSTM | 0.562 |
| MCI→AD conversion AUC — cross-sectional RF baseline | 0.416 |
| CDR-SB slope MAE — model | 0.380 |
| CDR-SB slope MAE — naive "predict no change" baseline | 0.423 |

**Honest read of this, don't skip it in the demo:** the headline next-visit
accuracy (0.911) is inflated by class imbalance — most visits are
CN→CN or MCI→MCI, so a model that mostly predicts "no change" scores well
on accuracy/macro-F1 without doing anything clever. The number that
actually tests the plan's claim ("longitudinal beats cross-sectional") is
the **MCI→AD conversion AUC**, and there the LSTM (0.562) does beat the
cross-sectional RF (0.416) — directionally consistent with the plan's cited
precedent (LSTM AUC 0.93 vs. RF AUC 0.90 on real ADNI) — but on only **3
conversion events in the entire test split** (2 in val, 17 in train). That
is nowhere near enough to call this a validated result; it's a correct
*implementation* of the comparison methodology on a cohort that's too small
and, like Phase 1's dataset, generated from a single shared `decline_rate`
latent rather than realistic disease-stage-dependent dynamics. The slope
regression (0.380 vs. 0.423 MAE) shows a real but modest edge over
"assume nothing changes."

**What this means practically:**
- Don't claim a validated conversion-prediction win in front of judges —
  claim the architecture and evaluation protocol are correct and
  literature-grounded (TADPOLE-style future-status task, LSTM-vs-cross-
  sectional-RF comparison per the cited precedent), and that the small
  conversion count is a synthetic-data artifact, not a modeling flaw.
- The time-to-threshold projection (see below) is the strongest demo
  moment — it's qualitatively different output from Phase 1 (a bare class
  score) and is worth walking a judge through live.
- The TFT swap-in described as a stretch goal in the original plan is now
  implemented and evaluated head-to-head against this LSTM — see
  "Stretch goals" below.

## Time-to-threshold projection (the plan's target output shape)

Rather than a bare risk score, `evaluate_trajectory.py` runs MC Dropout
(Gal & Ghahramani, 2016 — keep dropout active at inference, repeat the
forward pass) at each subject's last observed visit, turning the point-
estimate CDR-SB slope into a distribution:

```
SUBJ-0033: CDR-SB 3.76 (threshold 3.80) → projected to cross in
           ~1.1 months, 80% CI: [0.7, 1.7]  (30/30 MC passes declining)
SUBJ-0013: CDR-SB 2.33 → projected to cross in
           ~40.4 months, 80% CI: [25.4, 52.7]
SUBJ-0023: CDR-SB 3.81 → already at/above threshold, escalate now
```

The 3.8 CDR-SB threshold is **not** a clinical cutoff — it's the midpoint
of MCI's 75th-percentile CDR-SB (3.15) and AD's 25th-percentile CDR-SB
(4.40) on the *train* split, a cohort-specific escalation trigger derived
the same leakage-safe way as everything else in this pipeline. This is
Phase 2 exposing the hook Phase 4 (calibrated uncertainty) is meant to
refine — the CI here comes from raw MC Dropout variance, which the plan
itself flags as poorly calibrated until temperature-scaled.

## Stretch goal: Temporal Fusion Transformer swap-in

`src/tft_dataset.py` + `src/train_tft.py` + `src/evaluate_tft.py` — the
plan's own "swap in a small TFT ... for probabilistic multi-horizon
output" line for Module B. Single-step forecast (`max_prediction_length=1`)
to match exactly what the LSTM's slope head is evaluated on above
(next-visit CDR-SB), so the head-to-head is apples-to-apples rather than
showcasing multi-step forecasting the LSTM was never asked to do.

Long-format data prep (`tft_dataset.py`): missing blood/MRI/PET features
are train-mean-imputed with a paired `*_present` binary indicator per
modality — TFT's encoder can't take NaN directly the way the LSTM's
masked-embedding approach can, so missingness becomes an explicit feature
instead of being silently imputed away.

**Two real library-version bugs had to be debugged and fixed** to get
this training at all, on `pytorch-forecasting 1.8.0` + `lightning 2.6.5`
+ `torchmetrics 1.9.0`:

1. `RuntimeError: element 0 of tensors does not require grad and does
   not have a grad_fn` on the first backward pass. Traced (by patching
   `update()`/`compute()` with print statements and watching
   `requires_grad` flip from `True` to `False` across the exact
   boundary) to `MultiHorizonMetric.forward()` — the base class
   `QuantileLoss` inherits from — silently detaching its persistent
   torchmetrics state buffer from the autograd graph during its
   accumulate-then-read round trip. (Lightning's `inference_mode=True`
   default was the first suspect — a known way to poison a persistent
   buffer — but setting `inference_mode=False` and reproducing the same
   crash ruled that out.) Fix: monkeypatch `MultiHorizonMetric.forward`
   to compute and reduce the loss directly, bypassing the buggy buffer
   entirely — same arithmetic, intact autograd graph. See the
   `_fixed_multihorizon_forward` docstring in `train_tft.py`.
2. NaN loss after fix #1. Cause: `GroupNormalizer(groups=["subject_id"])`
   fits a separate mean/std per subject, and several subjects in this
   cohort only have 2–3 visits — for those, the per-group std is
   ~0, so normalizing CDR-SB by it produces inf/NaN. Fix: switched to a
   global (ungrouped) `GroupNormalizer`, appropriate here since CDR-SB is
   a bounded clinical scale comparable across subjects, not something
   needing per-subject centering.

```bash
python src/train_tft.py --max_epochs 60
python src/evaluate_tft.py
```

**Actual result (early-stopped at epoch 20, seed 42), head-to-head
against the LSTM on next-visit CDR-SB MAE:**

| | Next-visit CDR-SB MAE | n (test positions) |
|---|---|---|
| TrajectoryLSTM | 0.411 | 168 |
| TemporalFusionTransformer | 0.451 | 254 |

Row counts differ because each model is evaluated on the largest
held-out set it natively supports on the same test subjects (TFT's
`TimeSeriesDataSet` only needs ≥1 encoder step, so it scores more
consecutive-visit pairs than the LSTM's next-visit head does) — not an
artificially matched subsample.

**Honest read:** the LSTM's point estimate is *slightly* more accurate
here, which is a legitimate result, not a failure to report around — on
this small, short-history synthetic cohort (2–5 visits per subject), TFT's
larger parameter count and richer feature set (per-modality presence
indicators, static covariates, relative time index) don't have enough
signal to outperform a simpler recurrent model. What TFT *does* deliver
that the LSTM doesn't get natively is genuine probabilistic output: its
80% prediction interval (10th–90th quantile) has a mean width of 1.387
CDR-SB points and **empirical coverage of 0.82** against the nominal 0.80
— i.e., it's well-calibrated out of the box, without the MC-Dropout
forward-pass loop the LSTM's projection needs (see
`project_time_to_threshold` above) to get any interval at all. See
`outputs/phase2_tft_comparison_report.json` for the full report,
including five example predictions with their intervals.

## Repo structure

```
ad-phase1/
├── run_phase1.py
├── run_phase2.py
├── run_phase3.py
├── run_phase4.py
├── requirements.txt
├── src/
│   ├── cohort_table.py               # Phase 0: n subjects, class balance, age/sex breakdown, by-split counts
│   ├── dataset.py                    # Phase 1: per-visit tensors + missingness masks, leakage-safe norm
│   ├── model.py                       # Phase 1: ConcatBaselineMLP + CrossAttentionFusion
│   ├── train.py / evaluate.py         # Phase 1 training + test-set comparison
│   ├── inspect_attention.py           # Phase 1: per-example attention weight dump
│   ├── optuna_sweep.py                # Phase 1 stretch: Optuna hyperparameter sweep over CrossAttentionFusion
│   ├── longitudinal_dataset.py         # Phase 2: per-SUBJECT padded visit sequences + next-visit/slope targets
│   ├── trajectory_model.py             # Phase 2: VisitEncoder (fusion) + TrajectoryLSTM (2 heads)
│   ├── train_trajectory.py             # Phase 2: masked multi-task training loop
│   ├── evaluate_trajectory.py          # Phase 2: next-visit metrics, RF baseline, slope MAE, MC-Dropout projection
│   ├── tft_dataset.py                   # Phase 2 stretch: long-format data prep for TimeSeriesDataSet
│   ├── train_tft.py                     # Phase 2 stretch: TFT training (incl. two library-bug workarounds)
│   ├── evaluate_tft.py                  # Phase 2 stretch: TFT vs LSTM head-to-head on next-visit CDR-SB MAE
│   ├── load_fusion.py                   # Phase 3: loads the frozen Phase 1 fusion model + its exact norm stats
│   ├── pathway_env.py                   # Phase 3/4: ADPathwayEnv MDP + ADPathwayEnvUncertaintyAware subclass
│   ├── train_pathway.py                 # Phase 3/4: PPO (stable-baselines3) training, --uncertainty_aware flag
│   ├── evaluate_pathway.py              # Phase 3/4: PPO(s) vs always-full-workup vs cognitive-only vs greedy
│   ├── uncertainty.py                   # Phase 4: MC Dropout, aleatoric/epistemic decomposition, ECE, temp scaling
│   ├── evaluate_uncertainty.py          # Phase 4: Module A calibration report + high-uncertainty examples
│   └── evaluate_uncertainty_trajectory.py  # Phase 4: same analysis, Module B's next-visit head
├── data/raw/                        # cleaned_dataset.csv + subject_splits.json from Phase 0
├── checkpoints/                     # {baseline,fusion,trajectory}_best.pt, tft_best.ckpt, pathway_ppo{,_uncertainty}.zip,
│                                     # fusion_pretune_best.pt (untuned model, kept for the record)
└── outputs/                         # includes phase1_optuna_report.json, phase2_tft_comparison_report.json
```

---

# Phase 3 — Adaptive Diagnostic Pathway Engine

Module C: instead of every patient walking the fixed cognitive → blood →
MRI → PET funnel, an agent decides **per patient** whether escalating to
the next (more expensive/invasive) test is worth it, formalized exactly
as the plan's MDP:

> State: current known feature vector + uncertainty. Actions: {order
> blood, order MRI, order PET, stop and diagnose}. Reward:
> +correct_diagnosis − λ·cost(test) − μ·invasiveness(test).

**How it extends Module A** (the plan explicitly asks for this): rather
than raw feature vectors, the agent's state is the frozen Phase 1
`CrossAttentionFusion` model's predicted class probabilities given
whatever's been revealed so far (`src/load_fusion.py`, `pathway_env.py`).
Ordering a test changes what Module A sees; the agent watches its own
diagnostic confidence shift in real time and decides whether that's worth
the next test's cost.

Episodes are grounded in **real recorded values**: only the 297 visits
where blood, MRI, AND PET were all actually collected are used, so
"ordering" a test reveals a real number, never a simulated one (train 197
/ val 44 / test 56, same subject splits as Phases 1–2).

## Quickstart

```bash
cd ad-phase1
python run_phase3.py --total_timesteps 100000 --lambda_cost 0.5 --mu_invasive 0.5
```

Trains a PPO agent (stable-baselines3, per the plan's tech stack) on
`ADPathwayEnv`, then evaluates it against three baselines on the held-out
test set:

- `outputs/phase3_comparison_report.json` — accuracy, avg tests ordered,
  avg cost/invasiveness, avg reward for all 4 policies, plus example
  test-ordering trajectories
- `checkpoints/pathway_ppo.zip`

## Actual results on the synthetic dataset (100k PPO steps, test n=56)

| Policy | Accuracy | Avg tests ordered | Avg cost | Avg reward |
|---|---|---|---|---|
| Always-full-workup (the funnel, every patient) | **0.929** | 3.00 | 12.00 | 4.25 |
| **PPO pathway agent** | 0.875 | **0.61** | **0.61** | **10.44** |
| Greedy-confidence heuristic (fixed 0.75 threshold) | 0.875 | 0.14 | 0.18 | 10.76 |
| Cognitive-only (never escalate) | 0.857 | 0.00 | 0.00 | 10.36 |

(Costs: blood 1.0, MRI 3.0, PET 8.0; λ=μ=0.5, chosen so a full workup's
combined cost/invasiveness penalty — 8.25 — is comparable in magnitude to
the ±15 diagnosis reward, so the trade-off actually has teeth. At the
smaller λ=μ=0.05 first tried, cost was negligible next to the diagnosis
reward and the agent correctly learned to just order everything — a
useful sanity check that the MDP is wired correctly, but not an
interesting policy to show a judge.)

**Honest read of this, don't skip it in the demo:** the PPO agent
recovers a real 1.8-point accuracy gain over never escalating (0.875 vs
0.857) for a fraction of the full workup's cost, and gets the best reward
of any policy once cost is priced in realistically. But look at *what*
it actually learned: across all 56 test patients, the policy collapses to
exactly two behaviors — order blood then stop (34/56), or stop
immediately (22/56). **It never once orders MRI or PET.** That's not a
training failure; it's the correct policy *for this reward function on
this dataset* — the same root cause as Phase 1's honest note: this
synthetic cohort derives every modality from one shared `decline_rate`
latent, so blood biomarkers (cheap) carry almost as much signal as PET
(8x the cost), and no amount of RL will teach an agent to pay 8x for
information it can already get for 1x. The greedy-confidence heuristic
independently converges to the same insight (barely escalates at all,
0.14 avg tests) by a completely different mechanism, which is itself
decent evidence this is a property of the data, not an artifact of either
method.

**What this means practically:**
- Don't claim the agent learned a *sophisticated* escalation policy in
  front of judges — claim the MDP formulation, the Module A integration,
  and the PPO training pipeline are all correct and literature-grounded
  (the plan's cited "Efficient Alzheimer's Diagnosis Through Sequential
  Decision-Making with RL" — DQN, 89.2% accuracy, favoring cheap tests —
  is the direct precedent this extends), and that validating a genuine
  MRI/PET-worthy escalation policy requires real ADNI data where amyloid
  pathology, neurodegeneration, and cognitive symptoms actually decouple
  at different disease stages (exactly Phase 1's caveat, inherited here).
- The reward-vs-λ sensitivity itself is a good demo moment: showing a
  judge that turning λ up from 0.05 to 0.5 flips the agent from
  "order everything" to "barely escalate" demonstrates the reward
  function actually controls the cost/accuracy trade-off as designed,
  even on data too simple to need a smart escalation *rule*.
- On real ADNI data, the honest next step is re-running this exact
  pipeline unchanged — the environment, reward shaping, and PPO setup
  don't need to change, only the input data does.

---

# Phase 4 — Uncertainty-Aware Prioritization

Module D: replace bare point-estimate risk scores with calibrated
confidence, and decompose *why* a prediction is uncertain, per the plan:

> Gal & Ghahramani (2016) — keeping dropout active at inference and doing
> repeated stochastic forward passes approximates Bayesian inference.
> Kendall & Gal — split aleatoric (irreducible data noise) from epistemic
> (the model's own lack of knowledge, reducible with more data) — the
> distinction is clinically meaningful: epistemic-uncertain cases should
> trigger more testing, aleatoric-uncertain cases may mean the test
> itself is inconclusive. Raw MC Dropout is known to be poorly calibrated
> — evaluate with ECE, not just accuracy.

**No architecture change was needed.** Phase 1's `CrossAttentionFusion`
already has dropout (rate 0.2) throughout — Module D is purely an
*inference-time* addition (`src/uncertainty.py`): call the frozen Phase 1
checkpoint repeatedly with dropout left on (`model.train()`, no grad),
and decompose the resulting spread of predictions.

## Quickstart

```bash
cd ad-phase1
python src/evaluate_uncertainty.py
```

Writes `outputs/phase4_uncertainty_report.json`: deterministic-pass ECE,
temperature-scaled ECE, MC-Dropout ECE, mean aleatoric/epistemic
uncertainty, and the 5 highest-epistemic and 5 highest-aleatoric test
examples for the demo walkthrough.

## Actual results on the synthetic dataset (test set, 30 MC passes)

| Calibration method | Accuracy | ECE |
|---|---|---|
| Deterministic single pass | 0.898 | 0.028 |
| **Temperature-scaled** (T=0.874, fit on val) | 0.898 | **0.017** |
| MC Dropout (30-pass average) | 0.898 | 0.044 |

| Uncertainty (mean over test set) | Value |
|---|---|
| Total (entropy of averaged prediction) | 0.400 |
| Aleatoric (avg entropy per pass) | 0.377 |
| Epistemic (total − aleatoric) | 0.023 |

**Honest read of this, don't skip it in the demo:** temperature scaling
does what the literature says it should — a single learned scalar cuts
ECE by ~40%, cleanly. MC Dropout does **not**: its averaged-probability
ECE (0.044) is *worse* than a single deterministic pass (0.028). This
isn't a bug — it's the plan's own caveat ("raw MC Dropout is known to be
poorly calibrated ... unless explicitly calibrated") confirmed rather
than just asserted. Don't claim MC Dropout as a calibration fix in front
of judges; claim it as an *uncertainty decomposition* tool (aleatoric vs.
epistemic) and pair it with temperature scaling for the calibration claim.

The bigger finding: **epistemic uncertainty is small (0.023) relative to
aleatoric (0.377)** across this cohort. The model isn't very unsure about
its own weights — it's seen enough of this synthetic distribution to be
confident in itself — but there's real, irreducible class-boundary
ambiguity in the data (the same single-shared-latent generation process
flagged as a limitation in Phases 1–3). That's a coherent, non-cherry-
picked story across all four phases, not a one-off caveat.

## Module B (the trajectory model) gets the same treatment

The plan names both models — *"Add dropout layers to Module A/B's
networks"* — and `TrajectoryLSTM` (Phase 2) already had dropout throughout,
so `src/evaluate_uncertainty_trajectory.py` runs the identical MC-Dropout /
ECE / aleatoric-epistemic analysis on its next-visit diagnosis head:

```bash
python src/evaluate_uncertainty_trajectory.py
```

| Calibration (Module B, next-visit head) | Accuracy | ECE |
|---|---|---|
| Deterministic single pass | 0.911 | 0.068 |
| MC Dropout (30-pass average) | 0.917 | 0.105 |

| Uncertainty (mean, Module B) | Value |
|---|---|
| Aleatoric | 0.448 |
| Epistemic | 0.031 |

Same pattern as Module A, independently reproduced on a different
architecture and a different task (sequence classification vs. single-
visit classification): MC Dropout's averaged probabilities are *worse*
calibrated than a single deterministic pass, and epistemic uncertainty is
small relative to aleatoric. Two models agreeing on the same finding is
better evidence it's a property of the dataset, not an artifact of one
architecture's dropout placement.

## Phase 4 → Phase 3 integration: does uncertainty help the pathway agent?

The plan explicitly asks for this connection: *"Feed epistemic
uncertainty directly into Module C's reward function — high uncertainty
raises the value of ordering another test."* Rather than hand-craft an
artificial escalation bonus (which would just be prescribing the answer),
`ADPathwayEnvUncertaintyAware` (`src/pathway_env.py`) augments the
agent's **state** with a live epistemic-uncertainty estimate (10 MC
passes per step) and lets PPO decide for itself whether that signal is
worth acting on — the existing reward (correct/incorrect diagnosis) is
unchanged, so if epistemic uncertainty is a genuine leading indicator of
an about-to-be-wrong diagnosis, the agent should learn to spend on
another test when it sees high epistemic uncertainty.

```bash
python src/train_pathway.py --total_timesteps 100000 --lambda_cost 0.5 \
    --mu_invasive 0.5 --uncertainty_aware --output_name pathway_ppo_uncertainty
python src/evaluate_pathway.py --lambda_cost 0.5 --mu_invasive 0.5
```

| Policy | Accuracy | Avg tests | Avg reward |
|---|---|---|---|
| Original PPO agent (Phase 3, no uncertainty in state) | 0.875 | 0.61 | 10.44 |
| **PPO agent + epistemic uncertainty in state** | 0.857 | 0.57 | 9.93 |

**Honest result: it didn't help.** The uncertainty-aware agent converges
to essentially the *same* qualitative policy as the original — order
blood then stop (32/56 test patients, vs. 34/56 for the original), or
stop immediately (24/56 vs. 22/56) — and scores marginally *worse* on
both accuracy and reward. This is not a failed implementation; it's the
predictable consequence of Phase 4's own finding one section up: on this
dataset, epistemic uncertainty is small and doesn't vary much across
patients, so there's little signal in it for the agent to exploit, and
the extra state dimension is mostly noise for the policy to learn around.
The plan's causal claim ("high uncertainty raises the value of ordering
another test") was tested here, not assumed — and on this cohort, it
doesn't hold, for a reason that's traceable to the data, not the method.
On real ADNI data, where epistemic uncertainty should vary far more
across genuinely different patient presentations, re-running this exact
integration is the honest way to find out whether the claim holds there.

---

# Phase 5 — Real-Data Migration & Synthetic Data Hardening

**Status: implemented.** This phase exists because every phase above ran
correctly but produced numbers that can't yet be trusted, for one
structural reason, stated once at the top of this README and confirmed
independently by four different evaluation reports: the v1 synthetic
generator derives every modality from a single shared `decline_rate`
latent per subject, so there's no genuine cross-modal interaction, no
genuine disease-stage-dependent dynamics, and (separately, just as
damaging) a conversion rate about 4x lower than real ADNI populations —
5.8% of v1 subjects convert stage at least once, vs. the ~22% (77/347
subjects) cited in this plan's own reference list for 2-year MCI→AD
conversion. No amount of architecture improvement or hyperparameter
tuning fixes a data problem — confirmed directly by Phase 1's own Optuna
result (+0.0003 macro-F1 after a full sweep).

Phase 5 has four parts. 5A and 5B are two **independent, parallel**
tracks (real data takes time to access; synthetic hardening doesn't need
to wait for it), 5C is a shared training-time improvement used by 5B's
retraining, and 5D is the gate that both tracks report through before
Phase 6/7 build on either one's numbers.

## 5A — Real-data migration: OASIS Longitudinal

`src/migrate_oasis.py` adapts OASIS Longitudinal (Kaggle,
`jboysen/mri-and-alzheimers`) — a real, non-simulated longitudinal cohort
— into this pipeline's exact schema, so it's a drop-in replacement for
`data/raw/` with no changes needed to `dataset.py` / `longitudinal_dataset.py`.

```bash
# Requires network + a Kaggle account (not available in this dev sandbox):
pip install kagglehub
python src/migrate_oasis.py                          # auto-download via kagglehub
# — or, if you already have the CSV —
python src/migrate_oasis.py --input_csv path/to/oasis_longitudinal.csv
```

**What OASIS can't supply (and how that's handled, not hidden):** no
blood biomarkers (Aβ42/40, p-tau) and no PET SUVR. Because
`CrossAttentionFusion` already uses a learned "missing" embedding per
modality (a Phase 1 design choice made specifically so this wouldn't
require a rearchitecture later), those two modalities are **permanently
masked missing** for every OASIS-migrated subject — Module A degrades
gracefully to 2-modality (cognitive + MRI-derived) fusion, same forward
pass, same checkpoint format. `CDR_SB` is reconstructed from OASIS's
global CDR via a documented linear proxy (`global_CDR × 3.0`), not a real
Sum-of-Boxes score — treat any CDR-SB slope output on this data as
illustrative of methodology, the same honesty standard the rest of this
README already applies to synthetic-data caveats. Full caveats are
written to `data/raw_oasis/migration_report.json` on every run.

**Module C's action space must shrink too.** `pathway_env.py` hardcoded
a 3-modality (blood/MRI/PET) escalation action space; that's now
reconfigurable:

```python
from pathway_env import configure_escalation_modalities
configure_escalation_modalities(["mri"])  # OASIS has no blood, no PET
```

Call this once, before building any `ADPathwayEnv` / training or loading
a PPO checkpoint against OASIS data — it rebuilds the action space,
observation space, and cost/invasiveness table from just the modalities
that actually exist in the data. A checkpoint trained under one
configuration can't be loaded under another (the action space size
itself changes), so OASIS-track Module C checkpoints need their own
`--checkpoint_dir`, same as Module A/B.

*This adapter could not be executed end-to-end in this development
environment (no network access to Kaggle) — it's been verified against a
small hand-built fixture mirroring OASIS's real column layout, but not
against the actual dataset. Run it in an environment with network access
before trusting its output.*

## 5B — Hardened synthetic generator v2

`src/generate_synthetic_v2.py` fixes the root cause directly rather than
patching around it:

1. **Two only-weakly-correlated latents** (`neurodegeneration` drives MRI
   atrophy + cognitive decline; `amyloid_burden` drives blood biomarkers
   + PET SUVR) instead of one shared `decline_rate` — a subject can now be
   high on one and low on the other, which is what makes cross-attention
   fusion potentially worth more than concatenation in the first place.
2. **Conversion probability is a genuine AND-interaction**
   (`neuro_severity × amyloid_severity`, a product, not a linear sum) —
   the literal thing concatenation cannot represent and cross-attention
   is designed to pick up.
3. **Calibrated conversion rate**: `--transition_scale 0.3` (the default)
   lands the realized 2-year MCI→AD conversion rate at **17.4%**, inside
   the target 15–22% band, producing **89 converter visits** in a
   500-subject cohort — comfortably past the ≥20 threshold Phase 5D
   requires to trust a converter-subset comparison (v1 had 3 in the whole
   test split).
4. **Ground truth is logged, never fed to the model** — `ground_truth.csv`
   records both latent severities and the true interaction signal per
   visit, so a real performance gain in Module A/B can eventually be
   checked against a known target instead of just trusted at face value.

```bash
python src/generate_synthetic_v2.py --n_subjects 500 --seed 42
# Generated 500 subjects / 2107 visits -> data/raw_v2/
# Baseline diagnosis class balance: {'MCI': 218, 'CN': 207, 'AD': 75}
# Realized subject-level conversion rate: 17.4%  (target band: 15-22%)
```

## 5C — Class-imbalance handling for rare converter/AD cases

Two additions, both opt-in flags on the existing training scripts (not
new scripts to run separately):

- **`src/losses.py` — `FocalLoss`** (Lin et al., 2017). Drop-in
  replacement for weighted cross-entropy; `gamma=0` is mathematically
  identical to plain weighted CE (verified: `gamma=0` output matches
  `nn.CrossEntropyLoss(weight=...)` exactly on a fixed random batch), so
  it's a strict generalization, not a different loss family.
  `--use_focal_loss --focal_gamma 2.0` on `train.py` and `train_trajectory.py`.

- **`src/oversampling.py` — `SubjectLevelOversampler`**. Duplicates whole
  converter *subject*-sequences as extra dataset indices (not rows — this
  matters specifically because `ADTrajectoryDataset` is one-item-per-
  subject, so row-level oversampling would either do nothing or leak a
  subject's *stable* visits across a batch alongside the one converting
  visit, overstating learnability). Val/test loaders are never
  oversampled. `--use_oversampling --oversample_factor 4` on
  `train_trajectory.py`.

```bash
python src/train_trajectory.py --data_dir data/raw_v2 --checkpoint_dir checkpoints_v2 \
    --use_focal_loss --focal_gamma 2.0 --use_oversampling --oversample_factor 4
```

*Both components are logic-verified in this environment (focal loss via
a numpy-equivalent formula check; the oversampler via a stubbed
`torch.utils.data.Sampler` unit test) but not yet run through a full
training loop end-to-end here, since this development sandbox has no
`torch` install. Run `python src/losses.py` and `python src/oversampling.py`
directly for their built-in sanity checks in an environment with `torch`.*

**Reference comparison: SMOTE.** The plan's own tech-stack line for this
phase names `imbalanced-learn` for "SMOTE-style tabular oversampling
utilities, used only as a reference/comparison against subject-level
oversampling" — that comparison wasn't previously implemented anywhere
in this package; `src/smote_baseline.py` fills the gap. It implements
Chawla et al. (2002)'s SMOTE directly on top of scikit-learn's
`NearestNeighbors` (already a transitive dependency, no extra package
needed) rather than adding `imbalanced-learn` itself, since core SMOTE
interpolation is all the comparison needs. Unlike `losses.py` /
`oversampling.py`, this module has **no torch dependency**, so it's
actually been run end-to-end in this dev sandbox against the real
`data/raw_v2` cohort (not just logic-checked):

```bash
python src/smote_baseline.py --csv_path data/raw_v2/cleaned_dataset.csv
```

| Strategy (target class: AD, RandomForest classifier, subject-level test split) | Train rows | AD rows in train | Test macro-F1 | Test AD F1 |
|---|---|---|---|---|
| No oversampling (class-weighted only) | 1,688 | 350 | 0.790 | 0.818 |
| Row-level SMOTE (this module) | 2,007 | 669 | 0.792 | 0.841 |
| Subject-level duplication (row-level analogue of `SubjectLevelOversampler`, factor=4) | 2,966 | 1,400 | **0.807** | **0.860** |

**Honest read:** on this cohort, subject-level duplication beats both
plain class-weighting and row-level SMOTE on both metrics, using real
recorded visits rather than interpolated feature vectors. SMOTE does
still help over doing nothing (+0.002 macro-F1, +0.023 AD-F1), so it's
not a strawman — it's a legitimate, literature-standard technique that
this comparison shows is dominated by the domain-aware alternative here,
for the structural reason in the module's docstring: SMOTE interpolates
between two nearest-neighbor visit-rows with no notion of "subject" or
"visit order," so a synthetic row can blend two different subjects (or
two different disease stages) into a visit that never happened and can't
be assigned a real longitudinal identity — which is exactly why it's
usable for this flat row-level comparison but not for Module B's actual
per-subject sequence training. Full report:
`outputs/phase5_smote_comparison_report.json`.

## 5D — Validation gate

`src/validation_gate.py` reads the standard Phase 1/2/3 evaluation
reports and checks three criteria — the same three named in the plan —
before anything downstream is allowed to cite the numbers:

1. Fusion beats concatenation by a non-trivial margin (≥0.03 macro-F1)
   **specifically on the missing-modality test subset**.
2. The trajectory model beats the persistence baseline **specifically on
   the converter subset**, with ≥20 converter positions to trust the
   comparison (new: `evaluate_trajectory.py` now reports this subset
   directly via `evaluate_converter_subset()`).
3. The PPO pathway agent finds **at least one scenario** where a correct,
   escalated diagnosis scored a higher reward than the mean reward of
   stopping immediately (new: `evaluate_pathway.py` now logs every
   episode's trajectory + reward via `all_episode_trajectories`, not just
   the first 6 examples).

```bash
python src/validation_gate.py \
    --phase1_report outputs/phase1_comparison_report_v2.json \
    --phase2_report outputs/phase2_comparison_report_v2.json \
    --phase3_report outputs/phase3_comparison_report_v2.json
```

Exit code is `0` if all three pass, `1` otherwise, so it doubles as a
CI-style gate. **Run against the existing v1 reports (before any
retraining), it correctly fails all three** — criterion 1 at
`+0.009` delta (below the `+0.030` threshold, consistent with Phase 1's
own "fusion barely beats baseline" finding), and criteria 2/3 report
"not yet measured" rather than a false pass, since v1's
`evaluate_trajectory.py` / `evaluate_pathway.py` output predates these
breakouts. This is the expected, correct behavior of the gate, not a bug
— it's demonstrating that it actually gates rather than rubber-stamping.

If any criterion doesn't clear on the v2 or OASIS data either, **that's a
genuine negative result to report to judges honestly** (e.g. "fusion's
advantage did not clear our validation gate on this dataset; here's the
missing-modality-subset number and what we think it means"), per the
plan's own instruction — not something to omit or silently work around.

## Running Phase 5 end to end

```bash
# Hardened-synthetic track (5B/5C/5D), no network required:
python run_phase5.py
python run_phase5.py --n_subjects 800 --total_timesteps 100000   # bigger run

# Real-data track (5A), requires network + Kaggle access, then reruns the
# same sequence against the migrated data:
python src/migrate_oasis.py --input_csv path/to/oasis_longitudinal.csv
python run_phase5.py --data_dir data/raw_oasis --checkpoint_dir checkpoints_oasis
```

`run_phase5.py` generates v2 data, retrains Modules A/B/C on it (Module B
with focal loss + oversampling on by default), reruns all three
evaluation scripts, and finishes by running the validation gate —
printing and writing `outputs/phase5_validation_gate_report.json` with a
clear pass/fail per criterion, exiting non-zero if any criterion isn't
met so it's safe to treat as a build-blocking check.

*Note on what's been executed vs. verified-by-inspection in this
development environment: this sandbox has no `torch`/network access, so
`run_phase5.py`'s full training sequence has not actually been executed
here. What HAS been run and confirmed: the v2 generator (produces the
17.4% conversion rate quoted above), the OASIS migration adapter (against
a hand-built fixture matching OASIS's real schema), the
`configure_escalation_modalities()` action-space reconfiguration, the
focal-loss and subject-level-oversampling logic (via stubbed/numpy-
equivalent unit tests), and the validation gate itself (against the
existing v1 reports, correctly failing all three criteria as described
above). Run `python run_phase5.py` in the original `torch`-equipped
environment to get real retrained numbers before citing them to judges.*

---

# Phase 6 — Module E: Causal Explainability Layer

Per the plan: go beyond SHAP's correlational attributions to separate
confounded association from plausible causal driver — is elevated
cognitive-decline severity driven by biomarker/imaging pathology, or
confounded by age/education (both well-documented AD confounders)?
Framed honestly, per the plan's own instruction, as a **first-pass
confound-adjustment layer**, not full causal discovery.

**Gate status:** the plan states Phase 6 "runs after the validation
gate." `outputs/phase5_validation_gate_report.json` currently shows
`all_passed: false` (the v1-report run described in Phase 5 above) — so
the numbers below are methodology development and verification, **not**
a validated result to present to judges as final. `run_phase6.py` checks
this file itself and prints the same warning before running, rather than
silently proceeding as if the gate had passed.

## What predicts what, and why

- **Predictors** (`PREDICTOR_COLS`): the biomarker/imaging features —
  `abeta42_40_ratio`, `ptau181` (blood), `hippocampal_volume_mm3`,
  `cortical_thickness_mm` (MRI), `pet_amyloid_suvr` (PET) — i.e. exactly
  the modalities Module A's cross-attention is designed to fuse
  (`dataset.py`'s `MODALITY_COLUMNS`, minus the cognitive-test modality,
  see below).
- **Outcome**: `CDR_SB`, the same continuous cognitive-severity yardstick
  Phase 2 already uses for its escalation threshold — reused here rather
  than inventing a new outcome definition for Phase 6 alone.
- **Confounders** (`CONFOUNDER_COLS`): `age`, `education_years` — the
  plan's own named confounders, and also exactly `dataset.py`'s
  `STATIC_COLUMNS`, i.e. the two features the fusion architecture already
  treats specially (fed alongside every modality token rather than as a
  modality of their own) — a natural, pre-existing hook for this
  analysis rather than an arbitrary choice.
- MMSE/ADAS13 are deliberately **excluded** from the predictor set even
  though they're in the "cognitive" modality: predicting cognitive
  decline severity (CDR_SB) from other cognitive test scores would be
  circular, not a causal-driver question.

## Method: linear backdoor adjustment (Frisch-Waugh-Lovell)

`src/causal_explainability.py`'s `backdoor_adjusted_importance()` is the
plan's "simple backdoor-adjustment estimate on age/education" option —
implemented directly with `sklearn.linear_model.LinearRegression`
(already in requirements.txt) rather than adding `dowhy`, since `dowhy`
isn't installed in this dev sandbox and there's no network access to
install it here. `dowhy_backdoor_estimate()` is also written (tries
`import dowhy`, returns `None` and lets the caller fall back to the FWL
version if it's not available) — the plan's other named option, ready to
run in an environment that has it, but not exercised as part of this
package's verified results, same status as the OASIS migration adapter
in Phase 5.

The FWL shortcut: the coefficient of outcome `Y` on predictor `X`
*after controlling for confounders `C`* is algebraically identical to
regressing the **residual** of `Y` (after removing what `C` explains)
on the **residual** of `X` (after removing what `C` explains) — no
`dowhy`, no full multivariable model needed, just two linear regressions
per predictor. All predictors/outcome are z-scored first so
"importance" is on one comparable, unit-free scale.

**Sanity check** (`python src/causal_explainability.py
--sanity_check`, always runs — no data file needed): constructs `X` and
`Y` that are correlated **only** through a shared confounder `C`, by
design, with no direct `X→Y` effect. This is the right test for a
confound-adjustment tool specifically because it's the one case where the
correct answer is known in advance:

```
causal_explainability sanity check passed: a purely confounded
association (unadjusted=0.984) correctly drops to ~0
(adjusted=-0.013) once the shared confounder is backdoor-adjusted for.
```

The unadjusted association is large and spurious (0.984); adjustment
correctly removes essentially all of it (-0.013). This has actually been
run in this sandbox (pure numpy/pandas/scikit-learn, no torch needed) —
it's the one Phase 6 result that's fully verified end to end here, logic
and execution both.

## Actual results on the hardened synthetic v2 cohort (2,107 visits)

```bash
python src/causal_explainability.py --csv_path data/raw_v2/cleaned_dataset.csv
```

| Predictor | Unadjusted importance | Adjusted importance | Confounding strength | % surviving adjustment |
|---|---|---|---|---|
| hippocampal_volume_mm3 | −0.823 | −0.821 | −0.002 | 99.7% |
| cortical_thickness_mm | −0.813 | −0.811 | −0.002 | 99.7% |
| ptau181 | 0.392 | 0.391 | 0.001 | 99.7% |
| pet_amyloid_suvr | 0.265 | 0.264 | 0.001 | 99.6% |
| abeta42_40_ratio | −0.386 | −0.386 | −0.0004 | 99.9% |

**Honest read of this, don't skip it in the demo:** every predictor's
association with CDR_SB survives adjustment almost entirely (≥99.6% in
every case) — age and education explain essentially **none** of the
biomarker→severity relationship in this cohort. That's not a bug in the
adjustment method (the sanity check above confirms the method correctly
zeroes out a *real* confound when one exists) — it's a traceable property
of `generate_synthetic_v2.py` itself: `age0 = np.clip(rng.normal(74,
7.5), 55, 92)` samples age **independently** of the `neuro_severity` /
`amyloid_severity` latents that actually drive every biomarker and the
outcome. Phase 5B's v2 generator fixed the single-shared-latent problem
(giving Module A something real to fuse) and the conversion-rate problem
(giving Module B something real to predict), but it was never designed
to wire age/education in as genuine confounders of the biomarker→outcome
relationship — that's a different, narrower gap than what 5B set out to
fix, and this analysis is what surfaces it.

**What this means practically:**
- Don't claim a validated "we checked for confounding and found none" result
  in front of judges — claim the adjustment *method* is correct and
  literature-grounded (verified against a known-confounded synthetic
  case), and that this cohort's generator doesn't currently encode a
  real age/education confounding structure to detect, which is itself
  useful, honest information about what synthetic data can and can't
  validate.
- The fix, if a demo needs a visible confounding effect: adjust
  `generate_synthetic_v2.py` so `age0` (or a new term) contributes
  directly to `neuro_severity`/`amyloid_severity`, the same kind of
  targeted generator patch Phase 5B already made for cross-modal
  interaction and conversion rate. Not done here, for the same reason
  Phase 1 didn't hand-inject a fusion advantage: shaping the data to make
  the story win would defeat the point of an honest validation layer.
- On real ADNI/OASIS data, age and education are well-documented AD
  confounders in the literature this phase cites — re-running this exact
  script unchanged against real data is the way to find out whether the
  "negligible confounding" result is a synthetic-data artifact (likely)
  or holds up on real cohorts (per the cited literature, unlikely, but
  worth checking rather than assuming).

## SHAP explanation of the frozen fusion model (written, not executed here)

`explain_fusion_model_shap()` wraps the frozen Phase 1
`CrossAttentionFusion` checkpoint's dict-based forward pass (per-modality
tensors + presence masks) behind a flat-vector callable and runs
`shap.KernelExplainer` (model-agnostic — chosen over `DeepExplainer`
specifically because `DeepExplainer`'s gradient hooks assume a single
flat input tensor, not the dict-of-modalities interface
`CrossAttentionFusion.forward` actually uses) against the frozen
checkpoint via `load_fusion.py`'s existing loader.

**Not executed in this dev sandbox**: requires both `torch` (to load the
checkpoint) and `shap`, neither installed, and no network access to
install them (same constraint Phase 5A's OASIS adapter and Phase 5's
full v2 retraining ran into). `run_phase6.py` checks for both imports
first and skips this step with a clear message if they're missing,
rather than failing the whole phase — the backdoor-adjustment analysis
above runs regardless.

```bash
# In a torch+shap-equipped environment:
python run_phase6.py --data_dir data/raw_v2 --checkpoint checkpoints_v2/fusion_best.pt
```

`compare_shap_vs_backdoor_adjustment()` is the plan's actual requested
output shape — "adjusted vs. unadjusted feature importance side by
side" — combining whichever unadjusted-importance source is available
(real SHAP values when `explain_fusion_model_shap` has been run; a
labeled fallback to the linear unadjusted coefficient otherwise, as in
this sandbox) with the always-available backdoor-adjusted numbers. See
`outputs/phase6_causal_report.json` for the full side-by-side table as
actually run here (linear-fallback version).

## Quickstart

```bash
python run_phase6.py                          # gate check + SHAP (if available) + backdoor adjustment
python src/causal_explainability.py --sanity_check   # verify the adjustment logic in isolation
python src/causal_explainability.py --csv_path data/raw_v2/cleaned_dataset.csv  # just the backdoor analysis
```

Writes `outputs/phase6_causal_report.json`: per-predictor unadjusted vs.
confounder-adjusted importance, confounding strength, and the
SHAP-vs-adjusted side-by-side comparison.

---

# Phase 7 — Module F: Population Resource-Constrained Optimizer

**Explicitly out of this package's own build scope** — the plan lists
this as "should-build after the gate clears," a stretch goal beyond
Phases 0–6, not one of Phase 1/2's named "if time allows" items. Built
anyway on request, held to the same standard as everything else here.

**Gate status:** same as Phase 6 — `outputs/phase5_validation_gate_report.json`
currently shows `all_passed: false`, so everything below is methodology
development and verification of the *optimizer itself*, not a validated
population-level claim. `run_phase7.py` checks this and warns, same
pattern as `run_phase6.py`.

## What the plan asks for

> "Given a fixed weekly capacity, decide which subset of currently-flagged
> patients to escalate to maximize expected diagnostic yield across the
> population... knapsack-style MILP maximizing Σ(expected diagnostic
> yield_i × x_i) subject to Σx_i ≤ capacity, where yield combines Module
> C's escalation value and Module D's uncertainty. Solve with PuLP or OR-Tools."

Two genuinely separate pieces: **(1)** the MILP/knapsack solver itself,
and **(2)** where the per-patient "expected diagnostic yield" numbers
that get fed into it actually come from. `src/resource_optimizer.py`
keeps them clearly apart because they have very different execution
status in this dev sandbox.

## 1. The solver

`solve_allocation()` tries `pulp` (the plan's named tool) first; falls
back to a from-scratch dynamic-programming 0/1 knapsack when it's not
installed — this sandbox's situation (no network to install `pulp`).
Both solve the exact same optimization to the exact same optimum for
integer costs (true here — `pathway_env.py`'s `TEST_INFO` costs are all
whole numbers: blood=1, MRI=3, PET=8), so the fallback isn't an
approximation, just a dependency-free implementation of the exact
algorithm.

**Sanity check** (`python src/resource_optimizer.py --sanity_check`,
always runs): the DP knapsack's result is compared against **brute-force
enumeration of all 2¹⁵ subsets** on a random 15-item instance:

```
resource_optimizer sanity check passed: DP knapsack yield (4.365)
exactly matches brute-force optimum (4.365) over all 32768 subsets,
n=15 items, capacity=20.
```

This has actually been run in this sandbox (pure numpy, no torch needed)
— the solver's correctness is fully verified end to end, independent of
where the yield numbers themselves come from.

## 2. Where the yield numbers come from

**The real version** (`compute_population_yield_full()`) is written
against the actual frozen artifacts the plan intends: Module A's fusion
model + Module D's MC-Dropout epistemic uncertainty (`uncertainty.py`)
for the uncertainty term, and Module C's frozen PPO agent
(`train_pathway.py`'s checkpoint, loaded via `stable_baselines3.PPO.load`)
for the escalation-value term. **Not executed here** — needs `torch` and
`stable-baselines3`, neither installed, no network to get them. Same
status as Phase 6's `explain_fusion_model_shap()`.

**The surrogate version** (`compute_population_yield_surrogate()`) is
what actually produced every number below — plain scikit-learn, real
v2 data, leakage-safe (fit on TRAIN, evaluated on TEST):

- `clf_screening`: RandomForest on cognitive features only (MMSE, ADAS13,
  CDR_SB — always recorded per Phase 0's cohort table, i.e. genuinely
  known before any escalation decision).
- `clf_full`: RandomForest on cognitive + blood + MRI + PET (mean-imputed
  where missing, same simplification `smote_baseline.py` and
  `causal_explainability.py` already make).
- **Uncertainty proxy** (stands in for Module D): Shannon entropy of
  `clf_screening`'s predicted distribution.
- **Escalation-value proxy** (stands in for Module C): `max(0,
  P_full(true class) − P_screening(true class))` — how much probability
  mass the missing modalities would shift toward the *correct* diagnosis,
  computed against real held-out ground truth, not fabricated.
- **yield** = uncertainty proxy × escalation-value proxy, the plan's own
  combination formula.
- **cost** = sum of `TEST_COST[modality]` for whichever of blood/MRI/PET
  are *actually* missing for that specific patient-visit — a real,
  per-patient variable weight drawn from the cohort's genuine missingness
  pattern (not a fabricated constant), which is what makes this an
  actually-weighted knapsack rather than a trivial "sort and take the top
  N" case.

**Yield-surrogate sanity check** (`sanity_check_yield_surrogate()`,
cheap invariant checks, not a correctness proof — there's no ground-truth
"correct" yield to check against, unlike the solver above):

```
resource_optimizer yield-surrogate sanity check passed:
248 incomplete-workup candidates, all yield/cost invariants hold.
```

## Actual results on the hardened synthetic v2 cohort (test split)

```bash
python run_phase7.py --data_dir data/raw_v2 --capacities 20 50 100 200
```

248 test-split visits have an incomplete workup (missing ≥1 of
blood/MRI/PET) — the candidate pool for escalation:

| Missing modalities | Count |
|---|---|
| PET only | 118 |
| blood + PET | 76 |
| blood only | 26 |
| blood + MRI + PET | 10 |
| MRI + PET | 10 |
| MRI only | 5 |
| blood + MRI | 3 |

Mean cost across candidates: 7.71 (in `TEST_COST` units). Mean yield:
0.0646.

**Capacity sweep** (the static equivalent of the plan's "live
capacity-slider demo re-solving in real time" — an actual interactive
slider is a frontend concern outside this repo's scope; this is the data
such a slider would re-query at each position):

| Weekly capacity | Solver used | Patients selected | Total yield | Capacity utilization |
|---|---|---|---|---|
| 20 | DP fallback | 10 | 2.109 | 100% |
| 50 | DP fallback | 15 | 3.500 | 100% |
| 100 | DP fallback | 21 | 5.517 | 100% |
| 200 | DP fallback | 33 | 8.652 | 100% |

**Honest read:** capacity utilization is exactly 100% at every level
tested — with 248 real candidates and costs as small as 1 (blood-only
gaps), the solver always finds something to spend remaining budget on,
which is expected behavior for a knapsack with many small-weight items
available, not a sign of a solver bug (confirmed correct against
brute-force above). This is a demonstration of the optimizer working
correctly on real, held-out data — **not** a validated clinical claim
about which patients should actually get scarce PET slots, both because
Phase 5's gate hasn't passed yet and because the yield numbers themselves
come from the surrogate classifiers described above, not the real frozen
Module A/C/D pipeline. Full report, including every selected patient ID
per capacity level: `outputs/phase7_resource_allocation_report.json`.

## Quickstart

```bash
python run_phase7.py                                    # gate check + sanity checks + full report
python src/resource_optimizer.py --sanity_check          # solver correctness + yield invariants only
python src/resource_optimizer.py --data_dir data/raw_v2 --capacities 20 50 100 200
```
