"""
uncertainty.py
--------------
Phase 4: Module D, Uncertainty-Aware Prioritization. Per the plan:

    "Gal & Ghahramani (2016) established that keeping dropout active at
    inference time and doing repeated stochastic forward passes
    approximates Bayesian inference ... Kendall & Gal formalized the split
    between aleatoric uncertainty (irreducible noise in the data itself)
    and epistemic uncertainty (the model's own lack of knowledge,
    reducible with more data) ... raw MC Dropout is known to be poorly
    calibrated ... evaluate with Expected Calibration Error (ECE), not
    just accuracy."

No architecture change was needed: Phase 1's `CrossAttentionFusion` and
`ConcatBaselineMLP` already have dropout (rate 0.2) throughout. MC Dropout
just means calling forward repeatedly with `model.train()` (dropout
active) and `torch.no_grad()`, which is what `mc_predict` below does.

Aleatoric/epistemic decomposition uses the standard classification-entropy
split (used e.g. in Kwon et al., 2020, in the same Bayesian-NN spirit
Kendall & Gal describe):

    total   = H[ mean_t(p_t) ]                     -- entropy of the averaged prediction
    aleatoric = mean_t[ H[p_t] ]                     -- average entropy of each individual pass
    epistemic = total - aleatoric                     -- mutual information between the
                                                          prediction and the (dropout-sampled)
                                                          model weights; how much the passes
                                                          DISAGREE with each other

A genuinely ambiguous case (aleatoric-dominant) has every pass confidently
agreeing on a blurry answer. A case the model just hasn't learned enough
about (epistemic-dominant) has passes disagreeing with each other -- and
per the plan, THAT is the case worth ordering another test for; the
aleatoric-dominant case may mean the test itself is inconclusive regardless
of how many more you order.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# MC Dropout forward passes + uncertainty decomposition
# ---------------------------------------------------------------------

def mc_predict(model, batch, n_passes: int = 30):
    """Runs N stochastic forward passes with dropout left ACTIVE
    (model.train(), no grad). Returns probs_stack: (n_passes, B, n_classes)."""
    model.train()  # keep dropout on -- this is the whole trick (Gal & Ghahramani, 2016)
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(batch)
            probs.append(torch.softmax(logits, dim=-1))
    model.eval()
    return torch.stack(probs, dim=0)  # (n_passes, B, n_classes)


def entropy(p, eps: float = 1e-8):
    """Shannon entropy along the last dim. p: (..., n_classes)."""
    return -(p * torch.log(p + eps)).sum(dim=-1)


def decompose_uncertainty(probs_stack):
    """probs_stack: (n_passes, B, n_classes).
    Returns dict of (B,) tensors: mean_probs, total, aleatoric, epistemic."""
    mean_probs = probs_stack.mean(dim=0)                  # (B, n_classes)
    total = entropy(mean_probs)                             # (B,)
    aleatoric = entropy(probs_stack).mean(dim=0)             # (B,)
    epistemic = (total - aleatoric).clamp(min=0.0)           # (B,) -- mutual information, >=0
    return {"mean_probs": mean_probs, "total": total, "aleatoric": aleatoric, "epistemic": epistemic}


# ---------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------

def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Standard ECE: bin by predicted confidence, compare mean confidence
    to mean accuracy in each bin, weight by bin occupancy.
    Returns (ece, per_bin_diagnostics) for reliability-diagram plotting."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"bin_range": [float(lo), float(hi)], "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(avg_conf - acc)
        bins.append({"bin_range": [float(lo), float(hi)], "count": count, "avg_confidence": avg_conf, "accuracy": acc})
    return float(ece), bins


# ---------------------------------------------------------------------
# Temperature scaling (Guo et al., 2017 -- the "logit scaling" the plan
# cites as the standard MC-Dropout-calibration follow-up)
# ---------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """A single learned scalar T applied as logits / T before softmax.
    Fit on held-out (val) logits by minimizing NLL -- never fit on test."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # T starts at 1.0

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 200):
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())
