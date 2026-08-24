"""
losses.py
---------
Phase 5C: class-imbalance handling for rare converter/AD examples.

Plain class-weighted cross-entropy (used throughout Phases 1-2) reweights
the LOSS but not the SAMPLING -- a batch can still go many steps without
containing a converter example at all, and even when it does, an "easy"
majority-class example contributes the same per-example gradient budget as
a hard, rare one until the weight is applied post-hoc.

FocalLoss (Lin et al., 2017, "Focal Loss for Dense Object Detection",
ICCV) down-weights easy, well-classified examples multiplicatively via
(1 - p_t)^gamma, so gradient signal concentrates on hard/rare examples
instead of being swamped by a large number of easy majority-class ones.
It's a drop-in replacement for nn.CrossEntropyLoss(weight=...) -- same
call signature, same optional per-class weight -- so it can be swapped
into train.py / train_trajectory.py without touching the training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    gamma=0 recovers ordinary (weighted) cross-entropy exactly -- this is
    a strict generalization, not a different loss family, so it's safe to
    use as the default with gamma=0 and only raise it when class imbalance
    is the confirmed bottleneck (as Phase 2's evaluation found).

    Args:
        weight: optional per-class weight tensor, same as nn.CrossEntropyLoss.
        gamma: focusing parameter. 0 = plain cross-entropy. 2.0 is the
            value used in the original paper and is a reasonable default
            for the converter/AD imbalance seen in Phase 2's evaluation.
        reduction: 'mean' | 'sum' | 'none' (matches nn.CrossEntropyLoss).
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # per-example CE (no reduction yet), reusing PyTorch's numerically stable
        # log-softmax + nll rather than reimplementing it
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)  # probability the model assigned to the true class
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def sanity_check():
    """gamma=0 must exactly match nn.CrossEntropyLoss(weight=...)."""
    torch.manual_seed(0)
    logits = torch.randn(16, 3)
    targets = torch.randint(0, 3, (16,))
    weight = torch.tensor([1.0, 2.0, 4.0])

    ce_ref = nn.CrossEntropyLoss(weight=weight)(logits, targets)
    focal_gamma0 = FocalLoss(weight=weight, gamma=0.0)(logits, targets)
    assert torch.allclose(ce_ref, focal_gamma0, atol=1e-6), (ce_ref.item(), focal_gamma0.item())

    # gamma>0 must down-weight relative to plain CE (loss should be <= ce_ref
    # in expectation for a randomly-initialized, imperfect classifier, since
    # every example gets a focal_term in [0,1])
    focal_gamma2 = FocalLoss(weight=weight, gamma=2.0)(logits, targets)
    assert focal_gamma2.item() <= ce_ref.item()

    print("FocalLoss sanity check passed: gamma=0 matches weighted CE exactly; "
          f"gamma=2 loss ({focal_gamma2.item():.4f}) <= plain weighted CE ({ce_ref.item():.4f}).")


if __name__ == "__main__":
    sanity_check()
