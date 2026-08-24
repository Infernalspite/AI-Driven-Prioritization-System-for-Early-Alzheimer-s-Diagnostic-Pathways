"""
load_fusion.py
---------------
Small shared helper: loads the Phase 1 CrossAttentionFusion checkpoint,
frozen (eval mode, requires_grad=False), for reuse as Module C's fused-state
feature extractor. Also reconstructs the exact NormalizationStats the
checkpoint was trained with, so Phase 3 episodes are normalized identically
to what the frozen model expects (never refit -- that would be a subtle
train/inference skew, not textbook leakage but still a mismatch worth
avoiding).
"""

import torch

from dataset import NormalizationStats
from model import CrossAttentionFusion


def load_frozen_fusion(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CrossAttentionFusion(embed_dim=ckpt["embed_dim"], n_classes=3, dropout=ckpt.get("dropout", 0.2))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    norm_stats = NormalizationStats()
    norm_stats.means = ckpt["norm_stats"]["means"]
    norm_stats.stds = ckpt["norm_stats"]["stds"]

    return model, norm_stats
