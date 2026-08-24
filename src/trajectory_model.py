"""
trajectory_model.py
--------------------
Phase 2: TrajectoryLSTM, per the plan's "practical build" for this phase:

    "implement a stacked LSTM taking the 2-3 available visit timepoints per
    patient ... predicting next-visit stage + a regression head for
    MMSE/CDR trajectory slope."

Two stages:

1. VisitEncoder -- fuses ONE visit's multimodal features into a single
   embedding. Reuses Phase 1's missing-modality convention (a learned
   "missing" embedding substituted per absent modality, BERT/ViT mask-token
   style, rather than zero-imputation) and a small attention-pooling step
   over the modality tokens. Unlike Phase 1's CrossAttentionFusion, the
   token sequence is pooled to ONE vector per visit here, because in this
   phase the LSTM -- not attention -- is the layer responsible for
   cross-*time* structure.

2. TrajectoryLSTM -- runs the per-visit embeddings through a stacked LSTM
   (packed/padded properly, so pad timesteps never get treated as real
   visits) and reads two heads off each timestep's hidden state:
     - next_diagnosis_head: 3-way classifier for the diagnosis at the
       NEXT visit (TADPOLE-style future-status prediction)
     - slope_head: regression for CDR-SB trajectory slope (points per
       6-month interval) between this visit and the next

Dropout is left active throughout (not just at train time) so
`project_last_visit` can be called repeatedly under `model.train()` with
`torch.no_grad()` for MC Dropout uncertainty bands (Gal & Ghahramani, 2016)
on the time-to-threshold projection in evaluate_trajectory.py. This is
Phase 2 building the hook that Phase 4's calibrated uncertainty layer is
meant to plug into -- not a substitute for that module.
"""

import torch
import torch.nn as nn

from dataset import MODALITY_COLUMNS, STATIC_COLUMNS


class VisitEncoder(nn.Module):
    def __init__(self, embed_dim: int = 32, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.modalities = list(MODALITY_COLUMNS.keys())
        self.embed_dim = embed_dim

        self.encoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(len(MODALITY_COLUMNS[m]), embed_dim), nn.ReLU(),
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
            ) for m in self.modalities
        })
        self.static_encoder = nn.Sequential(
            nn.Linear(len(STATIC_COLUMNS), embed_dim), nn.ReLU(),
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
        )
        self.missing_embeddings = nn.ParameterDict({
            m: nn.Parameter(torch.randn(embed_dim) * 0.02) for m in self.modalities
        })
        self.token_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool_query = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def forward(self, modality_features_t, modality_mask_t, static_features_t):
        """modality_features_t/modality_mask_t: dict[m -> (B, ...)] for ONE
        timestep. static_features_t: (B, len(STATIC_COLUMNS)).
        Returns (B, embed_dim)."""
        B = static_features_t.shape[0]
        tokens = []
        for m in self.modalities:
            encoded = self.encoders[m](modality_features_t[m])
            present = modality_mask_t[m].unsqueeze(-1).float()
            missing_tok = self.missing_embeddings[m].unsqueeze(0).expand(B, -1)
            token = present * encoded + (1 - present) * missing_tok
            tokens.append(token.unsqueeze(1))
        tokens.append(self.static_encoder(static_features_t).unsqueeze(1))
        seq = torch.cat(tokens, dim=1)  # (B, n_modalities+1, embed_dim)

        query = self.pool_query.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        pooled, _ = self.token_attn(query, seq, seq, need_weights=False)
        return self.norm(pooled.squeeze(1))


class TrajectoryLSTM(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64, n_layers: int = 2,
                 n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.visit_encoder = VisitEncoder(embed_dim=embed_dim, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.next_diagnosis_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.slope_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_sequence(self, batch):
        B, T = batch["seq_mask"].shape
        device = batch["seq_mask"].device
        visit_embeds = torch.zeros(B, T, self.visit_encoder.embed_dim, device=device)
        for t in range(T):
            mf_t = {m: batch["modality_features"][m][:, t, :] for m in MODALITY_COLUMNS}
            mm_t = {m: batch["modality_mask"][m][:, t] for m in MODALITY_COLUMNS}
            static_t = batch["static_features"][:, t, :]
            visit_embeds[:, t, :] = self.visit_encoder(mf_t, mm_t, static_t)

        lengths = batch["length"].clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(visit_embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        hidden_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        return hidden_seq  # (B, T, hidden_dim)

    def forward(self, batch):
        """Per-timestep predictions for every real visit in the batch (used
        for training/eval of the next-visit and slope heads at every
        position that has a valid target)."""
        hidden_seq = self.encode_sequence(batch)
        h = self.head_dropout(hidden_seq)
        next_logits = self.next_diagnosis_head(h)   # (B, T, n_classes)
        slope_pred = self.slope_head(h).squeeze(-1)  # (B, T)
        return next_logits, slope_pred

    def project_last_visit(self, batch):
        """Next-diagnosis probs + slope prediction read off each subject's
        OWN last real visit -- this is the forward-looking prediction used
        for the time-to-threshold projection demo."""
        hidden_seq = self.encode_sequence(batch)
        lengths = batch["length"].clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden_seq.shape[-1])
        last_hidden = hidden_seq.gather(1, idx).squeeze(1)  # (B, hidden_dim)
        h = self.head_dropout(last_hidden)
        next_logits = self.next_diagnosis_head(h)
        slope_pred = self.slope_head(h).squeeze(-1)
        return next_logits, slope_pred
