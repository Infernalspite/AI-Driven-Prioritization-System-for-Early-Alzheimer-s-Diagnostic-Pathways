"""
model.py
--------
Two models, trained and compared head-to-head:

1. ConcatBaselineMLP — the "what the problem statement literally asks for":
   zero-impute missing modalities, concatenate everything, MLP classifier.
   This is the thing novelty is being measured AGAINST, so it has to exist
   and be trained fairly, not strawmanned.

2. CrossAttentionFusion — the novel model, following the MCAD / NeuroNet-AD
   pattern from the literature review: a small encoder per modality, then a
   cross-attention block that lets modalities inform each other's
   representation, then a pooled classification head.

   Missing-modality handling: rather than silently zero-imputing and hoping
   attention "figures it out", each modality has a learned MISSING embedding
   (same idea as BERT/ViT mask tokens) substituted whenever that modality is
   absent for a visit. This means missingness itself becomes a learnable
   signal the model can use — clinically relevant, since *which* tests a
   clinician already chose to order is itself informative (e.g. a patient who
   already got a PET scan was probably already flagged as high-risk).
"""

import torch
import torch.nn as nn

from dataset import MODALITY_COLUMNS, STATIC_COLUMNS


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConcatBaselineMLP(nn.Module):
    """Baseline: concatenate zero-imputed modality features + static features -> MLP."""

    def __init__(self, n_classes: int = 3, hidden_dim: int = 64):
        super().__init__()
        input_dim = sum(len(cols) for cols in MODALITY_COLUMNS.values()) + len(STATIC_COLUMNS)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, batch):
        parts = [batch["modality_features"][m] for m in MODALITY_COLUMNS]
        parts.append(batch["static_features"])
        x = torch.cat(parts, dim=-1)
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 32, n_heads: int = 4, n_classes: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.modalities = list(MODALITY_COLUMNS.keys())

        # one encoder per modality, mapping its raw feature vector -> shared embed space
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(len(MODALITY_COLUMNS[m]), embed_dim) for m in self.modalities
        })
        self.static_encoder = ModalityEncoder(len(STATIC_COLUMNS), embed_dim)

        # learned "this modality was missing" embeddings (one per modality)
        self.missing_embeddings = nn.ParameterDict({
            m: nn.Parameter(torch.randn(embed_dim) * 0.02) for m in self.modalities
        })

        # a CLS-style pooling token, as in BERT/ViT
        self.cls_token = nn.Parameter(torch.randn(embed_dim) * 0.02)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, batch, return_attention: bool = False):
        B = batch["static_features"].shape[0]
        tokens = []

        for m in self.modalities:
            encoded = self.encoders[m](batch["modality_features"][m])  # (B, embed_dim)
            present = batch["modality_mask"][m].unsqueeze(-1).float()  # (B, 1)
            missing_tok = self.missing_embeddings[m].unsqueeze(0).expand(B, -1)  # (B, embed_dim)
            token = present * encoded + (1 - present) * missing_tok
            tokens.append(token.unsqueeze(1))  # (B, 1, embed_dim)

        static_tok = self.static_encoder(batch["static_features"]).unsqueeze(1)  # always present
        tokens.append(static_tok)

        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, embed_dim)
        tokens.append(cls)

        seq = torch.cat(tokens, dim=1)  # (B, n_tokens, embed_dim)

        attn_out, attn_weights = self.cross_attention(seq, seq, seq, need_weights=True,
                                                        average_attn_weights=True)
        seq = self.norm1(seq + attn_out)
        seq = self.norm2(seq + self.ffn(seq))

        cls_out = seq[:, -1, :]  # pooled representation = CLS token after attention
        logits = self.classifier(cls_out)

        if return_attention:
            return logits, attn_weights
        return logits

    def token_names(self):
        return self.modalities + ["static", "CLS"]
