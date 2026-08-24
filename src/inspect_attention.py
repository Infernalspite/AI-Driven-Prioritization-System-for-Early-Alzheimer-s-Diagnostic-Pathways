"""
inspect_attention.py
---------------------
Pulls the cross-attention weights for a handful of test-set examples so you
can show, in the demo, *what the model attended to* for a given patient —
e.g. "for this borderline MCI case with no PET data, the model leaned
heavily on the blood-biomarker token." This is the direct payoff of the
cross-attention design over a black-box concat MLP: the attention weights
are a natural, cheap interpretability layer, on top of anything SHAP-based
added later (Phase 5).
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ADFusionDataset, collate_fn, load_splits, LABEL_NAMES
from model import CrossAttentionFusion
from evaluate import load_model


def main(n_examples: int = 8, checkpoint_dir: str = "checkpoints",
         data_dir: str = "data/raw", output: str = "outputs/attention_examples.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm_stats = load_model(f"{checkpoint_dir}/fusion_best.pt", device)
    assert isinstance(model, CrossAttentionFusion)

    _, _, test_df, _ = load_splits(data_dir)
    test_ds = ADFusionDataset(test_df.reset_index(drop=True), norm_stats)
    loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    token_names = model.token_names()
    examples = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_examples:
                break
            batch["modality_features"] = {k: v.to(device) for k, v in batch["modality_features"].items()}
            mask = {k: bool(v.item()) for k, v in batch["modality_mask"].items()}
            batch["modality_mask"] = {k: v.to(device) for k, v in batch["modality_mask"].items()}
            batch["static_features"] = batch["static_features"].to(device)

            logits, attn_weights = model(batch, return_attention=True)
            pred = int(logits.argmax(dim=-1).item())
            true_label = int(batch["label"].item())

            # attention FROM the CLS token (last row) TO every other token — this is
            # "what did the final prediction actually lean on"
            cls_attn = attn_weights[0, -1, :].cpu().numpy().tolist()

            examples.append({
                "true_label": LABEL_NAMES[true_label],
                "predicted_label": LABEL_NAMES[pred],
                "correct": pred == true_label,
                "modality_present": mask,
                "cls_attention_to_tokens": dict(zip(token_names, [round(w, 4) for w in cls_attn])),
            })

    with open(output, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Wrote {len(examples)} attention examples -> {output}\n")
    for ex in examples:
        status = "CORRECT" if ex["correct"] else "WRONG"
        present = ", ".join(m for m, p in ex["modality_present"].items() if p) or "none"
        top_attn = sorted(ex["cls_attention_to_tokens"].items(), key=lambda x: -x[1])[:3]
        print(f"[{status}] true={ex['true_label']:>3s} pred={ex['predicted_label']:>3s} | "
              f"present modalities: {present}")
        print(f"    top attended tokens: {top_attn}")


if __name__ == "__main__":
    main()
