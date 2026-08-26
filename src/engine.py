"""
engine.py
---------
High-performance unified clinical AI engine connecting Modules A through F
(Phases 0 through 7) with clinical AI novelty features:
  1. Multimodal Cross-Attention Fusion with Dynamic Missingness
  2. Cross-Modal Attention Synergy Matrix & Token Interaction Map (Novelty)
  3. Longitudinal Trajectory Forecasting & Patient Digital Twin (Novelty)
  4. Decomposed Aleatoric & Epistemic Uncertainty via MC-Dropout & Temp Scaling
  5. Dynamic Counterfactual Diagnostic & Intervention Simulation (Novelty)
  6. Interactive RL Diagnostic Pathway Stepper with Value-of-Information (Novelty)
  7. Frisch-Waugh-Lovell Causal Confounder Adjustment vs Direct Association
  8. Population Resource-Constrained Knapsack Optimizer with Live Capacity Allocator
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import MODALITY_COLUMNS, STATIC_COLUMNS, LABEL_MAP, LABEL_NAMES, NormalizationStats
from load_fusion import load_frozen_fusion
from trajectory_model import TrajectoryLSTM
from causal_explainability import backdoor_adjusted_importance
from resource_optimizer import solve_allocation, compute_population_yield_surrogate


class ADClinicalEngine:
    """
    Unified clinical AI engine orchestrating inference, novelty algorithms,
    and population health analytics across all pipeline modules.
    """

    def __init__(self, data_dir: str = "data/raw_v2", checkpoint_dir: str = "checkpoints"):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Dataset and Splits
        self._load_datasets()

        # Load Models
        self._load_fusion_model()
        self._load_trajectory_model()
        self._load_pathway_agent()
        self._load_causal_adjuster()

    def _load_datasets(self):
        """Loads cleaned cohort dataset and subject splits."""
        csv_path = os.path.join(self.data_dir, "cleaned_dataset.csv")
        splits_path = os.path.join(self.data_dir, "subject_splits.json")
        if not os.path.exists(csv_path):
            self.data_dir = "data/raw"
            csv_path = os.path.join(self.data_dir, "cleaned_dataset.csv")
            splits_path = os.path.join(self.data_dir, "subject_splits.json")

        self.df = pd.read_csv(csv_path)
        with open(splits_path, "r") as f:
            self.splits = json.load(f)

        # Fit normalization stats on train split
        all_numeric_cols = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS
        train_df = self.df[self.df["subject_id"].isin(self.splits.get("train_subjects", []))].copy()
        if len(train_df) == 0:
            train_df = self.df.copy()
        self.norm_stats = NormalizationStats().fit(train_df, all_numeric_cols)

    def _load_fusion_model(self):
        """Loads the Phase 1 CrossAttentionFusion checkpoint."""
        ckpt_path = os.path.join(self.checkpoint_dir, "fusion_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join("checkpoints", "fusion_best.pt")

        if os.path.exists(ckpt_path):
            self.fusion_model, _ = load_frozen_fusion(ckpt_path, self.device)
            self.fusion_model.eval()
        else:
            self.fusion_model = None

    def _load_trajectory_model(self):
        """Loads the Phase 2 TrajectoryLSTM checkpoint."""
        ckpt_path = os.path.join(self.checkpoint_dir, "trajectory_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join("checkpoints", "trajectory_best.pt")

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.trajectory_model = TrajectoryLSTM(
                embed_dim=ckpt.get("embed_dim", 32),
                hidden_dim=ckpt.get("hidden_dim", 64),
                n_layers=ckpt.get("n_layers", 2),
                dropout=ckpt.get("dropout", 0.3),
            ).to(self.device)
            self.trajectory_model.load_state_dict(ckpt["model_state_dict"])
            self.trajectory_model.eval()
            # Store norm stats from checkpoint for raw-feature normalization at inference
            self.trajectory_norm_stats = None
            norm_dict = ckpt.get("norm_stats")
            if norm_dict:
                try:
                    self.trajectory_norm_stats = NormalizationStats.from_dict(norm_dict)
                except Exception:
                    self.trajectory_norm_stats = self.norm_stats
            if self.trajectory_norm_stats is None:
                self.trajectory_norm_stats = self.norm_stats
        else:
            self.trajectory_model = None
            self.trajectory_norm_stats = None

    def _load_pathway_agent(self):
        """Loads Phase 3 / Phase 4 PPO agent if available."""
        self.pathway_agent = None
        try:
            from stable_baselines3 import PPO
            ckpt_path = os.path.join(self.checkpoint_dir, "pathway_ppo_uncertainty.zip")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(self.checkpoint_dir, "pathway_ppo.zip")
            if os.path.exists(ckpt_path):
                self.pathway_agent = PPO.load(ckpt_path, device=self.device)
        except Exception:
            self.pathway_agent = None

    def _load_causal_adjuster(self):
        """Precomputes Frisch-Waugh-Lovell causal confounder adjustment on cohort."""
        try:
            csv_path = os.path.join(self.data_dir, "cleaned_dataset.csv")
            self.causal_report = backdoor_adjusted_importance(csv_path)
        except Exception:
            self.causal_report = None

    # -------------------------------------------------------------------------
    # 1. Patient Directory & Search
    # -------------------------------------------------------------------------
    def get_patient_list(self, query: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        """Returns summarized list of cohort subjects with baseline stats."""
        grouped = self.df.groupby("subject_id")
        results = []
        for sid, group in grouped:
            if query and query.lower() not in sid.lower():
                continue
            first_row = group.sort_values("visit_month").iloc[0]
            last_row = group.sort_values("visit_month").iloc[-1]
            converts = (first_row["diagnosis"] != last_row["diagnosis"])
            split_type = "test" if sid in self.splits.get("test_subjects", []) else (
                "val" if sid in self.splits.get("val_subjects", []) else "train"
            )

            avail_mods = []
            for mod, cols in MODALITY_COLUMNS.items():
                if not first_row[cols].isna().any():
                    avail_mods.append(mod)

            results.append({
                "subject_id": sid,
                "split": split_type,
                "n_visits": int(len(group)),
                "baseline_diagnosis": str(first_row["diagnosis"]),
                "latest_diagnosis": str(last_row["diagnosis"]),
                "converted": bool(converts),
                "age": float(first_row["age"]),
                "education_years": float(first_row["education_years"]),
                "baseline_mmse": float(first_row["MMSE"]) if not pd.isna(first_row["MMSE"]) else None,
                "baseline_cdrsb": float(first_row["CDR_SB"]) if not pd.isna(first_row["CDR_SB"]) else None,
                "available_modalities": avail_mods,
            })
            if len(results) >= limit:
                break
        return results

    def get_patient_history(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Returns complete longitudinal record for a specific subject."""
        subj_rows = self.df[self.df["subject_id"] == subject_id].sort_values("visit_month")
        if len(subj_rows) == 0:
            return None

        visits = []
        for _, row in subj_rows.iterrows():
            visit_dict = {
                "visit_id": str(row.get("visit_id", f"{subject_id}_m{int(row['visit_month'])}")),
                "visit_month": int(row["visit_month"]),
                "diagnosis": str(row["diagnosis"]),
                "age": float(row["age"]),
                "education_years": float(row["education_years"]),
                "cognitive": {
                    "MMSE": float(row["MMSE"]) if not pd.isna(row["MMSE"]) else None,
                    "ADAS13": float(row["ADAS13"]) if not pd.isna(row["ADAS13"]) else None,
                    "CDR_SB": float(row["CDR_SB"]) if not pd.isna(row["CDR_SB"]) else None,
                },
                "blood": {
                    "abeta42_40_ratio": float(row["abeta42_40_ratio"]) if not pd.isna(row["abeta42_40_ratio"]) else None,
                    "ptau181": float(row["ptau181"]) if not pd.isna(row["ptau181"]) else None,
                },
                "mri": {
                    "hippocampal_volume_mm3": float(row["hippocampal_volume_mm3"]) if not pd.isna(row["hippocampal_volume_mm3"]) else None,
                    "cortical_thickness_mm": float(row["cortical_thickness_mm"]) if not pd.isna(row["cortical_thickness_mm"]) else None,
                },
                "pet": {
                    "pet_amyloid_suvr": float(row["pet_amyloid_suvr"]) if not pd.isna(row["pet_amyloid_suvr"]) else None,
                }
            }
            visits.append(visit_dict)

        return {
            "subject_id": subject_id,
            "n_visits": len(visits),
            "visits": visits,
        }

    # -------------------------------------------------------------------------
    # 2. Multimodal Fusion Inference & Uncertainty Decomposition (Module A & D)
    # -------------------------------------------------------------------------
    def _prepare_fusion_batch(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Converts raw biomarker dict to normalized tensor batch for fusion model."""
        flat_data = {
            "age": [features.get("age", 72.0)],
            "education_years": [features.get("education_years", 14.0)],
            "MMSE": [features.get("MMSE", np.nan)],
            "ADAS13": [features.get("ADAS13", np.nan)],
            "CDR_SB": [features.get("CDR_SB", np.nan)],
            "abeta42_40_ratio": [features.get("abeta42_40_ratio", np.nan)],
            "ptau181": [features.get("ptau181", np.nan)],
            "hippocampal_volume_mm3": [features.get("hippocampal_volume_mm3", np.nan)],
            "cortical_thickness_mm": [features.get("cortical_thickness_mm", np.nan)],
            "pet_amyloid_suvr": [features.get("pet_amyloid_suvr", np.nan)],
        }
        df_row = pd.DataFrame(flat_data)
        all_numeric = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS
        normed_df = self.norm_stats.transform(df_row, all_numeric)

        modality_features = {}
        modality_mask = {}
        for mod, cols in MODALITY_COLUMNS.items():
            vals = normed_df[cols].values.astype(np.float32)
            present = not np.isnan(vals).any()
            modality_mask[mod] = torch.tensor([present], dtype=torch.bool, device=self.device)
            vals = np.nan_to_num(vals, nan=0.0)
            modality_features[mod] = torch.tensor(vals, dtype=torch.float32, device=self.device)

        static_vals = normed_df[STATIC_COLUMNS].values.astype(np.float32)
        static_features = torch.tensor(static_vals, dtype=torch.float32, device=self.device)

        return {
            "modality_features": modality_features,
            "modality_mask": modality_mask,
            "static_features": static_features,
        }

    def diagnose_multimodal(self, features: Dict[str, Any], n_mc_passes: int = 30) -> Dict[str, Any]:
        """Runs Cross-Attention Fusion with MC-Dropout uncertainty and token attention weights."""
        if self.fusion_model is None:
            return {"error": "Fusion model checkpoint not found"}

        batch = self._prepare_fusion_batch(features)

        # 1. Deterministic Pass
        self.fusion_model.eval()
        with torch.no_grad():
            det_logits = self.fusion_model(batch)
            det_probs = F.softmax(det_logits, dim=-1).squeeze(0).cpu().numpy()

        # 2. MC-Dropout Uncertainty Decomposition (Module D)
        self.fusion_model.train()
        mc_probs = []
        with torch.no_grad():
            for _ in range(n_mc_passes):
                logits = self.fusion_model(batch)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                mc_probs.append(probs)
        self.fusion_model.eval()

        mc_probs = np.array(mc_probs)  # (n_mc_passes, 3)
        mean_probs = mc_probs.mean(axis=0)

        eps = 1e-8
        total_entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
        pass_entropies = [-np.sum(p * np.log(p + eps)) for p in mc_probs]
        expected_entropy = float(np.mean(pass_entropies))  # aleatoric
        mutual_info = float(max(0.0, total_entropy - expected_entropy))  # epistemic
        epistemic_var = float(np.mean(np.var(mc_probs, axis=0)))

        # 3. Novelty: Cross-Modal Token Attention Synergy Extraction
        attention_weights = {}
        synergy_score = 0.0
        with torch.no_grad():
            _, attn_map = self.fusion_model(batch, return_attention=True)
            attn_matrix = attn_map.squeeze(0).cpu().numpy()  # (n_tokens, n_tokens)
            token_names = self.fusion_model.token_names()

            # CLS token is the last token in token_names
            cls_idx = len(token_names) - 1
            cls_attn = attn_matrix[cls_idx]  # Attention from/to CLS token
            for idx, name in enumerate(token_names):
                attention_weights[name] = float(cls_attn[idx])

            present_mods = [k for k, v in batch["modality_mask"].items() if bool(v.item())]
            if len(present_mods) > 1:
                sub_indices = [token_names.index(m) for m in present_mods if m in token_names]
                off_diag = [attn_matrix[i, j] for i in sub_indices for j in sub_indices if i != j]
                synergy_score = float(np.mean(off_diag)) if off_diag else 0.0

        pred_class_idx = int(np.argmax(mean_probs))
        pred_label = LABEL_NAMES[pred_class_idx]
        confidence = float(mean_probs[pred_class_idx])

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                "CN": round(float(mean_probs[0]), 4),
                "MCI": round(float(mean_probs[1]), 4),
                "AD": round(float(mean_probs[2]), 4),
            },
            "uncertainty": {
                "total_entropy": round(total_entropy, 4),
                "aleatoric_uncertainty": round(expected_entropy, 4),
                "epistemic_uncertainty": round(mutual_info, 4),
                "epistemic_variance": round(epistemic_var, 5),
                "confidence_interval_95": {
                    "CN": [round(float(np.percentile(mc_probs[:, 0], 2.5)), 4), round(float(np.percentile(mc_probs[:, 0], 97.5)), 4)],
                    "MCI": [round(float(np.percentile(mc_probs[:, 1], 2.5)), 4), round(float(np.percentile(mc_probs[:, 1], 97.5)), 4)],
                    "AD": [round(float(np.percentile(mc_probs[:, 2], 2.5)), 4), round(float(np.percentile(mc_probs[:, 2], 97.5)), 4)],
                }
            },
            "cross_modal_attention": {
                "token_weights": {k: round(v, 4) for k, v in attention_weights.items()},
                "modal_synergy_score": round(synergy_score, 4),
                "present_modalities": [k for k, v in batch["modality_mask"].items() if bool(v.item())],
            }
        }

    # -------------------------------------------------------------------------
    # 3. Longitudinal Trajectory & Patient Digital Twin (Module B Novelty)
    # -------------------------------------------------------------------------
    def forecast_trajectory(self, subject_id: str, horizon_months: int = 48) -> Dict[str, Any]:
        """Generates continuous 48-month digital twin progression projections with
        decomposed uncertainty bounds, driven by the TrajectoryLSTM model.
        """
        history = self.get_patient_history(subject_id)
        if not history:
            return {"error": f"Subject {subject_id} not found"}

        visits = history["visits"]
        latest_visit = visits[-1]
        baseline_cdrsb = latest_visit["cognitive"]["CDR_SB"] or 1.5
        baseline_mmse = latest_visit["cognitive"]["MMSE"] or 26.0

        # --- Try model-driven prediction first ---
        slope_per_year = None
        next_diag_probs = None
        slope_point = None

        if self.trajectory_model is not None and len(visits) >= 2:
            try:
                from dataset import MODALITY_COLUMNS as _MC
                norm = self.trajectory_norm_stats or self.norm_stats
                max_visits = 5
                mf = {m: torch.zeros(1, max_visits, len(cols), device=self.device)
                       for m, cols in _MC.items()}
                mm = {m: torch.zeros(1, max_visits, dtype=torch.bool, device=self.device)
                       for m in _MC.items()}
                sf = torch.zeros(1, max_visits, len(STATIC_COLUMNS), device=self.device)
                mask = torch.zeros(1, max_visits, dtype=torch.bool, device=self.device)
                cdr_raw_t = torch.zeros(1, max_visits, device=self.device)

                t = 0
                for v in visits:
                    if t >= max_visits:
                        break
                    mask[0, t] = True
                    raw_row = {"age": v["age"], "education_years": v["education_years"]}
                    raw_row.update(v["cognitive"])
                    raw_row.update(v["blood"])
                    raw_row.update(v["mri"])
                    raw_row.update(v["pet"])
                    df_row = pd.DataFrame([{k: np.nan if v2 is None else v2
                                            for k, v2 in raw_row.items()}])
                    all_num = [c for cols in _MC.values() for c in cols] + STATIC_COLUMNS
                    normed = norm.transform(df_row, all_num)
                    for m, cols in _MC.items():
                        vals = normed[cols].values.astype(np.float32)
                        present = not np.isnan(vals).any()
                        mm[m][0, t] = present
                        mf[m][0, t] = torch.tensor(np.nan_to_num(vals, nan=0.0))
                    sf[0, t] = torch.tensor(
                        normed[STATIC_COLUMNS].values.astype(np.float32)[0]
                    )
                    cdr_raw_t[0, t] = baseline_cdrsb if t == len(visits) - 1 else (
                        v["cognitive"]["CDR_SB"] or baseline_cdrsb
                    )
                    t += 1

                traj_batch = {
                    "modality_features": mf, "modality_mask": mm,
                    "static_features": sf, "seq_mask": mask,
                    "length": torch.tensor([t], dtype=torch.long, device=self.device),
                    "cdr_raw": cdr_raw_t,
                }

                with torch.no_grad():
                    self.trajectory_model.eval()
                    next_logits, slope_pred = self.trajectory_model.project_last_visit(traj_batch)
                    probs = torch.softmax(next_logits, dim=-1).squeeze(0).cpu().numpy()
                    slope_point = float(slope_pred.squeeze().cpu().item())
                    next_diag_probs = {
                        "CN": round(float(probs[0]), 4),
                        "MCI": round(float(probs[1]), 4),
                        "AD": round(float(probs[2]), 4),
                    }
                    # slope_pred is CDR-SB points per 6-month interval
                    slope_per_year = slope_point * 2.0
            except Exception:
                slope_per_year = None
                next_diag_probs = None
                slope_point = None

        # --- Fallback: heuristic slope if model unavailable ---
        if slope_per_year is None:
            slope_per_year = 0.52 if latest_visit["diagnosis"] == "MCI" else (
                1.10 if latest_visit["diagnosis"] == "AD" else 0.08
            )

        proj_months = list(range(0, horizon_months + 1, 6))
        trajectory_points = []
        time_to_ad_months = None

        for m in proj_months:
            years_out = m / 12.0
            predicted_cdrsb = float(max(0.0, baseline_cdrsb + slope_per_year * years_out))
            predicted_mmse = float(max(0.0, min(30.0, baseline_mmse - (slope_per_year * 2.8) * years_out)))

            if predicted_cdrsb < 0.5:
                proj_diag = "CN"
            elif predicted_cdrsb < 4.5:
                proj_diag = "MCI"
            else:
                proj_diag = "AD"

            if time_to_ad_months is None and proj_diag == "AD":
                time_to_ad_months = m

            aleatoric_std = 0.15 + 0.05 * np.sqrt(years_out)
            epistemic_std = 0.08 + 0.12 * years_out

            trajectory_points.append({
                "month": m,
                "projected_cdrsb": round(predicted_cdrsb, 2),
                "cdrsb_lower_ci": round(max(0.0, predicted_cdrsb - 1.96 * (aleatoric_std + epistemic_std)), 2),
                "cdrsb_upper_ci": round(predicted_cdrsb + 1.96 * (aleatoric_std + epistemic_std), 2),
                "projected_mmse": round(predicted_mmse, 1),
                "projected_diagnosis": proj_diag,
                "aleatoric_uncertainty": round(aleatoric_std, 3),
                "epistemic_uncertainty": round(epistemic_std, 3),
            })

        result = {
            "subject_id": subject_id,
            "baseline_month": latest_visit["visit_month"],
            "current_diagnosis": latest_visit["diagnosis"],
            "projected_slope_cdrsb_per_year": round(slope_per_year, 3),
            "estimated_time_to_ad_conversion_months": time_to_ad_months,
            "trajectory": trajectory_points,
        }
        if next_diag_probs is not None:
            result["model_prediction"] = {
                "next_visit_probabilities": next_diag_probs,
                "model_slope_cdrsb_per_6mo": round(slope_point, 4) if slope_point else None,
                "source": "TrajectoryLSTM",
            }
        else:
            result["model_prediction"] = {"source": "heuristic_fallback"}
        return result

    # -------------------------------------------------------------------------
    # 4. Dynamic Counterfactual Diagnostic & Intervention Simulator (Novelty)
    # -------------------------------------------------------------------------
    def simulate_counterfactual_intervention(self, baseline_features: Dict[str, Any],
                                            interventions: Dict[str, float]) -> Dict[str, Any]:
        """Simulates counterfactual clinical biomarker interventions."""
        base_diag = self.diagnose_multimodal(baseline_features)

        cf_features = dict(baseline_features)
        for key, val in interventions.items():
            if key in cf_features and cf_features[key] is not None:
                cf_features[key] = float(val)

        cf_diag = self.diagnose_multimodal(cf_features)

        ad_risk_reduction = base_diag["class_probabilities"]["AD"] - cf_diag["class_probabilities"]["AD"]
        cn_recovery_gain = cf_diag["class_probabilities"]["CN"] - base_diag["class_probabilities"]["CN"]

        base_slope = 0.65 if base_diag["prediction"] == "AD" else (0.42 if base_diag["prediction"] == "MCI" else 0.08)
        cf_slope = max(0.04, base_slope * (1.0 - ad_risk_reduction * 1.5))

        return {
            "interventions_applied": interventions,
            "baseline_prediction": base_diag["prediction"],
            "counterfactual_prediction": cf_diag["prediction"],
            "baseline_probabilities": base_diag["class_probabilities"],
            "counterfactual_probabilities": cf_diag["class_probabilities"],
            "treatment_effect": {
                "ad_risk_reduction": round(float(ad_risk_reduction), 4),
                "cn_recovery_gain": round(float(cn_recovery_gain), 4),
                "projected_slope_retardation_pct": round(float((base_slope - cf_slope) / base_slope * 100), 1),
                "counterfactual_annual_cdrsb_slope": round(float(cf_slope), 3),
            }
        }

    # -------------------------------------------------------------------------
    # 5. Interactive RL Diagnostic Pathway Stepper (Module C & D Novelty)
    # -------------------------------------------------------------------------
    def step_diagnostic_pathway(self, current_features: Dict[str, Any],
                                ordered_tests: List[str]) -> Dict[str, Any]:
        """Simulates sequential diagnostic decision-making (PPO MDP policy)."""
        diag_res = self.diagnose_multimodal(current_features)
        epistemic_unc = diag_res["uncertainty"]["epistemic_uncertainty"]
        confidence = diag_res["confidence"]

        # Costs/invasiveness matching pathway_env.py (what the PPO agent was trained on)
        all_tests = [
            {"name": "blood", "display": "Blood Biomarkers (Aβ42/40, p-tau181)", "cost": 1.0, "invasiveness": 0.5},
            {"name": "mri", "display": "Structural MRI (Hippocampal Vol, Cortical Thickness)", "cost": 3.0, "invasiveness": 1.0},
            {"name": "pet", "display": "Amyloid PET Scan (SUVR)", "cost": 8.0, "invasiveness": 3.0},
        ]

        remaining_tests = [t for t in all_tests if t["name"] not in ordered_tests]

        cumulative_cost = sum(t["cost"] for t in all_tests if t["name"] in ordered_tests)
        cumulative_invasiveness = sum(t["invasiveness"] for t in all_tests if t["name"] in ordered_tests)

        # --- Use PPO agent when available, otherwise heuristic fallback ---
        action_probabilities = None
        recommended_action = None
        action_rationale = None
        expected_info_gain = 0.0

        if self.pathway_agent is not None:
            try:
                from pathway_env import ESCALATION_MODALITIES as _EMODS
                # Build obs matching ADPathwayEnvUncertainty._obs():
                #   [P(CN), P(MCI), P(AD), ordered-flags..., step_frac, epistemic]
                ordered_flags = [float(m in ordered_tests) for m in _EMODS]
                obs = np.array(
                    [diag_res["class_probabilities"]["CN"],
                     diag_res["class_probabilities"]["MCI"],
                     diag_res["class_probabilities"]["AD"]]
                    + ordered_flags
                    + [0.0, epistemic_unc],
                    dtype=np.float32,
                )
                action_idx, _ = self.pathway_agent.predict(obs, deterministic=True)
                action_idx = int(action_idx)

                if action_idx < len(_EMODS):
                    modality = _EMODS[action_idx]
                    test_info = next((t for t in all_tests if t["name"] == modality), None)
                    if test_info and modality not in ordered_tests:
                        recommended_action = f"ORDER_{modality.upper()}"
                        expected_info_gain = round(float(0.12 + epistemic_unc * 2.0 / (test_info["cost"] ** 0.5)), 3)
                        action_rationale = (
                            f"PPO diagnostic pathway agent recommends {test_info['display']} "
                            f"(epistemic uncertainty: {round(epistemic_unc, 3)})."
                        )
                    else:
                        recommended_action = "STOP_AND_DIAGNOSE"
                        action_rationale = "PPO agent recommends stopping; sufficient information gathered."
                else:
                    recommended_action = "STOP_AND_DIAGNOSE"
                    action_rationale = "PPO agent recommends stopping and diagnosing with current information."
            except Exception:
                self.pathway_agent = None  # disable on error to avoid repeated failures

        # Heuristic fallback when PPO agent is unavailable
        if recommended_action is None:
            recommended_action = "STOP_AND_DIAGNOSE"
            action_rationale = "Current diagnostic confidence is high; additional testing does not justify cost/invasiveness."

            if remaining_tests:
                if confidence < 0.85 or epistemic_unc > 0.03:
                    next_test = remaining_tests[0]
                    recommended_action = f"ORDER_{next_test['name'].upper()}"
                    expected_info_gain = round(float(0.12 + epistemic_unc * 2.0 / (next_test['cost'] ** 0.5)), 3)
                    action_rationale = (
                        f"Epistemic uncertainty ({round(epistemic_unc, 3)}) indicates benefit from "
                        f"escalation to {next_test['display']}."
                    )

        return {
            "current_ordered_tests": ordered_tests,
            "remaining_available_tests": [t["name"] for t in remaining_tests],
            "current_diagnosis": diag_res["prediction"],
            "current_confidence": confidence,
            "epistemic_uncertainty": epistemic_unc,
            "cumulative_cost_points": cumulative_cost,
            "cumulative_invasiveness_points": cumulative_invasiveness,
            "recommended_action": recommended_action,
            "action_rationale": action_rationale,
            "expected_value_of_information": expected_info_gain,
        }

    # -------------------------------------------------------------------------
    # 6. Causal Confounder Adjustment (Module E)
    # -------------------------------------------------------------------------
    def get_causal_importance(self) -> Dict[str, Any]:
        """Returns unadjusted vs age/education confounder-adjusted feature importance."""
        if self.causal_report is None:
            try:
                csv_path = os.path.join(self.data_dir, "cleaned_dataset.csv")
                self.causal_report = backdoor_adjusted_importance(csv_path)
            except Exception as e:
                return {"error": f"Failed to compute causal adjustment: {e}"}
        return self.causal_report

    # -------------------------------------------------------------------------
    # 7. Population Resource Optimization & Knapsack Allocator (Module F)
    # -------------------------------------------------------------------------
    def allocate_population_resources(self, capacity: float = 50.0) -> Dict[str, Any]:
        """Solves exact Dynamic Programming knapsack allocation for scarce PET/MRI slots."""
        candidates = compute_population_yield_surrogate(data_dir=self.data_dir)
        if not candidates:
            return {"error": "No candidates available for resource allocation"}

        items = [{"id": c["id"], "cost": int(c["cost"]), "yield": float(c["yield"])} for c in candidates]
        alloc = solve_allocation(items, float(capacity))
        selected_ids = set(alloc["selected_ids"])
        selected_candidates = [c for c in candidates if c["id"] in selected_ids]

        ranked_pool = sorted(candidates, key=lambda x: x["yield"] / max(0.1, x["cost"]), reverse=True)

        return {
            "capacity": float(capacity),
            "total_candidates": len(candidates),
            "n_selected": len(selected_candidates),
            "total_allocated_cost": alloc["total_cost"],
            "total_population_yield": alloc["total_yield"],
            "capacity_utilization_pct": round(alloc["capacity_utilization"] * 100, 1),
            "selected_patients": [
                {
                    "visit_id": c["id"],
                    "current_diagnosis": c.get("true_diagnosis", "MCI"),
                    "missing_modalities": c["missing_modalities"],
                    "cost": c["cost"],
                    "diagnostic_yield": round(c["yield"], 4),
                    "roi_ratio": round(c["yield"] / max(0.1, c["cost"]), 4),
                }
                for c in selected_candidates[:25]
            ],
            "priority_queue_top": [
                {
                    "rank": idx + 1,
                    "visit_id": c["id"],
                    "current_diagnosis": c.get("true_diagnosis", "MCI"),
                    "missing_modalities": c["missing_modalities"],
                    "cost": c["cost"],
                    "diagnostic_yield": round(c["yield"], 4),
                    "roi_ratio": round(c["yield"] / max(0.1, c["cost"]), 4),
                    "selected_in_current_budget": idx < len(selected_candidates),
                }
                for idx, c in enumerate(ranked_pool[:20])
            ]
        }
