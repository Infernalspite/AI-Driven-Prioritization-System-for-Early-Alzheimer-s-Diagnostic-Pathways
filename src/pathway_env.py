"""
pathway_env.py
---------------
Phase 3: Module C, the Adaptive Diagnostic Pathway Engine, formalized as an
MDP per the plan:

    "State: current known feature vector (whatever tests exist so far) +
     uncertainty from Module D. Actions: {order blood biomarkers, order
     MRI, order PET, stop and diagnose}. Reward: +correct_diagnosis -
     lambda*cost(test) - mu*invasiveness(test)."

Extending prior art (the plan explicitly names this as the way to extend
"Efficient Alzheimer's Diagnosis Through Sequential Decision-Making with
Reinforcement Learning"): rather than feed the agent raw test values, the
STATE is the Phase 1 CrossAttentionFusion model's predicted class
probabilities given whatever modalities have been revealed so far. The
frozen Module A model is reused as-is (eval mode, no grad) -- ordering a
test changes what's revealed, Module A re-fuses, and the agent sees the
resulting (and shifting) diagnostic confidence. This is the concrete
"feed it the fused multimodal state from Module A" extension the plan
calls out.

Episodes are grounded in REAL patient data, not simulated test outcomes:
only visits where blood, MRI, AND PET were all actually collected are used
(so "ordering" a test reveals that patient's real recorded value, never an
invented one). Cognitive screening + demographics are always given for
free, matching the plan's 4-stage funnel where cognitive screening is
step 1 and always happens.

Costs/invasiveness (illustrative, not real billing data -- documented in
the Phase 3 README section):
    blood: cost 1.0, invasiveness 0.5   (cheapest, least invasive)
    MRI:   cost 3.0, invasiveness 1.0
    PET:   cost 8.0, invasiveness 3.0   (most expensive, most invasive)

Reward at "stop and diagnose": +reward_correct if the fusion model's
prediction from currently-revealed data matches the true diagnosis, else
-reward_wrong -- with an extra penalty for the clinically worse mistake of
missing a true AD case (a false negative on the highest-stakes class),
since a missed AD diagnosis and an unnecessary blood draw are not
equally costly errors.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from dataset import MODALITY_COLUMNS, STATIC_COLUMNS, LABEL_MAP, LABEL_NAMES

ESCALATION_MODALITIES = ["blood", "mri", "pet"]  # cognitive is always free/given
TEST_INFO = {
    "blood": {"cost": 1.0, "invasiveness": 0.5},
    "mri": {"cost": 3.0, "invasiveness": 1.0},
    "pet": {"cost": 8.0, "invasiveness": 3.0},
}
ACTION_NAMES = ["order_blood", "order_mri", "order_pet", "stop_and_diagnose"]


def configure_escalation_modalities(available_modalities):
    """
    Phase 5A: when running against OASIS-migrated data, PET (and blood, in
    OASIS's case) don't exist as an action at all -- not just missing at
    some visits, the way build_complete_cases() already handles, but
    structurally absent from the source data for every subject. Ordering
    an action that can never reveal a real value isn't a cost/invasiveness
    tuning question, it's an action space that shouldn't exist.

    Call this ONCE, before constructing any ADPathwayEnv / building
    complete-case data / training or loading a PPO agent, to rebuild
    ESCALATION_MODALITIES, TEST_INFO, and ACTION_NAMES to only the
    modalities actually present in the data. This mutates this module's
    globals (every function/class here reads them directly, not via a
    per-instance config), so it must be called before any of those
    functions run, and a checkpoint trained under one configuration should
    not be loaded under a different one (the action space size itself
    changes: len(available_modalities) + 1).

    Example (OASIS migration, no blood or PET):
        from pathway_env import configure_escalation_modalities
        configure_escalation_modalities(["mri"])
    """
    global ESCALATION_MODALITIES, TEST_INFO, ACTION_NAMES
    full_test_info = {
        "blood": {"cost": 1.0, "invasiveness": 0.5},
        "mri": {"cost": 3.0, "invasiveness": 1.0},
        "pet": {"cost": 8.0, "invasiveness": 3.0},
    }
    unknown = set(available_modalities) - set(full_test_info.keys())
    if unknown:
        raise ValueError(f"Unknown modalities requested: {unknown}. "
                          f"Known modalities: {list(full_test_info.keys())}")
    ESCALATION_MODALITIES = list(available_modalities)
    TEST_INFO = {m: full_test_info[m] for m in ESCALATION_MODALITIES}
    ACTION_NAMES = [f"order_{m}" for m in ESCALATION_MODALITIES] + ["stop_and_diagnose"]
    print(f"pathway_env: escalation action space reconfigured -> {ACTION_NAMES}")


def build_complete_cases(df, norm_stats):
    """Filters to visits with ALL modalities present (so every action in
    the episode reveals a real recorded value), and pre-normalizes."""
    escalation_cols = [c for m in ESCALATION_MODALITIES for c in MODALITY_COLUMNS[m]]
    complete = df.dropna(subset=escalation_cols).reset_index(drop=True)

    all_numeric_cols = [c for cols in MODALITY_COLUMNS.values() for c in cols] + STATIC_COLUMNS
    complete_norm = norm_stats.transform(complete, all_numeric_cols)
    return complete_norm


class ADPathwayEnv(gym.Env):
    """One episode = one patient visit. Agent sequentially decides which
    (if any) of blood/MRI/PET to order before stopping to diagnose."""

    def __init__(self, complete_df, fusion_model, device,
                 lambda_cost=0.05, mu_invasive=0.05,
                 reward_correct=15.0, reward_wrong=15.0, reward_missed_ad_extra=10.0,
                 redundant_action_penalty=0.5, max_steps=4, deterministic_order=False, seed=None):
        super().__init__()
        self.df = complete_df
        self.fusion_model = fusion_model
        self.device = device
        self.lambda_cost = lambda_cost
        self.mu_invasive = mu_invasive
        self.reward_correct = reward_correct
        self.reward_wrong = reward_wrong
        self.reward_missed_ad_extra = reward_missed_ad_extra
        self.redundant_action_penalty = redundant_action_penalty
        self.max_steps = max_steps
        self.deterministic_order = deterministic_order  # eval mode: iterate rows in order instead of sampling
        self._eval_cursor = 0

        self.action_space = spaces.Discrete(len(ESCALATION_MODALITIES) + 1)
        # obs = [P(CN), P(MCI), P(AD), one ordered-flag per escalation modality, step_frac]
        # Phase 5A note: this shape now derives from ESCALATION_MODALITIES rather than
        # being hardcoded to 7 -- call configure_escalation_modalities() BEFORE
        # constructing any env (e.g. for OASIS-migrated data, which has no PET/blood),
        # since a checkpoint trained under one action/observation space can't be
        # loaded under a different one.
        obs_dim = 3 + len(ESCALATION_MODALITIES) + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _pick_row(self):
        if self.deterministic_order:
            idx = self._eval_cursor % len(self.df)
            self._eval_cursor += 1
            return idx
        return int(self.rng.integers(0, len(self.df)))

    def _build_model_batch(self):
        """Builds a batch-of-1 input for the frozen Phase 1 fusion model
        reflecting which modalities are currently revealed."""
        row = self.df.iloc[self.row_idx]
        modality_features, modality_mask = {}, {}
        for m, cols in MODALITY_COLUMNS.items():
            vals = row[cols].values.astype(np.float32)
            revealed = True if m == "cognitive" else self.ordered.get(m, False)
            modality_features[m] = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(0)
            modality_mask[m] = torch.tensor([revealed], dtype=torch.bool, device=self.device)
        static_vals = row[STATIC_COLUMNS].values.astype(np.float32)
        static_features = torch.tensor(static_vals, dtype=torch.float32, device=self.device).unsqueeze(0)
        return {"modality_features": modality_features, "modality_mask": modality_mask,
                "static_features": static_features}

    def _fused_probs(self):
        batch = self._build_model_batch()
        with torch.no_grad():
            logits = self.fusion_model(batch)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs

    def _obs(self):
        probs = self._fused_probs()
        ordered_mask = np.array([float(self.ordered[m]) for m in ESCALATION_MODALITIES], dtype=np.float32)
        step_frac = np.array([self.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([probs.astype(np.float32), ordered_mask, step_frac])

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.row_idx = self._pick_row()
        self.ordered = {m: False for m in ESCALATION_MODALITIES}
        self.step_count = 0
        self.trajectory = []  # for demo/inspection: list of actions taken
        obs = self._obs()
        info = {"true_diagnosis": self.df.iloc[self.row_idx]["diagnosis"]}
        return obs, info

    def step(self, action):
        self.step_count += 1
        terminated = False
        info = {}

        if action < len(ESCALATION_MODALITIES):  # order a test
            modality = ESCALATION_MODALITIES[action]
            self.trajectory.append(ACTION_NAMES[action])
            if self.ordered[modality]:
                reward = -self.redundant_action_penalty
            else:
                self.ordered[modality] = True
                t = TEST_INFO[modality]
                reward = -(self.lambda_cost * t["cost"] + self.mu_invasive * t["invasiveness"])
        else:  # stop and diagnose
            self.trajectory.append(ACTION_NAMES[action])
            reward, info = self._diagnose_and_score()
            terminated = True

        if not terminated and self.step_count >= self.max_steps:
            # forced stop -- budget exhausted, must diagnose with whatever was revealed
            extra_reward, info = self._diagnose_and_score()
            reward += extra_reward
            terminated = True
            info["forced_stop"] = True

        obs = self._obs()
        truncated = False
        info.setdefault("true_diagnosis", self.df.iloc[self.row_idx]["diagnosis"])
        info["ordered"] = dict(self.ordered)
        info["n_tests_ordered"] = int(sum(self.ordered.values()))
        info["trajectory"] = list(self.trajectory)
        return obs, reward, terminated, truncated, info

    def _diagnose_and_score(self):
        probs = self._fused_probs()
        pred_idx = int(np.argmax(probs))
        pred_name = LABEL_NAMES[pred_idx]
        true_name = self.df.iloc[self.row_idx]["diagnosis"]
        correct = pred_name == true_name

        if correct:
            reward = self.reward_correct
        else:
            reward = -self.reward_wrong
            if true_name == "AD":
                reward -= self.reward_missed_ad_extra  # missing a true AD case is the costliest error

        total_cost = sum(TEST_INFO[m]["cost"] for m in ESCALATION_MODALITIES if self.ordered[m])
        total_invasive = sum(TEST_INFO[m]["invasiveness"] for m in ESCALATION_MODALITIES if self.ordered[m])

        info = {
            "predicted_diagnosis": pred_name, "true_diagnosis": true_name, "correct": correct,
            "predicted_probs": {LABEL_NAMES[i]: float(probs[i]) for i in range(3)},
            "total_cost": total_cost, "total_invasiveness": total_invasive,
        }
        return reward, info


class ADPathwayEnvUncertaintyAware(ADPathwayEnv):
    """
    Phase 4 -> Phase 3 integration point, per the plan:

        "Feed epistemic uncertainty directly into Module C's reward
         function -- high uncertainty raises the value of ordering
         another test."

    Rather than hand-craft an artificial escalation bonus (which would
    just be prescribing the answer the plan wants Module C to discover),
    this subclass enriches the STATE with Module D's epistemic uncertainty
    (a cheap MC Dropout estimate, n_mc_passes stochastic passes through
    the SAME frozen Module A model) at every step. The agent's existing
    reward already penalizes wrong diagnoses; if epistemic uncertainty is
    a genuine leading indicator of an error about to happen, PPO should
    learn on its own to spend the cost of another test when it sees high
    epistemic uncertainty -- exactly the plan's causal claim, made
    checkable rather than assumed.
    """

    def __init__(self, *args, n_mc_passes: int = 10, **kwargs):
        self.n_mc_passes = n_mc_passes
        super().__init__(*args, **kwargs)
        # obs = [P(CN), P(MCI), P(AD), one ordered-flag per escalation modality, step_frac, epistemic]
        # (Phase 5A: derives from ESCALATION_MODALITIES, same reasoning as the base class.)
        obs_dim = 3 + len(ESCALATION_MODALITIES) + 2
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

    def _fused_probs_and_epistemic(self):
        from uncertainty import mc_predict, decompose_uncertainty  # local import: avoid a hard dep for Phase 3-only use
        batch = self._build_model_batch()
        probs_stack = mc_predict(self.fusion_model, batch, n_passes=self.n_mc_passes)
        decomp = decompose_uncertainty(probs_stack)
        mean_probs = decomp["mean_probs"].squeeze(0).cpu().numpy()
        epistemic = float(decomp["epistemic"].squeeze(0).cpu().item())
        return mean_probs, epistemic

    def _fused_probs(self):
        # used by _diagnose_and_score -- keep the diagnosis itself based on the
        # MC-averaged probability too, for consistency with what the agent sees
        probs, self._last_epistemic = self._fused_probs_and_epistemic()
        return probs

    def _obs(self):
        probs, epistemic = self._fused_probs_and_epistemic()
        ordered_mask = np.array([float(self.ordered[m]) for m in ESCALATION_MODALITIES], dtype=np.float32)
        step_frac = np.array([self.step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([probs.astype(np.float32), ordered_mask, step_frac,
                                np.array([epistemic], dtype=np.float32)])
