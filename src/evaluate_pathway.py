"""
evaluate_pathway.py
---------------------
Head-to-head test-set evaluation for Phase 3, same honest-comparison style
as Phases 1-2. Four policies run over the SAME held-out, complete-case test
visits (blood/MRI/PET all actually recorded, so every "order" reveals a
real value):

1. **PPO pathway agent** -- the trained Module C policy.
2. **Always-full-workup** -- order blood, MRI, AND PET every time, then
   diagnose. This is literally the plan's "standard 4-stage funnel for
   every patient," i.e. what Module C is trying to improve on.
3. **Cognitive-only** -- stop immediately, diagnose from cognitive +
   demographics alone (0 extra cost, 0 extra invasiveness). The cheapest
   possible extreme; shows what accuracy costs if you escalate nothing.
4. **Greedy-confidence heuristic** -- the plan's own named fallback
   ("a contextual bandit / greedy expected-information-gain heuristic
   dressed as a simplified MDP is acceptable to demo if full DQN training
   doesn't converge"): order blood -> MRI -> PET in that fixed priority
   order until the fused model's max class probability clears a confidence
   threshold (0.75, chosen by inspection of val-set confidence
   distribution, not tuned against test), then stop.

Reports per policy: diagnostic accuracy, average tests ordered, average
raw cost/invasiveness (not the RL reward -- the reward already bundles cost
and invasiveness into one scalar, which is useful for training but not for
reading off a demo table), and average RL-style reward for reference.
"""

import argparse
import json

import numpy as np
import torch
from stable_baselines3 import PPO

from load_fusion import load_frozen_fusion
from pathway_env import ADPathwayEnv, ADPathwayEnvUncertaintyAware, build_complete_cases, ESCALATION_MODALITIES, TEST_INFO
from train_pathway import load_split_dfs


def run_episode(env, policy_fn):
    obs, info = env.reset()
    total_reward = 0.0
    terminated = False
    while not terminated:
        action = policy_fn(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return {
        "reward": total_reward,
        "correct": info["correct"],
        "predicted_diagnosis": info["predicted_diagnosis"],
        "true_diagnosis": info["true_diagnosis"],
        "n_tests_ordered": info["n_tests_ordered"],
        "total_cost": info["total_cost"],
        "total_invasiveness": info["total_invasiveness"],
        "trajectory": info["trajectory"],
    }


def policy_always_full(obs, info, env):
    for i, m in enumerate(ESCALATION_MODALITIES):
        if not env.ordered[m]:
            return i
    return 3  # all ordered -> stop


def policy_cognitive_only(obs, info, env):
    return 3  # always stop immediately


def make_policy_greedy_confidence(threshold=0.75):
    def _policy(obs, info, env):
        probs = obs[:3]
        if max(probs) >= threshold:
            return 3
        for i, m in enumerate(ESCALATION_MODALITIES):
            if not env.ordered[m]:
                return i
        return 3  # nothing left to order
    return _policy


def make_policy_ppo(ppo_model):
    def _policy(obs, info, env):
        action, _ = ppo_model.predict(obs, deterministic=True)
        return int(action)
    return _policy


def run_policy_over_test_set(env, policy_fn, n_episodes):
    results = [run_episode(env, policy_fn) for _ in range(n_episodes)]
    accs = [r["correct"] for r in results]
    return {
        "accuracy": float(np.mean(accs)),
        "avg_n_tests_ordered": float(np.mean([r["n_tests_ordered"] for r in results])),
        "avg_total_cost": float(np.mean([r["total_cost"] for r in results])),
        "avg_total_invasiveness": float(np.mean([r["total_invasiveness"] for r in results])),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "n_episodes": n_episodes,
        "examples": results[:6],
        # Phase 5D validation gate criterion 3 needs to check whether the PPO
        # agent ever finds escalating to MRI/PET worth it -- that requires
        # every episode's trajectory + reward, not just the first 6 examples.
        "all_episode_trajectories": [
            {"reward": r["reward"], "correct": r["correct"], "trajectory": r["trajectory"],
             "n_tests_ordered": r["n_tests_ordered"]}
            for r in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 pathway policies on the test set")
    parser.add_argument("--fusion_checkpoint", type=str, default="checkpoints/fusion_best.pt")
    parser.add_argument("--ppo_checkpoint", type=str, default="checkpoints/pathway_ppo.zip")
    parser.add_argument("--lambda_cost", type=float, default=0.05)
    parser.add_argument("--mu_invasive", type=float, default=0.05)
    parser.add_argument("--confidence_threshold", type=float, default=0.75)
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--uncertainty_ppo_checkpoint", type=str, default="checkpoints/pathway_ppo_uncertainty.zip",
                         help="Phase 4 integration agent; skipped if the file doesn't exist.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model, norm_stats = load_frozen_fusion(args.fusion_checkpoint, device)

    _, _, test_df = load_split_dfs(args.data_dir)
    complete_test = build_complete_cases(test_df, norm_stats)
    n_episodes = len(complete_test)
    print(f"Test episodes (complete-case visits, test split): {n_episodes}")

    env_kwargs = dict(lambda_cost=args.lambda_cost, mu_invasive=args.mu_invasive, deterministic_order=True)

    def fresh_env():
        return ADPathwayEnv(complete_test, fusion_model, device, seed=0, **env_kwargs)

    def fresh_uncertainty_env():
        return ADPathwayEnvUncertaintyAware(complete_test, fusion_model, device, seed=0, **env_kwargs)

    ppo_model = PPO.load(args.ppo_checkpoint, device=device)

    policies = {
        "ppo_agent": (make_policy_ppo(ppo_model), fresh_env),
        "always_full_workup": (policy_always_full, fresh_env),
        "cognitive_only": (policy_cognitive_only, fresh_env),
        "greedy_confidence_heuristic": (make_policy_greedy_confidence(args.confidence_threshold), fresh_env),
    }

    import os
    if os.path.exists(args.uncertainty_ppo_checkpoint):
        uncertainty_ppo_model = PPO.load(args.uncertainty_ppo_checkpoint, device=device)
        policies["ppo_agent_uncertainty_aware"] = (make_policy_ppo(uncertainty_ppo_model), fresh_uncertainty_env)

    report = {}
    for name, (policy_fn, env_factory) in policies.items():
        env = env_factory()
        report[name] = run_policy_over_test_set(env, policy_fn, n_episodes)
        r = report[name]
        print(f"{name:28s} acc {r['accuracy']:.3f}  avg_tests {r['avg_n_tests_ordered']:.2f}  "
              f"avg_cost {r['avg_total_cost']:.2f}  avg_reward {r['avg_reward']:.2f}")

    report["_meta"] = {
        "confidence_threshold_used": args.confidence_threshold,
        "lambda_cost": args.lambda_cost, "mu_invasive": args.mu_invasive,
        "test_costs": TEST_INFO,
        "note": ("All policies evaluated on the SAME complete-case test visits (blood/MRI/PET all "
                 "actually recorded), iterated once each -- deterministic, not sampled. Accuracy differences "
                 "reflect real recorded modality values revealed progressively, not simulated outcomes."),
    }

    with open("outputs/phase3_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote outputs/phase3_comparison_report.json")


if __name__ == "__main__":
    main()
