"""
train_pathway.py
------------------
Trains a PPO agent (stable-baselines3, per the plan's tech stack) on
ADPathwayEnv -- the plan's Module C, "Efficient Alzheimer's Diagnosis
Through Sequential Decision-Making with Reinforcement Learning" extended to
consume Module A's fused diagnostic state (see pathway_env.py / load_fusion.py).

Uses the SAME Phase 0 subject splits as Phases 1-2, restricted to visits
where blood/MRI/PET were all actually recorded (so the agent only ever
"orders" a test that has a real value to reveal). Train/val/test subject
sets never overlap; the frozen Module A fusion model was itself only ever
trained on the train split.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from load_fusion import load_frozen_fusion
from pathway_env import ADPathwayEnv, ADPathwayEnvUncertaintyAware, build_complete_cases


def load_split_dfs(data_dir="data/raw"):
    df = pd.read_csv(f"{data_dir}/cleaned_dataset.csv")
    with open(f"{data_dir}/subject_splits.json") as f:
        splits = json.load(f)
    train_df = df[df["subject_id"].isin(splits["train_subjects"])].copy()
    val_df = df[df["subject_id"].isin(splits["val_subjects"])].copy()
    test_df = df[df["subject_id"].isin(splits["test_subjects"])].copy()
    return train_df, val_df, test_df


def make_env_fn(complete_df, fusion_model, device, env_kwargs, seed, uncertainty_aware=False):
    env_cls = ADPathwayEnvUncertaintyAware if uncertainty_aware else ADPathwayEnv

    def _init():
        env = env_cls(complete_df, fusion_model, device, seed=seed, **env_kwargs)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Phase 3 PPO pathway agent")
    parser.add_argument("--total_timesteps", type=int, default=60000)
    parser.add_argument("--lambda_cost", type=float, default=0.05)
    parser.add_argument("--mu_invasive", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion_checkpoint", type=str, default="checkpoints/fusion_best.pt")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--uncertainty_aware", action="store_true",
                         help="Phase 4 integration: augment state with MC-Dropout epistemic uncertainty.")
    parser.add_argument("--output_name", type=str, default="pathway_ppo",
                         help="checkpoint filename stem, e.g. pathway_ppo or pathway_ppo_uncertainty")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_model, norm_stats = load_frozen_fusion(args.fusion_checkpoint, device)

    train_df, _, _ = load_split_dfs(args.data_dir)
    complete_train = build_complete_cases(train_df, norm_stats)
    print(f"Training episodes available (complete-case visits, train split): {len(complete_train)}")

    env_kwargs = dict(lambda_cost=args.lambda_cost, mu_invasive=args.mu_invasive)
    env = make_vec_env(
        make_env_fn(complete_train, fusion_model, device, env_kwargs, args.seed, args.uncertainty_aware),
        n_envs=1, seed=args.seed,
    )

    model = PPO(
        "MlpPolicy", env, verbose=1, seed=args.seed,
        n_steps=512, batch_size=64, n_epochs=10, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
    )
    model.learn(total_timesteps=args.total_timesteps)
    model.save(f"{args.checkpoint_dir}/{args.output_name}")
    print(f"\nSaved PPO pathway agent -> {args.checkpoint_dir}/{args.output_name}.zip")


if __name__ == "__main__":
    main()
