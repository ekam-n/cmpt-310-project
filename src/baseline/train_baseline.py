"""
Baseline: vanilla SB3 DQN on CarRacing-v3.

This is the reference point the whole project is measured against. SB3's DQN
is plain Deep Q-Learning -- no Double DQN, no Dueling, no prioritized replay.
Your contributions (Double DQN, Dueling DQN, reward shaping, alt averaging)
all get compared back to THIS run, trained under the same conditions.

Run from the src/ directory:
    python -m baseline.train_baseline
"""

import os
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from common import config
from common.env_factory import make_vec

from agents import DoubleDQN
from agents import DuelingDQNPolicy
from agents import NoisyDoubleDuelingDQN
from agents import NoisyDQNPolicy
from common import available_activation_names, get_activation_fn

import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
    "--algorithm",
    default="baseline",
    choices=["baseline", "double", "dueling", "noisy"],
    help="Which DQN variant to train.",
    )
    parser.add_argument(
        "--activation",
        default="elu",
        choices=available_activation_names(),
        help="Activation function used by the policy network.",
    )
    args = parser.parse_args()

    algorithm_dirs = {
    "baseline": "baseline_dqn_1-000",
    "double": "double_dqn_1-000",
    "dueling": "dueling_dqn_1-000",
    "noisy": "noisy_dd_dqn_1-000",}

    log_dir = os.path.join(config.LOG_ROOT, algorithm_dirs[args.algorithm])
    os.makedirs(log_dir, exist_ok=True)

    # Use default reward (no shaping) for the baseline so contributions that
    # change the reward have a clean reference. Flip use_reward_wrapper if your
    # group decides the baseline should include the passthrough wrapper.
    train_env = make_vec(use_reward_wrapper=False)
    eval_env = make_vec(use_reward_wrapper=False)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=config.EVAL_FREQ,
        save_path=os.path.join(log_dir, "checkpoints"),
    )

    common_kwargs = dict(
    verbose=1,
    tensorboard_log=os.path.join(log_dir, "tensorboard"),
    policy_kwargs=dict(
        activation_fn=get_activation_fn(args.activation)
    ),
    **config.dqn_kwargs(),
    )

    if args.algorithm == "baseline":
        model = DQN(
            "CnnPolicy",
            train_env,
            **common_kwargs,
            )
    elif args.algorithm == "double":
        model = DoubleDQN(
            "CnnPolicy",
            train_env,
            **common_kwargs,
        )
    elif args.algorithm == "dueling":
        DQN.policy_aliases["DuelingDQNPolicy"] = DuelingDQNPolicy

        model = DQN(
            "DuelingDQNPolicy",
            train_env,
            **common_kwargs,

        )
    elif args.algorithm == "noisy":
        DQN.policy_aliases["NoisyDQNPolicy"] = NoisyDQNPolicy
        model = NoisyDoubleDuelingDQN(
            "NoisyDQNPolicy",
            train_env,
            **common_kwargs,
        )

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=CallbackList([ckpt_cb, eval_cb]),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    model.save(os.path.join(log_dir, f"final_model_{stamp}"))
    print(f"\nSaved baseline model to {log_dir}")
    print("Best model (by eval reward) is at best_model.zip in the same dir.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
