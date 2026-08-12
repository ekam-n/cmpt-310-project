"""
Baseline: vanilla SB3 DQN on CarRacing-v3.

This is the reference point the whole project is measured against. SB3's DQN
is plain Deep Q-Learning -- no Double DQN, no Dueling, no prioritized replay.
Your contributions (Double DQN, Dueling DQN, reward shaping, alt averaging)
all get compared back to THIS run, trained under the same conditions.

Run from the src/ directory:
    python -m baseline.train_baseline
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

from common.activation_functions import available_activation_names, get_activation_fn
from common import config as config
from common import run_tracking
from common.env_factory import make_vec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation",
        default="elu",
        choices=available_activation_names(),
        help="Activation function used by the policy network.",
    )
    run_tracking.add_common_args(parser)
    args = parser.parse_args()

    # Every run gets its OWN directory (logs/baseline_dqn/<tags>_<stamp>/) plus a
    # run_config.json manifest, so repeated runs no longer overwrite each other.
    ctx = run_tracking.start_run("baseline_dqn", args, "DQN")
    log_dir = str(ctx.run_dir)

    # Reward shaping is now an explicit, recorded per-run choice (--reward-wrapper),
    # not a hardcoded value. This script historically used the wrapper (on).
    use_reward_wrapper = args.reward_wrapper == "on"
    train_env = make_vec(seed=args.seed, use_reward_wrapper=use_reward_wrapper)
    eval_env = make_vec(seed=args.seed, use_reward_wrapper=use_reward_wrapper)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=os.path.join(log_dir, "checkpoints"),
    )

    model = DQN(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        policy_kwargs=dict(activation_fn=get_activation_fn(args.activation)),
        **{**config.dqn_kwargs(), "seed": args.seed},
    )

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=CallbackList([ckpt_cb, eval_cb]),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    model.save(os.path.join(log_dir, f"final_model_{stamp}"))
    print(f"\nSaved baseline model to {log_dir}")
    print("Best model (by eval reward) is at best_model.zip in the same dir.")

    # Writes results.json: best eval reward + its timestep + wall-clock seconds.
    run_tracking.finish_run(ctx)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
