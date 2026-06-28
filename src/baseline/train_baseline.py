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

from src.common import config
from src.common.env_factory import make_vec


def main():
    log_dir = os.path.join(config.LOG_ROOT, "baseline_dqn")
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

    model = DQN(
        "CnnPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **config.dqn_kwargs(),
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
