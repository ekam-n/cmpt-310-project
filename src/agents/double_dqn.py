"""
Double DQN
Collaborators: Ekam + Lex

REASONING:
Base DQN systematically overestimates action values, as a result negatively
impacting performance and stability. One reason why this occurs is because 
the base system uses the target network to select and evaluate the next action.

Base DQN calculation: 
    target = r + gamma * max_a' Q_target(s', a') # selection and evaluation

As an alternative, double DQN uses an online network to select an action, and
the target network to evaluate it.

Double DQN calculations:
    a*     = argmax_a' Q_online(s', a')      # selection with online network
    target = r + gamma * Q_target(s', a*)     # evaluation with target network

IMPLEMENTATION:
- Imported DQN from stablebaseline3 and overrid train() function.
- Changed the th.no_grad() block in train() to use double DQN, changing the
way the next-state target values are calculated.
- Changed the main() function to use our edited DoubleDQN class instead of
stablebaseline3's DQN class.
- Keep hyperparameters and training identical to stablebaseline3 for fair
comparison (see common.config and common.env_factory).
- Modified the main() function from dqn.py to use our DoubleDQN class.

RUNNING:
Run from src/: python -m agents.double_dqn
"""
import argparse
from stable_baselines3 import DQN
import torch as th
import torch.nn.functional as F
from common import config
from datetime import datetime
import os
from baseline.train_baseline import make_vec
import numpy as np

from stable_baselines3.common.callbacks import (
   EvalCallback,
   CheckpointCallback,
   CallbackList
)

from common.activation_functions import available_activation_names, get_activation_fn
from common import run_tracking

class DoubleDQN(DQN):
    # override stable-baselines3/dqn/dqn.py train function
    # change next_q calculation
     def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # changed: use current online network to calculate a*
                # then use target network to evaluate Q value and get target value
                # stablebaseline3 defines a target and online network
                # self.q_net: online network
                # self.q_net_target: target network

                # use online network to determine action a*
                # use the one with highest value
                next_online_q_values = self.q_net(replay_data.next_observations)
                next_action = next_online_q_values.argmax(dim=1, keepdim=True)

                # evaluate Q value of action with target network
                new_target_q_value = self.q_net_target(replay_data.next_observations)
                next_q_values = new_target_q_value.gather(1, next_action) # replaces need for reshape as in DQN
                
                # keep these lines from stablebaseline3's implementation
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

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

    # brought over from train_baseline py in src/baseline
    # changed to use DoubleDQN instead of DQN
    # each run now gets its own logs/double_dqn/<tags>_<stamp>/ dir + manifest,
    # so two runs with different settings no longer overwrite each other

    ctx = run_tracking.start_run("double_dqn", args, "DoubleDQN")
    log_dir = str(ctx.run_dir)

    # Reward shaping is now an explicit, recorded per-run choice (--reward-wrapper),
    # not a hardcoded value. This script historically ran without the wrapper (off).
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

    model = DoubleDQN(
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