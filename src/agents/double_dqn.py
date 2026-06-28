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

from stable_baselines3 import DQN
import torch as th
import torch.nn.functional as F
from common import fast_config as config
from datetime import datetime
import os
from baseline.train_baseline import make_vec
import numpy as np

from stable_baselines3.common.callbacks import (
   EvalCallback,
   CheckpointCallback,
   CallbackList
)

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
    # brought over from train_baseline py in src/baseline
    # changed to use DoubleDQN instead of DQN
    # changed to save to logs/double_dqn, callbacks and budget identical to baseline
    
    log_dir = os.path.join(config.LOG_ROOT, "double_dqn")
    os.makedirs(log_dir, exist_ok=True)

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

    model = DoubleDQN(
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