"""
Dueling DQN  --  owners: Hargun + Evan

THE IDEA (what you need to implement):
Instead of the network outputting Q(s,a) directly, split the final part into
two streams (matches your handwritten notes):

  - Value stream      V(s)      -> one number: how good is this state
  - Advantage stream  A(s,a)    -> one per action: how much better is action a

Recombine with the mean-advantage stability constraint:

    Q(s,a) = V(s) + ( A(s,a) - mean_a' A(s,a') )

This architecture lets the network learn state value separate from action
advantages, improving learning efficiency and generalization.

Run from src/ once implemented:
    python -m agents.dueling_dqn
"""

import os
import sys
from datetime import datetime

from pathlib import Path

# Ensure the project `src/` directory is on sys.path so imports work
# whether the script is run as a script or with `python -m` from project root.
src_path = str(Path(__file__).resolve().parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import NatureCNN

from common import config
from common.env_factory import make_vec


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, features_dim: int = 512):
        super().__init__()
        self.cnn = NatureCNN(observation_space, features_dim=features_dim)
        n_actions = action_space.n
        self.value_head = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1), # V(s) -> single scalar
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(features_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions), # A(s,a) -> one per action
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute Q(s,a) = V(s) + [A(s,a) - mean(A)].
        
        Args:
            observations: Batch of observations [batch_size, channels, height, width]
            
        Returns:
            Q-values [batch_size, n_actions]
        """
        features = self.cnn(observations)
        
        value = self.value_head(features)  # [batch_size, 1]
        advantages = self.advantage_head(features)  # [batch_size, n_actions]
        
        # Normalize advantages by subtracting their mean
        advantages = advantages - advantages.mean(dim=1, keepdim=True)
        
        # Combine: Q(s,a) = V(s) + [A(s,a) - mean(A)]
        q_values = value + advantages
        
        return q_values


class DuelingDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net_architecture(self) -> None:
        """Override: use our dueling Q-network instead of linear Q-network."""
        # Replace the default Q-network with our dueling version
        self.q_net = DuelingQNetwork(
            self.observation_space,
            self.action_space,
            features_dim=512,
        )
        self.q_net = self.q_net.to(self.device)

        # Target network (same architecture)
        self.q_net_target = DuelingQNetwork(
            self.observation_space,
            self.action_space,
            features_dim=512,
        )
        self.q_net_target = self.q_net_target.to(self.device)


def main():
    """Train dueling DQN
    
    Mirrors the baseline training setup but with our custom dueling policy.
    Saves logs and best model to logs/dueling_dqn/.
    """
    log_dir = os.path.join(config.LOG_ROOT, "dueling_dqn")
    os.makedirs(log_dir, exist_ok=True)

    # Create train and eval environments (no reward shaping for fair comparison)
    train_env = make_vec(use_reward_wrapper=False)
    eval_env = make_vec(use_reward_wrapper=False)

    # Callbacks for evaluation and checkpointing
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

    # Create DQN with our custom dueling policy
    model = DQN(
        DuelingDQNPolicy,  # Use our custom policy instead of "CnnPolicy"
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **config.dqn_kwargs(),  # Use shared hyperparameters
    )

    print("Training Dueling DQN...")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=CallbackList([ckpt_cb, eval_cb]),
    )

    # Save final model with timestamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    model.save(os.path.join(log_dir, f"final_model_{stamp}"))
    print(f"\nSaved dueling DQN model to {log_dir}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
