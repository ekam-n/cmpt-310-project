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

<<<<<<< HEAD
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
=======
# using same imports as original DQN/policies.py in case imports require a specific setup.
import os
from datetime import datetime

>>>>>>> hargun
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
<<<<<<< HEAD
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
=======

from common import config as config
from common.env_factory import make_vec
from typing import Any
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.dqn.policies import DQNPolicy

""" I had to change the config, reward wrappers and whatnot to get the thing to work 
    I have no idea what we want to do about this, I dont want to modify the files in the
    github for everyone so I am including the values here 

   
# --- Environment ---
ENV_ID = "CarRacing-v3"
CONTINUOUS = False
GRAYSCALE = True
N_STACK = 4
N_ENVS = 1

# --- Training budget ---
TOTAL_TIMESTEPS = 40_000
SEED = 42

# --- DQN hyperparameters (aggressive but stable for debugging) ---
LEARNING_RATE = 1e-4              # fast learning
# bigger buffer so turning experiences don't get pushed out
BUFFER_SIZE = 50_000
# enough random data to fill a batch, but not too long
LEARNING_STARTS = 500
BATCH_SIZE = 64                   # larger batch for stable gradients
TAU = 1.0
GAMMA = 0.99
TRAIN_FREQ = 1                    # update every step (faster learning)
# do 4 gradient updates per environment step (overfit to turning)
GRADIENT_STEPS = 4
# update target more often (faster propagation)
TARGET_UPDATE_INTERVAL = 500
EXPLORATION_FRACTION = 0.8        # keep exploring for most of training
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05      # still some exploration at the end

# --- Evaluation ---
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10              # fewer episodes = faster eval
LAP_COMPLETE_PERCENT = 0.95

# --- Paths ---
LOG_ROOT = "./logs"


    def __init__(self, env,
                 speed_coef=0.01,
                 smoothness_coef=-0.02,
                 on_track_reward=0.05,
                 movement_penalty=-0.05,
                 position_threshold=0.05,
                 progress_coef=0.1):
        print("using custom wrapper")
"""


class DuelingQNetwork(BasePolicy):
    """Dueling DQN Q-network with Value and Advantage streams."""
    action_space: spaces.Discrete
    _forward_call_count = 0

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: nn.Module | None = None,
        features_dim: int = 512,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ELU,
        normalize_images: bool = True,
    ) -> None:
        # so not sure why, following the formatting of the stablebaselines, the feature extractor should be passed to the Q-network, but when using it later it
        # didnt work properly. So I created a fresh one here, meaning if we want to change which feature extractor we use we'll have to change it here.
        cnn = NatureCNN(observation_space, features_dim=features_dim)

        # 2. Call parent, passing the CNN so extract_features works
        super().__init__(
            observation_space,
            action_space,
            features_extractor=cnn,
            normalize_images=normalize_images,
        )
        self.features_extractor = cnn
        self.features_dim = cnn.features_dim

        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        action_dim = int(self.action_space.n)

        # secong problem was here, using that paper I shared I learned that splitting A and V from the get go
        # leads to errors, so you use a shared network that splits later

        shared_layers = create_mlp(
            self.features_dim,
            -1,
            net_arch,
            activation_fn,
            squash_output=False,
        )
        self.shared_net = nn.Sequential(*shared_layers)
        shared_dim = net_arch[-1] if net_arch else self.features_dim

        self.q_net = nn.Sequential(
            nn.Linear(shared_dim, 128),
            activation_fn(),
            nn.Linear(128, 1)
        )

        self.a_net = nn.Sequential(
            nn.Linear(shared_dim, 128),
            activation_fn(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        # extract_features() and features_extractor()  are 2 functions that seem to do the same thing but this is the one that works
        features = self.extract_features(obs, self.features_extractor)
        # extract using a shared buffer, and then from there we split into Advantage and Value before recombining them
        shared = self.shared_net(features)
        value = self.q_net(shared)
        advantages = self.a_net(shared)

        q = value + (advantages - advantages.mean(dim=1, keepdim=True))

        DuelingQNetwork._forward_call_count += 1

        # this is a debugging print block, from what I've seen it doesnt effect performance that much and helps me so I'm keeping it in
        if DuelingQNetwork._forward_call_count % 10000 == 0:
            with th.no_grad():
                q_max_min = (q.max(dim=1).values -
                             q.min(dim=1).values).mean().item()
                print("\n" + "="*60)
                print(f"Forward call #{DuelingQNetwork._forward_call_count}")
                print(f"  Features std: {features.std().item():.4f}")
                print(f"  Value (V) mean: {value.mean().item():.4f}  std: {
                      value.std().item():.4f}")
                print(f"  Advantages (A) mean: {advantages.mean().item():.4f}  std: {
                      advantages.std().item():.4f}")
                print(f"  Q max-min diff: {q_max_min:.6f}")
                print(f"  Selected action: {q.argmax(dim=1)[0].item()}")
                print("="*60 + "\n")
        return q

    # 1 to 1 with the normal predict function for now, we can work on implementing a new one later maybe

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)

    # again, the features extractor here should be passing the NatureCNN but it isnt working?
    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DuelingDQNPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ELU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: dict[str, Any] | None = None,
        features_dim: int | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,  # ????? y
            features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self):
        return DuelingQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_dim=512,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
        )


def main():

    log_dir = os.path.join(config.LOG_ROOT, "dueling_dqn")
    os.makedirs(log_dir, exist_ok=True)

    eval_env = make_vec(use_reward_wrapper=True)
    train_env = make_vec(use_reward_wrapper=True)

>>>>>>> hargun
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

<<<<<<< HEAD
    # Create DQN with our custom dueling policy
    model = DQN(
        DuelingDQNPolicy,  # Use our custom policy instead of "CnnPolicy"
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **config.dqn_kwargs(),  # Use shared hyperparameters
    )

    print("Training Dueling DQN...")
=======
    # DQN has an array of policy, cant begin trainint without adding the child to said array
    DQN.policy_aliases["DuelingDQNPolicy"] = DuelingDQNPolicy

    model = DQN(
        "DuelingDQNPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **config.dqn_kwargs(),
    )

>>>>>>> hargun
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=CallbackList([ckpt_cb, eval_cb]),
    )

<<<<<<< HEAD
    # Save final model with timestamp
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    model.save(os.path.join(log_dir, f"final_model_{stamp}"))
    print(f"\nSaved dueling DQN model to {log_dir}")
=======
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    model.save(os.path.join(log_dir, f"final_model_{stamp}"))
    print(f"\nSaved baseline model to {log_dir}")
    print("Best model (by eval reward) is at best_model.zip in the same dir.")
>>>>>>> hargun

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()


# TODO (Hargun/Evan): wire DuelingQNetwork into a custom DQNPolicy, then
# mirror the train/eval setup in baseline/train_baseline.py.
# Save to logs/dueling_dqn/ and keep callbacks + budget identical.
