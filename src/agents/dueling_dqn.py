"""
Dueling DQN  --  owners: Hargun + Evan

THE IDEA (what you need to implement):
Instead of the network outputting Q(s,a) directly, split the final part into
two streams (matches your handwritten notes):

  - Value stream      V(s)      -> one number: how good is this state
  - Advantage stream  A(s,a)    -> one per action: how much better is action a

Recombine with the mean-advantage stability constraint:

    Q(s,a) = V(s) + ( A(s,a) - mean_a' A(s,a') )

HOW TO IMPLEMENT IT IN SB3:
Unlike Double DQN (which changes the target math), Dueling DQN changes the
NETWORK ARCHITECTURE. The clean path is a custom Q-network with the two
streams, wired in via a custom DQNPolicy and passed to DQN. Keep env,
hyperparameters, and training budget identical to the baseline (pull from
common.config and common.env_factory) so the comparison is fair.

Run from src/ once implemented:
    python -m agents.dueling_dqn
"""

import torch.nn as nn

# using same imports as original DQN/policies.py in case imports require a specific setup.
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


# Starting from the base policy so we have a framework for what to implement
class DuelingQNetwork(BasePolicy):
    """TODO (Hargun/Evan):
      1. Use a CNN feature extractor (SB3's NatureCNN is a good base).
      2. Build two heads: value_head (-> 1) and advantage_head (-> n_actions).
      3. Combine: Q = V + (A - A.mean(dim=1, keepdim=True)).
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: NatureCNN,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            # no idea what this effects later so just using the same as the default
            net_arch = [64, 64]
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        # in this case its 5 the discreet actions
        action_dim = self.action_space.n

        # this is where we start to differ from the baseline, like the above comment says we need 2 parralel networks

        # this is the "value head"
        q_net = create_mlp(self.features_dim, action_dim,
                           self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

        # I think this doesnt need all the dimensions of action_dim?
        # this is the "action head"
        a_net = create_mlp(self.features_dim, action_dim,
                           self.net_arch, self.activation_fn)
        self.a_net = nn.Sequential(*a_net)

    # this is where we  need to change the architecture to use both of the above mlp's

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        pass

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        pass

    def _get_constructor_parameters(self) -> dict[str, Any]:
        pass


def main():

    # TODO (Hargun/Evan): wire DuelingQNetwork into a custom DQNPolicy, then
    # mirror the train/eval setup in baseline/train_baseline.py.
    # Save to logs/dueling_dqn/ and keep callbacks + budget identical.
    raise NotImplementedError("Dueling DQN not implemented yet")


if __name__ == "__main__":
    main()
