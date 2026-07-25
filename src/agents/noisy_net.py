"""
Implement class that controls the noisy network
Source: https://scispace.com/pdf/noisy-networks-for-exploration-1zp75ueyjb.pdf
Noisy Networks For Exploration Fortunato Azar Piot et. al.

Notes:
- noisy network agent samples new set of parameters after every optimization step
  - between steps act according to fixed parameters (weights and biases)
  - agent should always act according to parameters drawn from current noise distribution
- replaces epsilon-greedy entirely
  - policy now greedily optimized randomized action value function
- noise function: suggests factorised faussian noise
- replay: current noisy network parameter sample fixed across batch
- DQN and dueling one step of optimization for every action step
  - noisy network parameters are re sampled before every action
  - reset before action selection?
- create alternatives: noisynet-DQN, noisynet-Dueling
  - noisynet-DQN is referencing double DQN
- replace all linear layers in value function with noisy linear layers
- from linear layer use in and out features to generate epsilon i and epsilon j used in reset_noise
- unchanged:
  - bellman loss
  - double DQN target
  - dueling decomposition (however modify linear layers)
- changed:
  - linear layer y = Wx + b to noisy layer, eqs 8 and 9 in the paper
  - gaussian noise to reduce compute time of rng in algs, use factorized eq 10 in paper
  - initialization of noisy network parameters: section 3.2
  - new parameters for Q(s, a; θ), replace θ={W,b} with params={mu, sigma}

- flow:
  - reset noise
  - generate new epsilon values
  - forward the new noisy layer
"""

import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp
)
from stable_baselines3.common.type_aliases import PyTorchObs

# NoisyQNetwork is very similar to dueling network (only with noisy not linear layers)
# credit to Hargun/Evan as most of this class is copied from their DuelingQNetwork
class NoisyQNetwork(BasePolicy):
  action_space: spaces.Discrete
  _forward_call_count = 0
  def __init__(
      self, 
      observation_space: spaces.Space, 
      action_space: spaces.Discrete,
      features_extractor: nn.Module | None = None,
      features_dim: int = 512,
      net_arch: list[list] | None=None,
      activation_fn: type[nn.Module] = nn.ELU,
      normalize_images: bool = True
    ) -> None:
    cnn = NatureCNN(observation_space, features_dim=features_dim)

    super().__init__(
      observation_space,
      action_space,
      features_extractor=cnn,
      normalize_images=normalize_images
    )

    self.features_extractor = cnn
    self.features_dim = cnn.features_dim

    if net_arch is None:
      net_arch = [64, 64]
    self.net_arch = net_arch
    self.activation_fn = activation_fn
    action_dim = int(self.action_space.n)

    shared_layers = create_mlp(
      self.features_dim,
      -1
      net_arch,
      activation_fn,
      squash_output=False
    )
    self.shared_net = nn.Sequential(*shared_layers)
    shared_dim = net_arch[-1] if net_arch else self.features_dim

    self.q_net = nn.Sequential(
      NoisyLinear(shared_dim, 128),
      activation_fn(),
      NoisyLinear(128, 1)
    )

    self.a_net = nn.Sequential(
      NoisyLinear(shared_dim, 128),
      activation_fn(),
      NoisyLinear(128, action_dim)
    )

  def forward(self, obs: PyTorchObs) -> th.Tensor:
    # pytorch should handle calling noisylinear forward
    features = self.extract_features(obs, self.features_extractor)
    shared = self.shared_net(features)
    value = self.q_net(shared)
    advantages = self.a_net(shared)

    q = value + (advantages - advantages.mean(dim=1, keepdim=True))

    NoisyQNetwork._forward_call_count += 1

  def predict(self, observation: PyTorchObs, deterministics: bool = True) -> th.Tensor:
    q_values = self(observation)
    return q_values.argmax(dim=1).reshape(-1)

  def _get_constructor_parameters(self) -> dict[str, Any]:
    data = super()._get_construction_parameters()
    data.update(
      dict(
        net_arch=self.net_arch,
        features_dim=self.features_dim,
        activation_fn=self.activation_fn,
        features_extractor = self.features_extractor
      )
    )
    return data

  def reset_noise(self):
    # reset noise for each module
    # pytorch should handle calling noisylinear reset noise



class NoisyLinear(nn.Module):
  def __init__(self):
    # initialize parameters and buffers

    # learned
    mu_weight = 0
    sigma_weight = 0
    mu_bias = 0
    sigma_bias = 0
    
    epsilon_weight = 0
    epsilon_bias = 0
    sigma_0 = 0.5
    in_features = 0
    out_features = 0
    
  def scale_noise(self):
    # helper function for reset_noise
    # functions from section 3.2
    # eq 10

  def reset_noise(self):
    # call scale_noise
    # set epsilon_weight and epsilon_bias

  def forward_layer(self):
    # use eqs 8 and 9
    # take values set in reset_noise and from init

class NoisyDoubleDuelingDQN(DQN):
  # combine noisy, double, and dueling? noisy needs to be put on top of this architecture
  # dueling is used in NoisyQNetwork, do we need to inherit the DuelingDQNPolicy?
  # modify DQN train function in here like DoubleDQN class, train should also reset noise?
  # q_net and q_net_target should use NoisyQNetwork

