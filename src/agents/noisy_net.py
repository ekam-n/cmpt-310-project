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

# imports for NoisyQNetwork and NoisyLinear
import torch as th
import numpy as np
from stable_baselines3 import DQN
from gymnasium import spaces
from torch import nn
rng = np.random.default_rng()
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp
)

# remaining imports for NoisyDoubleDuelingDQN and main
import argparse
import torch.nn.functional as F
import os
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from common import config
from datetime import datetime
from typing import Any
from baseline.train_baseline import make_vec
from stable_baselines3.common.callbacks import (
   EvalCallback,
   CheckpointCallback,
   CallbackList
)
from common.activation_functions import available_activation_names, get_activation_fn

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
      -1,
      net_arch,
      activation_fn,
      squash_output=False
    )
    self.shared_net = nn.Sequential(*shared_layers)
    shared_dim = net_arch[-1] if net_arch else self.features_dim

    # change q_net and a_net so they use NoisyLinear layers instead of Linear
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
    return q

  def predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
    q_values = self(observation)

    if deterministic:
      self.reset_noise()
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
    # need to get all modules, check if they are a noisylinear layer, then call reset if it is
    for module in self.modules():
      if isinstance(module, NoisyLinear):
        module.reset_noise() # calls the NoisyLinear reset noise function

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    # initialize parameters and buffers
    self.in_features = in_features
    self.out_features = out_features
    self.register_buffer(
    "epsilon_weight",
    th.zeros(out_features, in_features)
    )
    self.register_buffer(
    "epsilon_bias",
    th.zeros(out_features)
    )

    # learned
    # need to ensure the sizes are correct, use nn.Parameter
    sigma_init = 0.5 / np.sqrt(self.in_features) # from paper
    mu_bound = 1 / np.sqrt(self.in_features) # from paper
    self.mu_weight = nn.Parameter(th.empty(out_features, in_features))
    nn.init.uniform_(self.mu_weight, -mu_bound, mu_bound) # from paper
    self.sigma_weight = nn.Parameter(th.empty(out_features, in_features))
    with th.no_grad():
      self.sigma_weight.fill_(sigma_init)

    self.mu_bias = nn.Parameter(th.empty(out_features))
    nn.init.uniform_(self.mu_bias, -mu_bound, mu_bound)
    self.sigma_bias = nn.Parameter(th.empty(out_features))
    with th.no_grad():
      self.sigma_bias.fill_(sigma_init)
    
  def scale_noise(self, epsilon):
    # helper function for reset_noise
    # functions from section 3.2
    # eq 10

    # f(x) = sign(x) * sqrt(abs(x)) from paper
    return th.sign(epsilon) * th.sqrt(th.abs(epsilon))


  def reset_noise(self):
    # set epsilon_weight and epsilon_bias
    # epsilon_weight = epsilon_j * epsilon_i ^T
    # epsilon_bias = epsilon_j
    epsilon_i = th.randn(self.in_features)
    epsilon_j = th.randn(self.out_features)
    epsilon_i = self.scale_noise(epsilon_i)
    epsilon_j = self.scale_noise(epsilon_j)

    self.epsilon_weight.copy_(th.outer(epsilon_j, epsilon_i)) # from paper
    self.epsilon_bias.copy_(epsilon_j) # from paper


  def forward(self, input):
    # use eqs 8 and 9
    # take values set in reset_noise and from init
    weight = self.mu_weight + self.sigma_weight * self.epsilon_weight # from paper
    bias = self.mu_bias + self.sigma_bias * self.epsilon_bias # from paper

    return F.linear(input, weight, bias)


# again credit to Hargun/Evan for the __init__, and make_q_net additions from dueling
class NoisyDoubleDuelingDQN(DQN):
  # combine noisy, double, and dueling? noisy needs to be put on top of this architecture
  # dueling is used in NoisyQNetwork, do we need to inherit the DuelingDQNPolicy?
  # modify DQN train function in here like DoubleDQN class, train should also reset noise?
  # q_net and q_net_target should use NoisyQNetwork
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
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        net_arch=net_arch,
        activation_fn=activation_fn,
        features_extractor_class=features_extractor_class,  # ????? y
        features_extractor_kwargs=features_extractor_kwargs,
        normalize_images=normalize_images,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
    )

  def make_q_net(self):
    return NoisyQNetwork(
        observation_space=self.observation_space,
        action_space=self.action_space,
        features_dim=512,
        net_arch=self.net_arch,
        activation_fn=self.activation_fn,
        normalize_images=self.normalize_images,
    )

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

      # here is the double DQN additions
      with th.no_grad():
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
      # paper recommends to reset noise after optimizer
      self.q_net.reset_noise()
      self.q_net_target.reset_noise()

    # Increase update counter
    self._n_updates += gradient_steps

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/loss", np.mean(losses))

# credit to hargun/evan, this main function is from dueling and edited slightly
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    log_dir = os.path.join(config.LOG_ROOT, "NoisyDD_dqn")
    os.makedirs(log_dir, exist_ok=True)

    eval_env = make_vec(use_reward_wrapper=True)
    train_env = make_vec(use_reward_wrapper=True)

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
   # DQN has an array of policy, cant begin trainint without adding the child to said array
    DQN.policy_aliases["NoisyDQNPolicy"] = NoisyQNetwork # network replacement

    model = NoisyDoubleDuelingDQN( # model replacement
        "NoisyDQNPolicy",
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