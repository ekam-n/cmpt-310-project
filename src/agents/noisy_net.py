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

class NoisyNetwork:
  def __init__(self):
    # initialize parameters and buffers

    # learned
    mu_weight = 0
    sigma_weight = 0
    mu_bias = 0
    sigma_bias = 0

    scale_noise()
    
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

class NoisyDoubleDuelingDQN:
  # combine noisy, double, and dueling? noisy needs to be put on top of this architecture
