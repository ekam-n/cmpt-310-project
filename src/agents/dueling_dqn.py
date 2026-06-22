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


class DuelingQNetwork(nn.Module):
    """TODO (Hargun/Evan):
      1. Use a CNN feature extractor (SB3's NatureCNN is a good base).
      2. Build two heads: value_head (-> 1) and advantage_head (-> n_actions).
      3. Combine: Q = V + (A - A.mean(dim=1, keepdim=True)).
    """
    pass


def main():
    # TODO (Hargun/Evan): wire DuelingQNetwork into a custom DQNPolicy, then
    # mirror the train/eval setup in baseline/train_baseline.py.
    # Save to logs/dueling_dqn/ and keep callbacks + budget identical.
    raise NotImplementedError("Dueling DQN not implemented yet")


if __name__ == "__main__":
    main()
