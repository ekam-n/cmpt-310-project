"""
Double DQN  --  owners: Ekam + Lex

THE IDEA (what you need to implement):
Vanilla DQN computes the target using the target network to BOTH select and
evaluate the next action:

    target = r + gamma * max_a' Q_target(s', a')

This overestimates Q-values. Double DQN decouples the two roles:

    a*     = argmax_a' Q_online(s', a')      # online net SELECTS
    target = r + gamma * Q_target(s', a*)     # target net EVALUATES

HOW TO IMPLEMENT IT IN SB3:
Subclass DQN and override train(), changing ONLY the next-state target value
computation. Everything else (env, hyperparameters, training budget) must stay
identical to the baseline so the comparison is fair -- pull all of that from
common.config and common.env_factory.

Run from src/ once implemented:
    python -m agents.double_dqn
"""

from stable_baselines3 import DQN


class DoubleDQN(DQN):
    """DQN with the Double-DQN target.

    TODO (Ekam/Lex):
      - Override train() (start from SB3's DQN.train as a reference).
      - In the no-grad target block, replace the vanilla
            next_q = Q_target(s').max(...)
        with the Double DQN version:
            a* = Q_online(s').argmax(...)
            next_q = gather(Q_target(s'), a*)
      - Leave the loss, optimizer step, and target-net update as in vanilla DQN.
    """
    pass


def main():
    # TODO (Ekam/Lex): mirror the train/eval setup in
    # baseline/train_baseline.py, but instantiate DoubleDQN instead of DQN.
    # Save to logs/double_dqn/ and keep callbacks + budget identical to baseline.
    raise NotImplementedError("Double DQN not implemented yet")


if __name__ == "__main__":
    main()
