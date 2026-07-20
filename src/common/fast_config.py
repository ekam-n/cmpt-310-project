"""
Shared configuration for the whole group.

This is the SINGLE SOURCE OF TRUTH for training conditions. Every agent
(baseline DQN, Double DQN, Dueling DQN, ...) imports from here so that the
ONLY thing differing between experiments is the algorithm itself -- not the
environment, not the hyperparameters, not the training budget.

This is exactly what the prof meant by "compare against the baseline using
the same training conditions." Do NOT hardcode hyperparameters in your own
agent files; change them here (or override explicitly and document why).
"""

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


def dqn_kwargs():
    """Return the shared DQN hyperparameters as a dict.

    Pass this straight into DQN(...) (or a subclass) so every agent starts
    from identical settings. Override individual keys only when your
    contribution requires it, and note it in your agent file + the writeup.
    """
    return dict(
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=EXPLORATION_INITIAL_EPS,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        seed=SEED,
    )
