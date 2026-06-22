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
CONTINUOUS = False          # discrete action space (5 actions) -> required for DQN
GRAYSCALE = True            # WarpFrame converts to 84x84 grayscale
N_STACK = 4                 # frame stacking for motion information
N_ENVS = 1                  # parallel envs (keep at 1 for reproducibility/simplicity)

# --- Training budget (keep identical across all agents for fair comparison) ---
TOTAL_TIMESTEPS = 500_000
SEED = 42

# --- DQN hyperparameters (SB3 defaults shown explicitly so everyone sees them) ---
LEARNING_RATE = 1e-4
BUFFER_SIZE = 100_000
LEARNING_STARTS = 10_000     # steps of random play before learning begins
BATCH_SIZE = 32
TAU = 1.0                    # 1.0 = hard target update; <1.0 = Polyak (Ekam's research thread)
GAMMA = 0.99                 # discount factor
TRAIN_FREQ = 4              # gradient update every N steps
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1_000   # copy online -> target every N steps
EXPLORATION_FRACTION = 0.2       # fraction of training over which epsilon decays
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05

# --- Evaluation ---
EVAL_FREQ = 25_000          # run eval every N timesteps during training
N_EVAL_EPISODES = 20        # episodes averaged per evaluation
LAP_COMPLETE_PERCENT = 0.95  # % of tiles before a lap counts as complete

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
