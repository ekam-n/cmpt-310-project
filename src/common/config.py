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
# discrete action space (5 actions) -> required for DQN
CONTINUOUS = False
GRAYSCALE = True            # WarpFrame converts to 84x84 grayscale
N_STACK = 4                 # frame stacking for motion information
# parallel envs (keep at 1 for reproducibility/simplicity)
N_ENVS = 1

# --- Training budget (keep identical across all agents for fair comparison) ---
TOTAL_TIMESTEPS = 7_50_000
SEED = 42

# --- DQN hyperparameters (SB3 defaults shown explicitly so everyone sees them) ---
LEARNING_RATE = 5e-4
BUFFER_SIZE = 100_000
LEARNING_STARTS = 5_000     # steps of random play before learning begins
BATCH_SIZE = 64
# 1.0 = hard target update; <1.0 = Polyak (Ekam's research thread)
TAU = 1.0
GAMMA = 0.99                 # discount factor
TRAIN_FREQ = 4              # gradient update every N steps
GRADIENT_STEPS = 2
TARGET_UPDATE_INTERVAL = 500   # copy online -> target every N steps
EXPLORATION_FRACTION = 0.4       # fraction of training over which epsilon decays
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05

FEATURES_DIMS = 4
# --- Evaluation ---
EVAL_FREQ = 25_000          # run eval every N timesteps during training
N_EVAL_EPISODES = 20        # episodes averaged per evaluation
LAP_COMPLETE_PERCENT = 0.95  # % of tiles before a lap counts as complete

# Fixed evaluation tracks. CarRacing generates a new random track per reset, so
# evaluating two models on different tracks confounds the comparison with track
# luck. Every model in the final comparison is rolled out on THESE 20 seeds, in
# this order, so the numbers differ only by the agent. Do NOT edit this list
# once results are collected -- it would invalidate every earlier comparison.
EVAL_SEEDS = [
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
]

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
        # features_dims=FEATURES_DIMS,
    )
