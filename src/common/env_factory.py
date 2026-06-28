"""
Shared environment factory.

Every agent builds its env through make_env() / make_vec() here, so the
observation pipeline (grayscale, frame stacking, channel order) is IDENTICAL
across the baseline and every contribution. If you change preprocessing,
change it HERE so the comparison stays fair.
"""

import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import WarpFrame

from common import config
from envs.reward_wrapper import CarRacingRewardWrapper


def make_env(render_mode=None, use_reward_wrapper=True):
    """Build a single wrapped CarRacing env (not vectorized).

    Used as the callable passed into SB3's make_vec_env.
    """
    env = gymnasium.make(
        config.ENV_ID,
        continuous=config.CONTINUOUS,
        lap_complete_percent=config.LAP_COMPLETE_PERCENT,
        render_mode=render_mode,
    )
    if use_reward_wrapper:
        env = CarRacingRewardWrapper(env)
    if config.GRAYSCALE:
        env = WarpFrame(env)   # -> 84x84 grayscale
    return env


def make_vec(n_envs=None, seed=None, render_mode=None, use_reward_wrapper=True):
    """Build the vectorized, frame-stacked, channel-transposed env.

    This is what you actually pass to DQN(...). Returns an env ready for a
    CnnPolicy (channels-first, 4 stacked frames).
    """
    n_envs = n_envs if n_envs is not None else config.N_ENVS
    seed = seed if seed is not None else config.SEED

    venv = make_vec_env(
        lambda: make_env(render_mode=render_mode,
                         use_reward_wrapper=use_reward_wrapper),
        n_envs=n_envs,
        seed=seed,
    )
    venv = VecFrameStack(venv, n_stack=config.N_STACK)
    venv = VecTransposeImage(venv)   # HWC -> CHW for PyTorch CNN
    return venv
