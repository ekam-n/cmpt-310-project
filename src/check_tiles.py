"""How close does an agent get to lap completion? Usage (from src/):
    python check_tiles.py <run_dir> [--episodes 5]
"""
import argparse
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from common import env_factory

parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--episodes", type=int, default=5)
args = parser.parse_args()

raw = {}
def make():
    e = env_factory.make_env(use_reward_wrapper=False)
    raw["env"] = e
    return e

venv = DummyVecEnv([make])
venv = VecFrameStack(venv, n_stack=4)
venv = VecTransposeImage(venv)
model = DQN.load(Path(args.run_dir) / "best_model.zip", env=venv)

for ep in range(args.episodes):
    # Seed AFTER model load, per episode: DQN.load(env=...) re-seeds the env
    # with the model's saved TRAINING seed (SB3 set_random_seed), which used
    # to silently replace the track requested here. venv.seed before reset is
    # how evaluate.py / speed_check.py pin the true EVAL_SEEDS tracks.
    venv.seed(1001 + ep)               # walk the first EVAL_SEEDS
    obs = venv.reset()
    done = [False]
    steps = 0
    max_tiles = 0
    total = len(raw["env"].unwrapped.track)   # track size, read at episode start
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(action)
        steps += 1
        if not done[0]:  # only sample while THIS episode is still live
            max_tiles = max(max_tiles, raw["env"].unwrapped.tile_visited_count)
    print(f"ep{ep+1}: {max_tiles}/{total} tiles = {max_tiles/total*100:.1f}% "
          f"(need 95%) in {steps} steps")
venv.close()