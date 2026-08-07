"""Watch a trained agent drive. Usage (from src/):
    python watch.py logs/noisy_dqn/act-relu_rw-off_seed-1_20260806-0034
    python watch.py <run_dir> --seed 1001     # fixed track from EVAL_SEEDS
    python watch.py <run_dir> --episodes 3
    python watch.py <run_dir> --headless      # no window; saves 3 PNG frames

Needs the project venv (stable_baselines3 is not in the system Python):
    ..\\.venv\\Scripts\\python.exe watch.py <run_dir>
"""
import argparse
import sys
from pathlib import Path

import cv2
from stable_baselines3 import DQN

from common import config, env_factory

# Steps at which --headless saves a frame (after the initial zoom-in animation).
HEADLESS_FRAME_STEPS = (50, 150, 250)


def resolve_model_path(run_dir: Path) -> Path:
    """Return <run_dir>/best_model.zip, or exit loudly with suggestions."""
    model_path = run_dir / "best_model.zip"
    if model_path.is_file():
        return model_path

    print(f"ERROR: {model_path} does not exist.", file=sys.stderr)
    # Sibling runs of the same agent; if the agent dir itself was passed by
    # mistake, its own children.
    hints = sorted(p.parent for p in run_dir.parent.glob("*/best_model.zip"))
    if not hints and run_dir.is_dir():
        hints = sorted(p.parent for p in run_dir.glob("*/best_model.zip"))
    if hints:
        print("Run directories that do have a best_model.zip:", file=sys.stderr)
        for hint in hints:
            print(f"  {hint}", file=sys.stderr)
    else:
        print(f"No best_model.zip found near {run_dir}; check the path "
              f"(runs live under {config.LOG_ROOT}/<agent>/).", file=sys.stderr)
    sys.exit(1)


def save_frame(env, step: int) -> None:
    frame = env.render(mode="rgb_array")
    path = f"watch_frame_step{step:03d}.png"
    cv2.imwrite(path, frame[:, :, ::-1])   # RGB -> BGR for OpenCV
    print(f"[watch]   saved {path} (frame std {frame.std():.1f}; "
          f"0.0 would mean a blank frame)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="run directory containing best_model.zip")
    parser.add_argument("--seed", type=int, default=None, help="fix the track seed")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--headless", action="store_true",
                        help="render off-screen and save 3 sample frames as PNGs "
                             "instead of opening a window")
    args = parser.parse_args()

    model_path = resolve_model_path(Path(args.run_dir))
    print(f"[watch] model: {model_path}", flush=True)

    # make_vec falls back to config.SEED when seed is None, so the track is
    # NEVER random -- say which seed actually applies. The seed fixes episode
    # 1's track exactly; episodes 2+ follow a deterministic sequence from it,
    # so two invocations with the same seed always see the same tracks.
    seed = args.seed if args.seed is not None else config.SEED
    if args.seed is None:
        print(f"[watch] no --seed given -> config default {config.SEED}", flush=True)
    print(f"[watch] track seed {seed}: same seed = same track sequence, "
          f"so agents can be compared on identical tracks", flush=True)

    render_mode = "rgb_array" if args.headless else "human"
    env = env_factory.make_vec(render_mode=render_mode,
                               use_reward_wrapper=False, seed=seed)
    model = DQN.load(model_path, env=env)
    print(f"[watch] model loaded (device: {model.device})", flush=True)
    if not args.headless:
        print("[watch] a pygame window opens at the first step -- it may start "
              "behind other windows; an episode is up to 1000 steps (~20 s)",
              flush=True)

    for ep in range(args.episodes):
        obs = env.reset()
        print(f"[watch] episode {ep + 1} starting", flush=True)
        done = [False]
        total = 0.0
        steps = 0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += float(reward[0])
            steps += 1
            if steps % 100 == 0:
                print(f"[watch]   step {steps}, reward so far {total:.1f}", flush=True)
            if args.headless and ep == 0 and steps in HEADLESS_FRAME_STEPS:
                save_frame(env, steps)
        print(f"episode {ep+1}: reward {total:.1f} in {steps} steps", flush=True)
    env.close()


if __name__ == "__main__":
    main()
