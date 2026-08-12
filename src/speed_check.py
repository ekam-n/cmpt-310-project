"""Measure driving pace directly: per-step speed, action mix, and tile
progress for two trained agents on the SAME fixed eval tracks.

Built to test the hypothesis that noisy_dqn drives competently but slower
than baseline_dqn (which would turn a small pace deficit into a large
lap-completion gap via the 95%-tiles-in-1000-frames cutoff).

Usage (from src/):
    python speed_check.py <run_dir_a> <run_dir_b> [--episodes 5]

Tracks are pinned exactly the way evaluate.py pins them (venv.seed(s) before
reset, walking EVAL_SEEDS), so episodes here are directly comparable to the
official evaluation. Reward wrapper off, deterministic predict.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecFrameStack, VecTransposeImage)

from common import config, env_factory

ACTION_NAMES = ["noop", "left", "right", "gas", "brake"]
NEAR_ZERO_FRACTION = 0.05      # "near-zero" = below 5% of the agent's max speed
SLOW_SEGMENT_MIN_STEPS = 15    # contiguous near-zero runs at least this long

# Chart tokens (validated categorical pair + neutral ink/grid).
AGENT_COLORS = ["#2a78d6", "#eb6834"]   # fixed by CLI order: first, second
INK, INK_MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def resolve_model_path(run_dir: Path) -> Path:
    model_path = run_dir / "best_model.zip"
    if model_path.is_file():
        return model_path
    print(f"ERROR: {model_path} does not exist.", file=sys.stderr)
    hints = sorted(p.parent for p in run_dir.parent.glob("*/best_model.zip"))
    for hint in hints:
        print(f"  did you mean: {hint}", file=sys.stderr)
    sys.exit(1)


def make_env_with_handle():
    """Vec env matching eval conditions, plus a handle on the raw env so we
    can read Box2D state (car physics, tiles) that the vec API hides."""
    raw = {}

    def make():
        e = env_factory.make_env(use_reward_wrapper=False)
        raw["env"] = e
        return e

    venv = DummyVecEnv([make])
    venv = VecFrameStack(venv, n_stack=config.N_STACK)
    venv = VecTransposeImage(venv)
    return venv, raw


def run_episode(model, venv, raw, seed):
    """One deterministic episode on the track fixed by `seed`."""
    venv.seed(seed)                      # exactly how evaluate.py pins tracks
    obs = venv.reset()
    unwrapped = raw["env"].unwrapped
    track_total = len(unwrapped.track)

    speeds, tiles, actions = [], [], []
    done = [False]
    steps = 0
    while not done[0] and steps < 1100:  # TimeLimit truncates at 1000 anyway
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(action)
        steps += 1
        actions.append(int(action[0]))
        if not done[0]:                  # after done the env has auto-reset
            v = unwrapped.car.hull.linearVelocity
            speeds.append(float((v.x * v.x + v.y * v.y) ** 0.5))
            tiles.append(int(unwrapped.tile_visited_count))
    return {
        "seed": seed,
        "steps": steps,
        "track_total": track_total,
        "final_tiles": tiles[-1] if tiles else 0,
        "speeds": np.array(speeds),
        "tiles": np.array(tiles),
        "actions": np.array(actions),
    }


def slow_segments(speeds, threshold):
    """Contiguous runs of near-zero speed: list of (start, end) step ranges."""
    below = speeds < threshold
    segments = []
    start = None
    for i, b in enumerate(below):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= SLOW_SEGMENT_MIN_STEPS:
                segments.append((start, i - 1))
            start = None
    if start is not None and len(below) - start >= SLOW_SEGMENT_MIN_STEPS:
        segments.append((start, len(below) - 1))
    return segments


def action_mix(actions):
    counts = np.bincount(actions, minlength=len(ACTION_NAMES))
    return counts / max(len(actions), 1)


def report_agent(name, episodes, max_speed):
    threshold = NEAR_ZERO_FRACTION * max_speed
    print(f"\n=== {name}  (max speed over all episodes: {max_speed:.1f}; "
          f"near-zero = < {threshold:.1f}) ===")
    for ep in episodes:
        s = ep["speeds"]
        cov = ep["final_tiles"] / ep["track_total"] * 100
        pace = ep["final_tiles"] / ep["steps"] * 100
        near_zero = float((s < threshold).mean() * 100) if len(s) else 0.0
        mix = action_mix(ep["actions"])
        mix_str = " ".join(f"{n}={f * 100:.0f}%"
                           for n, f in zip(ACTION_NAMES, mix))
        segs = slow_segments(s, threshold)
        seg_str = (", ".join(f"steps {a}-{b}" for a, b in segs[:4])
                   + ("..." if len(segs) > 4 else "")) if segs else "none"
        print(f"track {ep['seed']}: {ep['final_tiles']}/{ep['track_total']} "
              f"tiles ({cov:.1f}%) in {ep['steps']} steps | "
              f"speed mean {s.mean():.1f} median {np.median(s):.1f} "
              f"max {s.max():.1f} | tiles/100 steps {pace:.1f} | "
              f"near-zero {near_zero:.1f}%")
        print(f"    actions: {mix_str}")
        print(f"    slow segments (>= {SLOW_SEGMENT_MIN_STEPS} steps): {seg_str}")


def save_figure(names, all_episodes, out_path, rep_idx=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150,
                                     constrained_layout=True)
    fig.patch.set_facecolor("white")
    seeds = [ep["seed"] for ep in all_episodes[0]]

    # (a) mean speed per track, both agents side by side
    x = np.arange(len(seeds))
    width = 0.38
    for i, (name, eps) in enumerate(zip(names, all_episodes)):
        means = [ep["speeds"].mean() for ep in eps]
        offset = (i - 0.5) * (width + 0.04)   # small gap between paired bars
        ax_a.bar(x + offset, means, width, color=AGENT_COLORS[i], label=name)
    ax_a.set_xticks(x, [str(s) for s in seeds])
    ax_a.set_xlabel("track seed", color=INK_MUTED)
    ax_a.set_ylabel("mean speed", color=INK_MUTED)
    ax_a.set_title("(a) Mean speed per eval track", color=INK, loc="left")

    # (b) speed traces on the representative track, overlaid
    for i, (name, eps) in enumerate(zip(names, all_episodes)):
        s = eps[rep_idx]["speeds"]
        ax_b.plot(np.arange(len(s)), s, color=AGENT_COLORS[i],
                  linewidth=1.4, label=name)
    ax_b.set_xlabel("step", color=INK_MUTED)
    ax_b.set_ylabel("speed", color=INK_MUTED)
    ax_b.set_title(f"(b) Speed over time, track {seeds[rep_idx]} "
                   f"(largest pace gap)", color=INK, loc="left")

    for ax in (ax_a, ax_b):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_MUTED)
        ax.legend(frameon=False, labelcolor=INK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nfigure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs=2,
                        help="two run directories to compare "
                             "(each containing best_model.zip)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="how many EVAL_SEEDS tracks to drive (default 5)")
    args = parser.parse_args()

    seeds = config.EVAL_SEEDS[:args.episodes]
    names, all_episodes = [], []
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        model_path = resolve_model_path(run_dir)
        name = run_dir.parent.name          # logs/<agent>/<run> -> agent
        names.append(name)
        venv, raw = make_env_with_handle()
        model = DQN.load(model_path, env=venv)
        print(f"[speed_check] {name}: {model_path}", flush=True)
        episodes = []
        for seed in seeds:
            ep = run_episode(model, venv, raw, seed)
            print(f"[speed_check]   track {seed} done "
                  f"({ep['steps']} steps)", flush=True)
            episodes.append(ep)
        venv.close()
        all_episodes.append(episodes)

    # SANITY: both agents must have seen identical tracks.
    for i, seed in enumerate(seeds):
        totals = [eps[i]["track_total"] for eps in all_episodes]
        if len(set(totals)) != 1:
            print(f"ERROR: track {seed} differs between agents "
                  f"(tile totals {totals}); comparison is invalid.",
                  file=sys.stderr)
            sys.exit(1)
    print("\nsanity: identical tracks for both agents "
          f"(tile totals: {[eps['track_total'] for eps in all_episodes[0]]})")

    max_speeds = [max(ep["speeds"].max() for ep in eps) for eps in all_episodes]
    for name, eps, ms in zip(names, all_episodes, max_speeds):
        report_agent(name, eps, ms)

    # Cross-agent summary on the same tracks. Gap = second relative to first.
    a_name, b_name = names
    print(f"\n=== {b_name} relative to {a_name}, same tracks ===")
    print(f"{'track':>6} | {'mean speed':>21} | {'gap':>7} | "
          f"{'tiles/100 steps':>21} | {'gap':>7}")
    speed_gaps, pace_gaps = [], []
    for ep_a, ep_b in zip(*all_episodes):
        sa, sb = ep_a["speeds"].mean(), ep_b["speeds"].mean()
        pa = ep_a["final_tiles"] / ep_a["steps"] * 100
        pb = ep_b["final_tiles"] / ep_b["steps"] * 100
        sg, pg = (sb - sa) / sa * 100, (pb - pa) / pa * 100
        speed_gaps.append(sg)
        pace_gaps.append(pg)
        print(f"{ep_a['seed']:>6} | {sa:>9.1f} vs {sb:>7.1f} | {sg:>+6.1f}% | "
              f"{pa:>9.1f} vs {pb:>7.1f} | {pg:>+6.1f}%")
    print(f"aggregate: mean speed gap {np.mean(speed_gaps):+.1f}%, "
          f"tiles/100-steps gap {np.mean(pace_gaps):+.1f}%")

    rep_idx = int(np.argmax(np.abs(pace_gaps)))
    save_figure(names, all_episodes, Path("figures") / "speed_comparison.png",
                rep_idx=rep_idx)


if __name__ == "__main__":
    main()
