"""
Shared plotting & comparison module.  --  owner: Ekam

Turns the data our training runs ALREADY produce into the comparison figures
and tables the writeup needs. No extra logging required: every agent's
EvalCallback writes logs/<agent>/evaluations.npz (eval rewards vs timesteps)
and saves best_model.zip. This module reads those.

WHAT IT PRODUCES (all saved under figures/):
  1. learning_curves.png   -- mean eval reward vs timesteps for every agent
                              found in logs/, on one axes, with std bands.
                              THE key "ours vs baseline" figure.
  2. comparison.csv/.md    -- final-metrics table (reward, completion rate,
                              collision rate, steps-to-lap) built by running
                              the shared evaluate_agent on each best_model.zip.
  3. <metric>_bar.png      -- one bar chart per metric from that table.

USAGE (from src/):
    # Learning curves only (fast -- just reads npz files):
    python -m evaluate.plots

    # Everything, including re-evaluating each best model (slower):
    python -m evaluate.plots --full --episodes 20

Figures save to files (no display window), so this works on Colab and
headless machines too.

NOTE for fair comparisons: the canonical writeup figures should come from ONE
machine that trained all agents under the same shared config. Locally, you'll
just see whichever runs exist in your own logs/ folder.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: save files, never open a window
import matplotlib.pyplot as plt

from common import config

# Where agents write logs (train scripts run from src/, so this is src/logs)
LOG_ROOT = Path(config.LOG_ROOT)
FIG_DIR = Path("figures")

# Fixed colors so every figure uses the same color per agent
AGENT_COLORS = {
    "baseline_dqn": "#d62728",   # red
    "double_dqn": "#9467bd",     # purple
    "dueling_dqn": "#2ca02c",    # green
}
FALLBACK_COLORS = ["#1f77b4", "#ff7f0e", "#8c564b", "#e377c2"]


def find_agent_runs():
    """Return {agent_name: run_dir} for every logs/<agent>/ that has data."""
    runs = {}
    if not LOG_ROOT.exists():
        return runs
    for d in sorted(LOG_ROOT.iterdir()):
        if d.is_dir() and ((d / "evaluations.npz").exists()
                           or (d / "best_model.zip").exists()):
            runs[d.name] = d
    return runs


def color_for(name, idx):
    return AGENT_COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ---------------------------------------------------------------------------
# 1) Learning curves from evaluations.npz
# ---------------------------------------------------------------------------

def plot_learning_curves(runs, out_path=None):
    """Overlay mean eval reward vs timesteps for every agent, with std bands."""
    out_path = out_path or (FIG_DIR / "learning_curves.png")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plotted = 0
    for idx, (name, run_dir) in enumerate(runs.items()):
        npz = run_dir / "evaluations.npz"
        if not npz.exists():
            print(f"  (skip {name}: no evaluations.npz -- was it trained "
                  f"with the EvalCallback?)")
            continue
        data = np.load(npz)
        timesteps = data["timesteps"]
        results = data["results"]              # (n_evals, n_eval_episodes)
        mean_r = results.mean(axis=1)
        std_r = results.std(axis=1)

        c = color_for(name, idx)
        ax.plot(timesteps, mean_r, label=name, color=c, linewidth=2,
        marker="o", markersize=4)
        ax.fill_between(timesteps, mean_r - std_r, mean_r + std_r,
                        color=c, alpha=0.15)
        plotted += 1

    if plotted == 0:
        print("No evaluations.npz files found under "
              f"{LOG_ROOT}/ -- train something first.")
        plt.close(fig)
        return None

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title("Learning curves (mean eval reward ± std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2) Final comparison table (runs the shared evaluator on each best model)
# ---------------------------------------------------------------------------

def build_comparison_table(runs, n_episodes=20):
    """Evaluate each agent's best_model.zip with the SHARED evaluator and
    return {agent_name: metrics dict}. Slow: rolls out real episodes."""
    # Imported here so plain plotting doesn't require torch/SB3 to load.
    from stable_baselines3 import DQN
    from common.env_factory import make_vec
    from evaluate.evaluate import evaluate_agent, print_metrics

    results = {}
    for name, run_dir in runs.items():
        best = run_dir / "best_model.zip"
        if not best.exists():
            print(f"  (skip {name}: no best_model.zip)")
            continue
        print(f"Evaluating {name} ({n_episodes} episodes)...")
        # NOTE: evaluation always uses the UNSHAPED env (native reward) so the
        # -100 death signal is intact and numbers are comparable across agents.
        env = make_vec(use_reward_wrapper=False)
        # DQN.load works for our subclasses too (they only override train()).
        # If an agent later changes its policy/architecture in a way that
        # breaks this, load it with its own class here.
        model = DQN.load(best, env=env)
        metrics = evaluate_agent(model, env, n_episodes=n_episodes)
        print_metrics(name, metrics)
        env.close()
        results[name] = metrics
    return results


TABLE_FIELDS = [
    ("mean_reward", "Mean reward"),
    ("std_reward", "Std reward"),
    ("completion_rate", "Completion rate"),
    ("collision_rate", "Collision rate"),
    ("mean_episode_len", "Mean episode len"),
    ("mean_steps_to_lap", "Steps to lap"),
]


def save_comparison_table(results, csv_path=None, md_path=None):
    """Write the metrics table as CSV (for analysis) and Markdown (for the
    writeup / README)."""
    csv_path = csv_path or (FIG_DIR / "comparison.csv")
    md_path = md_path or (FIG_DIR / "comparison.md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent"] + [k for k, _ in TABLE_FIELDS])
        for name, m in results.items():
            w.writerow([name] + [m.get(k) for k, _ in TABLE_FIELDS])

    with open(md_path, "w") as f:
        header = "| Agent | " + " | ".join(l for _, l in TABLE_FIELDS) + " |"
        sep = "|" + "---|" * (len(TABLE_FIELDS) + 1)
        f.write(header + "\n" + sep + "\n")
        for name, m in results.items():
            cells = []
            for k, _ in TABLE_FIELDS:
                v = m.get(k)
                if v is None:
                    cells.append("--")
                elif "rate" in k:
                    cells.append(f"{v * 100:.0f}%")
                else:
                    cells.append(f"{v:.1f}")
            f.write(f"| {name} | " + " | ".join(cells) + " |\n")

    print(f"Saved {csv_path} and {md_path}")


def plot_metric_bars(results):
    """One bar chart per headline metric."""
    bar_metrics = [
        ("mean_reward", "Mean evaluation reward"),
        ("completion_rate", "Lap completion rate"),
        ("collision_rate", "Collision rate"),
        ("mean_steps_to_lap", "Timesteps to complete a lap"),
    ]
    names = list(results.keys())
    for idx_m, (key, label) in enumerate(bar_metrics):
        values = [results[n].get(key) for n in names]
        # skip charts where no agent has the metric (e.g. nobody finished a lap)
        if all(v is None for v in values):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = np.arange(len(names))
        vals = [0 if v is None else v for v in values]
        colors = [color_for(n, i) for i, n in enumerate(names)]
        ax.bar(xs, vals, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        for x, v in zip(xs, values):
            ax.text(x, (0 if v is None else v),
                    "n/a" if v is None else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out = FIG_DIR / f"{key}_bar.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also evaluate each best_model.zip and build the "
                             "comparison table + bar charts (slow)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="episodes per agent for --full evaluation")
    args = parser.parse_args()

    runs = find_agent_runs()
    if not runs:
        print(f"No agent runs found under {LOG_ROOT}/. Train something first "
              f"(e.g. python -m baseline.train_baseline), then re-run this.")
        return

    print("Found runs:", ", ".join(runs))
    plot_learning_curves(runs)

    if args.full:
        results = build_comparison_table(runs, n_episodes=args.episodes)
        if results:
            save_comparison_table(results)
            plot_metric_bars(results)


if __name__ == "__main__":
    main()
