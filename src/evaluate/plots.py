"""
Shared plotting & comparison module.  --  owner: Ekam

Turns the data our training runs ALREADY produce into the comparison figures
and tables the writeup needs. No extra logging required: every agent's
EvalCallback writes evaluations.npz (eval rewards vs timesteps) and saves
best_model.zip into its run directory. This module reads those, plus the
run_config.json / results.json manifests written by common.run_tracking.

WHAT IT PRODUCES (all saved under figures/):
  1. learning_curves.png   -- mean eval reward vs timesteps for every run
                              found in logs/, on one axes, with std bands.
                              THE key "ours vs baseline" figure.
  2. comparison.csv/.md    -- final-metrics table (reward, completion rate,
                              collision rate, steps-to-lap) built by running
                              the shared evaluate_agent on each best_model.zip.
  3. <metric>_bar.png      -- one bar chart per metric from that table.
  4. runs_index.csv        -- every run and its settings, one row each.

USAGE (from src/):
    # Learning curves for everything in logs/ (fast -- just reads npz files):
    python -m evaluate.plots

    # Slice the runs: hold the agent fixed, one curve per activation.
    python -m evaluate.plots --filter agent=double_dqn --group-by activation

    # Multiple filters are ANDed:
    python -m evaluate.plots --filter agent=double_dqn --filter reward_wrapper=off

    # What runs do I even have?
    python -m evaluate.plots --index          # -> figures/runs_index.csv
    python -m evaluate.plots --list           # -> printed to the terminal

    # Everything, including re-evaluating each best model (slower):
    python -m evaluate.plots --full --episodes 20

--filter / --group-by accept ANY field recorded in run_config.json or
results.json (agent, activation, seed, reward_wrapper, timesteps, git.commit,
dqn_kwargs.learning_rate, ...). Run --list to see what is available.

SEED AGGREGATION: runs that are identical except for their seed are averaged
into a single curve with a mean +/- std band across seeds, rather than drawn as
separate lines. That is usually what you want when reporting a result.

BACKWARD COMPATIBILITY: older flat logs/<agent>/ runs (from before per-run
directories existed) are still discovered and plotted. They carry no manifest,
so they show up as agent-named curves and are dropped by any --filter on a
field they do not have.

Figures save to files (no display window), so this works on Colab and
headless machines too.
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
from common.run_tracking import load_runs, write_runs_index

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

# Runs within one agent are told apart by linestyle, so a mixed figure keeps
# each agent's colour while still separating its variants.
LINESTYLES = ["-", "--", "-.", ":"]

# Used instead when the figure is a sweep over ONE agent, where colouring by
# agent would make every curve identical. Colour-blind-safe ordering.
SWEEP_COLORS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7",
                "#d55e00", "#56b4e9", "#f0e442", "#000000"]

# Fields that legitimately differ between repeats of the "same" experiment.
# Everything else being equal, runs differing only in these are seed-repeats.
NON_IDENTIFYING = {
    "seed", "dqn_kwargs.seed", "cli_args.seed", "run_dir", "run_name",
    "started_at", "finished_at", "train_seconds", "legacy", "complete",
    "has_curve", "has_model", "command", "note", "cli_args.note",
    "best_eval_reward", "best_eval_timestep", "final_eval_reward",
    "n_evaluations", "device", "platform", "python",
    "git.commit", "git.dirty", "git.branch",
}


def color_for(name, idx):
    return AGENT_COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ---------------------------------------------------------------------------
# Run discovery, filtering, grouping
# ---------------------------------------------------------------------------

def discover_runs(log_root=None):
    """Every run under logs/ -- new nested run dirs AND legacy flat dirs."""
    return load_runs(log_root if log_root is not None else LOG_ROOT)


def parse_filters(pairs):
    """Turn ['agent=double_dqn', 'seed=42'] into {'agent': 'double_dqn', ...}."""
    filters = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(
                f"--filter expects key=value, got '{pair}'. "
                f"Example: --filter agent=double_dqn")
        key, value = pair.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def apply_filters(runs, filters):
    """Keep runs matching every key=value. Compared as strings, so
    `--filter seed=42` matches the integer 42 in the manifest.

    A run that does not have the field at all is dropped -- which is how
    legacy runs (no manifest) fall out of a --filter activation=relu."""
    if not filters:
        return runs
    kept = []
    for run in runs:
        if all(key in run and str(run[key]) == value
               for key, value in filters.items()):
            kept.append(run)
    return kept


def config_signature(run, group_by=None):
    """Identity of a run's *configuration*, ignoring seed and run metadata.

    Two runs with the same signature are seed-repeats of one experiment and get
    aggregated into one curve.
    """
    if run.get("legacy"):
        # No manifest to compare; each legacy dir stands alone.
        return (("run_dir", run["run_dir"]),)
    items = tuple(sorted(
        (k, str(v)) for k, v in run.items()
        if k not in NON_IDENTIFYING and not k.startswith("cli_args.")
    ))
    return items


def group_runs(runs, group_by=None):
    """Return {label: [runs]}.

    With --group-by, one group per distinct value of that field (this is what
    makes `--group-by activation` overlay one curve per activation, averaging
    over seeds within each). Without it, one group per distinct configuration,
    still averaging over seeds.
    """
    groups = {}
    if group_by:
        for run in runs:
            if group_by not in run:
                print(f"  (skip {run['run_name']}: no '{group_by}' field -- "
                      f"legacy run or older manifest)")
                continue
            groups.setdefault(f"{group_by}={run[group_by]}", []).append(run)
    else:
        by_sig = {}
        for run in runs:
            by_sig.setdefault(config_signature(run), []).append(run)
        for members in by_sig.values():
            groups[label_for(members)] = members
    return dict(sorted(groups.items()))


def label_for(runs):
    """A short human label for a group of seed-repeat runs."""
    first = runs[0]
    agent = first.get("agent", "?")
    if first.get("legacy"):
        return f"{agent} (legacy)"
    bits = [agent]
    if first.get("activation"):
        bits.append(str(first["activation"]))
    if first.get("reward_wrapper"):
        bits.append(f"rw={first['reward_wrapper']}")
    label = " ".join(bits)
    if len(runs) > 1:
        label += f" (n={len(runs)})"
    return label


def agent_of(runs):
    return runs[0].get("agent", "?")


# ---------------------------------------------------------------------------
# 1) Learning curves from evaluations.npz
# ---------------------------------------------------------------------------

def load_curve(run):
    """(timesteps, per_eval_mean_reward) from a run's evaluations.npz, or None."""
    npz = Path(run["run_dir"]) / "evaluations.npz"
    if not npz.exists():
        return None
    try:
        data = np.load(npz)
        timesteps = np.asarray(data["timesteps"], dtype=float)
        results = np.asarray(data["results"], dtype=float)
    except Exception as exc:
        print(f"  (skip {run['run_name']}: unreadable evaluations.npz: {exc})")
        return None
    if timesteps.size == 0 or results.size == 0:
        return None
    return timesteps, results.mean(axis=1)


def aggregate_curves(curves):
    """Mean +/- std across seed-repeat runs, on a common timestep grid.

    Runs can have different numbers of evaluations (different budgets or
    eval_freq), so each curve is interpolated onto the overlapping range before
    averaging rather than assuming the x-axes line up.
    """
    if len(curves) == 1:
        timesteps, values = curves[0]
        return timesteps, values, np.zeros_like(values)

    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    if hi <= lo:
        # No overlap at all (e.g. wildly different budgets): fall back to the
        # longest single curve rather than inventing an average.
        timesteps, values = max(curves, key=lambda c: c[0][-1])
        return timesteps, values, np.zeros_like(values)

    n_points = max(len(c[0]) for c in curves)
    grid = np.linspace(lo, hi, n_points)
    stacked = np.vstack([np.interp(grid, ts, vals) for ts, vals in curves])
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def plot_learning_curves(groups, out_path=None, title=None):
    """Overlay mean eval reward vs timesteps for every group, with std bands."""
    out_path = Path(out_path or (FIG_DIR / "learning_curves.png"))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Colour normally means "which agent". But in a single-agent ablation
    # (--filter agent=X --group-by activation) that would make every curve the
    # same colour, so there we colour by group instead and let the fixed agent
    # be stated in the title.
    agents = {agent_of(runs) for runs in groups.values()}
    single_agent_sweep = len(agents) == 1 and len(groups) > 1

    style_by_agent = {}
    plotted = 0
    for idx, (label, runs) in enumerate(groups.items()):
        curves = [c for c in (load_curve(r) for r in runs) if c is not None]
        if not curves:
            print(f"  (skip {label}: no evaluations.npz -- was it trained "
                  f"with the EvalCallback, or stopped before the first eval?)")
            continue

        timesteps, mean_r, std_r = aggregate_curves(curves)
        agent = agent_of(runs)
        if single_agent_sweep:
            colour = SWEEP_COLORS[idx % len(SWEEP_COLORS)]
            style = "-"
        else:
            colour = color_for(agent, idx)
            # Same agent, different config -> same colour, different linestyle.
            seen = style_by_agent.setdefault(agent, 0)
            style_by_agent[agent] = seen + 1
            style = LINESTYLES[seen % len(LINESTYLES)]

        suffix = f" [{len(curves)} seeds]" if len(curves) > 1 else ""
        ax.plot(timesteps, mean_r, label=label + suffix, color=colour,
                linewidth=2, linestyle=style, marker="o", markersize=4)
        if np.any(std_r):
            ax.fill_between(timesteps, mean_r - std_r, mean_r + std_r,
                            color=colour, alpha=0.15)
        plotted += 1

    if plotted == 0:
        print(f"No usable evaluations.npz files found under {LOG_ROOT}/ -- "
              f"train something first.")
        plt.close(fig)
        return None

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title(title or "Learning curves (mean eval reward ± std)")
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

def row_names(label, runs):
    """Unique table-row name per run in a group -> {run_dir: name}.

    Two runs can share a group AND a seed (a repeat of the same config, or one
    launched by a sweep). Naming rows by label+seed alone would then make two
    rows collide and silently drop a result, so fall back to the run directory
    name, which is unique by construction.
    """
    if len(runs) == 1:
        return {runs[0]["run_dir"]: label}
    seeds = [r.get("seed") for r in runs]
    unique_seeds = all(s is not None for s in seeds) and len(set(seeds)) == len(seeds)
    return {
        r["run_dir"]: (f"{label} seed={r['seed']}" if unique_seeds
                       else f"{label} [{r['run_name']}]")
        for r in runs
    }


def build_comparison_table(groups, n_episodes=None):
    """Evaluate each group's best_model.zip with the SHARED evaluator on the
    FIXED eval tracks, and return {label: metrics dict}. Slow: rolls out real
    episodes."""
    # Imported here so plain plotting doesn't require torch/SB3 to load.
    from stable_baselines3 import DQN
    from common.env_factory import make_eval_vec
    from evaluate.evaluate import evaluate_agent, print_metrics

    n_episodes = n_episodes or len(config.EVAL_SEEDS)
    results = {}
    for label, runs in groups.items():
        names = row_names(label, runs)
        for run in runs:
            best = Path(run["run_dir"]) / "best_model.zip"
            if not best.exists():
                print(f"  (skip {run['run_name']}: no best_model.zip)")
                continue
            name = names[run["run_dir"]]
            # Never let one run's numbers overwrite another's -- losing a run
            # silently is the exact failure this tracking work exists to stop.
            if name in results:
                name = f"{name} [{run['run_name']}]"
            print(f"Evaluating {name} ({n_episodes} episodes)...")
            # Evaluation always uses the UNSHAPED env (native reward) so the
            # -100 death signal is intact, and the FIXED seed set so every
            # model is scored on exactly the same 20 tracks.
            env = make_eval_vec()
            # DQN.load works for our subclasses too (they only override train()).
            # If an agent later changes its policy/architecture in a way that
            # breaks this, load it with its own class here.
            try:
                model = DQN.load(best, env=env)
            except Exception as exc:
                print(f"  (skip {name}: could not load model: {exc})")
                env.close()
                continue
            metrics = evaluate_agent(model, env, n_episodes=n_episodes,
                                     seeds=config.EVAL_SEEDS)
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
    csv_path = Path(csv_path or (FIG_DIR / "comparison.csv"))
    md_path = Path(md_path or (FIG_DIR / "comparison.md"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run"] + [k for k, _ in TABLE_FIELDS])
        for name, m in results.items():
            w.writerow([name] + [m.get(k) for k, _ in TABLE_FIELDS])

    with open(md_path, "w") as f:
        header = "| Run | " + " | ".join(l for _, l in TABLE_FIELDS) + " |"
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
    for key, label in bar_metrics:
        values = [results[n].get(key) for n in names]
        # skip charts where no agent has the metric (e.g. nobody finished a lap)
        if all(v is None for v in values):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = np.arange(len(names))
        vals = [0 if v is None else v for v in values]
        colors = [color_for(n.split()[0], i) for i, n in enumerate(names)]
        ax.bar(xs, vals, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        for x, v in zip(xs, values):
            ax.text(x, (0 if v is None else v),
                    "n/a" if v is None else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = FIG_DIR / f"{key}_bar.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 3) Run index
# ---------------------------------------------------------------------------

def print_run_list(runs):
    """Human-readable summary of what is in logs/, and what you can filter on."""
    print(f"\n{len(runs)} run(s) under {LOG_ROOT}/:\n")
    for run in runs:
        tag = " [legacy]" if run.get("legacy") else ""
        best = run.get("best_eval_reward")
        best_str = f"best={best:.1f}" if isinstance(best, (int, float)) else "best=--"
        print(f"  {run.get('agent','?'):<14} {run['run_name']:<44} "
              f"act={run.get('activation', '--'):<10} "
              f"rw={run.get('reward_wrapper', '--'):<4} "
              f"seed={run.get('seed', '--'):<5} {best_str}{tag}")
    fields = sorted({k for r in runs for k in r})
    print(f"\nFilterable fields: {', '.join(fields)}\n")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="also evaluate each best_model.zip and build the "
                             "comparison table + bar charts (slow)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="episodes per agent for --full evaluation "
                             f"(default: {len(config.EVAL_SEEDS)}, one per eval seed)")
    parser.add_argument("--filter", action="append", metavar="KEY=VALUE",
                        help="keep only runs where KEY equals VALUE; repeatable "
                             "(e.g. --filter agent=double_dqn). Fields come "
                             "from run_config.json -- see --list.")
    parser.add_argument("--group-by", metavar="FIELD",
                        help="one curve per distinct value of FIELD "
                             "(e.g. --group-by activation)")
    parser.add_argument("--index", action="store_true",
                        help="write every run and its settings to "
                             "figures/runs_index.csv")
    parser.add_argument("--list", action="store_true",
                        help="print the runs found and the fields you can "
                             "filter/group on, then exit")
    parser.add_argument("--log-root", default=None,
                        help=f"where to look for runs (default: {LOG_ROOT})")
    parser.add_argument("--out", default=None,
                        help="output path for the learning-curves figure")
    args = parser.parse_args()

    runs = discover_runs(args.log_root)
    if not runs:
        root = args.log_root or LOG_ROOT
        print(f"No runs found under {root}/. Train something first "
              f"(e.g. python -m baseline.train_baseline --reward-wrapper off), "
              f"then re-run this.")
        return

    if args.list:
        print_run_list(runs)
        return

    if args.index:
        out = write_runs_index(runs, FIG_DIR / "runs_index.csv")
        print(f"Saved {out} ({len(runs)} runs)")
        if not (args.filter or args.group_by or args.full):
            return

    selected = apply_filters(runs, parse_filters(args.filter))
    if not selected:
        print(f"No runs matched {args.filter}. Run --list to see what exists.")
        return

    groups = group_runs(selected, args.group_by)
    if not groups:
        print("Nothing left to plot after grouping.")
        return

    print(f"Plotting {len(selected)} run(s) in {len(groups)} group(s): "
          f"{', '.join(groups)}")

    title = None
    if args.group_by or args.filter:
        bits = []
        if args.filter:
            bits.append(", ".join(args.filter))
        if args.group_by:
            bits.append(f"by {args.group_by}")
        title = "Learning curves (" + "; ".join(bits) + ")"
    plot_learning_curves(groups, out_path=args.out, title=title)

    if args.full:
        results = build_comparison_table(groups, n_episodes=args.episodes)
        if results:
            save_comparison_table(results)
            plot_metric_bars(results)


if __name__ == "__main__":
    main()
