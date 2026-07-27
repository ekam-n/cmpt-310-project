"""
Sequential sweep runner: launch a batch of training runs unattended.

Milestone 2 is ablations -- agent x activation x seed x reward-shaping. Rather
than babysitting each command, describe the grid in a JSON file and let this
run through it. Each combination becomes one normal training command, so the
runs it produces are indistinguishable from ones you launched by hand.

RESUMABLE: before launching anything, it reads back every run already in logs/
and skips combinations that have already COMPLETED (i.e. have a results.json).
Kill it, restart it, and it picks up where it left off. A run that crashed
half-way has no results.json, so it gets retried.

USAGE (from src/):
    python -m experiments.sweep --config experiments/sweeps/activation.json --dry-run
    python -m experiments.sweep --config experiments/sweeps/activation.json
    python -m experiments.sweep --config experiments/sweeps/activation.json --force

SWEEP FILE (JSON; YAML also works if you happen to have pyyaml installed):

    {
      "agents": ["baseline_dqn", "double_dqn"],
      "activation": ["relu", "elu"],
      "seed": [42, 43],
      "reward_wrapper": ["off"],
      "timesteps": 750000
    }

List-valued keys are swept over (full cartesian product); scalar keys are
passed to every run. That example is 2 x 2 x 2 x 1 = 8 runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from common import config
from common.run_tracking import load_runs

# Agent name -> the module that trains it. Add a line here when you add an
# agent (e.g. "noisy_dqn": "agents.noisy_net") and it becomes sweepable.
AGENT_MODULES = {
    "baseline_dqn": "baseline.train_baseline",
    "double_dqn": "agents.double_dqn",
    "dueling_dqn": "agents.dueling_dqn",
}

# Keys that are swept (each becomes a CLI flag on the training command).
SWEPT_KEYS = ["activation", "reward_wrapper", "seed"]
# Keys passed straight through to every run.
PASSTHROUGH_KEYS = ["timesteps", "eval_freq", "n_eval_episodes", "log_root", "note"]


def load_sweep_file(path):
    """Read a sweep definition. JSON always; YAML if pyyaml is installed."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Sweep file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                f"{path} is YAML but pyyaml is not installed (it is not a "
                f"project dependency). Either `pip install pyyaml` or convert "
                f"the sweep file to JSON.")
        spec = yaml.safe_load(text)
    else:
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path} is not valid JSON: {exc}")
    if not isinstance(spec, dict):
        raise SystemExit(f"{path} must contain a JSON object, got {type(spec).__name__}")
    return spec


def expand(spec):
    """Cartesian product of the swept keys -> list of run dicts."""
    agents = spec.get("agents") or spec.get("agent")
    if not agents:
        raise SystemExit("Sweep file needs an 'agents' list, e.g. "
                         '"agents": ["baseline_dqn"]')
    if isinstance(agents, str):
        agents = [agents]
    unknown = [a for a in agents if a not in AGENT_MODULES]
    if unknown:
        raise SystemExit(f"Unknown agent(s) {unknown}. Known: "
                         f"{sorted(AGENT_MODULES)}. Add new ones to "
                         f"AGENT_MODULES in experiments/sweep.py.")

    axes = {}
    for key in SWEPT_KEYS:
        value = spec.get(key)
        if value is None:
            continue
        axes[key] = value if isinstance(value, list) else [value]

    if "reward_wrapper" not in axes:
        raise SystemExit(
            "Sweep file must set 'reward_wrapper' (\"on\", \"off\", or a list "
            "of both). It has no default anywhere in this project on purpose "
            "-- see docs/EXPERIMENTS.md.")

    passthrough = {k: spec[k] for k in PASSTHROUGH_KEYS if k in spec}

    runs = []
    keys = list(axes)
    for agent in agents:
        for combo in itertools.product(*(axes[k] for k in keys)):
            runs.append({"agent": agent, **dict(zip(keys, combo)), **passthrough})
    return runs


def command_for(run):
    """The exact `python -m ...` command this run corresponds to."""
    cmd = [sys.executable, "-m", AGENT_MODULES[run["agent"]]]
    for key in SWEPT_KEYS + PASSTHROUGH_KEYS:
        if key in run and run[key] is not None:
            cmd += [f"--{key.replace('_', '-')}", str(run[key])]
    return cmd


def matches(existing, run):
    """Does an already-recorded run correspond to this sweep combination?"""
    if not existing.get("complete"):
        return False          # crashed / still running -> re-run it
    if existing.get("agent") != run["agent"]:
        return False
    for key in SWEPT_KEYS:
        if key not in run:
            continue
        if str(existing.get(key)) != str(run[key]):
            return False
    # Budget matters too: a 40k run does not satisfy a 750k sweep entry.
    if "timesteps" in run and str(existing.get("timesteps")) != str(run["timesteps"]):
        return False
    return True


def already_done(existing_runs, run):
    return any(matches(e, run) for e in existing_runs)


def describe(run):
    return (f"{run['agent']:<14} act={run.get('activation', '-'):<10} "
            f"rw={run.get('reward_wrapper', '-'):<4} "
            f"seed={run.get('seed', '-'):<5} "
            f"steps={run.get('timesteps', config.TOTAL_TIMESTEPS)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="path to the sweep definition (JSON, or YAML if "
                             "pyyaml is installed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would run (and what would be skipped) "
                             "without training anything")
    parser.add_argument("--force", action="store_true",
                        help="re-run every combination, even ones that already "
                             "have a completed run in logs/")
    parser.add_argument("--log-root", default=None,
                        help="where runs live, for both the skip check and the "
                             "training runs themselves")
    args = parser.parse_args()

    spec = load_sweep_file(args.config)
    if args.log_root:
        spec["log_root"] = args.log_root
    runs = expand(spec)

    existing = [] if args.force else load_runs(spec.get("log_root"))
    todo, skipped = [], []
    for run in runs:
        (skipped if already_done(existing, run) else todo).append(run)

    print(f"\nSweep: {args.config}")
    print(f"  {len(runs)} combination(s): {len(todo)} to run, "
          f"{len(skipped)} already complete")
    for run in skipped:
        print(f"  [skip] {describe(run)}")
    for run in todo:
        print(f"  [run ] {describe(run)}")

    if args.dry_run:
        print("\n--dry-run: nothing executed.\n")
        return

    if not todo:
        print("\nNothing to do -- every combination already has a completed "
              "run. Use --force to redo them.\n")
        return

    failures = []
    started = time.perf_counter()
    for idx, run in enumerate(todo, 1):
        cmd = command_for(run)
        # shlex.join so a --note with spaces prints as something you can paste.
        print(f"\n{'=' * 70}\n[{idx}/{len(todo)}] {describe(run)}\n"
              f"  $ {shlex.join(cmd)}\n{'=' * 70}\n")
        # stdout/stderr are inherited so SB3's progress bar shows up live.
        result = subprocess.run(cmd)
        if result.returncode != 0:
            # One bad run must not abort an overnight batch.
            print(f"\n!! FAILED (exit {result.returncode}): {describe(run)}\n")
            failures.append((run, result.returncode))

    elapsed = (time.perf_counter() - started) / 60
    print(f"\n{'=' * 70}")
    print(f"Sweep finished in {elapsed:.1f} min: "
          f"{len(todo) - len(failures)}/{len(todo)} succeeded")
    for run, code in failures:
        print(f"  FAILED (exit {code}): {describe(run)}")
    print(f"{'=' * 70}\n")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
