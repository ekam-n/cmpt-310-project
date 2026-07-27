"""
Per-run tracking: one directory per experiment, with a manifest.

WHY THIS EXISTS:
Every training script used to write to logs/<agent>/, so running the same agent
twice with different settings SILENTLY OVERWROTE the first run's best_model.zip
and evaluations.npz. Milestone 2 is all ablations (activation sweep, reward
shaping on/off, target-update variants), so we need every run to survive and to
carry enough metadata to tell them apart afterwards.

WHAT A RUN DIRECTORY LOOKS LIKE:

    logs/baseline_dqn/act-relu_rw-off_seed-42_20260727-1430/
        run_config.json     <- everything needed to reproduce this run
        results.json        <- final metrics + best eval + wall-clock seconds
        evaluations.npz     <- written by SB3's EvalCallback (learning curve)
        best_model.zip      <- written by SB3's EvalCallback
        final_model_*.zip
        checkpoints/
        tensorboard/

TRAINING SCRIPTS ONLY NEED THREE CALLS:

    parser = argparse.ArgumentParser(...)
    parser.add_argument("--activation", ...)
    run_tracking.add_common_args(parser)          # 1. shared flags
    args = parser.parse_args()

    ctx = run_tracking.start_run("baseline_dqn", args, "DQN")   # 2. dir + manifest
    log_dir = ctx.run_dir
    ...train...
    run_tracking.finish_run(ctx)                  # 3. results.json

READING RUNS BACK:

    from common.run_tracking import load_runs, write_runs_index
    runs = load_runs()                 # list of flat dicts, one per run
    write_runs_index(runs, "figures/runs_index.csv")
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import config

RUN_CONFIG_NAME = "run_config.json"
RESULTS_NAME = "results.json"
EVALUATIONS_NAME = "evaluations.npz"

# How the run-directory name is built. Kept short so paths stay usable on
# Windows, where the whole path (not just the name) is length-limited.
MAX_TAG_VALUE_LEN = 16
MAX_DIR_NAME_LEN = 100

# Underscores are kept: agent names are snake_case (baseline_dqn) and the
# directory under logs/ must keep matching the agent name that goes into the
# manifests and into `--filter agent=...`, including for pre-existing flat dirs.
_SLUG_STRIP = re.compile(r"[^a-z0-9._-]+")


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------

def slugify(value: Any, max_len: int = MAX_TAG_VALUE_LEN) -> str:
    """Make an arbitrary tag value safe for a directory name."""
    text = str(value).strip().lower()
    text = _SLUG_STRIP.sub("-", text).strip("-._")
    if not text:
        text = "na"
    return text[:max_len]


def make_run_dir(agent_name: str, tags: dict | None = None,
                 log_root: str | Path | None = None,
                 timestamp: str | None = None) -> Path:
    """Create and return logs/<agent>/<tag1>-<val1>_..._<YYYYmmdd-HHMM>/.

    Tags are what makes two runs of the same agent distinguishable at a glance,
    e.g. {"act": "relu", "rw": "off", "seed": 42}. If a directory with the same
    name already exists (two runs launched inside the same minute) a -1, -2, ...
    suffix is appended rather than reusing it -- never overwrite a run.
    """
    root = Path(log_root) if log_root is not None else Path(config.LOG_ROOT)
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M")

    parts = [f"{slugify(k, 8)}-{slugify(v)}" for k, v in (tags or {}).items()]
    parts.append(stamp)
    name = "_".join(parts)[:MAX_DIR_NAME_LEN].strip("_-")

    base = root / slugify(agent_name, 40)
    run_dir = base / name
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{name}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    """Run a git command, returning None on any failure (no git, no repo, ...)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info() -> dict:
    """Commit hash / branch / dirty state, degrading gracefully to 'unknown'.

    'dirty' matters: a run made from an uncommitted tree is not reproducible
    from the commit hash alone, so we record that fact instead of pretending.
    """
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        # None (not False) when we genuinely could not tell.
        "dirty": None if status is None else bool(status.strip()),
    }


def reward_wrapper_coefficients() -> dict:
    """The reward wrapper's coefficient defaults, read off its signature.

    Recorded on every run (even ones with the wrapper off) so that if somebody
    later tunes the coefficients we can tell which runs used which values.
    """
    try:
        import inspect
        from envs.reward_wrapper import CarRacingRewardWrapper
        sig = inspect.signature(CarRacingRewardWrapper.__init__)
        return {
            name: p.default
            for name, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty
        }
    except Exception:                                  # pragma: no cover
        return {}


def env_settings() -> dict:
    """Environment/preprocessing settings that must match across compared runs."""
    return {
        "env_id": config.ENV_ID,
        "continuous": config.CONTINUOUS,
        "grayscale": config.GRAYSCALE,
        "n_stack": config.N_STACK,
        "n_envs": config.N_ENVS,
        "lap_complete_percent": config.LAP_COMPLETE_PERCENT,
    }


# ---------------------------------------------------------------------------
# Writing the manifests
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Best-effort conversion so json.dump never explodes on a run manifest."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    # numpy scalars and anything else exotic
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(obj)


def write_run_config(run_dir: str | Path, **fields: Any) -> Path:
    """Write run_config.json -- everything needed to reproduce this run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(fields)
    path = run_dir / RUN_CONFIG_NAME
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def write_results(run_dir: str | Path, metrics: dict | None = None,
                  **extra: Any) -> Path:
    """Write results.json -- final metrics plus best-eval and timing info."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _json_safe({**(metrics or {}), **extra})
    path = run_dir / RESULTS_NAME
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def best_eval_from_npz(run_dir: str | Path) -> dict:
    """Peak of the learning curve, from the EvalCallback's evaluations.npz.

    Returns {} when the file is absent (short run, no EvalCallback) or unreadable.
    """
    npz_path = Path(run_dir) / EVALUATIONS_NAME
    if not npz_path.exists():
        return {}
    try:
        import numpy as np
        data = np.load(npz_path)
        timesteps = data["timesteps"]
        results = data["results"]              # (n_evals, n_eval_episodes)
        if len(timesteps) == 0:
            return {}
        per_eval_mean = results.mean(axis=1)
        best_idx = int(per_eval_mean.argmax())
        return {
            "best_eval_reward": float(per_eval_mean[best_idx]),
            "best_eval_timestep": int(timesteps[best_idx]),
            "final_eval_reward": float(per_eval_mean[-1]),
            "n_evaluations": int(len(timesteps)),
        }
    except Exception as exc:                           # pragma: no cover
        print(f"  (could not read {npz_path}: {exc})")
        return {}


# ---------------------------------------------------------------------------
# Reading runs back
# ---------------------------------------------------------------------------

def flatten(prefix: str, obj: Any, out: dict) -> dict:
    """Flatten nested dicts into dotted keys, so one run == one flat record."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj
    return out


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None (with a warning) if it is unusable.

    Partial files are expected in the wild: a run killed mid-write, or a run
    still in progress that has run_config.json but not yet results.json.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  (skipping unreadable {path}: {exc})")
        return None
    return data if isinstance(data, dict) else None


def _looks_like_run(d: Path) -> bool:
    """A directory is a run if it holds a manifest or any training output."""
    return any((d / n).exists() for n in (
        RUN_CONFIG_NAME, RESULTS_NAME, EVALUATIONS_NAME, "best_model.zip"))


def load_runs(log_root: str | Path | None = None) -> list[dict]:
    """Return one flat record per run found under log_root.

    Handles BOTH layouts:
      - new, nested:  logs/<agent>/<run_name>/{run_config,results}.json
      - legacy, flat: logs/<agent>/evaluations.npz        (no manifests)

    Legacy runs still appear (so teammates' existing logs keep plotting) but
    carry only agent/run_dir/legacy -- there is no metadata to filter them on.
    Missing or malformed JSON never raises; the run is reported with whatever
    could be read.
    """
    root = Path(log_root) if log_root is not None else Path(config.LOG_ROOT)
    runs: list[dict] = []
    if not root.exists():
        return runs

    for agent_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        nested = sorted(p for p in agent_dir.iterdir()
                        if p.is_dir() and _looks_like_run(p))
        for run_dir in nested:
            runs.append(_record_for(run_dir, agent_dir.name, legacy=False))
        # A flat legacy run: outputs sit directly in logs/<agent>/.
        if _looks_like_run(agent_dir):
            runs.append(_record_for(agent_dir, agent_dir.name, legacy=True))
    return runs


def _record_for(run_dir: Path, agent_fallback: str, legacy: bool) -> dict:
    record: dict[str, Any] = {}
    for name in (RUN_CONFIG_NAME, RESULTS_NAME):
        data = _read_json(run_dir / name)
        if data:
            flatten("", data, record)
    record.setdefault("agent", agent_fallback)
    record["run_dir"] = str(run_dir)
    record["run_name"] = run_dir.name
    record["legacy"] = legacy
    record["has_curve"] = (run_dir / EVALUATIONS_NAME).exists()
    record["has_model"] = (run_dir / "best_model.zip").exists()
    record["complete"] = (run_dir / RESULTS_NAME).exists()
    return record


def write_runs_index(runs: list[dict], csv_path: str | Path) -> Path:
    """Dump every run to one CSV row. Stdlib csv -- pandas is not a dependency."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Stable, readable column order: identity first, then everything else.
    lead = ["agent", "run_name", "activation", "reward_wrapper", "seed",
            "timesteps", "best_eval_reward", "best_eval_timestep",
            "train_seconds", "git.commit", "git.dirty", "run_dir"]
    keys = sorted({k for r in runs for k in r})
    header = [k for k in lead if k in keys] + [k for k in keys if k not in lead]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow(run)
    return csv_path


def runs_to_dataframe(runs: list[dict] | None = None,
                      log_root: str | Path | None = None):
    """Optional pandas view of the runs. pandas is NOT required by this repo."""
    try:
        import pandas as pd
    except ImportError as exc:                         # pragma: no cover
        raise ImportError(
            "runs_to_dataframe() needs pandas, which is not in "
            "requirements.txt. Either `pip install pandas` or use "
            "write_runs_index() / load_runs() instead."
        ) from exc
    return pd.DataFrame(load_runs(log_root) if runs is None else runs)


# ---------------------------------------------------------------------------
# The training-script API
# ---------------------------------------------------------------------------

# Historical hardcoded reward-wrapper setting per script, quoted in --help so
# people know what they used to get. The flag is REQUIRED precisely because
# these disagree and picking a default would silently change someone's results;
# which one is "right" is a pending group decision.
HISTORICAL_REWARD_WRAPPER = {
    "baseline_dqn": "on",
    "double_dqn": "off",
    "dueling_dqn": "on",
}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the flags every training script shares. Defaults come from config."""
    parser.add_argument(
        "--reward-wrapper", choices=["on", "off"], required=True,
        help="REQUIRED, no default on purpose: use the shaped reward wrapper "
             "(envs.reward_wrapper) for TRAINING or not. The three scripts used "
             "to hardcode different values (baseline=on, double=off, "
             "dueling=on), so any default would silently change somebody's "
             "results. Evaluation always ignores this and uses the raw env "
             "reward. Your choice is recorded in run_config.json.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED,
        help=f"Random seed for env + agent (default: {config.SEED}).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=config.TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {config.TOTAL_TIMESTEPS:,}).",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=config.EVAL_FREQ,
        help=f"Evaluate/checkpoint every N steps (default: {config.EVAL_FREQ:,}).",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=config.N_EVAL_EPISODES,
        help=f"Episodes per evaluation (default: {config.N_EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--log-root", default=config.LOG_ROOT,
        help=f"Where run directories are created (default: {config.LOG_ROOT}).",
    )
    parser.add_argument(
        "--note", default="",
        help="Free-text note stored in run_config.json (e.g. why you ran this).",
    )
    return parser


@dataclass
class RunContext:
    """Handle returned by start_run() and passed back to finish_run()."""
    run_dir: Path
    agent_name: str
    args: argparse.Namespace
    start_time: float = field(default_factory=time.perf_counter)
    started_at: str = ""


def start_run(agent_name: str, args: argparse.Namespace,
              model_cls_name: str, extra_tags: dict | None = None,
              extra_config: dict | None = None) -> RunContext:
    """Create the run directory, write run_config.json, start the clock."""
    activation = getattr(args, "activation", None)
    tags = {}
    if activation:
        tags["act"] = activation
    tags["rw"] = args.reward_wrapper
    tags["seed"] = args.seed
    tags.update(extra_tags or {})

    run_dir = make_run_dir(agent_name, tags, log_root=getattr(args, "log_root", None))
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    write_run_config(
        run_dir,
        agent=agent_name,
        agent_class=model_cls_name,
        activation=activation,
        seed=args.seed,
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        reward_wrapper=args.reward_wrapper,
        reward_wrapper_coefs=reward_wrapper_coefficients(),
        note=getattr(args, "note", ""),
        # `seed` inside dqn_kwargs comes from config; the CLI --seed above is
        # what actually gets used, and both are recorded so a mismatch is visible.
        dqn_kwargs=config.dqn_kwargs(),
        env=env_settings(),
        cli_args=vars(args),
        git=git_info(),
        started_at=started_at,
        python=platform.python_version(),
        platform=platform.platform(),
        device=_torch_device(),
        command=" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        **(extra_config or {}),
    )

    print(f"\n[run] {run_dir}")
    print(f"[run] reward wrapper: {args.reward_wrapper} | "
          f"activation: {activation} | seed: {args.seed} | "
          f"timesteps: {args.timesteps:,}\n")
    return RunContext(run_dir=run_dir, agent_name=agent_name, args=args,
                      started_at=started_at)


def _torch_device() -> str:
    """Which device SB3 will pick ('auto'), recorded for timing comparisons."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:                                  # pragma: no cover
        return "unknown"


def finish_run(ctx: RunContext, metrics: dict | None = None,
               **extra: Any) -> Path:
    """Stop the clock, scrape the best eval off evaluations.npz, write results."""
    train_seconds = time.perf_counter() - ctx.start_time
    payload = {
        "agent": ctx.agent_name,
        "train_seconds": round(train_seconds, 2),
        "started_at": ctx.started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **best_eval_from_npz(ctx.run_dir),
        **(metrics or {}),
        **extra,
    }
    path = write_results(ctx.run_dir, payload)
    print(f"\n[run] finished in {train_seconds / 60:.1f} min -> {ctx.run_dir}")
    if "best_eval_reward" in payload:
        print(f"[run] best eval reward {payload['best_eval_reward']:.1f} "
              f"at timestep {payload['best_eval_timestep']:,}")
    return path
