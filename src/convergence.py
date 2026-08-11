"""First timestep where each run reaches >= THRESH of its best eval reward.
Usage (from src/):  python convergence.py [--thresh 0.9]
"""
import argparse
from pathlib import Path
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("--thresh", type=float, default=0.9)
args = parser.parse_args()

rows = []
for npz in sorted(Path("logs").glob("*/*/evaluations.npz")):
    run_dir = npz.parent
    data = np.load(npz)
    t, r = data["timesteps"], data["results"].mean(axis=1)
    best = r.max()
    # threshold on the span above the floor so negative early rewards don't distort
    floor = r.min()
    target = floor + args.thresh * (best - floor)
    idx = np.argmax(r >= target)          # first index meeting target
    conv = int(t[idx])
    # label from manifest if present
    label = run_dir.parent.name
    cfg = run_dir / "run_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        label = f"{c.get('agent', label)} rw={c.get('reward_wrapper','?')}"
    rows.append((label, conv, best))

print(f"{'run':<28} {'converged_at':>12} {'best_reward':>12}")
for label, conv, best in rows:
    print(f"{label:<28} {conv:>12,} {best:>12.1f}")