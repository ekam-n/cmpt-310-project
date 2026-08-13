# CMPT 310 — Autonomous Racecar (Group 20)

Reinforcement learning agents that learn to drive [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/).

**Goal:** implement DQN improvements (Double DQN, Dueling DQN, and more) and
measure them against the Stable-Baselines3 vanilla DQN baseline under identical
training conditions.

---

## Quick start

```bash
# 1. clone
git clone <your-repo-url> cmpt310-racecar
cd cmpt310-racecar

# 2. create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt
```

### macOS note (box2d)

If `pip install -r requirements.txt` fails while building **box2d**, install
SWIG first, then retry:

```bash
brew install swig
pip install -r requirements.txt
```

### Verify the environment works

```bash
cd src
python -c "from common.env_factory import make_vec; e=make_vec(); print('obs shape:', e.observation_space.shape); e.close()"
```

You should see an observation shape of `(4, 84, 84)` — 4 stacked grayscale frames.

---

## Running things

All commands run **from the `src/` directory** (so the package imports resolve):

```bash
cd src
```

| What                        | Command                                  | Owners        |
|-----------------------------|------------------------------------------|---------------|
| Train baseline DQN          | `python -m baseline.train_baseline`      | shared ref    |
| Train Double DQN            | `python -m agents.double_dqn`            | Ekam, Lex     |
| Train Dueling DQN           | `python -m agents.dueling_dqn`           | Hargun, Evan  |
| Train Noisy Network         | `python -m agents.noisy_net`             | Lex           |

Each training script accepts `--activation <name>` so you can compare
nonlinearities without changing model code. Available options are `relu`,
`elu`, `leaky_relu`, `tanh`, `sigmoid`, `gelu`, `silu`, `swish`, `mish`, and
`identity`.

**`--reward-wrapper {on,off}` is required** — it has no default, because the
three scripts used to hardcode different values and any default would silently
change somebody's results:

```bash
python -m baseline.train_baseline --activation relu --reward-wrapper off
```

Each run writes to its own directory, `logs/<agent>/<settings>_<timestamp>/`,
along with a `run_config.json` recording exactly what produced it (including the
git commit). Runs no longer overwrite each other. The best model (by eval
reward) is `best_model.zip` inside that run directory.

**See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** for launching runs, comparing
them (`--filter` / `--group-by`), the fixed evaluation seeds, running sweeps, and
the test suite.

### Before you merge

```bash
pytest src/tests/ -v        # ~10 seconds
```

### Watch training progress

TensorBoard must be pointed at `src/logs/` — training scripts run from `src/`,
so that's where logs land. From the **repo root**:

```bash
tensorboard --logdir src/logs/
```

(Running `tensorboard --logdir logs/` from the repo root finds nothing — that
was why it appeared broken.) Then open http://localhost:6006.

### Comparison charts & tables (for the writeup)

`evaluate/plots.py` turns training logs into the figures we need. From `src/`:

```bash
# learning curves for every run, overlaid (fast):
python -m evaluate.plots

# slice the runs: one agent, one curve per activation:
python -m evaluate.plots --filter agent=baseline_dqn --group-by activation

# list every run and its settings:
python -m evaluate.plots --list
python -m evaluate.plots --index      # -> figures/runs_index.csv

# + final comparison table and per-metric bar charts (slow — rolls out episodes):
python -m evaluate.plots --full
```

Outputs land in `src/figures/`: `learning_curves.png`, `comparison.csv`,
`comparison.md`, `runs_index.csv`, and one bar chart per metric. It auto-discovers
every run under `src/logs/` — including older flat `logs/<agent>/` runs from
before per-run directories existed. Runs that differ only by seed are averaged
into one curve with a std band. The `--full` table evaluates every model on the
same 20 fixed tracks (`config.EVAL_SEEDS`) so the comparison isn't confounded by
track luck. For the canonical writeup figures, ONE machine should train all
agents under the shared config and generate the figures in one place.

### Analysis scripts (per-run diagnostics)

Four standalone scripts in `src/` for digging into trained runs. All run from
`src/`; where a run directory is expected, pass the whole
`logs/<agent>/<settings>_<timestamp>/` path (the script loads its
`best_model.zip`):

```bash
# watch a trained agent drive (window pops up):
python watch.py logs/noisy_dqn/act-relu_rw-off_seed-1_20260806-0034
python watch.py <run_dir> --seed 1001       # fixed track from EVAL_SEEDS
python watch.py <run_dir> --headless        # no window; saves 3 PNG frames instead

# first timestep each run reaches 90% of its best eval reward (scans all of logs/):
python convergence.py [--thresh 0.9]

# how close an agent gets to lap completion (tiles visited per episode):
python check_tiles.py <run_dir> [--episodes 5]

# driving pace + action mix for TWO agents on the SAME fixed eval tracks:
python speed_check.py <run_dir_a> <run_dir_b> [--episodes 5]
```

`speed_check.py` and `check_tiles.py` pin tracks exactly the way
`evaluate.py` does (`EVAL_SEEDS`, reward wrapper off, deterministic actions),
so their numbers are directly comparable to the official evaluation.

---

## Project layout

```
cmpt-310-project/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── EXPERIMENTS.md          # launching runs, sweeps, comparisons, tests
├── notebooks/                  # scratch / exploration
├── box2d/                      # copy of Gymnasium's CarRacing source (reference only — not imported)
└── src/
    ├── common/
    │   ├── config.py           # SHARED hyperparameters — single source of truth
    │   ├── env_factory.py      # SHARED env construction (grayscale, frame stack)
    │   ├── activation_functions.py
    │   ├── fast_config.py
    │   ├── run_tracking.py     # per-run directories + run_config.json manifests
    │   └── visualize.py
    ├── envs/
    │   └── reward_wrapper.py   # SHARED reward shaping (--reward-wrapper on/off)
    ├── baseline/
    │   └── train_baseline.py   # vanilla SB3 DQN — the reference to beat
    ├── agents/
    │   ├── double_dqn.py       # Ekam + Lex
    │   ├── dueling_dqn.py      # Hargun + Evan
    │   ├── noisy_net.py        # Lex
    │   └── README.md           # ownership + conventions
    ├── evaluate/
    │   ├── evaluate.py         # SHARED metrics: reward, completion, collisions...
    │   └── plots.py            # learning curves, comparison table, bar charts
    ├── experiments/
    │   ├── sweep.py            # batch-run many configs (see docs/EXPERIMENTS.md)
    │   └── sweeps/             # sweep definitions (activation.json, ...)
    ├── tests/                  # smoke tests — run before merging
    ├── notes/                  # per-person scratch notes
    ├── watch.py                # ┐
    ├── convergence.py          # │ analysis scripts — see
    ├── check_tiles.py          # │ "Analysis scripts" above
    └── speed_check.py          # ┘
```

## Why it's structured this way

`config.py` and `env_factory.py` are **shared and imported by everyone**, so the
only thing that differs between the baseline and each contribution is the
algorithm itself — not the environment or the hyperparameters. That's what makes
the comparison fair (and is exactly what the proposal feedback asked for).

## Evaluation metrics

`evaluate/evaluate.py` reports, for any trained model:
- mean ± std cumulative reward
- lap completion rate
- collision / off-track rate
- mean episode length
- mean timesteps to first completed lap

Use the same function on every model so the comparison table in the writeup is
apples-to-apples.

## Workflow / git etiquette

- Branch per person/feature: `double-dqn-ekam`, `dueling-hargun`, etc.
- Don't commit `logs/` (already gitignored — models are big).
- Touch shared files (`config.py`, `env_factory.py`) only with a heads-up to
  the group, since everyone depends on them.

## License
### Stable-Baselines3
MIT License

Copyright (c) 2020 Stable-Baselines Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Gymnasium
The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
