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

Each writes models + logs to `logs/<name>/`. The best model (by eval reward)
is saved as `best_model.zip` in that folder.

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
# learning curves for every trained agent, overlaid (fast):
python -m evaluate.plots

# + final comparison table and per-metric bar charts (slow — rolls out episodes):
python -m evaluate.plots --full --episodes 20
```

Outputs land in `src/figures/`: `learning_curves.png`, `comparison.csv`,
`comparison.md`, and one bar chart per metric. It auto-discovers every agent
under `src/logs/` — no configuration needed. For the canonical writeup
figures, ONE machine should train all agents under the shared config and
generate the figures in one place.

---

## Project layout

```
cmpt310-racecar/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── common/
    │   ├── config.py           # SHARED hyperparameters — single source of truth
    │   └── env_factory.py      # SHARED env construction (grayscale, frame stack)
    │   └── activation_functions.py
    │   └── fast_config.py
    │   └── visualize.py
    ├── envs/
    │   └── reward_wrapper.py   # SHARED reward shaping (off by default)
    ├── baseline/
    │   └── train_baseline.py   # vanilla SB3 DQN — the reference to beat
    ├── agents/
    │   ├── double_dqn.py       # Ekam + Lex (stub — to implement)
    │   ├── dueling_dqn.py      # Hargun + Evan (stub — to implement)
    │   ├── noisy_net.py        # Lex (stub — to implement)
    │   └── README.md           # ownership + conventions
    ├── evaluate/
    │   └── evaluate.py         # SHARED metrics: reward, completion, collisions...
    │   └── plots.py
    └── notebooks/              # scratch / exploration
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
