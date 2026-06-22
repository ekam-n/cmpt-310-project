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

Each writes models + logs to `logs/<name>/`. The best model (by eval reward)
is saved as `best_model.zip` in that folder.

### Watch training progress

```bash
tensorboard --logdir logs/
```

---

## Project layout

```
cmpt310-racecar/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── common/
    │   ├── config.py          # SHARED hyperparameters — single source of truth
    │   └── env_factory.py     # SHARED env construction (grayscale, frame stack)
    ├── envs/
    │   └── reward_wrapper.py   # SHARED reward shaping (off by default)
    ├── baseline/
    │   └── train_baseline.py   # vanilla SB3 DQN — the reference to beat
    ├── agents/
    │   ├── double_dqn.py       # Ekam + Lex (stub — to implement)
    │   ├── dueling_dqn.py      # Hargun + Evan (stub — to implement)
    │   └── README.md           # ownership + conventions
    ├── evaluate/
    │   └── evaluate.py         # SHARED metrics: reward, completion, collisions...
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
