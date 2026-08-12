# Running experiments

Everything here runs from `src/`:

```bash
cd src
```

## Why this changed

Every training script used to write into `logs/<agent>/`. Running the same agent
twice — say, once with ReLU and once with GELU — **silently overwrote** the first
run's `best_model.zip` and `evaluations.npz`. Since Milestone 2 is entirely
ablations (activations, reward shaping, target-update variants), that meant a
whole sweep left behind exactly one usable result.

Now every run gets its own directory plus a manifest recording what produced it,
so runs accumulate instead of clobbering each other and you can tell them apart
months later.

---

## 1. Launching a run

```bash
python -m baseline.train_baseline --activation relu --reward-wrapper off
python -m agents.double_dqn       --activation gelu --reward-wrapper off --seed 43
python -m agents.dueling_dqn      --activation elu  --reward-wrapper on
```

### `--reward-wrapper` is required and has no default

This is the one thing that will break your muscle memory. A bare
`python -m agents.double_dqn` now exits with:

```
error: the following arguments are required: --reward-wrapper
```

That is deliberate. The three scripts used to hardcode *different* values
(baseline `on`, double `off`, dueling `on`), so any default we picked would
silently change somebody's results. **Which one we standardise on is still a
group decision** — until it's made, you state it per run and it gets recorded.

Reward shaping applies to **training only**. Evaluation always uses the raw env
reward (see §4), so `--reward-wrapper on` never inflates your reported numbers.

### All shared flags

| Flag | Default | What it does |
|---|---|---|
| `--reward-wrapper {on,off}` | **required** | shaped reward during training |
| `--activation` | `elu` | policy nonlinearity (`relu`, `gelu`, `mish`, …) |
| `--seed` | `config.SEED` (42) | env + agent seed |
| `--timesteps` | `config.TOTAL_TIMESTEPS` | training budget |
| `--eval-freq` | `config.EVAL_FREQ` | evaluate/checkpoint every N steps |
| `--n-eval-episodes` | `config.N_EVAL_EPISODES` | episodes per evaluation |
| `--log-root` | `./logs` | where run directories go |
| `--note` | `""` | free text saved in the manifest ("why did I run this?") |

Defaults still come from `common/config.py`, so leaving a flag off gives you the
shared settings exactly as before.

---

## 2. What gets recorded

```
logs/baseline_dqn/act-relu_rw-off_seed-42_20260727-1430/
├── run_config.json      # everything needed to reproduce this run
├── results.json         # best eval reward, when it happened, wall-clock time
├── evaluations.npz      # the learning curve (written by SB3's EvalCallback)
├── best_model.zip       # best model by eval reward
├── final_model_*.zip
├── checkpoints/
└── tensorboard/
```

The directory name encodes the settings that vary, so `ls logs/baseline_dqn/` is
already a readable experiment list.

**`run_config.json`** holds: agent + agent class, activation, seed, timesteps,
reward-wrapper on/off *and* the wrapper's coefficients, all of
`config.dqn_kwargs()`, the env settings (frame stack, grayscale, discrete, lap
percent), the exact command line, device, Python/platform, and the **git commit
+ branch + whether the tree was dirty**. If `dirty` is `true`, the commit hash
alone won't reproduce that run — commit before long runs you intend to report.

**`results.json`** holds `best_eval_reward`, `best_eval_timestep`,
`final_eval_reward`, and `train_seconds` (wall clock).

Nothing here is required for training to work — if git is missing or a file is
half-written, it degrades to `"unknown"` rather than taking down your run.

### TensorBoard

```bash
# from the repo root
tensorboard --logdir src/logs/
```

---

## 3. Comparing runs

```bash
# What have I got?
python -m evaluate.plots --list

# Everything in logs/, overlaid
python -m evaluate.plots

# The ablation figure: one agent, one curve per activation
python -m evaluate.plots --filter agent=baseline_dqn --group-by activation

# Filters are ANDed, and repeatable
python -m evaluate.plots --filter agent=double_dqn --filter reward_wrapper=off

# Dump every run and its settings to figures/runs_index.csv
python -m evaluate.plots --index

# Also re-evaluate each best_model.zip -> comparison table + bar charts (slow)
python -m evaluate.plots --full
```

`--filter` and `--group-by` accept **any** field in `run_config.json` or
`results.json`, including nested ones flattened with dots:
`agent`, `activation`, `seed`, `reward_wrapper`, `timesteps`, `git.commit`,
`dqn_kwargs.learning_rate`, `env.n_stack`, … Run `--list` to see the full list
for your logs.

**Seeds are averaged automatically.** Runs identical except for `--seed` collapse
into one curve with a mean ± std band, labelled `[n seeds]`. That's what you want
in the writeup — a single seed proves very little on CarRacing.

**Old runs still work.** Pre-existing flat `logs/<agent>/` directories are still
discovered and plotted, marked `(legacy)`. They have no manifest, so they drop
out of any `--filter` on a field they don't have — that's expected, not a bug.

---

## 4. Evaluation: fixed tracks

CarRacing generates a **new random track on every reset**, so evaluating two
models on different tracks means part of the difference you measure is just track
luck. `config.EVAL_SEEDS` is 20 fixed seeds, and every model in the comparison is
rolled out on exactly those tracks in that order:

```python
from common import config
from common.env_factory import make_eval_vec
from evaluate.evaluate import evaluate_agent

env = make_eval_vec()                       # n_envs=1, NO reward wrapper
metrics = evaluate_agent(model, env,
                         n_episodes=len(config.EVAL_SEEDS),
                         seeds=config.EVAL_SEEDS)
```

`plots.py --full` does this for you. **Don't edit `EVAL_SEEDS`** once results are
collected — it would invalidate every earlier comparison.

Two rules `make_eval_vec()` bakes in so nobody has to remember them:

1. **Never use the reward wrapper for evaluation.** Beyond fairness, collision
   detection depends on the raw `-100` off-playfield spike, which shaping masks.
2. `n_envs=1`, because `evaluate_agent` rolls out one episode at a time.

**Caveat:** the during-training curve in `evaluations.npz` comes from SB3's
`EvalCallback`, which gives no hook for a fixed seed set — those points are on
random tracks and are noisier. The *learning curves* are for trends; the
*comparison table* from `--full` is the seed-fixed, quotable number.

---

## 5. Sweeps

Describe the grid in JSON, then let it run unattended:

```bash
python -m experiments.sweep --config experiments/sweeps/activation.json --dry-run
python -m experiments.sweep --config experiments/sweeps/activation.json
```

```json
{
  "agents": ["baseline_dqn"],
  "activation": ["relu", "elu", "gelu", "mish"],
  "reward_wrapper": ["off"],
  "seed": [42, 43],
  "timesteps": 750000
}
```

List values are swept (full cartesian product — that's 8 runs); scalars apply to
every run. Always `--dry-run` first to see the count before committing GPU hours.

**It's resumable.** Before launching anything it reads back what's already in
`logs/` and skips combinations that already have a *completed* run (one with a
`results.json`). Kill it, restart it, and it continues. A run that crashed
half-way has no `results.json`, so it gets retried. `--force` redoes everything.
A failing run is reported and the batch continues rather than aborting.

---

## 6. Tests — run these before you merge

```bash
pytest src/tests/ -v                  # ~10s, everything
pytest src/tests/ -v -m "not slow"    # ~3s, skips the ones that train
```

52 tests covering: every module imports (a merge once shipped a syntax error that
broke two of three agents — this catches that class of bug), the observation
pipeline is `(4, 84, 84)` / `Discrete(5)`, each agent trains 200 steps, the
death-vs-timeout classification the collision metric depends on, the run-tracking
round trip, and plotting from synthetic log dirs.

The suite takes seconds precisely so there's no excuse to skip it.

---

## 7. Adding a new agent

Say you're adding `agents/noisy_net.py`. Three calls wire it into all of the
above — copy the pattern from `baseline/train_baseline.py`:

```python
from common import run_tracking

parser.add_argument("--activation", ...)
run_tracking.add_common_args(parser)                     # 1. shared flags
args = parser.parse_args()

ctx = run_tracking.start_run("noisy_dqn", args, "NoisyDQN")   # 2. dir + manifest
log_dir = str(ctx.run_dir)
use_reward_wrapper = args.reward_wrapper == "on"
...
model.learn(total_timesteps=args.timesteps, ...)
run_tracking.finish_run(ctx)                             # 3. results.json
```

Then, to make it sweepable, add one line to `AGENT_MODULES` in
`experiments/sweep.py`, add the module to `MODULES` in `tests/test_smoke.py`, and
give it a colour in `AGENT_COLORS` in `evaluate/plots.py` so it's consistent
across every figure.
