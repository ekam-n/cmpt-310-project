"""
Smoke tests -- run these before you merge.

    pytest src/tests/ -v                  # everything
    pytest src/tests/ -v -m "not slow"    # skip the ones that train / roll out

WHY THIS EXISTS: a merge once shipped a syntax error that broke two of three
agents and nobody noticed until a training run failed minutes in. The import
test below alone would have caught that. The rest cover the things that are
easy to break silently: the observation pipeline, the death-vs-timeout
classification the collision metric depends on, and the run-tracking round trip
that every experiment now relies on.

These are smoke tests, not correctness proofs -- they check that things run and
produce the right shapes/classifications, not that any agent learns well.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# 1. Everything imports (catches syntax errors and broken merges)
# ---------------------------------------------------------------------------

MODULES = [
    "common.config",
    "common.fast_config",
    "common.env_factory",
    "common.activation_functions",
    "common.run_tracking",
    "envs.reward_wrapper",
    "evaluate.evaluate",
    "evaluate.plots",
    "baseline.train_baseline",
    "agents.double_dqn",
    "agents.dueling_dqn",
    "agents.noisy_net",
    "experiments.sweep",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    """Every module must at least import. This is the cheap merge guard."""
    import importlib
    importlib.import_module(module_name)


def test_activation_functions_all_constructible():
    from common.activation_functions import (
        ACTIVATION_FUNCTIONS, available_activation_names, get_activation_fn)

    for name in available_activation_names():
        get_activation_fn(name)()          # must be instantiable, not just present
    assert set(available_activation_names()) == set(ACTIVATION_FUNCTIONS)
    with pytest.raises(ValueError):
        get_activation_fn("definitely-not-an-activation")


def test_eval_seeds_are_fixed_and_unique():
    """The comparison depends on these never drifting."""
    from common import config

    assert len(config.EVAL_SEEDS) == 20
    assert len(set(config.EVAL_SEEDS)) == 20, "duplicate eval seeds"
    assert all(isinstance(s, int) for s in config.EVAL_SEEDS)


# ---------------------------------------------------------------------------
# 2. The observation pipeline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vec_env():
    """One CarRacing vec env shared by the tests that need a real env."""
    from common.env_factory import make_vec
    env = make_vec(use_reward_wrapper=False)
    yield env
    env.close()


@pytest.mark.slow
def test_make_vec_observation_and_action_space(vec_env):
    """4 stacked 84x84 grayscale frames, channels-first, and 5 discrete actions.

    If this changes, every trained model in logs/ becomes incomparable.
    """
    from gymnasium import spaces

    assert vec_env.observation_space.shape == (4, 84, 84)
    assert isinstance(vec_env.action_space, spaces.Discrete)
    assert vec_env.action_space.n == 5

    obs = vec_env.reset()
    assert obs.shape == (1, 4, 84, 84)
    assert obs.dtype == np.uint8


@pytest.mark.slow
def test_make_eval_vec_has_no_reward_wrapper():
    """Eval must use the raw env reward -- collision detection depends on the
    unmasked -100 death spike."""
    from common.env_factory import make_eval_vec
    from envs.reward_wrapper import CarRacingRewardWrapper

    env = make_eval_vec()
    try:
        inner = env.venv.venv.envs[0]
        wrappers = []
        while hasattr(inner, "env"):
            wrappers.append(type(inner))
            inner = inner.env
        assert CarRacingRewardWrapper not in wrappers
    finally:
        env.close()


@pytest.mark.slow
def test_seeded_reset_is_reproducible(vec_env):
    """Fixed eval seeds only work if seeding really does fix the track."""
    def first_obs(seed):
        vec_env.seed(seed)
        return np.asarray(vec_env.reset()).copy()

    assert np.array_equal(first_obs(1001), first_obs(1001))
    assert not np.array_equal(first_obs(1001), first_obs(1002))


# ---------------------------------------------------------------------------
# 3. Each agent trains a few steps without blowing up
# ---------------------------------------------------------------------------

TINY = dict(
    buffer_size=300,
    learning_starts=50,
    batch_size=8,
    train_freq=8,
    gradient_steps=1,
    target_update_interval=50,
    learning_rate=1e-4,
    verbose=0,
    seed=0,
)


def _build_agent(agent, env):
    """Construct each agent exactly the way its training script does."""
    from stable_baselines3 import DQN

    if agent == "baseline_dqn":
        return DQN("CnnPolicy", env, **TINY)
    if agent == "double_dqn":
        from agents.double_dqn import DoubleDQN
        return DoubleDQN("CnnPolicy", env, **TINY)
    if agent == "dueling_dqn":
        from agents.dueling_dqn import DuelingDQNPolicy
        DQN.policy_aliases["DuelingDQNPolicy"] = DuelingDQNPolicy
        return DQN("DuelingDQNPolicy", env, **TINY)
    if agent == "noisy_dqn":
        from agents.noisy_net import NoisyDoubleDuelingDQN, NoisyQNetwork
        DQN.policy_aliases["NoisyDQNPolicy"] = NoisyQNetwork
        return NoisyDoubleDuelingDQN("NoisyDQNPolicy", env, **TINY)
    raise AssertionError(agent)


@pytest.mark.slow
@pytest.mark.parametrize("agent", ["baseline_dqn", "double_dqn", "dueling_dqn",
                                   "noisy_dqn"])
def test_agent_trains_a_few_steps(agent, vec_env):
    """~200 steps end to end: builds, collects, does gradient updates, predicts.

    Enough to catch a broken train() override, a bad policy wiring, or a shape
    mismatch -- all of which have happened here.
    """
    model = _build_agent(agent, vec_env)
    model.learn(total_timesteps=200, progress_bar=False)
    assert model.num_timesteps >= 200

    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,)
    assert 0 <= int(action[0]) < 5


# ---------------------------------------------------------------------------
# 4. Death vs timeout classification (the collision metric)
# ---------------------------------------------------------------------------

class FakeVecEnv:
    """Minimal stand-in for a 1-env SB3 VecEnv.

    evaluate_agent() only needs reset/step/seed, so we can exercise the
    termination classification exhaustively without paying for Box2D or
    waiting 1000 frames per episode.
    """

    def __init__(self, episode_len, final_reward, truncated, step_reward=-0.1):
        self.episode_len = episode_len
        self.final_reward = final_reward
        self.truncated = truncated
        self.step_reward = step_reward
        self.seeds_seen = []
        self.t = 0

    def seed(self, seed):
        self.seeds_seen.append(seed)
        return [seed]

    def reset(self):
        self.t = 0
        return np.zeros((1, 4, 84, 84), dtype=np.uint8)

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        reward = self.final_reward if done else self.step_reward
        info = {}
        if done:
            total = self.step_reward * (self.episode_len - 1) + self.final_reward
            info["episode"] = {"r": total, "l": self.episode_len}
            if self.truncated:
                info["TimeLimit.truncated"] = True
        obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        return obs, np.array([reward]), np.array([done]), [info]


class FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([0]), None


def test_eval_classifies_off_playfield_death():
    """A -100 spike with terminated=True is a collision, not a lap."""
    from evaluate.evaluate import evaluate_agent

    env = FakeVecEnv(episode_len=50, final_reward=-100.0, truncated=False)
    m = evaluate_agent(FakeModel(), env, n_episodes=3)

    assert m["collision_rate"] == 1.0
    assert m["completion_rate"] == 0.0
    assert m["mean_episode_len"] == 50
    assert m["mean_steps_to_lap"] is None


def test_eval_classifies_timeout():
    """Running out of time is neither a collision nor a completed lap.

    This is the random-policy case: the car pootles around for the full 1000
    frames without leaving the playfield.
    """
    from evaluate.evaluate import evaluate_agent

    env = FakeVecEnv(episode_len=1000, final_reward=-0.1, truncated=True)
    m = evaluate_agent(FakeModel(), env, n_episodes=3)

    assert m["collision_rate"] == 0.0
    assert m["completion_rate"] == 0.0
    assert m["mean_episode_len"] == 1000


def test_eval_classifies_completed_lap():
    """terminated without the -100 spike == lap completed (v2+ semantics)."""
    from evaluate.evaluate import evaluate_agent

    env = FakeVecEnv(episode_len=400, final_reward=5.0, truncated=False)
    m = evaluate_agent(FakeModel(), env, n_episodes=2)

    assert m["completion_rate"] == 1.0
    assert m["collision_rate"] == 0.0
    assert m["mean_steps_to_lap"] == 400


def test_eval_uses_the_fixed_seed_set_in_order():
    """Every model must see the same tracks, in the same order."""
    from common import config
    from evaluate.evaluate import evaluate_agent

    env = FakeVecEnv(episode_len=10, final_reward=-100.0, truncated=False)
    m = evaluate_agent(FakeModel(), env, n_episodes=len(config.EVAL_SEEDS),
                       seeds=config.EVAL_SEEDS)

    assert env.seeds_seen == config.EVAL_SEEDS
    assert m["eval_seeds"] == config.EVAL_SEEDS


def test_eval_without_seeds_does_not_seed():
    """Backward compatibility: existing callers keep random tracks."""
    from evaluate.evaluate import evaluate_agent

    env = FakeVecEnv(episode_len=10, final_reward=-100.0, truncated=False)
    m = evaluate_agent(FakeModel(), env, n_episodes=2)

    assert env.seeds_seen == []
    assert m["eval_seeds"] is None


def test_eval_seed_list_wraps_when_more_episodes_than_seeds():
    from evaluate.evaluate import evaluate_agent

    env = FakeVecEnv(episode_len=5, final_reward=-100.0, truncated=False)
    evaluate_agent(FakeModel(), env, n_episodes=5, seeds=[1, 2])

    assert env.seeds_seen == [1, 2, 1, 2, 1]


# ---------------------------------------------------------------------------
# 5. run_tracking round trip
# ---------------------------------------------------------------------------

def test_run_dir_is_unique_per_run(tmp_path):
    """The whole point: two runs of one agent must not collide."""
    from common.run_tracking import make_run_dir

    a = make_run_dir("baseline_dqn", {"act": "relu", "seed": 42},
                     log_root=tmp_path, timestamp="20260727-1430")
    b = make_run_dir("baseline_dqn", {"act": "tanh", "seed": 42},
                     log_root=tmp_path, timestamp="20260727-1430")
    # Same agent, same minute, same tags -> still must not reuse a directory.
    c = make_run_dir("baseline_dqn", {"act": "relu", "seed": 42},
                     log_root=tmp_path, timestamp="20260727-1430")

    assert a != b != c and a != c
    assert len({a, b, c}) == 3
    assert a.parent.name == "baseline_dqn"
    assert "act-relu" in a.name and "20260727-1430" in a.name


def test_run_dir_names_are_filesystem_safe(tmp_path):
    from common.run_tracking import make_run_dir

    run_dir = make_run_dir("weird agent/name", {"act": "Leaky ReLU!!"},
                           log_root=tmp_path)
    for part in (run_dir.name, run_dir.parent.name):
        assert all(ch.isalnum() or ch in "-._" for ch in part), part
        assert len(part) <= 100


def test_run_tracking_round_trip(tmp_path):
    """create dir -> write configs -> load_runs reads it back."""
    from common.run_tracking import (
        load_runs, make_run_dir, write_results, write_run_config)

    run_dir = make_run_dir("double_dqn", {"act": "gelu", "seed": 7},
                           log_root=tmp_path)
    write_run_config(run_dir, agent="double_dqn", activation="gelu", seed=7,
                     reward_wrapper="off", timesteps=1000,
                     dqn_kwargs={"learning_rate": 5e-4, "batch_size": 64},
                     git={"commit": "abc123", "dirty": False})
    write_results(run_dir, {"mean_reward": 12.5},
                  best_eval_reward=30.0, best_eval_timestep=500,
                  train_seconds=61.5)

    runs = load_runs(tmp_path)
    assert len(runs) == 1
    run = runs[0]

    assert run["agent"] == "double_dqn"
    assert run["activation"] == "gelu"
    assert run["seed"] == 7
    assert run["reward_wrapper"] == "off"
    assert run["mean_reward"] == 12.5
    assert run["best_eval_reward"] == 30.0
    assert run["best_eval_timestep"] == 500
    assert run["train_seconds"] == 61.5
    assert run["complete"] is True
    assert run["legacy"] is False
    # nested dicts are flattened, so they can be filtered/grouped on
    assert run["dqn_kwargs.learning_rate"] == 5e-4
    assert run["git.commit"] == "abc123"


def test_load_runs_tolerates_missing_and_broken_files(tmp_path):
    """Half-written runs are normal (killed jobs); they must not crash a plot."""
    from common.run_tracking import load_runs, write_run_config

    # a run that started but never finished: config, no results
    started = tmp_path / "baseline_dqn" / "act-relu_20260727-1400"
    started.mkdir(parents=True)
    write_run_config(started, agent="baseline_dqn", activation="relu")

    # a run whose JSON got truncated mid-write
    broken = tmp_path / "baseline_dqn" / "act-elu_20260727-1500"
    broken.mkdir(parents=True)
    (broken / "run_config.json").write_text('{"agent": "baseline_dqn", "acti')
    (broken / "results.json").write_text("{}")

    # a directory with nothing in it at all -> not a run
    (tmp_path / "baseline_dqn" / "empty").mkdir(parents=True)

    runs = load_runs(tmp_path)
    by_name = {r["run_name"]: r for r in runs}

    assert "act-relu_20260727-1400" in by_name
    assert "act-elu_20260727-1500" in by_name    # survives the bad JSON
    assert "empty" not in by_name

    # The unfinished run has no results.json, so it is not complete...
    unfinished = by_name["act-relu_20260727-1400"]
    assert unfinished["complete"] is False
    assert unfinished["activation"] == "relu"

    # ...while the one with a corrupt config still reports what could be read:
    # results.json exists, so it counts as complete, but the unparseable
    # run_config.json contributes nothing and the agent falls back to the dir.
    corrupt = by_name["act-elu_20260727-1500"]
    assert corrupt["complete"] is True
    assert corrupt["agent"] == "baseline_dqn"
    assert "activation" not in corrupt


def test_load_runs_finds_legacy_flat_dirs(tmp_path):
    """Teammates' pre-existing logs/<agent>/ runs must still be discoverable."""
    from common.run_tracking import load_runs

    legacy = tmp_path / "dueling_dqn"
    legacy.mkdir(parents=True)
    np.savez(legacy / "evaluations.npz",
             timesteps=np.array([100, 200]), results=np.zeros((2, 3)))

    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["legacy"] is True
    assert runs[0]["agent"] == "dueling_dqn"
    assert runs[0]["has_curve"] is True


def test_best_eval_is_read_off_the_npz(tmp_path):
    from common.run_tracking import best_eval_from_npz

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.savez(run_dir / "evaluations.npz",
             timesteps=np.array([100, 200, 300]),
             results=np.array([[0.0, 0.0], [50.0, 70.0], [10.0, 10.0]]))

    best = best_eval_from_npz(run_dir)
    assert best["best_eval_reward"] == 60.0      # mean of the middle eval
    assert best["best_eval_timestep"] == 200
    assert best["final_eval_reward"] == 10.0
    assert best["n_evaluations"] == 3


def test_best_eval_missing_npz_is_not_an_error(tmp_path):
    from common.run_tracking import best_eval_from_npz
    assert best_eval_from_npz(tmp_path) == {}


def test_git_info_never_raises():
    """Must degrade to 'unknown' rather than take down a training run."""
    from common.run_tracking import git_info

    info = git_info()
    assert set(info) == {"commit", "branch", "dirty"}
    assert isinstance(info["commit"], str)


def test_runs_index_csv_has_one_row_per_run(tmp_path):
    import csv as _csv
    from common.run_tracking import load_runs, write_runs_index

    for act in ("relu", "tanh"):
        d = tmp_path / "baseline_dqn" / f"act-{act}_20260727-1400"
        d.mkdir(parents=True)
        (d / "run_config.json").write_text(json.dumps(
            {"agent": "baseline_dqn", "activation": act, "seed": 42}))
        (d / "results.json").write_text(json.dumps({"best_eval_reward": 1.0}))

    out = write_runs_index(load_runs(tmp_path), tmp_path / "runs_index.csv")
    rows = list(_csv.DictReader(open(out)))

    assert len(rows) == 2
    assert {r["activation"] for r in rows} == {"relu", "tanh"}
    assert rows[0]["agent"] == "baseline_dqn"


# ---------------------------------------------------------------------------
# 6. Plots from synthetic log dirs
# ---------------------------------------------------------------------------

def _fake_run(root, agent, run_name, curve=(100, 200, 300), **cfg):
    d = root / agent / run_name
    d.mkdir(parents=True)
    (d / "run_config.json").write_text(json.dumps({"agent": agent, **cfg}))
    (d / "results.json").write_text(json.dumps({"best_eval_reward": 1.0}))
    timesteps = np.array(curve)
    rng = np.random.default_rng(abs(hash(run_name)) % 2**32)
    np.savez(d / "evaluations.npz", timesteps=timesteps,
             results=rng.normal(size=(len(timesteps), 3)))
    return d


@pytest.fixture
def synthetic_logs(tmp_path):
    """Two activations x two seeds, plus one legacy flat dir."""
    for act in ("relu", "tanh"):
        for seed in (42, 43):
            _fake_run(tmp_path, "baseline_dqn", f"act-{act}_seed-{seed}_2026-1",
                      activation=act, seed=seed, reward_wrapper="off")
    _fake_run(tmp_path, "double_dqn", "act-relu_seed-42_2026-1",
              activation="relu", seed=42, reward_wrapper="on")

    legacy = tmp_path / "legacy_dqn"
    legacy.mkdir(parents=True)
    np.savez(legacy / "evaluations.npz", timesteps=np.array([100, 200]),
             results=np.zeros((2, 3)))
    return tmp_path


def test_plots_discovers_nested_and_legacy(synthetic_logs):
    from evaluate.plots import discover_runs

    runs = discover_runs(synthetic_logs)
    assert len(runs) == 6                                  # 5 nested + 1 legacy
    assert sum(r["legacy"] for r in runs) == 1


def test_plots_filter_and_group(synthetic_logs):
    from evaluate.plots import apply_filters, discover_runs, group_runs, parse_filters

    runs = discover_runs(synthetic_logs)
    selected = apply_filters(runs, parse_filters(["agent=baseline_dqn"]))
    assert len(selected) == 4                              # legacy + double drop out

    groups = group_runs(selected, "activation")
    assert set(groups) == {"activation=relu", "activation=tanh"}
    assert all(len(v) == 2 for v in groups.values())        # both seeds per group


def test_plots_group_by_default_aggregates_seeds(synthetic_logs):
    """Runs differing only by seed collapse into one curve, not two."""
    from evaluate.plots import apply_filters, discover_runs, group_runs, parse_filters

    selected = apply_filters(discover_runs(synthetic_logs),
                             parse_filters(["agent=baseline_dqn"]))
    groups = group_runs(selected, None)

    assert len(groups) == 2                                # relu and tanh
    assert all(len(members) == 2 for members in groups.values())


def test_plots_bad_filter_syntax_is_a_clear_error():
    from evaluate.plots import parse_filters

    with pytest.raises(SystemExit, match="key=value"):
        parse_filters(["agent double_dqn"])


def test_learning_curves_png_is_produced(synthetic_logs, tmp_path):
    from evaluate.plots import discover_runs, group_runs, plot_learning_curves

    out = tmp_path / "figures" / "learning_curves.png"
    result = plot_learning_curves(group_runs(discover_runs(synthetic_logs)),
                                 out_path=out)

    assert result == out and out.exists()
    assert out.stat().st_size > 5000                       # a real image
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_legacy_only_logs_still_plot(tmp_path):
    """Backward compatibility: an old flat logs/<agent>/ dir on its own plots."""
    from evaluate.plots import discover_runs, group_runs, plot_learning_curves

    legacy = tmp_path / "logs" / "baseline_dqn"
    legacy.mkdir(parents=True)
    np.savez(legacy / "evaluations.npz", timesteps=np.array([100, 200, 300]),
             results=np.random.default_rng(0).normal(size=(3, 4)))

    out = tmp_path / "legacy.png"
    runs = discover_runs(tmp_path / "logs")
    assert len(runs) == 1 and runs[0]["legacy"] is True
    assert plot_learning_curves(group_runs(runs), out_path=out) == out
    assert out.exists()


def test_comparison_row_names_are_unique_per_run():
    """Two runs with the SAME config and seed must still get separate rows.

    Regression: they used to collide into one dict key, so a run was evaluated
    and then silently overwritten -- losing a result, which is the whole thing
    this tracking work exists to prevent.
    """
    from evaluate.plots import row_names

    same_seed = [
        {"run_dir": "logs/b/act-tanh_seed-42_1430", "run_name": "a", "seed": 42},
        {"run_dir": "logs/b/act-tanh_seed-42_1435", "run_name": "b", "seed": 42},
    ]
    names = row_names("activation=tanh", same_seed)
    assert len(set(names.values())) == 2

    diff_seed = [
        {"run_dir": "logs/b/r1", "run_name": "r1", "seed": 42},
        {"run_dir": "logs/b/r2", "run_name": "r2", "seed": 43},
    ]
    names = row_names("activation=relu", diff_seed)
    assert set(names.values()) == {"activation=relu seed=42",
                                   "activation=relu seed=43"}

    solo = [{"run_dir": "logs/b/r1", "run_name": "r1", "seed": 42}]
    assert list(row_names("activation=elu", solo).values()) == ["activation=elu"]


def test_aggregate_curves_handles_mismatched_lengths():
    """Runs with different eval counts still average, without crashing."""
    from evaluate.plots import aggregate_curves

    curves = [
        (np.array([100.0, 200.0, 300.0]), np.array([0.0, 10.0, 20.0])),
        (np.array([100.0, 300.0]), np.array([0.0, 30.0])),
    ]
    grid, mean, std = aggregate_curves(curves)

    assert len(grid) == len(mean) == len(std)
    assert grid[0] == 100.0 and grid[-1] == 300.0
    assert mean[-1] == pytest.approx(25.0)
    assert std[-1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 7. Sweep expansion
# ---------------------------------------------------------------------------

def test_sweep_expands_the_cartesian_product():
    from experiments.sweep import expand

    runs = expand({"agents": ["baseline_dqn", "double_dqn"],
                   "activation": ["relu", "elu"],
                   "seed": [42, 43],
                   "reward_wrapper": ["off"],
                   "timesteps": 1000})

    assert len(runs) == 8
    assert all(r["timesteps"] == 1000 for r in runs)
    assert {r["agent"] for r in runs} == {"baseline_dqn", "double_dqn"}


def test_sweep_requires_reward_wrapper():
    """No default anywhere -- including here."""
    from experiments.sweep import expand

    with pytest.raises(SystemExit, match="reward_wrapper"):
        expand({"agents": ["baseline_dqn"], "activation": ["relu"]})


def test_sweep_rejects_unknown_agent():
    from experiments.sweep import expand

    with pytest.raises(SystemExit, match="Unknown agent"):
        expand({"agents": ["transformer_dqn"], "reward_wrapper": ["off"]})


def test_sweep_skips_completed_runs_only():
    """Resumability: finished runs are skipped, crashed ones are retried."""
    from experiments.sweep import already_done

    combo = {"agent": "baseline_dqn", "activation": "relu",
             "reward_wrapper": "off", "seed": 42, "timesteps": 1000}
    done = {"agent": "baseline_dqn", "activation": "relu",
            "reward_wrapper": "off", "seed": 42, "timesteps": 1000,
            "complete": True}

    assert already_done([done], combo)
    assert not already_done([{**done, "complete": False}], combo)   # crashed
    assert not already_done([{**done, "seed": 43}], combo)          # other seed
    assert not already_done([{**done, "timesteps": 40000}], combo)  # other budget
    assert not already_done([], combo)


def test_sweep_command_includes_required_flags():
    from experiments.sweep import command_for

    cmd = command_for({"agent": "double_dqn", "activation": "mish",
                       "reward_wrapper": "on", "seed": 42, "timesteps": 500})
    joined = " ".join(cmd)

    assert "agents.double_dqn" in joined
    assert "--activation mish" in joined
    assert "--reward-wrapper on" in joined
    assert "--seed 42" in joined
    assert "--timesteps 500" in joined


# ---------------------------------------------------------------------------
# 8. The training scripts' CLI contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", [
    "baseline.train_baseline", "agents.double_dqn", "agents.dueling_dqn",
    "agents.noisy_net"])
def test_reward_wrapper_flag_is_required(module_name):
    """--reward-wrapper has no default on purpose: the three scripts used to
    hardcode different values, so any default would silently change results."""
    import argparse
    import importlib

    from common.run_tracking import add_common_args

    importlib.import_module(module_name)      # must import for the flag to matter
    parser = argparse.ArgumentParser()
    add_common_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([])                 # missing --reward-wrapper

    args = parser.parse_args(["--reward-wrapper", "off"])
    assert args.reward_wrapper == "off"


def test_common_args_default_to_shared_config():
    """Defaults must track config.py, so the sweep and a manual run agree."""
    import argparse

    from common import config
    from common.run_tracking import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--reward-wrapper", "on"])

    assert args.seed == config.SEED
    assert args.timesteps == config.TOTAL_TIMESTEPS
    assert args.eval_freq == config.EVAL_FREQ
    assert args.n_eval_episodes == config.N_EVAL_EPISODES
