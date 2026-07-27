"""
Shared evaluation.

Computes the metrics the prof asked for, the SAME WAY for every agent, so the
baseline and each contribution are directly comparable:

  - mean / std cumulative reward
  - lap completion rate (% of episodes that finished a lap)
  - collision / off-track rate (% of episodes that died off the playfield)
  - mean episode length (frames)
  - mean timesteps to first completed lap

HOW TERMINATION IS CLASSIFIED (verified against the CarRacing-v3 source):
  The environment's info dict is empty mid-episode and carries no
  "lap_finished" flag, so we classify from the termination signals instead:

    - off-playfield DEATH: the env emits a one-time step reward of -100 and
      sets terminated=True. We detect a final raw reward <= -90 with
      terminated (not truncated).
    - LAP COMPLETED: terminated=True without the -100 spike (v2+ changed lap
      completion from truncation to termination).
    - TIME-LIMIT TIMEOUT: TimeLimit.truncated=True -- neither a completion nor
      a collision (the car just ran out of time).

  NOTE: this assumes the eval env is built WITHOUT the reward wrapper (so the
  raw -100 death signal is intact). Evaluate every agent on the same
  unshaped env via make_vec(use_reward_wrapper=False) for fair, comparable
  numbers. (Shaped reward is for *training*; evaluation should use the env's
  native reward.) common.env_factory.make_eval_vec() builds exactly that env.

  FIXED TRACKS: pass seeds=config.EVAL_SEEDS so every model is scored on the
  same 20 tracks -- otherwise each model gets freshly randomised tracks and
  part of the difference between agents is just track luck.
"""

import json
import numpy as np

DEATH_REWARD_THRESHOLD = -90.0   # env emits -100 on leaving the playfield


def evaluate_agent(model, env, n_episodes=20, seeds=None):
    """Roll out a trained model and return a metrics dict.

    `env` should be a vectorized env from common.env_factory.make_eval_vec()
    (n_envs=1, use_reward_wrapper=False). `model` is any SB3 model with
    .predict().

    `seeds`: pass config.EVAL_SEEDS to evaluate on a FIXED set of tracks.
    CarRacing generates a new random track on every reset, so without this two
    agents are scored on different tracks and the comparison is partly track
    luck. With it, episode i always runs on seeds[i] for every model. If there
    are more episodes than seeds the list wraps around.

    Leaving `seeds=None` keeps the original random-track behaviour.
    """
    if seeds:
        # Fixed tracks are the point of the comparison table; say so out loud
        # if the caller asks for fewer episodes than seeds and silently drops some.
        if n_episodes < len(seeds):
            print(f"  (note: {n_episodes} episodes < {len(seeds)} eval seeds -- "
                  f"only the first {n_episodes} tracks will be used)")

    rewards = []
    lengths = []
    completed = []          # 1 if lap completed
    collided = []           # 1 if died off playfield
    first_lap_steps = []    # episode length for completed episodes

    for ep in range(n_episodes):
        if seeds:
            # Seed BEFORE reset: SB3's DummyVecEnv passes the stored seed into
            # the next env.reset(), which is what regenerates the same track.
            env.seed(seeds[ep % len(seeds)])
        obs = env.reset()
        done = False
        last_reward = 0.0
        ep_reward = 0.0
        ep_len = 0
        truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            last_reward = float(reward[0])
            ep_reward += last_reward
            ep_len += 1
            done = bool(dones[0])

        info = infos[0]
        # Prefer the Monitor "episode" record for true reward/length if present.
        if "episode" in info:
            ep_reward = float(info["episode"]["r"])
            ep_len = int(info["episode"]["l"])
        # Was this a time-limit truncation (vs a real termination)?
        truncated = bool(info.get("TimeLimit.truncated", False))

        is_death = (not truncated) and (last_reward <= DEATH_REWARD_THRESHOLD)
        is_lap = (not truncated) and (not is_death)

        rewards.append(ep_reward)
        lengths.append(ep_len)
        collided.append(1 if is_death else 0)
        completed.append(1 if is_lap else 0)
        if is_lap:
            first_lap_steps.append(ep_len)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "completion_rate": float(np.mean(completed)),
        "collision_rate": float(np.mean(collided)),
        "mean_episode_len": float(np.mean(lengths)),
        "mean_steps_to_lap": (
            float(np.mean(first_lap_steps)) if first_lap_steps else None
        ),
        "n_episodes": n_episodes,
        # Recorded so a results.json makes clear whether the numbers came from
        # the fixed track set or from random tracks.
        "eval_seeds": list(seeds) if seeds else None,
    }


def print_metrics(name, metrics):
    print(f"\n=== {name} ===")
    print(f"  mean reward      : {metrics['mean_reward']:.1f} "
          f"+/- {metrics['std_reward']:.1f}")
    print(f"  completion rate  : {metrics['completion_rate'] * 100:.0f}%")
    print(f"  collision rate   : {metrics['collision_rate'] * 100:.0f}%")
    print(f"  mean episode len : {metrics['mean_episode_len']:.0f} frames")
    stl = metrics["mean_steps_to_lap"]
    print(f"  steps to lap     : {stl:.0f}" if stl is not None
          else "  steps to lap     : (no laps completed)")


def save_metrics(path, name, metrics):
    with open(path, "w") as f:
        json.dump({"agent": name, **metrics}, f, indent=2)
