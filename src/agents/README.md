# Agents — ownership & conventions

Each agent is one self-contained file that trains a variant and saves to its
own `logs/<agent_name>/` folder, so nobody steps on anyone else's runs.

| File             | Owners        | Milestone | Status      |
|------------------|---------------|-----------|-------------|
| `double_dqn.py`  | Ekam, Lex     | M1        | Stub — TODO |
| `dueling_dqn.py` | Hargun, Evan  | M1        | Stub — TODO |

## Rules to keep comparisons fair

1. **Import hyperparameters from `common/config.py`.** Don't hardcode them.
   If your contribution needs a different value, override it explicitly in
   your file and write a one-line comment saying why (and note it in the
   writeup).
2. **Build envs via `common/env_factory.py`.** Same preprocessing for everyone.
3. **Keep `TOTAL_TIMESTEPS` and `SEED` identical** to the baseline unless the
   group agrees otherwise. "Same training conditions" is a grading criterion.
4. **Save to your own `logs/<name>/` dir** (the scripts already do this).
5. **One agent = one file.** If you need shared helpers, put them in `common/`.

## How each variant differs (quick reference)

- **Baseline DQN** — vanilla SB3, the reference to beat.
- **Double DQN** — changes only the *target computation* (online net selects
  the action, target net evaluates it). Architecture unchanged.
- **Dueling DQN** — changes the *network architecture* (value + advantage
  streams). Target computation unchanged.

These two are deliberately orthogonal, which makes for a clean writeup: one
changes the math, one changes the model. A natural M2 extension is combining
them (Dueling Double DQN).
