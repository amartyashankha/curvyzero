# JS Reuse Probe - 2026-05-09

## Question

Can CurvyZero reuse the original CurvyTron JavaScript as the ground-truth
environment instead of continuing to port every source behavior into Python?

## Prototype

Added `tools/js_reuse_probe/curvytron_env_cli.js`, a batch CLI that:

- loads 25 original files from `third_party/curvytron-reference` into a Node
  `vm` context;
- replaces `Math.random` with a deterministic constant or tape from
  `source_setup.random`;
- replaces timers with a manually advanced queue and a controlled `Date`;
- creates an original `Game`;
- performs a reset-like call:
  - `forced_state` for existing state-step fixtures;
  - `source_new_round` when a lifecycle fixture asks the source to spawn players;
- performs step-like calls by applying each move through
  `avatar.updateAngularVelocity(move)` and then calling `game.update(stepMs)`;
- emits compact JSON with `reset`, `steps`, `reward`, `roundDone`, `gameDone`,
  source events, and `randomCalls`.

Added `tools/js_reuse_probe/curvytron_env_worker.js`, a persistent newline-JSON
worker that:

- loads the same 25 original files once at process startup;
- exposes one live env through `reset`, `step`, `snapshot`, and `close`
  commands;
- keeps deterministic random, controlled timers, score deltas, events, and
  compact source snapshots across step calls;
- reports `sourceLoadCount` so tests can prove separate step commands are not
  reloading the original JS VM.

Added `src/curvyzero/fidelity/js_reuse_probe.py` as the reusable Python
subprocess boundary:

```python
from curvyzero.fidelity.js_reuse_probe import (
    CurvytronJsEnvWorker,
    run_js_reuse_env_probe,
)

payload = run_js_reuse_env_probe(
    "scenarios/environment/source_kinematics_turn_multistep.json"
)

with CurvytronJsEnvWorker() as env:
    reset = env.reset("scenarios/environment/source_kinematics_turn_multistep.json")
    frame = env.step({"moves": [0, 0], "step_ms": 1000 / 60})
```

## What Works

- The original CurvyTron JS can be loaded once in Node and driven with
  deterministic reset/step-like calls.
- Existing source state-step scenarios work without rewriting their mechanics in
  Python. The `source_kinematics_turn_multistep` fixture returns the expected
  final positions `[[21.063765, 39.925415], [58.935365, 39.937827]]` and
  headings `[-0.046667, 3.188259]`.
- Multi-step source behavior with randomness also works in the same shape. The
  `source_print_manager_random_cadence_multistep` fixture consumes the taped
  random values `[0, 0.5, 0.25]` and exposes the original print-manager events.
- Natural source reset is partially proven: running the lifecycle spawn fixture
  with `source_new_round` calls the original `game.newRound(...)`, produces the
  source spawn positions/headings, and records the spawn random calls.
- Persistent stepping now works for the env-shaped movement fixture. The worker
  resets once, accepts multiple independent `step` commands, and returns the same
  frames as the batch CLI while `sourceLoadCount` remains `1`.
- A longer 1v1/no-bonus source rollout works through the persistent worker. The
  fixture `source_lifecycle_long_1v1_no_bonus_wall_round_done.json` performs one
  deterministic `source_new_round` reset and 111 separate `step` calls while
  `sourceLoadCount` stays `1`. It reaches source `round:end`: avatar 2 dies on
  the wall, avatar 1 wins, scores are `[1, 0]`, and `roundDone` is true.

## What Does Not Work Yet

- Existing lifecycle fixtures are not interpreted as step scripts. The CLI uses
  lifecycle scenarios only to prove source `newRound` reset behavior.
- There is no Python in-process JS bridge, vectorization, or production-speed
  story here. The useful result is the API boundary, not throughput.
- The JSON state is compact but not yet the final trainer observation contract.
  It includes enough state for later env stepping work: player kinematics,
  alive/present/printing, scores, trail/body counters, world flags, events, and
  simple score-delta rewards.
- The worker owns one env per Node process. It is not yet a vectorized backend,
  a Gymnasium-compatible API, or a lifecycle-complete training environment.
- The long 1v1 proof stops at round terminal state. It does not drive the
  source warmdown timer to `game:stop`/`gameDone`.

## Plain-Language Answer

Yes, we can reuse their JavaScript and make it work as a source-faithful scalar
environment backend/oracle. The persistent worker proves the important missing
piece: Python can keep one original CurvyTron VM alive, reset it, send separate
step commands, and receive deterministic snapshots without paying source reload
cost on every step.

That does not make it the final fast batched training environment by itself. It
is still one env per Node worker, crossing a subprocess JSON boundary, with no
vectorization and no final trainer observation contract. The short-term use is
good: use it as the source env backend for fidelity checks, scalar prototypes,
and oracle-driven development. For high-throughput self-play, keep the native
batched Python/vector path as the likely training backend unless measurements
show a pool of Node workers is acceptable.

## Needed Refactor For Production Use

The next refactor, if we promote this, should keep the worker narrow and harden
it into an adapter:

- keep deterministic random/timer state inside the worker between calls;
- define a stable observation/reward/done adapter on top of the current compact
  snapshot;
- add lifecycle command support for `advance_timers`, `set_avatar_state`, and
  round start/stop transitions instead of treating lifecycle fixtures as oracle
  dumps.

This probe answers the narrower question: yes, the original JS can be loaded
once and driven deterministically with reset/step-like calls from Python via a
persistent subprocess. Production training still needs a measured throughput
decision and a final observation/action contract.

## Verification

Commands run:

```bash
node tools/js_reuse_probe/curvytron_env_cli.js scenarios/environment/source_kinematics_turn_multistep.json
node tools/js_reuse_probe/curvytron_env_cli.js scenarios/environment/source_print_manager_random_cadence_multistep.json
node tools/js_reuse_probe/curvytron_env_cli.js scenarios/environment/source_lifecycle_spawn_rng_warmup_print_start_2p.json
uv run pytest tests/test_js_reuse_probe.py
```

Focused test:

- `tests/test_js_reuse_probe.py` asserts deterministic repeated runs, reset
  state, four step moves, final source positions/headings, reward/done fields,
  source root metadata, and source `newRound` reset spawn/random behavior.
- The worker test asserts `reset` plus four separate `step` commands match the
  batch CLI frames and that `sourceLoadCount` stays at `1` through reset, steps,
  and snapshot.
- The long-rollout test asserts one source reset plus 111 separate worker steps
  reaches the pinned 1v1/no-bonus wall-terminal outcome without reloading the
  source VM.

Latest focused run:

```bash
uv run pytest tests/test_js_reuse_probe.py
# 4 passed
```
