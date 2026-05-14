# Current Source Of Truth

Date: 2026-05-13

## What We Are Doing

We are starting a refactor phase for the CurvyTron stock LightZero training
lane. The goal is not to add new training features. The goal is to make the
current working path easier to trust, test, and maintain.

Scope clarification from Coach: this lane is about training code and trainer
scaffolding. Do not wander into environment redesign. The environment matters
only as a contract the trainer consumes.

## Trusted Lane

The trusted learning lane is:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
```

That path should call stock `lzero.entry.train_muzero`. The old custom
`--mode two-seat-selfplay` lane is historical and should not guide learning
claims.

## Immediate Bug To Lock Down

DI-engine can create timestamped LightZero experiment directories when
`compile_config` sees an existing configured experiment path. Then checkpoints
may land under:

```text
train/lightzero_exp_YYMMDD_HHMMSS/ckpt
```

while CurvyZero status, poller, resume, eval, GIF, or manifest code may still
look only at:

```text
train/lightzero_exp/ckpt
```

This can make a healthy run look stale. The fix should be small, but it must be
covered before the refactor.

Current status: fixed in the trainer/status scaffolding and covered by focused
regression tests. The pure path/parsing/candidate contract now lives in:

```text
src/curvyzero/training/lightzero_checkpoints.py
```

The large Modal trainer keeps thin wrappers for compatibility, but checkpoint
exp-dir discovery, LightZero iteration-name parsing, checkpoint candidate
collection, and latest checkpoint selection should be treated as the helper
module's responsibility.

Kant's follow-up audit found one additional real bug: resume-sidecar saving
also used the fixed `lightzero_exp/ckpt` path. That is now covered and fixed:
when the matching checkpoint lives in `lightzero_exp_*`, the sidecar is saved
beside that checkpoint and mirrored to the run checkpoint area.

Focused validation after the first extraction:

```text
uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py -q
76 passed, 1 skipped

uv run ruff check tests/test_lightzero_timestamped_checkpoint_discovery.py src/curvyzero/training/lightzero_checkpoints.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py src/curvyzero/infra/modal/lightzero_curvytron_run_status.py
All checks passed
```

The focused test file now also covers:

- helper candidate collection and deterministic latest selection;
- optional empty checkpoint filtering;
- timestamped LightZero resume-sidecar saving.

Opponent ref immutability is also fixed for the top-level frozen-opponent
entry point. Both top-level frozen refs and mixture frozen refs should now use
exact immutable `iteration_N.pth.tar` refs, never `latest` or `ckpt_best`.

First opponent-assignment parser slice is implemented in:

```text
src/curvyzero/training/opponent_registry.py
```

It parses a frozen assignment snapshot into the existing opponent-mixture
contract. It is intentionally pure: no Modal Dict, no volume reload, no
tournament ranking, no checkpoint loading. Explicit assignment refs are now
wired into the trainer and background poller as static launch inputs.

End-to-end smoke status: a tiny Modal CPU `--mode train` smoke has now run
successfully after the refactor. It called stock `train_muzero`, wrote
iteration checkpoints, wrote resume sidecars, wrote `progress_latest.json`, and
the run-status reader found `latest_checkpoint=iteration_2`.

Artifact smoke status: a tiny Modal CPU `--mode train` smoke with background
eval/GIF enabled also completed. The background poller found three checkpoints,
ran three inspection jobs, ran three GIF jobs, and wrote both website GIF
variants for each checkpoint: greedy `raw.gif` and sampled `collect_t1.gif`.

This is not a learning claim and it did not prove GPU training or an exact
resume-equivalence story. Those are still separate validation gates.

Fresh `--mode train` is still stock LightZero in the training-loop ownership
sense: LightZero owns collector, search, replay, learner, and stock checkpoint
creation. CurvyZero supplies config, env, progress/status/resume scaffolding,
and background artifact workers. The detailed discrepancy table now lives in:

```text
docs/working/training/lightzero_train_refactor_2026-05-13/stock_lightzero_parity_audit.md
```

Important caveat: resumed runs are not exact stock fresh runs. The sidecar path
is operational continuity and should not claim exact uninterrupted replay
equivalence.

Latest focused local validation:

```text
uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py -q
102 passed, 1 skipped
```

The latest added guard proves fresh resume hooks pass through the original
LightZero `call_hook`, `eval`, and `random_collect` return values when resume
is inactive.

The latest small source extraction moved exact LightZero resume-state candidate
collection/selection into `src/curvyzero/training/lightzero_checkpoints.py`.
The trainer still owns run/attempt path construction and the public
`resume_state_lookup` payload shape.

Auto-resume checkpoint scanning now also uses the shared checkpoint candidate
helper instead of hand-parsing checkpoint filenames inside the trainer. The
trainer still owns source-root metadata, Modal refs, and the final
auto-resume payload.

Checkpoint progress payload construction is now split from the JSON write and
covered directly. The hook wrappers still only call the original LightZero save
path first, then write CurvyZero observability state.

Opponent assignment snapshots now require exact
`curvyzero_opponent_assignment/v0` schema ids and have a canonical SHA-256
helper. Explicit assignment-ref wiring now exists in the trainer and background
checkpoint eval/GIF poller: an immutable `assignment.json` is resolved into the
existing opponent-mixture contract before LightZero starts. This is still a
static trainer input, not live tournament control: no Modal Dict reads, no
tournament ranking, and no trainer polling inside `train_muzero`.

Leaderboard-to-training current truth lives in:

```text
docs/working/training/leaderboard_to_training_2026-05-13/HANDOFF.md
```

Recent focused fixes after critique:

- the real checkpoint eval poller function now accepts and forwards
  `opponent_assignment_ref`;
- the assignment artifact writer has direct regression coverage for writing
  assignment/audit files under an attempt and committing the Volume;
- the pure leaderboard selector sorts eligible rows deterministically before
  choosing the champion and excludes retired rows even when provisional rows
  are allowed.

New immediate requirement from Coach: the trusted stock LightZero lane should
support one policy action per granular CurvyTron game step. Any action repeat
or multi-tick decision hold must be explicit. The active plan is in:

```text
docs/working/training/lightzero_train_refactor_2026-05-13/granular_action_cadence_plan_2026-05-13.md
```

Current action-cadence status: fixed for the trusted
`source_state_fixed_opponent` stock train lane. The default is one source
physics frame per policy action. Trusted `--mode train` and `--mode dry` reject
stale multi-frame `decision_ms` values; use `policy_action_repeat_*` if a run
intentionally wants repeated actions. The active survivaldiag and opponent
mixture manifest builders now emit the one-frame timing value.

Fresh post-cadence smoke status: after adding a final train-mode Volume commit,
the waited CPU smoke
`curvytron-cadence-e2e-smoke-20260513-203914/train-smoke-001` completed with
`ok=true`, called stock `train_muzero`, wrote telemetry, wrote final summary
artifacts, and mirrored 18 LightZero checkpoints. The downloaded summary showed
`decision_source_frames=1`, `decision_ms=16.666666666666668`, and
`policy_action_repeat_min=max=1`.

Background eval/GIF cadence pass-through is also patched locally: the remote
eval/GIF APIs, poller path, and eval env builder now receive cadence fields
explicitly instead of relying on imported defaults. Focused tests with sentinel
cadence values passed.

Latest action-cadence validation:

```text
uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_survivaldiag_manifest.py tests/test_curvytron_opponent_mixture_manifest.py -q
112 passed, 1 skipped

uv run pytest tests/test_lightzero_timestamped_checkpoint_discovery.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_run_status.py tests/test_opponent_mixture.py tests/test_opponent_registry.py tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_curvytron_survivaldiag_manifest.py tests/test_curvytron_opponent_mixture_manifest.py -q
161 passed, 1 skipped

uv run ruff check src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py scripts/build_curvytron_survivaldiag_manifest.py scripts/build_curvytron_opponent_mixture_manifest.py tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_survivaldiag_manifest.py tests/test_curvytron_opponent_mixture_manifest.py
All checks passed
```

## Current Refactor Stance

- Keep the environment as an environment: reset, step, observation, reward,
  done, info. Do not hide trainer or Modal behavior in it.
- Keep LightZero ownership clear: collector, replay, learner, target semantics,
  and search stay in stock LightZero wherever possible.
- Move CurvyZero scaffolding toward small modules with plain contracts:
  checkpoint discovery, resume selection, progress/status writing,
  background eval/GIF scheduling, manifest support, and Modal wrappers.
- Do not change reward, opponent semantics, or matrix settings in this lane
  unless a regression test proves the current behavior is wrong.
- Treat action cadence as part of the trainer/env contract: if one LightZero
  env step advances more than one granular source game step, that must be
  explicit and tested.

Primary files for this lane are the training launcher, trainer support tests,
and small helper modules/scripts it already uses. Environment modules are only
read to understand the interface expected by the trainer.

New focused regression file:

```text
tests/test_lightzero_timestamped_checkpoint_discovery.py
```

## Non-Goals For This First Pass

- Launching more training runs.
- Optimizing render speed.
- Redesigning reward.
- Redesigning opponent mixtures.
- Rewriting the trainer.
- Moving large blocks of code before tests describe their current contract.
