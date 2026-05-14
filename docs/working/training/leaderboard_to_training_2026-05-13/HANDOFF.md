# Handoff: Leaderboard-To-Training Loop

## Read This First

Start here:

1. `README.md`
2. `current_state.md`
3. `implementation_log.md`
4. `closed_loop_spec.md`
5. `gaps_and_tests.md`

## Plain Status

A tiny manual closed-loop smoke works.

What was proven:

1. A leaderboard rating can be published into a public leaderboard snapshot.
2. A public leaderboard snapshot can be converted into an assignment.
3. An assignment can be written to the training Volume.
4. A trainer can consume that assignment and run.
5. That trainer writes checkpoints.
6. Tournament discovery can see those checkpoints.
7. Intake can enqueue and drain those checkpoints.
8. A tiny rating run can complete.
9. The new rating can be published into a new leaderboard.
10. A new assignment can be selected from that leaderboard.
11. A second trainer smoke can consume that new assignment and run.

This is not production automation yet.

## Coach / Inspector Boundary

Coach owns training:

- launch/runtime config;
- exact checkpoint writing;
- immutable assignment consumption;
- eval/GIF using the same assignment as training;
- deciding whether leaderboard evidence is good enough to steer training.

Inspector owns tournament observability:

- checkpoint discovery from the training Volume;
- subscriber/intake Dict and Queue coordination;
- tournament games, GIF samples, Elo/rating artifacts, pair history, and
  scheduler state;
- public leaderboard snapshot publishing.

The clean exchange is:

```text
Coach -> Inspector: exact checkpoint refs on the training Volume
Inspector -> Coach: public leaderboard snapshot, then immutable assignment.json
```

The trainer must not poll live tournament state or Modal Dict while learning.
Modal Dict/Queue are coordination helpers. Volume JSON snapshots are the
durable truth.

Subscriber watch rule:

- run IDs or run prefixes are live watches;
- explicit checkpoint refs are frozen seeds;
- continuation into an existing rating run must be explicit
  (`continue_from_latest=True`) and must keep the full known checkpoint pool.

## Important Smoke Runs

Assignment write:

```text
leaderboard-assignment-smoke / assignment-smoke
leaderboard-assignment-smoke-b / assignment-smoke-b
```

Training smokes:

```text
leaderboard-assignment-train-smoke-20260513a
leaderboard-assignment-train-smoke-20260513c
```

Tiny tournament:

```text
arena-closed-loop-smoke-20260513b / elo-closed-loop-smoke
```

Smoke leaderboard:

```text
curvytron-closed-loop-smoke-20260513b
```

## Tests That Passed

```text
uv run pytest \
  tests/test_lightzero_checkpoint_opponent_provider.py \
  tests/test_opponent_leaderboard.py \
  tests/test_opponent_registry.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_artifact_writer_stores_assignment_and_audit \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_ref_resolves_to_existing_mixture_contract \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_resolves_assignment_ref \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_function_accepts_assignment_ref_and_resolves_command \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_rejects_assignment_and_inline_mixture \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_local_launcher_passes_gif_config_to_poller_and_prints_enabled \
  tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer -q
```

Result:

```text
27 passed
```

Lint/compile:

```text
ruff: all checks passed
py_compile: passed
```

## Code Added Or Changed

New:

- `src/curvyzero/training/opponent_leaderboard.py`
- `tests/test_opponent_leaderboard.py`
- `tests/test_lightzero_checkpoint_opponent_provider.py`
- `scripts/materialize_curvytron_leaderboard_assignment.py`

Changed:

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py`
- `tests/test_curvytron_checkpoint_tournament.py`
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`

## Important Bug Found And Fixed

The trainer failed when a leaderboard-selected checkpoint had a different
LightZero support head shape than the current trainer default.

Fix:

```text
lightzero_checkpoint_opponent_provider.py
```

now infers reward/value support sizes from checkpoint state dict head shapes
before building the MuZero model.

## What Still Needs Work

Do next:

1. Add a repair command for missing/stale leaderboard Dict pointers.
2. Promote the assignment artifact writer/operator flow from smoke path to a
   documented production runbook.
3. Add safe refresh policy for long-running trainers.
4. Harden intake continuation from existing `latest.json`.
5. Run a larger but still bounded closed-loop smoke.
6. Validate one-frame tournament context for the new public leaderboard.
7. Decide whether non-neural scripted policies stay assignment-only or become
   tournament participants later.

Recent focused fixes after critique:

- the real checkpoint eval/GIF poller now accepts and forwards
  `opponent_assignment_ref`;
- the assignment artifact writer has direct regression coverage;
- the pure assignment selector excludes retired rows when provisional rows are
  allowed, and sorts rows before choosing the champion.
- the trainer-facing leaderboard publisher refuses provisional ratings unless
  explicitly allowed, commits the Volume before moving the live pointer, and
  leaves the pointer unchanged if commit fails.

## Do Not Overstate

Do not say production closed-loop training is done.

Say this:

```text
A tiny manual closed-loop smoke works. Production automation still needs refresh,
continuation, repair, and scale tests.
```
