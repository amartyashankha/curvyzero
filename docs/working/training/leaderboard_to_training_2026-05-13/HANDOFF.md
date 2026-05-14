# Handoff: Leaderboard-To-Training Loop

## Read This First

Start here:

1. `README.md`
2. `current_state.md`
3. `implementation_log.md`
4. `closed_loop_spec.md`
5. `gaps_and_tests.md`

## Plain Status

A tiny manual closed-loop smoke works, including the `stable_slots_v1`
materializer path. The first stable-slot smoke also found a real bug: checkpoint
recency metadata was not preserved in rating rows, so `recent_strong` could pick
an older high-ranked checkpoint. Local code/tests now preserve that metadata;
rerun the remote smoke before trusting automatic refresh.

Latest local rule:

- The tournament service defaults to a top-100 active pool.
- Mature rows below rank 100 are marked `retired`, not deleted.
- Retired rows stay in rating history and public leaderboard snapshots, but are
  not scheduled for future games.
- New or under-tested rows stay `provisional` even if their temporary rank is
  below 100, so they can still receive placement games.
- Pointer summaries count retired rows separately from provisional rows.

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

## Coach / Tournament Job Boundary

Coach owns training:

- launch/runtime config;
- exact checkpoint writing;
- immutable `assignment.json` consumption;
- eval/GIF using the same assignment as training;
- deciding whether leaderboard evidence is good enough to steer training.

The long-running tournament job owns tournament observability:

- checkpoint discovery from the training Volume;
- subscriber/intake Dict and Queue coordination;
- tournament games, GIF samples, Elo/rating artifacts, pair history, and
  scheduler state;
- public leaderboard snapshot publishing.

The clean exchange is:

```text
Coach -> tournament job: exact checkpoint refs on the training Volume
tournament job -> Coach: public leaderboard snapshot, then immutable assignment.json
```

The trainer must not poll live tournament state or Modal Dict while learning.
Modal Dict/Queue are coordination helpers. Volume JSON snapshots are the
durable truth.

Subscriber watch rule:

- run IDs or run prefixes are live watches;
- explicit checkpoint refs are frozen seeds;
- continuation into an existing rating run must be explicit
  (`continue_from_latest=True`) and must keep the full known checkpoint pool.

Tournament service rule:

- `intake-seed` is the admin/configure path for scheduler/evaluator policy.
- `tournament-submit` / `intake-submit` is the normal candidate path.
- Submit accepts candidate checkpoint refs or run IDs, not scheduler knobs.
- Drain uses the manifest policy by default. Ad-hoc rating overrides are an
  explicit internal/operator escape hatch, not the product contract.

Checkpoint discovery foot gun:

- discovery must scan `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
- DI-engine can create timestamped folders like `lightzero_exp_260513_123802`;
- scanning only `train/lightzero_exp/ckpt/...` misses real checkpoints.

Publication boundary:

- The tournament job owns public leaderboard snapshots and the compact live pointer;
- Coach owns assignment selection, assignment refresh timing, and launch policy;
- Modal Dict may point at the latest public snapshot, but the durable handoff is
  the Volume JSON snapshot plus a selected immutable `assignment.json`.

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

Latest focused local verification:

```text
uv run pytest tests/test_curvytron_checkpoint_tournament.py -q
139 passed, 12 skipped

uv run pytest \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_opponent_leaderboard.py \
  tests/test_lightzero_checkpoint_opponent_provider.py -q
70 passed, 1 skipped

uv run ruff check \
  src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  src/curvyzero/training/opponent_leaderboard.py \
  src/curvyzero/training/lightzero_checkpoint_opponent_provider.py \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_opponent_leaderboard.py \
  tests/test_lightzero_checkpoint_opponent_provider.py
All checks passed

git diff --check
clean
```

Latest extra local coverage:

- `tests/test_curvytron_checkpoint_intake_repair.py`
- `tests/test_curvytron_tournament_scheduler_fairness.py`
- `tests/test_curvytron_opponent_leaderboard_pointer_repair.py`
- no-active-row leaderboard publish guard in
  `tests/test_curvytron_checkpoint_tournament.py`
- one-frame leaderboard publish guard in
  `tests/test_curvytron_checkpoint_tournament.py`

Combined focused tournament check:

```text
uv run pytest \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_intake_repair.py \
  tests/test_curvytron_tournament_scheduler_fairness.py \
  tests/test_curvytron_opponent_leaderboard_pointer_repair.py -q
151 passed, 12 skipped
```

Earlier focused smoke regression:

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

- `src/curvyzero/tournament/checkpoint_intake_service.py`
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

1. Remote-smoke the repair command for missing/stale leaderboard Dict pointers.
2. Rerun the `stable_slots_v1` closed-loop smoke after the recency metadata
   repair and inspect the concrete assignment entries.
3. Promote the `stable_slots_v1` assignment writer/operator flow from local
   helper/smoke path to a documented production runbook.
4. Remote-smoke one-frame public publish and continuation with the active-pool
   rule enabled.
5. Add safe refresh policy for long-running trainers.
6. Run a larger but still bounded closed-loop smoke.
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
- checkpoint intake now has regression coverage for live run-id watches versus
  frozen explicit checkpoint refs.
- checkpoint intake drain now has regression coverage that `spawn_if_existing`
  does not touch an existing rating run unless `continue_from_latest=True`.
- continuation specs use the full `seen_checkpoint_refs` pool, so adding a new
  checkpoint does not silently drop older rated checkpoints.
- `stable_slots_v1` is now the production-direction materializer. It writes
  ordinary assignment entries, uses nested `recency.latest_for_run`, supports
  blank or wall-avoidant immortal sentinel entries, dedupes checkpoint id/ref,
  and records per-slot evidence in the audit.

## Do Not Overstate

Do not say production closed-loop training is done.

Say this:

```text
A tiny manual closed-loop smoke works. Local recency metadata is repaired, but
the remote stable-slot smoke must be rerun before automatic refresh. Production
automation still needs refresh, continuation, repair, and scale tests.
```
