# Implementation Log

## 2026-05-14 stable_slots_v1 Slice

Implemented:

- `select_stable_slots_v1_assignment` in
  `src/curvyzero/training/opponent_leaderboard.py`
  - materializes one verified leaderboard snapshot into ordinary assignment
    entries;
  - supports `stable_3` and `stable_5` profiles;
  - supports `none`, `blank_canvas`, and `wall_avoidant_immortal` sentinels;
  - supports normal or immortal death mode for leaderboard checkpoint entries;
  - uses nested `recency.latest_for_run` for `recent_strong`;
  - dedupes checkpoint slots by checkpoint id and checkpoint ref;
  - records source snapshot hash, profile, context gate, per-slot evidence, and
    selection reasons in `audit.json`.
- Removed `slot_rules_v0` from the production helper direction. The trainer
  still consumes only immutable `assignment.json`.
- `scripts/materialize_curvytron_leaderboard_assignment.py`
  - defaults to `stable_slots_v1`;
  - accepts an already-published leaderboard snapshot directly and verifies its
    leaderboard id, snapshot id, and canonical hash;
  - keeps `top_slots_v0` as an explicit smoke/legacy option.
- Docs now describe Volume as truth, Dict as cache, Queue/subscriber as wakeups,
  and `stable_slots_v1` as the Coach-owned materializer.

Checks:

```text
uv run pytest tests/test_opponent_leaderboard.py -q
19 passed

uv run pytest tests/test_opponent_leaderboard.py tests/test_opponent_registry.py \
  tests/test_materialize_curvytron_leaderboard_assignment.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_ref_resolves_to_existing_mixture_contract \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_artifact_writer_stores_assignment_and_audit \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_resolves_assignment_ref \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_rejects_assignment_and_inline_mixture -q
36 passed

uv run ruff check src/curvyzero/training/opponent_leaderboard.py \
  scripts/materialize_curvytron_leaderboard_assignment.py \
  tests/test_opponent_leaderboard.py \
  tests/test_materialize_curvytron_leaderboard_assignment.py
All checks passed

git diff --check -- src/curvyzero/training/opponent_leaderboard.py \
  scripts/materialize_curvytron_leaderboard_assignment.py \
  tests/test_opponent_leaderboard.py \
  docs/working/training/leaderboard_to_training_2026-05-13
clean
```

CLI smoke:

```text
scripts/materialize_curvytron_leaderboard_assignment.py
```

materialized a `stable_5` assignment with champion, recent strong, diverse
challenger, anchor, and wall-avoidant immortal sentinel under
`/private/tmp/stable_slots_materialized/`.

## 2026-05-14 Stable-Slot Recency Repair

Found during reorientation:

- The manual `stable_slots_v1` closed-loop smoke worked mechanically, but the
  published leaderboard rows did not carry enough checkpoint metadata.
- Without `run_id`, `attempt_id`, `iteration`, and `latest_for_run`, the
  `recent_strong` slot can silently fall back to "best remaining checkpoint."
  In the smoke, that meant an older checkpoint could be selected as
  `recent_strong`.

Implemented locally:

- checkpoint refs now produce `run_id`, `attempt_id`, and `iteration` metadata;
- normalized checkpoint lists mark exactly the latest iteration per run as
  `latest_for_run`;
- rating rows preserve checkpoint recency metadata so public leaderboard rows
  can expose it to `stable_slots_v1`.
- intake manifests that contain only checkpoint refs, with no discovery `rows`,
  still build rating specs correctly; metadata rows are optional enrichment, not
  a required field.

Focused check:

```text
uv run pytest \
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_spec_extracts_assignment_slot_metadata_from_ref \
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_specs_mark_only_latest_iteration_per_run \
  tests/test_opponent_leaderboard.py::test_stable_slots_v1_uses_nested_recency_latest_for_run -q
3 passed

uv run pytest tests/test_curvytron_checkpoint_tournament.py \
  tests/test_opponent_leaderboard.py \
  tests/test_materialize_curvytron_leaderboard_assignment.py -q
170 passed, 12 skipped

uv run ruff check src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  src/curvyzero/tournament/curvytron_checkpoint_tournament.py \
  src/curvyzero/training/opponent_leaderboard.py \
  scripts/materialize_curvytron_leaderboard_assignment.py \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_opponent_leaderboard.py \
  tests/test_materialize_curvytron_leaderboard_assignment.py
All checks passed
```

Next proof:

- rerun the remote closed-loop smoke and inspect `assignment.json`; the
  `recent_strong` entry should point at the latest useful checkpoint for the
  watched run, not just the highest-rated remaining row.

## 2026-05-13 Pure Bridge Slice

Implemented:

- `src/curvyzero/training/opponent_leaderboard.py`
  - builds public leaderboard snapshots from tournament rating snapshots;
  - validates leaderboard snapshots;
  - builds/validates live pointer payloads;
  - selects top-slot opponent assignments;
  - writes assignment audit payloads;
  - keeps assignment JSON compatible with `parse_opponent_assignment_snapshot`.
- `tests/test_opponent_leaderboard.py`
  - snapshot active/provisional validation;
  - immutable checkpoint ref rejection;
  - pointer compact summary;
  - assignment selector output;
  - deterministic selection.
- `scripts/materialize_curvytron_leaderboard_assignment.py`
  - local-only CLI that consumes exported rating JSON/API payloads;
  - emits `leaderboard_snapshot.json`, `leaderboard_pointer.json`,
    `assignment.json`, and `audit.json`.
- Trainer-side resolver helper:
  - `_resolve_opponent_assignment_for_env` in
    `lightzero_curvyzero_stacked_debug_visual_survival_train.py`;
  - reads assignment ref/path;
  - validates assignment;
  - resolves the assignment's mixture through the existing mixture resolver.

Smoke artifact:

```text
artifacts/local/curvytron_status_snapshots/analysis_20260513e/leaderboard_latest212/materialized_assignment_smoke/
```

Checks:

```text
uv run pytest tests/test_opponent_leaderboard.py tests/test_opponent_registry.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_ref_resolves_to_existing_mixture_contract -q

17 passed

uv run ruff check src/curvyzero/training/opponent_leaderboard.py \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  tests/test_opponent_leaderboard.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  scripts/materialize_curvytron_leaderboard_assignment.py

All checks passed
```

## 2026-05-13 Publisher And Trainer Plumbing Slice

Implemented:

- `curvytron_opponent_leaderboard_publish`
  - reads tournament rating `latest.json`;
  - builds the public leaderboard snapshot;
  - writes snapshot and leaderboard `latest.json` on the tournament Volume;
  - writes a compact Modal Dict pointer.
- Trainer assignment-ref plumbing
  - `_run_visual_survival_train` accepts `opponent_assignment_ref`;
  - assignment refs are mutually exclusive with inline `opponent_mixture_spec`;
  - command metadata records assignment id, source, ref, sha256, and file summary;
  - checkpoint eval poller command resolves the same assignment ref so eval/GIF
    sees the same mixture.

New tests:

```text
tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer
tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_ref_resolves_to_existing_mixture_contract
tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_resolves_assignment_ref
tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_rejects_assignment_and_inline_mixture
```

Checks:

```text
20 passed
ruff: all checks passed
```

## 2026-05-13 Closed-Loop Smoke

Smoke path:

1. Published latest-212 rating into smoke public leaderboard:
   `curvytron-latest212-smoke-20260513`.
2. Materialized assignment and audit locally.
3. Wrote assignment/audit to `curvyzero-runs` via
   `mode=write-assignment`.
4. Ran a tiny train smoke consuming the assignment:
   `leaderboard-assignment-train-smoke-20260513a`.
5. Discovered 17 checkpoints from that training run via tournament broad
   checkpoint discovery.
6. Seeded intake with the new smoke checkpoint plus an existing champion
   checkpoint.
7. Drained intake and spawned a tiny one-frame rating run:
   `arena-closed-loop-smoke-20260513b / elo-closed-loop-smoke`.
8. Rating completed: `1` pair, `3` games, `ratings_written=true`.
9. Published that rating into smoke public leaderboard:
   `curvytron-closed-loop-smoke-20260513b`.
10. Fetched the public leaderboard, selected a fresh 3-slot assignment, wrote it
    to `curvyzero-runs`, and ran a second trainer smoke:
    `leaderboard-assignment-train-smoke-20260513c`.

Important blocker found and fixed:

- The first second-generation assignment smoke failed because a selected
  checkpoint used a different LightZero model support shape than the trainer's
  current default opponent-provider reconstruction.
- Fix: `lightzero_checkpoint_opponent_provider.py` now infers support head sizes
  from checkpoint state dict shapes before constructing the MuZero model.

Final train smoke evidence:

```text
run_id=leaderboard-assignment-train-smoke-20260513c
ok=true
assignment_id=closed-loop-smoke-assignment-b
mixture_entries=slot_champion, slot_recent_strong, slot_sentinel_blank_canvas
```

Remaining caveat:

- This proves a tiny manual closed-loop smoke, not the full automated periodic
  production loop. Online continuation, scheduled refresh, pointer repair, and
  assignment refresh policy remain to be proven remotely and scaled. Local
  pointer repair landed later in this file.

## Still Not Done

- Production runbook for Modal Dict pointer repair. Tiny remote smoke passed,
  but operator flow is not documented enough yet.
- Production runbook for the `stable_slots_v1` operator workflow.
- Automatic `stable_slots_v1` workflow that writes assignments on a schedule.
- Periodic safe assignment refresh during long training.
- Online Elo continuation from existing `latest.json` at production scale.
- Automated end-to-end test for the full loop.
- One-frame public leaderboard validation at real scale. Tiny two-checkpoint
  remote rating/publish smoke passed.

Remote smoke:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode leaderboard-publish \
  --tournament-id arena-curvytron-latest212-allpairs-gpp11-gifs3-20260513-145153 \
  --rating-run-id elo-latest212-allpairs-gpp11-gifs3-20260513-145153 \
  --leaderboard-id curvytron-latest212-smoke-20260513 \
  --leaderboard-snapshot-id latest212-smoke-20260513
```

Result:

```text
row_count=212
active_count=212
provisional_count=0
commit_error=null
snapshot_ref=tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/snapshots/latest212-smoke-20260513.json
pointer_key=current:curvytron-latest212-smoke-20260513
```

## 2026-05-13 Contract Audit After Coach/Tournament Job Split

Locked down in docs and tests:

- Coach owns training, exact checkpoint writes, and immutable assignment
  consumption.
- The tournament job owns checkpoint discovery, rating, public leaderboard snapshots, and
  compact live pointers.
- Run-id and run-prefix intake scans are live watches.
- Explicit checkpoint refs are frozen seeds, not live watches.
- Checkpoint discovery must scan `train/lightzero_exp*/ckpt/iteration_*.pth.tar`
  because DI-engine may use timestamped `lightzero_exp_*` folders.
- Intake drain must not touch an existing rating run unless
  `continue_from_latest=True`.
- Continuation uses the full `seen_checkpoint_refs` pool so older rated
  checkpoints are not silently dropped.
- Submit-only intake rejects scheduler/evaluator knobs from candidate
  submissions.
- `tournament-submit` / `intake-submit` can add exact refs or run IDs without
  changing service policy.
- Drain uses manifest policy by default; ad-hoc rating overrides require
  explicit opt-in.

Checks:

```text
uv run pytest tests/test_curvytron_checkpoint_tournament.py -q
139 passed, 12 skipped

uv run pytest \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_opponent_leaderboard.py \
  tests/test_lightzero_checkpoint_opponent_provider.py -q
70 passed, 1 skipped

ruff: all checks passed
git diff --check: clean
```

## 2026-05-13 Testing Repair Pass

Added local coverage for the highest-risk tournament-job gaps:

- explicit `all_pairs` and `random` scheduler branches;
- multi-round adaptive placement fairness;
- pair-history freshness preference for useful unplayed pairs;
- queue-loss repair from manifest `queued_checkpoint_refs`;
- stale intake claim repair after the claim age threshold;
- no-active-row public leaderboard publish gate;
- one-frame public leaderboard publish gate;
- live pointer repair from immutable public leaderboard snapshots.

Latest focused checks:

```text
uv run pytest \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_intake_repair.py \
  tests/test_curvytron_tournament_scheduler_fairness.py \
  tests/test_curvytron_opponent_leaderboard_pointer_repair.py -q
151 passed, 12 skipped

uv run ruff check \
  src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_intake_repair.py \
  tests/test_curvytron_tournament_scheduler_fairness.py \
  tests/test_curvytron_opponent_leaderboard_pointer_repair.py
All checks passed

git diff --check
clean
```

## 2026-05-14 Active Pool Limit Pass

Implemented locally:

- The default rating active pool is now top 100 via
  `DEFAULT_RATING_ACTIVE_POOL_LIMIT`.
- Rating snapshots keep every known checkpoint row, but mature rows below the
  active cutoff are marked:

```json
{"status": "retired", "retired_reason": "below_active_pool_limit"}
```

- New or under-tested checkpoints are not retired just because their temporary
  rank is below 100. They stay `provisional` so placement games can pull them
  into the pool if they are good.
- Adaptive scheduling excludes retired rows while still scheduling active top
  rows and provisional/new entrants.
- Intake continuation preserves the full known checkpoint history in the rating
  spec. The scheduler, not the intake layer, decides who is active for games.
- Public leaderboard snapshots mirror the same rule, and live pointer compact
  summaries now report `retired_count` separately from `provisional_count`.

Focused checks:

```text
uv run python -B -m pytest \
  tests/test_opponent_leaderboard.py::test_build_leaderboard_snapshot_defaults_to_top_100_active_pool_and_retired_tail \
  tests/test_opponent_leaderboard.py::test_build_leaderboard_snapshot_keeps_unplaced_tail_rows_provisional \
  tests/test_opponent_leaderboard.py::test_leaderboard_pointer_compact_summary_does_not_count_retired_rows_as_provisional \
  tests/test_curvytron_checkpoint_tournament.py::test_rating_snapshot_defaults_to_top_100_active_pool_and_retains_retired_tail \
  tests/test_curvytron_checkpoint_tournament.py::test_adaptive_scheduler_uses_top_pool_and_provisional_entrants_not_retired_tail \
  tests/test_curvytron_checkpoint_intake_repair.py::test_intake_drain_continue_from_latest_preserves_retired_history_in_rating_spec -q
6 passed

uv run python -B -m pytest \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_intake_repair.py \
  tests/test_curvytron_tournament_scheduler_fairness.py \
  tests/test_curvytron_opponent_leaderboard_pointer_repair.py \
  tests/test_opponent_leaderboard.py -q
177 passed, 12 skipped
```

Remaining caveat:

- This is local contract coverage. Remote smokes are still needed for pointer
  repair, one-frame public publish, queue/stale-claim repair, and a bounded
  closed-loop run.

Remote smoke evidence added:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode leaderboard-pointer-repair \
  --leaderboard-id curvytron-latest212-smoke-20260513

result:
pointer_published=true
previous_pointer_status=current
pointer_key=current:curvytron-latest212-smoke-20260513
snapshot_ref=tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/snapshots/latest212-smoke-20260513.json
compact_summary.retired_count=0
```

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode rating \
  --tournament-id arena-oneframe-public-publish-smoke-20260514b \
  --rating-run-id elo-oneframe-public-publish-smoke-20260514b \
  --run-ids curvy-survive-bonus-blank-browser-steady-base-r034-s1110171,curvy-survive-bonus-blank-fast-steady-base-r039-s1110201 \
  --expected-checkpoint-count 2 \
  --round-count 1 \
  --pair-selection all_pairs \
  --games-per-pair 3 \
  --games-per-shard 3 \
  --max-steps 64 \
  --num-simulations 1 \
  --decision-source-frames 1 \
  --decision-ms 16.666666666666668 \
  --source-physics-step-ms 16.666666666666668 \
  --policy-mode eval \
  --wait

result:
found_count=2
pair_count=1
game_count=3
rated_pair_count=1
rating_spec.decision_source_frames=1
rating_spec.decision_ms=16.666666666666668
rating_spec.active_pool_limit=100
latest_ref=tournaments/curvytron/arena-oneframe-public-publish-smoke-20260514b/ratings/elo-oneframe-public-publish-smoke-20260514b/latest.json
```

```text
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode leaderboard-publish \
  --tournament-id arena-oneframe-public-publish-smoke-20260514b \
  --rating-run-id elo-oneframe-public-publish-smoke-20260514b \
  --leaderboard-id curvytron-oneframe-public-smoke-20260514b \
  --leaderboard-snapshot-id oneframe-public-smoke-20260514b \
  --leaderboard-active-min-distinct-opponents 1 \
  --leaderboard-active-min-valid-games 3

result:
active_count=2
diagnostic_only=false
pointer_published=true
commit_error=null
pointer_key=current:curvytron-oneframe-public-smoke-20260514b
snapshot_ref=tournaments/curvytron/leaderboards/curvytron-oneframe-public-smoke-20260514b/snapshots/oneframe-public-smoke-20260514b.json
compact_summary.retired_count=0
```
