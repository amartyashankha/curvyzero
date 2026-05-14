# Implementation Log

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
  assignment refresh policy remain to be automated and scaled.

## Still Not Done

- Modal Dict pointer repair.
- Automatic selector/operator workflow that writes assignments on a schedule.
- Periodic safe assignment refresh during long training.
- Online Elo continuation from existing `latest.json` at production scale.
- Automated end-to-end test for the full loop.
- One-frame public leaderboard launch and validation at real scale.

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
