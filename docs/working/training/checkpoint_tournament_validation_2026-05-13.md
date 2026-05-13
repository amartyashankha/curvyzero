# Checkpoint Tournament Validation, 2026-05-13

## What Must Be True

- Pair specs are deterministic.
- Game specs use stable IDs and seeds.
- Score extraction matches the env terminal info.
- Artifact refs cannot escape the tournament namespace.
- A remote game can load two real checkpoints.
- A remote game can write a summary and GIF.
- The browser can list tournaments and serve GIF/JSON files.

## Adaptive Elo Must Also Prove

- Scheduler output is deterministic from pool, snapshot, round index, and seed.
- Scheduler output is bounded by the requested battle budget.
- Scheduler does not build all possible pairs for huge pools.
- Each adaptive pair has a clear schedule reason.
- New checkpoints get enough distinct opponents before active status.
- Replay battles create new battle refs and do not overwrite old summaries.
- Ratings can still be recomputed from immutable battle summaries.
- Website reads rankings/progress from small snapshot/index files, not all game
  summaries.

## Local Tests

Run:

`uv run pytest tests/test_curvytron_checkpoint_tournament.py`

This proves the pure helpers and local browser listing. It does not prove
checkpoint loading.

Latest focused run:

`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_curvytron_checkpoint_tournament.py -q`

Result:

`78 passed, 10 skipped`

Coverage added in this run:

- `adaptive_v0` requires a pair budget.
- adaptive pair specs are budgeted, unique, and tagged with schedule metadata.
- schedule metadata survives pair summary creation.
- pair history accumulates by canonical pair key across seat order.
- pair history rejects pool-hash mismatch.
- discovery can select latest, exact iteration, or all checkpoints while
  scanning timestamped `lightzero_exp_*` directories.
- Modal rating output helpers write `pair_history.json` and
  `scheduler_state.json`, and snapshots point at those refs.

Checkpoint discovery regression to keep:

- discovery scans `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
- timestamped DI-engine dirs can beat stale fixed `lightzero_exp/ckpt`;
- resume sidecars and empty checkpoint files do not enter tournament refs;
- `checkpoint_selection=all` can return more checkpoints than selected runs, so
  `max_runs` is not treated as the expected checkpoint count in that mode.

Remote discovery smoke:

```text
uv run --extra modal python -B -m modal run \
  -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode discover \
  --run-id-prefix survivaldiag-v1b-20260513h \
  --max-runs 3 \
  --checkpoint-selection latest
```

Result:

- `found_count=3`
- `missing_count=0`
- `checkpoint_scan_glob=train/lightzero_exp*/ckpt/iteration_*.pth.tar`
- returned refs included timestamped `lightzero_exp_260513_*` dirs, proving the
  tournament path is not limited to stale fixed `lightzero_exp/ckpt`.

## Modal Smoke

Run one pair with one game:

`uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode pair --tournament-id arena-smoke-YYYYMMDD --checkpoint-refs <ref-a>,<ref-b> --games-per-pair 1 --max-steps 16 --num-simulations 1 --wait`

Success means:

- command returns `ok: true` or a clear failure summary
- `tournaments/curvytron/<id>/show_in_tournament_browser.flag` exists in
  `curvyzero-curvytron-tournaments`
- `tournaments/curvytron/<id>/battles/.../battle.json` exists in
  `curvyzero-curvytron-tournaments`
- `tournaments/curvytron/<id>/battles/.../games/game-000000/summary.json`
  exists in `curvyzero-curvytron-tournaments`
- `game.gif` exists when GIF saving is enabled
- game summary has `frame_size: 704`
- game summary has `artifacts.gif.pixel_size: [704, 704]`

## Website Smoke

Open the deployed tournament browser.

Check:

- tournament appears in the dropdown/list
- battle card appears
- JSON link opens battle JSON
- GIF link opens the sample GIF

The `/api/tournaments` endpoint should answer quickly even when no tournament
exists. If it hangs, first suspect slow web-container import or a Volume listing
path that scans too much.

Observed fix on 2026-05-13: after removing the trainer import, the tournament
image needed an explicit `fastapi` dependency for the ASGI browser.

The website reads from `curvyzero-curvytron-tournaments`, not `curvyzero-runs`.

## Failure Handling

If a game fails, it should still write a failure summary. That is useful because
checkpoint loading failures are product evidence, not just infrastructure noise.

## Latest Smoke, 2026-05-13

Command shape:

`uv run --extra modal python -B -m modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode tournament --tournament-id arena-v2volume-704-smoke-20260513a --checkpoint-refs <iteration_0>,<iteration_7> --games-per-pair 1 --max-steps 8 --num-simulations 1 --wait`

Result:

- completed ok
- checkpoint volume: `curvyzero-runs`
- artifact volume: `curvyzero-curvytron-tournaments`
- tournament id: `arena-v2volume-704-smoke-20260513a`
- website `/api/tournaments` lists this tournament from the new artifact volume
- website `/api/battles` lists one battle
- game summary reports `frame_size: 704`
- game summary reports `artifacts.gif.pixel_size: [704, 704]`
- downloaded GIF header reports `GIF image data, version 89a, 704 x 704`

Smoke caveat:

- It was a tiny 8-step timeout smoke, not a meaningful policy comparison.
- It proves plumbing, storage split, website read path, and GIF size.

## Scale Probe Later

Before 300 checkpoints:

- run 4 checkpoints, unordered pairs, 2 games each
- run 10 checkpoints, unordered pairs, 1 game each
- measure total wall time, failures, and file count
- decide whether one-game-per-input is still good or needs game shards
