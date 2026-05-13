# Checkpoint Tournament Validation, 2026-05-13

## What Must Be True

- Pair specs are deterministic.
- Game specs use stable IDs and seeds.
- Score extraction matches the env terminal info.
- Artifact refs cannot escape the tournament namespace.
- A remote game can load two real checkpoints.
- A remote game can write a summary and GIF.
- The browser can list tournaments and serve GIF/JSON files.

## Local Tests

Run:

`uv run pytest tests/test_curvytron_checkpoint_tournament.py`

This proves the pure helpers and local browser listing. It does not prove
checkpoint loading.

## Modal Smoke

Run one pair with one game:

`uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode pair --tournament-id arena-smoke-YYYYMMDD --checkpoint-refs <ref-a>,<ref-b> --games-per-pair 1 --max-steps 16 --num-simulations 1 --wait`

Success means:

- command returns `ok: true` or a clear failure summary
- `tournaments/curvytron/<id>/show_in_tournament_browser.flag` exists
- `tournaments/curvytron/<id>/battles/.../battle.json` exists
- `tournaments/curvytron/<id>/battles/.../games/game-000000/summary.json`
  exists
- `game.gif` exists when GIF saving is enabled

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

## Failure Handling

If a game fails, it should still write a failure summary. That is useful because
checkpoint loading failures are product evidence, not just infrastructure noise.

## Scale Probe Later

Before 300 checkpoints:

- run 4 checkpoints, unordered pairs, 2 games each
- run 10 checkpoints, unordered pairs, 1 game each
- measure total wall time, failures, and file count
- decide whether one-game-per-input is still good or needs game shards
