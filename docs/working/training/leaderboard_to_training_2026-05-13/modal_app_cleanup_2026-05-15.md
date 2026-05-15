# Modal App Cleanup - 2026-05-15

Historical cleanup note: this was written during the invalidated v2 period.
Current storage/app names and cleanup guidance are in `NOW.md`, `TODO.md`, and
`cleanup_lane_2026-05-15.md`.

Scope: app-level cleanup only. No Modal Volumes, Dicts, Queues, or artifacts were
deleted or purged.

## Evidence Checked

- At the time, `src/curvyzero/infra/modal/curvytron_volume_names.py` defined
  v2 lane names. That file is now deleted and must not be treated as current:
  - train: `curvyzero-lightzero-curvytron-visual-survival-train-v2`
  - tournament: `curvyzero-checkpoint-tournament-v2`
  - GIF browser: `curvyzero-curvytron-gif-browser-v2`
  - Volumes: `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`,
    `curvyzero-curvytron-control-v2`
- `docs/working/training/leaderboard_to_training_2026-05-13/live_loop_parallel_execution_2026-05-14.md`
  records `curvy-v2refresh18p-20260514b` as the current patched 18-row loop.
- `docs/working/training/leaderboard_to_training_2026-05-13/operator_runbook.md`
  allows stopping clearly old detached tournament apps, but says to preserve
  deployed training/tournament apps and the current proof app.
- `modal app list --json` was checked before and after stopping apps.
- Recent logs from detached `curvyzero-checkpoint-tournament-v2` apps referenced
  current live tournament IDs such as `curvy-v2refresh18p-live-20260514b`, so
  those detached apps were treated as live or uncertain.

## Stopped

| App ID | Description | Reason |
| --- | --- | --- |
| `ap-4AA7bzK3CC6CIWlC6b1L6S` | `curvyzero-curvytron-gif-browser` | Old non-v2 GIF browser; replaced by v2 browser. |
| `ap-6kHiOJS9aWTfchsc3nZdOQ` | `curvyzero-checkpoint-tournament` | Old non-v2 tournament app; replaced by v2 tournament app. |
| `ap-SznaKiODxdS4n5uMyC3289` | `curvyzero-background-spawn-probe` | Zero-task probe app from earlier Modal testing. |

Stop commands:

```bash
modal app stop ap-4AA7bzK3CC6CIWlC6b1L6S
modal app stop ap-6kHiOJS9aWTfchsc3nZdOQ
modal app stop ap-SznaKiODxdS4n5uMyC3289
```

Post-stop `modal app list --json` shows all three stopped at about
`2026-05-15 00:49 EDT`.

## Kept

| App ID | Description | Reason |
| --- | --- | --- |
| `ap-1qlYXmmofwXNmrtZ4XTyjC` | `curvyzero-curvytron-gif-browser-v2` | Current v2 GIF browser. |
| `ap-7JwT5bBirxOTgst6yV1slg` | `curvyzero-checkpoint-tournament-v2` | Current deployed v2 tournament/browser/service app. |
| `ap-jHbblnzHiTfhKh57P53iNg` | `curvyzero-lightzero-curvytron-visual-survival-train-v2` | Current deployed v2 training app; tasks active. |
| `ap-hgRfLZZa1y9JS6U06hIH9k` | `curvyzero-checkpoint-tournament-v2` detached | Hundreds of tasks; not clearly stale from available evidence. |
| `ap-cIdYnGnowhAaDtn1VRQJw5` | `curvyzero-checkpoint-tournament-v2` detached | Logs reference current/near-current v2 tournament work. |
| `ap-uJTlKnuoh3q4uoa6UeOReu` | `curvyzero-checkpoint-tournament-v2` detached | Logs reference `curvy-v2refresh18p-live-20260514b` current proof work. |
| `ap-fiuernb2RQSnfIz9yQC0gW` | `curvyzero-checkpoint-tournament-v2` detached | Newest detached v2 tournament app; not safe to classify stale. |

## Cataloged, Not Stopped

| App ID | Description | Reason |
| --- | --- | --- |
| `ap-DmCKutNc5xDg2x2N1hdEtS` | `curvyzero-lightzero-curvytron-visual-survival-eval` | Zero tasks and pre-v2 name, but it may still be useful as an eval utility. Leave for a separate retirement decision. |

## Notes

- Several already-stopped one-shot inspector/status apps remain in Modal history;
  they were not modified.
- No attempt was made to delete or compact `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, or `curvyzero-curvytron-control-v2`.
