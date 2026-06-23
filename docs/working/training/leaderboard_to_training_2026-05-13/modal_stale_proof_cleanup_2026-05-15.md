# Modal Stale Proof Cleanup - 2026-05-15

Historical cleanup note: this predates the all-v2 reset and the active nonzero
source rerate. Current cleanup authority is `cleanup_lane_2026-05-15.md`; do
not use this file to decide whether to stop `ap-j21sPzVU0Ow0OS6RUZSXV0`.

Scope: Modal app catalog/cleanup only in environment `shankha-dev`. No volumes,
dicts, queues, artifacts, or deployed apps were deleted or stopped.

Checked between about `2026-05-15 05:00` and `05:07 EDT` with:

```bash
modal app list --env shankha-dev --json
modal app logs --env shankha-dev <app-id> --since 2h --search curvy-looplive-fast-proof-20260515a
modal app logs --env shankha-dev <app-id> --tail 500
modal container list --env shankha-dev --app-id <app-id> --json
```

## Stopped

None.

Reason: the app IDs with confirmed stale `curvy-looplive-fast-proof-20260515a`
`FileExistsError` logs also had later/current
`curvy-looplive-controllong-proof-20260515d` activity. App-level stopping would
therefore risk stopping current proof/tournament work.

## Active/Recent Ephemeral Curvyzero Catalog

| App ID | App | Tasks observed | Evidence | Action |
| --- | --- | ---: | --- | --- |
| `ap-eXVq2pDG90HQgKiHMcEew6` | `curvyzero-checkpoint-tournament` | 91 | Stale `curvy-looplive-fast-proof-20260515a` `FileExistsError` at `04:16`, but later `curvy-looplive-controllong-proof-20260515d` `FileExistsError` at `05:03`. | Preserved: mixed old/current app, not safe for app-level stop. |
| `ap-jrI3fJzTSZ6PbSmHzsuH5t` | `curvyzero-checkpoint-tournament` | 2 | No matching `curvy-looplive-fast-proof-20260515a`, `controlfast`, or `controllong` log evidence in recent filtered checks; recent tail shows intake subscriber scheduling only. | Preserved: ambiguous/orphan-like, but not a proven stale proof worker. |
| `ap-B39g5ipoNDvvLChlB8TU2C` | `curvyzero-checkpoint-tournament` | 2 | Stale `curvy-looplive-fast-proof-20260515a` `FileExistsError` from `04:11`-`04:20`, but later `curvy-looplive-controllong-proof-20260515d` rating output at `05:02`. | Preserved: mixed old/current app, not safe for app-level stop. |
| `ap-PHnYh4ayhPXW7BBRIWPm32` | `curvyzero-checkpoint-tournament` | 6 | Stale `curvy-looplive-fast-proof-20260515a` `FileExistsError` from `04:11`-`04:20`, but later `curvy-looplive-controllong-proof-20260515d` / `elo-looplive-controllong-proof-fresh-20260515e` output at `05:00`-`05:03`. | Preserved: mixed old/current app, not safe for app-level stop. |
| `ap-nv50D3kdiYT0JLBaBIpQmy` | `curvyzero-checkpoint-tournament` | 10 | Current `curvy-looplive-controllong-proof-20260515d` logs; no `curvy-looplive-fast-proof-20260515a` match in the 2h search. | Preserved: current proof/tournament worker. |
| `ap-ciAzi7ByfRueLxZLtqxuEf` | `curvyzero-lightzero-curvytron-visual-survival-train` | 1 initially, then 0 | `try-looplive-proof-controllong-20260515d` / `curvy-looplive-proof-controllong-20260515d` trainer logs, initialized from `curvy-looplive-proof-controlfast-20260515c`. Final app list showed it stopped at `2026-05-15 05:05:35-04:00`; this cleanup did not stop it. | Preserved/no operator stop: user explicitly said the longer trainer must not be stopped. |
| `ap-W9XMZcXqeWqdtL3Kcg4DYr` | `curvyzero-checkpoint-tournament` | 2 | Current `curvy-looplive-controllong-proof-20260515d` rating output; no `curvy-looplive-fast-proof-20260515a` match in the 2h search. | Preserved: current proof/tournament worker. |
| `ap-5DIbxrRFyp0xLzdbPal9Tf` | `curvyzero-checkpoint-tournament` | 2 | Current `curvy-looplive-controllong-proof-20260515d` / `elo-looplive-controllong-proof-fresh-20260515e` output; no `curvy-looplive-fast-proof-20260515a` match in the 2h search. | Preserved: current proof/tournament worker. |

## Deployed Apps Preserved

| App ID | App | Reason |
| --- | --- | --- |
| `ap-7JwT5bBirxOTgst6yV1slg` | `curvyzero-checkpoint-tournament-v2` | Current deployed tournament app; user said not to stop. |
| `ap-jHbblnzHiTfhKh57P53iNg` | `curvyzero-lightzero-curvytron-visual-survival-train-v2` | Current deployed trainer app; user said not to stop. |
| `ap-1qlYXmmofwXNmrtZ4XTyjC` | `curvyzero-curvytron-gif-browser-v2` | Current deployed GIF browser; user said not to stop. |
| `ap-DmCKutNc5xDg2x2N1hdEtS` | `curvyzero-lightzero-curvytron-visual-survival-eval` | Deployed eval app, outside this stale-proof cleanup. |

## Final Verification

Final `modal app list --env shankha-dev --json` still showed the deployed v2
apps running and no operator-stopped apps from this cleanup. The active
ephemeral tournament apps remaining were:

- `ap-eXVq2pDG90HQgKiHMcEew6`
- `ap-jrI3fJzTSZ6PbSmHzsuH5t`
- `ap-B39g5ipoNDvvLChlB8TU2C`
- `ap-PHnYh4ayhPXW7BBRIWPm32`
- `ap-nv50D3kdiYT0JLBaBIpQmy`
- `ap-W9XMZcXqeWqdtL3Kcg4DYr`
- `ap-5DIbxrRFyp0xLzdbPal9Tf`

## Follow-up

If app-level cleanup is still desired, `ap-jrI3fJzTSZ6PbSmHzsuH5t` is the only
active ephemeral app that looked orphan-like from logs, but it was not stopped
because it did not have direct stale-proof evidence. The stale proof failures in
`ap-eXVq2pDG90HQgKiHMcEew6`, `ap-B39g5ipoNDvvLChlB8TU2C`, and
`ap-PHnYh4ayhPXW7BBRIWPm32` should be handled only with finer-grained controls
than `modal app stop`, or after confirming they no longer own current
`controllong` work.
