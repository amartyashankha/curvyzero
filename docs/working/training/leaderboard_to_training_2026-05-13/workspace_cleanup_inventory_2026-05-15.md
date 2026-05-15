# Workspace Cleanup Inventory - 2026-05-15

Historical cleanup inventory: this was captured during the invalidated v2
period and should not be used as current preserve/kill guidance. Current truth
is in `NOW.md`, `TODO.md`, and `cleanup_lane_2026-05-15.md`.

Scope: read-only inventory and cleanup plan. No apps, volumes, dicts, queues, or
artifacts were stopped, deleted, hidden, cleared, or archived by this pass.

Observed around `2026-05-15 09:40-09:50 EDT` in Modal environment
`shankha-dev` using:

```bash
modal app list --json
modal volume list --json
modal volume ls curvyzero-curvytron-tournaments-v2 tournaments/curvytron --json
modal dict list --json
modal queue list --json
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode visibility --visibility-action list
modal app logs <app-id> --tail N --timestamps
```

One read-only visibility command created short-lived helper app
`ap-cZBWswHRBbKjNZGn2OqTDt`; it completed and stopped itself.

## Historical Summary

- At capture time, these v2 deployed apps were alive:
  - train: `ap-jHbblnzHiTfhKh57P53iNg`,
    `curvyzero-lightzero-curvytron-visual-survival-train-v2`, deployed,
    `51` tasks on final app-list check.
  - tournament/browser/service: `ap-7JwT5bBirxOTgst6yV1slg`,
    `curvyzero-checkpoint-tournament-v2`, deployed, `509` tasks.
  - GIF browser: `ap-1qlYXmmofwXNmrtZ4XTyjC`,
    `curvyzero-curvytron-gif-browser-v2`, deployed, `1` task.
- At capture time, the corrected rerate was alive:
  - `ap-MKU8vQNXqZWCqX6Dle0ztG`,
    `curvyzero-checkpoint-tournament-v2`, ephemeral detached, `505` tasks.
    Logs show active `elo-v2real18-rerate67-allpairs-16ms-20260515a` game
    workers writing successful games under `curvy-v2real18-live-20260515a`.
- At capture time, these v2 storage/control objects existed:
  - Volumes: `curvyzero-runs-v2`,
    `curvyzero-curvytron-tournaments-v2`,
    `curvyzero-curvytron-control-v2`.
  - Dicts: `curvyzero-curvytron-checkpoint-intake-v2`,
    `curvyzero-curvytron-opponent-leaderboard-live-v2`.
  - Queue: `curvyzero-curvytron-checkpoint-events-v2`, `2` partitions,
    total size `136`.
- Modal queue/dict cleanup is not safe right now. The v2 queue is non-empty and
  is part of the live intake/control lane.
- Tournament visibility currently has `7` visible arenas. Some are useful live
  evidence, some are smoke/proof lanes that can be hidden after operator
  confirmation.
- The legacy non-v2 lane is still alive in multiple apps and old volumes. It
  looks stale or debug-only, but not safe to kill blindly because recent logs
  show successful `curvy-looplive-proof-20260515a` rating output as late as
  `07:53 EDT`.

## Curvyzero Apps

| Class | App ID | App | State | Tasks | Reason |
| --- | --- | --- | --- | ---: | --- |
| Historical v2 lane | `ap-jHbblnzHiTfhKh57P53iNg` | `curvyzero-lightzero-curvytron-visual-survival-train-v2` | deployed | 51 | V2 trainer app at capture time. Historical. |
| Historical v2 lane | `ap-7JwT5bBirxOTgst6yV1slg` | `curvyzero-checkpoint-tournament-v2` | deployed | 509 | V2 tournament/web/intake service at capture time. Historical. |
| Historical v2 lane | `ap-1qlYXmmofwXNmrtZ4XTyjC` | `curvyzero-curvytron-gif-browser-v2` | deployed | 1 | V2 GIF browser at capture time. Historical. |
| Necessary current lane | `ap-MKU8vQNXqZWCqX6Dle0ztG` | `curvyzero-checkpoint-tournament-v2` | ephemeral detached | 505 | Corrected 16.6667ms v2real18 rerate. Preserve until final/latest exists and is promoted or abandoned intentionally. |
| Necessary current lane | `ap-DmCKutNc5xDg2x2N1hdEtS` | `curvyzero-lightzero-curvytron-visual-survival-eval` | deployed | 0 | Eval utility; zero tasks but likely still useful. Preserve until eval path owner retires it. |
| Smoke/debug lane | `ap-V7tpkneuQJv0tgM48reBfa` | `curvyzero-checkpoint-tournament-v2` | ephemeral detached | 2 | Detached v2 tournament worker. Created during v2real18 era; preserve unless logs prove no current claim/work. |
| Smoke/debug lane | `ap-ofmTZIujPWu8btf9HFx9UI` | `curvyzero-checkpoint-tournament-v2` | ephemeral detached | 2 | Same. |
| Smoke/debug lane | `ap-mQFFZB4S88AJ1rxp4dQTAj` | `curvyzero-checkpoint-tournament-v2` | ephemeral detached | 2 | Same. |
| Smoke/debug lane | `ap-B0iDbG2mlI5PcPCL7GkJOu` | `curvyzero-checkpoint-tournament-v2` | ephemeral detached | 2 | Same. |
| Smoke/debug lane | `ap-C4WZyxRAZUNyhMY1xtEbMW` | `curvyzero-checkpoint-tournament-v2` | ephemeral detached | 2 | Same. |
| Smoke/debug lane | `ap-ODbEluCyZM1eup8KzRMnxN` | `curvyzero-lightzero-curvytron-run-status` | ephemeral | 2 | Run-status helper appeared during observation. Do not touch unless owner confirms it is just a stuck helper. |
| Smoke/debug lane | `ap-mhwnWo7dFV0RrT99HMFXoB` | `curvyzero-lightzero-curvytron-run-status` | deployed | 0 | Deployed status helper with only build logs in tail. Candidate to stop after confirming no dashboard depends on it. |
| Stale but preserve temporarily | `ap-uBnGcOwOsE8FoB2K7mEQXF` | `curvyzero-checkpoint-tournament` | deployed | 2 | Legacy non-v2 tournament app. Tail shows `curvy-looplive-proof-20260515a` completed rounds and wrote final snapshots this morning. Preserve until legacy proof lane is closed. |
| Stale but preserve temporarily | `ap-botK9LTClDI1WH2tBDwxgI` | `curvyzero-lightzero-curvytron-visual-survival-train` | deployed | 0 | Legacy non-v2 trainer. Logs query hit resource limit, so no safe proof it is idle. Preserve temporarily. |
| Stale but preserve temporarily | `ap-eXVq2pDG90HQgKiHMcEew6` | `curvyzero-checkpoint-tournament` | ephemeral detached | 2 | Legacy detached tournament worker. Earlier cleanup notes saw mixed stale/current proof activity. Preserve until old proof lane is declared done. |
| Stale but preserve temporarily | `ap-jrI3fJzTSZ6PbSmHzsuH5t` | `curvyzero-checkpoint-tournament` | ephemeral detached | 2 | Legacy detached worker; logs query hit resource limit in this pass. Preserve. |
| Stale but preserve temporarily | `ap-B39g5ipoNDvvLChlB8TU2C` | `curvyzero-checkpoint-tournament` | ephemeral detached | 2 | Legacy detached worker. Preserve pending targeted log check. |
| Stale but preserve temporarily | `ap-PHnYh4ayhPXW7BBRIWPm32` | `curvyzero-checkpoint-tournament` | ephemeral detached | 2 | Legacy detached worker. Preserve pending targeted log check. |
| Stale but preserve temporarily | `ap-nv50D3kdiYT0JLBaBIpQmy` | `curvyzero-checkpoint-tournament` | ephemeral detached | 2 | Legacy detached worker. Preserve pending targeted log check. |
| Stale but preserve temporarily | `ap-W9XMZcXqeWqdtL3Kcg4DYr` | `curvyzero-checkpoint-tournament` | ephemeral detached | 2 | Legacy detached worker. Preserve pending targeted log check. |
| Stale but preserve temporarily | `ap-5DIbxrRFyp0xLzdbPal9Tf` | `curvyzero-checkpoint-tournament` | ephemeral detached | 2 | Legacy detached worker. Preserve pending targeted log check. |
| Already stopped | `ap-uIXpEjsU0Iy0lM0NHs8qEk` | `curvyzero-checkpoint-tournament-v2` | stopped | 0 | Wrong-tick rerate app already stopped by prior cleanup. No action. |
| Already stopped | `ap-8mEuAfBCW296hOhIHYvNt9` | `curvyzero-checkpoint-tournament-v2` | stopped | 0 | Prior short rerate/status app. No action. |
| Already stopped | `ap-YwpMWrzZ24eSAV1XllNOiC` | `curvyzero-lightzero-curvytron-run-status` | stopped | 0 | Prior status helper. No action. |
| Already stopped | `ap-cZBWswHRBbKjNZGn2OqTDt` | `curvyzero-checkpoint-tournament-v2` | stopped | 0 | Read-only visibility helper from this pass; completed. No action. |

## Other Deployed Apps

These are visible in `modal app list --json`, but they are outside the CurvyZero
training/tournament cleanup lane. Do not stop them as part of this task:

- Active/nonzero-task non-Curvy apps: `dark-star` (`8`),
  `benchmark-dashboard` (`1`), `humanx-opencode` (`1`),
  `giphius-5.9.3-mini-xhigh-slow` (`2`), `giphius-opus46-avatar` (`2`),
  `lovable-hidden-endpoint-probe` (`1`),
  `imaginator-hidden-endpoint-shared-store` (`1`),
  `lovable-codex-chat` (`5`).
- Zero-task deployed non-Curvy apps look like historical probes or service
  shells, but they are not related to the current garbage-arena concern:
  `benchmark-sweeper`, `plot-playbook-v4`, `dflash-profile-playbook-v4`,
  `vibevoice-latency-bench`, many `lovable-*` probe apps, and
  `lovable-recent-history-remote-shelf`.

## Volumes

| Class | Volume | Reason |
| --- | --- | --- |
| Historical v2 lane | `curvyzero-runs-v2` | V2 training checkpoints/artifacts at capture time. Historical. |
| Historical v2 lane | `curvyzero-curvytron-tournaments-v2` | V2 tournament, ratings, GIF/browser artifacts at capture time. Historical. |
| Historical v2 lane | `curvyzero-curvytron-control-v2` | V2 control assignments/pointers at capture time. Historical. |
| Stale but preserve temporarily | `curvyzero-runs` | Legacy non-v2 apps are still alive and logs show legacy proof output. Preserve until all legacy apps are stopped and evidence is archived. |
| Stale but preserve temporarily | `curvyzero-curvytron-tournaments` | Legacy non-v2 tournament artifacts; preserve while legacy apps are alive. |
| Stale but preserve temporarily | `curvyzero-curvytron-control` | Legacy control volume created today. Preserve until legacy proof/control lane is closed. |

Other listed volumes are outside this CurvyZero lane. Do not include them in
CurvyZero cleanup commands.

Important storage warning from legacy logs: `/runs` on the legacy volume reported
`97.6%` inode usage. That supports a later archival/purge project, but it does
not justify deleting current v2 evidence.

## Dicts And Queues

| Class | Object | Observed state | Reason |
| --- | --- | --- | --- |
| Necessary current lane | Dict `curvyzero-curvytron-checkpoint-intake-v2` | exists | Current v2 intake state. Preserve. |
| Necessary current lane | Dict `curvyzero-curvytron-opponent-leaderboard-live-v2` | exists | Current v2 leaderboard pointer/cache. Preserve. |
| Necessary current lane | Queue `curvyzero-curvytron-checkpoint-events-v2` | `2` partitions, total size `136` | Live v2 checkpoint event queue. Do not clear. |
| Stale but preserve temporarily | Dict `curvyzero-curvytron-checkpoint-intake-v0` | exists | Legacy intake state. Preserve while legacy apps/volumes remain alive. |
| Stale but preserve temporarily | Dict `curvyzero-curvytron-opponent-leaderboard-live` | exists | Legacy leaderboard pointer/cache. Preserve while legacy artifacts remain referenced. |
| Stale but preserve temporarily | Queue `curvyzero-curvytron-checkpoint-events-v0` | `7` partitions, total size `938` | Legacy queue. Do not clear until legacy apps are stopped and no old proof needs repair. |
| Safe cleanup candidate later | Many `bench-*`, `pc-*`, `wormhole-*` Dicts/Queues | mostly old, many queues size `0` | Outside CurvyZero lane; inventory separately before deleting. |

## Tournament Arenas In V2 Volume

`modal volume ls curvyzero-curvytron-tournaments-v2 tournaments/curvytron --json`
showed these roots:

| Class | Arena/artifact root | Visible? | Reason |
| --- | --- | --- | --- |
| Necessary current lane | `curvy-v2real18-live-20260515a` | yes | Current v2real18 tournament. Corrected rerate `elo-v2real18-rerate67-allpairs-16ms-20260515a` is running under this tournament. Keep visible. |
| Stale but preserve temporarily | `curvy-v2champ18-live-20260514a` | yes | Still had active logs from deployed v2 tournament app around `09:41 EDT` (`elo-v2champ18` round 5 games). Keep visible until final status is captured, then archive/hide. |
| Stale but preserve temporarily | `curvy-v2refresh18p-live-20260514b` | yes | Important historical full-loop evidence, but current docs say old `v2refresh18p` readings are not final proof. Keep visible only if operators still need side-by-side comparison. Otherwise hide after a final screenshot/status capture. |
| Stale but preserve temporarily | `curvy-v2refresh18-live-20260514a` | yes | Superseded by v2refresh18p and v2real18. Preserve artifacts, hide from public/current browser. |
| Smoke/debug lane | `curvy-v2tiny-loop-20260514a` | yes | Mechanics proof only. Preserve temporarily, hide from browser. |
| Smoke/debug lane | `curvy-v2-looplive-proof3-20260515a` | yes | V2 storage/refresh proof. Useful evidence, not current leaderboard truth. Preserve temporarily, hide from browser after evidence capture. |
| Smoke/debug lane | `curvy-v2-looplive-proof3-direct-20260515a` | yes | Direct proof companion. Preserve temporarily, hide from browser after evidence capture. |
| Safe cleanup candidate later | `arena-closed-loop-smoke-20260513b` | no | Old smoke arena already hidden. Delete only after confirming no docs/tests still reference it. |
| Safe cleanup candidate later | `arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d` | no | Old top20/intake/GIF smoke already hidden. Delete later by exact path only. |
| Stale but preserve temporarily | `curvy-loop18-live-main-20260514f` | no | Legacy/non-v2 loop18 main, currently hidden. Preserve while legacy apps/volumes remain alive. |
| Stale but preserve temporarily | `curvy-loop18-live-clean3-20260514e` | no | Legacy stress target, currently hidden. Preserve while legacy apps/volumes remain alive. |
| Stale but preserve temporarily | `curvy-v2refresh18-proof-20260514b` | no | Hidden proof evidence. Preserve for now. |
| Necessary current lane | `leaderboards` | no | Not an arena; durable leaderboard snapshots. Never delete as arena cleanup. |

Recommended visible set right now:

1. Keep visible: `curvy-v2real18-live-20260515a`.
2. Keep visible temporarily: `curvy-v2champ18-live-20260514a` while active logs
   are still being reduced/captured.
3. Optional temporary keep: `curvy-v2refresh18p-live-20260514b` only if the
   operator still needs the old full-loop comparison visible.
4. Hide from tournament/GIF websites after evidence capture:
   `curvy-v2refresh18-live-20260514a`, `curvy-v2tiny-loop-20260514a`,
   `curvy-v2-looplive-proof3-20260515a`,
   `curvy-v2-looplive-proof3-direct-20260515a`.

## Safe Cleanup Plan

Do not run destructive cleanup while the corrected rerate is active. The safest
sequence is staged and reversible first.

### 1. Recheck State

```bash
modal app list --json
modal volume ls curvyzero-curvytron-tournaments-v2 tournaments/curvytron --json
modal dict list --json
modal queue list --json
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode visibility \
  --visibility-action list
```

Confirm `ap-MKU8vQNXqZWCqX6Dle0ztG` is still active before deciding anything
about v2 tournament app cleanup:

```bash
modal app logs ap-MKU8vQNXqZWCqX6Dle0ztG --tail 100 --timestamps
```

### 2. Reversible Website Cleanup

Conservative visibility cleanup, keeping current v2real18 plus active v2champ18
and old v2refresh18p comparison visible:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode visibility \
  --visibility-action hide_except \
  --visibility-keep-tournament-ids curvy-v2real18-live-20260515a,curvy-v2champ18-live-20260514a,curvy-v2refresh18p-live-20260514b
```

If the dry run looks right, apply it:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode visibility \
  --visibility-action hide_except \
  --visibility-keep-tournament-ids curvy-v2real18-live-20260515a,curvy-v2champ18-live-20260514a,curvy-v2refresh18p-live-20260514b \
  --no-visibility-dry-run
```

Cleaner visibility cleanup after old comparison evidence is captured:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode visibility \
  --visibility-action hide_except \
  --visibility-keep-tournament-ids curvy-v2real18-live-20260515a,curvy-v2champ18-live-20260514a
```

Then apply only after confirming:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament \
  --mode visibility \
  --visibility-action hide_except \
  --visibility-keep-tournament-ids curvy-v2real18-live-20260515a,curvy-v2champ18-live-20260514a \
  --no-visibility-dry-run
```

Do not use broad `modal volume rm` for arenas before visibility cleanup and
final evidence capture.

### 3. App Stop Candidates, Later Only

Only after corrected rerate completes and legacy proof owners confirm the
non-v2 lane is closed, stop app-level candidates in this order:

```bash
# status/debug helpers first, after confirming no one is using them
modal app stop ap-mhwnWo7dFV0RrT99HMFXoB

# legacy deployed non-v2 apps, only after old proof lane is closed
modal app stop ap-botK9LTClDI1WH2tBDwxgI
modal app stop ap-uBnGcOwOsE8FoB2K7mEQXF

# legacy detached non-v2 workers, only after logs show no active writes
modal app stop ap-eXVq2pDG90HQgKiHMcEew6
modal app stop ap-jrI3fJzTSZ6PbSmHzsuH5t
modal app stop ap-B39g5ipoNDvvLChlB8TU2C
modal app stop ap-PHnYh4ayhPXW7BBRIWPm32
modal app stop ap-nv50D3kdiYT0JLBaBIpQmy
modal app stop ap-W9XMZcXqeWqdtL3Kcg4DYr
modal app stop ap-5DIbxrRFyp0xLzdbPal9Tf
```

Do not stop these in the same batch:

```bash
# preserve current deployed v2 lane
# modal app stop ap-jHbblnzHiTfhKh57P53iNg
# modal app stop ap-7JwT5bBirxOTgst6yV1slg
# modal app stop ap-1qlYXmmofwXNmrtZ4XTyjC

# preserve corrected rerate until complete/abandoned intentionally
# modal app stop ap-MKU8vQNXqZWCqX6Dle0ztG
```

For the small v2 detached worker apps, stop only after confirming they are not
current claim holders or child workers:

```bash
modal app logs ap-V7tpkneuQJv0tgM48reBfa --tail 100 --timestamps
modal app logs ap-ofmTZIujPWu8btf9HFx9UI --tail 100 --timestamps
modal app logs ap-mQFFZB4S88AJ1rxp4dQTAj --tail 100 --timestamps
modal app logs ap-B0iDbG2mlI5PcPCL7GkJOu --tail 100 --timestamps
modal app logs ap-C4WZyxRAZUNyhMY1xtEbMW --tail 100 --timestamps
```

If all are idle/stale after the corrected rerate is complete:

```bash
modal app stop ap-V7tpkneuQJv0tgM48reBfa
modal app stop ap-ofmTZIujPWu8btf9HFx9UI
modal app stop ap-mQFFZB4S88AJ1rxp4dQTAj
modal app stop ap-B0iDbG2mlI5PcPCL7GkJOu
modal app stop ap-C4WZyxRAZUNyhMY1xtEbMW
```

### 4. Artifact/Volume Cleanup, Later Only

After all relevant apps are stopped and visibility is clean, delete exact stale
v2 smoke roots only. Do not delete `leaderboards`, current v2real18, v2champ18,
or any root still needed for evidence.

Candidate exact-path deletions after confirmation:

```bash
modal volume rm --recursive curvyzero-curvytron-tournaments-v2 tournaments/curvytron/arena-closed-loop-smoke-20260513b
modal volume rm --recursive curvyzero-curvytron-tournaments-v2 tournaments/curvytron/arena-curvytron-top20-furthest-intake-gifs5-gpp21-20260513d
```

Possible archive/hide-only for now, not deletion:

```bash
# keep artifacts, hide from browser instead of deleting
# curvy-v2refresh18-live-20260514a
# curvy-v2refresh18p-live-20260514b
# curvy-v2tiny-loop-20260514a
# curvy-v2-looplive-proof3-20260515a
# curvy-v2-looplive-proof3-direct-20260515a
# curvy-v2refresh18-proof-20260514b
```

Do not clear/delete these in this cleanup:

```bash
# current v2 control/data
# modal volume rm --recursive curvyzero-runs-v2 ...
# modal volume rm --recursive curvyzero-curvytron-tournaments-v2 tournaments/curvytron/leaderboards
# modal volume rm --recursive curvyzero-curvytron-control-v2 ...
# modal dict clear curvyzero-curvytron-checkpoint-intake-v2
# modal dict delete curvyzero-curvytron-checkpoint-intake-v2
# modal dict clear curvyzero-curvytron-opponent-leaderboard-live-v2
# modal queue clear curvyzero-curvytron-checkpoint-events-v2
# modal queue delete curvyzero-curvytron-checkpoint-events-v2
```

## Open Checks Before Any Destructive Cleanup

- Wait for `ap-MKU8vQNXqZWCqX6Dle0ztG` corrected rerate to write final
  `latest.json`, or explicitly mark it abandoned.
- Confirm whether `curvy-v2champ18-live-20260514a` is still actively reducing or
  only replaying old logs.
- Confirm no operator still needs `v2refresh18p` visible on the tournament/GIF
  sites.
- Resolve `ap-ODbEluCyZM1eup8KzRMnxN` run-status helper if it remains alive.
- Do a separate non-Curvy Modal workspace cleanup pass for old `lovable-*`,
  `bench-*`, `pc-*`, and `wormhole-*` objects. They are real clutter, but not
  part of the current CurvyZero evidence lane.
