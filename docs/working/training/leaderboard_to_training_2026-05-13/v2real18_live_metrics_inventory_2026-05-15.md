# v2real18 Live Metrics Inventory - 2026-05-15

Read-only inventory. Code was not modified. Sources used: local manifests under
`artifacts/local/curvytron_tonight18_manifests/`, local tournament artifacts
under `artifacts/local/curvytron_v2real18_live/`, one bounded Modal run-status
read, `modal app list --env shankha-dev`, and a short log tail for the corrected
rerate app `ap-MKU8vQNXqZWCqX6Dle0ztG`.

## Summary

| Area | Current number |
| --- | ---: |
| Original manifest rows | 18 |
| Refresh manifest rows | 18 |
| Actually launched/tracked training rows | 21 = 18 original + 3 replacements |
| Running launched rows | 14 |
| Failed launched rows | 7 |
| Stopped launched rows | 0 observed |
| Unlaunched/missing refresh-manifest rows | 15 |
| Current launched-row checkpoints | 215 |
| Current launched-row max checkpoint iteration | 160000 |
| Canonical eval manifests in run status | 0 |
| Background eval completions in run status | 41 |
| GIF/selfplay artifacts in run status | 215 |
| Old live rating latest rows | 53 |
| Old live rating latest max iteration | 30000 |
| Corrected 16ms rerate input | 67 checkpoints / 2211 pairs / 46431 games |

Status as-of from the fresh run-status read: `2026-05-15T13:44:19Z`.

## Training Rows

Wall time is `elapsed_sec` from status when progress was readable. For the two
`unreadable` progress rows, checkpoint count and mtime were visible but elapsed
wall time was not.

| Row | Source | Reward/noise | Train status | Event | Ckpts | Latest iter | Wall time |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| r001 | original | sparse/clean | failed | checkpoint | 8 | 70000 | 1.52h |
| r002 | original | sparse/noisy | running | checkpoint | 17 | 160000 | 3.59h |
| r003 | original | sparse/clean | failed | checkpoint | 4 | 30000 | 0.61h |
| r004 | original | sparse/noisy | running | unreadable | 14 | ~130000 | n/a |
| r005 | original | sparse/clean | failed | checkpoint | 14 | 130000 | 2.86h |
| r006 | original | sparse/noisy | running | checkpoint | 10 | 90000 | 3.59h |
| r007 | original | survbonusnoout/clean | running | checkpoint | 10 | 90000 | 3.59h |
| r008 | original | survbonusnoout/noisy | failed | checkpoint | 1 | 0 | 0.35h |
| r009 | original | survbonusnoout/clean | failed | checkpoint | 0 | none | 0.01h |
| r010 | original | survbonusnoout/noisy | running | checkpoint | 13 | 120000 | 3.59h |
| r011 | original | survbonusnoout/clean | failed | checkpoint | 2 | 10000 | 0.20h |
| r012 | original | survbonusnoout/noisy | running | checkpoint | 16 | 150000 | 3.59h |
| r013 | original | survbonusout/clean | running | checkpoint | 13 | 120000 | 3.59h |
| r014 | original | survbonusout/noisy | running | unreadable | 10 | ~90000 | n/a |
| r015 | original | survbonusout/clean | running | checkpoint | 12 | 110000 | 3.59h |
| r016 | original | survbonusout/noisy | running | checkpoint | 13 | 120000 | 3.60h |
| r017 | original | survbonusout/clean | failed | checkpoint | 12 | 110000 | 2.67h |
| r018 | original | survbonusout/noisy | running | checkpoint | 10 | 90000 | 3.60h |
| r008r | replacement | survbonusnoout/noisy | running | checkpoint | 14 | 130000 | 3.08h |
| r009r | replacement | survbonusnoout/clean | running | checkpoint | 11 | 100000 | 3.08h |
| r011r | replacement | survbonusnoout/clean | running | checkpoint | 11 | 100000 | 3.08h |

The full refresh manifest has 18 rows, but only replacement rows `r008`,
`r009`, and `r011` are launched in `relaunch-r008-r009-r011.json`; the other
15 refresh rows read as missing/unlaunched, not stopped.

## Survival / Reward Signal

Trustworthy survival eval is still absent: run status reports
`eval_manifest_count=0` across the 21 launched rows. The best available
survival-like metric is therefore a weak proxy: GIF selfplay `physical_steps`
from 215 single-episode artifacts across 20 rows with GIFs.

| GIF proxy metric | Value |
| --- | ---: |
| GIF artifacts | 215 |
| Runs with GIFs | 20/21 |
| Comparable runs with >1 GIF | 19 |
| Mean first GIF steps | 114.85 |
| Mean latest GIF steps | 152.00 |
| Mean best GIF steps | 250.45 |
| Median first/latest GIF steps | 131 / 139 |
| Latest > first | 13/19 |
| Latest = first | 2/19 |
| Latest < first | 4/19 |
| Best > first | 18/19 |
| Latest GIF collapse warnings | 6 rows |

Interpretation: this is improving as a weak monitoring proxy, but it is not
proof of learning because each GIF is one sampled episode and opponent samples
vary. Outcome/private reward trend cannot be quantified from current artifacts:
the 215 GIF records expose `0` `reward`, `0` `private_reward`, and `0`
`outcome_reward` fields, and the status rows have `mean_steps/max_steps` null.

## Tournament State

The old live rating latest artifact is
`rating_latest_after_seed67.json` for
`curvy-v2real18-live-20260515a / elo-v2real18-live-20260515a`:

| Field | Value |
| --- | ---: |
| Latest old-live round | `round-000004` |
| Rating rows / checkpoint roster | 53 |
| Rated pairs / games | 300 / 6300 |
| Stable | false |
| Max rating delta | 43.513 |
| Max admitted iteration | 30000 |
| Replacement checkpoints admitted | 3 |

Current training has already advanced to 215 visible checkpoints and max
iteration `160000`, so the old live leaderboard is lagging. It is admitting
some new checkpoints from the 18-run lane, including the three replacement
rows, but not the current high-iteration checkpoints.

Discovery for the rerate found `67` checkpoint refs from `20/21` requested
launched runs, max discovered iteration `40000`, with `1` missing run. The
corrected rerate documented in `TOURNAMENT_DEBUG.md` is
`elo-v2real18-rerate67-allpairs-16ms-20260515a`; Modal app list shows
`ap-MKU8vQNXqZWCqX6Dle0ztG` still `ephemeral (detached)` with `505` tasks.
Recent app logs at `2026-05-15 09:47 EDT` show successful games with
`max_steps=1048576`. The progress reader did not return quickly enough for a
bounded status check, so final corrected leaderboard rows are not available yet.

## Weak-Run Immortal Intervention

The five weak-run immortal/blank intervention has not happened. The existing
weak-run doc says no row-scoped `~50%` blank/immortal pointer was written, and
the fresh status read shows only the normal shared refresh assignment hashes
applied to 15 original rows. There is no evidence of a row-scoped intervention
for only `r007`, `r008`, `r009`, `r011`, and `r016`.

## Recommendation

Recommendation: fix-first for leaderboard decisions, continue training where it
is already running. Do not stop/relaunch the 14 running rows. The training side
has real liveness and checkpoint production, but canonical eval is absent and
the old live leaderboard is stale at 53 rows / max iteration 30000. Wait for, or
repair progress visibility on, the corrected 16ms rerate before using tournament
rankings for promotion/stop decisions. Do not apply the five-run immortal
intervention through shared recipe pointers; it needs a row-scoped path or
dedicated relaunches.
