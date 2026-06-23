# v2real18 Training Signal - 2026-05-15

Read-only note from local artifacts. No live Modal apps were changed.

## Sources

- Manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2real18-20260515a/curvy-v2real18-20260515a.json`
- Latest status output inspected:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2real18-20260515a/run_status_poll_raw.txt`
- Refresh/replacement manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2real18-refresh-r1-20260515a/curvy-v2real18-refresh-r1-20260515a.json`
- Rerate / discovery artifacts:
  `artifacts/local/curvytron_v2real18_live/rating_latest_round1_corrected_gate.json`,
  `artifacts/local/curvytron_v2real18_live/rating_latest_monitor2.json`,
  `artifacts/local/curvytron_v2real18_live/discover_all_current.json`,
  `artifacts/local/curvytron_v2real18_live/rerate67_progress_poll2.json`

## Best Available Answer

The v2real18 lane has strong historical progress/liveness evidence, but it is
diagnostic only and is not the current restart launch source.

The strongest trustworthy signal is checkpoint and loop progress:

- Tracked rows in latest status: `21` total = `18` original real18 rows plus
  `3` refresh-r1 replacements.
- Status: `17` running, `4` failed originals.
- Durable checkpoint files in status: `90`.
- Per-row checkpoint count range: `0..7`.
- Max latest checkpoint: `iteration_60000`.
- Original 18 rows only: `80` checkpoint files, `14` running, `4` failed.
- Replacement rows:
  - `r008` replacement: `4` checkpoints, latest `iteration_30000`, running.
  - `r009` replacement: `3` checkpoints, latest `iteration_20000`, running.
  - `r011` replacement: `3` checkpoints, latest `iteration_20000`, running.

The best direct survival-like signal currently available is GIF selfplay
`physical_steps`, but it is weak and should not be used as proof of improvement.
For the original 18 rows:

- Rows with any GIF artifact: `17/18`.
- Rows with at least two GIF artifacts: `16/18`.
- GIF artifact count: `79`.
- Mean first GIF steps over rows with any GIF: `120.3529411764706`.
- Mean latest GIF steps over rows with any GIF: `131.2941176470588`.
- Mean best GIF steps over rows with any GIF: `210.88235294117646`.
- For rows with at least two GIFs:
  - Mean first GIF steps: `121.625`.
  - Mean latest GIF steps: `133.25`.
  - Mean best GIF steps: `217.8125`.
  - Latest > first: `8/16`.
  - Latest = first: `1/16`.
  - Latest < first: `7/16`.
  - Best > first: `15/16`.
- Latest GIF collapse warnings: `1/17` rows with GIFs.

This is suggestive but not trustworthy: each GIF is one sampled selfplay episode,
the opponent mixture entry can differ across checkpoints, and the real eval
manifest path is still empty.

## What Is Trustworthy

- The batch manifest is a real H100 v2 batch: `18` rows,
  `source_max_steps=1048576`, `collector_env_num=256`,
  assignment refresh interval `2000`, background eval and GIF enabled.
- Current status over tracked rows shows real training progress:
  `90` checkpoint files and max latest checkpoint `iteration_60000`.
- Assignment refresh uptake is real: `15/21` tracked rows have applied a
  refreshed assignment. Applied shas are evenly split:
  - `4db8fe399ce6d423f50cb30d8269c2d18bbf1b7025f8c40ffb8163972604fb5a`: `5`
    rows.
  - `9717c8b00d1e4a030026ca4188611f04d961b6d6a6f477f8758f11489d8f8d45`: `5`
    rows.
  - `e348714b7c960ea62423fd5a8cedaf20427778f764957a4142d5968bc2080f36`: `5`
    rows.
- Tournament feedback exists:
  - Corrected `round-000001`: `22` active ratings, `231` rated pairs,
    `4,851` games, `stable=false`, `max_abs_delta=37.52693218215696`.
  - Later monitor `round-000003`: `40` active ratings, `300` rated pairs,
    `6,300` games, `stable=false`, `max_abs_delta=102.88711507514478`.
- Discovery found `67` checkpoint refs across `20/21` tracked run ids.
- Fresh all-pairs rerate is running with `2,211` pairs / `46,431` games, but
  its progress artifact still reports `0` completed games in the inspected
  snapshot.

## What Is Not Trustworthy Yet

- Survival improvement is not proven. Latest status has
  `eval_manifest_count_sum=0` across `21` tracked rows.
- Background eval completions total only `7`, and they have not produced
  canonical eval manifests in the run-status rollup.
- GIF `physical_steps` is a weak proxy only. It has high variance and mixed
  opponent samples, so the `121.625 -> 133.25` latest-vs-first mean over the
  `16` comparable original rows should be treated as a monitoring hint, not a
  learning claim.
- Tournament Elo/rating progress proves the loop is evaluating checkpoints
  against each other; it is not the same as absolute survival improvement.
