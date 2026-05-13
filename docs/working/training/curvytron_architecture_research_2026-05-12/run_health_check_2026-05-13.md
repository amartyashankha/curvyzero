# CurvyTron Run Health Check 2026-05-13

Purpose: read-only health check of the 212 preserved CurvyTron runs after the
volume prune. No code changes were made for this check.

Fresh snapshot:

- current: `artifacts/local/curvytron_pruning/status_chunks_20260513e/`
- previous comparison snapshot:
  `artifacts/local/curvytron_pruning/status_chunks_20260513d/`
- elapsed wall time between snapshots: roughly 40 minutes

## Plain Read

Most preserved runs are alive and making progress, but not all of them are
healthy. The useful progress signal is checkpoint file advancement plus eval/GIF
artifact creation. `train_status=running` and `progress_latest.json` alone are
not enough: some rows report `running` even when their latest checkpoint has not
changed for hours, and some healthy rows have stale `progress_latest.json`.

## Counts

| Check | Result |
| --- | ---: |
| preserved runs checked | 212 |
| train status says running | 212 |
| train root exists | 212 |
| heartbeat file exists | 212 |
| progress file exists | 211 |
| eval manifest present | 211 |
| GIF artifact present | 212 |
| checkpoint advanced since previous snapshot | 157 |
| checkpoint unchanged since previous snapshot | 55 |
| checkpoint regressed | 0 |

Checkpoint age:

| Latest checkpoint age | Count |
| --- | ---: |
| within 30 minutes | 132 |
| within 60 minutes | 158 |
| within 90 minutes | 160 |
| within 3 hours | 177 |
| older than 3 hours | 36 |
| older than 6 hours | 10 |

Latest checkpoint iteration:

- min: `iteration_0`
- median: `iteration_100000`
- max: `iteration_285000`

## By Batch

| Batch | Rows | Advanced | Unchanged | Notes |
| --- | ---: | ---: | ---: | --- |
| survival | 33 | 25 | 8 | mostly moving; a few heavy/search/batch rows look stale |
| mix2 | 52 | 30 | 22 | weakest health; several old rows are stuck for many hours |
| mix3 | 126 | 102 | 24 | mostly moving; some compute probe/control rows are stale |
| v1b dependency | 1 | 0 | 1 | preserved dependency artifact, not a useful live-training row |

## Obvious Issues

1. There are 55 unchanged rows since the previous snapshot. 54 of those also
   did not add eval or GIF artifacts.
2. 36 rows have not produced a checkpoint for more than 3 hours.
3. 10 rows have not produced a checkpoint for more than 6 hours. These are
   mostly `mix2`; the preserved `v1b` dependency is also in this group.
4. 6 rows are still at `iteration_0` after hours. That is suspicious:
   2 `mix2` rows and 4 `mix3` rows.
5. One `mix2` row has no eval manifest:
   `curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031`.
6. One non-dependency live row is missing `progress_latest.json`:
   `curvy-mix3cur-scr100-rb-s8-c32-l32-repM-k10-c2-s2309021`. It still has
   checkpoints, evals, and GIFs, so this looks like a status/progress-file
   issue rather than proof that the row never ran.

## Pattern

The stale rows are not random. They are concentrated in:

- `mix2`, especially recipes containing scripted or older checkpoint opponents:
  `r50-scr50`, `r50-old50`, and `r50-blank25-scr25`;
- `mix3` compute or heavier probes: `sim16`, `c64`, or `l64` rows show up more
  often among stale rows;
- a few older survival heavy/search/batch rows.

This does not prove the configs are bad. It does mean these rows should not be
treated as equally healthy when reading learning signal.

## Working Interpretation

The live batch is broadly healthy enough to keep monitoring: most rows advanced
between snapshots, no checkpoints regressed, and eval/GIF artifacts are still
appearing. But there is a real stale tail. The stale rows may be slow, hung, or
reporting stale `running` heartbeats. A later fix/investigation should use
checkpoint mtime and eval/GIF freshness as the primary health criteria, not just
`train_status`.
