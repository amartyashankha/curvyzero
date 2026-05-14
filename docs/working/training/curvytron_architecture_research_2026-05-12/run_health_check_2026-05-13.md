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

Correction after deeper investigation: this health check only looked at the
fixed CurvyZero path `train/lightzero_exp/ckpt`. Some rows that looked stuck at
`iteration_0` have later checkpoints under DI-engine timestamped directories
such as `train/lightzero_exp_260513_123802/ckpt`. See
[stale_checkpoint_bug_investigation_2026-05-13.md](stale_checkpoint_bug_investigation_2026-05-13.md).
The counts below are still useful for describing what the website/status reader
saw, but they are not the true highest-checkpoint counts across all LightZero
experiment directories.

Later correction: all six rows called out below as fixed-path `iteration_0`
have now been checked and have later checkpoints under timestamped
`lightzero_exp_260513_*` directories. They are still status/poller failures, but
not proof that the learner failed to checkpoint.

Broader correction: a partial 212-row audit found at least 50 preserved rows
where fixed-path status undercounted the true broad checkpoint state, including
at least 45 rows whose fixed-path status was already nonzero. This file should
now be treated as a snapshot of the old fixed-path reader, not as a true run
health table. See
[checkpoint_discovery_audit_2026-05-13.md](checkpoint_discovery_audit_2026-05-13.md).

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
4. 6 rows were still at `iteration_0` in the fixed-path status reader after
   hours: 2 `mix2` rows and 4 `mix3` rows. Later broad scans found timestamped
   checkpoint streams for all six, so this is now best read as a checkpoint
   discovery bug rather than six clear training failures.
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

This does not prove the configs are bad. After the timestamped-directory
investigation, it also does not prove those configs truly stopped
checkpointing. It means fixed-path status was especially misleading on these
rows and must be replaced by a broad `lightzero_exp*/ckpt` scan before reading
learning signal.

## Working Interpretation

The live batch is broadly healthy enough to keep monitoring: most rows advanced
between snapshots, no checkpoints regressed, and eval/GIF artifacts are still
appearing. The fixed-path reader found a stale tail, but later checks showed the
six fixed-path `iteration_0` rows were actually saving into timestamped
LightZero directories. Future health checks should use the highest checkpoint
across all `lightzero_exp*/ckpt` directories, checkpoint mtime, and eval/GIF
freshness as the primary criteria, not just `train_status` or the fixed
`lightzero_exp/ckpt` path.
