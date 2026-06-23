# Data Sources

## Local Sources Already Known

Manifest:

```text
artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json
```

Submit/relaunch records:

```text
artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.submit_launch.json
artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.relaunch_20260517_1955.json
```

Canary manifest:

```text
artifacts/local/curvytron_next_batch_manifests/cz26c-e2e-20260516a/cz26c-e2e-20260516a.json
```

Prior r18fresh local analysis inputs:

```text
/tmp/r18fresh_eval_status.json
/tmp/r18fresh_eval_status_clean.json
/tmp/r18fresh_eval_curve_scores.json
artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json
```

## Likely Modal Sources Needed

Run status / eval status:

```text
src/curvyzero/infra/modal/lightzero_curvytron_run_status.py
```

Live loop / tournament status:

```text
scripts/curvytron_live_loop_control.py
```

Tournament:

```text
tournament_id = cz26-live-20260517a
rating_run_id = elo-cz26-live-20260517a
leaderboard_id = cz26-live-20260517a-elo-cz26-live-20260517a-training
```

## Data Needed For First Real Pass

- [x] 136-row trainer status:
  - latest iteration;
  - checkpoint count;
  - status/completion/failure;
  - eval checkpoints and metrics.
- [x] Tournament rating rows:
  - checkpoint ref;
  - run id;
  - iteration;
  - rank/rating/status;
  - games and distinct opponents.
- [ ] Battle/game summaries:
  - game duration/survival;
  - failed games;
  - sample GIF refs if relevant later.
- [ ] Trainer export/assignment proof:
  - export generation;
  - assignment SHA;
  - trainer applied/provider-ok rows.

## Current Read

Local data is enough to answer what was launched and to run the first deep
learning analysis:

- `136` rows total;
- `96` Grid A rows;
- `40` Grid B rows;
- all rows seeded from the same pinned old rank-1 checkpoint;
- submit/relaunch records exist for all rows;
- eval/status exists for all 136 rows;
- tournament rating rows exist for all 136 rows.

Current core local snapshots:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_eval_status_latest.json
artifacts/local/cz26_analysis_2026-05-18/cz26_live_loop_status_latest.json
artifacts/local/cz26_analysis_2026-05-18/cz26_rating_latest.json
artifacts/local/cz26_analysis_2026-05-18/cz26_rating_progress.json
```

Current generated analysis:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_joined_analysis.json
artifacts/local/cz26_analysis_2026-05-18/cz26_joined_analysis.md
artifacts/local/cz26_analysis_2026-05-18/cz26_matched_contrasts.json
artifacts/local/cz26_analysis_2026-05-18/cz26_matched_contrasts.md
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
```

Battle summaries and trainer consumption proof are not required for the current
Grid A/B learning read. They are still useful if we diagnose tournament-pool
strengthening over time or exact trainer feedback timing.

Canonical local analysis output directory:

```text
artifacts/local/cz26_analysis_2026-05-18/
```

## First Pull Contract

Pull compact snapshots into the canonical artifact directory, then analyze them
locally.

Required first snapshots:

1. trainer/eval status for the 136-row manifest;
2. current tournament rating or leaderboard export for `cz26-live-20260517a`;
3. game/exposure summary only if the rating snapshot does not include games and
   opponents.

Do not start by scraping raw Volumes or rebuilding lineage tables. Those are
fallback diagnostics, not the first analysis path.

Exact first commands:

```bash
CZ26_RUN_IDS=$(jq -r '.rows[] | .run_id' artifacts/local/curvytron_next_batch_manifests/cz26-full-20260517a/cz26-full-20260517a.json | paste -sd,)

uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "$CZ26_RUN_IDS" \
  --output eval-json \
  --chunk-size 1 \
  --chunk-workers 16

uv run --extra modal python scripts/curvytron_live_loop_control.py \
  --action status \
  --activity-probe-pairs 0 \
  --lookahead-batches 64 \
  --no-drain-call-probe \
  --full-status
```

Expected saved outputs:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_eval_status_latest.json
artifacts/local/cz26_analysis_2026-05-18/cz26_live_loop_status_latest.json
```

Note: `--chunk-size 16` timed out on the first eval pull because each remote
input had too much work. The current safe pull uses `--chunk-size 1` so one slow
run cannot kill a multi-run chunk.

## Missingness To Track

- runs with no eval curves;
- runs with fewer checkpoints;
- rows that completed versus lagged or were interrupted;
- checkpoints written but not rated;
- rated checkpoints with too few games;
- rows with tournament exposure far below the median.
