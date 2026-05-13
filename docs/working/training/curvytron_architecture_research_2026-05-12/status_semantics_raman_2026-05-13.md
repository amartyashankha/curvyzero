# Status Semantics: Stale Checkpoints vs `running`

Date: 2026-05-13

Scope: read-only inspection of local Modal-volume status snapshots:

- `artifacts/local/curvytron_pruning/status_chunks_20260513d`
- `artifacts/local/curvytron_pruning/status_chunks_20260513e`

No source code was changed for this note.

## Short Read

The most reliable evidence that training is still making progress is checkpoint movement: latest checkpoint iteration and latest checkpoint mtime, ideally compared across snapshots.

The `train_status=running` field is not reliable by itself. In both snapshots every checked row says `running`, but many rows have checkpoints that are hours old and did not advance between snapshots.

## Snapshot-Level Evidence

| Snapshot | Checked rows | `train_status=running` | Progress files | Eval manifests | GIF artifacts | Stale latest checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `20260513d` | 212 | 212 | 210 | 211 | 212 | 25 rows older than 180 min; 3 older than 360 min; max 659.63 min |
| `20260513e` | 212 | 212 | 211 | 211 | 212 | 36 rows older than 180 min; 10 older than 360 min; max 707.37 min |

Between `d` and `e`, 157 rows advanced and 55 rows did not advance. The advanced rows are the clearest live-progress evidence. The unchanged rows are the rows that need suspicion even if their heartbeat says `running`.

Bucket movement from `20260513e`:

| Bucket | Advanced | Unchanged |
|---|---:|---:|
| `mix2` | 30 | 22 |
| `mix3` | 102 | 24 |
| `survival` | 25 | 8 |
| `v1b_dependency` | 0 | 1 |

## Field Reliability

| Field | Use It For | Do Not Use It For | Evidence |
|---|---|---|---|
| `latest_checkpoint`, `latest_checkpoint_mtime`, `checkpoint_count`, `checkpoints[]` | Actual checkpoint existence and training movement | Full health by itself | Cross-snapshot movement showed 157 advanced and 55 unchanged rows. |
| Cross-snapshot checkpoint comparison | Best liveness signal in these artifacts | Instantaneous process state | It catches rows that still say `running` but have not checkpointed for hours. |
| `train_status`, `train_stage`, `status_heartbeat_exists` | Last written trainer heartbeat state | Current liveness | Both snapshots have `train_status=running` for all 212 rows, while `20260513e` has 36 rows stale over 180 min. |
| `progress_exists` | Whether `progress_latest.json` exists | Current liveness | `20260513e` has 211 progress files but 55 unchanged rows. |
| Flattened `progress_latest` fields: `iteration`, `timestamp`, `elapsed_sec` | Context, with caution | Single source of progress | Some rows report `iteration=0` while latest checkpoint is far beyond zero. Some rows have recent progress timestamps but stale checkpoints. |
| `eval_manifest_count`, `latest_eval_checkpoint` | Eval artifact existence and eval lag | Trainer liveness | Eval can lag latest checkpoint, and one row has zero eval manifests while GIF artifacts exist. |
| `gif_artifact_count`, `latest_gif_checkpoint` | GIF artifact existence and GIF lag | Trainer liveness | GIF artifacts exist for all 212 rows, including stale rows. |
| `background_poller_status`, `background_poller_seen_count`, `background_poller_scheduled_count` | Poller context | Trainer liveness or completed eval count | All 212 rows have `background_poller_status=running` and `background_poller_completed_count=0`; many still have eval/GIF artifacts. |
| `background_poller_completed_count`, `background_poller_eval_completed_count`, `background_poller_gif_completed_count` | Possibly terminal poller summaries after exit | Live worker completion while poller is running | In `20260513e`, 211 rows have eval and GIF artifacts while `background_poller_completed_count=0` on all 212 rows. |

## Exact Contradictions

### All Rows Say Running, But Some Are Very Stale

In `20260513e`, `train_status_counts` is:

```json
{"running": 212}
```

The same summary reports:

- `stale_gt_180_min_count`: 36
- `stale_gt_360_min_count`: 10
- max latest checkpoint age: 707.37 minutes

So `running` means "last heartbeat said running", not "checkpoint is currently advancing".

### Stale Dependency Row Still Says Running

From `20260513e`:

| Field | Value |
|---|---|
| `run_id` | `survivaldiag-v1b-20260513h-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40` |
| `train_status` | `running` |
| `train_stage` | `auto_resume_checked` |
| `status_heartbeat_exists` | `true` |
| `progress_exists` | `false` |
| `latest_checkpoint` | `iteration_20000` |
| checkpoint age | 707.37 min |
| `eval_manifest_count` | 4 |
| `gif_artifact_count` | 5 |
| `background_poller_status` | `running` |

This row has no progress file and an extremely stale checkpoint, but still says `running`.

### Progress Exists, Eval Exists, GIF Exists, But Training Is Stale At Iteration 0

From `20260513e`:

| Field | Value |
|---|---|
| `run_id` | `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` |
| `train_status` | `running` |
| `train_stage` | `auto_resume_checked` |
| `progress_exists` | `true` |
| `iteration` | 0 |
| `timestamp` | `2026-05-13T19:14:41.058875Z` |
| `elapsed_sec` | 23798.137314 |
| `latest_checkpoint` | `iteration_0` |
| checkpoint age | 449.87 min |
| `eval_manifest_count` | 1 |
| `gif_artifact_count` | 1 |

This is the clearest example that `progress_exists`, heartbeat, eval count, and GIF count do not imply current progress.

### GIFs Can Exist While Eval Manifest Count Is Zero

From `20260513e`:

| Field | Value |
|---|---|
| `run_id` | `curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031` |
| `train_status` | `running` |
| `progress_exists` | `true` |
| `iteration` | 20000 |
| `latest_checkpoint` | `iteration_20000` |
| checkpoint age | 404.77 min |
| `eval_manifest_count` | 0 |
| `latest_eval_checkpoint` | empty |
| `gif_artifact_count` | 3 |
| `latest_gif_checkpoint` | `iteration_20000` |

This means GIF artifact presence and eval manifest presence are separate artifact signals.

### Poller Can Lag Latest Checkpoint

Run: `curvy-survive-bonus-blank-browser-steady-base-r034-s1110171`

| Snapshot | `latest_checkpoint` | `checkpoint_count` | `latest_eval_checkpoint` | `latest_gif_checkpoint` | `background_poller_status` | Seen / scheduled / completed |
|---|---:|---:|---:|---:|---|---|
| `20260513d` | `iteration_225000` | 16 | `iteration_225000` | `iteration_225000` | `running` | 16 / 16 / 0 |
| `20260513e` | `iteration_255000` | 18 | `iteration_240000` | `iteration_240000` | `running` | 17 / 17 / 0 |

The training checkpoint advanced from `iteration_225000` to `iteration_255000`, but eval/GIF only reached `iteration_240000` by the later snapshot. That is normal lag, not proof of failure.

The same row also shows stale progress fields:

| Snapshot | `iteration` | `timestamp` | `elapsed_sec` | `latest_checkpoint` |
|---|---:|---|---:|---|
| `20260513d` | 0 | `2026-05-13T09:38:30.815086Z` | 10.019549 | `iteration_225000` |
| `20260513e` | 0 | `2026-05-13T09:38:30.815086Z` | 10.019549 | `iteration_255000` |

So the flattened `progress_latest` fields can be stale even while checkpoint files are advancing.

### Progress Iteration Can Disagree With Latest Checkpoint

In `20260513e`, 25 rows have `progress_exists=true` but flattened `iteration` does not match the latest checkpoint iteration. Examples:

| Run | `iteration` | `timestamp` | `latest_checkpoint` | `checkpoint_count` |
|---|---:|---|---|---:|
| `curvy-survive-bonus-blank-fast-light-base-r063-s1111121` | 0 | `2026-05-13T09:47:03.737100Z` | `iteration_285000` | 20 |
| `curvy-survive-bonus-blank-fast-medium-base-r109-s1112151` | 0 | `2026-05-13T09:50:57.835085Z` | `iteration_285000` | 20 |
| `curvy-survive-bonus-passive-browser-medium-base-r230-s1132607` | 0 | `2026-05-13T10:00:22.134858Z` | `iteration_210000` | 15 |

This makes checkpoint listing more trustworthy than the progress fields for progress status.

## Practical Interpretation Rules

Use this order when reading rows:

1. First check `latest_checkpoint` and `latest_checkpoint_mtime`.
2. Then compare latest checkpoint iteration against a previous snapshot.
3. Treat `train_status=running` and `status_heartbeat_exists=true` as weak context only.
4. Treat eval/GIF fields as observability artifact status, not trainer status.
5. Treat poller `running` as "poller was last seen running"; use latest eval/GIF checkpoint to see whether it is caught up.
6. Treat `background_poller_completed_count=0` as expected while the poller is running, not as "nothing completed".
7. Treat `progress_latest` fields as mixed-quality context. If they disagree with checkpoint files, trust checkpoint files.

## Bottom Line

Rows that say `running` but have stale checkpoints are real in these snapshots. The status UI should separate:

- trainer heartbeat state
- checkpoint freshness
- checkpoint advancement since previous snapshot
- eval/GIF artifact coverage
- poller lag

The current artifact fields already contain enough information to make that distinction, but `running` should not be displayed as if it means active training progress.
