# Fast Survival Readout Path

Date: 2026-05-12

Problem: `lightzero_curvytron_run_status.py --output eval-summary` is better
than full status because it skips GIF summary scans, but it still recursively
walks each run's attempt eval tree and reopens every historical eval manifest.
Across many runs this turns a simple survival-curve read into repeated Modal
Volume directory traversal.

## Current Layout

Useful refs already exist:

- `training/<task>/<run>/run.json`
- `training/<task>/<run>/latest_attempt.json`
- `training/<task>/<run>/attempts/<attempt>/attempt.json`
- `training/<task>/<run>/attempts/<attempt>/train/progress_latest.json`
- `training/<task>/<run>/attempts/<attempt>/train/checkpoint_eval_poller.json`
- `training/<task>/<run>/attempts/<attempt>/eval/<eval_id>/manifest_*.json`
- `training/<task>/<run>/checkpoints/lightzero/iteration_<n>.pth.tar`

The slow part is not JSON size; it is discovery. The status tool calls
`eval_root.rglob("manifest_*.json")` per run, then sorts and merges manifests.
Full status additionally calls `rglob("selfplay/summary.json")`.

## Target Readout Contract

Add one cached, small JSON per run:

`training/<task>/<run>/readout/survival_curve_latest.json`

Shape:

```json
{
  "schema_id": "curvyzero_curvytron_survival_readout/v1",
  "task_id": "lightzero-curvytron-visual-survival",
  "run_id": "...",
  "attempt_id": "...",
  "updated_at": "...",
  "source_manifest_refs": ["training/.../eval/.../manifest_*.json"],
  "checkpoint_count": 0,
  "latest_checkpoint": "iteration_3000",
  "poller": {"status": "completed", "seen_count": 31, "completed_count": 31},
  "train": {"status": "completed", "stage": "finished"},
  "latest": {"iteration": 3000, "mean_steps": 123.0, "max_steps": 512},
  "best": {"iteration": 2400, "mean_steps": 180.0, "max_steps": 1024},
  "curve": [
    {
      "iteration": 0,
      "checkpoint": "iteration_0",
      "mean_steps": 12.0,
      "median_steps": 11.0,
      "min_steps": 4,
      "max_steps": 30,
      "seeds": 8,
      "capped_count": 0,
      "failure_count": 0,
      "top_action": "1",
      "top_action_fraction": 0.51,
      "collapsed": false,
      "manifest_ref": "training/.../manifest_*.json"
    }
  ]
}
```

This is the object humans and subagents should read. It is tiny, stable, and
does not require walking `eval/**`.

## Writers

Preferred writer: update the eval manifest writer path.

After `curvytron_visual_survival_eval_manifest` writes a manifest, also fold
that manifest's `survival_aggregate_table` into the per-run readout JSON. The
live checkpoint eval path in `lightzero_curvyzero_stacked_debug_visual_survival_train.py`
should do the same after writing its manifest. This makes the readout hot as
evals complete and avoids a separate scan.

Fallback writer: add a rebuild mode to the status tool.

`--output rebuild-readout-cache` can do the existing expensive `rglob` once,
write `readout/survival_curve_latest.json`, and exit. This is useful for old
runs and migrations.

## Cached Run Index

Add one matrix-level index:

`training/<task>/_indexes/<matrix_or_preset>/survival_latest.json`

Shape:

```json
{
  "schema_id": "curvyzero_curvytron_survival_index/v1",
  "updated_at": "...",
  "preset": "stock-high-signal-v1",
  "runs": [
    {
      "run_id": "...",
      "attempt_id": "...",
      "readout_ref": "training/.../readout/survival_curve_latest.json",
      "latest_iter": 3000,
      "latest_mean": 123.0,
      "best_iter": 2400,
      "best_mean": 180.0,
      "poller_status": "completed",
      "train_status": "completed"
    }
  ]
}
```

Refresh it only from per-run readouts, never by scanning eval directories. A
single Modal call can refresh the index for a preset. Subagents should consume
this index or a local fetched copy.

## Status Tool Changes

Add two fast modes:

- `readout-summary`: reads only `readout/survival_curve_latest.json`,
  `progress_latest.json`, and `checkpoint_eval_poller.json` per run.
- `readout-json`: returns the same small payloads as JSON for downstream tools.

Keep `eval-summary` as a slow diagnostic/migration path. Rename in help text to
`eval-summary-scan` or document that it scans eval manifests.

Implementation detail: avoid `rglob` in the default path. Use direct refs only:

- latest attempt pointer;
- per-run readout;
- poller status;
- progress latest;
- optionally checkpoint directory `iterdir` only if readout lacks checkpoint
  count.

## Local Snapshot Flow

Add one fetch target for subagents:

`artifacts/local/curvytron_status_snapshots/<stamp>/<preset>/`

Contents:

- `survival_latest.json` copied from the Volume index;
- `runs/<run_id>.survival_curve_latest.json` copied from each readout;
- `README.md` with source refs and fetch time.

Subagents should be handed this directory and told not to call Modal. They can
rank runs, plot curves, and summarize regressions from local JSON only.

## Practical Order

1. Add per-run readout writer in eval-manifest write paths.
2. Add one migration/rebuild command for existing runs.
3. Add `readout-summary` mode that uses direct file refs only.
4. Add preset index writer/refresh.
5. Add local snapshot fetch script or mode.

This preserves the current artifact tree but changes the read path from
"discover everything every time" to "append/update once, read one file."
