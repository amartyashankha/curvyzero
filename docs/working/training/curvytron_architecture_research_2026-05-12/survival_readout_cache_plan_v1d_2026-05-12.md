# Survival Readout Cache Plan for v1d

Date: 2026-05-12

Context: `eval-summary` is now the right compact human view for v1d, but it
still took about a minute for 32 rows because it discovers and opens historical
eval manifests on the Modal Volume. The old curve summary was worse because it
also scanned GIF summaries. The next fix should make survival readout a direct
file read, not a directory walk.

## Target

Humans and subagents should read one cheap table:

`training/lightzero-curvytron-visual-survival/_indexes/<matrix>/survival_latest.json`

This table should be small enough to copy locally and complete enough to answer:

- latest checkpoint survival for each run;
- best checkpoint survival for each run;
- compact curve points for plotting;
- eval health, failure count, and action-collapse hints;
- source refs back to per-run readouts and eval manifests.

## Volume-Side Files

Per run:

`training/<task>/<run>/readout/survival_curve_latest.json`

Matrix index:

`training/<task>/_indexes/<matrix>/survival_latest.json`

Optional latest pointer:

`training/<task>/_indexes/latest_survival_<matrix>.json`

The status tool should read the matrix index first. If it is missing or stale,
it can report that explicitly instead of silently falling back to an expensive
eval/GIF scan.

## Per-Run Readout Shape

Keep the per-run file denormalized enough that no reader needs eval traversal:

```json
{
  "schema_id": "curvyzero_curvytron_survival_readout/v1",
  "matrix_id": "stock-high-signal-v1d",
  "run_id": "curvytron-stock-...",
  "attempt_id": "stock-high-signal-v1d-attempt-...",
  "updated_at": "2026-05-12T...",
  "source_manifest_count": 31,
  "source_manifest_refs": ["training/.../manifest_*.json"],
  "poller": {"status": "completed", "completed_count": 31, "failure_count": 0},
  "latest": {
    "iteration": 3000,
    "checkpoint_label": "iteration_3000",
    "mean_steps": 123.0,
    "max_steps": 512,
    "win_rate": 0.25,
    "failure_count": 0,
    "top_action": "1",
    "top_action_fraction": 0.42,
    "collapsed": false
  },
  "best": {
    "metric": "mean_steps",
    "iteration": 2400,
    "checkpoint_label": "iteration_2400",
    "mean_steps": 180.0,
    "max_steps": 1024,
    "win_rate": 0.375
  },
  "curve": [
    {
      "iteration": 100,
      "checkpoint_label": "iteration_100",
      "mean_steps": 13.5,
      "median_steps": 12.0,
      "min_steps": 4,
      "max_steps": 24,
      "win_rate": 0.375,
      "failure_count": 0,
      "manifest_ref": "training/.../manifest_*.json"
    }
  ]
}
```

Use only fields already present in the eval manifest tables where possible.
This keeps the writer simple and avoids coupling the readout to GIF artifacts.

## Matrix Index Shape

The index should be optimized for terminal readout and ranking:

```json
{
  "schema_id": "curvyzero_curvytron_survival_index/v1",
  "matrix_id": "stock-high-signal-v1d",
  "updated_at": "2026-05-12T...",
  "run_count": 32,
  "rows": [
    {
      "row": "01",
      "run_id": "curvytron-stock-...",
      "attempt_id": "stock-high-signal-v1d-attempt-...",
      "readout_ref": "training/.../readout/survival_curve_latest.json",
      "latest_iter": 3000,
      "latest_mean_steps": 123.0,
      "latest_max_steps": 512,
      "best_iter": 2400,
      "best_mean_steps": 180.0,
      "poller_status": "completed",
      "failure_count": 0,
      "collapsed": false
    }
  ]
}
```

This is the file subagents should receive by default.

## Writer Strategy

Preferred path:

1. After each numeric eval manifest is written, fold that manifest into the
   per-run readout.
2. After the per-run readout is updated, refresh the matrix index row for that
   run.
3. Do not wait for GIF generation and do not read GIF summaries.

Migration path:

Add one explicit slow command such as `--output rebuild-survival-cache` or a
small script that performs the current manifest scan once, writes all per-run
readouts, then writes the matrix index. This is appropriate for v1d because the
expensive scan already succeeded and the compact results are documented.

## Status Tool Modes

Fast default for humans:

`--output survival-summary --matrix stock-high-signal-v1d`

Behavior:

- read exactly one index JSON;
- print latest/best rows;
- never call `rglob`;
- never inspect GIF directories.

Fast JSON for subagents:

`--output survival-json --matrix stock-high-signal-v1d`

Behavior:

- return the index JSON plus optional per-run curves if requested;
- fail closed with a clear "cache missing/stale" message.

Slow diagnostic:

`--output eval-summary-scan`

Behavior:

- keep the current eval-manifest traversal;
- label it as a scan;
- reserve it for migration, auditing, and cache rebuilds.

## Local Snapshot

Add a single fetch/export target:

`artifacts/local/curvytron_status_snapshots/<stamp>/<matrix>/`

Contents:

- `survival_latest.json`
- `runs/<run_id>.survival_curve_latest.json`
- `source_refs.json`
- `README.md`

Subagent instruction should be simple: use this directory only; do not run
Modal status commands. That turns a 60 second shared-volume read into local JSON
loads and prevents many agents from repeating the same traversal.

## Smallest Useful Implementation

1. Implement the per-run readout builder as pure functions in the status module
   or a tiny helper module.
2. Add tests using synthetic eval manifests that verify latest, best, curve
   sorting, failure counts, and action-collapse fields.
3. Add the slow rebuild mode to populate v1d caches from existing artifacts.
4. Add `survival-summary` and `survival-json` that read only the cache.
5. Later, wire the eval writer to keep the cache hot for future runs.

This keeps the first patch low risk: no trainer behavior changes, no Modal
launch changes, no old two-seat trainer changes, and no dependency on GIFs.
