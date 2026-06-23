# Tooling Inventory

## Existing Tools

### Eval Curves

```text
scripts/analyze_curvytron_eval_curves.py
src/curvyzero/analysis/eval_curves.py
```

Known capability:

- consumes local JSON/JSONL snapshots;
- builds per-run eval curves;
- scores metrics such as `mean_survival` and `mean_training_reward`;
- can attach manifest axes for some manifest shapes;
- outputs JSON or table formats.

Likely use:

- survival and reward row summaries;
- curve scoring;
- first/best/latest and peak-then-crash flags.

Gap:

- does not by itself join tournament rows or build Grid A/Grid B projection
  tables.

### Run Status

```text
src/curvyzero/infra/modal/lightzero_curvytron_run_status.py
```

Known capability:

- compact Modal-side reader for training progress and eval summaries;
- has assignment-proof support;
- intended to replace noisy raw Volume polling.

Likely use:

- pull current cz26 training/eval status.

Gap:

- need exact command and output mode for the 136-run manifest.

### Live Loop Control

```text
scripts/curvytron_live_loop_control.py
```

Known capability:

- compact live status for intake/tournament/export/trainer-proof;
- avoids raw Volume spelunking as first-line operation.

Likely use:

- current tournament status;
- trainer consumption proof;
- maybe JSON for current rating/export state.

Gap:

- may need a separate rating snapshot pull or debug bundle for full analysis.

### Tournament Debug Bundle

```text
scripts/curvytron_tournament_debug_bundle.py
```

Known capability:

- candidate for packaging tournament state.

Gap:

- not yet inspected in this pass.

### Mixture Status

```text
scripts/analyze_curvytron_mixture_status.py
```

Known capability:

- older status analyzer for render/mixture experiments.

Likely use:

- probably not directly useful for cz26 Grid A/B, but has patterns for joining
  manifest and status rows.

## Tool We Probably Need

A cz26-specific grid analyzer:

```text
scripts/analyze_curvytron_cz26_grid.py
```

Inputs:

- cz26 manifest;
- run-status/eval JSON;
- tournament rating JSON;
- optional battle summary JSON.

Outputs:

- row-level JSON;
- axis-projection JSON;
- Markdown summary tables.

First version can be local-only and consume snapshots. It should not perform
Modal pulls itself.

## Current Tooling State

Implemented first local-only version:

```text
scripts/analyze_curvytron_cz26_grid.py
```

Implemented matched-contrast helper:

```text
scripts/analyze_curvytron_cz26_contrasts.py
```

Implemented second-pass deep report generator:

```text
scripts/analyze_curvytron_cz26_deep_report.py
```

Validated so far:

- compiles cleanly;
- reads the `136`-row `cz26-full-20260517a` manifest;
- emits row-level JSON;
- emits Markdown projection summaries;
- separates Grid A and Grid B axes;
- reports missing eval/tournament data instead of hiding it.
- separates all-checkpoint tournament metrics from learned-only tournament
  metrics so shared `iteration 0` seed checkpoints do not masquerade as learned
  progress.
- emits reward component/outcome-residual tables from eval-status rows;
- emits exact-horizon projections at 30k, 170k, and 300k;
- emits matched pairwise contrasts at exact horizons;
- emits tournament exposure for top learned checkpoints;
- emits action-collapse and per-run tables.

Smoke outputs:

```text
artifacts/local/cz26_analysis_2026-05-18/manifest_only_analysis.json
artifacts/local/cz26_analysis_2026-05-18/manifest_only_analysis.md
artifacts/local/cz26_analysis_2026-05-18/cz26_matched_contrasts.json
artifacts/local/cz26_analysis_2026-05-18/cz26_matched_contrasts.md
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
```

Known limitation:

- generated reports depend on the local snapshots listed in `DATA_SOURCES.md`;
- the tool does not fetch Modal data by design.

Lineage tooling exists in the repo, but it is parked for this pass. We will use
it only if status/eval/rating snapshots contradict the assumption that the run
and tournament data are trustworthy enough for analysis.
