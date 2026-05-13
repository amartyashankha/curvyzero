# Eval Curve Tooling Plan

Purpose: build a reusable analysis package for large training tensors. The
tooling should make many runs easy to compare without pretending one score is
the whole truth.

## Goal

Given a matrix manifest or run list, produce curves for each run and compare
runs across axes.

Metrics should be pluggable:

- outcome win rate;
- mean/median/max survival;
- training reward;
- bonus pickup count/reward;
- action collapse;
- terminal reasons;
- eval failures.

## Data Flow

1. Read matrix manifest and run IDs.
2. Fetch or rebuild per-run curve data from Modal Volume artifacts.
3. Write a local snapshot under:

```text
artifacts/local/curvytron_status_snapshots/<stamp>/<matrix>/
```

4. Analyze local JSON only.
5. Emit tables and optionally CSV/JSON for plots.

## Curve Object

Keep it simple:

```json
{
  "run_id": "...",
  "attempt_id": "...",
  "axes": {"reward": "...", "render": "..."},
  "points": [
    {
      "iteration": 0,
      "metrics": {
        "win_rate": 0.125,
        "mean_survival": 12.875,
        "mean_reward": 0.0
      }
    }
  ]
}
```

## First Scoring Functions

These are rough filters, not final truth:

- `latest(metric)`;
- `best(metric)`;
- `delta(metric) = latest - first`;
- `best_delta(metric) = best - first`;
- `early_slope(metric)`;
- `late_slope(metric)`;
- `peak_then_crash(metric)`;
- `is_flat(metric)`;
- `collapse_flag`;
- `eval_health`.

Be cautious about false negatives. A run that learns late or briefly peaks
should remain visible for manual review.

## First Implementation Scope

- Use existing eval manifests and status helpers.
- Add tests with synthetic curves.
- Do not touch training behavior.
- Do not depend on GIFs.
- Prefer local JSON snapshots for subagents.

## 2026-05-12 First Local Patch

Implemented a Modal-free first pass:

- `src/curvyzero/analysis/eval_curves.py` defines `CurvePoint`,
  `EvalCurve`, local JSON/JSONL loaders, manifest-axis extraction,
  eval-summary row conversion, and simple scoring.
- `scripts/analyze_curvytron_eval_curves.py` is a thin CLI wrapper for local
  curve/eval-summary snapshots.
- Scoring is metric-agnostic: callers can score one metric or multiple metrics
  such as `win_rate`, `mean_survival`, and `mean_reward` side by side. There is
  no built-in single truth metric.
- Scores include `best_delta`, `peak_signal`, `late_bloom`, and
  `peak_then_crash` so temporary or late-moving runs remain visible for manual
  review.
- Default flat/signal/crash thresholds are metric-scale aware: rate-like
  metrics such as `win_rate` use smaller thresholds than survival-step metrics,
  and callers can override them from the CLI.
- `tests/test_eval_curves.py` covers increasing, flat, peak-then-crash,
  late-blooming, collapse-flag, multi-metric, eval-summary, and canonical curve
  inputs.

The first patch intentionally does not fetch Modal Volume artifacts, scan GIFs,
plot curves, or make run recommendations.

## 2026-05-13 v1d Validation Patch

Added the missing bridge from Modal status to local curve analysis:

- `lightzero_curvytron_run_status.py` supports `--output eval-json`;
- `eval_curves.py` can read rows with full `eval_checkpoints`;
- outcome histograms become `win_rate`, `loss_rate`, `draw_rate`, and
  `cap_rate`;
- checkpoint `mean_steps` becomes `mean_survival`.

Validated on the old `stock-tensor-v1d` matrix:

- local snapshot:
  `artifacts/local/curvytron_status_snapshots/2026-05-13/stock-tensor-v1d-eval.clean.json`
- score table:
  `artifacts/local/curvytron_status_snapshots/2026-05-13/stock-tensor-v1d-curve-scores.md`

Result: the tooling agrees with the manual projection. Fixed and old opponents
show survival lift; recent/mid frozen opponents mostly show saturated outcome
with flat survival.

## 2026-05-13 Survivaldiag Readout Patch

Small analysis-only extension for future large survivaldiag matrices:

- `eval_curves.py` now exposes `MetricSpec`, `METRIC_SCHEMA`, and
  `METRIC_SCHEMA_BY_NAME` so readouts can group metrics as outcome, survival,
  reward, bonus, action, terminal, and health instead of inventing columns per
  report.
- `summarize_curve_metrics(curve)` returns a family-grouped summary for all
  available schema metrics, plus latest terminal cause, latest top action,
  collapse flag, and eval health.
- `eval_checkpoints` parsing keeps v1d fields compatible and additionally
  normalizes future fields when present:
  `mean_reward`/`mean_training_reward`, `bonus_count`,
  `bonus_pickup_count`, `bonus_reward`, `action_histogram`, `action_entropy`,
  `terminal_reason_histogram`/`terminal_cause_histogram`, and
  `failure_rate`.
- Terminal histograms become scalar rates such as `wall_rate`,
  `own_trail_rate`, `opponent_trail_rate`, `timeout_rate`, plus specific
  `terminal_<cause>_rate` metrics for causes not known ahead of time.

Current gap: the parser can consume survivaldiag telemetry once it is in local
JSON, but the fetch/status path still needs to carry reward components, bonus
counts, terminal-cause histograms, and action entropy from the raw eval
manifests into `eval_checkpoints`.

## Later

- Plot curves.
- Group by arbitrary axes from manifest names/flags.
- Add beam-search style recommendations for next matrix candidates.
