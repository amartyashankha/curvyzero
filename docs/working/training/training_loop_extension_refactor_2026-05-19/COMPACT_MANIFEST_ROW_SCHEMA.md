# Compact Manifest Row Schema

Last updated: 2026-05-19

## Purpose

This doc defines the target launch-manifest row shape after the grouped-submit parameter shrink. It is separate from `VisualSurvivalExperimentSpec`, which is a local config-builder input, not the whole launch artifact.

## Current Truth

- `TRAIN_KWARGS_REQUIRED_FOR_GROUPED_SUBMIT` requires only `mode`, `seed`, `run_id`, and `attempt_id`.
- The submitter accepts legacy full `row["train_kwargs"]`.
- The submitter also normalizes compact/minimal rows before validation and Modal spawn.
- `build_curvytron_tonight18_manifest.py` now emits compact `train_kwargs` by default: default-equal trainer fields are omitted, while initial checkpoint, opponent source, assignment refresh, background eval/GIF non-defaults, reward/noise axes, and explicit CLI overrides stay visible.
- Non-migrated builders still often emit full flat `train_kwargs` because some row values intentionally differ from trainer defaults.

## Minimal Compact `train_kwargs`

This is launchable when all omitted values should use deployed trainer defaults:

```json
{
  "mode": "train",
  "seed": 11,
  "run_id": "example-run",
  "attempt_id": "example-attempt"
}
```

The submitter copies safe run identity fields into `poller_kwargs` before spawn. Do not put train-only fields in `poller_kwargs`.

## Optional Compact `experiment_spec`

Rows may omit `train_kwargs` and provide:

```json
{
  "experiment_spec": {
    "seed": 11,
    "reward_variant": "survival_plus_bonus_plus_outcome",
    "reward_outcome_alpha": 0.5,
    "opponent_policy_kind": "fixed_straight",
    "action_noise_probability": 0.1,
    "scale_preset": "current_broad"
  }
}
```

The submitter currently expands only current reward/noise/opponent-kind/current-scale fields. Unknown scale presets fail. This is deliberately narrower than the old flat launch surface.

## Runtime Fields

Runtime fields that are not pure config-builder concerns should stay out of `VisualSurvivalExperimentSpec`. For compact rows, use `runtime_spec`, top-level row metadata, or explicit `train_overrides` only when needed:

- initial policy checkpoint ref and load mode;
- exactly one opponent source: assignment ref or mixture spec;
- assignment refresh pointer/interval;
- own-checkpoint refresh flag;
- background eval/GIF overrides;
- non-default collector/search/batch/checkpoint-cadence overrides.

## Migration Rule

Compact by omission only when the omitted value equals the deployed trainer default. If a builder intentionally changes a value, keep it explicit until the compact schema has a named field for that axis.

Known non-default-heavy builders:

- `build_curvytron_survivaldiag_manifest.py`;
- `build_curvytron_opponent_mixture_manifest.py`.

Migrated proof case:

- `build_curvytron_tonight18_manifest.py` uses compact-by-omission rows plus `train_kwargs_schema_id=curvyzero_tonight18_compact_train_kwargs/v0`. Its internal manifest validator expands the compact row from the same default table before checking assignment, slot, render, learner-seat, and checkpoint contracts.

## Test Gates

Relevant local gates:

```bash
uv run pytest -q -p no:cacheprovider tests/test_curvytron_shared_contracts.py tests/test_curvytron_survivaldiag_submitter.py tests/test_curvytron_tonight18_manifest.py::test_grouped_submit_accepts_compact_train_kwargs_with_current_defaults
```

These tests prove:

- legacy full `train_kwargs` dry-run;
- minimal compact `train_kwargs` dry-run;
- compact `experiment_spec` normalization;
- `tonight18` compact-by-default rows still dry-run through grouped submit;
- trainer default filling for compact payloads;
- poller rejection for train-only fields;
- stale internal fields do not become required again.
