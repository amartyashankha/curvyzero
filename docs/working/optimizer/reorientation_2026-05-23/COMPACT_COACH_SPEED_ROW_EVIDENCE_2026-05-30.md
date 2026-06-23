# Compact Coach Speed-Row Evidence

Date: 2026-05-30

Status: local evidence contract landed and hardened; local producer smoke
passed; durable H100 loaded-checkpoint threshold row accepted and attached to
compatibility.
Role: prevent `coach_speed_row` from being closed by a prefixed string or a
profile-only row.
Authority: artifact-specific record; `CURRENT_STATE.md` and `goal.md` remain
the current operating truth.

## What Landed

Added:

```text
src/curvyzero/training/compact_coach_speed_row.py
tests/test_compact_coach_speed_row.py
```

The contract schema is:

```text
curvyzero_compact_coach_speed_row_evidence/v1
```

Coach compatibility still stores a short evidence ref in metadata, but
`coach_speed_row=true` now requires structured evidence that validates and
regenerates the exact ref. A string like `compact_coach_speed_row:anything`
no longer closes the gate.

## Required Shape

The evidence object binds:

- unified lifecycle report path/hash and checkpoint id;
- manifest path/hash and row hash;
- result JSON path/hash plus summary/payload hashes;
- exact speed-row manifest/result schemas;
- producer/run provenance;
- candidate checkpoint id and route through lifecycle, manifest, row, result,
  summary, and compact payload;
- model identity: the lifecycle checkpoint identity is derived from the
  compact checkpoint path/hash/model digest, and final Coach-gate validation
  requires the speed-row result to prove `candidate_loaded_checkpoint` identity
  for the same checkpoint/model;
- result row snapshot matching the manifest row;
- route and speed currency;
- row/result `profile_only=false`;
- row/result `touches_live_runs=false`;
- `calls_train_muzero` matching the route;
- no promotion, live-run, stock-resume, or rating-quality claims;
- denominator arithmetic with no fallback denominator and values bound back to
  result-summary fields;
- finite positive numbers.

Profile-only currencies such as `compact_profile_active_roots_per_sec` are
explicitly rejected for this required gate. A profile-only row tied to the
unified lifecycle candidate can still be useful support evidence, but it is
not the promotion-closing `coach_speed_row`.

There is now an explicit identity split:

```text
candidate_named_support_only:
  evidence names and hashes the lifecycle candidate, but the measured row does
  not prove it loaded that exact checkpoint/model.

candidate_loaded_checkpoint:
  evidence names the lifecycle candidate and the measured row proves the same
  loaded checkpoint id, trainer id, policy/model refs, policy source, checkpoint
  hash/path when available, and model-state digest.
```

Only `candidate_loaded_checkpoint` can close Coach compatibility. The support
scope is allowed for local producer hardening and shape checks, but it must fail
any `coach_speed_row=true` compatibility refresh.

## Current State

The unified lifecycle candidate remains:

```text
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json
```

It still reports:

```text
lifecycle_gates_complete=true
missing_required_gates=[coach_speed_row]
promotion_eligible=false
```

A first local CPU producer smoke exists:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-smoke-20260530/compact_coach_speed_row_smoke_report.json
```

It records `env_steps_collected=8.0`, `training_wall_sec=0.8652770410408266`,
and `compact_trainer_env_steps_per_sec=9.245593746919415`, with real local
env/search/replay/sample/compact-MuZero learner work and all non-claims false.
The raw support payload remains `profile_only=true`; this local CPU smoke is
support evidence only. Its evidence has
`model_identity.scope=candidate_named_support_only`. The
recorded lifecycle identity is useful and now includes the compact checkpoint
path, compact checkpoint SHA256, policy/model refs, trainer id, policy source,
learner update count, and model-state digest. The measured local result does
not include matching `result_loaded_checkpoint_identity`.

The accepted H100 threshold evidence is:

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-h100-b1024a16-threshold-20260530/row_001_result.json.compact_coach_speed_row.evidence.json
```

It records:

```text
model_identity.scope=candidate_loaded_checkpoint
env_steps_collected=184320.0
training_wall_sec=18.397932468
compact_trainer_env_steps_per_sec=10018.517043727199
uses_fallback_denominator=false
promotion_claim=false
```

The compatibility refresh that consumes it is:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json
```

That report has `coach_speed_row=true`, no missing current local compatibility
gates/evidence, and `promotion_eligible=true`, while preserving
`promotion_claim=false` and all live-run/stock-resume/rating-quality/
training-speedup non-claims.

## Validation

```text
uv run pytest tests/test_compact_coach_speed_row.py tests/test_compact_coach_compatibility.py -q
uv run pytest tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py tests/test_compact_eval_gif_tournament_load.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_coach_compatibility.py tests/test_compact_coach_speed_row.py -q
uv run ruff check src/curvyzero/training/compact_coach_speed_row.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_compatibility.py
uv run ruff check scripts/build_compact_coach_compatibility_speed_row_refresh.py tests/test_compact_coach_compatibility_speed_row_refresh.py
uv run pytest tests/test_compact_coach_compatibility.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_speed_row_smoke.py tests/test_compact_coach_compatibility_speed_row_refresh.py -q
```

Result:

```text
33 passed
71 passed, 2 warnings
ruff passed
refresh ruff passed
41 passed, 2 warnings
```

## Later Readiness State

The speed-row evidence gate is structurally closed. OPT-060 later bound
compatibility, lifecycle, matched quality, stock resume/load, isolated live-run
safety, sandbox assignment/rating, and longer-horizon metrics into a
no-promotion review packet. OPT-061 then chose
`require_larger_same_surface_study` for promotion readiness. OPT-068 later
produced the larger 32x2048 packet and accepted it only for a named
non-production manual review step, not promotion. OPT-062 later
produced the speed-row floor bundle:
`artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-20260530/compact_speed_row_floor_bundle_report.json`.
It keeps this evidence gate intact while adding same-denominator H100 siblings:
accepted device-target `10018.5`, compact Torch `5005.0`, and fixed-shape
floor `13759.0` compact trainer env steps/sec. The residual-aware engineering
read is `compact_rollout_slab_non_service_dominant`; the promotion P0 is now
the explicit manual review decision on the larger packet.
