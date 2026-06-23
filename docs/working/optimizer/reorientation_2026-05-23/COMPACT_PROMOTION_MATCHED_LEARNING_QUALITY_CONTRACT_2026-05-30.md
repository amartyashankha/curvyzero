# Compact Promotion Matched Learning-Quality Contract

Date: 2026-05-30

Status: local contract/schema validator, provenance-required and artifact-bound
capture schema, preview-only source inventory helper, strict capture-from-
artifacts packager, capture-derived report builder, source-fingerprint
hardening, artifact-content validation, minimum seed/horizon evidence, and
saturated-eval rejection implemented. First real two-arm 1024x8 matched canary
validated on 2026-05-30.

## Scope

This is a post-compatibility readiness contract for this readiness lane:

```text
matched_stock_vs_compact_learning_quality
```

It prevents old profile/speed rows from being reused as learning-quality
evidence. The old 2026-05-28 matched denominator pair is still useful speed
orientation, but it has no matched eval movement and does not satisfy this
contract. The accepted 2026-05-30 H100 speed row also stays speed-row evidence,
not quality evidence.

Current validated canary:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-canary-current-env16train2-20260530/matched_learning_quality_canary_report.json
```

It closes the matched-learning-quality canary lane, but not promotion readiness.
Later on 2026-05-30 the isolated live-run safety canary, sandbox
assignment/rating proof, OPT-059 longer-horizon compact metrics, and OPT-060
readiness-bundle review also landed. The final review keeps promotion false and
requires manual acceptance or a larger/more durable matched-quality study before
any actual promotion.

## Code

Added:

```text
src/curvyzero/training/compact_promotion_readiness_learning_quality.py
scripts/build_compact_matched_learning_quality_canary.py
scripts/build_compact_matched_learning_quality_capture_preview.py
scripts/build_compact_matched_learning_quality_capture_from_artifacts.py
tests/test_compact_promotion_readiness_learning_quality.py
tests/test_compact_matched_learning_quality_canary_builder.py
tests/test_compact_matched_learning_quality_capture_from_artifacts.py
```

Schema:

```text
curvyzero_compact_promotion_matched_learning_quality_canary/v1
```

Arm schema:

```text
curvyzero_compact_matched_learning_quality_arm/v1
```

Raw capture schema:

```text
curvyzero_compact_matched_learning_quality_capture/v1
```

Capture provenance schema:

```text
curvyzero_compact_matched_learning_quality_capture_provenance/v1
```

Support-only preview schema:

```text
curvyzero_compact_matched_learning_quality_capture_preview/v1
```

The validator requires a hash-bound compatibility report and unified lifecycle
report, exactly two arms, route-scoped `train_muzero` labels, matched eval
settings, semantic matched source fingerprints, visible denominators, at least
two eval points per arm, at least two fixed eval seeds, `eval_max_steps >= 128`,
and no saturated eval curves.
Builder-eligible raw captures must also carry `capture_provenance` proving that
the producer ran training, ran pre-eval, ran post-eval, is not support-only, and
is allowed to feed the builder. Role-specific provenance must say stock called
`train_muzero`, while compact-owned training did not call `train_muzero` and did
perform real compact-owned training work. The provenance refs are not free-text
claims: they must match hashed capture `artifact_refs` by kind or path.

Builder-eligible captures must include hashed `artifact_refs` for:

```text
training_artifact
pre_eval_summary
post_eval_summary
initial_checkpoint
final_checkpoint
```

The report builder records SHA256s for the raw `stock_reference_capture` and
`compact_candidate_capture` files; validation fails if either input capture file
drifts after report assembly.

Artifact hashes are no longer enough by themselves. The validator reads the
referenced `training_artifact`, `pre_eval_summary`, and `post_eval_summary`
JSON files. Training artifacts must match role schema, route, run id,
profile/support/live-run non-claims, model digests, stock-vs-compact call
contract, and checkpoint refs/paths/hashes. Eval summaries must normalize back
to the capture eval points and match the fixed seed set plus eval horizon.
Initial and final checkpoint artifact hashes must differ.

The report-builder CLI consumes externally produced raw capture JSON files,
derives validator arms, then validates and writes the matched canary report:

```text
scripts/build_compact_matched_learning_quality_canary.py \
  --compatibility-report <compatibility_report.json> \
  --unified-lifecycle-report <unified_lifecycle_report.json> \
  --stock-reference-capture <stock_reference_capture.json> \
  --compact-candidate-capture <compact_candidate_capture.json>
```

Builder outputs:

```text
manifest.json
stock_reference_arm.json
compact_candidate_arm.json
matched_learning_quality_canary_report.json
```

The `*_arm.json` files are derived validator artifacts. They are not the public
evidence source and should not be hand-authored as the main path.

## Support-Only Capture Preview

A helper that only inspects existing artifacts must emit:

```text
curvyzero_compact_matched_learning_quality_capture_preview/v1
usable_as_quality_capture=false
feeds_builder=false
support_only=true
```

Preview outputs must not be named `stock_reference_capture.json` or
`compact_candidate_capture.json`, must not write
`matched_learning_quality_canary_report.json`, and must not close OPT-051. The
current helper is:

```text
scripts/build_compact_matched_learning_quality_capture_preview.py
```

It normalizes source eval summaries into preview points and hashes source files,
but it does not run stock `train_muzero`, does not run compact-owned training,
does not run matched pre/post eval, and cannot feed the matched-quality builder.

## Builder-Eligible Capture From Artifacts

The real capture packager is:

```text
scripts/build_compact_matched_learning_quality_capture_from_artifacts.py
```

It writes exactly one builder-eligible capture file:

```text
stock_reference_capture.json
compact_candidate_capture.json
```

This script is deliberately not the training/eval producer. It only packages
already-produced source/model/eval/denominator/provenance JSON plus five
required artifact files into
`curvyzero_compact_matched_learning_quality_capture/v1`, validates the derived
arm, and writes the capture. It is valid only when its inputs came from an
upstream real stock `train_muzero` run or upstream real compact-owned training
run with matched pre/post eval.

Required artifact kinds:

```text
training_artifact
pre_eval_summary
post_eval_summary
initial_checkpoint
final_checkpoint
```

Forbidden outputs:

```text
stock_reference_capture_preview.json
compact_candidate_capture_preview.json
stock_reference_arm.json
compact_candidate_arm.json
matched_learning_quality_canary_report.json
```

The `--output` filename must agree with the role. A stock run cannot write
`compact_candidate_capture.json`; a compact run cannot write
`stock_reference_capture.json`; neither role may use preview filenames. The
builder report remains a separate step.

## Required Arm Shape

Stock reference:

```text
role=stock_reference
route=stock_train_muzero_reference
calls_train_muzero=true
profile_only=false
denominator_currency=stock_train_muzero_learning_quality
env_step_currency=stock_train_muzero_raw_env_steps
uses_fallback_denominator=false
```

Compact candidate:

```text
role=compact_candidate
route=compact_owned_trainer
calls_train_muzero=false
profile_only=false
real_compact_owned_training_work=true
model_identity_scope=candidate_loaded_checkpoint
denominator_currency=compact_owned_learning_quality
env_step_currency=compact_owned_trainer_env_steps
uses_fallback_denominator=false
```

Both arms must share the same:

```text
denominator_id
quality_horizon
git_commit
matched_surface
observation_schema_id
policy_observation_backend
eval_seed_set
eval_episode_count
source_max_steps
eval_max_steps
num_simulations
batch_size
reward_variant
reward_target_effect
death_mode=normal
terminal_target_mode
root_noise
dirichlet_alpha
policy_noise
rnd_enabled=false
exploration_bonus_mode
opponent_policy_ref
opponent_policy_kind
opponent_runtime_mode
opponent_death_mode
natural_bonus_spawn
training_seed_policy
initialization_source
num_unroll_steps
td_steps
discount
support_scale
```

Hardware class, initialization source, and training seed policy remain
arm-scoped. Stock may be a fresh Modal `train_muzero` run while compact may be a
local loaded-candidate compact run; the matched report records
`stock_hardware_class` and `compact_hardware_class` separately and marks the
pair `hardware_class=mixed` when they differ.

## Quality Movement

The primary metric is:

```text
mean_survival_delta = final_mean_survival - initial_mean_survival
```

The validator computes:

```text
stock_reference_delta
compact_candidate_delta
compact_minus_stock_delta
```

It requires finite metrics, monotonic checkpoint steps, `pre_train` and
`post_train` eval points, nonzero observed movement, `cap_rate <= 0.5` on each
point, model-state digest changes for both arms, and existing artifact refs
with matching SHA256 hashes and matching contents.

## Non-Claims

This contract does not close the readiness lane by itself. It validates the
shape a real producer must emit.

Kept false:

```text
promotion_claim=false
promotion_readiness_complete=false
training_speedup_claim=false
live_run_safety_claim=false
stock_resume_claim=false
stock_training_resume_claim=false
rating_or_promotion_quality_claim=false
compact_quality_superiority_claim=false
leaderboard_claim=false
touches_live_runs=false
compact_calls_train_muzero=false
```

The 1024x8 env16/train2 matched learning-quality canary now passes. Do not
inflate that into promotion, speedup, live-run safety, stock resume,
rating-quality, compact-quality-superiority, or leaderboard claims.

## 2026-05-30 Validated Canary

Report:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-canary-current-env16train2-20260530/matched_learning_quality_canary_report.json
```

Inputs:

```text
artifacts/local/curvytron_compact_matched_learning_quality_results/optimizer-stock-reference-quality-producer-modal-current-diagnostic1024x8-20260530/matched_1024x8_stock_reference_capture/stock_reference_capture.json
artifacts/local/curvytron_compact_matched_learning_quality_results/optimizer-compact-candidate-env-search-replay-local-gate-1024x8-env16train2-20260530/compact_candidate_capture.json
```

Headline:

```text
ok=true
status=matched_learning_quality_movement_observed
denominator_id=matched_learning_quality_current_1024x8_denominator_v1
quality_horizon=matched_learning_quality_pre_post_eval_1024x8_current
hardware_class=mixed
stock_hardware_class=modal-gpu-l4-t4-cpu40
compact_hardware_class=local-cpu-producer-smoke
stock_mean_survival=134.0 -> 133.125
compact_mean_survival=141.5 -> 146.25
compact_minus_stock_delta=5.625
```

The stock arm ran real Modal `train_muzero`, evaluated eight fixed seeds at
`eval_max_steps=1024`, and used denominators `collector_envstep_delta=533`,
`learner_train_calls=133`, `replay_sample_calls=133`, and
`training_wall_sec=73.74316`. The compact arm used env/search/replay rows,
`CompactOwnedTrainerV1.record_step`, eight fixed seeds at `eval_max_steps=1024`,
and denominators `compact_rollout_rows=64`, `compact_sample_rows=30`,
`learner_update_count_delta=30`, `sample_batch_count_delta=16`, and
`training_wall_sec=0.47014087717980146`.

## Validation

```text
uv run ruff check src/curvyzero/training/compact_promotion_readiness_learning_quality.py tests/test_compact_promotion_readiness_learning_quality.py
uv run pytest tests/test_compact_promotion_readiness_learning_quality.py -q
uv run ruff check src/curvyzero/training/compact_promotion_readiness_learning_quality.py scripts/build_compact_matched_learning_quality_canary.py tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py
uv run pytest tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py -q
uv run ruff check src/curvyzero/training/compact_promotion_readiness_learning_quality.py scripts/build_compact_matched_learning_quality_canary.py scripts/build_compact_matched_learning_quality_capture_preview.py tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py
uv run pytest tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py -q
uv run ruff check src/curvyzero/training/compact_promotion_readiness_learning_quality.py scripts/build_compact_matched_learning_quality_canary.py scripts/build_compact_matched_learning_quality_capture_preview.py scripts/build_compact_matched_learning_quality_capture_from_artifacts.py tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_matched_learning_quality_capture_from_artifacts.py tests/test_compact_promotion_readiness_learning_quality.py
uv run pytest tests/test_compact_matched_learning_quality_capture_from_artifacts.py tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py -q
uv run pytest tests/test_compact_matched_learning_quality_capture_from_artifacts.py tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py tests/test_compact_promotion_readiness.py tests/test_compact_coach_compatibility.py tests/test_compact_coach_compatibility_speed_row_refresh.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_speed_row_smoke.py -q
uv run ruff check scripts/build_compact_matched_learning_quality_stock_reference_producer.py scripts/build_compact_matched_learning_quality_compact_candidate_producer.py scripts/build_compact_matched_learning_quality_capture_from_artifacts.py scripts/build_compact_matched_learning_quality_canary.py tests/test_compact_matched_learning_quality_stock_reference_producer.py tests/test_compact_matched_learning_quality_compact_candidate_producer.py tests/test_compact_matched_learning_quality_capture_from_artifacts.py tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py
uv run pytest tests/test_compact_matched_learning_quality_stock_reference_producer.py tests/test_compact_matched_learning_quality_compact_candidate_producer.py tests/test_compact_matched_learning_quality_capture_from_artifacts.py tests/test_compact_matched_learning_quality_canary_builder.py tests/test_compact_promotion_readiness_learning_quality.py -q
```

Result:

```text
ruff passed
14 passed
17 passed
22 passed
25 passed
51 passed
98 passed, 2 warnings
compact-candidate producer tests 4 passed
producer+packager+builder+contract slice 55 passed, 2 warnings
broader readiness-adjacent slice 116 passed, 2 warnings
stock+compact producer tests 18 passed, 2 warnings
stock producer CLI --help passed
touched matched-quality ruff passed
focused matched-quality slice 85 passed, 2 warnings
compact env/search/replay producer core test passed
focused matched-quality producer/packager/builder slice 87 passed, 2 warnings
final matched-quality producer/packager/builder slice 96 passed, 2 warnings
```

## Compact-Candidate Producer

`scripts/build_compact_matched_learning_quality_compact_candidate_producer.py`
now has a default compact env/search/replay training mode. It loads the unified
lifecycle compact checkpoint, restores a stock-LightZero-shaped model/optimizer,
steps `HybridBatchedObservationProfileManager`, uses `CompactRolloutSlab` plus
`CompactTorchSearchServiceV1`, stores device replay rows in `_CompactReplayRingV1`,
trains through `CompactOwnedLoopV1`, and advances the trainer through
`CompactOwnedTrainerV1.record_step(...)`. It then exports pre/post stock-shaped
`iteration_0.pth.tar` and `iteration_1.pth.tar`, runs matched standalone eval
hooks, and writes the strict packager inputs.

The deterministic resident-sample path remains only as an explicit scaffold
mode. The strict compact quality gate rejects it as final evidence: a
builder-eligible compact training artifact must use
`sample_source=compact_env_search_replay_rows`, must name
`CompactOwnedTrainerV1.record_step`, must set `synthetic_resident_sample=false`,
and must set `real_env_search_replay_rows=true`. Previous resident-sample local
attempts against the unified lifecycle checkpoint failed closed at the packager
because pre/post eval both capped with identical mean survival.
The less-toy failed run is:
`artifacts/local/curvytron_compact_matched_learning_quality_results/optimizer-compact-candidate-quality-producer-local-eval200-20260530/`.
It contains pre/post exports, eval summaries, and packager inputs, but no
builder-eligible compact capture. The env/search/replay mode has since been run
against the current unified lifecycle checkpoint and emitted
`artifacts/local/curvytron_compact_matched_learning_quality_results/optimizer-compact-candidate-env-search-replay-local-gate-20260530a/compact_candidate_capture.json`.
That compact capture is builder-eligible canary evidence with
`sample_source=compact_env_search_replay_rows`, `real_compact_owned_training_work=true`,
two fixed eval seeds, `eval_max_steps=128`, cap rate `0.0`, and pre/post mean
survival `117.5 -> 100.0`; it was compact-only evidence and is superseded for
matched readiness by the 1024x8 env16/train2 capture:
`artifacts/local/curvytron_compact_matched_learning_quality_results/optimizer-compact-candidate-env-search-replay-local-gate-1024x8-env16train2-20260530/compact_candidate_capture.json`.
The superseding compact capture used eight fixed seeds, `eval_max_steps=1024`,
and moved mean survival `141.5 -> 146.25`.

## Residual Fake-Proof Surfaces

- The direct arm-building function remains a structural/unit-test helper; the
  public evidence path is capture files through
  `scripts/build_compact_matched_learning_quality_canary.py`.
- The current positive movement proves a canary shape, not promotion-quality
  superiority. Larger rows should use fixed seeds and enough episodes/checkpoint
  horizon to make the quality read meaningful.
- Eval settings are matched by declared config plus inspected eval-summary
  contents. A future producer should add per-seed/per-episode summary hashes
  once the real eval artifacts exist.
- `scripts/build_compact_matched_learning_quality_capture_from_artifacts.py`
  is a strict packager only. Calling it on fixture JSON proves the boundary,
  not training. The stock 1024x8 matched copy was repackaged from already
  produced stock artifacts.
- `scripts/build_compact_matched_learning_quality_compact_candidate_producer.py`
  now has a real env/search/replay training mode and one builder-eligible local
  compact capture. Future capped/no-movement runs must still fail closed and
  must not be renamed into final matched evidence.

## Current P0

Do not keep grinding this exact missing-stock loop. The matched-quality canary,
isolated live-run safety canary, sandbox assignment/rating proof, OPT-059
longer-horizon compact metrics, and OPT-060 final bundle review now exist. The
next P0 question is whether manual review accepts the current canary-scale
quality evidence for a non-production next step, or asks for a larger/more
durable matched-quality study.

If this matched-quality study is rerun, keep the same rules: fresh output
directories, fixed eval seeds, matched reward/death/noise/RND/horizon settings,
real pre/post eval summaries, visible stock/compact denominators,
builder-eligible `capture_provenance`, mixed-hardware honesty, and all
promotion/speed/live/resume/rating/leaderboard non-claims false.
