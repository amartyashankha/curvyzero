# Compact Promotion Stock Resume/Load Canary

Date: 2026-05-30

Status: first post-compatibility readiness lane implemented and passed locally.

## Artifact

Report:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-stock-resume-load-canary-20260530/stock_resume_load_canary_report.json
```

Inputs:

```text
artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json
```

Produced:

```text
train/lightzero_exp/ckpt/iteration_0.pth.tar
train/lightzero_exp/ckpt/iteration_0.pth.tar.metadata.json
train/lightzero_exp/ckpt/iteration_0.pth.tar.evidence.json
verification_report.json
tournament_loader_report.json
```

## What It Proves

This canary starts from the compatibility-eligible compact candidate, reloads
the compact trainer checkpoint from disk, re-exports the loaded checkpoint as a
fresh stock-shaped payload under `model`, runs the stock LightZero strict-load
verifier, selects the export through the LightZero `iteration_*.pth.tar`
discovery shape, patches `policy.learn.learner.hook.load_ckpt_before_run` to the
selected checkpoint, and constructs the tournament loader from that resumed
export.

Headline:

```text
schema_id=curvyzero_compact_promotion_stock_resume_load_canary/v1
ok=true
readiness_lane=stock_resume_load_canary
candidate_checkpoint_id=optimizer-compact-unified-lifecycle-smoke-20260530
stock_resume_load_canary=true
strict_stock_model_load_verified_after_compact_checkpoint_reload=true
tournament_loader_constructed_after_compact_checkpoint_reload=true
promotion_readiness_complete=false
```

Stock resume selection:

```text
selection_policy=latest_lightzero_iteration_checkpoint_from_dirs
selected_checkpoint_name=iteration_0.pth.tar
selected_iteration=0
selected_checkpoint_path=.../train/lightzero_exp/ckpt/iteration_0.pth.tar
load_ckpt_before_run_patch.path=policy.learn.learner.hook.load_ckpt_before_run
resume_state_found=false
```

Loaded compact checkpoint identity:

```text
checkpoint_id=optimizer-compact-unified-lifecycle-smoke-20260530
trainer_id=optimizer-compact-unified-lifecycle-smoke-20260530:compact-current-chain-trainer
policy_version_ref=optimizer-compact-unified-lifecycle-smoke-20260530:policy-update-1
model_version_ref=optimizer-compact-unified-lifecycle-smoke-20260530:model-update-1
policy_source=compact_current_chain_eval_gif_tournament_smoke
learner_update_count=1
sample_batch_count=1
model_state_digest=37b7321c054d433e063dbc95d3a233b33eed83b98d12bcb3cc46034989e40d3a
```

Strict-load and loader summary:

```text
verification ok=true
strict_stock_model_load_verified=true
state_key=model
tournament_loader ok=true
checkpoint_state_key=model
eval_loader_ok=true
policy_observation_backend=cpu_oracle
```

## Non-Claims

This closes exactly one readiness lane. It is not a full promotion readiness
bundle and not a stock trainer resume claim.

Kept false:

```text
promotion_claim=false
stock_resume_claim=false
stock_training_resume_claim=false
training_speedup_claim=false
live_run_safety_claim=false
rating_or_promotion_quality_claim=false
touches_live_runs=false
calls_train_muzero=false
```

Later status: isolated live-run safety, sandbox assignment/rating proof, and
OPT-059 longer-horizon compact metrics have also passed. OPT-060 then produced
the final hash-bound readiness-bundle review. Still remaining before any actual
promotion is manual acceptance or a stronger matched-quality study.

The matched stock-vs-compact learning-quality canary passed after this stock
resume/load document was written. It is recorded at
`artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-canary-current-env16train2-20260530/matched_learning_quality_canary_report.json`.

## Code

Added:

```text
src/curvyzero/training/compact_promotion_readiness.py
scripts/build_compact_promotion_stock_resume_load_canary.py
tests/test_compact_promotion_readiness.py
```

The builder validates that the input compatibility report is locally eligible
but not claimed, verifies the current-chain evidence, reloads the compact
checkpoint, checks identity against the current-chain candidate, re-exports
from the loaded checkpoint, validates stock checkpoint selection and
`load_ckpt_before_run` patching, and hash-binds every produced file in the
canary report.

## Validation

```text
uv run ruff check src/curvyzero/training/compact_promotion_readiness.py scripts/build_compact_promotion_stock_resume_load_canary.py tests/test_compact_promotion_readiness.py src/curvyzero/training/compact_coach_compatibility.py tests/test_compact_coach_compatibility.py scripts/build_compact_coach_compatibility_speed_row_refresh.py tests/test_compact_coach_compatibility_speed_row_refresh.py
uv run pytest tests/test_compact_promotion_readiness.py tests/test_compact_coach_compatibility.py tests/test_compact_coach_compatibility_speed_row_refresh.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_speed_row_smoke.py -q
```

Result:

```text
ruff passed
47 passed, 2 warnings
```
