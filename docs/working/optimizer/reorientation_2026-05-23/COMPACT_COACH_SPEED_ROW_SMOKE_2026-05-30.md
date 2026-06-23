# Compact Coach Speed-Row Smoke

Date: 2026-05-30

Status: local producer smoke passed; durable H100 loaded-checkpoint rows passed;
threshold row accepted and compatibility refresh written.
Role: record the first non-profile speed-row manifest/result/evidence artifact
for the unified lifecycle candidate.
Authority: artifact-specific record; `CURRENT_STATE.md` and `goal.md` remain
the current operating truth.

## What Landed

Added:

```text
scripts/build_compact_coach_speed_row_smoke.py
src/curvyzero/infra/modal/compact_coach_speed_row.py
scripts/run_compact_coach_speed_row_modal_smoke.py
```

The script is a sibling producer, not the hybrid profile manifest runner. It
runs the local env/search/replay/sample/compact-MuZero learner path, preserves
the raw profile payload as support evidence, and emits separate speed-row
manifest/result schemas:

```text
curvyzero_compact_coach_speed_row_manifest/v1
curvyzero_compact_coach_speed_row_result/v1
curvyzero_compact_coach_speed_row_evidence/v1
```

## Durable Local Artifact

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-smoke-20260530/compact_coach_speed_row_smoke_report.json
```

Headline:

```text
candidate_checkpoint_id=optimizer-compact-unified-lifecycle-smoke-20260530
speed_currency=compact_trainer_env_steps_per_sec
env_steps_collected=8.0
training_wall_sec=0.8652770410408266
steps_per_sec=9.245593746919415
real_compact_owned_training_work=true
profile_support_profile_only=true
calls_train_muzero=false
touches_live_runs=false
promotion_claim=false
model_identity_scope=candidate_named_support_only
```

Evidence ref:

```text
compact_coach_speed_row:optimizer-compact-unified-lifecycle-smoke-20260530:optimizer-compact-coach-speed-row-smoke-20260530:001:row=optimizer-compact-coach-speed-row-smoke-20260530/001:result_sha256=7cd019768076aa039de7ad56b250aa3243d9c2f2e90d13904c5803c0d8b8c127
```

## Important Caveat

This is a local CPU smoke and producer proof. It is not the final durable H100
Coach-comparable speed row, not a stock `train_muzero` run, not live-run safety,
not stock resume, not rating quality, and not a promotion claim.

The raw support payload remains `profile_only=true`, as it should. The
non-profile row is the sibling speed-row artifact binding measured compact-owned
training work to the already non-profile unified lifecycle candidate. Do not
relabel old hybrid profile rows as Coach rows.

The smoke is also intentionally support-only for model identity. Its evidence
records the lifecycle checkpoint path/hash/model digest, but the measured row
does not prove it loaded that exact checkpoint. Coach compatibility now requires
`model_identity.scope=candidate_loaded_checkpoint` before `coach_speed_row=true`
can pass.

## Durable H100 Row

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-h100-b128a8-20260530/compact_coach_speed_row_modal_report.json
```

Headline:

```text
candidate_checkpoint_id=optimizer-compact-unified-lifecycle-smoke-20260530
speed_currency=compact_trainer_env_steps_per_sec
env_steps_collected=2560.0
training_wall_sec=2.327328861
steps_per_sec=1099.973468682903
real_compact_owned_training_work=true
profile_support_profile_only=true
calls_train_muzero=false
touches_live_runs=false
promotion_claim=false
model_identity_scope=candidate_loaded_checkpoint
learner_device=cuda
resident_sample_used=true
device_replay_index_rows_sample=true
learner_observation_h2d_bytes=0
learner_input_h2d_bytes=0
```

This row loads the unified lifecycle compact checkpoint and matches checkpoint
id, trainer id, policy/model refs, policy source, checkpoint SHA, support
scale, and model-state digest. It is the first durable loaded-checkpoint row,
but it was superseded as the accepted threshold input by the row below.

## Accepted Threshold H100 Row

```text
artifacts/local/curvytron_compact_coach_speed_row_results/optimizer-compact-coach-speed-row-h100-b1024a16-threshold-20260530/compact_coach_speed_row_modal_report.json
```

Headline:

```text
candidate_checkpoint_id=optimizer-compact-unified-lifecycle-smoke-20260530
speed_currency=compact_trainer_env_steps_per_sec
shape=B1024/A16, measured_steps=180, warmup_steps=45
sample_gate=512 rows every 8 steps
env_steps_collected=184320.0
training_wall_sec=18.397932468
steps_per_sec=10018.517043727199
real_compact_owned_training_work=true
profile_support_profile_only=true
calls_train_muzero=false
touches_live_runs=false
promotion_claim=false
model_identity_scope=candidate_loaded_checkpoint
learner_device=cuda
resident_sample_used=true
device_replay_index_rows_sample=true
learner_observation_h2d_bytes=0
learner_input_h2d_bytes=0
materialized_timestep_count=0
learner_sample_rows=11264
sample_gate_rows=11264
```

Compatibility refresh:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json
coach_speed_row_gate=true
missing_required_gates=[]
missing_required_evidence=[]
promotion_eligible=true
promotion_claim=false
```

This closes the structural speed-row compatibility gate. It does not claim
stock `train_muzero`, live-run safety, stock resume, rating quality, training
speedup, or actual promotion.

## Validation

```text
uv run python scripts/build_compact_coach_speed_row_smoke.py --run-id optimizer-compact-coach-speed-row-smoke-20260530 --unified-lifecycle-report artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json --steps 4 --warmup-steps 1 --batch-size 2 --actor-count 1 --sample-batch-size 2 --sample-interval 1 --learner-device cpu
uv run pytest tests/test_compact_coach_speed_row.py tests/test_compact_coach_compatibility.py -q
uv run pytest tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py tests/test_compact_eval_gif_tournament_load.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_coach_compatibility.py tests/test_compact_coach_speed_row.py -q
uv run ruff check src/curvyzero/training/compact_coach_speed_row.py src/curvyzero/training/compact_coach_compatibility.py scripts/build_compact_coach_speed_row_smoke.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_compatibility.py
uv run pytest tests/test_compact_coach_speed_row.py tests/test_compact_coach_compatibility.py tests/test_compact_coach_speed_row_smoke.py -q
uv run pytest tests/test_compact_trainer_checkpoint.py tests/test_compact_owned_trainer.py tests/test_compact_eval_gif_tournament_load.py tests/test_compact_stock_checkpoint_export.py tests/test_compact_coach_compatibility.py tests/test_compact_coach_speed_row.py tests/test_compact_coach_speed_row_smoke.py -q
```

Result:

```text
local speed-row smoke passed
33 passed
71 passed, 2 warnings
ruff passed
H100 B128/A8 row passed
35 passed, 2 warnings
73 passed, 2 warnings
B1024/A16 threshold row passed
ruff passed on speed-row refresh script/tests
41 passed, 2 warnings
```

## Later Promotion-Readiness State

The real matched stock-vs-compact learning-quality canary later passed at
`artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-canary-current-env16train2-20260530/matched_learning_quality_canary_report.json`.
Local `promotion_eligible=true` is still not a promotion/live-run/stock-resume/
rating-quality claim; after the later isolated live safety canary and sandbox
assignment/rating proof, OPT-059 longer-horizon metrics, and OPT-060 final
bundle review, remaining work is matched-quality sufficiency/manual acceptance
before any actual promotion. OPT-062 later extended this speed-row producer
with backend selection and produced same-denominator H100 compact Torch and
fixed-shape floor siblings. The resulting floor bundle is
`artifacts/local/curvytron_compact_speed_row_floor_bundle_results/optimizer-compact-speed-row-floor-bundle-20260530/compact_speed_row_floor_bundle_report.json`;
it is optimizer engineering evidence only. The residual-aware read points next
at compact rollout slab non-service wall decomposition, with all promotion/
live/rating/speedup claims false.
