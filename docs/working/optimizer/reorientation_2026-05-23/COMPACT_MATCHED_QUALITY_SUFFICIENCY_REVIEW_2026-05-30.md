# Compact Matched-Quality Sufficiency Review

Date: 2026-05-30

Status: OPT-061 complete. OPT-068 larger 32x2048 execution and refreshed
sufficiency are complete. This is still a no-promotion decision artifact.

## Result

The matched-quality sufficiency review now exists:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-quality-sufficiency-review-20260530a/matched_quality_sufficiency_review_report.json
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-quality-sufficiency-review-20260530a/manifest.json
```

It validates under:

```text
curvyzero_compact_matched_quality_sufficiency_review/v1
```

The decision is:

```text
status=larger_same_surface_study_required
matched_quality_sufficiency_decision=require_larger_same_surface_study
promotion_claim=false
automatic_promotion_allowed=false
current_evidence_sufficient_for_promotion=false
manual_review_required_before_any_promotion=true
evidence_ref=compact_matched_quality_sufficiency_review:optimizer-compact-unified-lifecycle-smoke-20260530:13855b2ddddd14c6
```

## Why

OPT-060 made a valid manual-review packet, but it did not make the current
matched-quality canary promotion-sufficient. The sufficiency review hash-binds
the OPT-060 bundle and acknowledges the current evidence limits:

- `eval_seed_count=8`
- `eval_max_steps=1024`
- `hardware_class=mixed`
- stock side `modal-gpu-l4-t4-cpu40`
- compact side `local-cpu-producer-smoke`
- compact-minus-stock mean-survival delta `+5.625`
- longer-horizon compact trace has three checkpoints and only two learner
  update deltas

Because there is no explicit external human acceptance attached for a named
non-production next step, the artifact chooses the larger-study branch.

## Larger Study Plan

The attached same-surface study plan is intentionally a plan, not evidence. It
requires fresh outputs before any promotion decision:

```text
study_id=optimizer-compact-matched-learning-quality-larger-2048x32-env64train8-20260531
min_eval_seed_count=32
min_eval_max_steps=2048
stock_reference_min_max_env_step=2048
stock_reference_min_max_train_iter=4
compact_candidate_min_env_steps=64
compact_candidate_min_train_steps=8
```

Guardrails:

- same eval surface required;
- fresh outputs required;
- no fallback denominators;
- no preview captures;
- no profile-only speed currency;
- no live runs;
- compact candidate must keep `calls_train_muzero=false`;
- final readiness bundle must be refreshed before the next sufficiency review.

The read-only plan artifacts now exist:

```text
artifacts/local/curvytron_compact_matched_quality_study_plan_results/optimizer-compact-matched-quality-larger-study-plan-20260531/larger_study_plan_report.json
artifacts/local/curvytron_compact_matched_quality_study_plan_results/optimizer-compact-matched-quality-larger-study-plan-20260531/manifest.json
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-quality-sufficiency-review-20260531a/matched_quality_sufficiency_review_report.json
artifacts/local/curvytron_compact_matched_quality_study_bundle_results/optimizer-compact-matched-quality-larger-study-bundle-canonical-20260531/larger_study_bundle_report.json
artifacts/local/curvytron_compact_matched_quality_study_bundle_results/optimizer-compact-matched-quality-larger-study-bundle-canonical-20260531/manifest.json
artifacts/local/curvytron_compact_matched_quality_study_preflight_results/optimizer-compact-matched-quality-larger-study-preflight-20260531/larger_study_preflight_report.json
artifacts/local/curvytron_compact_matched_quality_study_preflight_results/optimizer-compact-matched-quality-larger-study-preflight-20260531/manifest.json
```

Historical plan/preflight status:

```text
status=larger_matched_quality_study_required
plan_manifest_emitted=true
plan_only_not_evidence=true
preflight_only=true
runner_manifest_only=true
fresh_outputs_produced=false
stock_capture_produced=false
compact_capture_produced=false
matched_canary_refreshed=false
matched_pair_verification_refreshed=false
readiness_bundle_refreshed=false
sufficiency_review_refreshed=false
quality_evidence=false
promotion_claim=false
automatic_promotion_allowed=false
current_evidence_sufficient_for_promotion=false
stock_train_muzero_speedup_claim=false
```

The standalone plan and canonical bundle now agree on the fresh `20260531`
future run ids. The canonical bundle is bound to the refreshed 2026-05-31
sufficiency review's embedded larger-study plan and records the ordered future
execution steps plus post-step validators. The readiness-bundle refresh command
explicitly carries compatibility, lifecycle, stock resume/load, isolated
live-run, sandbox assignment/rating, longer-horizon, matched-quality, and pair
reports; preflight rejects implicit default input fallbacks. The preflight says
only `stock_reference_capture_producer` and
`compact_candidate_capture_producer` are executable now. No artifact in this
set runs stock/compact captures or refreshes quality evidence.

## Larger Study Execution

The larger 32x2048 packet now exists and supersedes the plan-only status above
as the current quality-evidence frontier:

```text
compact_capture=artifacts/local/curvytron_compact_matched_learning_quality_results/optimizer-compact-candidate-env-search-replay-larger-2048x32-env64train8-20260531/compact_candidate_capture.json
stock_capture=artifacts/local/curvytron_compact_matched_learning_quality_results/optimizer-stock-reference-quality-producer-larger-2048x32-20260531-evalenv32/stock_reference_capture.json
matched_canary=artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-larger-2048x32-env64train8-20260531-canary/matched_learning_quality_canary_report.json
pair_verifier=artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-larger-2048x32-env64train8-20260531-pair-verifier/matched_pair_verification_report.json
readiness_bundle=artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-readiness-bundle-review-larger-2048x32-env64train8-20260531/readiness_bundle_review_report.json
sufficiency_review=artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-quality-sufficiency-review-larger-2048x32-env64train8-20260531/matched_quality_sufficiency_review_report.json
```

Result:

```text
eval_seed_count=32
eval_max_steps=2048
stock_delta=+5.0625
compact_delta=+5.4375
compact_minus_stock_delta=+0.375
status=current_canary_accepted_for_next_non_production_step
next_non_production_step=larger_32x2048_matched_quality_manual_review_not_promotion
current_evidence_sufficient_for_promotion=false
promotion_claim=false
automatic_promotion_allowed=false
```

Important caveat: canonical stock
`optimizer-stock-reference-quality-producer-larger-2048x32-20260531` failed
closed before emitting a capture because the LightZero evaluator hit
`IndexError: tuple index out of range` after the initial 32 evaluator episodes
with `evaluator_env_num=2`. The accepted stock capture is the explicit
`evalenv32` rerun, which keeps the matched 32-episode eval surface but avoids
that evaluator edge.

## Code And Tests

Code:

```text
src/curvyzero/training/compact_matched_quality_sufficiency_review.py
src/curvyzero/training/compact_matched_quality_study_plan.py
src/curvyzero/training/compact_matched_quality_larger_study_bundle.py
src/curvyzero/training/compact_matched_quality_larger_study_preflight.py
scripts/build_compact_matched_quality_sufficiency_review.py
scripts/build_compact_matched_quality_larger_study_plan.py
scripts/build_compact_matched_quality_larger_study_bundle.py
scripts/build_compact_matched_quality_larger_study_preflight.py
tests/test_compact_matched_quality_sufficiency_review.py
tests/test_compact_matched_quality_study_plan.py
tests/test_compact_matched_quality_larger_study_bundle.py
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_matched_quality_sufficiency_review.py scripts/build_compact_matched_quality_sufficiency_review.py tests/test_compact_matched_quality_sufficiency_review.py
uv run pytest tests/test_compact_matched_quality_sufficiency_review.py -q
uv run python scripts/build_compact_matched_quality_sufficiency_review.py
uv run ruff check src/curvyzero/training/compact_matched_quality_study_plan.py src/curvyzero/training/compact_matched_quality_larger_study_bundle.py src/curvyzero/training/compact_matched_quality_larger_study_preflight.py scripts/build_compact_matched_quality_larger_study_plan.py scripts/build_compact_matched_quality_larger_study_bundle.py scripts/build_compact_matched_quality_larger_study_preflight.py tests/test_compact_matched_quality_study_plan.py tests/test_compact_matched_quality_larger_study_bundle.py
uv run pytest tests/test_compact_matched_quality_sufficiency_review.py tests/test_compact_matched_quality_study_plan.py tests/test_compact_matched_quality_larger_study_bundle.py tests/test_compact_promotion_readiness_bundle_review.py -q
```

Focused sufficiency tests passed: `7 passed, 2 warnings`. Focused
matched-quality plan/bundle/preflight/readiness slice passed:
`26 passed, 2 warnings`.

## Non-Claims

This artifact does not claim:

- compact promotion;
- automatic promotion;
- compact quality superiority;
- live-run safety;
- production live-run safety;
- stock training resume;
- rating or public leaderboard quality;
- training speedup.

## Next

For promotion readiness, manually review the larger 32x2048 packet and decide
whether it supports a named non-production promotion proposal, requires a
larger repeat, or leaves compact candidate-only.

For optimizer engineering, the next smallest honest performance artifact is a
speed-row floor bundle: compare the accepted H100 Coach speed row with real
`CompactTorchSearchServiceV1` and fixed/no-search floor siblings before
starting GPU game-mechanics work.
