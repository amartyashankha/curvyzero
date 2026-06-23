# Compact Promotion Readiness Bundle Review - 2026-05-30

Purpose: final post-compatibility review packet for the current compact
candidate. This artifact assembles and hash-binds the readiness evidence that
already exists. It does not publish, promote, rewrite production pointers, claim
compact superiority, or allow automatic promotion.

## Durable Artifact

Report:
`artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-readiness-bundle-review-20260530a/readiness_bundle_review_report.json`

Manifest:
`artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-readiness-bundle-review-20260530a/manifest.json`

Schema:
`curvyzero_compact_promotion_readiness_bundle_review/v1`

Candidate:
`optimizer-compact-unified-lifecycle-smoke-20260530`

Evidence ref:
`compact_promotion_readiness_bundle_review:optimizer-compact-unified-lifecycle-smoke-20260530:4e2675c2ccd9bf9d`

## Bound Inputs

- Compatibility refresh:
  `artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json`
- Unified lifecycle:
  `artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json`
- Matched learning-quality canary:
  `artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-canary-current-env16train2-20260530/matched_learning_quality_canary_report.json`
- Matched pair verifier:
  `artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-pair-verifier-current-env16train2-20260530/matched_pair_verification_report.json`
- Stock resume/load canary:
  `artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-stock-resume-load-canary-20260530/stock_resume_load_canary_report.json`
- Isolated live-run safety canary:
  `artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-isolated-live-run-safety-canary-20260530d/isolated_live_run_safety_canary_report.json`
- Sandbox assignment/rating proof:
  `artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-sandbox-assignment-rating-proof-20260530a/sandbox_assignment_rating_proof_report.json`
- Longer-horizon compact metrics:
  `artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-longer-horizon-learning-metrics-local-20260530a/longer_horizon_learning_metrics_report.json`

The validator re-runs each lane validator, checks top-level hashes, checks the
matched verifier points to the matched canary, checks stock-resume hashes are
bound into isolated/sandbox reports, and checks the shared compact checkpoint
hash across compatibility, matched compact, stock resume, sandbox proof, and
longer-horizon metrics.

## Decision

- `status=local_bundle_reviewed_no_promotion`
- `ready_for_manual_review=true`
- `promotion_claim=false`
- `automatic_promotion_allowed=false`
- `manual_review_required_before_any_promotion=true`
- `matched_quality_sufficiency_decision=canary_scale_manual_acceptance_or_larger_study_required_before_promotion`

The current matched-quality evidence is still canary-scale: eight 1024-step eval
seeds, mixed hardware, stock Modal `train_muzero` reference, and local compact
env/search/replay candidate. The longer-horizon compact trace is also local and
tiny: three checkpoints and two learner updates. Together they are enough for a
manual review packet, not a compact superiority or production promotion claim.

Later OPT-061 consumed this packet and made the sufficiency branch explicit:
without external human acceptance for a named non-production step, the current
canary is not promotion-sufficient and a larger same-surface matched-quality
study is required before any promotion claim.

Later OPT-068 produced that larger 32x2048 packet and refreshed this bundle
shape under:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-readiness-bundle-review-larger-2048x32-env64train8-20260531/readiness_bundle_review_report.json
```

The refreshed sufficiency review accepts the larger packet only for
`larger_32x2048_matched_quality_manual_review_not_promotion`; promotion and
automatic promotion remain false.

## Validation

Focused validation:

```bash
uv run ruff check src/curvyzero/training/compact_promotion_readiness_bundle_review.py scripts/build_compact_promotion_readiness_bundle_review.py tests/test_compact_promotion_readiness_bundle_review.py
uv run pytest tests/test_compact_promotion_readiness_bundle_review.py -q
```

Result: ruff passed; `6 passed, 2 warnings`.

Real artifact build:

```bash
uv run python scripts/build_compact_promotion_readiness_bundle_review.py --overwrite
```

Result: report and manifest emitted under
`optimizer-compact-promotion-readiness-bundle-review-20260530a`; the report
validated after build.
