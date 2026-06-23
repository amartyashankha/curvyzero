# Compact Promotion Sandbox Assignment/Rating Proof

Date: 2026-05-30

Status: OPT-058 closed as local assignment/rating plumbing proof.

## Artifact

Report:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-sandbox-assignment-rating-proof-20260530a/sandbox_assignment_rating_proof_report.json
```

Producer manifest:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-sandbox-assignment-rating-proof-20260530a/sandbox_assignment_rating_proof_producer_manifest.json
```

Schema:

```text
curvyzero_compact_promotion_sandbox_assignment_rating_proof/v1
```

## What It Proves

The proof binds the current compatibility and lifecycle candidate through:

- unified lifecycle report;
- speed-row compatibility report;
- stock resume/load canary;
- isolated live-run safety canary;
- source compact checkpoint hash;
- resumed stock export hash;
- local rating snapshot;
- local leaderboard snapshot and pointer;
- stable assignment JSON plus assignment audit;
- forbidden-touch audit.

The local rating signal is deliberately tiny: one rated pair, five valid games,
and `max_abs_delta=1.6`. The leaderboard has two active local rows. The stable
assignment uses `stable_slots_v1`, has three entries, selects the candidate, and
passes `parse_opponent_assignment_snapshot` plus assignment-audit validation.

## What It Does Not Prove

This is not:

- a rating-quality claim;
- a public leaderboard publication;
- a production rating spawn;
- a training-candidate assignment pointer refresh;
- a live-run proof;
- a stock-resume claim;
- a speedup claim;
- a promotion claim.

The report keeps `promotion_readiness_complete=false`. Later OPT-059 closed the
named longer-horizon compact metrics lane, OPT-060 supplied the hash-bound
readiness-bundle review, OPT-061 supplied the original matched-quality
sufficiency decision, and OPT-068 produced the larger 32x2048 packet. The
remaining promotion boundary is now an explicit manual/policy decision on that
larger packet before any actual promotion.

## Command

```text
uv run python scripts/build_compact_promotion_sandbox_assignment_rating_proof_producer.py --run-id optimizer-compact-promotion-sandbox-assignment-rating-proof-20260530a --unified-lifecycle-report artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results/optimizer-compact-unified-lifecycle-smoke-20260530/unified_lifecycle_report.json --compatibility-report artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json --stock-resume-load-canary-report artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-stock-resume-load-canary-20260530/stock_resume_load_canary_report.json --isolated-live-run-safety-canary-report artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-isolated-live-run-safety-canary-20260530d/isolated_live_run_safety_canary_report.json
```

## Validation

```text
uv run ruff check src/curvyzero/training/compact_promotion_readiness.py scripts/build_compact_promotion_sandbox_assignment_rating_proof.py scripts/build_compact_promotion_sandbox_assignment_rating_proof_producer.py tests/test_compact_promotion_readiness.py
uv run pytest tests/test_compact_promotion_readiness.py tests/test_opponent_leaderboard.py tests/test_opponent_registry.py -q
```

Result:

```text
ruff=passed
tests=69 passed, 2 warnings
report_revalidation=passed
```
