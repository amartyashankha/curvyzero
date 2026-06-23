# Compact Promotion Readiness Gate

Date: 2026-05-30

Status: selected as the post-compatibility gate.

Update: the first readiness lane, stock resume/load canary, is now implemented
and passed locally. See
`COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_2026-05-30.md`. This does not
complete promotion readiness.

Update: the matched learning-quality contract/tooling lane now includes the
validator, capture provenance requirements, preview-only source inventory, and
capture-derived report builder. See
`COMPACT_PROMOTION_MATCHED_LEARNING_QUALITY_CONTRACT_2026-05-30.md`.

Later update: the matched learning-quality canary passed at
`artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-matched-learning-quality-canary-current-env16train2-20260530/matched_learning_quality_canary_report.json`.
The isolated live-run safety canary and sandbox assignment/rating proof have
now passed too. OPT-059 then added longer-horizon compact metrics, and OPT-060
added the final hash-bound readiness-bundle review. The review is ready for
manual review, but still records `promotion_claim=false` and
`automatic_promotion_allowed=false`.

## Current Input

The local compact Coach compatibility refresh is:

```text
artifacts/local/curvytron_compact_coach_compatibility_results/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530/compatibility_report.json
```

Read it as local compatibility-eligible input only. It has:

```text
coach_speed_row_gate=true
missing_required_gates=[]
missing_required_evidence=[]
promotion_eligible=true
promotion_claim=false
calls_train_muzero=false
touches_live_runs=false
```

The attached H100 speed row is real compact-owned training work, but it is not
stock `train_muzero`, not a live-run safety canary, not a stock-resume proof,
not rating/leaderboard proof, and not a training-speedup claim by itself.

## Decision

Do not treat local `promotion_eligible=true` as a promotion or deployment
claim. It means the current compact-owned candidate has hash-bound local
compatibility evidence for the required lifecycle and speed-row gates.

Actual `promotion_claim=true` remains blocked after OPT-060 and OPT-061. The
readiness bundle now exists and is hash-bound, and the matched-quality
sufficiency review now chooses the larger-study branch. These are manual review
and decision artifacts, not an automatic promotion switch. The generic
compatibility builder rejects the boolean flip with a
`post_compatibility_promotion_readiness_required` blocker.

This is intentional. The speed row answered "can the compact-owned path perform
measured non-profile training work for this candidate?" It did not answer "is
this candidate good enough and safe enough to deploy through Coach?"

## Required Evidence

Before any real promotion, deployment, live-run, stock-resume, or
rating-quality claim, use the OPT-060 hash-bound readiness bundle and OPT-061
matched-quality sufficiency review as the current review packet. If replacing
or upgrading it, the bundle must include at least these lanes:

1. Matched stock-vs-compact learning-quality rows from the same repo state and
   hardware class where possible. The rows must record observation, reward,
   death, noise, learner, replay, checkpoint horizon, and bucket-timing
   settings, plus actual quality movement. Speed alone is not enough.
2. Stock resume/load canary for the compact-exported checkpoint path. The
   canary must prove metadata acceptance, checkpoint reload, model identity and
   provenance preservation, and eval/tournament loader behavior after resume.
3. Isolated live-run safety canary in a non-mutating Coach lane. It must prove
   assignment plumbing, checkpoint write/read, trainer consumption, and metrics
   movement without touching production live runs.
4. Assignment/rating/leaderboard proof in a sandbox namespace. Produced
   artifacts must be accepted by the same surfaces that would consume a
   promoted candidate.
5. Longer-horizon compact learning metrics across multiple checkpoints, not
   just the tiny lifecycle smoke or one threshold speed row.

## Non-Claims

Keep these false until the matching evidence exists:

```text
promotion_claim=false
training_speedup_claim=false
live_run_safety_claim=false
stock_resume_claim=false
rating_or_promotion_quality_claim=false
touches_live_runs=false
```

`calls_train_muzero=false` also stays false for the compact-owned route unless a
future stock bridge literally enters stock `train_muzero` and proves it.

## Code Guard

The current compatibility code now has two layers:

1. If `promotion_claim=true` and local compatibility is incomplete, it reports
   the missing local gates or evidence.
2. If local compatibility is complete, `promotion_claim=true` still fails with
   the post-compatibility readiness blocker.

Direct `CompactCoachCompatibilityReportV1(..., promotion_claim=True)`
construction also fails, so callers cannot bypass the public builder by
instantiating the exported report type. This prevents an already eligible
compatibility report from being rebuilt with one boolean changed.

Validation:

```text
uv run ruff check src/curvyzero/training/compact_coach_compatibility.py tests/test_compact_coach_compatibility.py
uv run pytest tests/test_compact_coach_compatibility.py -q
```

Result:

```text
ruff passed
18 passed
```

## Current Task

The named readiness lanes now have local evidence through OPT-059, and OPT-060
binds them into:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-promotion-readiness-bundle-review-20260530a/readiness_bundle_review_report.json
```

Next work is not another local lane. It is either explicit manual acceptance of
the current review packet for a non-production next step, or a larger/more
durable matched-quality study if the current 1024x8 mixed-hardware canary plus
local longer-horizon trace are too small.
