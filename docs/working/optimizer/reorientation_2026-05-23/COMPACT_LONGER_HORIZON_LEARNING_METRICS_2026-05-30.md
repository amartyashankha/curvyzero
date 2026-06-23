# Compact Longer-Horizon Learning Metrics

Date: 2026-05-30

## Scope

This closes OPT-059 as compact-only learning-metrics evidence.

It does not claim promotion, compact superiority, production live-run safety,
rating quality, leaderboard publication, stock resume, or training speedup.

## Artifact

Report:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-longer-horizon-learning-metrics-local-20260530a/longer_horizon_learning_metrics_report.json
```

Producer manifest:

```text
artifacts/local/curvytron_compact_promotion_readiness_results/optimizer-compact-longer-horizon-learning-metrics-local-20260530a/longer_horizon_learning_metrics_producer_manifest.json
```

Command:

```text
uv run python scripts/build_compact_longer_horizon_learning_metrics_producer.py --run-id optimizer-compact-longer-horizon-learning-metrics-local-20260530a --compact-env-steps 8 --compact-warmup-steps 1 --compact-replay-pair-capacity 16 --eval-seed-count 2 --eval-steps 128 --num-simulations 1 --batch-size 2 --compact-sample-batch-size 2 --train-steps 1
```

## What It Proves

- The producer runs compact env/search/replay training through
  `CompactOwnedTrainerV1.record_step`.
- It emits three compact-native checkpoints and three stock-shaped eval exports.
- Each eval summary is bound to the exact exported checkpoint path.
- Model digests move between adjacent trained checkpoints.
- Cumulative denominators are monotonic.
- Trained points carry `curvyzero_compact_training_metrics_lineage/v1`.
- Eval seeds and horizon are fixed across the checkpoint series.
- All promotion, speedup, live-run, rating, leaderboard, and `train_muzero`
  claims remain false.

## Headline

```text
schema_id=curvyzero_compact_longer_horizon_learning_metrics/v1
checkpoint_count=3
learner_update_count_delta=2
sample_batch_count_delta=3
compact_rollout_rows=12
compact_sample_rows=4
replay_store_entry_count=3
replay_store_index_row_count=12
mean_survival=107.5,88.5,117.5
final_minus_first_mean_survival=10.0
metrics_lineage_refs=2
promotion_readiness_complete=false
final_promotion_bundle_still_required=true
```

At artifact emission time, `final_promotion_bundle_still_required=true`.
OPT-060 later supplied that bundle, and OPT-061 supplied the matched-quality
sufficiency decision. OPT-068 later supplied the larger 32x2048 matched-quality
packet. The remaining promotion boundary is now explicit manual/policy review
of that packet before any actual promotion.

## Validation

```text
uv run ruff check src/curvyzero/training/compact_longer_horizon_learning_metrics.py scripts/build_compact_longer_horizon_learning_metrics_producer.py tests/test_compact_longer_horizon_learning_metrics.py scripts/build_compact_matched_learning_quality_compact_candidate_producer.py tests/test_compact_matched_learning_quality_compact_candidate_producer.py

uv run pytest tests/test_compact_longer_horizon_learning_metrics.py tests/test_compact_matched_learning_quality_compact_candidate_producer.py -q
```

Result:

```text
ruff passed
15 passed, 2 warnings
```

The real report was also revalidated directly with
`validate_compact_longer_horizon_learning_metrics_v1`.

## Remaining Boundary

The named longer-horizon lane is closed, but promotion is still not automatic.
OPT-060 now provides the final readiness-bundle review that binds all readiness
artifacts by hash. The remaining question is whether manual review accepts the
current 1024x8 matched-quality canary plus this compact-only trace, or requires
a larger matched study before any actual promotion.
