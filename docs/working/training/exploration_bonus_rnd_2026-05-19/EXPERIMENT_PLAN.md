# Experiment Plan

Created: 2026-05-19.

This is a gate plan, not a sweep plan. Do not broaden until the previous gate
has a clean proof.

## Common Measurements

Record these for every gate that launches training:

- latest and best checkpoint sparse outcome;
- survival reward and survival length;
- tournament rank/Elo when available;
- action distribution and action collapse indicators;
- target reward min/mean/max/p95;
- support clipping and requested vs effective support;
- checkpoint metadata hash and observation surface hash.

RND gates also record:

- RND input spec and observed input shape;
- RND predictor loss;
- train/estimate call ratio and small-buffer skip count;
- raw RND MSE mean/std/p50/p95 before any normalization;
- raw intrinsic mean/std/p95;
- normalized intrinsic mean/std/p95;
- intrinsic/extrinsic target ratio;
- RND update count and sample count;
- predictor/target/optimizer/normalizer state hashes when resumable.

## E0: Baseline Control

Purpose: prove the current trainer still behaves as expected.

Config:

```text
exploration_bonus_mode=none
```

Success:

- stock `train_muzero` path is selected;
- no RND imports or reward-model construction on the hot path;
- env reward, target reward, sidecars, and tournament behavior remain
  baseline-equivalent;
- existing CZ26-style metrics continue to write.

## E0m: Meter-Only RND

Purpose: prove RND plumbing without changing learning targets.

Config:

```text
exploration_bonus_mode=rnd_meter_v0
exploration_bonus_weight=0.0
exploration_bonus_feature_source=policy_gray64_latest/v0
exploration_bonus_rnd_batch_size=64 or the normal learner batch
exploration_bonus_rnd_update_per_collect=100
require_rnd_metrics=true for tiny proof smokes
```

Future command shape:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute gpu-l4-t4-cpu40 \
  --reward-variant survival_plus_bonus_plus_outcome \
  --reward-outcome-alpha 0.67 \
  --exploration-bonus-mode rnd_meter_v0 \
  --exploration-bonus-weight 0.0 \
  --exploration-bonus-feature-source policy_gray64_latest/v0 \
  --exploration-bonus-rnd-batch-size 64 \
  --exploration-bonus-rnd-update-per-collect 100 \
  --require-rnd-metrics
```

Success:

- `train_muzero_with_reward_model` path is selected;
- Curvy adapter emits `(N,1,64,64)` `float32` in `[0,1]`;
- RND trains/logs;
- augmented target rewards equal extrinsic target rewards exactly;
- `rnd_reward_model_metrics_latest.json` proves construction, collect/train/
  estimate calls, predictor hash changed, target hash stayed frozen, and
  `last_target_reward_changed=false`;
- `train_cnt_rnd`, `estimate_cnt_rnd`, train/estimate ratio, skip count, and
  raw MSE percentiles are present;
- metadata says `training_effect: reward_target_unchanged` and
  `target_reward_effect: unchanged`;
- resume either restores RND state or clearly marks the run diagnostic and
  non-resumable.

Stop if:

- target rewards differ at `weight=0.0`;
- hook patching targets a different entrypoint from the one called;
- LightZero RND support assumes flat obs or a hardcoded unroll that the adapter
  does not override;
- sidecars cannot identify the exploration mode and input spec.

## E1: Positive-Weight RND Canary

Purpose: test whether curiosity pressure helps retained extrinsic quality.

This remains blocked. `rnd_replay_target_v0` exists as plumbing, but
positive-weight RND should not launch as a recommendation until intrinsic
normalization is decided. Do not make `rnd_meter_v0` accept nonzero weight.

Config:

```text
exploration_bonus_mode=rnd_replay_target_v0
exploration_bonus_weight=<small>
exploration_bonus_feature_source=policy_gray64_latest/v0
exploration_bonus_cap=<bounded>
exploration_bonus_schedule=<decayed or explicitly constant>
```

Required new surface before launch:

- exact CLI/Modal kwargs for cap, schedule, and decay horizon;
- explicit `rnd_reward_norm=running_std_v0` or equivalent global normalizer;
- support math that includes the bounded intrinsic term;
- checkpoint metadata fields `model_exploration_bonus_*` and RND state hashes;
- resume save/load of predictor, target, optimizer, normalizers, counters, and
  config hash;
- manifest/grid labels and tournament compatibility fields so objective-changing
  policies are not silently pooled with baseline policies.

Success:

- support calculation accounts for bounded intrinsic reward;
- intrinsic/extrinsic ratio stays within the planned range;
- tournament/eval scoring remains extrinsic-only;
- latest-checkpoint quality does not collapse relative to E0;
- best-checkpoint quality improves or exposes a clear diagnostic signal.

Stop if:

- novelty dominates sparse outcome;
- action entropy collapses;
- target support clips frequently;
- resume changes novelty statistics discontinuously;
- tournament ranking improves only through intrinsic-contaminated scoring.

Current normalization recommendation:

```text
raw_mse = mean((predictor(obs) - target(obs))^2, feature_dim)
scale = sqrt(running_var(raw_mse_stream)) + eps
normalized_intrinsic = raw_mse / scale
bounded_intrinsic = clip(normalized_intrinsic, 0, exploration_bonus_cap)
target_reward += exploration_bonus_weight * bounded_intrinsic
```

Do not mean-center the intrinsic reward. RND error is nonnegative; the first
positive lane needs scale stabilization and a cap, not negative novelty.
Suggested first weights after the contract lands: `0.003`, `0.01`, `0.03`,
with `exploration_bonus_cap=1.0` and `rnd_update_per_collect=100`.

Optimizer smoke note: do not use learner `batch_size=1` for RND speed smokes.
A tiny batch-1 CPU-oracle profile reached `train_muzero_with_reward_model` and
then failed with a training-layer “more than 1 value per channel” error. Use
`batch_size >= 32` for RND plumbing/speed proof rows and prefer the normal
run batch. The current code default is `rnd_update_per_collect=100`; treat
smaller values as explicit ablations, not the default recommendation.

## E2: Input Variant Canary

Purpose: compare `policy_gray64_stack4/v0` only if latest-frame RND is stable.

Decision needed:

- latest-frame RND may miss motion and near-death dynamics;
- stack4 RND may reward stale temporal changes and is more shape-risky.

Do not run E2 until E1 establishes whether the simple input is useful enough to
justify expanding the surface.

## Parking Lot Experiments

- Source-state count/coverage bonus: useful native novelty canary, but separate
  from true RND.
- MuZero latent RND: research-only because the representation is nonstationary
  and entangles the curiosity target with learner internals.
- Broad hyperparameter sweep: premature until E0m is clean and the positive-RND
  normalization contract is chosen.
