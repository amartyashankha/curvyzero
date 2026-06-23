# RND Integration Prep Interfaces

Created: 2026-05-19.

Status: concrete prep notes. This pins the obvious interfaces before touching
training code.

Planning update after parallel critique: the first patch should be narrower than
the full interface list below. Start with `mode=none|rnd_meter_v0`,
`weight=0.0`, and `feature_source=policy_gray64_latest/v0`. Treat broader
modes, schedules, caps, count bonuses, and positive-weight support scaling as
post-meter-only work.

## Short Answer

Run the first true RND trainer inside the same Modal container and same trainer
process as the LightZero trainer.

Reason: the RND model needs the same replay samples, policy/model device,
checkpoint cadence, learner metrics, and resume hooks. A separate service would
mainly create synchronization and freshness bugs before we know the bonus helps.

Current trainer image already installs `LightZero==0.2.0`, JAX CUDA, NumPy,
Cloudpickle, and Pillow in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
The first integration should extend that image only if the installed LightZero
package lacks `lzero.entry.train_muzero_with_reward_model` or
`lzero.reward_model.rnd_reward_model`.

## First Decision Points

| Question | First answer |
| --- | --- |
| Where does true RND run? | Same Modal trainer image, same LightZero train process. |
| Is RND env-side? | No for true RND. Env-side is only for cheap count/coverage canaries. |
| How does true RND affect training? | Replay-target augmentation, matching LightZero's reward-model entrypoint. |
| Does eval/tournament use intrinsic reward? | No. Metadata only. Scoring stays extrinsic/source-game. |
| Is `reward_variant` changed? | No. Add an orthogonal `exploration_bonus_*` contract. |
| Default RND input? | Latest policy gray64 frame, `float32`, channel-first, shape `(1,64,64)`, range `[0,1]`. |
| First nonzero schedule? | Deferred until meter-only RND is behavior-neutral. |
| First arm before nonzero reward? | Meter-only `weight=0`; target rewards unchanged, trainer path changes. |

## Interfaces To Prepare

### `ExplorationBonusSpec`

Home: `src/curvyzero/training/exploration_bonus.py`.

Purpose: a typed/pure config contract, not tied to Modal.

Fields:

```text
mode: none | rnd_meter_v0 | rnd_replay_target_v0 later | count_coverage_v0 later
schema_id: curvyzero_exploration_bonus/<mode>/v0
config_hash: stable hash of all fields below
training_only: true
rnd_batch_size: int
rnd_update_per_collect: int
rnd_buffer_size: int
rnd_learning_rate: float
rnd_weight_decay: float
feature_source: policy_gray64_latest | policy_gray64_stack4 | muzero_latent | source_state_compact
rnd_input_norm: bool
seed: int | null
target_reward_effect: unchanged | intrinsic_weighted_addition
trainer_effect: uses_stock_muzero_entrypoint | uses_reward_model_entrypoint_and_trains_rnd_meter
```

Initial defaults:

```text
mode = none
weight = 0.0
feature_source = policy_gray64_latest/v0
rnd_batch_size = 64
rnd_update_per_collect = 100
rnd_buffer_size = 100000
rnd_learning_rate = 3e-4
rnd_weight_decay = 1e-4
rnd_input_norm = false
seed = trainer seed
```

Open choice: LightZero's stock RND uses batch min-max normalization of RND error.
The OpenAI reference normalizes intrinsic reward by running statistics of
discounted intrinsic returns. LightZero-style batch min/max is acceptable only
for meter-only compatibility and overhead smokes. Positive-weight RND remains
blocked until the intrinsic-normalization contract is chosen; batch-relative
min/max must not be treated as a serious curiosity signal. Current code logs raw
RND MSE before batch min/max so the metric is not only batch-relative.

### `RndInputSpec`

Home: same contract module.

Purpose: make the RND network input explicit and hashable.

Default input:

```text
id = policy_gray64_latest/v0
source = LightZero replay observation, produced by the current policy observation contract
layout = CHW
shape = (1, 64, 64)
dtype = float32
range = [0.0, 1.0]
extraction = obs_stack[-1:, :, :]
perspective = learner/ego player perspective
surface = browser_lines + simple_symbols
backend = cpu_oracle initially
trail_render_mode = from policy observation contract
bonus_render_mode = from policy observation contract
```

Why latest frame first:

- OpenAI RND used the newest grayscale frame rather than the whole frame stack.
- CurvyTron novelty should mostly mean novel board/trail geometry, not a novel
  four-frame temporal pattern.
- It keeps the RND target smaller and easier to normalize.

Keep explicit alternatives:

```text
policy_gray64_stack4/v0: shape (4,64,64), exact policy input
muzero_latent/v0: latent from MuZero representation network
source_state_compact/v0: quantized source features, for native count/coverage not true RND first
```

Do not feed raw source/vector state into true RND until replay can carry that
state with provenance. Source-state features are better as diagnostics or a
native count bonus first.

### `RndReplayBatchAdapter`

Home: new module, perhaps `src/curvyzero/training/rnd_replay_adapter.py`.

Purpose: isolate all LightZero batch-shape assumptions.

Interface:

```python
class RndReplayBatchAdapter:
    def flatten_inputs(self, train_data, input_spec) -> RndFlatBatch:
        ...

    def restore_augmented_rewards(self, train_data, augmented_rewards, shape_meta):
        ...
```

Responsibilities:

- read `obs_batch_orig` and `target_reward` from LightZero `train_data`;
- derive unroll length from `target_reward.shape`, not a hardcoded `6`;
- support image obs with shape `(B, K, C, H, W)` or whatever LightZero actually
  provides after sampling;
- produce RND tensor with shape `(B*K, C_rnd, H, W)` for image modes;
- deep-copy target rewards before writing augmented rewards;
- return shape metadata for audits: original obs shape, target reward shape,
  inferred unroll length, extracted RND input shape.

This is the adapter that avoids copying upstream
`lzero/reward_model/rnd_reward_model.py` assumptions directly.

### `CurvyRNDRewardModel`

Home: `src/curvyzero/training/exploration_bonus.py`.

Purpose: a Curvy-safe wrapper around the LightZero RND idea.

Interface should mirror LightZero enough to plug into
`train_muzero_with_reward_model`:

```python
class CurvyRNDRewardModel:
    def collect_data(self, new_data) -> None: ...
    def train_with_data(self) -> dict[str, float]: ...
    def estimate(self, train_data) -> Any: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
```

State to own:

- target network;
- predictor network;
- optimizer;
- update counters;
- train-data counters;
- config hash.

For the first adapter, do not let this class know about Modal Volumes. It should
only expose state and metrics. The Modal trainer owns persistence.

### `TrainingEntrypointSpec`

Home: config builder or a tiny helper next to it.

Purpose: choose which LightZero entrypoint to call.

```text
exploration_bonus.mode == none:
  entrypoint = lzero.entry.train_muzero

exploration_bonus.mode == rnd_meter_v0:
  entrypoint = lzero.entry.train_muzero_with_reward_model
```

The current hooks accept a `train_muzero` callable and inspect its globals for
`BaseLearner`, collector/evaluator classes, etc. Rename internally to
`train_entrypoint` only after the smoke passes; the first patch can keep the
variable name and set metadata field:

```text
trainer_entrypoint = lzero.entry.train_muzero_with_reward_model
```

Add a dry compatibility check:

```text
hasattr(lzero.entry, "train_muzero_with_reward_model")
can import lzero.reward_model.rnd_reward_model
signature accepts input_cfg, seed, max_train_iter, max_env_step
```

Implementation note: `rnd_meter_v0` currently patches
`RNDRewardModel = CurvyRNDRewardModel` into the selected entrypoint globals and,
when importable, `lzero.reward_model.rnd_reward_model.RNDRewardModel`. The patch
is restored after the trainer exits. This is deliberately small and should be
replaced with registry/config wiring only when the upstream LightZero interface
is clear enough to avoid a broader fork.

### `LightZeroRNDConfigPatch`

Home: `lightzero_config_builder.py` once the core contract exists.

Patch when RND is enabled:

```text
main_config.reward_model = {
  type: rnd_muzero or curvy_rnd_muzero,
  intrinsic_reward_type: add,
  input_type: obs,
  obs_shape: (1,64,64) or adapter-specific shape,
  hidden_size_list: ...,
  intrinsic_reward_weight: spec.weight,
  input_norm: spec.input_norm,
  input_norm_clamp_min: spec.input_norm_clamp_min,
  input_norm_clamp_max: spec.input_norm_clamp_max,
  extrinsic_reward_norm: false,
  rnd_buffer_size: spec.rnd_buffer_size,
  update_per_collect: spec.update_per_collect,
  learning_rate: spec.learning_rate,
  weight_decay: spec.weight_decay,
}

main_config.policy.use_rnd_model = true
main_config.policy.use_momentum_representation_network = true
main_config.policy.target_model_for_intrinsic_reward_update_type = spec.target_update_type
main_config.policy.target_update_freq_for_intrinsic_reward = spec.target_update_freq
main_config.policy.target_update_theta_for_intrinsic_reward = spec.target_update_theta
```

If using `policy_gray64_latest`, the adapter can expose RND obs shape as
`(1,64,64)` even though the policy observation shape remains `(4,64,64)`.

### `TargetSupportPatch`

Home: `reward_contracts.py`.

Purpose: include bounded intrinsic reward in requested/effective reward/value
support metadata when reward targets are augmented.

Inputs:

```text
extrinsic reward policy
source_max_steps
policy_action_repeat_max
exploration_bonus weight/cap/schedule
model_support_cap
```

For first bounded RND:

```text
per_step_intrinsic_max = weight * cap
reward_support_requested += ceil(per_step_intrinsic_max)
value_support_requested += ceil(source_max_steps * per_step_intrinsic_max)
```

Also record:

```text
model_intrinsic_reward_support_requested_scale
model_intrinsic_value_support_requested_scale
model_intrinsic_support_capped
```

### `ExplorationTelemetry`

Home: learner metrics writer plus attempt summaries.

For true replay-target RND, do not write intrinsic reward into env telemetry as
if it came from the environment. Put it beside learner metrics.

Target metric surface:

```text
schema_id = curvyzero_rnd_reward_model_metrics/v0
train_iter
envstep
exploration_bonus_mode
exploration_bonus_config_hash
rnd_input_shape
rnd_input_source
rnd_loss
rnd_intrinsic_raw_mean/std/min/max/p95
rnd_intrinsic_normalized_mean/std/min/max/p95
augmented_reward_mean/std/min/max
extrinsic_reward_mean/std/min/max
intrinsic_to_extrinsic_abs_ratio
obs_rms_mean_summary
obs_rms_std_summary
reward_normalizer_state_summary
rnd_update_count
rnd_train_data_count
```

Current implementation:

```text
command/config/surface metadata:
  exploration_bonus.mode
  exploration_bonus.config_hash
  exploration_bonus.input_spec
  exploration_bonus.target_reward_effect = unchanged
  exploration_bonus.trainer_effect = uses_reward_model_entrypoint_and_trains_rnd_meter
  trainer_entrypoint
  policy_use_rnd_model
  reward_model_type

TensorBoard scalars:
  rnd_reward_model/rnd_mse_loss
  rnd_reward_model/rnd_reward_mean

JSON sidecars under attempt train root:
  rnd_reward_model_metrics_latest.json
  rnd_reward_model_metrics.jsonl
  rnd_reward_model_metrics_scan.json

Hard proof gate:
  require_rnd_metrics = true
  fails unless collect_data/train_with_data/estimate were called, predictor
  hash changed, target hash stayed frozen, and weight=0 target rewards stayed
  unchanged.
```

For meter-only, JSON sidecar metrics must include:

```text
weight = 0.0
augmented_reward_equal_to_extrinsic = true
```

### `ExplorationCheckpointState`

Home: existing `lightzero_resume_state` machinery in the Modal trainer.

Status: not implemented for `rnd_meter_v0`. Treat the current meter as
diagnostic/non-resumable until the reward-model holder field is wired into the
existing full-resume hook.

Files to write under the attempt train root:

```text
lightzero_resume_state/rnd_reward_model_state.pt
lightzero_resume_state/rnd_reward_model_state.json
lightzero_resume_state/rnd_reward_model_metrics_latest.json
```

The `.pt` should contain model/optimizer/normalizer/counter state. The `.json`
should contain schema id, config hash, tensor-shape summary, and SHA256 of the
binary state.

Checkpoint policy metadata sidecar additions:

```text
model_exploration_bonus_mode
model_exploration_bonus_schema_id
model_exploration_bonus_config_hash
model_exploration_bonus_weight
model_exploration_bonus_cap
model_exploration_bonus_schedule
model_exploration_bonus_feature_source
model_exploration_bonus_eval_behavior = disabled
rnd_state_ref/hash if stateful and available
```

### `TournamentCompatibility`

Home: `src/curvyzero/tournament/curvytron/contracts.py`.

Deferred. Do not add exploration fields to `rating_pool_hash` for meter-only
RND. Tournament scoring remains extrinsic-only and current pool hashing stays
unchanged until an RND-trained checkpoint can actually enter evaluation.

Later, add to rating pool hash and roster:

```text
model_exploration_bonus_mode
model_exploration_bonus_config_hash
```

Tournament does not score intrinsic reward. This is only so RND-trained and
non-RND-trained policies can be stratified and not silently pooled as identical
training objectives.

### CLI / Manifest Surface

Patch-one flags should be only the disabled mode and meter-only RND. The broader
surface below is for the positive-weight or native-count phase after the first
gate:

```text
--exploration-bonus-mode none|rnd_meter_v0
--exploration-bonus-weight 0.0
--exploration-bonus-feature-source policy_gray64_latest/v0
--exploration-bonus-rnd-batch-size 64
--exploration-bonus-rnd-update-per-collect 100
--exploration-bonus-rnd-buffer-size 100000
--exploration-bonus-rnd-learning-rate 0.0003
--exploration-bonus-rnd-weight-decay 0.0001
--exploration-bonus-rnd-input-norm / --no-exploration-bonus-rnd-input-norm
--require-rnd-metrics
```

Positive-weight RND needs a new mode, for example `rnd_replay_target_v0`, and
must add explicit cap/schedule/decay kwargs before launch. Do not make
`rnd_meter_v0` objective-changing.

Suggested first future command shape:

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

That is the meter-only RND smoke. Nonzero reward comes after this proves
behavioral equivalence, metrics proof, support math, checkpoint metadata,
manifest labels, tournament stratification, and state persistence.

## Same Container Details

Use the same container initially.

Pros:

- RND model sees the same replay data as the learner;
- no cross-container model freshness protocol;
- no extra Modal Dict/Queue hot path;
- checkpoint/resume can be tied to existing LightZero checkpoint hooks;
- GPU/CPU placement follows the trainer's existing `cuda` choice.

Costs:

- extra learner-process memory;
- extra forward/backward pass for RND predictor;
- reward-model training competes with MuZero learner for GPU time.

Those costs are acceptable for the canary. If RND becomes useful and expensive,
then split only the reward-model training behind a stable state/checkpoint
contract.

## Initial Smoke Sequence

1. Pure dry config:
   - `mode=none` equals current config.
   - `mode=rnd_meter_v0`, `weight=0` injects reward_model and entrypoint.
2. Shape test:
   - fake LightZero train batch with `(4,64,64)` obs.
   - adapter extracts `(1,64,64)` latest-frame RND input.
   - target reward shape round-trips without hardcoded unroll length.
3. Meter-only trainer smoke:
   - one tiny profile/train call;
   - writes RND metrics;
   - augmented reward equals extrinsic reward.
4. Resume smoke:
   - save RND state;
   - resume;
   - counters/normalizers/predictor hash continue instead of resetting.
5. Sidecar/tournament smoke:
   - checkpoint metadata includes exploration config;
   - tournament hash changes for RND versus non-RND;
   - eval reward remains extrinsic.

## Things To Avoid

- Do not put RND predictor state inside individual env workers.
- Do not mutate replay rewards in-place without a deep copy and shape audit.
- Do not let `reward_variant` imply intrinsic reward.
- Do not let background eval/tournament instantiate RND for scoring.
- Do not compare RND and non-RND runs without recording the config hash.
- Do not use LightZero's hardcoded six-step RND reshape for Curvy observations.
