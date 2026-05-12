# CurvyTron Native Train MuZero Probe - 2026-05-10

Question: can we build a minimal native LightZero `train_muzero` CurvyTron
visual/survival path from existing registered env wrappers, instead of leaning
on the custom bounded adapter?

Probe framing: treat CurvyTron as Pong-like unless the code proves otherwise.
The useful question is not "why is CurvyTron special?" It is "what wrapper/API
change makes CurvyTron look enough like LightZero Atari/Pong that the existing
`train_muzero` path can drive it?"

## Answer

Yes, in the Pong-like single-agent sense. A real native LightZero entrypoint
already exists:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

It imports `lzero.entry.train_muzero`, patches the stock
`zoo.atari.config.atari_muzero_config`, and calls:

```text
train_muzero([main_config, create_config], seed=..., max_train_iter=..., max_env_step=...)
```

Active registered env:

```text
env.type: curvyzero_stacked_debug_visual_survival_lightzero
env.import_names: ["curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env"]
env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
reward_schema_id: curvyzero_survival_time/v0
observation_schema_id: curvyzero_stacked_debug_occupancy_gray64_player_aware_survival_time/v1
observation_shape: [4,64,64]
env.frame_stack_num: 1
policy.model.frame_stack_num: 1
policy.model.image_channel: 4
frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
debug_fidelity_only: true
source_fidelity_claim: none
```

That is the smallest existing native-LightZero path. It uses LightZero's
collector, replay, MCTS, learner, checkpoints, and `train_muzero` entrypoint.
It does not use the bounded two-seat train smoke.

Important latest result: the custom accumulated-replay two-seat run was
mechanically clean, but the survival curve was weak/flat: `191.688` at
`iteration_0`, then `201.844` from `iteration_1` onward through `iteration_32`.
Treat scaling the custom bounded adapter as not promising. Prioritize the
native LightZero trainer/replay/checkpointing path for CurvyTron-as-Pong-like
first.

Bluntly: CurvyTron fits the existing LightZero Pong/Atari path when the wrapper
turns it into a one-player control problem with the other player hidden inside
the env, exactly as Pong hides the emulator opponent/world dynamics behind
`env.step(action)`.

The wrapper/API shape required for that fit is:

```text
reset(seed) -> {
  observation: float32[4,64,64],
  action_mask: int8[3],
  to_play: -1,
  timestep: int,
}

step(ego_action: scalar 0|1|2) -> BaseEnvTimestep(
  obs=same shape,
  reward=scalar survival reward,
  done=bool,
  info={opponent policy metadata, schema ids, terminal metadata}
)
```

Existing code already provides that through
`curvyzero_stacked_debug_visual_survival_lightzero`. No new big system is
needed for fixed/frozen-opponent native `train_muzero`.

## What It Does

`CurvyZeroStackedDebugVisualSurvivalLightZeroEnv` is registered with
DI-engine's `ENV_REGISTRY` and subclasses the local stacked debug visual
survival wrapper. The wrapper:

- wraps `CurvyTronSourceEnv`;
- exposes one LightZero ego action per `env.step(action)`;
- renders player-aware debug occupancy frames;
- maintains a wrapper-owned four-frame FIFO stack;
- returns survival-time reward, `1.0` when the ego is alive after the transition
  and `0.0` otherwise;
- writes env-step telemetry when configured by the Modal trainer.

The Modal train wrapper hard-patches the Atari MuZero config to match this
surface: conv model, action space size `3`, observation shape `[4,64,64]`,
`image_channel=4`, `frame_stack_num=1`, base env manager, one or more
collector/evaluator envs, and the registered CurvyTron env import name.

## Pong-Like Fit

For native `train_muzero`, the right mental model is Atari Pong:

```text
LightZero policy emits one action.
Env owns the rest of the world transition.
Replay stores one policy row per env step.
```

The existing CurvyTron wrapper matches that model by making opponent control an
env-internal policy. In other words, the missing API change was not "teach
LightZero about simultaneous CurvyTron"; it was "hide simultaneous CurvyTron
behind a Pong-like single-agent wrapper." That change already exists for the
stacked visual survival wrapper.

Exact fit points:

```text
BaseEnv registration: yes
create_config.env.type/import_names: yes
BaseEnvTimestep return: yes
single scalar action: yes
conv observation: yes, wrapper emits [4,64,64]
Atari-style train config: yes, patched from zoo.atari.config.atari_muzero_config
survival reward: yes
opponent inside env: yes
```

Exact remaining mismatches:

```text
browser/canvas pixels: no, optional later debug/human evidence
source-state gray64 target: yes, active current observation path
native LightZero frame stacking proof for unstacked CurvyTron frames: no
current-policy two-seat self-play: no
```

## Fixed-Opponent / Single-Ego Status

The current native `train_muzero` lane is single-ego by design. That is fine for
a Pong-like wrapper. It is not fine if the claim is current-policy two-seat
self-play.

Exact modes:

```text
opponent_policy_kind=fixed_straight
  LightZero chooses player_0 action.
  The env fills player_1 with action 1.
  opponent_training_relation: learner_vs_fixed_straight

opponent_policy_kind=frozen_lightzero_checkpoint
  LightZero chooses player_0 action.
  The env fills player_1 through a frozen checkpoint provider.
  opponent_training_relation: learner_vs_frozen_lightzero_checkpoint
```

The env telemetry and summary intentionally report:

```text
current_policy_self_play: false
current_policy_self_play_blocker:
  LightZero train_muzero calls this env with only the ego action. The live
  collector policy and learner weights are outside env.step, so the env cannot
  ask the current policy for the opponent action without a larger collector or
  two-seat env change.
```

This is not a naming issue. The stock `train_muzero` collector calls
`env.step(action)` with one action for one policy row. If CurvyTron is being
treated as Pong-like, that is exactly the desired API. If CurvyTron is being
treated as two-seat current-policy self-play, it is the blocker.

So the exact split is:

```text
Pong-like CurvyTron: existing wrapper is enough.
Current-policy two-seat CurvyTron: existing train_muzero API is not enough.
```

## Registered Env Inventory

CurvyTron-relevant registered env wrappers already in the repo:

```text
curvyzero_v0_lightzero
  module: curvyzero.training.curvyzero_lightzero_env
  env_id: CurvyZeroLightZero-v0
  surface: scalar/ray, sparse outcome reward, single ego, fixed action opponent

curvyzero_survival_time_lightzero
  module: curvyzero.training.curvyzero_survival_time_lightzero_env
  env_id: CurvyZeroSurvivalTimeLightZero-v0
  surface: scalar/ray, survival-time reward, single ego, fixed action opponent

curvyzero_debug_visual_tensor_lightzero
  module: curvyzero.training.curvyzero_debug_visual_lightzero_env
  env_id: CurvyZeroDebugVisualTensorLightZero-v0
  surface: unstacked debug visual [1,64,64], sparse outcome reward, single ego,
  fixed straight opponent

curvyzero_stacked_debug_visual_survival_lightzero
  module: curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env
  env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
  surface: wrapper-stacked debug visual [4,64,64], survival-time reward,
  single ego, fixed or frozen opponent
```

Only the last one currently has a real Modal `train_muzero` visual/survival
trainer wrapper.

## Native Reuse Available Today

Tiny no-code config/command change that reuses more native LightZero than the
bounded adapter: use the existing Modal native trainer in `profile` mode with a
one-learner-call cap. This still uses the registered wrapper-stacked env, but
the collector, replay, MCTS, learner call, and checkpoint hooks come from
LightZero `train_muzero`.

Use this instead of the bounded profile when the question is "does native
LightZero still drive this registered env end to end?":

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute cpu \
  --wait-for-train \
  --seed 0 \
  --run-id curvytron-native-train-muzero-profile-s0 \
  --attempt-id profile-native-train-muzero-smoke-20260510 \
  --max-env-step 64 \
  --max-train-iter 4 \
  --source-max-steps 32 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --n-episode 1 \
  --num-simulations 2 \
  --batch-size 4 \
  --save-ckpt-after-iter 1 \
  --stop-after-learner-train-calls 1 \
  --opponent-policy-kind fixed_straight
```

Expected smoke criteria:

```text
ok: true
called_train_muzero: true
trainer_entrypoint: lzero.entry.train_muzero
surface.env_type: curvyzero_stacked_debug_visual_survival_lightzero
surface.observation_shape: [4,64,64]
surface.reward_schema_id: curvyzero_survival_time/v0
surface.current_policy_self_play: false
phase_profile.counts.collector_collect_calls >= 1
phase_profile.counts.mcts_search_calls >= 1
phase_profile.counts.learner_train_calls >= 1
action_observability.row_count > 0
```

For a tiny real train rather than a capped profile:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute cpu \
  --wait-for-train \
  --seed 0 \
  --run-id curvytron-native-train-muzero-train-s0 \
  --attempt-id train-native-train-muzero-smoke-20260510 \
  --max-env-step 128 \
  --max-train-iter 4 \
  --source-max-steps 32 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --n-episode 1 \
  --num-simulations 2 \
  --batch-size 4 \
  --save-ckpt-after-iter 1 \
  --opponent-policy-kind fixed_straight
```

Expected train criteria:

```text
ok: true
called_train_muzero: true
checkpoint_mirror.count > 0
action_observability.row_count > 0
action_observability.opponent_action_histogram["1"] > 0
current_policy_self_play: false
```

Recommended priority after the bounded flat result:

```text
1. Smoke native train_muzero fixed-straight Pong-like path.
2. Run small native train_muzero fixed-straight checkpoints/eval panels.
3. Run native train_muzero frozen-checkpoint opponent, using the same
   LightZero trainer/replay/checkpoint path.
4. Only if fixed/frozen native runs expose an opponent-contract ceiling, run a
   small joint-action collector experiment.
```

Do not spend more scale on the custom bounded accumulated-replay adapter unless
it is serving as a diagnostic control.

## Frozen Opponent Smoke

If checking the existing frozen-opponent single-ego path, first point at an
existing mirrored LightZero checkpoint, then run:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute cpu \
  --wait-for-train \
  --seed 0 \
  --run-id curvytron-native-train-muzero-frozen-profile-s0 \
  --attempt-id profile-native-frozen-opponent-smoke-20260510 \
  --max-env-step 64 \
  --max-train-iter 4 \
  --source-max-steps 32 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --n-episode 1 \
  --num-simulations 2 \
  --batch-size 4 \
  --save-ckpt-after-iter 1 \
  --stop-after-learner-train-calls 1 \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/<run>/checkpoints/lightzero/<checkpoint>.pth.tar \
  --snapshot-ref curvytron_native_train_muzero_frozen_smoke
```

Expected frozen criteria:

```text
ok: true
called_train_muzero: true
surface.opponent_policy_kind: frozen_lightzero_checkpoint
surface.opponent_training_relation: learner_vs_frozen_lightzero_checkpoint
surface.current_policy_self_play: false
action_observability.first_rows[*].opponent_checkpoint_ref is populated
```

## Tiny Wrapper/API Changes That Would Help

The safest small change, if we wanted an even cleaner Pong-like native path, is
not to add a new trainer. It would be a small wrapper/config cleanup around the
existing registered visual survival wrapper:

```text
Keep env.type: curvyzero_stacked_debug_visual_survival_lightzero
Keep env.step(action) as one scalar ego action
Keep opponent policy internal to the env
Keep observation emitted as [4,64,64]
Keep env.frame_stack_num=1 and policy.model.frame_stack_num=1
```

The useful cleanup would be naming/documentation, for example exposing an alias
like `curvyzero_ponglike_visual_survival_lightzero` or adding a config preset
that says "Atari/Pong-style single-agent wrapper." That is not functionally
necessary, so this probe does not implement it.

The not-safe "tiny" change would be switching the trainer to
`curvyzero_debug_visual_tensor_lightzero` and hoping LightZero stacks the frames
for us. That would be more native if proven, but the repo does not currently
prove it.

## Why Not Use The Unstacked Visual Wrapper By Config Only

The unstacked visual wrapper exists:

```text
env.type: curvyzero_debug_visual_tensor_lightzero
env_id: CurvyZeroDebugVisualTensorLightZero-v0
payload: float32[1,64,64]
reward: sparse outcome
```

There is no existing native trainer wrapper for it. The known working visual
trainer path deliberately uses the wrapper-stacked survival env with:

```text
env observation_shape: [4,64,64]
policy.model.observation_shape: [4,64,64]
env.frame_stack_num: 1
policy.model.frame_stack_num: 1
policy.model.image_channel: 4
```

Earlier evidence showed the wrong stack config can make LightZero treat the
wrapper-stacked `[4,64,64]` observation as if it should be stacked again,
producing a channel mismatch. So switching to the unstacked visual env plus
LightZero-owned stacking is not a safe one-line config change.

Exact blockers for an unstacked, more Atari-native wrapper:

```text
1. Need a registered survival-reward unstacked visual env, not only sparse outcome.
2. Need proof LightZero/DI-engine frame stacking wraps this registered env into
   the model's expected [4,64,64] tensor.
3. Need a config smoke that proves env observation_shape, model observation_shape,
   image_channel, and frame_stack_num agree before train_muzero.
```

That is still small, but it is not no-code and not already proven.

## Exact Blocker

Native `train_muzero` can be reused today for CurvyTron visual/survival as a
Pong-like single-agent environment:

```text
current learner ego vs fixed-straight opponent
current learner ego vs frozen checkpoint opponent
```

It cannot be reused today for full current-policy CurvyTron self-play. Exact
blocker: the registered env receives one ego action per step and does not
receive the live policy object, current learner weights, both player observation
rows, or learner update events.

A real self-play fix requires one of:

```text
1. a two-seat collector/trainer contract;
2. a custom collector that asks the live policy for both seats before
   env.step(joint_action);
3. a separate actor/learner loop with explicit weight refresh.
```

That blocker does not apply to the Pong-like fixed/frozen-opponent path.

Decision: the next serious probe should stay on native `train_muzero`, not the
custom bounded adapter. Use single-ego fixed/frozen opponents first because that
keeps LightZero's native trainer, replay buffer, checkpoint cadence, evaluator,
and artifact conventions in the loop. A joint-action collector is the fallback
experiment, not the default path.

No code change was made for this probe because the only clearly correct
single-file changes would be cosmetic aliases or extra logging. The functional
blocker is architectural, not a missing env registration.

## Local Non-Pytest Sanity Commands

These commands do not train and do not require pytest:

```bash
uv run python -m py_compile \
  src/curvyzero/training/curvyzero_debug_visual_lightzero_env.py \
  src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_env.py \
  src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py

uv run python -m curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke

uv run python -c 'from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env import CurvyZeroStackedDebugVisualSurvivalLightZeroEnv; env=CurvyZeroStackedDebugVisualSurvivalLightZeroEnv({"seed": 3, "source_max_steps": 4}); obs=env.reset(seed=3); ts=env.step(1); print({"obs_shape": list(obs["observation"].shape), "step_shape": list(ts.obs["observation"].shape), "reward": ts.reward, "schema": ts.info["reward_schema_id"], "opponent_training_relation": ts.info["opponent_training_relation"], "current_policy_self_play": ts.info["current_policy_self_play"], "frame_stack_owner": ts.info["frame_stack_owner"]})'
```
