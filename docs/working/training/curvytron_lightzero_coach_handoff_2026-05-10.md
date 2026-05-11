# CurvyTron LightZero Coach Handoff - 2026-05-10

Purpose: simple Coach-facing readiness note after reviewing the current
LightZero Pong proof path, CurvyTron scalar wrapper, runtime probe, and visual
handoffs.

## Plain Verdict

The small meaningful hookups are ready: CurvyTron has no-train installed
LightZero/DI-engine config/import smokes for both the scalar single-ego wrapper
and the debug visual single-ego wrapper, and both pass.

This is not CurvyTron training readiness. It proves only that installed
LightZero can import, construct, reset, seed, step, and terminate the current
CurvyTron wrappers through the real DI-engine `BaseEnvTimestep` boundary.

Final CurvyTron target is still visual stacked frames. The current proven visual
surface is only debug occupancy, not source-faithful pixels. Optimizer owns
adapter plumbing/profiling. Coach owns whether a trainer run learns anything.

## Pong Proof Path

Coach's current proof path is official/control Atari Pong through:

```text
src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py
```

That wrapper uses installed `LightZero==0.2.0`, imports
`zoo.atari.config.atari_muzero_config`, and calls `lzero.entry.train_muzero`.
The exact stock surface it guards is:

```text
env_id: PongNoFrameskip-v4
env.type: atari_lightzero
env.import_names: ["zoo.atari.envs.atari_lightzero_env"]
env_manager.type: subprocess
policy.type: muzero
policy.import_names: ["lzero.policy.muzero"]
model_type: conv
observation_shape: [4, 64, 64]
action_space_size: 6
collector_env_num: 8
evaluator_env_num: 3
num_simulations: 50
batch_size: 256
frame_stack_num: 4
stock max_env_step: 200000
```

The Pong wrapper's dry mode patches only artifact placement. Its
`max_env_step_override` path is a labeled faithful-short rehearsal, not exact
stock training. Its positive `survival_reward_per_step` path switches to a
separate `atari_lightzero_survival_shaped` env type and must not be reported as
stock/control Pong.

Do not copy Pong's ALE identity into CurvyTron. ALE is only the Atari emulator
for the Pong control path.

## CurvyTron Env Identity

Current CurvyTron LightZero scalar wrapper:

```text
file: src/curvyzero/training/curvyzero_lightzero_env.py
class: curvyzero.training.curvyzero_lightzero_env.CurvyZeroLightZeroEnv
env type: curvyzero_v0_lightzero
env id: CurvyZeroLightZero-v0
import_names: ["curvyzero.training.curvyzero_lightzero_env"]
```

The runtime probe builds this create config:

```text
create_config.env.type = "curvyzero_v0_lightzero"
create_config.env.import_names = ["curvyzero.training.curvyzero_lightzero_env"]
create_config.env_manager.type = "base"
create_config.policy.type = "muzero"
create_config.policy.import_names = ["lzero.policy.muzero"]
```

The wrapper intentionally reuses
`CurvyZeroLightZeroLocalSmokeEnv`, so there is one scalar semantics source
instead of a second drifting adapter.

Current CurvyTron LightZero debug visual wrapper:

```text
file: src/curvyzero/training/curvyzero_debug_visual_lightzero_env.py
class: curvyzero.training.curvyzero_debug_visual_lightzero_env.CurvyZeroDebugVisualLightZeroEnv
env type: curvyzero_debug_visual_tensor_lightzero
env id: CurvyZeroDebugVisualTensorLightZero-v0
import_names: ["curvyzero.training.curvyzero_debug_visual_lightzero_env"]
```

This wrapper is separate from `curvyzero_v0_lightzero`. It uses
`CurvyTronSourceEnv` plus the debug occupancy renderer and does not mutate the
scalar/ray path.

## Current Observation, Action, Reward

Reset and step observations are LightZero dicts:

```text
observation: np.float32 shape (106,)
action_mask: np.int8 shape (3,)
to_play: -1
timestep: int
```

Observation schema:

```text
curvyzero_egocentric_rays/v0
24 rays x 4 ray channels + 10 scalars = 106 values
```

Debug visual observation schema:

```text
curvyzero_debug_occupancy_gray64/v0
raw renderer: uint8[1,64,64] CHW
LightZero-facing env payload: float32[1,64,64] CHW in [0,1]
model stack target: float32[4,64,64]
truth level: debug_non_fidelity
source_fidelity_level: none
```

Action schema:

```text
curvyzero_turn3/v0
0 = left
1 = straight
2 = right
```

Current scalar wrapper policy shape:

```text
LightZero controls one fixed ego player: player_0
wrapper fills player_1 with fixed straight action: opponent_action_id=1
opponent_policy_id: curvyzero_fixed_action_opponent
to_play stays -1
```

Current debug visual wrapper policy shape:

```text
LightZero controls one fixed ego player: player_0
wrapper fills player_1 with fixed straight action: opponent_action_id=1
opponent_policy_id: curvyzero_debug_visual_fixed_straight_opponent
to_play stays -1
```

Reward schema:

```text
curvyzero_sparse_round_outcome/v0
nonterminal reward: 0.0
winner reward: 1.0
loser reward: -1.0
draw/truncation reward: 0.0
returned reward shape: scalar
```

Done is `terminated[ego] or truncated[ego]`. Terminal info includes
`eval_episode_return`, `final_observation`, `final_reward_map`,
`terminal_reason`, `winner_ids`, `loser_ids`, `terminated`, `truncated`,
`needs_reset`, `joint_action`, schema ids/hashes, and `trace_hash`.

## What Passed

Local fallback probe:

```text
uv run python -m curvyzero.training.curvyzero_lightzero_runtime_probe
```

Result:

```text
ok: true
call_policy: does_not_train; does_not_call_lzero_entrypoints
local packages: LightZero missing, DI-engine missing, torch missing, gym missing
direct env: ok
terminal env: ok
real_ding_base_env_timestep: false
```

Focused local tests:

```text
uv run pytest tests/test_curvyzero_lightzero_runtime_probe.py tests/test_curvyzero_lightzero_env.py tests/test_curvyzero_lightzero_smoke.py -q
```

Result:

```text
10 passed, 1 skipped in 0.07s
```

The skip is expected when local DI-engine/LightZero is not installed.

Installed-runtime smoke:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_config_import_smoke --seed 0
```

Result:

```text
ok: true
packages:
  LightZero: 0.2.0
  DI-engine: 0.5.3
  torch: 2.11.0
  gym: 0.25.1
  numpy: 1.26.4
imports lzero/ding/torch/gym/custom env: ok
env_factory: ok
direct_env: ok
terminal_env: ok
real_ding_base_env_timestep: true
problems: []
remote_elapsed_sec: 5.745196
```

Installed-runtime debug visual smoke:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_debug_visual_config_import_smoke --seed 0
```

Result:

```text
ok: true
packages:
  LightZero: 0.2.0
  DI-engine: 0.5.3
  torch: 2.11.0
  gym: 0.25.1
  numpy: 1.26.4
imports lzero/ding/torch/gym/custom env: ok
env_factory: ok
direct_env: ok
real_ding_base_env_timestep: true
LightZero-facing env payload: float32[1,64,64]
model stack target: float32[4,64,64]
action_space_size: 3
problems: []
remote_elapsed_sec: 6.192897
```

Modal run URL from visual debug pass:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-B7QAUgQwVKY5TktrgQOjrQ
```

Modal run URL from scalar pass:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-hsStkD8eC2dPmuxRQd5W8y
```

## New Scalar Trainer-Config Scaffold

Coach now has a dry scalar/ray trainer-plumbing config scaffold:

```text
src/curvyzero/training/curvyzero_lightzero_train_config_smoke.py
```

Local dry command:

```text
uv run python -m curvyzero.training.curvyzero_lightzero_train_config_smoke --seed 0
```

This command does not train, does not call `lzero.entry.train_muzero`, and does
not touch visual adapter plumbing. In a local runtime without LightZero it uses
a fallback surface-only template and marks that clearly:

```text
mode: dry_config_builder_validator_only
call_policy: does_not_train; does_not_call_lzero_entrypoints
called_train_muzero: false
target_boundary: scalar/ray curvyzero_v0_lightzero only
quality_claim: none
source_fidelity_claim: none
template.status: fallback_surface_only_template
template.trainable_config_status: not_verified_without_installed_lightzero_template
```

The patched scalar surface it validates is:

```text
env.type: curvyzero_v0_lightzero
env.import_names: ["curvyzero.training.curvyzero_lightzero_env"]
env_manager.type: base
policy.type: muzero
policy.import_names: ["lzero.policy.muzero"]
model_type: mlp
observation_shape: 106
action_space_size: 3
frame_stack_num: 1
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
num_simulations: 2
batch_size: 8
update_per_collect: 1
max_env_step: 8
max_train_iter: 1
support_scale: 5
reward_support_size: 11
value_support_size: 11
opponent: fixed straight action, opponent_action_id=1
```

If this helper is run inside an installed LightZero runtime, Coach can require
the real CartPole MLP template and inspect DI-engine's compiled config before
any trainer entrypoint:

```text
uv run python -m curvyzero.training.curvyzero_lightzero_train_config_smoke \
  --seed 0 \
  --require-lightzero-template \
  --compile-installed-lightzero
```

That installed-runtime command is still a dry config/compile check only. It
must be reported as trainer plumbing only, not as CurvyTron learning evidence.

## What Is Ready

- The scalar/ray `curvyzero_v0_lightzero` env can be imported by installed
  LightZero/DI-engine.
- The debug visual `curvyzero_debug_visual_tensor_lightzero` env can be
  imported by installed LightZero/DI-engine.
- DI-engine's env factory can construct collector/evaluator config rows for it.
- Reset, fixed seed, fixed-action step, direct env step, env-factory step, and
  a tiny terminal step pass.
- Installed runtime returns real `ding.envs.env.base_env.BaseEnvTimestep`.
- Scalar spaces are present in installed runtime:
  `Box(-1.0, 1.0, (106,), float32)`, `Discrete(3)`, and scalar reward `Box`.
- Debug visual spaces are present in installed runtime:
  `Box(0.0, 1.0, (1,64,64), float32)`, `Discrete(3)`, and scalar reward
  `Box`.
- The smoke does not call `lzero.entry.train_muzero`.
- Identity guards say this is not CartPole, not Atari, not ALE, and not
  `PongNoFrameskip-v4`.
- A dry scalar trainer-config scaffold now validates the tiny MLP surface Coach
  would use for a first trainer plumbing attempt. Local fallback validation
  passes, but installed LightZero compile inspection still needs to be run in a
  LightZero runtime before any real trainer call.

## Survival-Time Config Scaffold

Coach now also has a separate dry scalar/ray trainer-plumbing config scaffold
for the survival-time wrapper:

```text
src/curvyzero/training/curvyzero_survival_time_lightzero_train_config_smoke.py
```

Local dry command:

```text
uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke --seed 0
```

This command does not train, does not call `lzero.entry.train_muzero`, and keeps
reward survival-only:

```text
env.type: curvyzero_survival_time_lightzero
env.import_names: ["curvyzero.training.curvyzero_survival_time_lightzero_env"]
env_id: CurvyZeroSurvivalTimeLightZero-v0
reward_schema_id: curvyzero_survival_time/v0
terminal outcome bonus: 0.0
loser penalty: 0.0
winner bonus: 0.0
model_type: mlp
observation_shape: 106
action_space_size: 3
frame_stack_num: 1
collector_env_num: 1
evaluator_env_num: 1
num_simulations: 2
batch_size: 8
max_env_step: 8
max_train_iter: 1
support_scale: 8
```

Local fallback validation passes. The exact next installed-runtime no-train
compile command is:

```text
uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke \
  --seed 0 \
  --require-lightzero-template \
  --compile-installed-lightzero
```

That command is still dry config/compile inspection only. It must not be
reported as CurvyTron learning evidence, and it must not be reported as the
visual training path. It is scalar/ray MLP plumbing.

## Stacked Debug Visual Survival Adapter

Optimizer feedback is correct: the current debug visual env emits
`float32[1,64,64]`, while the visual MuZero target is `float32[4,64,64]`.
There is no current proof that LightZero stacks CurvyTron debug visual frames.

Small safe follow-up added a separate wrapper-stacked debug visual survival
adapter:

```text
local module: curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke
registered module: curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env
registered env.type: curvyzero_stacked_debug_visual_survival_lightzero
registered env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
observation_schema_id: curvyzero_stacked_debug_occupancy_gray64_survival_time/v0
raw_observation_schema_id: curvyzero_debug_occupancy_gray64/v0
raw frame: float32[1,64,64]
LightZero payload: float32[4,64,64]
frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
truth level: debug/non-fidelity only
reward_schema_id: curvyzero_survival_time/v0
```

Local command:

```text
uv run python -m curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke
```

Result:

```text
ok: true
mode: no_train_stacked_debug_visual_survival_collect_replay_sample_only
collected fixed-action rows: 3
sample row observation_shape: [4,64,64]
mcts_search: not_run
learner_profile: not_run
called_train_muzero: false
```

This is not a full loop claim. It proves only fixed-action debug visual collect,
temporary replay-row packaging, and sample shape against a wrapper-owned stack.

## Installed Stacked Visual Survival Profile

The bounded installed-LightZero visual profile now passes in Modal:

```text
command:
  uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_profile \
    --seed 0 \
    --steps 4 \
    --num-simulations 2
app: ap-EubQiSXUN6L5PD1q0wfkfC
LightZero: 0.2.0
DI-engine: 0.5.3
torch: 2.11.0
gym: 0.25.1
numpy: 1.26.4
ok: true
```

Passed stages:

```text
wrapper-stacked debug visual collect: 4 rows
MCTS/search API: MuZeroPolicy.eval_mode.forward
replay row build: ok
replay sample/batch: observation [2,4,64,64]
learner forward/loss API: MuZeroPolicy.learn_mode.forward
optimizer step: blocked by no-op patch
model_parameters_changed: false
model_state_restored: true
called_train_muzero: false
trainer_claim: none
quality_claim: none
```

This is still not a trainer run. It is a debug-fidelity, survival-only plumbing
gate for collect -> MCTS/search -> replay -> sample -> learner forward/loss.

## What Is Not Ready

- No CurvyTron LightZero training run is proven.
- No source-faithful visual tensor exists yet.
- No proof exists that LightZero/DI-engine stacks the original CurvyTron debug
  visual env's `[1,64,64]` frames into `[4,64,64]`; the new stacked adapter does
  wrapper-local stacking only.
- No separate visual trainer config/run artifact exists yet.
- No full simultaneous multiplayer self-play is proven.
- No rotating ego rows, current-policy learned opponents, or joint-action MCTS
  are in this wrapper. A narrow frozen-checkpoint opponent hook now exists for
  the stacked debug visual single-ego trainer lane; label it
  learner-vs-frozen-checkpoint, not self-play.
- No claim of source-faithful browser/canvas visuals, exact trail gaps, holes,
  bonuses, lifecycle/spawn fidelity, or production reset/autoreset completeness.
- No claim that the scalar sparse reward is the final Coach reward. It is not
  the desired survival-time objective yet.
- No replay quality, scorecard, eval ladder, or policy-quality claim exists for
  CurvyTron.

## What Coach Can Try Next

1. Keep Pong as the proof lane. Use the stock/control Pong results to decide
   whether the LightZero setup is credible enough for more CurvyTron work.
2. Re-run the CurvyTron installed no-train smoke before any CurvyTron trainer
   attempt:

   ```text
   uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_config_import_smoke --seed 0
   ```

3. Inspect the dry scalar trainer-plumbing config scaffold:

   ```text
   uv run python -m curvyzero.training.curvyzero_lightzero_train_config_smoke --seed 0
   ```

   In an installed LightZero runtime, require the real CartPole MLP template and
   compile the config before any trainer call:

   ```text
   uv run python -m curvyzero.training.curvyzero_lightzero_train_config_smoke \
     --seed 0 \
     --require-lightzero-template \
     --compile-installed-lightzero
   ```

   This is still dry config/compile inspection only.

3a. Prefer the survival-time scaffold for the first CurvyTron Coach reward:

   ```text
   uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke --seed 0
   ```

   In an installed LightZero runtime, require the real CartPole MLP template and
   compile the survival-time config before any trainer call:

   ```text
   uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke \
     --seed 0 \
     --require-lightzero-template \
     --compile-installed-lightzero
   ```

4. If trying a first CurvyTron LightZero trainer config, keep it explicitly
   scalar and tiny:

   ```text
   env.type: curvyzero_survival_time_lightzero
   env.import_names: ["curvyzero.training.curvyzero_survival_time_lightzero_env"]
   reward_schema_id: curvyzero_survival_time/v0
   model_type: mlp
   observation_shape: 106
   action_space_size: 3
   frame_stack_num: 1
   collector_env_num: 1
   evaluator_env_num: 1
   num_simulations: 2
   opponent: fixed straight, named in info
   ```

   Report it as a trainer plumbing experiment only.

5. For visual CurvyTron, coordinate with Optimizer. The active current target
   is source-state gray64 `uint8[1,64,64]` / stacked training tensor.
   Browser/canvas pixels are optional later debug/human evidence, and the old
   `curvyzero_debug_occupancy_gray64/v0` surface is historical debug/profiling
   smoke only. Either prove LightZero stacks the current source-state env, or
   use a clearly labeled wrapper-stacked env:

   ```text
   env.type: curvyzero_stacked_debug_visual_survival_lightzero
   env.import_names: ["curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env"]
   model_type: conv
   observation_shape: [4,64,64]
   action_space_size: 3
   env.frame_stack_num: 1
   policy.model.frame_stack_num: 1
   policy.model.image_channel: 4
   frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
   reward_schema_id: curvyzero_survival_time/v0
   ```

6. The visual trainer prerequisite is the bounded profile artifact:

   ```text
   visual collect -> MCTS/search -> replay -> sample -> learner profile
   env.type: curvyzero_stacked_debug_visual_survival_lightzero
   debug_fidelity_only: true
   source_fidelity_claim: none
   reward_schema_id: curvyzero_survival_time/v0
   collector_env_num: 1
   num_simulations: 2
   ```

   This bounded profile now passes in Modal through learner forward/loss. Report
   it as visual plumbing/profile only, not learning evidence and not
   source-fidelity visual CurvyTron.

   The separate trainer artifact now exists:

   ```text
   src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
   ```

   The first successful waited run:

   ```text
   run_id: curvytron-visual-survival-debug-lz-s0
   attempt_id: train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510
   Modal app: ap-zXhm4AFaq8MI78MDtRNdjd
   called_train_muzero: true
   ok: true
   env.type: curvyzero_stacked_debug_visual_survival_lightzero
   reward_schema_id: curvyzero_survival_time/v0
   observation_shape: [4,64,64]
   env.frame_stack_num: 1
   policy.model.frame_stack_num: 1
   policy.model.image_channel: 4
   save_ckpt_after_iter: 1
   action rows: 461
   ego_action_histogram: {"0": 226, "1": 108, "2": 127}
   collapse_warning: null
   mirrored checkpoints: 75
   summary_ref:
     training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/train/summary.json
   ```

   Full run note:

   ```text
   docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md
   ```

   This is now a trainer plumbing success, still debug-fidelity only and not a
   policy-quality claim. Next Coach-facing work should be eval/scorecard and
   checkpoint inspection, not more scalar-only plumbing.

## Must Not Be Claimed

- Do not claim CurvyTron uses ALE.
- Do not claim CurvyTron is Atari-style. Say "LightZero visual stacked-frame
  CurvyTron, non-ALE" when discussing the final visual target.
- Do not claim scalar/ray `[106]` is the final visual target.
- Do not claim the debug occupancy visual tensor is source-faithful.
- Do not claim current scalar smoke proves learning, self-play, or policy
  quality.
- Do not claim current debug visual smoke proves learning, source-fidelity
  visuals, or the final CurvyTron trainer.
- Do not claim the scalar survival wrapper is the visual training path.
- Do not claim the wrapper-stacked debug visual adapter proves LightZero's own
  frame stacking.
- Do not claim visual replay, MCTS/search, or learner-profile readiness until a
  bounded same-artifact profile proves those pieces; current proof is debug
  visual survival profile only, not a trainer.
- Do not claim fixed-opponent single-ego learning is full CurvyTron multiplayer
  self-play.
- Do not claim frozen-checkpoint-opponent single-ego learning is full CurvyTron
  multiplayer self-play.
- Do not claim survival-shaped Pong or dummy Pong proves stock/control Pong.
- Do not claim CurvyTron policy quality until there is a same-run baseline/eval
  story. A debug-fidelity visual trainer run now exists, but it is not an eval.

## File Ownership Note

This handoff update touched scalar plumbing and added a clearly labeled
wrapper-stacked debug visual adapter. It did not add source-fidelity visuals,
prove LightZero frame stacking, or run a trainer.

```text
src/curvyzero/training/curvyzero_lightzero_train_config_smoke.py
src/curvyzero/training/curvyzero_survival_time_lightzero_train_config_smoke.py
src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py
src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_env.py
src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_profile.py
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
docs/experiments/2026-05-10-curvytron-visual-survival-lightzero-train.md
```
