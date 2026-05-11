# CurvyTron LightZero Survival-Time Contract - 2026-05-10

Purpose: first tiny CurvyTron survival-time reward wrapper/smoke for LightZero
plumbing. This is not a trainer run and does not claim learning.

## Wrapper Identity

Local no-train smoke:

```text
module: curvyzero.training.curvyzero_survival_time_lightzero_smoke
class: CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv
env.type: curvyzero_survival_time_lightzero_local_smoke
env_id: CurvyZeroSurvivalTimeLightZeroLocalSmoke-v0
```

Registered DI-engine/LightZero wrapper:

```text
module: curvyzero.training.curvyzero_survival_time_lightzero_env
class: CurvyZeroSurvivalTimeLightZeroEnv
env.type: curvyzero_survival_time_lightzero
env_id: CurvyZeroSurvivalTimeLightZero-v0
import_names: ["curvyzero.training.curvyzero_survival_time_lightzero_env"]
```

## Observation

The wrapper keeps the existing scalar/ray interface:

```text
observation: np.float32 shape (106,)
schema: curvyzero_egocentric_rays/v0
layout: 24 rays x 4 channels + 10 scalars
action_mask: np.int8 shape (3,)
to_play: -1
timestep: current toy-env tick
```

## Actions

The wrapper keeps the existing 3-action interface:

```text
0 = left
1 = straight
2 = right
```

LightZero controls one ego player, default `player_0`. The opponent is the same
fixed straight opponent used by the scalar smoke path.

## Terminal Condition

The survival wrapper reports the terminal transition before reset. The practical
done boundary is:

```text
done = ego died OR current toy round ended OR max_ticks truncation
```

The current toy CurvyTron env requires reset when either player is the last
survivor, so the wrapper preserves that no-hidden-autoreset boundary. The reward
is still survival-time only; there is no winner bonus and no loser penalty.

## Reward

Reward schema:

```text
reward_schema_id: curvyzero_survival_time/v0
terminal step counting rule: post_transition_alive
```

Reward rule:

```text
reward = 1.0 if the controlled player is alive after the wrapper step
reward = 0.0 if the controlled player is dead after the wrapper step
```

Episode return is the sum of survived wrapper steps. A terminal win, loss, draw,
or timeout does not add any extra outcome reward.

## Verification

No pytest and no trainer entrypoint were run.

Commands used:

```text
uv run python -m py_compile src/curvyzero/training/curvyzero_survival_time_lightzero_smoke.py src/curvyzero/training/curvyzero_survival_time_lightzero_env.py
uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_smoke
uv run python -c 'from curvyzero.env import CurvyTronConfig; from curvyzero.training.curvyzero_survival_time_lightzero_env import CurvyZeroSurvivalTimeLightZeroEnv; env=CurvyZeroSurvivalTimeLightZeroEnv({"env_config": CurvyTronConfig(action_repeat=1)}); obs=env.reset(seed=11); ts=env.step(1); print({"obs_shape": obs["observation"].shape, "mask_dtype": str(obs["action_mask"].dtype), "reward": ts.reward, "done": ts.done, "reward_schema_id": ts.info["reward_schema_id"]})'
```

Smoke result:

```text
ok: true
reset observation: float32[106]
reset action_mask: int8[3]
first alive step reward: 1.0
ego-death terminal reward: 0.0
ego-death final_reward_map: {"player_0": 0.0, "player_1": 0.0}
reward_schema_id: curvyzero_survival_time/v0
registered wrapper reset/step reward: 1.0
```

## Config Scaffold Status

Added a no-train survival-time trainer-plumbing config scaffold:

```text
module: curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke
call policy: does_not_train; does_not_call_lzero_entrypoints
active env.type: curvyzero_survival_time_lightzero
active env_id: CurvyZeroSurvivalTimeLightZero-v0
active import_names: ["curvyzero.training.curvyzero_survival_time_lightzero_env"]
reward_schema_id: curvyzero_survival_time/v0
reward policy: survival_only, no terminal outcome bonus, no loser penalty
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

Local verification completed:

```text
uv run python -m py_compile src/curvyzero/training/curvyzero_lightzero_train_config_smoke.py src/curvyzero/training/curvyzero_survival_time_lightzero_train_config_smoke.py src/curvyzero/training/curvyzero_survival_time_lightzero_env.py src/curvyzero/training/curvyzero_survival_time_lightzero_smoke.py
uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_smoke
uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke --seed 0
uv run python -m curvyzero.training.curvyzero_lightzero_train_config_smoke --seed 0
```

Result:

```text
survival env smoke: ok=true
survival dry config smoke: ok=true
sparse dry config smoke regression guard: ok=true
called_train_muzero: false
local template status: fallback_surface_only_template
installed LightZero compile: not run locally
```

Exact next no-train installed-runtime command:

```text
uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke \
  --seed 0 \
  --require-lightzero-template \
  --compile-installed-lightzero
```

Blocked gap before any trainer call:

```text
The survival-time config scaffold has only local fallback-template validation in
this workspace. It still needs the installed LightZero/DI-engine compile check
above in a runtime that provides zoo.classic_control.cartpole.config and
ding.config.compile_config. This is still no-train plumbing only.
```

## Visual Distinction

The scalar survival-time wrapper/config is not the CurvyTron visual training
path. It proves only the scalar/ray survival reward contract and a dry MLP
config surface. Do not report it as evidence that visual MuZero can consume
CurvyTron frames.

Optimizer feedback pins the visual blocker:

```text
existing debug visual env payload: float32[1,64,64]
visual MuZero model target: float32[4,64,64]
current proof that LightZero stacks CurvyTron env frames: none
```

Small safe follow-up implemented a separate, clearly labeled stacked debug
visual survival adapter. It stacks frames inside the wrapper; it does not prove
LightZero/DI-engine frame stacking.

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
terminal outcome bonus: 0.0
loser penalty: 0.0
winner bonus: 0.0
```

Verification:

```text
uv run python -m py_compile src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_env.py
uv run python -m curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke
uv run python -c 'from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env import CurvyZeroStackedDebugVisualSurvivalLightZeroEnv; env=CurvyZeroStackedDebugVisualSurvivalLightZeroEnv({"seed": 3, "source_max_steps": 4}); obs=env.reset(seed=3); ts=env.step(1); print({"obs_shape": obs["observation"].shape, "step_shape": ts.obs["observation"].shape, "reward": ts.reward, "schema": ts.info["reward_schema_id"], "frame_stack_owner": ts.info["frame_stack_owner"]})'
```

Result:

```text
stacked debug visual survival smoke: ok=true
reset observation_shape: [4,64,64]
step observation_shape: [4,64,64]
collected fixed-action rows: 3
sample row shape: [4,64,64]
reward per alive step: 1.0
mcts_search: not_run
learner_profile: not_run
called_train_muzero: false
```

Exact missing pieces for the useful visual profile artifact:

```text
Need installed LightZero conv MuZero policy/eval-mode search wired to the
stacked debug visual env before claiming collect -> MCTS/search.

Need a replay-row builder for visual MuZero rows containing stacked observation,
action, search policy/action weights, root value, reward, done, and next stacked
observation before claiming replay compatibility.

Need a sample/batch adapter that feeds those visual rows into the learner input
shape [B,4,64,64] before claiming sample -> learner profile.

Need a bounded profile command that runs only tiny/debug-fidelity visual collect
-> MCTS/search -> replay -> sample -> learner forward/loss plumbing. It must
stay no-train unless explicitly promoted later.
```

## Visual Survival Trainer Update

The survival-time reward contract has now been used by a real installed
LightZero visual trainer run, but only on the debug-fidelity wrapper-stacked
visual env.

```text
run_id: curvytron-visual-survival-debug-lz-s0
attempt_id: train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510
trainer: lzero.entry.train_muzero
env.type: curvyzero_stacked_debug_visual_survival_lightzero
reward_schema_id: curvyzero_survival_time/v0
terminal_outcome_bonus: 0.0
loser_penalty: 0.0
winner_bonus: 0.0
debug_fidelity_only: true
source_fidelity_claim: none
```

Action observability was recorded from the env side:

```text
action telemetry rows: 461
ego_action_histogram: {"0": 226, "1": 108, "2": 127}
done_count: 4
reward_sum: 459.0
reward_mean: 0.9956616052060737
collapse_warning: null
```

Artifacts:

```text
summary_ref:
  training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/train/summary.json
action_observability_ref:
  training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/attempts/train-gpu-l4t4-survival-debug-4096x32-stackfix-20260510/train/action_observability.json
checkpoint_root:
  training/lightzero-curvytron-visual-survival/curvytron-visual-survival-debug-lz-s0/checkpoints/lightzero/
mirrored_checkpoint_count: 75
```

Do not use this as scalar evidence or source-fidelity evidence. It proves only
that the visual survival reward path can enter installed LightZero training,
write checkpoints, and emit action histograms.
