# CurvyTron LightZero Integration Recon - 2026-05-10

Purpose: handoff for the Coach/LightZero CurvyTron lane after reading the
2026-05-10 handoff and the current env/interface code.

## Verdict

CurvyTron is ready for the next no-train LightZero plumbing step, not for a
Coach training claim.

The scalar env and debug visual env already prove import, construction, reset,
seed, step, and `BaseEnvTimestep` conversion in an installed LightZero/DI-engine
runtime. The blocking mismatch for first Coach training is reward: both current
CurvyTron LightZero wrappers return sparse round-outcome rewards. The first
CurvyTron reward for Coach should be survival time only.

Do not start by changing this inside the existing sparse wrapper. Make the
reward contract explicit first, then add a separate survival-time env/config so
old smoke evidence stays honest.

Update after the next tiny plumbing steps: the separate survival-time scalar
wrapper and local no-train dry config scaffold now exist. Local fallback-template
validation passes, and the sparse scaffold still passes as a regression guard.
The scalar survival path is not the visual training path.

Optimizer feedback also pinned the visual shape blocker: the existing debug
visual env emits `float32[1,64,64]`, while visual MuZero wants
`float32[4,64,64]`. There is no proof that LightZero stacks CurvyTron frames.
A separate wrapper-stacked debug visual survival adapter now exists for the
shape-smoke lane only; no MCTS/search, replay compatibility, learner profile,
or trainer run has been launched.

## Current Interfaces

Scalar/ray wrapper:

```text
module: curvyzero.training.curvyzero_lightzero_env
class: CurvyZeroLightZeroEnv
env.type: curvyzero_v0_lightzero
env_id: CurvyZeroLightZero-v0
import_names: ["curvyzero.training.curvyzero_lightzero_env"]
source env: CurvyTronEnv through CurvyZeroLightZeroLocalSmokeEnv
```

Scalar LightZero observation:

```text
dict keys: observation, action_mask, to_play, timestep
observation: np.float32 shape (106,)
schema: curvyzero_egocentric_rays/v0
action_mask: np.int8 shape (3,)
actions: 0=left, 1=straight, 2=right
to_play: -1
reward now: curvyzero_sparse_round_outcome/v0
reward shape: scalar float
done: terminated[ego] OR truncated[ego]
```

Debug visual wrapper:

```text
module: curvyzero.training.curvyzero_debug_visual_lightzero_env
class: CurvyZeroDebugVisualLightZeroEnv
env.type: curvyzero_debug_visual_tensor_lightzero
env_id: CurvyZeroDebugVisualTensorLightZero-v0
import_names: ["curvyzero.training.curvyzero_debug_visual_lightzero_env"]
source env: CurvyTronSourceEnv through debug occupancy renderer
```

Debug visual LightZero observation:

```text
dict keys: observation, action_mask, to_play, timestep
raw env payload: np.float32 shape (1,64,64), range [0,1]
schema: curvyzero_debug_occupancy_gray64/v0
model target: np.float32 shape (4,64,64)
model_type: conv
frame_stack_num: 4
truth level: debug_non_fidelity
source_fidelity_level: none
reward now: sparse terminal outcome, not survival time
```

Both wrappers are single-ego:

```text
ego: player_0 by default
opponent: fixed straight action, action id 1
opponent is not learned
not full multiplayer self-play
not joint-action MCTS
not ALE or Atari
```

## Exact LightZero Contract To Preserve

`reset(seed)` returns:

```text
{
  "observation": np.float32 tensor,
  "action_mask": np.int8[3],
  "to_play": -1,
  "timestep": int
}
```

`step(action)` accepts one scalar action id in `{0,1,2}` and returns:

```text
BaseEnvTimestep(next_obs_dict, reward_float, done_bool, info_dict)
```

The info dict must keep:

```text
ego_player_id
opponent_player_id
joint_action
opponent_action_id
opponent_policy_id
opponent_policy_version
terminal_reason
winner_ids
loser_ids
death_player_ids
done
terminated
truncated
needs_reset
final_observation on done
final_reward_map on done
eval_episode_return on done
schema ids and hashes
trace_hash
```

No hidden autoreset may drop the terminal transition.

## Reward Decision Needed

Current code:

```text
curvyzero_sparse_round_outcome/v0
nonterminal reward: 0.0
winner reward: +1.0
loser reward: -1.0
draw/truncation reward: 0.0
```

First Coach CurvyTron reward should instead be a new survival-time reward:

```text
proposed schema id: curvyzero_survival_time/v0
reward unit: one fixed wrapper decision step, or elapsed source ms for source-backed visual
nonterminal reward: positive survival increment for ego
terminal outcome bonus: none
loser penalty: none
draw bonus: none
winner bonus: none
episode return: total ego survival time
```

The exact terminal-step counting rule still needs to be pinned before code:

```text
option A: reward +1 for every transition started while ego is alive
option B: reward +1 only if ego remains alive after the transition
```

Picked in the survival-time contract and artifacts:

```text
terminal step counting rule: post_transition_alive
reward = 1.0 if ego is alive after the wrapper step, else 0.0
```

## Blockers Before A Training Claim

- No CurvyTron LightZero trainer run exists.
- The survival-time reward wrapper and local dry config scaffold exist, but
  installed LightZero/DI-engine compile inspection has not been run for that
  scaffold yet.
- No survival-time trainer run or scorecard exists.
- The scalar wrapper is toy/project `CurvyTronEnv`, not source-fidelity CurvyTron.
- The visual wrapper is debug occupancy only, not source-faithful pixels.
- The original debug visual wrapper emits `[1,64,64]`; no installed LightZero
  stack proof exists for it.
- The new stacked debug visual survival wrapper emits `[4,64,64]`, but the
  stack is wrapper-owned and debug-fidelity only.
- No bounded visual collect -> MCTS/search -> replay -> sample -> learner
  profile exists yet.
- The opponent is fixed straight, not learned or checkpointed.
- No rotating ego rows or full simultaneous multi-agent search are proven.
- No CurvyTron scorecard/eval ladder proves policy quality.

## First Safe Integration Step

Completed locally: added a separate no-train survival-time scalar wrapper and
dry config scaffold, without changing the sparse wrapper.

Implemented scope:

```text
env.type: curvyzero_survival_time_lightzero
env_id: CurvyZeroSurvivalTimeLightZero-v0
reuse observation: curvyzero_egocentric_rays/v0, float32[106]
reuse actions: curvyzero_turn3/v0
reuse single ego + fixed straight opponent
change only reward schema and reward calculation
do not call train_muzero
do not touch debug visual wrapper
```

Acceptance checks:

```text
local reset/step test passes
terminal test proves no outcome bonus or loser penalty
runtime probe returns real ding.envs.BaseEnvTimestep when installed
identity guard says non-ALE, non-Atari, non-Pong
info records reward_schema_id=curvyzero_survival_time/v0
eval_episode_return equals accumulated survival time
```

Exact next no-train installed-runtime command:

```text
uv run python -m curvyzero.training.curvyzero_survival_time_lightzero_train_config_smoke \
  --seed 0 \
  --require-lightzero-template \
  --compile-installed-lightzero
```

Only after that succeeds should Coach consider a tiny trainer plumbing run:

```text
env.type: curvyzero_survival_time_lightzero
model_type: mlp
observation_shape: 106
action_space_size: 3
frame_stack_num: 1
collector_env_num: 1
evaluator_env_num: 1
num_simulations: 2
opponent: fixed straight
```

Report that run as trainer plumbing only unless a separate scorecard proves
learning.

## Visual Shape Follow-Up

Small safe follow-up added a separate stacked debug visual survival wrapper:

```text
local module: curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke
registered module: curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env
registered env.type: curvyzero_stacked_debug_visual_survival_lightzero
registered env_id: CurvyZeroStackedDebugVisualSurvivalLightZero-v0
observation_schema_id: curvyzero_stacked_debug_occupancy_gray64_survival_time/v0
raw_observation_schema_id: curvyzero_debug_occupancy_gray64/v0
raw frame shape: [1,64,64]
LightZero payload shape: [4,64,64]
frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
reward_schema_id: curvyzero_survival_time/v0
terminal outcome bonus: 0.0
loser penalty: 0.0
winner bonus: 0.0
source_fidelity_claim: none
```

Local verification:

```text
uv run python -m py_compile src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_smoke.py src/curvyzero/training/curvyzero_stacked_debug_visual_survival_lightzero_env.py
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

Exact missing pieces for Optimizer's useful next artifact:

```text
1. Installed LightZero conv MuZero eval-mode search against the stacked debug
   visual env.
2. Visual MuZero replay rows with stacked obs, action, search policy/action
   weights, root value, survival reward, done, and next stacked obs.
3. Replay sampler/batcher that produces learner inputs shaped [B,4,64,64].
4. Bounded profile command for debug-fidelity visual collect -> MCTS/search ->
   replay -> sample -> learner forward/loss plumbing.
```

Do not call that a full loop or learning result. It is a tiny visual plumbing
profile only.

## Files Read

- `docs/working/training/curvytron_lightzero_coach_handoff_2026-05-10.md`
- `docs/working/training/lightzero_environment_handoff_2026-05-09.md`
- `docs/working/environment/trainer_observation_reward_contract_v0_2026-05-09.md`
- `src/curvyzero/training/curvyzero_lightzero_env.py`
- `src/curvyzero/training/curvyzero_lightzero_smoke.py`
- `src/curvyzero/training/curvyzero_lightzero_runtime_probe.py`
- `src/curvyzero/training/curvyzero_debug_visual_lightzero_env.py`
- `src/curvyzero/training/curvyzero_debug_visual_lightzero_smoke.py`
- `src/curvyzero/training/curvyzero_debug_visual_lightzero_runtime_probe.py`
- `src/curvyzero/training/curvytron_visual_observation.py`
- `src/curvyzero/env/trainer_contract.py`
- `src/curvyzero/env/trainer_observation.py`
- `src/curvyzero/env/core.py`
- `src/curvyzero/env/source_trainer_adapter.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_config_import_smoke.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_debug_visual_config_import_smoke.py`
- `tests/test_curvyzero_lightzero_smoke.py`
- `tests/test_curvyzero_debug_visual_lightzero_smoke.py`
