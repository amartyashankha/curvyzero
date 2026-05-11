# Trainer Observation And Reward Contract V0

Status: narrow contract plus first toy/grid helper and small fixture canaries
Date: 2026-05-09

This page pins the first trainer-facing observation, action-mask, reward,
done/truncated, and info contract. It is deliberately smaller than production
observation work: the first pure ray/scalar helper exists, but there is no
production LightZero env wrapper, no public full env API, and no broad
source-fidelity claim. The current narrow optimizer bridge is source-backed
CurvyTron state -> trainer `float32[B,P,106]` rays/scalars -> replay-v0 chunks.
The strict public 1v1/no-bonus env can feed that bridge and expose a strict
replay/profile manifest, but it remains a narrow fixed-decision wrapper over
source controls. It is proof/profiling infrastructure, not the destination and
not the product runtime. The one runtime under hardening is
`VectorMultiplayerEnv`. This bridge is not a ROM, emulator,
pixel-observation, or visual LightZero path. Local and installed no-train
LightZero-shaped smokes exist separately; they prove adapter/config plumbing
only, not training quality or environment fidelity.
The strict wrapper's restrictions are temporary explicit non-fidelity profile
choices, not the reconstruction path and not an excuse to skip source-default
CurvyTron behavior.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

This is a CurvyZero trainer wrapper/schema contract, not native CurvyTron source
semantics. Native source behavior is real-time player control state advanced by
elapsed-millisecond server frames; `step()`, `joint_action`, fixed decision
cadence, and action ids are adapter/replay conveniences.
The current strict public env reports this boundary with
`native_control_model_id`, `trainer_control_wrapper_id`, and `decision_ms` in
reset and step info.

Code source of truth:

- `src/curvyzero/env/trainer_contract.py`
- `src/curvyzero/env/trainer_observation.py`
- `src/curvyzero/env/vector_trainer_observation.py`
- `tests/test_trainer_contract.py`
- `tests/test_vector_lifecycle.py`
- `scenarios/environment/observation/obs_empty_arena_geometry_v0.json`
- `scenarios/environment/observation/obs_source_movement_empty_multistep_v0.json`

Current implementation status:

- `observe_egocentric_rays_v0(state, config, ego_player_id, ...)` returns
  structured `rays float32[24,4]`, `scalars float32[10]`, flat
  `observation float32[106]`, env `action_mask bool[3]`, LightZero
  `action_mask int8[3]`, `to_play=-1`, sparse reward, and compact reward info.
- The helper is deterministic and pure for the current `EnvState`; it reads toy
  grid occupancy for trail channels and current player positions for the
  opponent-head channel.
- `observe_vector_1v1_egocentric_rays_v0(...)` returns the same pinned
  trainer surface from vector runtime arrays for the strict 1v1/no-bonus slice.
  It raycasts against vector body circles, not fake occupancy.
- `build_final_trainer_transition_1v1_no_bonus_rows(...)` builds
  `float32[B,2,106]` final observation arrays and sparse final reward maps from
  `terminal_reason`/`winner`/`draw`, then feeds the autoreset planner in the
  strict lifecycle proof.
- A first analytic observation manifest exists at
  `scenarios/environment/observation/obs_empty_arena_geometry_v0.json`. Tests
  load it to pin empty-arena p0/p1 perspective symmetry, a scoped
  no-absolute-position leak check for non-wall rays plus scalars, and the
  borderless wall-channel no-hit rule.
- A second, deliberately small source-referenced observation canary exists at
  `scenarios/environment/observation/obs_source_movement_empty_multistep_v0.json`.
  It distills ticks `0` and `3` from the trusted
  `source_kinematics_straight_multistep` expected frames into empty-occupancy
  `EnvState` snapshots. Tests pin schema/hash, source fixture reference, empty
  trail channels, p0/p1 non-wall perspective symmetry, forward opponent-head ray
  distance decreasing as source movement closes the gap, and tick scalar
  encoding.
- Focused local check after this slice:
  `uv run pytest tests/test_trainer_contract.py -q` protects the pinned trainer
  observation/reward helper contract.
- Focused ruff after this slice:
  `uv run ruff check tests/test_trainer_contract.py` is static hygiene around
  the touched tests.

Do not overclaim these helpers or the canaries. The scalar helper is not a full
CurvyTron trail/body/radius oracle. The vector helper is a narrow 1v1/no-bonus
body-circle handoff, not a LightZero adapter, not broad lifecycle, and not
replay-writer evidence. The empty-arena
manifest is analytic only. The movement manifest is a distilled trusted-state
canary, not a source learned-observation oracle, not browser pixel fidelity, and
not lifecycle/spawn/RNG, trail, body, or terminal coverage. The original source
has no learned observation.

## Contract Ids

| Surface | Id | Hash |
| --- | --- | --- |
| Adapter contract | `curvyzero_trainer_adapter_contract/v0` | `c25810c9cc197d27` |
| Observation schema | `curvyzero_egocentric_rays/v0` | `61767187ffa4a3a6` |
| Action space | `curvyzero_turn3/v0` | `957cf262e9a3fb1f` |
| Reward schema | `curvyzero_sparse_round_outcome/v0` | `0ab8bebd84fcb2c5` |
| Native control model | `curvytron_realtime_controls_elapsed_frames/v0` | n/a |
| Trainer control wrapper | `curvyzero_fixed_decision_wrapper/v0` | n/a |

The older research name `curvyzero-observe-v0-rays` is a legacy alias only. New
trainer rows should emit `curvyzero_egocentric_rays/v0`.

The debug schemas remain separate:

- `curvyzero_debug_global_player_obs/v0`
- `curvyzero_vector_debug_obs_reward_packing/v1`
- `curvyzero_debug_score_round_delta_death_penalty/v0`

Do not relabel debug tensors as this trainer contract.

## Observation

Canonical structured observation:

```text
rays      float32[24, 4]
scalars   float32[10]
```

LightZero adapter observation:

```text
{
  "observation": float32[106],
  "action_mask": int8[3],
  "to_play": -1
}
```

The flat `float32[106]` pack order is all `rays` values row-major by ray index
then channel index, followed by all `scalars` in the listed order. The legal
action mask is separate and is not included in the 106 floats.

Ray convention:

- Ego head is the local origin.
- Ego heading is local `+x`.
- Ego-left is local `+y`.
- Ray angles are degrees ego-left/counter-clockwise from heading, modulo 360.
- Angles are:
  `0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345`.
- Distances are clipped to the arena diagonal and divided by the arena
  diagonal.
- `1.0` means no hit in that channel before the clipped limit.
- `0.0` means immediate contact.
- Borderless wrap boundaries are not wall hits; they return no-hit for the wall
  channel unless another obstacle channel is hit.

Ray channel order:

```text
0 wall_or_out_of_bounds
1 own_trail
2 opponent_trail
3 opponent_head
```

Scalar order:

```text
0 ego_alive
1 opponent_alive
2 tick_fraction
3 opponent_rel_x_clipped
4 opponent_rel_y_clipped
5 opponent_heading_sin_relative
6 opponent_heading_cos_relative
7 speed_norm
8 turn_rate_norm
9 trail_radius_norm
```

This v0 scalar table is 1v1. A 3+ player observation needs either a new schema
or a documented opponent aggregation before it can be used for training.

Excluded from the learned observation by default: stable player index, seat id,
color, absolute `x/y`, source trace fields, and debug event rows.

## Legal Action Mask

Trainer action ids are fixed:

```text
0 left
1 straight
2 right
```

Canonical env mask dtype is `bool[3]`; the LightZero adapter converts it to
`int8[3]` in the same order.

Mask rules:

- Live active row with all three moves enabled: `[true, true, true]`.
- Strict left/right ruleset: `[true, false, true]`.
- Dead player, inactive row, terminal padding row, or row needing reset:
  `[false, false, false]`.

Source-style moves `-1/0/1` are adapter internals only and must not become the
policy action ids. The fixed `0/1/2` ids are not source-native action ids.

## Reward

Reward schema id: `curvyzero_sparse_round_outcome/v0`.

Reward is aligned as `reward_{t+1}` after the wrapper decision that applies the
wrapper action map for decision `t`.

Rules:

- Nonterminal decision step: `0.0`.
- One survivor at terminal round end: survivor gets `+1.0`; every loser gets
  `-1.0`.
- All-dead terminal draw: every player gets `0.0`.
- Pure truncation with no game outcome: every player gets `0.0`.
- If a row is both terminated and truncated, the terminal outcome reward wins
  and `truncated=true` remains in info.

No shaping terms are part of this schema. Any shaped or debug reward must use a
different schema id or stay in sidecar telemetry.

## Done, Truncated, And Info

`done = terminated OR truncated`.

Meaning:

- `terminated=true`: game rules produced a round outcome.
- `truncated=true`: a horizon, max tick, event overflow, artifact cap, or
  infrastructure limit stopped the row.
- Hidden autoreset is not part of this contract. Final observation, reward, and
  info must be returned before any row reset.

Reset info must include:

```text
episode_id
seed
ruleset_id
rules_hash
observation_schema_id
observation_schema_hash
action_space_id
action_space_hash
reward_schema_id
reward_schema_hash
player_ids
max_players
env_impl_id
native_control_model_id
trainer_control_wrapper_id
decision_ms
```

Step info must include the reset keys plus:

```text
ego_player_id
step_index
tick_index
joint_action
opponent_policy_id
opponent_policy_version
terminal_reason
winner_ids
loser_ids
death_player_ids
draw
timeout
truncation_reason
done
terminated
truncated
needs_reset
final_observation
final_reward_map
event_ref
event_range
state_ref
trace_ref
trace_hash
```

Here `joint_action` is the wrapper/replay action map, not a native CurvyTron
source object. `decision_ms` is the fixed wrapper decision window; native
source behavior remains held controls advanced over elapsed-ms server frames.

LightZero terminal info must also include `eval_episode_return`.

Allowed terminal reasons for this v0 contract are:

```text
none
survivor_win
all_dead_draw
timeout
horizon_truncated
event_overflow_truncated
infra_truncated
```

## Exact Remaining Implementation Tasks

Done in this slice:

- Implemented the first pure `curvyzero_egocentric_rays/v0` helper against the
  current `EnvState` and toy/grid occupancy.
- Added focused tests for shape, dtype, finite values, ray range, flat pack
  order, legal masks, purity/fresh arrays, player perspective, stable schema
  hash, basic trail/head channel separation, survivor/loser reward,
  all-dead draw, pure truncation, and terminal-plus-truncated precedence.
- Added one analytic observation canary manifest and tests for empty-arena
  perspective symmetry, non-wall/scalar no-absolute-position leak scope, and
  borderless wall-channel no-hit behavior.
- Added one distilled source-state canary manifest and tests tied to
  `source_kinematics_straight_multistep` expected frames for empty trail
  channels, non-wall ego perspective, source movement closing distance, and tick
  scalar encoding.
- Added the strict 1v1/no-bonus vector trainer handoff:
  `observe_vector_1v1_egocentric_rays_v0(...)`,
  `build_final_trainer_transition_1v1_no_bonus_rows(...)`, and lifecycle proof
  coverage that no longer uses `target['pos']` or `target['score']` as fake
  terminal observation/reward payloads.

Still remaining:

1. Keep the movement observation canary small while higher-priority lifecycle,
   spawn, and RNG work remains open; broaden source-state-backed observation
   coverage later for `obs_movement_turn_perspective`,
   `obs_trail_gap_hole_safe`, `obs_trail_gap_stored_body`,
   `obs_trail_gap_boundary_body`, `obs_borderless_wrap`,
   `obs_collision_same_frame_death`, and `obs_normal_wall_terminal`.
2. Replace or validate the scalar toy/grid occupancy ray internals against
   source-backed simulator occupancy/collision semantics. Keep the vector
   body-circle path narrow until broader lifecycle and replay wiring land.
3. Add broader perspective and identity-leak checks beyond the symmetric reset
   canary before using this as learned-observation evidence.
4. The local no-train LightZero-shaped smoke wrapper now exists at
   `src/curvyzero/training/curvyzero_lightzero_smoke.py`. It flattens
   rays/scalars to `float32[106]`, converts the mask to `int8[3]`, fills the
   opponent action from a named deterministic policy, preserves final
   observation/reward info, and refuses hidden post-terminal stepping until
   reset.
5. The thin registered wrapper now exists at
   `src/curvyzero/training/curvyzero_lightzero_env.py` as
   `curvyzero_v0_lightzero`. It deliberately reuses the local smoke semantics
   and only adds the DI-engine env type/timestep boundary. Still missing:
   installed-runtime config/import smoke and real training. Keep those separate
   from the local smoke so we do not confuse "wrapper shape works" with
   "trainer is ready."
6. Wire vector trainer observation/reward arrays into replay writer/reader
   compatibility checks that reject mismatched observation, action, reward,
   adapter, and rules hashes.
