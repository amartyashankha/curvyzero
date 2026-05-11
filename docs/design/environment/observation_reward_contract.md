# Observation, Reward, And Terminal Info Contract

Status: Draft boundary
Date: 2026-05-09

This page is the first concrete contract for what a trainer or coach may expect
from CurvyZero observations, action masks, rewards, done flags, terminal info,
debug events, and replay rows.

## Short Contract

- Real now: `CurvyTronEnv` exists as a toy-v0 single environment. It returns a
  flat privileged debug observation, sparse terminal rewards, per-agent
  `terminated`/`truncated` dicts, and per-agent `infos` with toy-v0 step and
  schema metadata.
- Real now: toy-v0 has `observe(ego_player)`,
  `legal_action_mask(ego_player)`, and errors on missing live-player actions.
  `reset(seed)` still returns only the observation dict, but it now populates
  `last_reset_info`.
- Real now: after toy-v0 returns a terminal or truncated step, another `step()`
  raises until `reset()` is called. There is no hidden autoreset.
- Real now: benchmark/debug vector packers can emit fixed arrays such as
  `obs[B,P,9]`, `reward[B,P]`, `done[B]`, `truncated[B]`, and
  `legal_action_mask[B,P,3]`. These are benchmark surfaces, not the final
  training API.
- Real now, narrowly: `VectorTrainerEnv1v1NoBonus` returns strict public
  1v1/no-bonus `float32[B,2,106]` trainer observations, final arrays, sparse
  terminal rewards, replay-v0 chunks, and reset/step info with
  `native_control_model_id`, `trainer_control_wrapper_id`, and `decision_ms`.
  This is still not a public full CurvyTron env API.
- Not real yet: broad public trainer `reset_many`/`step_many`, broad
  autoreset, JAX/GPU envs, and full source-faithful training observations.
- Current optimizer bridge: source-backed CurvyTron state -> trainer
  `float32[B,P,106]` rays/scalars -> replay-v0 chunks. This is not an
  emulator, ROM, or pixel-observation path; visual LightZero comes later.
- The first trainer action space is `curvyzero_turn3/v0`: `0` left,
  `1` straight, `2` right. The legal mask order is always
  `[left, straight, right]`.
- `step()`, `joint_action`, action ids, and fixed decision cadence are
  trainer-wrapper/replay abstractions. Native CurvyTron source behavior is
  real-time control state advanced by elapsed-millisecond server frames.
  The strict public env is a fixed decision wrapper over that source control
  state, not native discrete simultaneous actions.
- The first debug observation surface is
  `curvyzero_debug_global_player_obs/v0`. It is useful for wiring and
  deterministic smoke tests because it is already close to current code, but it
  is privileged and must be labeled as debug.
- The first trainable observation target is
  `curvyzero_egocentric_rays/v0`, a ray/scalar helper with LightZero flat
  shape `float32[106]`. It should replace debug observations in new
  coach-facing work, while broader source-backed observation coverage remains
  open.
- The first reward target for toy-v0 training is sparse round outcome:
  `0` during play, `+1` for the round winner, `-1` for the loser, and `0` for a
  no-survivor draw or pure truncation.
- `done = terminated OR truncated`. `terminated` means the game produced a real
  round outcome. `truncated` means an external or horizon limit stopped the
  episode.
- Debug events and source/common-trace refs are evidence links. They do not
  belong inside policy observations and must not be required in the hot training
  step.

## Real Now Versus Contract Target

Current single-env reality:

- `CurvyTronEnv.reset(seed)` returns `dict[player_id, np.ndarray]`, not
  `(obs, info)`. It also populates `env.last_reset_info` with reset/schema
  metadata.
- `CurvyTronEnv.observe(ego_player)` returns a copied current debug observation
  for that player.
- `CurvyTronEnv.legal_action_mask(ego_player)` returns the current toy-v0
  `bool[3]` mask.
- With the current 2-player toy-v0 state, each observation is a copied
  `float32[9]` vector containing both players' positions, both headings, both
  alive flags, and normalized tick.
- The trainer-wrapper step method returns `StepResult` with
  `observations`, `rewards`, `terminated`, `truncated`, and `infos`.
- `infos[player_id]` is populated with toy-v0 schema ids/hashes, step/tick
  metadata, terminal reason fields, `done`, `needs_reset`, final observations
  on done transitions, and ref placeholders.
- After a terminal or truncated toy-v0 step, a second `step()` raises
  `RuntimeError` until `reset()` starts a new episode.
- Missing live-player actions are `ValueError`s. Dead players do not require
  actions.

Current batch/debug reality:

- The vector actor/debug benchmarks can pack `obs[B,P,9]`, `reward[B,P]`,
  `done[B]`, `truncated[B]`, `terminated_agent[B,P]`,
  `truncated_agent[B,P]`, `legal_action_mask[B,P,3]`, ego ids, and
  `ego_mask[B,P]`.
- The debug batch reward uses score/round-score deltas plus death evidence. It
  is not the toy-v0 sparse outcome reward contract.
- The batch path is fixture seeded and benchmark local. It is not a declared
  `reset_many`/`step_many` API.

Contract target before coach training treats this as an environment API:

- `reset(seed=...) -> (obs_by_player, reset_info)`. Current toy-v0 still stores
  reset metadata in `last_reset_info` instead.
- `observe(ego_player) -> observation`.
- `legal_action_mask(ego_player) -> bool[3]`.
- `step(joint_action) -> StepResult` for the CurvyZero trainer wrapper. This
  advances one wrapper decision and may hold the converted source controls for
  `decision_ms`.
- `joint_action` is wrapper/replay metadata. It must include every live player
  exactly once in the wrapper decision. Dead players do not require actions.
- Terminal and truncation steps must fill standard info fields.
- Replay rows must store schema ids and enough refs to recreate or debug the
  transition.
- Strict public env reset/step info must state the native control model id, the
  trainer control wrapper id, and the decision window in milliseconds.

## Minimal Debug-To-Training Migration

1. Keep the current flat observation as
   `curvyzero_debug_global_player_obs/v0` and label every run that uses it as
   `debug-observation`.
2. Keep the implemented single-env pieces covered by tests:
   `last_reset_info`, `observe`, `legal_action_mask`, live-action validation,
   and step info/schema metadata.
3. Finish the public API boundary without changing learning semantics:
   `reset(...)->(obs, info)`, replay chunks, and broader terminal/truncation
   coverage.
4. Make the batch debug packer emit the same field names and shapes as the
   future batch contract, while keeping the `debug` schema ids.
5. Write replay chunks from those fields. The replay may contain debug
   observations, but the metadata must say so.
6. Implement the first ego-relative learned observation schema and switch only
   the observation schema id/hash and observation arrays. The action, reward,
   done, info, and replay keys should stay stable.
7. Promote to trainer evidence only after replay readers reject mismatched
   rules, observation, action, and reward hashes, and after terminal-info tests
   cover win, loss, draw, and truncation.

## Action And Legal Mask

Trainer action space id: `curvyzero_turn3/v0`.

Single-player action ids:

| Id | Meaning |
| ---: | --- |
| `0` | turn left |
| `1` | go straight |
| `2` | turn right |

Legal mask:

- Shape: `bool[3]`.
- Order: `[left, straight, right]`.
- Live toy-v0 player: `[true, true, true]`.
- Dead player, post-terminal padding, or inactive ego row:
  `[false, false, false]`.
- Source-style moves `-1/0/1` are adapter internals only. Policy, replay, and
  trainer code use wrapper ids `0/1/2`; those ids are not native CurvyTron
  action ids.

Single-env joint action:

```python
{"player_0": 0, "player_1": 2}
```

Batch action arrays:

- `action_id`: `int8[B,P]`, values in `{0,1,2}` for live ego rows.
- `action_valid`: `bool[B,P]`, true where the action row should be applied.
- `legal_action_mask`: `bool[B,P,3]`.
- `ego_mask`: `bool[B,P]`, true where the player row participates in policy,
  value, and reward losses.

## Observation Surfaces

### Debug Surface: Real Starting Point

Observation schema id: `curvyzero_debug_global_player_obs/v0`.

Single-env shape:

- `obs_by_player[player_id]`: `float32[9]` for the current 2-player toy-v0
  environment.
- Both players currently receive the same global vector.

Feature order:

1. `player_0_x`
2. `player_0_y`
3. `player_1_x`
4. `player_1_y`
5. `player_0_heading`
6. `player_1_heading`
7. `player_0_alive`
8. `player_1_alive`
9. `tick_over_max_ticks`

Batch debug shape:

- `obs`: `float32[B,P,9]`.
- `legal_action_mask`: `bool[B,P,3]`.
- `ego_mask`: `bool[B,P]`.

This surface is intentionally privileged. It can be used for plumbing, replay
format smoke tests, and deterministic canaries. It is not the final learned
observation.

### First Learned Target

Observation schema id: `curvyzero_egocentric_rays/v0`.
Observation schema hash: `61767187ffa4a3a6`.

Target single-env fields:

- `rays`: `float32[24,4]`.
- `scalars`: `float32[10]`.
- `legal_action_mask`: supplied beside the observation as `bool[3]`.
- LightZero wrapper flat pack: `observation` is `float32[106]`, built from
  `rays` row-major followed by `scalars`, and `action_mask` is a separate
  `int8[3]`.

Target batch fields:

- `rays`: `float32[B,P,24,4]`.
- `scalars`: `float32[B,P,10]`.
- `legal_action_mask`: `bool[B,P,3]`.
- `ego_mask`: `bool[B,P]`.

The exact ray angle table, scalar order, normalization, adapter shape, and
schema hashes are pinned in
`docs/working/environment/trainer_observation_reward_contract_v0_2026-05-09.md`
and `src/curvyzero/env/trainer_contract.py`. The ray observation implementation
now exists for the current helper surface, but it is not broad source-fidelity
evidence, a vector body-array implementation, or a visual observation path.

## Reward Contract

Reward schema id: `curvyzero_sparse_round_outcome/v0`.

Single-env shape:

- `rewards`: `dict[player_id, float]`.
- Reward is assigned after the trainer-wrapper joint-action step.

Batch shape:

- `reward`: `float32[B,P]`.

Rules:

- Nonterminal movement step: every live player receives `0.0`.
- One survivor at terminal round end: survivor receives `+1.0`; dead players
  receive `-1.0`.
- No-survivor terminal draw: every player receives `0.0`.
- Pure truncation without a game outcome: every player receives `0.0`.
- If a step is both terminated and truncated, terminal outcome reward wins and
  `truncated` records the external cap.

Do not mix this reward id with the current vector debug score/round-score delta
reward. That debug formula needs its own schema id and should remain labeled as
debug until it is intentionally promoted or replaced.

## Done, Terminated, And Truncated

Single-env fields:

- `terminated`: `dict[player_id, bool]`.
- `truncated`: `dict[player_id, bool]`.
- `done`: derived by callers as `terminated[player] or truncated[player]`.

Batch fields:

- `terminated`: `bool[B]` for environment row outcome.
- `truncated`: `bool[B]` for environment row truncation.
- `done`: `bool[B]`, equal to `terminated OR truncated`.
- `terminated_agent`: `bool[B,P]`.
- `truncated_agent`: `bool[B,P]`.

Meaning:

- `terminated=true` means normal game rules produced a round end.
- `truncated=true` means a time limit, rollout horizon, event overflow,
  artifact cap, or infrastructure limit stopped the row.
- Public autoreset is future work. The internal debug actor bridge may reset
  fixture-seeded rows after staging the terminal debug transition into its
  in-memory replay chunk; that narrow bridge behavior is not a public
  `reset_many`/`step_many` contract and is not the final training env.
- Current toy-v0 refuses post-terminal or post-truncation stepping until reset,
  so callers cannot accidentally train through hidden autoreset behavior.

## Standard Info Fields

Reset info should include:

- `episode_id`
- `seed`
- `ruleset_id`
- `rules_hash`
- `observation_schema_id`
- `observation_schema_hash`
- `action_space_id`
- `action_space_hash`
- `reward_schema_id`
- `reward_schema_hash`
- `player_ids`
- `max_players`

Current toy-v0 note: reset metadata is stored in `env.last_reset_info` after
`reset(seed)`. The reset call does not return `(obs, info)` yet.

Step info should include the same schema ids plus:

- `step_index`
- `tick_index`
- `terminal_reason`: one of `none`, `survivor_win`, `all_dead_draw`,
  `timeout`, `horizon_truncated`, `event_overflow_truncated`, or
  `infra_truncated`.
- `winner_ids`
- `loser_ids`
- `death_player_ids`
- `draw`
- `timeout`
- `truncation_reason`
- `done`
- `terminated`
- `truncated`
- `needs_reset`
- `final_observation`
- `event_ref`
- `event_range`
- `state_ref`
- `trace_ref`

Current toy-v0 note: per-player `infos` now fill these fields where the toy
env can know them, using `None` for the optional refs. On terminal or truncated
steps, `final_observation` is the final per-player debug observation and
`needs_reset` is true.

For single envs, `infos[player_id]` may repeat the same round-level fields for
each player. For batch envs, these fields should be arrays where possible and
refs where variable length data would make arrays awkward.

## Event And Ref Fields

Events are optional debug evidence, not policy input.

Use refs when event/state/trace data is large or unstable:

- `event_ref`: artifact id, file path, or in-memory handle for event rows.
- `event_range`: `[start, stop)` range for events caused by this transition.
- `state_ref`: optional state snapshot id or checksum.
- `trace_ref`: optional common-trace or source-trace artifact id.
- `scenario_ref`: optional scenario id/path when a reset came from a fixture.

The policy observation must be reconstructable without reading these refs.
Debugging and fidelity tools may follow refs after the step returns.

## Replay Metadata

Replay chunks must store enough metadata to reject incompatible rows before a
learner reads tensors.

Current actor-loop sample bridge status:

- `scripts/benchmark_vector_actor_loop_bridge.py --sample-only` emits
  `sample_contract_metadata` with replay, observation, action-space, and reward
  schema ids plus deterministic schema hashes.
- It also reports `ruleset_id` from the selected fixtures and a deterministic
  `rules_hash`, but that hash is only a selected-fixture compatibility guard.
  It is not a full source rules hash and must not be promoted to production
  replay compatibility.
- `created_at` is deliberately `null` in the deterministic sample. A real
  replay writer must fill it.

Chunk-level metadata:

- `replay_schema_id`
- `replay_schema_hash`
- `ruleset_id`
- `rules_hash`
- `observation_schema_id`
- `observation_schema_hash`
- `action_space_id`
- `action_space_hash`
- `reward_schema_id`
- `reward_schema_hash`
- `env_impl_id`
- `env_impl_version`
- `native_control_model_id`
- `trainer_control_wrapper_id`
- `decision_ms`
- `created_at`
- `producer`

Row-level metadata:

- `episode_id`
- `env_id`
- `player_id`
- `step_index`
- `seed`
- `obs` or `obs_ref`
- `next_obs` or `next_obs_ref`
- `legal_action_mask`
- `action_id`
- `joint_action_id` or `joint_action_ref` for the wrapper/replay action map
- `reward`
- `terminated`
- `truncated`
- `done`
- `terminal_reason`
- `winner_ids`
- `loser_ids`
- `death_player_ids`
- `draw`
- `timeout`
- `truncation_reason`
- `event_ref`
- `state_ref`
- `trace_ref`
- `behavior_policy_id`
- `opponent_policy_ids`

Policy logits, visit counts, root values, value targets, and priority fields are
learner/search metadata. They may be added beside this core contract, but they
do not replace the environment fields.

## Explicitly Not Ready

- No source-faithful training environment is declared ready.
- The narrow ray/scalar learned-observation schema and helpers exist; broader
  source-faithful learned-observation coverage is not declared ready.
- No final vector `reset_many` or `step_many` contract is declared ready.
- Public-facing autoreset planning/apply helpers exist, but no public
  trainer-env autoreset contract is declared ready.
- Toy-v0 explicitly refuses post-terminal/post-truncation `step()` calls until
  `reset()`; this is a guard against hidden autoreset, not an autoreset API.
- No JAX-native, GPU-native, distributed, or Modal hot-loop environment is
  declared ready.
- Narrow replay-v0 writer/reader compatibility checks exist; no production
  full-env replay compatibility guarantee is declared ready.
- Toy-v0 step infos now fill the standard toy fields above, but no replay or
  production source-faithful terminal-info guarantee is declared ready.
- No policy should treat debug global observations, source traces, browser
  screenshots, or event rows as proof of source-fidelity training readiness.
