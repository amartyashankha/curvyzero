# Source-State Multiplayer LightZero Adapter Spec - 2026-05-13

Status: repo-owned target-row v0, deterministic sample-batch v0, and
fake/injected native `GameSegment` mapping implemented. The separate opt-in
real-LightZero construction helper also exists, but it is construction-smoke
only; buffer/training parity is still open and false-claimed.
Owner surface: trainer replay to training-target adapter.
Depends on:
[source_state_multiplayer_trainer_surface_spec_2026-05-13.md](source_state_multiplayer_trainer_surface_spec_2026-05-13.md).

Purpose: define the smallest honest target step after
`SourceStateMultiplayerTrainerSurface` and
`SourceStateMultiplayerTrainerReplayRecorder`. The new replay arrays prove that
trainer-facing source-state visual batches can be copied over time. The target
row adapter now proves one-step target construction for repo-owned rows, and
the sample-batch layer now proves deterministic batching on top of those rows.
The fake/injected bridge maps rows into injected native `GameSegment`-like
objects without importing LightZero. A separate opt-in helper now constructs
real LightZero-shaped segments as smoke only, not as a change to the
injection-only bridge. That helper stays construction-smoke only until
sampled-target parity through the real `MuZeroGameBuffer` passes. It does not
yet prove real buffer insertion, learner updates, evaluation quality, or true
multiplayer self-play.

Implemented code:

- `src/curvyzero/training/multiplayer_source_state_target_rows.py`
- `src/curvyzero/training/multiplayer_source_state_native_bridge.py`
- `src/curvyzero/training/multiplayer_source_state_lightzero_native_bridge.py`
- `tests/test_multiplayer_source_state_target_rows.py`
- `tests/test_multiplayer_source_state_native_bridge.py`
- `tests/test_multiplayer_source_state_lightzero_native_bridge.py`

## Decision

The first target should be a repo-owned source-state multiplayer target-row
adapter, not a native LightZero `GameSegment` adapter.

Working name:
`SourceStateMultiplayerTargetAdapterV0`.

Output contract:
`curvyzero_source_state_multiplayer_muzero_target_rows/v0`.

Reason: the current replay chunk already has the source-state observations,
legal masks, live-seat maps, joint actions, rewards, done flags, final visual
observations, and per-record policy rows needed to build auditable target rows.
A repo-owned target-row adapter can validate transition alignment and target
semantics without depending on LightZero buffer internals first. That sequence
is now in place: repo-owned rows and sample batches first, injected mapping and
real-LightZero construction smoke after, and real buffer sampled parity later.

## Input Contract

Primary input is one
`SourceStateMultiplayerTrainerReplayChunkV0` produced by
`SourceStateMultiplayerTrainerReplayRecorder`.

Required chunk arrays:

- `observation`: `float32[T,B,P,4,64,64]`.
- `legal_action_mask`: `bool[T,B,P,3]`.
- `lightzero_action_mask`: `bool[T,B,P,3]`.
- `live_mask`: `bool[T,B,P]`.
- `joint_action`: `int16[T,B,P]`.
- `reward`: `float32[T,B,P]`.
- `done`, `terminated`, `truncated`: `bool[T,B]`.
- `final_observation`: `float32[T,B,P,4,64,64]`.
- `final_observation_row_mask`: `bool[T,B]`.
- `final_reward_map`: `float32[T,B,P]`.

Required per-record policy-row arrays:

- `policy_observation`: `float32[R_t,4,64,64]`.
- `policy_action_mask`: `bool[R_t,3]`.
- `policy_env_row`: `int32[R_t]`.
- `policy_player`: `int16[R_t]`.

Required metadata:

- Surface and replay contract ids, render-mode metadata, player count, cadence
  metadata, reward schema, final-observation policy, and native-control wrapper
  metadata from the trainer surface.
- Project-only mode metadata when present. This includes `profile_no_death`,
  no-death/profile modes, optimizer modes, and training-helper modes. These
  modes are valid project additions and must be preserved, but they must be
  marked as not original CurvyTron/source-fidelity behavior.

The adapter also needs policy-row records captured when the shared policy or
search produced actions. These records are sidecars to the replay arrays and
must stay explicit.

Minimal `PolicyRowRecordV0` fields:

- `record_index`: decision record in the replay chunk.
- `policy_row`: row within `chunk.policy_rows[record_index]`.
- `env_row`: must equal `policy_env_row[policy_row]`.
- `player`: must equal `policy_player[policy_row]`.
- `action`: selected action id in `[0, 3)`.
- `action_mask`: copied legal mask for the policy row.
- `policy_target`: normalized action distribution, usually MCTS child visits
  or a declared behavior-policy fallback.
- `root_value`: scalar value estimate from the search or policy.
- `policy_source`: enum such as `mcts_child_visits`, `policy_logits_softmax`,
  or `uniform_legal_fallback_for_smoke`.
- `source_record_ref`: optional run/chunk/episode metadata for audit.

Validation rules:

- Each policy-row record must point to a live row in the chunk at
  `record_index`.
- `policy_row` order must match `np.nonzero(live_mask[record_index])`.
- `action_mask` must equal `policy_action_mask[policy_row]`.
- `policy_target` must have shape `[3]`, be finite, be nonnegative, sum to
  `1.0` within tolerance, and assign zero mass to illegal actions unless the
  policy source explicitly names a diagnostic exception.
- `action` must be legal under the policy row mask.
- Records for terminal rows are allowed only if `live_mask` says the seat was
  live before the transition being built.

## Transition Alignment

Replay record `k` is the decision state. Replay record `k + 1` is the result of
executing the selected player-major joint action.

For each policy row at record `k`, build one transition target when:

- `k + 1 < T`;
- the row/player is live in `live_mask[k]`;
- a matching `PolicyRowRecordV0` exists;
- `joint_action[k + 1, env_row, player]` equals the policy-row action.
- record `k + 1` is a real action-result record. `remove_player(...)`,
  `advance_warmdown(...)`, and reset records are event/setup records with
  `joint_action=-1` padding; they may provide the next observation state, but
  they must not masquerade as the result of a selected policy action.

Transition fields:

- `observation = observation[k, env_row, player]`.
- `action = joint_action[k + 1, env_row, player]`.
- `action_mask = policy_action_mask[k][policy_row]`.
- `policy_target = PolicyRowRecordV0.policy_target`.
- `root_value = PolicyRowRecordV0.root_value`.
- `reward = reward[k + 1, env_row, player]`.
- `done = done[k + 1, env_row]`.
- `terminated = terminated[k + 1, env_row]`.
- `truncated = truncated[k + 1, env_row]`.
- `next_observation = final_observation[k + 1, env_row, player]` when
  `final_observation_row_mask[k + 1, env_row]` is true, otherwise
  `observation[k + 1, env_row, player]`.
- `final_reward = final_reward_map[k + 1, env_row, player]` when the final row
  mask is true, otherwise `reward`.
- `env_row`, `player`, `record_index`, `next_record_index`, and
  `policy_row` as audit metadata.

The reset record is therefore useful as an initial decision state, but its
`joint_action=-1` is not a training action. The terminal record is useful as
the outcome for the previous decision; it should not become a new starting row
unless it still has live policy rows under the explicit contract.
Leave and warmdown records follow the same rule: their live policy rows are
requests for a future decision from the post-event state, not proof that the
event was caused by policy/search.

## Output Rows

Implemented first output is target rows, not sampled native LightZero
segments.

`SourceStateMultiplayerTargetRowsV0`:

- `observation`: `float32[N,4,64,64]`.
- `action`: `int16[N]`.
- `action_mask`: `bool[N,3]`.
- `policy_target`: `float32[N,3]`.
- `root_value`: `float32[N]`.
- `reward`: `float32[N]`.
- `done`, `terminated`, `truncated`: `bool[N]`.
- `next_observation`: `float32[N,4,64,64]`.
- `to_play`: `int64[N]`, always `-1` for this first non-board-game contract.
- `env_row`: `int32[N]`.
- `player`: `int16[N]`.
- `record_index`, `next_record_index`, `policy_row`: integer audit arrays.
- `target_contract_id`,
  `source_replay_contract_id`,
  `surface_schema_id`,
  `policy_row_record_schema_id`,
  `project_mode_metadata`,
  `source_fidelity_claim`,
  `original_curvytron_behavior_claim`,
  `native_game_segment_claim=false`,
  `lightzero_training_integration_claim=false`.

Implemented second layer after row validation:
`SourceStateMultiplayerSampleBatchV0`.

`build_source_state_multiplayer_sample_batch_v0` builds deterministic sample
batches on top of the same target rows and remains repo-owned until native
LightZero buffer parity is proven. Focused target-row/sample-batch tests
reported `12 passed` locally per worker.

## Implementation Order

1. Done: repo-owned target rows from trainer replay arrays plus policy-row
   records.
2. Done: deterministic sample batches over those rows.
3. Done: injection-only `GameSegment`-like mapping that accepts injected
   constructors and does not import LightZero.
4. Done: separate opt-in real-LightZero construction helper in a separate
   module. Keep it construction-smoke only and keep buffer/training claims
   false.
5. Later: real `MuZeroGameBuffer` sampled-target parity for reward, value,
   policy, action, mask, observation, and `to_play`.

Do not mutate the injection-only bridge into a real-LightZero import path. Its
job is to keep the row-to-native-shaped mapping auditable without native
dependency or training claims.

## Native LightZero Bridge Option

The first native-shaped bridge is implemented as an injection-only mapping, not
a real LightZero integration.

Implemented spike: fake/injected `GameSegment` mapping from
`SourceStateMultiplayerTargetRowsV0`, without importing LightZero and without
claiming real buffer parity. Separate opt-in real-LightZero construction smoke
now lives in its own module. Later proof: real LightZero `MuZeroGameBuffer`
insertion and sampled-target parity.

Keep `lightzero_native_game_segment_claim=false` for the injection-only bridge
because it does not construct LightZero's real `GameSegment` class. Keep the
opt-in real-LightZero helper at construction-smoke claim level until it pushes
rows through the relevant `MuZeroGameBuffer` and validates sampled reward,
value, policy, action, mask, observation, and `to_play` targets against
repo-owned expected rows.

The existing tiny two-seat native replay bridge is useful as a parity probe. It
does not consume the new source-state trainer replay chunks and should not be
cited as product multiplayer training integration by itself.

## Multiplayer Semantics

Policy rows are shared-policy seat perspectives:

1. The surface emits `observation[B,P,4,64,64]` and masks.
2. `live_mask[B,P]` selects present, alive, non-terminal seats.
3. Live seats flatten into variable `R_t` policy rows.
4. One shared policy/search produces one record per live policy row.
5. Actions are mapped back with `(policy_env_row, policy_player)` into
   `joint_action[B,P]`.
6. The target adapter maps the next step reward and terminal state back to the
   same `(env_row, player)`.

Rewards are per seat, not row-global:

- `reward[t,b,p]` is the seat-perspective reward for player `p`.
- Terminal row reward comes from `final_reward_map[t,b,p]` when
  `final_observation_row_mask[t,b]` is true.
- A winner can receive reward even when no policy rows remain after terminal
  completion. That terminal reward belongs to the previous live decision row.

Final observations are per row and per seat:

- Use `final_observation` for terminal target `next_observation`.
- Keep reset observations and final observations distinct.
- Do not reconstruct final observations from metadata-only env observations.

Absent/dead seats:

- Must not create policy rows while absent, dead, or row-terminal.
- May appear in full `[B,P]` arrays for audit and final reward maps.
- Need explicit target-row metadata if their terminal reward is credited to an
  earlier live decision.

## Project Feature Flags

No-death, `profile_no_death`, profile, optimizer, and training-helper modes are
valid project additions. They are not original CurvyTron behavior.

The adapter must preserve these modes as explicit metadata from source replay
chunk to target rows and sample batches. It must not silently drop or normalize
them away.

Required metadata when any such mode exists upstream:

- `death_mode`, including `profile_no_death` when enabled.
- `death_suppression_for_profile` or equivalent boolean.
- `profile_mode_enabled` and `profile_mode_claim`.
- `training_only_mode_enabled` and `training_only_mode_claim`.
- `opponent_death_mode` / opponent immortality flags when present.
- `source_fidelity_claim`, downgraded when project-only modes are enabled.
- `original_curvytron_behavior_claim=false` when no-death, profile-only, or
  training-only behavior changes source rules.

Valid claim language:

- normal source-rules rows may claim source-state trainer replay consumption.
- project-only rows may claim project training/profile support.
- project-only rows must not claim original CurvyTron source fidelity for the
  modified rule.
- mixed chunks must keep row-level or chunk-level metadata clear enough that
  source-rules rows and project-only rows cannot be confused during sampling,
  reporting, or later native-bridge validation.

Tests must include at least one `profile_no_death` or no-death/training-helper
fixture where the target rows preserve the mode fields and mark the
source-fidelity claim as restricted.

## `to_play`

Use `to_play=-1` for every target row in this first contract.

Native CurvyTron uses real-time player control state plus elapsed-ms server
frames, not discrete simultaneous trainer decisions. CurvyZero wrapper
decisions are simultaneous/player-major only at the trainer/replay boundary,
and the observation/player mapping is already seat-perspective. Public player
ids `0..P-1` belong in `player` metadata, not in LightZero-style `to_play`,
unless a future board-game-style contract is deliberately implemented and
tested for value sign, backup, search, replay sampling, and evaluation
semantics.

Tests must assert:

- target rows emit only `-1`;
- no adapter writes CurvyTron player ids into `to_play`;
- any future non-`-1` `to_play` path fails closed unless its schema id names the
  tested board-game/multi-agent contract.

## Non-Claims

This spec does not claim:

- ALE integration.
- Browser canvas pixel parity.
- Fixed-opponent self-play.
- True multiplayer self-play training quality.
- Real native LightZero buffer/training integration.
- Stock `train_muzero` integration.
- Durable artifact format.
- Correct learner updates.
- Evaluation quality or policy improvement.
- Broad P=3/P=4 lifecycle readiness.
- Original CurvyTron source fidelity when no-death, profile-only, optimizer, or
  training-only rule changes are enabled.

The first adapter can claim only:

- source-state trainer replay arrays were consumed;
- policy-row records were matched to live rows;
- transition target rows were built with tested action/reward/final-observation
  alignment;
- project-only feature flags were preserved when present;
- `to_play=-1` was enforced;
- native LightZero buffer/training integration was not claimed;
- fake/injected native `GameSegment` mapping was built without importing
  LightZero and kept native/LightZero/training/buffer/learner claims false.
- the opt-in real-LightZero helper was construction smoke only and did not
  claim `MuZeroGameBuffer` sampled-target parity.

## Focused Tests

Minimum tests before implementation is accepted:

1. Contract validation rejects missing arrays, wrong dtypes, shape drift, bad
   policy-row maps, illegal actions, and policy targets with illegal mass.
2. Reset-to-first-step alignment uses the reset observation as the decision
   state, ignores reset `joint_action=-1`, and reads action/reward from the next
   step record.
3. Terminal transition alignment uses `final_observation` and
   `final_reward_map` for a terminal row and does not emit a new terminal
   starting row without live policy rows.
4. P=2 shared-policy rows map actions and rewards back to the correct
   `(env_row, player)` pairs.
5. P=3 or P=4 live-mask filtering excludes dead/absent seats while preserving
   full-row audit metadata.
6. `to_play` is always `-1`, and public player ids remain only in `player`
   metadata.
7. Metadata says `native_game_segment_claim=false` and
   `lightzero_training_integration_claim=false`.
8. Output rows are copied, not aliases to mutable replay arrays or policy-row
   records.
9. No-death/profile/training-helper rows preserve explicit feature flags,
   including `profile_no_death` when present, and downgrade source-fidelity
   claims for modified source rules.
10. The deterministic sample-batch layer returns stable row ids and stable
   targets for a fixed seed.
11. Fake/injected native bridge tests construct injected `GameSegment`-like
    objects from these rows while keeping real LightZero and buffer claims
    false. Real `MuZeroGameBuffer` sampled-target parity remains a later test.

## Implementation Order

1. Done: define dataclasses/constants for policy-row records and target rows.
2. Done: add a validator that joins chunk policy rows to policy-row records.
3. Done: build one-step transition target rows with reset/step/final alignment.
4. Done: add focused P=2 and P=4 tests around mapping, terminal rows,
   no-death/death-immunity metadata, and `to_play`.
5. Done: add deterministic sample-batch construction from target rows.
6. Done: spike fake/injected native `GameSegment` mapping from
   `SourceStateMultiplayerTargetRowsV0`.
7. Done: add separate opt-in real-LightZero construction smoke.
8. Next: prove real LightZero `MuZeroGameBuffer` sampled-target parity.
