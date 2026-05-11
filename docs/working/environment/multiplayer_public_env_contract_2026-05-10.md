# Multiplayer Public Env And Replay Contract

Status: draft contract; metadata-only public env surface has landed
Date: 2026-05-10
Scope: public base multiplayer contract and current implementation boundary

There is one runtime under hardening: `VectorMultiplayerEnv`. The name
is historical public-env wording; the product direction is one fast
source-faithful runtime, not two implementations. The JavaScript oracle and
`CurvyTronSourceEnv` are proof/oracle tools. They should not become product
environments or the production trainer runtime.

Current proof boundary:

- Strict 1v1/no-bonus reset-to-terminal is proven for one long fixture.
  This is only a proof/profiling boundary, not full fidelity and not the
  multiplayer destination.
- Direct fast-runtime 3P/4P no-bonus normal-wall scoring canaries exist.
- Source fixtures prove focused 3P/4P lifecycle facts, including spawn order,
  warmup, survivor scoring, present/absent rows, next round, and selected match
  end cases.
- `VectorMultiplayerEnv` is the intended public 2P/3P/4P
  multiplayer runtime under hardening. It is still a narrow metadata/public-
  state surface, not a trainer-ready env, not a full bonus env, and not
  visual/pixel parity. It has focused seeded `BonusSelfSmall`/`BonusGameClear`/
  `BonusGameBorderless` support including public borderless expiry, plus
  partial public natural source-default type selection and same-frame natural
  bonus plus PrintManager random-order accounting.
- Source/runtime/public seeded `BonusGameBorderless` duration/expiry now pass.
- Public lifecycle fields are state arrays from reset, not temporary values
  stitched only into bridge rows: `round_done`, `warmdown_pending`,
  `match_done`, `round_winner`, and `match_winner`. The narrow proof is
  `test_public_lifecycle_metadata_arrays_exist_from_reset` plus the focused
  round/match warmdown tests. This is not a natural reset or full lifecycle
  parity claim.
- One focused source-tape public metadata proof now exposes survivor warmdown
  movement/death through explicit `advance_warmdown_frame(...)` in match mode:
  ordinary `step()` stays blocked while `warmdown_pending=true`, the explicit
  frame advances before `game:stop`, and the survivor death does not re-resolve
  the already-ended round score.
- A narrow 3P/4P no-bonus scalar learned-observation schema/projection exists:
  `curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0` maps
  `VectorMultiplayerEnv.state` to `float32[R,27]` rows for present+alive
  ego players.
- `build_multiplayer_scalar_observation_replay_artifact_v0(...)` can package
  those scalar rows, masks, row ids, ego ids, source shape, and nested public
  metadata records into a replay-shaped artifact. It is not trainer replay.
- `source_lifecycle_tie_at_max_score_4p.json` now has JS oracle, Python source
  runner, and focused public-env parity for tied 4P leaders continuing to the
  next round. This does not prove broader 4P unique-leader match end,
  multi-round match end, or present/absent match variants.
- Public 3P/4P natural reset, warmdown/replay integration, trainer-ready env
  observation, visual/pixel observation, and LightZero training are not yet
  proven.
- Active-round leave has one public metadata bridge:
  `VectorMultiplayerEnv.remove_player(...)` is guarded for narrow
  3P/4P continuation paths plus immediate round-end rows. The immediate path is
  source-proven for 2P leave and has a 4P source-rule canary for survivor
  scoring after already-dead players. There is no broad public leave, warmdown
  leave, replay, trainer, visual, or bonus support.

This contract says what the public multiplayer no-bonus runtime and replay
sidecar must expose before any training or replay claim is made. The current
metadata-only surface is a stepping/debug boundary, not a trainer-ready
interface.
For the prioritized gap queue, see
[multiplayer_env_gap_targets_2026-05-10.md](multiplayer_env_gap_targets_2026-05-10.md).

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Remaining Public Gaps

Keep this list short here; put case detail in the gap catalog.

- Broader lifecycle: natural reset, warmdown/next-round, match-end,
  present/absent, survivor scoring, autoreset, and final-row policy.
- Leave variants: active-round public leave is narrow; warmdown leave and leave
  interactions with replay/trainer/visual paths are still open. Immediate
  round-end leave is only 2P fixture-backed plus one 4P source-rule canary.
- Trainer adapters: 3P/4P trainer observation, reward/mask policy, replay
  writer/reader/shards, policy/search/value targets, and visual replay are not
  done.
- Bonus public env: focused seeded public slices and partial natural selection
  exist, but remaining runtime/public effects, full replay/final state, and
  broad natural bonus support do not. Runtime bonus support is being
  consolidated toward a table/spec, and natural source-default type selection
  must not imply unsupported runtime effects.
- Pixel parity: no source-pixel renderer or source/browser pixel comparison
  gate exists yet.

## Contract Ids

Proposed ids:

```text
public_env_contract_id: curvyzero_public_multiplayer_env/v0
ruleset_id: curvytron_no_bonus/v0
native_control_model_id: curvytron_realtime_controls_elapsed_frames/v0
trainer_control_wrapper_id: curvyzero_fixed_decision_wrapper/v0
action_space_id: curvyzero_turn3/v0
reward_schema_id: curvyzero_sparse_round_outcome/v0
final_observation_policy: terminal_public_observation_before_autoreset/v0
death_order_policy: curvytron_source_game_deaths_order/v0
multiplayer_scalar_observation_schema_id: curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0
multiplayer_scalar_replay_shape_id: curvyzero_multiplayer_scalar_observation_replay_shape/v0
lifecycle_policy_id: curvyzero_public_explicit_reset_warmdown_bridge/v0
reset_episode_id_policy: vector_reset_episode_id_increments_on_explicit_reset_only/v0
source_round_id_policy: one_based_source_round_increments_on_next_round_spawn/v0
```

The existing 1v1 trainer observation id `curvyzero_egocentric_rays/v0` remains
1v1 only. Do not use it for 3P/4P.

Semantic guardrail:

- Native CurvyTron holds real-time player control state and advances elapsed-ms
  server frames.
- `action[player_count]`, `joint_action`, fixed `decision_ms`, and `step`
  belong to the CurvyZero trainer wrapper and replay sidecars.
- A wrapper may choose a fixed decision cadence, but that cadence is not a
  source-native rule.
- Wrapper restrictions are temporary explicit profile/adapter configs. The
  reconstruction path remains source-default CurvyTron behavior in
  `VectorMultiplayerEnv`.

## Row Shape

The per-row contract below applies to one env row. Vector envs add a leading
`B` dimension to every row field.

Supported player counts for this contract are:

```text
player_count: 2 | 3 | 4
player_ids: 0..player_count-1
source_player_ids: 1..player_count
```

Wrapper player ids are zero-based. Source avatar ids are one-based. Replay must
carry both when the data is tied to source fixtures or source event logs.

## Inputs

Reset input:

```text
player_count
present[player_count]
max_score
reset_seed
reset_source
decision_ms
capacities
episode_end_mode
opponent_policy_config optional
source_identity optional
```

Rules:

- `present` is the selected player set for the round.
- Absent players have `present=false`, `alive=false`, and no required action.
- `episode_end_mode` is `round` or `match`.
- First public 3P/4P implementation should use `round` until warmdown,
  next-round, and match-end metadata are proven end to end.
- `capacities` must include enough room for bodies, timers, event rows, death
  rows, and replay sidecar rows. Capacity truncation must set
  `truncated=true` and a concrete `truncation_reason`.

Step input:

```text
action[player_count]
```

Action ids:

```text
0 left
1 straight
2 right
```

Rules:

- Live present players may provide one action each.
- Dead or absent players do not need live actions. Their action slot may be
  ignored, filled with a sentinel, or copied from the previous control state,
  but the chosen policy must be recorded in the action sidecar.
- Source-native control values are adapter internals. They must not replace the
  public `0/1/2` action ids.

## Outputs

Reset returns:

```text
observation
action_mask[player_count, 3]
info
```

Step returns:

```text
observation
reward[player_count]
terminated
truncated
done
info
```

`done = terminated or truncated`.

Observation rules:

- A public observation schema id is always required.
- The public env still defaults to explicit debug or metadata-only observation
  ids. Do not relabel debug state as trainer observation.
- The 3P/4P scalar projection exists beside the public env. It is a pure
  projection over `VectorMultiplayerEnv.state`, not a second env and not
  a replay writer, visual/pixel, source-fidelity completion, or LightZero
  training claim.
- Terminal step output must include the final observation before any reset.
- Reset observation for the next episode must never replace the terminal
  observation.

Reward rules:

- Public env reward is a per-player vector.
- For `curvyzero_sparse_round_outcome/v0`, nonterminal rewards are all `0.0`.
- Survivor round terminal: survivor gets `+1.0`; losers get `-1.0`.
- All-dead round draw: all players get `0.0`.
- Pure truncation without a game outcome: all players get `0.0`.
- If a row is both terminated and truncated, the source game outcome chooses
  reward and `truncated=true` remains in metadata.

## Reset Metadata

Reset `info` must include:

```text
public_env_contract_id
env_impl_id
ruleset_id
rules_hash
native_control_model_id
trainer_control_wrapper_id
decision_ms
episode_id
round_id
lifecycle_policy_id
reset_episode_id
reset_episode_id_policy
source_round_id
source_round_id_policy
episode_end_mode
player_count
player_ids
source_player_ids
present[player_count]
alive[player_count]
score[player_count]
round_score[player_count]
round_done
warmdown_pending
match_done
round_winner
match_winner
death_player[player_count]
death_count
death_order_policy
reset_seed
reset_source
seed_sequence_id optional
random_tape_cursor optional
random_tape_draw_count optional
random_tape_source optional
random_tape_length optional
rng_impl_id optional
random_tape_history_ref optional
source_identity optional
observation_schema_id
observation_schema_hash
action_space_id
action_space_hash
reward_schema_id
reward_schema_hash
final_observation_policy
```

Reset values:

- `alive[p]` is true only for present spawned players after reset/warmup setup.
- `death_player` is padded with `-1`.
- `death_count` is the number of valid entries in `death_player`.
- If the source lifecycle places absent players in the death list at
  `round:new`, those absent players must appear in `death_player` in the same
  source order and `death_count` must include them.
- `score` is cumulative match score.
- `round_score` is the current round-local source score contribution. It is all
  zero at ordinary reset.
- `round_done`, `warmdown_pending`, `match_done`, `round_winner`, and
  `match_winner` are real public env state arrays from reset. Reset values are
  false for booleans and `-1` for winner arrays until a source-backed terminal
  fact sets them.
- `episode_id` is the reset episode for that vector row.
- `round_id` is the round inside that episode: public reset starts selected
  rows at `1`, warmdown increments it only when a next round is spawned, and
  match-end rows keep the final round id.
- `reset_episode_id` is the explicit reset episode id. It does not increment
  for a next round inside one match.
- `source_round_id` follows the same one-based source-round rule as `round_id`.
- `lifecycle_policy_id`, `reset_episode_id_policy`, and
  `source_round_id_policy` must travel with replay so callers do not mistake a
  multi-round match for multiple reset episodes.

## Step Metadata

Step `info` must include all reset metadata that can affect replay, plus:

```text
step_index
tick_index
elapsed_ms
action_sidecar
opponent_policy_sidecar optional
present[player_count]
alive[player_count]
death_player[player_count]
death_count
death_events optional
score[player_count]
round_score[player_count]
round_done
warmdown_pending
match_done
round_winner
match_winner
terminated
truncated
done
needs_reset
terminal_reason
truncation_reason
timeout
draw
winner_ids
round_winner_ids
match_winner_ids
loser_ids
final_observation optional
final_reward_map optional
source_event_ref optional
source_event_range optional
state_ref optional
trace_ref optional
trace_hash optional
eval_episode_return optional
```

Terminal reason enum:

```text
none
round_survivor_win
round_all_dead_draw
match_winner
match_draw
timeout
horizon_truncated
event_overflow_truncated
capacity_truncated
infra_truncated
```

Rules:

- `round_done=true` means source rules ended the current round.
- `warmdown_pending=true` means the source round has ended, the public match
  episode has not ended, and ordinary public `step()` is blocked until an
  explicit warmdown API advances the pending timer/frame.
- `match_done=true` means source rules ended the match.
- `terminated=true` means the configured public episode ended because of
  source game rules.
- In `episode_end_mode=round`, a round end is terminal.
- In `episode_end_mode=match`, only match end is terminal; round end is
  metadata unless the row also truncates.
- `draw=true` only means a source game outcome draw. Pure truncation is not a
  draw.
- `winner_ids` is empty for draw and nonterminal steps.
- `round_winner_ids` is empty for all-dead round draw.
- `match_winner_ids` is empty until match end. Tied leaders at max score that
  continue to the next round are not match winners.
- `needs_reset=true` only after a terminal public episode transition has been
  returned.

Warmdown frame bridge:

- In `episode_end_mode=match`, a source round end sets
  `warmdown_pending=true` and ordinary public `step()` remains blocked until the
  caller advances warmdown.
- `advance_warmdown_frame(actions, elapsed_ms=...)` is the only current public
  metadata bridge for source survivor movement during that blocked interval.
  It advances rows that are already waiting in warmdown, decrements the
  scheduled `game:stop` timer, and refuses elapsed values that would cross the
  `game:stop` boundary.
- The landed fixture-backed proof is only the focused 3P
  `source_lifecycle_survivor_score_3p_next_round` case: after `round:end`, the
  survivor moves for 1150 ms, dies at the source-pinned wall position, appends
  to the death list without terminal re-scoring, then `advance_warmdown(3850.0)`
  reaches `game:stop` and next `round:new`.
- This is metadata-only. It is not hidden autoreset, replay support, visual
  observation, trainer-ready match-mode stepping, or broad natural frame-loop
  lifecycle support.

Active-round leave bridge:

- `VectorMultiplayerEnv.remove_player(row, player_id)` is the only
  current public leave/disconnect bridge.
- It is metadata-only and scoped to active-round continuation rows guarded by
  the 3P and 4P source fixture paths, plus immediate round-end rows guarded by
  the 2P source fixture and one 4P source-rule canary.
- Public `player_id` values are zero-based. Source ids are public id plus one.
- A successful leave sets the leaver to `present=false` and `alive=false`,
  emits leave metadata, and does not append the leaver to `death_player`.
- If the leave ends the round, the public env scores the remaining survivor
  from source avatar-collection size, sets round terminal metadata, and exposes
  final debug metadata in `episode_end_mode=round`.
- The bridge rejects warmdown, terminal, absent-player, dead-player, and
  bad-shape calls.
- This does not prove broad public leave, warmdown leave, replay, trainer-ready
  env behavior, visual observation, or bonus support.

## Player Presence, Alive, And Death Order

Required fields:

```text
player_count: int
present[player_count]: bool
alive[player_count]: bool
death_player[player_count]: int, padded with -1
death_count: int
```

Rules:

- `present` says whether a player is in the round.
- `alive` says whether a present player is currently alive.
- Absent players are always `alive=false`.
- Death order follows the source game death list, not sorted player id order.
- Same-frame deaths use source processing order.
- Absent-at-round-new entries must be distinguishable from live deaths, either
  by `death_events` or by a source event ref.
- Scoring must follow source scoring order. In particular, do not recompute
  3P/4P scores from a sorted final alive mask.

Optional `death_events` shape:

```text
[
  {
    "player_id": int,
    "source_player_id": int,
    "death_index": int,
    "tick_index": int,
    "elapsed_ms": int,
    "reason": "absent_at_round_new" | "wall" | "body" | "leave" | "unknown"
  }
]
```

## Scores And Winners

Required fields:

```text
score[player_count]
round_score[player_count]
reward[player_count]
winner_ids
round_winner_ids
match_winner_ids
loser_ids
draw
```

Rules:

- `score` is cumulative source match score after the current step.
- `round_score` is the source round contribution after the current step.
- `reward` is the trainer/public reward schema, not necessarily the same as
  source score.
- `loser_ids` is the set of present players that did not win the terminal
  public outcome. It is empty on nonterminal steps.
- For all-dead draw, `winner_ids=[]`, `draw=true`, and `loser_ids=[]` unless a
  later trainer wrapper deliberately wants loser bookkeeping in sidecar only.

## Action Sidecar

Every step must record how the public action became native control state:

```text
action_sidecar: {
  "action_space_id": "curvyzero_turn3/v0",
  "decision_ms": int,
  "player_action[player_count]": int,
  "player_action_mask[player_count,3]": bool,
  "action_required[player_count]": bool,
  "action_source[player_count]": string,
  "native_control_value[player_count]": int,
  "ignored_action_policy": string,
  "joint_action_schema_id": string
}
```

`action_source` examples:

```text
external_joint_action
ego_policy
opponent_policy
dead_noop
absent_noop
terminal_padding
```

This sidecar is replay data. It is not native CurvyTron source state.

## Opponent Policy Sidecar

LightZero final shape is likely ego-player rows with opponent policies:

- The trainer chooses one ego action.
- The wrapper supplies actions for the other live present players.
- The wrapper writes the full wrapper action map into the `joint_action` replay
  sidecar.

Full wrapper joint-action MCTS is not the first target.

Required sidecar for ego rows:

```text
opponent_policy_sidecar: {
  "ego_player_id": int,
  "opponent_player_ids": int[],
  "opponent_policy_ids[player_count]": string,
  "opponent_policy_versions[player_count]": string,
  "opponent_policy_snapshot_refs[player_count]": string optional,
  "opponent_actions[player_count]": int,
  "opponent_action_logp[player_count]": float optional,
  "opponent_action_seed_refs[player_count]": string optional
}
```

Rules:

- Ego player id must be sidecar metadata, not a hidden learned-observation leak.
- Opponent policy ids and versions must be stable enough to reproduce or audit
  the transition.
- In external joint-action mode, use a policy id such as
  `external_joint_action/v0` and still record the actions.

## Final Observation Policy

Policy id:

```text
terminal_public_observation_before_autoreset/v0
```

Rules:

- The final observation is built after the source update/timer processing that
  produced the terminal public transition.
- The final observation is captured before any autoreset or next reset copy.
- If autoreset is later enabled, replay must store both:
  - `final_observation` for the terminal transition;
  - `next_reset_observation` or reset metadata for the next episode.
- A terminal replay row without final observation must explicitly say
  `final_observation_policy=absent_unproven_multiplayer_obs/v0` and must not
  be accepted as production trainer replay.

## Seed And Source Identity

Replay and public metadata must identify where the row came from:

```text
episode_id
round_id
reset_seed
reset_source
seed_sequence_id optional
random_tape_cursor optional
random_tape_draw_count optional
random_tape_source optional
random_tape_length optional
rng_impl_id optional
random_tape_history_ref optional
source_identity optional
```

`source_identity` may include:

```text
source_fixture_id
source_fixture_path
source_fixture_hash
source_oracle_id
source_env_id
comparator_id
comparator_hash
warmup_contract_id
evidence_ref
```

Rules:

- Reset seed alone is not enough for production replay.
- Row-local RNG cursor/draw count or a history ref must travel with replay
  before claiming reproducible 3P/4P reset/warmup behavior.
- Generated public reset uses `seed_generated_source_random_history` with
  `curvyzero_seeded_source_math_random_history/v0`.
- The only natural reset claim now allowed is
  `seeded_source_history_reset_spawn_warmup_call_order/v0`: tested 2P/3P/4P
  reset spawn plus warmup random-call order when the generated row history is
  fed to both public reset and `CurvyTronSourceEnv`.
- This is not V8 `Math.random` bit parity and not broad generated reset,
  lifecycle, replay, trainer, visual, or bonus parity.
- Source fixture ids are evidence refs, not a replacement for public env
  metadata.

## Replay Requirements

A replay row must be able to explain the transition without rerunning source
tools.

Episode-level replay metadata:

```text
public_env_contract_id
ruleset_id
rules_hash
env_impl_id
player_count
player_ids
source_player_ids
present
episode_end_mode
reset_seed
reset_source
lifecycle_policy_id
reset_episode_id
reset_episode_id_policy
source_round_id
source_round_id_policy
seed/source identity
observation/action/reward schema ids and hashes
native_control_model_id
trainer_control_wrapper_id
decision_ms
opponent policy ids and versions, if any
```

Step-level replay metadata:

```text
action_sidecar
opponent_policy_sidecar optional
present
alive
death_player
death_count
score
round_score
round_done
warmdown_pending
match_done
round_winner
match_winner
terminated
truncated
done
terminal_reason
truncation_reason
winner/draw fields
final_observation or explicit absent policy
final_reward_map on terminal rows
```

Reader policy:

- Reject 3P/4P trainer replay that lacks player count, present mask, death
  order, score vector, terminal flags, reset seed/source, and action sidecar.
- Reject terminal rows that silently replace final observation with reset
  observation.
- Reject rows that claim `curvyzero_egocentric_rays/v0` for 3P/4P.

## Explicitly Out Of Scope Now

- Bonuses.
- Visual observations or browser pixel fidelity.
- Full wrapper joint-action MCTS.
- A production LightZero adapter.
- Trainer-ready 3P/4P env observation support. The narrow scalar projection
  exists, but it is not enough by itself.
- A production replay shard/manifest format beyond the required sidecar fields.
  Current metadata replay packaging, metadata recorder, and scalar
  replay-shaped artifact are useful checks, but they are not trainer replay,
  visual replay, or policy/search/value targets.
- Hidden autoreset.
- Broad match lifecycle claims not pinned by source fixtures and public env
  tests.
- Broad public leave/disconnect support. Only the narrow 3P/4P metadata
  continuation bridge exists.
- Rewriting or widening `VectorTrainerEnv1v1NoBonus` in place.
- Treating JS or `CurvyTronSourceEnv` as the target fast runtime.

## Exact Next Implementation Chunks

Assuming comparator canaries and N-player warmup are available, land the next
work in this order.

1. Warmdown, next-round, and match terminal split.
   - Touch `src/curvyzero/env/vector_runtime.py` and
     `src/curvyzero/env/vector_lifecycle.py`.
   - Preserve the existing reset-owned public lifecycle arrays:
     `round_done`, `warmdown_pending`, `match_done`, `round_winner`, and
     `match_winner`.
   - Broaden those fields beyond the current narrow warmdown bridges only from
     named source fixtures. Do not reintroduce stitched metadata-only lifecycle
     fields.
   - Preserve source behavior for tied leaders at max score continuing to the
     next round.
   - Tests: 3P survivor next-round, 4P survivor next-round, 3P match-end,
     3P tie-at-max continuation, and
     `test_public_lifecycle_metadata_arrays_exist_from_reset`.

2. Public multiplayer env metadata hardening.
   - Continue `VectorMultiplayerEnv` in
     `src/curvyzero/env/vector_multiplayer_env.py`; add a trainer wrapper above
     it later only after metadata parity is stable.
   - Keep it separate from `VectorTrainerEnv1v1NoBonus`. Do not widen the
     strict 1v1 trainer env in place.
   - Inputs: `player_count`, `present`, `max_score`, `reset_seed`,
     `reset_source`, `decision_ms`, capacities, and `episode_end_mode`.
   - Tests: 3P and 4P reset metadata, warmup metadata, action shape `[B,P]`,
     absent/dead action handling, and no hidden trainer observation claim.

3. Public step metadata and final transition capture.
   - Touch `src/curvyzero/env/vector_runtime.py` and
     `src/curvyzero/env/vector_multiplayer_env.py`.
   - Populate `present`, `alive`, death order, score vectors, round/match
     terminal fields, winner/draw fields, terminal reason, final reward map,
     and final observation policy on every terminal step.
   - Public 4P canaries are being added now for the seeded ordered-deaths
     survivor and terminal-draw wall fixtures. They should prove metadata-only
     public stepping, not natural reset or trainer observations.
   - Tests: 3P two-dead survivor, 4P ordered deaths survivor, and 4P terminal
     draw canaries through the public env, not direct runtime seeding only.
   - Still missing after these canaries: natural public reset, replay
     integration, and trainer-ready 3P/4P observation/env support.

4. Replay sidecar integration.
   - Touch `src/curvyzero/training/vector_env_replay_recorder.py` and replay
     reader/writer compatibility checks.
   - Record player count, present mask, alive mask, death order, scores,
     lifecycle arrays, reset seed/source, RNG cursor/history ref when
     available, action sidecar, opponent policy sidecar, and final observation
     policy.
   - Tests: reject missing multiplayer fields, reject 3P/4P
     `curvyzero_egocentric_rays/v0` claims, reject terminal reset-observation
     substitution.

5. Ego-row wrapper with opponent policies.
   - Add a wrapper layer on top of the public env, not inside source rules.
   - One row is one ego player; opponents are supplied by named policies.
   - Record opponent policy ids, versions, snapshot refs, and actions in the
     sidecar.
   - Keep full wrapper joint-action MCTS out of this chunk.

6. 3P/4P observation decision.
   - Keep `src/curvyzero/env/vector_multiplayer_observation.py` as the narrow
     scalar projection unless a trainer wrapper proves it can own the next
     step.
   - Touch `src/curvyzero/env/vector_trainer_observation.py` only after public
     metadata and replay are stable.
   - Either wire the scalar projection into a trainer-ready wrapper with tests,
     keep public rows debug/metadata-only, or define a later visual schema.
   - Do not reuse the 1v1 ray/scalar schema.
