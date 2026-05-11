# Multiplayer Training Interface - 2026-05-10

Status: narrow training-facing shape note
Scope: CurvyTron/MuZero/LightZero multiplayer self-play interface only

This note answers the immediate 2P/3P/4P self-play questions without moving
the implementation boundary. It is about the wrapper shape a trainer sees, not
native CurvyTron source semantics.

There is one runtime under hardening: `VectorMultiplayerEnv`.
`CurvyTronSourceEnv` and the source JS oracle are proof tools. Strict
`VectorTrainerEnv1v1NoBonus` is a narrow 1v1 proof/profiling boundary, not the
destination.

## Short Answers

For 2P/3P/4P self-play, one player should observe an ego-perspective learned
observation for exactly one controlled seat. The strict 1v1 scalar/ray
proof/profiling schema is:

```text
curvyzero_egocentric_rays/v0
LightZero payload: float32[106] + int8[3] action_mask + to_play=-1
```

There is now also a narrow 3P/4P no-bonus learned-observation
schema/projection:

```text
curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0
projection: VectorMultiplayerEnv.state -> float32[R,27]
rows: present+alive ego players only
claim: scalar state projection only
```

For public 2P/3P/4P today, the implemented surface is metadata-only:

```text
curvyzero_debug_metadata_only/v0
shape: float32[B, P, 6]
claim: debug/public metadata only, not trainer observation
```

Do not reuse the 1v1 ray schema as a 3P/4P trainer observation. The next real
trainer-ready multiplayer interface may use the scalar projection, but it still
needs wrapper, reset/lifecycle, replay, reward, and policy proof. The scalar
projection is not a second env, not visual/pixel support, not source-fidelity
completion, and not LightZero training. A replay-shaped scalar artifact exists
for rows plus public metadata, but that is not trainer replay.

Opponent policies fit as wrapper-owned action suppliers. LightZero/MuZero v0
should choose one ego action. The wrapper fills the other live present players
from named/versioned opponent policies, then logs the full wrapper action map
as the `joint_action` sidecar and logs the opponent-policy sidecar. This is
learner-versus-policy-controlled opponents, not full simultaneous wrapper
joint-action MCTS.

The wrapper abstraction around source control is:

```text
source-native: held real-time control state + elapsed-ms server frames
trainer wrapper: fixed decision cadence + action ids 0/1/2 + joint_action map
```

`step(...)`, `joint_action`, `decision_ms`, action ids, and action sidecars are
CurvyZero trainer/replay abstractions. They are not native CurvyTron source API
facts. The wrapper maps `0 left`, `1 straight`, `2 right` to source control
values and holds those controls through the elapsed-ms frame window.
Wrapper restrictions are temporary explicit profile/adapter configs, not a
reason to avoid source-default CurvyTron reconstruction.

Replay metadata must be sufficient to explain a transition without rerunning
source tools. Minimum multiplayer metadata is: player ids, source player ids,
present/alive masks, action mask, full wrapper action map / `joint_action`,
reward vector,
done/terminated/truncated, round/match terminal facts, winner/draw, score and
round-score vectors, death order, reset seed/source, random tape cursor/draw
count or RNG ref, action sidecar, observation schema id, final-observation
policy, and opponent policy sidecar when opponents are wrapper-filled.

Still missing before trainer-ready multiplayer: a trainer-ready env/wrapper
around the 3P/4P scalar projection or a later visual schema, natural
reset/warmup and broader warmdown/match-mode policy, production replay
writer/manifest, opponent-policy sampling/checkpoint pool integration, ego-row
rotation, LightZero/MuZero trainer bridge for this multiplayer wrapper, and
later source-faithful visual stacked frames.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Current Proven Surfaces

Source CurvyTron facts already promoted:

- Runtime holds per-avatar control state and advances server frames by elapsed
  milliseconds.
- Movement uses elapsed milliseconds; the server target is 60 Hz but physics
  integrates elapsed wall-clock time.
- Multiplayer scoring/death-order, same-frame deaths, present/absent behavior,
  warmup/warmdown, and match/tie slices are proven only for named fixtures.
- `source_lifecycle_tie_at_max_score_4p.json` now has JS oracle, Python source
  runner, and focused public metadata parity for tied 4P leaders continuing to
  the next round. It is still not a trainer-ready observation/replay claim.

Strict trainer-facing 1v1 facts:

- `VectorTrainerEnv1v1NoBonus` and scalar/ray helpers expose the strict
  `curvyzero_egocentric_rays/v0` learned-observation boundary.
- The LightZero scalar wrappers are single-ego. LightZero controls one ego
  player; the wrapper fills the opponent action from a fixed named policy.
- The survival-time LightZero wrapper changes reward only; it is still 1v1,
  single-ego, fixed-opponent, no-train plumbing.

Public multiplayer facts:

- `VectorMultiplayerEnv` supports public 2P/3P/4P rows with
  action shape `[B,P]`.
- Its observation schema is `curvyzero_debug_metadata_only/v0`, not a learned
  trainer observation.
- `curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0` exists beside
  it as a pure state projection in
  `src/curvyzero/env/vector_multiplayer_observation.py`. It emits
  `float32[R,27]` present+alive ego rows and has explicit non-claims for
  trainer-ready env, visual/pixel, and source-fidelity completion.
- It emits per-player reward vectors, action masks, final-row metadata, explicit
  `autoreset_done_rows()`, reset/source/cursor fields, death order, score
  vectors, round/match facts, and full action sidecars.
- Public lifecycle facts now live in env state arrays from reset:
  `round_done`, `warmdown_pending`, `match_done`, `round_winner`, and
  `match_winner`. This is useful trainer-facing metadata, but still not
  trainer-ready lifecycle or natural reset parity.
- `MultiplayerMetadataReplayRecorder` can package those public rows into
  metadata-only replay chunks and rejects trainer-observation claims.
- `build_multiplayer_scalar_observation_replay_artifact_v0(...)` can package
  the 3P/4P scalar projection rows into a replay-shaped artifact with
  `observation`, scalar action masks, LightZero int masks, env row ids, ego
  player ids, row mask, source shape, and active-row public metadata records.
  It preserves public `round_id` meaning through the nested public record:
  reset starts at `1`, next-round warmdown increments it, and match-end rows
  keep the final round id. It is not full trainer replay, not visual replay,
  not source-fidelity completion, and not policy/search/value targets.

ALE is only relevant to official Atari Pong control runs. CurvyTron should use
LightZero env shapes directly and must not be described as an ALE environment.

## One Player's Observation

The intended multiplayer trainer row is an ego row:

```text
row identity: ego_player_id
policy input: observe(ego_player_id)
legal moves: action_mask[3]
policy output: one action id in {0,1,2}
value target: scalar ego return
hidden metadata: player ids, source ids, opponent policies, replay refs
```

The learned observation must not smuggle stable seat/color identity unless a
schema explicitly chooses that. The player id is replay/eval metadata, not a
free learned-observation leak.

For 2P, the current scalar/ray v0 schema can represent ego plus one opponent.
For 3P/4P, the current scalar schema is the narrow non-visual bridge. Broader
trainer work still has to decide whether to keep it, widen it, or replace it:

- aggregate nearest opponents/trails into fixed channels;
- allocate fixed slots with identity randomization/canonicalization;
- use a visual frame stack from source-faithful rendering;
- or keep the public env metadata-only until the visual path is ready.

Final target should include visual stacked frames later. The current visual
surface is debug occupancy only, and the current proven non-visual trainer
surface is scalar/ray for 1v1.

## Opponent Policies

The v0 multiplayer self-play shape should be searched/controlled ego only:

```text
LightZero/MuZero chooses action for ego_player_id.
Wrapper samples/fills non-ego actions from named policies.
Wrapper advances the elapsed-ms source-control window.
Replay stores the full wrapper action map / `joint_action` and policy sidecars.
```

Opponent policies should be explicit and versioned:

```text
policy_id
policy_version
policy_snapshot_ref optional
seed or seed_ref for stochastic policy choices
opponent_actions[player_count]
optional action logp/prob telemetry
```

Useful first opponents are fixed, random, heuristic, same-checkpoint
policy-only, and frozen checkpoints from a pool. Full joint-action search is
deferred because the action branching is `3^P` before horizon length or model
uncertainty.

## Wrapper Step Semantics

Public trainer action ids stay:

```text
0 left
1 straight
2 right
```

For each wrapper decision:

1. Build a full `[P]` action map for live present players.
2. Validate action masks for live players.
3. Mark dead, absent, or terminal slots as ignored/noop/padding in the sidecar.
4. Map trainer ids to source control values.
5. Advance the source-shaped runtime by `decision_ms` elapsed milliseconds.
6. Return observation, per-player reward vector or scalar ego reward, done
   flags, final observation metadata, and replay sidecars.

The source-native model remains held controls over elapsed-ms server frames.
The wrapper may choose a fixed decision cadence, but that cadence is a training
choice, not a source rule.

## Replay Metadata Required

A multiplayer replay row needs these fields before any training claim:

```text
public_env_contract_id
ruleset_id, rules_hash, env_impl_id
native_control_model_id
trainer_control_wrapper_id
decision_ms
player_count
player_ids
source_player_ids
present[player_count]
alive[player_count]
action_mask[player_count,3]
joint_action[player_count]
action_sidecar
opponent_policy_sidecar optional
reward[player_count] or ego reward plus reward_owner
done, terminated, truncated, needs_reset
round_done, warmdown_pending, match_done, terminal_reason, truncation_reason
winner, round_winner, match_winner, winner_ids, round_winner_ids, match_winner_ids, draw
score[player_count], round_score[player_count]
death_player[player_count], death_count, death_order_policy
reset_seed, reset_source
random_tape_cursor, random_tape_draw_count, rng_history_ref optional
observation_schema_id, observation_schema_hash
action_space_id, action_space_hash
reward_schema_id, reward_schema_hash
final_observation_policy
final_observation or explicit absence policy
episode_id, round_id, step_index, tick_index, elapsed_ms
lifecycle_policy_id, reset_episode_id_policy, source_round_id_policy
```

`episode_id` is the reset episode for that vector row. `round_id` is the round
inside the episode: reset starts at `1`, next-round warmdown increments it, and
match-end rows keep the final round id.

For metadata-only public multiplayer rows, replay must continue to say:

```text
metadata_only=true
trainer_observation_claim=false
trainer_replay_claim=false
learned_observation_claim=false
```

## Missing Before Trainer-Ready Multiplayer

Trainer-ready 2P/3P/4P multiplayer still needs:

- A trainer-ready decision around the landed 3P/4P scalar projection:
  keep `curvyzero_multiplayer_egocentric_scalars_3p4p_no_bonus/v0`, widen it,
  or replace it. The existing 1v1 ray schema is not enough.
- A public ego-row wrapper over `VectorMultiplayerEnv`:
  one ego action in, opponent policies filled, full action sidecar out.
- Opponent policy registry/sampler with stable ids, versions, snapshot refs,
  and stochastic seeds.
- Ego rotation across seats and replay ownership rules for all live players.
- Match-mode episode semantics, or an explicit round-mode-only training
  decision, with final observation before any reset.
- Broader natural reset/warmup/warmdown coverage beyond fixture-tape and
  metadata-only proofs.
- Production replay writer/manifest for multiplayer learned observations,
  search targets, policy targets, values, returns, and compatibility hashes.
  The current scalar row artifact only packages observations/masks plus public
  metadata trace; it is not a trainer replay shard.
- LightZero/MuZero custom-env collector/evaluator config for the multiplayer
  wrapper. Current installed LightZero smokes are scalar/debug/survival
  no-train or Atari Pong control evidence.
- Evaluation scorecards against fixed/random/heuristic/frozen checkpoint pools.
- Later visual frame source: source-faithful CurvyTron renderer plus
  LightZero-facing frame stack, not ALE.

## Narrow Code Hooks Landed

Keep these hooks outside `src/curvyzero/env/vector_runtime.py`:

1. `curvyzero.env.multiplayer_ego_wrapper`: narrow metadata-only ego wrapper
   landed. It chooses configured ego-player rows, accepts one action per live
   ego row, fills non-ego live slots through an opponent policy, feeds full
   `[B,P]` actions to `VectorMultiplayerEnv`, and emits wrapper/opponent
   sidecars without claiming learned observations or wrapper joint-action MCTS.
2. `curvyzero.training.multiplayer_opponent_policy`: fixed-action and seeded
   random legal opponent policies landed with ids, versions, seeds, actions,
   and deterministic slot seeds. Heuristic and frozen-checkpoint policies are
   still future hooks.
3. `curvyzero.training.multiplayer_ego_lightzero_coach_smoke`: local no-train
   Coach/LightZero-facing smoke landed around the ego wrapper path only. It
   instantiates `MetadataOnlyMultiplayerEgoWrapper` plus
   `FixedActionOpponentPolicy`, proves reset/action-map/sidecar shape, and
   reports optimizer/coach alignment as metadata-only: no learned observation,
   no trainer replay claim, no ALE, no `train_muzero`, and no duplicate
   CurvyTron trainer env.
4. `curvyzero.env.vector_multiplayer_observation`: narrow 3P/4P scalar
   projection landed. It is projection-only and has explicit non-claims.
   Trainer wrapper, replay writer, visual frames, and LightZero training remain
   separate.
5. `curvyzero.training.multiplayer_replay_writer`: promote the current
   metadata-only recorder into learned-observation replay only after the schema
   and lifecycle policy are stable.
6. Installed LightZero config/collector smoke remains deferred; the current
   multiplayer Coach smoke is pure local wrapper metadata and still does not
   call `train_muzero`.

## Non-Claims

- Not ALE CurvyTron.
- Not full source-fidelity CurvyTron.
- Not trainer-ready 3P/4P learned-observation env support.
- Not production multiplayer replay.
- Not full simultaneous self-play or wrapper joint-action MCTS.
- Not proof that current LightZero CurvyTron wrappers train.
