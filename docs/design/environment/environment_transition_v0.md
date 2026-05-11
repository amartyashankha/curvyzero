# EnvironmentTransitionV0

Status: draft contract
Date: 2026-05-09

## Decision

Production target: one fast, source-faithful environment.

`CurvyTronSourceEnv` and the JS oracle are temporary truth/proof tools. They are
where source rules become runnable, inspectable, and comparable before moving
into the production runtime.

The fast implementation is the production direction. It must migrate source
semantics from the executable spec and prove them with the shared transition
trace.

So the answer is: unify the public transition contract and trace format, then
use them to verify the fast implementation. Do not optimize an unverified fast
path or preserve the proof tools as a second production runtime.

## Transition Shape

`EnvironmentTransitionV0` is the comparison boundary for one environment row.
It is not the replay shard format and not a policy observation.
It is also not the native CurvyTron source API: source behavior is current
player control state advanced through elapsed-millisecond server frames, while
`joint_action`, action ids, and fixed decision cadence are CurvyZero
wrapper/schema/replay terms.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `schema` | `curvyzero_environment_transition/v0` |
| `ruleset_id` | Source target, initially `curvytron-v1-reference` |
| `transition_id` | Monotonic row-local transition id |
| `episode_id` | Row-local episode id |
| `reset_episode_id` | Alias for the row-local reset episode id; separated from source round identity |
| `round_id` | One-based source round id inside the current reset episode |
| `source_round_id` | Alias for the source round id; increments only when warmdown spawns a next round |
| `lifecycle_policy_id` | Versioned label for explicit reset plus narrow warmdown-bridge lifecycle metadata |
| `reset_seed` | Seed that created the current episode |
| `reset_source` | `manual`, `autoreset`, `fixture`, or `replay` |
| `player_count` | Real source player count for the row |
| `map_size` | Source map size for that player count |
| `at_ms_before`, `at_ms_after` | Source clock before and after the transition |
| `reset` | Present only when the transition includes reset work |
| `timer_advance_ms` | Elapsed timer advance before movement |
| `native_control_model_id` | Native control model label, currently `curvytron_realtime_controls_elapsed_frames/v0` |
| `trainer_control_wrapper_id` | Trainer wrapper label, currently `curvyzero_fixed_decision_wrapper/v0` |
| `decision_ms` | Elapsed milliseconds for one fixed wrapper decision |
| `joint_action` | CurvyZero wrapper/replay field: public action ids for live players, converted to source control state for the elapsed-ms transition window; not a native source field |
| `events` | Ordered normalized source events |
| `state_before`, `state_after` | Comparable source state snapshots |
| `terminal` | Done/terminated/truncated and source terminal metadata |
| `observation_refs` | Observation schema ids/hashes or refs, not raw policy inputs |
| `reward_refs` | Reward schema ids/hashes or refs, not hidden reward logic |
| `replay_refs` | Optional replay/debug refs for reproducing the transition |

## Reset Input

Reset is explicit. It is never hidden inside `step`.

Required reset input:

- `player_count`: `1`, `2`, `3`, or `4`.
- `present`: bool per player. Present and alive are separate.
- `reset_seed`: row-local seed.
- `reset_source`: one known reset-source code.
- `ruleset_id` and behavior-affecting rules hash.
- Optional source overrides: `players`, `max_score`, `borderless`,
  `warmup_ms`, and fixture/replay state.

Reset output must include:

- `round:new` and natural spawn event trace.
- Row-local RNG calls with call index, site, player, value, and `at_ms`.
- Spawn position and angle events in source order.
- Initial state fields after reset but before the first live movement tick.
- Timer schedule created by reset, especially the first `game:start` warmup
  timer.

Autoreset is a separate handoff after the terminal transition has been returned
and staged for replay. A row reset must snapshot final transition data before
overwriting the row.

## Timer Handling

Each transition has a timer phase before movement:

```text
reset/autoreset, if requested
  -> advance due timers in row-local source order
  -> emit timer-created events
  -> apply wrapper-provided control changes
  -> run movement/collision/PrintManager update
  -> build terminal, reward, observation, and replay refs
```

Timer state is part of `state_before` and `state_after`.

Minimum timer fields:

- `timer_active`
- `timer_remaining_ms`
- `timer_kind`
- `timer_player`
- `timer_seq`
- `timer_overflow`

Required timer kinds for 2P no-bonus:

- `game:start`: fires after warmup, activates the world, and schedules
  PrintManager starts.
- `print_manager:start`: fires after the trail-start delay for every avatar in
  reverse source order, including non-present avatars.
- `warmdown:end`: fires after `round:end` and emits `game:stop`, then either
  starts the next round or ends the match.

The optimized path must not smuggle timer effects into movement code. If a
timer fires before a zero-ms movement tick in the source, the transition trace
must show that order.

## Wrapper Joint Action

This section defines CurvyZero wrapper and replay metadata. Native CurvyTron
does not have a joint-action object; it reads current player control state while
advancing elapsed-ms frames.

The public trainer-wrapper action space is `curvyzero_turn3/v0`:

| Id | Meaning | Source move |
| ---: | --- | ---: |
| `0` | left | `-1` |
| `1` | straight | `0` |
| `2` | right | `1` |

`joint_action` must contain every live player exactly once at the wrapper
decision boundary. A fixed decision cadence, when used, belongs to the wrapper
profile and should be recorded as `decision_ms`. Dead players and absent
players do not require actions. Missing live-player actions are errors.

The transition trace stores public action ids plus the normalized source move
used by the implementation. That storage is for schema comparison and replay;
it does not mean the source exposes public action ids or a joint-action step.

## Event Schema

Events are an ordered array. Ordering is part of parity.

Required normalized event fields:

- `seq`: transition-local order.
- `at_ms`: source clock when emitted.
- `phase`: `reset`, `timer`, `movement`, `collision`, `scoring`, or `lifecycle`.
- `kind`: event name.
- `player_id`: primary player id, or null.
- `other_player_id`: killer, winner, or related player id, or null.
- `bool_value`, `int_values`, `float_values`: small typed payload slots.
- `fields`: readable debug payload using source names where helpful.

2P no-bonus must support these event kinds when they occur:

- `round:new`
- `random`
- `position`
- `angle`
- `game:start`
- `print_manager:start`
- `property`
- `point`
- `die`
- `score:round`
- `score`
- `round:end`
- `game:stop`
- `end`

Bonus events are reserved until the bonus source contract is written.

Vector fixed event rows may keep compact numeric arrays internally, but they
must project to this event schema for comparison.

## State Fields

`state_before` and `state_after` must include the smallest comparable source
state needed to prove the transition.

Required row fields:

- `started`, `in_round`, `world_active`, `world_body_count`
- `frame_scheduled`, `rendered`
- `borderless`
- `death_count`, `death_player_ids`
- timer fields listed above
- RNG cursor/history fields
- overflow flags for event/body/timer capacity

Required player fields:

- `player_id`
- `present`
- `alive`
- `x`, `y`
- `angle`
- `velocity`, `angular_velocity`
- `score`, `round_score`
- `printing`
- `print_manager_active`
- `print_manager_distance`
- `print_manager_last_x`, `print_manager_last_y`
- `trail_point_count`
- body counters needed for own-trail latency and world body insertion

Required body/world fields for no-bonus parity:

- stored body position, radius, owner, body number, birth time, and old-body
  metadata where a collision can observe it.
- body write cursor/count and capacity overflow status in the fast
  implementation.

## Terminal Info

Terminal info is row-level first, then projected per player for trainers.

Required fields:

- `terminated`
- `truncated`
- `done = terminated OR truncated`
- `terminal_reason`
- `winner_ids`
- `loser_ids`
- `death_player_ids`
- `draw`
- `match_done`
- `round_done`
- `score`
- `round_score`
- `final_observation_policy`

Source terminal reasons for v0:

- `survivor_win`
- `all_dead_draw`

Truncation reasons for v0:

- `timeout_truncated`
- `event_overflow_truncated`
- `body_overflow_truncated`

Source termination takes precedence over truncation reward. Pure truncation does
not pretend to be a source death or win.

## Observation, Reward, And Replay Boundaries

Observations and rewards are derived from `state_after` plus terminal metadata.
They are not part of the rule implementation.

Observation boundary:

- Policy code reads observations and legal masks only.
- Policy observations do not include raw source events, hidden source internals,
  or trace objects.
- Each observation carries schema id and schema hash.

Reward boundary:

- Reward schema id and hash are recorded on the transition.
- The first source-training reward remains sparse round outcome:
  `0` during play, `+1` survivor, `-1` loser, `0` draw, `0` pure truncation.
- Debug score-delta rewards must keep a debug schema id.

Replay boundary:

- Replay rows store wrapper trainer action ids, action weights, root values,
  observation and reward schema ids/hashes, `episode_id`/`reset_episode_id`,
  `round_id`/`source_round_id`, lifecycle policy id, `reset_seed`,
  `reset_source`, done flags, final observation policy, and optional transition
  trace refs.
- Terminal transition data must be staged before autoreset mutates the row.
- Replay readers reject mismatched rules, observation, action, or reward hashes.

## Why "Just Optimize Now" Was Blocked

The fast path was blocked by contract risk, not lack of effort.

- There was no shared transition contract tying reset, timers, events, state,
  terminal info, observation, reward, and replay into one row-local output.
- The vector path was fixture-backed and useful, but it had not migrated
  natural source lifecycle end to end.
- Scalar parity was still incomplete: broad no-bonus multiplayer lifecycle,
  match continuation, present/alive edges, bonuses, production observations,
  replay, and autoreset were missing.
- Debug event arrays and actor-loop benchmarks measured useful speed slices,
  but they did not prove arbitrary policy rollouts.
- GPU/JAX work would have optimized an unsettled semantic target.
- Amdahl's law applies to self-play: if MCTS/search or model inference
  dominates the loop, a faster env step has capped value.

The fix is not to stop optimizing. The fix is to make every fast-path result
prove the same transition contract as the executable spec, while measuring the
whole self-play loop.

## Current Speed Read

The current P2 no-event actor bridge says the env is visible in the fake loop:

- `actor_step_p50`: about `0.557 ms`
- `env_step_p50`: about `0.271 ms`
- `synthetic_policy_p50`: about `0.060 ms`
- `env_step`: about `48.5%` of the measured fake loop

This is useful, but it is not a production bottleneck proof. The policy/search
bucket is synthetic and intentionally cheap. Real MCTS/search or model
inference may dominate, which would make env-step optimization a smaller win by
Amdahl's law.

Next benchmark requirement: measure real search/MCTS, source-backed
observation packing, action mapping, replay staging, reset/autoreset, and env
step in the same loop before starting GPU/JAX env optimization.

## Implementation Plan

1. The executable spec emits `EnvironmentTransitionV0` traces first.
   - Add a projection from `CurvyTronSourceEnv` state/events into this schema.
   - Cover reset, timer advance, wrapper joint-action metadata, state before/after, terminal
     info, and observation/reward/replay refs for 2P no-bonus.
   - Compare the projected trace to existing JS oracle fixtures before using it
     as the fast implementation target.

2. The fast implementation matches source traces for 2P no-bonus.
   - Project vector fixed arrays into the same transition schema.
   - Start with reset -> warmup -> `game:start` -> PrintManager start -> live
     movement -> wall/body death -> `round:end` -> warmdown -> match/next-round.
   - Use strict trace equality for event order and typed tolerances for floats.
   - Keep unsupported rows explicit instead of silently falling back.

3. Only then widen no-bonus multiplayer.
   - Add 3P and 4P no-bonus lifecycle traces.
   - Cover source map sizes, reverse update order, present/absent players,
     same-frame deaths, all-dead draws, survivor scoring, match end, and
     multi-round continuation.

4. Add bonuses after no-bonus vector semantics are stable.
   - The executable spec defines bonus reset/spawn/catch/effect/expiry event
     traces.
   - The fast implementation matches those traces before bonus speed claims.

5. Measure the whole self-play loop before chasing env-only speed.
   - Report env step, observation packing, action mapping, replay staging,
     model inference, MCTS/search, host/device transfer, completed games/min,
     and p50/p95/p99 action latency.
   - Use Amdahl's law: optimize the env hard only when the full loop shows env
     stepping is still a production bottleneck.
   - If MCTS/search dominates, prioritize batching/search/model work before
     deeper env micro-optimization.

6. Postpone GPU/JAX env work.
   - Continue Modal/JAX/Mctx as model/search/runtime evidence only.
   - Revisit GPU/JAX env execution only after fast CPU/vector semantics are
     stable and an end-to-end actor-loop profile proves environment stepping is
     the real production bottleneck.

7. Retire the executable spec from the hot path.
   - Keep it as an oracle/debug harness while source gaps remain.
   - Production training should use the single fast faithful environment once
     the promoted transition slices cover the needed rules.

## Promotion Gate

A fast environment can claim optimized parity for a slice only when:

- Executable-spec trace for that slice matches the JS oracle.
- Fast-implementation trace for the same inputs matches the executable-spec
  trace.
- The trace includes reset/timer ordering, events, state fields, terminal info,
  and replay boundary metadata.
- Unsupported mechanics are listed as unsupported in the result.
- Speed numbers are reported for that exact promoted slice only.
- Full-loop measurements say whether env speed, MCTS/search, model inference,
  replay, or transfer is the limiting bucket.
