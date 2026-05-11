# Full CurvyTron Environment Spec

Status: working memory
Date: 2026-05-09

## Purpose

The environment goal is full multiplayer CurvyTron: source-faithful, fast, and
reproducible. There is one runtime under hardening:
`VectorMultiplayerEnv`.

The target is the original server game rules, not a convenient 1v1 toy. A
1v1/no-bonus slice is only the first milestone because it is the smallest useful
boundary for reset, step, observation, reward, replay, and speed work.
Native source behavior is held player control state advanced over elapsed-ms
server frames. Trainer `step`, action ids, `joint_action`, and `decision_ms`
are wrapper/replay terms.
Restricted wrappers are temporary proof/profile configs only. The
reconstruction path is source-default CurvyTron behavior, promoted into
`VectorMultiplayerEnv` with explicit evidence.

## Source Truth

Use the original server source as the rule authority. The source-map docs under
`docs/research/curvytron_source_map/` are the working index.

- Server gameplay state and source events are the first truth.
- Client code explains input mapping, wire payloads, and rendering only.
- JS oracle fixtures and probes are the proof path for source claims.
- Python parity and vector parity must stay tied to named source claims.
- Browser pixels are later visual checks, not gameplay proof.

Runtime rule: the end goal is one fast faithful runtime,
`VectorMultiplayerEnv`. `CurvyTronSourceEnv` and the JS oracle are
truth/proof tools while source rules move into that runtime. Optimized work is
useful only when it matches the source contract it claims.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

Amdahl guardrail: optimize env-step only with measurements against the whole
self-play loop, including MCTS/search/model cost, observation packing, reset,
and replay.

Optimizer contract: the optimizer may measure only proof/profiling surfaces such
as the strict wrapper
`VectorTrainerEnv1v1NoBonus` `[B,2,106]` trainer-observation plus replay-v0
plumbing for the strict 1v1/no-bonus slice. It may not redefine source truth,
broaden environment fidelity claims, turn plumbing throughput into a learning
claim, or generalize timings to bonuses, broad lifecycle, 3P/4P, visual
LightZero, or full CurvyTron. The strict 1v1 path is not the product runtime.

## Required Multiplayer Support

The final environment must support the multiplayer shape the source supports.

- Player counts: 1P edge cases plus normal 2P, 3P, and 4P games.
- Map sizes: 1P `80`, 2P `88`, 3P `95`, 4P `101`.
- Match max scores: 1P `1`, 2P `10`, 3P `20`, 4P `30`.
- Avatar order follows room player insertion order.
- Server update order is reverse avatar order.
- `present` and `alive` are separate states.
- Scoring uses total avatar count in source places where the source does that.
- Same-frame deaths share the source death collection count captured at frame
  start; absent avatars already in `game.deaths` count.
- In 1P, `resolveScores` treats the sole avatar as the round winner even if it
  just died.

## Required Mechanics

Full source-faithful behavior must include:

- Elapsed-ms movement, source turn rate, speed coupling, inverse controls, and
  straight-angle behavior.
- Normal wall death and borderless wrap.
- Endpoint-circle body collision with strict overlap and self-trail latency.
- Trail point emission, delayed printing, PrintManager holes, and death-frame
  point side effects.
- Source round scoring, match scoring, winner bonus, and draw cases.
- Bonuses: config, spawn timing, catch order, targets, stack math, expiry,
  speed/radius/inverse/color effects, clear, and borderless.
- Source event order for state traces.
- Wire compression only when claiming wire replay fidelity.

## Lifecycle

The environment must model natural source lifecycle, not only forced fixture
states.

- `round:new`, warmup, `game:start`, delayed PrintManager start, live updates,
  `round:end`, warmdown, `game:stop`, next round, and match end.
- Natural spawn order, spawn RNG, heading retries, non-present players, and
  present/alive edge cases.
- 2P, 3P, and 4P lifecycle cases, including survivor scoring, all-dead rounds,
  match end, tie-at-max-score, and multi-round matches.

## Observations

Trainer observations must come from trusted environment state.

- Observation schemas need ids and hashes.
- Egocentric observations must avoid hidden seat or absolute-position shortcuts
  unless the schema says they are present.
- Observation fixtures should reference source-backed states, not browser
  pixels or toy-only states.
- Legal action masks must be explicit for live, dead, terminal, and padded rows.

## Rewards

The trainer reward must be simple and tied to source round outcome.

- Reward schema needs an id and hash.
- Terminal reward maps must preserve each player's outcome.
- Same-frame deaths and all-dead draws must stay explicit.
- Pure truncation must not be confused with source death or win.

## Replay

Replay must make each sample reproducible.

- Store ruleset, observation schema, reward schema, player count, episode id,
  reset seed, reset source, done, terminated, truncated, terminal reason, and
  final observation policy.
- Store wrapper action ids, legal/action weights, root values, and optional
  event/state refs.
- Reject chunks with incompatible rules or schema hashes.

## Reset

Reset must be source-shaped and row-local.

- Natural reset must spawn from row-local seed/RNG state.
- Reset must preserve terminal transition data before overwriting a row.
- Autoreset must be public and explicit, not hidden inside a step.
- Seed history and random-call history must be available for replay and debug.

## Speed

The fast path must preserve the verified rules.

- Optimize only after a source claim has oracle/probe evidence and Python
  parity.
- The hot path should be batched and local to the process.
- Modal/JAX/Mctx jobs are coarse runtime or search jobs, never per-step env
  calls.
- `B>1`, mixed player counts, timers, RNG, overflow, observation packing, and
  replay staging must each have their own checks.

## Current Status

The project has strong source-backed slices for movement, selected borders,
body collision canaries, collision order, PrintManager behavior, trail cadence,
forced trail gaps, one natural trail-gap case, old-body metadata, selected
3P/4P scoring/order canaries, and 28 pinned lifecycle fixtures including
`source_lifecycle_spawn_rng_4p_next_round`,
`source_lifecycle_survivor_score_4p_next_round`,
`source_lifecycle_present_absent_3p_survivor_score_round_end`, and
`source_lifecycle_multi_round_match_end_3p`. The focused 4P next-round fixtures
cover all-present all-dead and survivor continuation paths with next natural 4P
spawn RNG/order. The focused multi-round fixture proves only the all-present 3P
`max_score: 3` path where avatar 1 carries score 2 through `game:stop` and
`round:new`, then reaches score 4 and emits `end`.

`CurvyTronSourceEnv` now exists as the source-shaped scalar no-bonus executable
spec/oracle harness.
It is checked directly against the JS lifecycle oracle for 2P warmup and delayed
PrintManager start, 2P simple wall death through match end, focused 3P/4P
reset/spawn order, and focused 4P all-dead next-round lifecycle. It also has a narrow long-rollout proof for
`source_lifecycle_long_1v1_no_bonus_wall_round_done.json`: 111 natural 1v1
no-bonus steps reach source `round:end` after a wall death, and the final
positions, alive flags, scores, deaths, round winner, reward, `roundDone`, and
`gameDone` summary match the persistent original-JS worker when `node` is
available.

The scalar source env now stores real source-shaped body/island/world state via
`SourceBodyState`, `SourceIslandState`, and `SourceWorldState`. Body insertion
happens when source point emission occurs while the game has started and the
world is active, and `worldBodyCount` is backed by the mutable world body count.
The scalar source env also has the first direct borderless branch support:
strict outside-only wrap, first-axis opposite-position wrap, first-axis corner
wrap, exact-edge safety, and destination-body lookup skipped on the wrap frame.
It carries `borderless` on reset/source game state. `tests/test_source_env.py`
has 35 focused direct source-env pytest cases, including checks for strict opponent
overlap versus tangent safety, own-trail latency, old body metadata at 2000 ms,
wall priority, same-frame reverse-order point insertion, the active
already-hole PrintManager stop property event, normal trail exact-threshold and
epsilon behavior, two borderless checks, direct borderless PrintManager
wrap/toggle behavior, 3P ordered death scoring, 3P same-frame wall-death scoring,
a 3P absent-player scoring corner, tie-at-max next-round behavior,
timer-drain large advance across `game:stop -> round:new -> game:start`, and an
infinite-loop guard for zero-delay timer loops, non-present delayed PrintManager
starts, 1P wall-death scoring, source-verified world island corner lookup,
mid-round 2P `removeAvatar`/`player:leave`, narrow active
`BonusSelfSmall` catch/no-catch, and forced `BonusGameClear` immediate clear.
Source delayed `print_manager:start` runs for every avatar in reverse avatar
order, including non-present avatars. In 1P, source scoring treats the sole
avatar as the round winner even if it just died. `advance_timers` now loops
through every due timer up to the target time, including newly scheduled due
timers.
Focused checks already passed: `uv run pytest tests/test_source_env.py -q` and
`uv run ruff check src/curvyzero/env/source_env.py tests/test_source_env.py`.
The integrated focused command,
`uv run pytest tests/test_source_env.py tests/test_vector_runtime.py tests/test_source_lifecycle_runner.py tests/test_lifecycle_oracle.py tests/test_env_reference_defaults.py -q`,
is the current combined source/interface guard, and ruff plus the doc guard
passed. Treat it by the source claims it protects, not as a pass-count
dashboard. The companion vector runtime coverage is focused runtime-boundary tests in
`tests/test_vector_runtime.py` with 21 focused tests; it marks the next vector
runtime extraction boundary, not production optimized lifecycle readiness.

Strict `VectorTrainerEnv1v1NoBonus` has now landed narrowly in
`src/curvyzero/env/vector_trainer_env.py::VectorTrainerEnv1v1NoBonus`. It owns B
rows of vector state, returns real `float32[B,2,106]` trainer observations,
maps trainer actions to source moves, stages terminal final observation/reward
before autoreset, and resets/spawns/warms only selected done rows. Public
no-bonus stepping now uses natural PrintManager mode through
`print_manager_mode="natural_toggle"`: live rows naturally toggle print/hole
state, and wall/body deaths run PrintManager death cleanup. The public env now
also handles horizon truncation, overflow truncation, terminal metadata,
terminal barrier replay policy, replay metadata defaults from env info,
truncation reason labeling, and live-step replay recording for the strict
1v1/no-bonus slice. Safe timer diagnostic cleanup keeps benchmark-only timer
instrumentation out of normal runtime calls. A metadata-only public
`VectorMultiplayerEnv` surface now exists for 2P/3P/4P stepping and is
the intended runtime under hardening, but it does not yet claim learned trainer
observations, natural public reset parity, replay parity, or broad lifecycle
parity. The latest combined focused validation after that bridge wave was green
on the touched code/test set.

`scripts/benchmark_source_env.py` records a local scout timing for that narrow
no-bonus lifecycle. The latest local main-thread run,
`uv run python scripts/benchmark_source_env.py --repeats 20 --js --js-repeats 3`,
reported Python source env `0.000849s/rollout` and `130,689 steps/s`; the
persistent JS worker reported `0.006148s/rollout` and `18,054 steps/s`. These
numbers are local scout numbers only for this narrow no-bonus lifecycle, not a
full speed or fidelity claim.

The promoted source slice also includes
`source_live_movement_event_trace_2p_no_bonus_multistep` for live movement event
order and `source_bonus_default_weights_type_rng_step` for default multi-type
bonus RNG/type selection. The vector path has a strict public 1v1/no-bonus
trainer surface plus fixture-backed comparators. It supports a narrow promoted
slice, uses natural PrintManager mode for public base stepping, and rejects
unsupported natural lifecycle fixtures honestly.

The current 1v1/no-bonus work is useful as a proof/profiling boundary for
trainer interfaces, reset/autoreset, replay, and speed. It is not the
destination.

## Next Gap Queue

1. No-bonus multiplayer parity: broader 2P/3P/4P lifecycle, survivor scoring,
   broader present/non-present variants, and broader long/randomized source-env
   comparisons.
2. Bonuses: deterministic catch order first, then random spawn/type, stack
   math, expiry, game clear, borderless, speed/radius/inverse/color effects,
   and death interactions.
3. Production training surface: row-local seed/RNG replay history, broader
   source-backed observations, rewards, replay refs/manifests, reset/autoreset
   beyond strict 1v1/no-bonus, and terminal info through broader public/DI-engine
   boundaries.
4. Fast path: optimized parity against promoted source contracts, including
   lifecycle timers, RNG, mixed player counts, policy row mapping, and speed
   gates.

## Missing Gaps

- Multiplayer beyond the strict 1v1/no-bonus bridge and the direct seeded 3P/4P
  no-bonus wall-scoring/order fast-runtime canaries.
- Broader present/alive leave and non-present continuation behavior beyond the
  current narrow proofs.
- Broader world/island boundary behavior beyond the current corner lookup proof.
- Moving broad source semantics into the fast runtime.
- Row-local seed/RNG replay history, bonuses, broad lifecycle, 3P/4P public
  reset/warmup/replay/observation coverage, visuals, real speed gates, real
  LightZero adapter checks, and wire/browser checks remain outside this short
  checklist.

## Milestones

1. Small boundary: 1v1/no-bonus reset, warmup, step, observation, reward, replay,
   and speed checks. This proves the trainer surface on the simplest source
   slice only.
2. 2P no-bonus source lifecycle: natural spawn, round end, warmdown, next round,
   match end, reset/autoreset, final observations, and replay.
3. 3P/4P no-bonus multiplayer: source order, map sizes, scoring, same-frame
   deaths, present/alive edges, and broader match variants.
4. Bonuses beyond the current narrow catch/no-catch/death-order, one-type
   spawn RNG, one game-world spawn retry, expiry/restore, and forced
   `BonusGameClear` proofs: caps, broader stack
   effects/expiry ordering, natural clear probability/selection, borderless,
   speed, radius, inverse, color, other bonus types, vector/runtime support,
   and death interactions.
5. Fast reproducible training path: batched vector lifecycle, source-backed
   observations/rewards/replay, policy row mapping, and real LightZero boundary
   checks.

The strict public 1v1/no-bonus vector env satisfies only the first trainer
surface slice. Do not treat its `float32[B,2,106]` path or latest focused
validation count as a broad lifecycle, 3P/4P, bonus, visual, speed, or learning
result.
