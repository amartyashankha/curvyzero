# Environment Reorientation Packet

Status: working-memory packet
Date: 2026-05-11

Use this when an environment agent needs to recover the current shape quickly.
Keep the top short and push evidence details into linked docs.

## Short Answer

Main priority: harden one fast, source-faithful CurvyTron runtime:
`VectorMultiplayerEnv`.

`VectorMultiplayerEnv` is the intended runtime under hardening. The
original JS oracle and `CurvyTronSourceEnv` are proof/oracle tools: they tell us
what the runtime must do, but they are not product environments. Speed work is
useful only when it moves promoted source behavior into that runtime or measures
the whole self-play loop.

Use source language when reasoning about the game: CurvyTron has held control
state advanced by elapsed-ms frames. `step` and `joint_action` are wrapper/API
terms for our adapters, not native source semantics.

Strict `VectorTrainerEnv1v1NoBonus` is a fixed decision wrapper over native
source control state. It is narrow proof/profiling infrastructure, not the
destination and not a second product runtime. Its reset/step info exposes
`native_control_model_id`, `trainer_control_wrapper_id`, and `decision_ms` so
replay/profile rows can state that boundary plainly.
Its restrictions are temporary explicit profile configs; the reconstruction
path remains source-default CurvyTron behavior in `VectorMultiplayerEnv`.

For LightZero/MuZero, treat multiplayer as one ego decision row at a time. The
wrapper fills non-ego players from explicit versioned opponent policies, then
advances held source controls over the elapsed-ms server-frame window and
records the full wrapper
`joint_action` as replay sidecar. Repo-native self-play can use
`obs[B,P,D] -> compact live ego rows -> policy/search ->
wrapper action_map[B,P] / joint_action sidecar -> trainer env.step`.

Do not turn fixture counts, test counts, rows/sec, or synthetic rollout numbers
into the goal. The 1v1/no-bonus slice is a trainer-boundary proof only, not the
game target.

Current runtime boundary: `VectorMultiplayerEnv` is the public base
2P/3P/4P runtime being hardened. It is narrow and metadata-heavy today. It is
not full fidelity, not trainer-ready, not a full bonus env, and not
visual/pixel parity. It has partial seeded/natural bonus support for promoted
effects, including `BonusSelfMaster` and `BonusAllColor`, but not broad natural
bonus parity.

Cleanup decision: `VectorMultiplayerEnv` is a historical public env
name. Product direction is one fast source-faithful runtime, not two
implementations.

Latest fixed state: focused environment validation reported `272 passed`, source
bonus validation reported `33 passed`, ruff passed, and the environment doc
guard passed. `BonusSelfMaster` wall/body parity is fixed, `BonusAllColor`
reverse target event order plus older-wins overlap behavior are fixed, and the
stale public metadata/replay unsupported-audit claim is retired.
Seed-generated public random tape auto-extends deterministically.
Seed-generated natural bonus position retry is no longer capped by
`natural_bonus_position_attempt_capacity`; that setting is a chunk/fixture
limit. Fixture/direct finite tapes remain strict and can exhaust by design.
Public natural bonus timer advancement no longer has an artificial callback
cap. Artificial/manual bonus stack overflow is an intentional fixed-array guard
for bad or undersized direct runtime fixtures, not a public natural/seeded env
bug. A fully blocked generated map may still need policy if retries never find
a position.

Current execution now: we are no longer debating versions. One public runtime
is being hardened, and the remaining work is specific feature work plus proof,
not vague exploration. Keep each public-env, replay, trainer, bonus, and visual
claim explicit.

Docs are a single-worker lane now. Keep them concise, and avoid broad refactors
unless they directly remove confusion or unblock parity tests.

Old toy/debug paths are quarantined as historical smoke/interface evidence
only. Do not use them as environment-fidelity, public-runtime, replay, or
bonus evidence.

## 2P Status

See [active_lanes.md](active_lanes.md#2p-status) for the live 2P status. This
packet should not duplicate that paragraph.

## Current Truth

- Lifecycle/spawn/RNG has 28 pinned lifecycle fixtures, including
  `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p`, with JS oracle and Python parity
  coverage through the focused lifecycle runner.
- The pinned lifecycle slice covers focused 2P warmup/start,
  terminal-to-next-round, heading retry, 3P/4P first-round spawn/RNG, focused
  4P all-present all-dead warmdown/next-round, 4P survivor next-round,
  present/non-present cases including survivor scoring, 3P warmup and delayed
  PrintManager start, 2P and 3P match-end paths, 3P all-dead continuation,
  survivor scoring,
  tie-at-max continuation, and focused all-present 3P multi-round match end.
- `BonusGameClear` immediate clear is promoted narrowly as a forced source-env
  parity claim. It does not promote broad bonus behavior.
- The active bonus slice also has narrow `BonusSelfSmall` catch/no-catch,
  death-order no-catch, one natural one-type spawn/type/position RNG path, one
  game-world retry path, one expiry/restore path, seeded public
  `BonusSelfSmall`/`BonusGameClear`/`BonusGameBorderless` including public
  borderless expiry, seeded public bonus replay audit preservation, forced
  runtime `BonusGameBorderless`, public natural source-default type selection,
  same-frame natural bonus plus PrintManager random-order accounting, and a
  low-level `vector_runtime.py` natural spawn helper with type/position/retry/
  cap tests.
- `vector_runtime` now uses the explicit bonus table for landed optional-array
  runtime effects: `BonusSelfSmall`, `BonusSelfSlow`, `BonusSelfFast`,
  `BonusEnemySlow`, `BonusEnemyFast`, `BonusEnemyBig`, `BonusEnemyInverse`,
  `BonusGameBorderless`, and `BonusGameClear`. Keep that as a runtime claim.
  Public `BonusEnemyInverse` remains unclaimed until public state wires inverse
  arrays.
- Promoted source mechanics include elapsed-ms movement, normal wall and
  borderless behavior, body collision/order canaries, PrintManager
  toggles/start/death stops, normal trail cadence, forced trail gaps, one
  natural trail-gap source case, old-body metadata, and focused scalar
  source-env scoring/timer guards.

## What Is Not True Yet

- The environment is not fidelity-done.
- `VectorMultiplayerEnv` is not a full multiplayer environment claim.
  Remaining public multiplayer gaps include broader lifecycle, warmdown and
  immediate-terminal leave variants, trainer visual/replay adapters, a public
  bonus env, and source/browser pixel parity.
- The strict public `VectorTrainerEnv1v1NoBonus` path now has reset/spawn/warmup,
  source-ordered runtime stepping for the narrow 1v1/no-bonus slice, scoring
  for that slice, final observation/reward handoff, autoreset staging,
  replay-v0 recording, terminal metadata, seed/reset metadata across autoreset,
  reset/step control-wrapper info (`native_control_model_id`,
  `trainer_control_wrapper_id`, `decision_ms`), and an optional strict
  replay/profile manifest. It also has a source-fixture reset hook for random
  tape and warmup policy, and proves public reset-to-terminal parity for the
  long 1v1/no-bonus wall-round-done fixture. It still does not have full
  lifecycle, warmdown/next-round, broad 3P/4P scheduling, bonuses, visuals, or
  full RNG state/history/ref.
- Do not start with full wrapper joint-action MCTS for multiplayer; branching is
  `3^P`.
  Keep opponent policy ids/actions in replay. Current replay-v0 is 1v1, so
  3P/4P needs a generalized replay schema later.
- Lifecycle fixtures still mostly act as seeder rejection/RNG metadata guards
  outside their focused source-env parity role.
- Debug observations/rewards, fixture-cycled actor samples, synthetic feedback,
  synthetic Mctx, and Modal boundary runs are not production CurvyTron self-play.
- Bonus coverage is still narrow. The source oracle has broad Python
  stack/effect support for self/enemy/all bonuses, and `vector_runtime`
  optional-array support now covers the table-backed runtime effects listed
  above. Natural source-default type selection must not imply public effect
  support. Remaining gaps are public effect wiring, public inverse state arrays,
  catch/effects for newly selectable types, stack math beyond promoted restore
  cases, full public replay/final state, broader 3P/4P lifecycle/leave, and
  visual/pixel parity.

## Next Tasks

1. Harden `VectorMultiplayerEnv` as the intended runtime.
2. Keep strict public 1v1/no-bonus reset-to-terminal parity green as a
   proof/profiling boundary only.
3. Keep multiplayer in the target: 3P/4P fast-runtime wall scoring has landed;
   next add comparator coverage, then general no-bonus warmup/reset timers,
   warmdown/next-round, survivor scoring, present/non-present variants,
   mid-round leave, and longer JS/Python comparisons.
4. Keep the multiplayer trainer interface ego-row based: versioned opponent
   policies fill non-ego actions, full wrapper action maps stay in replay
   sidecars, and generalized 3P/4P replay waits until the fast parity target
   exists.
5. Keep bonus work narrow unless bonuses are enabled for training: broaden
   beyond partial public natural `BonusSelfSmall` spawn, pin timer/random
   ordering for public scheduling, add stack/expiry math, natural
   `BonusGameClear`, other runtime/public effects, and replay.
6. Turn the trainer surface into source-backed payloads: observation, legal
   mask, reward, terminal info, final observation, replay v0, reset/autoreset,
   and policy-row mapping.
7. Quarantine old toy/debug paths as smoke/interface evidence only.
8. Measure speed on the whole actor loop, not only env stepping. Track env,
   observation packing, policy/search, replay, reset/autoreset, and transfer
   cost.

## Current Process

Fidelity loop:

```text
source claim -> source read -> JS oracle/probe -> Python parity -> optimized parity -> docs
```

Speed loop:

```text
promoted claim -> fixed arrays -> vector compare -> B>1 parity/timing -> docs
```

Use focused acceptance commands for the source/interface claim being changed.
Do not maintain pass-count dashboards in these front-door docs.

## Boundaries

- Source fidelity remains the semantic guardrail.
- Speed work supports the intended `VectorMultiplayerEnv` runtime, but
  it does not define the rules.
- Keep docs concise and point to evidence docs instead of copying logs.
- Do not expose final `reset_many`/`step_many` or trainer-facing vector APIs
  until state, observation, reward, final observation, replay, and batch
  semantics are pinned.
- Do not call Modal per environment step, player, trace row, or MCTS node.
