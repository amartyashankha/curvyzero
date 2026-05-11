# Environment Training Interface Contract

Status: Draft boundary
Date: 2026-05-09

This page is the boundary between environment reconstruction and anything that
trains or coaches an agent. It is intentionally conservative: training can use
the environment only when the ruleset, observation, reward, termination, and
evidence level are explicit.

## Short Contract

- The coach/training agent imports only `curvyzero.env` for environment
  interaction. Source runners, scenario tools, common-trace projection, diffing,
  and Modal fidelity jobs are evidence machinery, not trainer API.
- Source-fidelity reconstruction comes first. The fidelity lane should keep
  converting source facts into scenarios, JS traces, Python traces, common
  traces, and diffs before training treats source behavior as a target.
- `curvyzero-v0` is separate from source fidelity. It is a simplified training
  ruleset, not an exact CurvyTron clone.
- Native CurvyTron semantics are real-time player input/control state plus
  elapsed-millisecond server frames. `step()`, `joint_action`, fixed decision
  cadence, and action ids are CurvyZero trainer wrapper/schema/replay
  abstractions, not source API facts.
- The strict public 1v1 env is a fixed decision wrapper over that source
  control state. Its info exposes `native_control_model_id`,
  `trainer_control_wrapper_id`, and `decision_ms`; it is not native discrete
  simultaneous actions.
- The current optimizer bridge is source-backed CurvyTron state -> trainer
  `float32[B,P,106]` rays/scalars -> replay-v0 chunks. Visual LightZero is a
  later adapter path, not today's environment target.
- The public trainer wrapper may expose one decision as a complete
  `joint_action` map. That is wrapper API, not native source behavior. Use
  fixed action ids: `0` = left, `1` = straight, `2` = right. A live-player
  wrapper step must provide every live player's action; missing live-player
  actions are errors.
- Multiplayer training should start with ego decision rows, not full
  joint-action MCTS. LightZero/MuZero gets one ego action; the wrapper fills
  non-ego actions with explicit versioned opponent policies, advances the
  elapsed-ms source frame, and records the full wrapper action map as replay
  sidecar. Repo-native self-play can compact `obs[B,P,D]` into live ego rows,
  run policy/search, then rehydrate the wrapper `joint_action[B,P]`.
- Policy inference uses `observe(ego_player)` and `legal_action_mask`; policy
  code must not read `env.state` or raw simulator internals.
- The first concrete observation/reward/terminal-info boundary lives in
  `docs/design/environment/observation_reward_contract.md`. It labels the
  current flat global observation and vector packers as debug surfaces, not
  final learned observations.
- Public trainer `reset_many`/`step_many`, JAX/GPU-native environments, and
  broad vector execution are not declared ready. `vector_runtime.step_many`
  exists as an internal supported-fixture batch step, and
  `VectorTrainerEnv1v1NoBonus` is a narrow strict public env, but neither is
  the final full CurvyTron trainer API.
- The reconstruction loop remains scenario/common-trace based:
  `scenario JSON -> JS trace -> Python trace -> common trace -> diff`.
- Event comparison is opt-in. A scenario includes events in common-trace diffs
  only when `comparison.include_events` is exactly `true`.
- Scenario tooling is split for ownership clarity: shared schema/parsing lives
  in `curvyzero.env.scenario_schema`, the toy-v0 runner lives in
  `curvyzero.env.toy_runner`, source-fidelity runners live in
  `curvyzero.fidelity.source_runners`, and `curvyzero.env.scenarios` remains a
  compatibility facade/CLI.
- Modal belongs around whole scenarios, shards, or batches. It must not sit
  inside `env.step()`, MCTS expansion, action selection, or per-tick JS/Python
  trace loops.
- A training agent must not infer source-fidelity claims from toy-v0 tests,
  raw traces, browser screenshots, or command exit codes alone.

## Stable Enough To Rely On

- Ruleset names are documented: `curvyzero-v0` for the simplified training
  scaffold and `curvytron-v1-reference` for source-derived reconstruction.
- The local fidelity loop uses scenario JSON as input and common-trace diff as
  the default comparison path.
- The local batch runner wraps the same one-scenario loop and writes a compact
  batch summary.
- The source-fidelity Python runners are narrow by design:
  `source-kinematics`, `source-normal-wall`, `source-borderless-wrap`, and the
  mixed `source-border-rules` dispatcher, plus `source-body-canary` for the
  six narrow body/trail canaries: opponent tangent safe, opponent overlap kill,
  own delta `3` safe, own delta `4` kill, same-frame point kill, and
  same-frame point control safe, plus `source-print-manager-canary` for the
  stable eight-case deterministic print-manager toggle/death-stop batch.
- The source-lifecycle runner has 18 pinned JS/Python parity fixtures including
  `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p`. This proves named 2P/3P/4P
  lifecycle slices only; broader 4P match lifecycle and broader
  present/non-present variants are still open.
- `CurvyTronSourceEnv` has narrow source-backed bonus proofs for
  `BonusSelfSmall` catch/no-catch/death-order, one-type spawn RNG, default
  multi-type weight/type RNG, one spawn retry, one expiry/restore, and forced
  immediate `BonusGameClear` clear. This does not prove natural
  `BonusGameClear` probability, catch/effects for newly selectable bonus types,
  or vector/runtime bonus support.
- `CurvyTronSourceEnv` is a temporary executable source spec, not a trainer
  runtime. The production target is one fast faithful environment; the fast path
  must pass named source claims before training relies on it.
- `vector_runtime.py` now owns the supported fixture-backed source-ordered B>1
  transition kernel, input validation, and counter contract. Benchmark scripts
  route through it, though some wrappers and actor/debug call sites may still
  call the benchmark wrapper. This is not a full trainer-ready step API yet:
  the strict public 1v1 env adds final observations, trainer rewards, replay
  rows, terminal metadata, and control-wrapper info for that slice only. Broad
  3P/4P lifecycle, broad bonuses, source fixture random-tape reset parity, and
  the full CurvyTron trainer API remain open.
- The intended trainer-facing surface is `curvyzero.env` only. Anything under
  `curvyzero.fidelity.source_runners`, scenario loading, trace normalization,
  diffing, or Modal fidelity orchestration is allowed to change for evidence
  work without becoming a trainer compatibility promise.
- Existing docs record source-kinematics movement checks and narrow border
  state checks. Treat those as narrow slice evidence, not full environment
  fidelity.
- Common trace v1 covers per-step player state fields such as position, angle,
  alive state, and optional score fields. Runner-only metadata stays outside the
  common trace.
- The Modal rule is stable: run batches remotely, keep ticks local to the
  process, and write/fetch exact artifacts.

## Not Stable Yet

- No full source-faithful training environment is declared ready.
- `curvyzero-v0` behavior remains a toy-v0 choice where it differs from source
  movement, walls, trails, scoring, bonuses, timing, and multiplayer rules.
- No vectorized, JAX-native, GPU-native, or distributed environment contract is
  declared ready.
- Broad event fidelity is not promoted. The named live movement trace
  `source_live_movement_event_trace_2p_no_bonus_multistep` is promoted, but the
  whole event surface is not stable until docs record checked batch results.
- Scenario schema compatibility is still maintained through
  `curvyzero.env.scenarios`. New fixtures should use `scenario_id`, but some
  code still reads older aliases.
- The common trace is not a training replay format. It omits rewards,
  termination flags, observations, action masks, policy metadata, and many raw
  source internals.
- The future Modal `environment_fidelity` batch app is a design target. The repo
  currently has Python-only Modal smoke paths plus local fidelity tooling.
- Broad multiplayer source fidelity is still open beyond the named lifecycle,
  wall, body, and scoring slices, especially broader 4P match lifecycle, broader
  present/non-present and leave variants, long rollouts, full bonuses, and
  server/client message behavior.
- Multiplayer has not been dropped from the target. After strict 1v1 public
  reset-to-terminal parity, the next fast-path parity target is 3P/4P for
  promoted source claims.
- Current replay-v0 is 1v1. 3P/4P replay needs a generalized schema with
  opponent policy ids/actions and full wrapper action sidecars.

## Training Boundary

The coach/training agent may use `curvyzero-v0` for plumbing and toy training
only when results are labeled as toy-v0. It may use source-fidelity artifacts to
choose the next reconstruction mismatch, but not as a claim that the training
environment matches CurvyTron.

Trainer code should import only from `curvyzero.env` and should treat all
scenario/common-trace/source-runner code as offline evidence tooling. A trainer
may read artifact summaries as experiment metadata, but it must not call source
runners, diff tools, trace projectors, browser probes, or Modal fidelity entry
points as part of reset, step, search, action selection, rollout, or evaluation.
In particular, trainer-facing code should not import from
`curvyzero.fidelity.source_runners`, `curvyzero.env.scenario_schema`,
`curvyzero.env.toy_runner`, or the `curvyzero.env.scenarios` compatibility CLI.

Before training consumes an environment as a serious target, the environment
owner must publish:

- `ruleset_id` and behavior-affecting `rules_hash`.
- Observation schema and schema hash.
- Action schema. For the public env contract this means fixed ids `0` left, `1`
  straight, and `2` right, with any source-style `-1/0/1` values confined to
  scenario/source-fidelity adapters.
- Reward schema and terminal/tie semantics.
- Reset, seeding, scenario override, and curriculum input contract.
- Episode boundary semantics, including warmup/warmdown if source fidelity is
  the target.
- Standard terminal `info` fields for every terminal or truncated step.
- Control-wrapper info: `native_control_model_id`,
  `trainer_control_wrapper_id`, and `decision_ms`.
- Evaluation scenario set and required pass criteria.
- Artifact contract for replays, traces, and failure debugging.
- Opponent policy id/action metadata when wrappers fill non-ego actions.

## Public Env API Shape

The stable training-facing API should look like a small project-owned core, not
a direct clone of any external library:

```python
from curvyzero.env import CurvyTronConfig, CurvyTronEnv

env = CurvyTronEnv(CurvyTronConfig())
obs_by_player, info = env.reset(seed=123)
obs = env.observe("player_0")
mask = env.legal_action_mask("player_0")
result = env.step({"player_0": 0, "player_1": 2})
```

Required semantics:

- `reset(seed=...)` creates one deterministic episode and returns observations
  plus reset `info`. Scenario overrides may exist, but they are an env reset
  option, not a trace-runner API.
- `step(joint_action)` advances exactly one CurvyZero trainer-wrapper decision
  for all live players. The wrapper may map that decision to one or more
  elapsed-ms source frames; the native source does not expose
  `step(joint_action)` or public action ids.
- `joint_action` is a wrapper/replay mapping from live player id to action id.
  Every live player must be present. Dead players should not require actions.
- For LightZero/MuZero single-ego rows, non-ego actions must come from explicit
  versioned opponent policies and be recorded in replay.
- Legal trainer action ids are fixed: `0` left, `1` straight, `2` right.
- `observe(ego_player)` is the policy observation path. It must be pure with
  respect to env state.
- `legal_action_mask(ego_player)` is the policy action-validity path. For the
  current simple action set it may be all true, but policy code should still use
  the mask.
- Policy, search, and model code must not read `env.state`, internal arrays, raw
  traces, source events, or scenario internals.
- Terminal results must include standard `info` keys such as `terminal_reason`,
  `winner_ids`, `loser_ids`, `death_player_ids`, `draw`, `timeout`, and
  ruleset/schema identifiers where applicable.
- Observation, reward, action-mask, done/truncated, event/ref, and replay
  metadata shapes are specified in
  `docs/design/environment/observation_reward_contract.md`. Treat that page as
  the concrete contract target and this page as the broader API boundary.

This plain joint-action dict shape is lightly inspired by Gymnasium reset/step,
PettingZoo `ParallelEnv`, and RLlib `MultiAgentEnv`. CurvyZero still owns its
core semantics; wrappers can provide third-party compatibility later.

## Future Vector Boundary

The future vector contract is split on purpose:

- Reset/init creates a fresh row, initializes RNG/counters, and schedules source
  setup timers. It is not part of the movement hot step.
- A pre-step timer phase advances row-local timers and appends any timer-created
  event rows before movement events.
- `step_arrays(state, source_moves, rng_state)` is a backend abstraction that
  starts after reset/autoreset and pre-step timers. It owns one
  movement/collision/post-collision PrintManager tick; it is not a native
  CurvyTron API.
- Autoreset happens between returned trainer transitions. The terminal step's
  rewards, final observations, and event rows are returned before the row is
  reset.

Current debug-only note: the fixture-seeded actor-loop bridge has a narrow
internal autoreset path after replay chunk staging. It exists so the local
debug benchmark can keep fixed rows moving after terminal debug surfaces; it is
not the public vector `reset_many`/`step_many` contract and is not the final
training environment.

The delayed PrintManager start fixture is the first case requiring this split:
its `Game.onStart` timer fires before the second captured zero-ms movement tick,
so a vector backend must represent setup timers and pre-step event rows instead
of smuggling start behavior into `PrintManager.test()`.

## Next Interface Decisions

1. Choose the first training target explicitly: keep `curvyzero-v0` as the
   immediate toy target, wait for a source-fidelity ruleset, or train both with
   separate labels and evaluation gates.
2. Implement and test the single-env training API shape: reset/step returns,
   wrapper joint-action mapping, observation containers, legal masks, and
   `info` fields, including control-wrapper metadata.
3. Define adapter boundaries between public action ids `0/1/2` and
   source-fidelity scenario moves `-1/0/1`.
4. Define the first opponent policy metadata contract: id/version, deterministic
   seeding, selected non-ego actions, and replay sidecar shape.
5. Extend the first reward and standard terminal-info contract for source-level
   same-frame death, round win, match win, timeout, and truncation semantics.
6. Decide what the coach needs from events beyond refs: no events, narrow
   source event fields, or a replay/debug stream separate from observations.
7. Define promotion criteria from reconstruction to training: which local or
   Modal batches must pass, which fields must match, and how failures are
   recorded.
8. Decide how scenarios enter training: fixed eval fixtures only, curriculum
   seeds, forced-state resets, or generated scenario sets.
9. Decide if and when to add `reset_many`/`step_many`; keep it after single-env
   reset/step/observe/replay semantics are stable.
10. Define batch boundaries for training infrastructure so Modal, if used, wraps
   jobs or shards and never becomes part of the environment hot loop.
11. Revisit JAX/GPU-native and vectorized env execution only after the
    single-env observation contract has tests and learning runs that justify the
    extra surface area.

## Pre-Training Checklist

- [ ] Target ruleset is named and separated from toy-v0/source-reference labels.
- [ ] Observation, action, reward, termination, and info schemas are written.
- [ ] Public env import boundary is checked: trainer code imports `curvyzero.env`
      only for environment interaction.
- [ ] Public action ids are fixed at `0` left, `1` straight, `2` right.
- [ ] Wrapper `step` requires a complete wrapper action map for all live players
      and errors on missing live-player actions.
- [ ] Reset/step info identifies the native control model, trainer control
      wrapper, and `decision_ms`.
- [ ] Policy inference uses `observe(ego_player)` plus `legal_action_mask`; no
      policy/search/model code reads `env.state`.
- [ ] Terminal and truncation `info` fields are standardized.
- [ ] Rules hash and schema hashes are included in replay/checkpoint metadata.
- [ ] Evaluation scenario set is named with exact scenario paths or manifest.
- [ ] Fidelity evidence level is stated: toy-v0, narrow source slice, or broader
      source-fidelity batch.
- [ ] Event comparison policy is stated, including whether
      `comparison.include_events` is required for the eval set.
- [ ] Modal usage is batch-only and absent from per-step training code.
- [ ] Scenario/common-trace/source-runner tools are documented as evidence
      machinery, not trainer API.
- [ ] Vector envs, JAX-native envs, and GPU env execution are explicitly deferred
      until the single-env contract is stable.
- [ ] Known source deviations are labeled `v0-choice`, `source-inspired`,
      `source-derived`, or `unresolved`.

## Pointers

- Ruleset split: `docs/design/rulesets.md`
- Observation/reward/terminal-info contract:
  `docs/design/environment/observation_reward_contract.md`
- Current simulator contract: `docs/design/deterministic_environment.md`
- Reconstruction loop: `docs/design/environment/reconstruction_workflow.md`
- Scenario/common-trace contract: `docs/design/environment/trace_loop_contract.md`
- Local batch plan: `docs/design/environment/probe_automation_plan.md`
- Modal batch boundary: `docs/design/environment/modal_fidelity_jobs.md`
- External API pointers: Gymnasium Env reset/step/render
  (`https://gymnasium.farama.org/api/env/`), PettingZoo ParallelEnv simultaneous
  dict reset/step (`https://pettingzoo.farama.org/api/parallel/`), and RLlib
  MultiAgentEnv agent-id dict returns
  (`https://docs.ray.io/en/latest/rllib/multi-agent-envs.html`).
