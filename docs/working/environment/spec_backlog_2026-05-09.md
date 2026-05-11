# Environment Spec Backlog

Status: working memory for SPEC_LIST
Date: 2026-05-09
Owner: SPEC_LIST

This page lists the environment specs still needed before we can honestly say
the training setup has a real CurvyTron target. It uses simple status words and
keeps unsupported work visible instead of calling it done.

This page is the SPEC_LIST-owned backlog. Keep it aligned with
`active_lanes.md` and `coverage_tracker.md`.

There is one runtime under hardening: `VectorMultiplayerEnv`.
`CurvyTronSourceEnv` and the source JS oracle are proof tools. Strict
`VectorTrainerEnv1v1NoBonus` is a narrow proof/profiling boundary, not the
destination.

Top-level target spec:
[full_environment_spec_2026-05-09.md](full_environment_spec_2026-05-09.md).

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Current Snapshot

- `src/curvyzero/env/vector_runtime.py::step_many` is the supported
  fixture-backed source-ordered CPU transition kernel.
- `scripts/benchmark_vector_batch_rows.py` routes normal calls through that
  transition kernel. Private `_step_many_kernel(..., phase_timers=...)` is
  benchmark diagnostics only, and the dead duplicate old benchmark body has
  been removed.
- Runtime terminal survivor/draw rows now mark optional `done`, `terminated`,
  `reset_pending`, `terminal_reason`, `draw`, and `winner` arrays when present.
- `vector_lifecycle.run_warmup_start_step_1v1_no_bonus_rows` composes strict
  1v1/no-bonus reset/spawn/warmup/timer/runtime stepping. A focused test proves
  wall-death terminal state plus real vector trainer final-observation/reward
  handoff into autoreset planning.
- Do not claim a trainer-ready environment yet. The strict public 1v1/no-bonus
  vector trainer env has landed, public-step horizon truncation is now in that
  live path, and a live-step replay recorder can build replay-v0 chunks from
  returned env batches. Overflow truncation, terminal metadata, the
  terminal barrier replay policy, replay metadata defaults from env info,
  truncation reason labeling, reset/step control-wrapper info
  (`native_control_model_id`, `trainer_control_wrapper_id`, `decision_ms`), and
  safe timer diagnostic cleanup have also landed for that proof/profiling
  slice. Urgent
  gaps remain: row-local seed/RNG history, replay manifest hardening/compaction, broad
  lifecycle coverage for warmdown/next round/3P/4P/bonuses, the visual
  renderer/LightZero boundary, and performance integration.
- The source-pinned bridge-test wave has landed: same-frame wall draw replay,
  borderless wrap destination-body/next-frame kill, borderless PrintManager
  wrap toggle, collision-order batch support, seed/reset metadata across
  autoreset into recorder chunks, an optional strict replay/profile manifest,
  and the long 1v1 wall-round-done
  terminal-step public-vector bridge. Treat these as source-pinned bridge tests
  only. They are not broad lifecycle, bonus-system, 3P/4P, visual LightZero, or
  full CurvyTron coverage.
- The long public-vector source bridge now has two checks.
  `test_public_vector_env_matches_source_long_1v1_wall_round_done_terminal_step`
  seeds from the source penultimate frame and validates the terminal step.
  `test_public_vector_env_reset_to_terminal_matches_source_long_1v1_fixture`
  uses source fixture random tape plus exact warmup policy and validates public
  reset/spawn/warmup through terminal for the same long wall-round-done path.

## Current Priority Claims

- The destination is full multiplayer CurvyTron on
  `VectorMultiplayerEnv`: source-faithful, fast, and reproducible. The
  1v1/no-bonus work is a small proof/profiling boundary for trainer interfaces
  and speed checks, not the final environment shape.
- The environment is not full fidelity yet. Current claims are fixture-backed
  and must stay narrower than "real CurvyTron training."
- Runtime rule: the end goal is one fast faithful runtime under hardening.
  `CurvyTronSourceEnv` and the JS oracle are truth/proof tools while source
  rules move into `VectorMultiplayerEnv`. Optimized and speed work must
  name the source contract they match.
- Trainer-facing `step()`, `joint_action`, fixed decision cadence, and action ids
  are CurvyZero wrapper/schema/replay abstractions. The strict public env is a
  fixed decision wrapper over source control state, not native discrete
  simultaneous actions. The source target remains real-time player control
  state advanced through elapsed-millisecond server frames.
- Wrapper restrictions are temporary explicit profile configs, not the
  reconstruction path; keep source-default CurvyTron as the target.
- Multiplayer trainer interface rule: LightZero/MuZero should see one ego
  decision row at a time. The wrapper fills non-ego players with explicit
  versioned opponent policies, advances held source controls over the elapsed-ms
  server-frame window, and logs the full wrapper action map / `joint_action`
  sidecar. Repo-native self-play can
  use `obs[B,P,D] -> compact live ego rows -> policy/search ->
  wrapper action_map[B,P] / joint_action sidecar -> trainer env.step`.
- Amdahl guardrail: optimize env-step only with measurements against the whole
  self-play loop, including MCTS/search/model cost, observation packing, reset,
  and replay.
- Optimizer critique: native vector timing is useful only with explicit
  included components. Ray/observation-bound timing is actionable. Large CPU
  batch regressions need a breakdown by env step, observation, replay/reset,
  and policy/search before any rewrite claim.
- Current proof/profiling timings are fenced to strict wrapper
  `VectorTrainerEnv1v1NoBonus` `[B,2,106]` plus replay-v0 plumbing. Do not
  generalize them to bonuses, broad lifecycle, 3P/4P, visual LightZero, or full
  CurvyTron.
- The current top source claim is `natural lifecycle/spawn/RNG`. It now has JS
  oracle evidence and direct Python parity for 28 pinned lifecycle fixtures
  including `source_lifecycle_spawn_rng_4p_next_round`,
  `source_lifecycle_survivor_score_4p_next_round`,
  `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
  `source_lifecycle_multi_round_match_end_3p`: three 2P
  lifecycle fixtures, one focused 3P first-round spawn-order fixture, one
  focused 3P warmup/PrintManager-start fixture, one focused 4P first-round
  spawn-order fixture, one focused 4P all-present all-dead warmdown/next-round
  fixture, one focused 4P survivor warmdown/next-round fixture, one focused 3P
  first-round present/absent fixture, one focused 3P present/absent
  survivor-scoring fixture, and one focused 3P present/absent
  warmdown/next-round fixture, one focused 2P
  max-score match-end fixture, one focused all-present 3P `max_score: 2`
  match-end fixture, plus one focused 3P all-dead forced wall-death
  warmdown/next-round fixture, one focused 3P survivor-scoring `round:end` fixture, and
  one focused 3P survivor warmdown/next-round fixture, one focused 3P
  tie-at-max continuation fixture, plus one focused 3P all-present
  multi-round match-end fixture.
- That pinned slice proves natural round creation for those fixtures, 2P spawn
  order, spawn RNG calls, accepted and rejected 2P heading RNG attempts, print
  timer scheduling, terminal event order, warmdown, `game:stop`, next-round
  scheduling, focused 3P first-round spawn order, focused 3P warmup and
  delayed PrintManager start, focused 4P first-round spawn order, and one 3P
  non-present avatar skipped for spawn RNG and added to deaths. It also proves
  one focused 3P present/non-present continuation where `game:stop` resizes to
  present-player arena size, then the next `round:new` re-adds the absent avatar
  to deaths and spawns only present avatars. It also proves one `max_score: 1`
  match-end path with `round:end`, `game:stop`, `end`, and no immediate next
  `round:new`. The focused all-present 3P match-end fixture proves only
  `max_score: 2`: avatars 3 then 2 die, avatar 1 reaches score 2,
  `round:end` winner 1 emits at 3000 ms, `game:stop` and `end` emit at
  8000 ms, and no immediate `round:new` follows. The focused 3P continuation
  proves only all-three-dead forced wall deaths, `round:end` winner null,
  `game:stop`, next `round:new`, and next natural 3P spawn RNG/order. The focused 3P survivor
  continuation proves only avatar 1 warmdown movement/death at 4150 ms,
  `game:stop`, next `round:new`, and next natural 3P spawn RNG/order.
  The focused 3P tie-at-max fixture proves tied leaders continue to next round.
  The focused 3P multi-round fixture proves one all-present path where avatar 1
  carries score 2 into the next round and later reaches match end.
  `tests/test_source_lifecycle_runner.py` compares Python events,
  `randomCalls`, and snapshots against `tools/reference_oracle/lifecycle_oracle.js`.
  Still missing: broader 4P match lifecycle beyond the focused all-dead and
  survivor next-round paths, broader present/non-present variants, broad vector
  lifecycle, trainer/replay/final observation beyond the strict slice,
  optimized parity, and broader reset follow-through.
- Reset/autoreset, replay, observation, bonuses, broader vector semantics, and
  real LightZero remain limited to narrow claims until lifecycle/spawn/RNG
  expands beyond the current pinned slice and production row/reset contracts
  exist.
- Existing trail-gap, old-body, PrintManager, body, border, movement, and vector
  checks are regression guardrails. Keep their exact outputs beside acceptance
  commands, not in this priority list.
- Narrow reset/timer/autoreset-related slices have landed: toy-v0 refuses hidden
  post-terminal stepping and returns `final_observation` info;
  `reset_array_rows(target, source, reset_mask)` can copy selected vector rows;
  `_bool_row_mask(...)` validates row masks; `final_transition_mask(done,
  truncated=None)` builds final-row masks; batch rows have a focused B>1
  delayed-start pre-step timer helper; and the actor bridge has debug-only
  internal autoreset after replay staging. `src/curvyzero/env/vector_autoreset.py`
  now adds a public-facing autoreset planner and narrow
  `apply_autoreset_rows(...)` helper that stages final arrays before calling
  `vector_reset.reset_arrays(...)`.
- `src/curvyzero/env/vector_reset.py` now has a production-facing masked reset
  boundary. It does not natural-spawn, schedule timers, autoreset, preserve
  final observations, write replay, or expose a trainer API.
- `src/curvyzero/env/vector_lifecycle.py` composes reset plus source-shaped
  spawn, has `reset_spawn_warmup_1v1_no_bonus_rows(...)` for the strict
  1v1/no-bonus `GAME_START` warmup timer, and now has
  `run_warmup_start_step_1v1_no_bonus_rows(...)` for reset/spawn/warmup timer
  advancement plus runtime stepping. The matching runtime can advance through
  `game:start`, delayed PrintManager starts, and a focused wall-death terminal
  step; terminal survivor/draw rows mark optional lifecycle arrays when
  present. It still reports `full_lifecycle=false` and does not handle
  warmdown/next-round, replay semantics beyond the strict live-step recorder,
  broad public env semantics, bonuses, visual rendering, performance
  integration, or broad 3P/4P scheduling.
- Training is not ready. Horizon truncation, overflow truncation, terminal
  metadata, terminal barrier replay policy, replay metadata defaults from env
  info, truncation reason labeling, and live-step replay recording have landed
  only for the strict public 1v1/no-bonus path. Reset/step info also exposes
  the native control model id, trainer control wrapper id, and `decision_ms`.
  Still missing: production `reset_many` beyond that slice, row-local seed/RNG
  history, replay manifest hardening/compaction, broad lifecycle coverage for
  warmdown/next round/3P/4P/bonuses, source-backed trainer observation fixtures
  beyond the strict handoff, the visual renderer/LightZero boundary, and full
  terminal-info wiring beyond the strict public slice.
- The first trainer observation/action/reward adapter contract is now pinned by
  `docs/working/environment/trainer_observation_reward_contract_v0_2026-05-09.md`
  and `src/curvyzero/env/trainer_contract.py`. A first pure helper now lives in
  `src/curvyzero/env/trainer_observation.py` and returns the pinned
  `rays float32[24,4]`, `scalars float32[10]`, flat `float32[106]`, masks, and
  sparse reward info for current toy/grid `EnvState`. It now also has
  `observe_1v1_egocentric_rays_v0(...)`, which stacks both ego rows into stable
  `(2, 106)` observations, `(2, 3)` masks, sparse rewards, `reward_map`,
  terminal `final_reward_map`, and done/terminated/truncated metadata. This is
  not a LightZero wrapper, vector body-array implementation, or production
  source-faithful training claim.
- The strict vector handoff now has
  `src/curvyzero/env/vector_trainer_observation.py`, which builds pinned
  vector body-circle ray observations and `float32[B,2,106]` final observation
  arrays plus sparse final reward maps. `tests/test_vector_lifecycle.py` no
  longer uses `target['pos']` or `target['score']` as fake terminal payloads in
  the strict 1v1 proof.
- The strict public vector env now has
  `src/curvyzero/env/vector_trainer_env.py::VectorTrainerEnv1v1NoBonus`. It
  composes reset/spawn/warmup, `vector_runtime.step_many`, real vector trainer
  observations, sparse terminal rewards, terminal snapshot-before-autoreset,
  public-step horizon truncation, overflow truncation, terminal metadata,
  terminal barrier replay policy, live-step replay recording, reset/step
  control-wrapper info, and selected-row reset/spawn/warmup after terminal rows.
  It is still only 1v1/no-bonus and not a full trainer-ready CurvyTron
  environment.
- LightZero-specific work is contract/smoke only until the v0 adapter details
  are proven at the real boundary: `reset`, `step`, debug observation labeling,
  legal action mask, scalar reward, `done=terminated OR truncated`, terminal
  `info`, sidecar metadata, and the wrapper mapping between LightZero's
  single-action env rows and CurvyZero's joint-action replay/step abstraction
  over CurvyTron controls.
- The old local toy `CurvyTronEnv` LightZero-shaped smoke is historical adapter
  regression coverage only. Installed scalar/debug no-train smokes now cover
  config/import/reset/step plumbing where noted, but none of these is a
  training run, product environment, or stronger environment-fidelity claim.
- Policy ego-row mapping has a first pure helper:
  `src/curvyzero/training/policy_row_mapping.py` compacts or pads
  `obs[B,P,...]` plus live/legal masks into policy rows, records env row ids and
  player ids, and maps selected trainer action ids back to a wrapper action map
  (`joint_action[B,P]` sidecar) with dead/padded rows left as no-op. The actor
  bridge now uses it around the synthetic policy/search stand-in before
  rehydrating selected actions back to the wrapper action map; this is the
  repo-native
  shape for multiplayer self-play. The policy/search itself is still fake and
  later synthetic feedback steps are still not source-compared.
- Local debug replay chunk helpers are now wired to the actor bridge only for
  sample output: `--sample-only --sample-replay-chunk PATH` writes one
  validated `.npz` chunk with compatibility metadata. Timed benchmark runs
  still use the in-memory ring and do not create production replay shards.
- A separate replay v0 contract now exists in
  `src/curvyzero/training/replay_chunk_v0.py` for 1v1/no-bonus training chunks:
  observation, reward, action, action weights, root value, done/terminated/
  truncated, `episode_id`, `reset_seed`, `reset_source`, `final_observation`,
  `final_reward_map`, and compatibility hashes. The actor bridge sample path
  can now write and round-trip this contract for P=2 using
  `--sample-only --sample-replay-v0-chunk PATH`, but the payload is still debug
  obs/reward data and is marked blocked for production training.
- `src/curvyzero/training/vector_env_replay_recorder.py` now adds the narrow
  live-step recorder for `VectorTrainerEnv1v1NoBonus.step(...)` batches. It
  packs returned trainer observations/rewards, actions, action weights, root
  values, done/terminated/truncated flags, reset metadata, and terminal final
  arrays into replay-v0. This is still strict 1v1/no-bonus plumbing, not broad
  production replay.
- The JS reuse worker now has a long 1v1/no-bonus source proof:
  `source_lifecycle_long_1v1_no_bonus_wall_round_done.json` drives 111 real
  original-JS `step` calls after one source reset, keeps `sourceLoadCount == 1`,
  and reaches source `round:end`. It does not drive the 5s warmdown to
  `gameDone`.
- `src/curvyzero/env/source_env.py` now has `CurvyTronSourceEnv`, a
  source-shaped scalar no-bonus environment with real source-shaped
  `SourceBodyState`, `SourceIslandState`, and `SourceWorldState` storage.
  It now has first direct borderless branch support for strict outside-only
  wrap, first-axis opposite-position wrap, first-axis corner wrap, exact-edge
  safety, and destination-body lookup skipped on the wrap frame. It carries
  `borderless` on reset/source game state. `tests/test_source_env.py` has
  focused direct source-env tests. It checks the env directly
  against `tools/reference_oracle/lifecycle_oracle.js` for 2P warmup and
  delayed PrintManager start, 2P simple wall death through `max_score: 1` match
  end, focused 3P/4P reset/spawn order, and one long 111-step 1v1/no-bonus
  wall-round-done rollout. The direct source-env coverage is now 35 focused
  tests for opponent strict overlap versus tangent safety, own-trail latency,
  old body metadata at 2000 ms, wall priority over body collision, same-frame
  reverse-order point insertion, the active already-hole PrintManager stop
  property event, two borderless checks, direct borderless PrintManager
  wrap/toggle behavior, 3P ordered death scoring, 3P same-frame wall-death
  scoring, a 3P absent-player scoring corner, tie-at-max next-round
  behavior, timer-drain large advance across
  `game:stop -> round:new -> game:start`, an infinite-loop guard for zero-delay
  timer loops, normal trail exact-threshold and epsilon behavior, non-present
  delayed PrintManager starts, 1P wall-death scoring, source-verified world
  island corner lookup, focused 2P mid-round `removeAvatar` leave, and seeded
  active `BonusSelfSmall` catch/no-catch behavior.
  `advance_timers` now loops through every due timer up to the target time,
  including newly scheduled due timers.
  The long rollout reaches source `round:end`; when `node` is
  available, the final summary matches the persistent original-JS worker. It is
  not full fidelity yet: multiplayer beyond 1v1, broader present/alive leave,
  non-present continuation, broader bonuses, and moving source semantics into
  the fast runtime are still missing. The named live movement event trace
  `source_live_movement_event_trace_2p_no_bonus_multistep` is promoted.
  Focused checks already passed: `uv run pytest tests/test_source_env.py -q`
  and `uv run ruff check src/curvyzero/env/source_env.py tests/test_source_env.py`.
- `tests/test_vector_runtime.py` now has 21 focused vector runtime boundary
  tests for the first runtime extraction boundary. The integrated focused
  command,
  `uv run pytest tests/test_source_env.py tests/test_vector_runtime.py tests/test_source_lifecycle_runner.py tests/test_lifecycle_oracle.py tests/test_env_reference_defaults.py -q`,
  is the current focused command, and ruff plus the doc guard passed. This is
  current focused coverage for the scalar witness plus vector runtime boundary,
  not a production training or full optimized lifecycle claim.
- `scripts/benchmark_source_env.py` is the local scout benchmark for that
  narrow long source-env lifecycle. Latest local main-thread run,
  `uv run python scripts/benchmark_source_env.py --repeats 20 --js --js-repeats 3`,
  reported Python source env `0.000849s/rollout` and `130,689 steps/s`, and
  persistent JS worker `0.006148s/rollout` and `18,054 steps/s`. Treat these as
  local scout numbers for the narrow no-bonus lifecycle, not full speed or
  fidelity gates.
- Reset/timer/autoreset has an implementation checklist:
  `docs/working/environment/reset_timer_autoreset_plan_2026-05-09.md`.

## Why This Took Too Long

The backlog got hard to use because there was no single spec/gap queue, narrow
proofs were written like broader status, docs repeated stale claims, and
vector/speed work sometimes advanced without a tight source-contract label.
Working correction: promote gaps in the queue below, use source-env and the JS
oracle as temporary truth/proof tools, and make the fast path prove optimized
parity against that contract.

## Current Aggressive Gap Queue

Plain remaining issues before stronger training claims:

1. Public 1v1 vector env API: landed narrowly for strict 1v1/no-bonus
   reset/step/autoreset/final-observation/reward plus public-step horizon
   truncation, overflow truncation, terminal metadata, reset/step
   control-wrapper info, terminal barrier replay policy, and live-step replay
   recording. Source-fixture reset controls now prove the long 1v1 no-bonus
   wall-round-done rollout from public reset through terminal. Do not widen this
   claim to full CurvyTron.
2. Natural PrintManager mode: landed for public base stepping by combining
   live print/hole toggling with death cleanup.
3. Replay: B>1 replay packing and the live-step replay recorder have landed for
   the strict public env path, including terminal metadata, seed/reset metadata
   across autoreset, reset/step control-wrapper info, an optional strict
   replay/profile manifest, and the terminal barrier replay policy. The focused
   guard with the metadata-only multiplayer env is green on the touched set.
   Still missing are row-local full RNG state/history/ref,
   compaction, and broader source refs.
4. Bridge-test wave: landed for same-frame wall draw replay, borderless wrap
   destination-body/next-frame kill, borderless PrintManager wrap toggle,
   collision-order batch support, seed/reset metadata across autoreset, optional
   strict replay/profile manifest, direct seeded 3P/4P no-bonus runtime
   wall-scoring/order canaries, one narrow 3P warmdown/next-round helper, a
   metadata-only public multiplayer env surface, and the long 1v1
   wall-round-done terminal-step plus reset-to-terminal source bridges. These
   are source-pinned bridge tests, not broad lifecycle, bonuses, learned 3P/4P
   observations, natural public 3P/4P reset/warmdown/replay, or full CurvyTron.
5. Multiplayer is not dropped. Keep the current direct seeded 3P/4P
   wall-scoring/order canaries and metadata-only public env tests green, then
   move broader promoted 3P/4P source claims into the fast path:
   warmdown/next-round, natural public reset/replay/observations, match-end,
   present/non-present, survivor scoring, and leave/alive edges.
6. Multiplayer policy/search starts ego-row based. Do not start with full
   wrapper joint-action MCTS because branching is `3^P`; keep opponent policy
   ids/actions and the full wrapper action map in replay. Current replay-v0 is
   1v1, so 3P/4P needs a generalized schema later.
7. Bonuses: only narrow source proofs exist. Spawned multi-type effects,
   stacking/expiry, borderless interactions, death interactions, and
   vector/runtime support remain.
8. Replay metadata: row-local seed/RNG history is still thin; add it to reset
   info and replay metadata, then reject missing history once the metadata
   contract exists.
9. Fidelity comparisons: run longer source-env versus original-JS rollouts
   across lifecycle and bonus paths, not only the current 111-step
   1v1/no-bonus proof.
10. LightZero: visual/real adapter work stays later, after the public env,
   replay, observations, reward, masks, and terminal info are source-backed.
11. Speed: measure whole-loop performance after reproduction/parity; speed does
   not outrank source reproduction.

Recent source-gap planner next priorities: keep the strict public
1v1/no-bonus reset-to-terminal proof green; broaden 3P/4P fast parity beyond
the current direct seeded wall-scoring/order canaries only from promoted source
claims; then add broader 2P/3P/4P lifecycle, the next thin bonus slice, and
leave visual LightZero plus speed work behind source reproduction.

## Parallel Spec Chunks Now

1. `full multiplayer target`: keep
   [full_environment_spec_2026-05-09.md](full_environment_spec_2026-05-09.md)
   as the destination contract. Use 1v1/no-bonus only as the first boundary for
   reset, observation, reward, replay, and speed work.
2. `no-bonus multiplayer parity`: keep the current 28-fixture
   JS/Python-direct lifecycle slice including
   `source_lifecycle_spawn_rng_4p_next_round`,
   `source_lifecycle_survivor_score_4p_next_round`,
   `source_lifecycle_present_absent_3p_survivor_score_round_end`, and
   `source_lifecycle_multi_round_match_end_3p` as a regression guard, then add
   broader 3P present/non-present variants, broader 4P match lifecycle,
   vector lifecycle, trainer/replay/final observation, or
   broader present/alive leave cases only as named source claims.
   This remains the top source priority.
3. `bonuses`: source-read and draft deterministic catch/effect fixtures after
   the no-bonus multiplayer queue is moving; stack/expiry and random spawn/type
   follow.
4. `production reset/autoreset`: the strict public 1v1 vector env exists now,
   including public-step horizon truncation, overflow truncation, terminal
   metadata, reset/step control-wrapper info, terminal barrier replay policy,
   and source-fixture reset parity for the long 1v1 no-bonus rollout. Do not
   claim broader production semantics until reset state, timers, seed history,
   and broader lifecycle/spawn/RNG claims are pinned.
5. `production replay`: draft row metadata, final-observation policy,
   reward/return map, terminal refs, manifests, and schema/rules hash rejection;
   the replay v0 file-level contract, strict live-step recorder, replay row
   metadata, and optional strict replay/profile manifest exist. The next work is
   hardening seed/RNG history, broader source refs, manifests, compaction, and a
   later generalized 3P/4P schema with opponent policy ids/actions.
6. `trainer observation/reward`: keep schema contract tests moving, then back
   final observation and reward/return map with trusted source-state fixtures
   from the no-bonus and bonus source queue.
7. `fast-path vector semantics`: add row-local RNG arrays, mixed-P policy,
   overflow/truncation, timer rows, and vector comparators only from promoted
   source contracts.
8. `real LightZero`: keep the local smoke as plumbing; real visual/ray adapter
   semantics and registration wait for reset/autoreset, replay,
   observation/reward, and vector row contracts.
9. Modal and browser work stays as coarse artifact/runtime evidence unless it
   runs the real CurvyTron state path.

## Detailed Backlog

Status words:

- `open`: not specified or not implemented enough.
- `partial`: some evidence exists, but the spec is not broad enough.
- `blocked`: waiting on another piece of work.
- `ready-to-run`: the next command or implementation step is clear.
- `done-for-now`: enough for the current narrow claim, but not a broad claim.

Training block words:

- `yes`: serious source-faithful training should not start without it.
- `indirect`: does not block toy plumbing, but blocks a stronger source claim.
- `no`: useful guardrail or speed work, but not a training API blocker.

Parallel words:

- `yes`: safe to split out now.
- `after source`: wait for the source fixture or contract to land first.
- `later`: keep visible, but do not spend a worker now.

| ID | Spec still needed | Status | Evidence now | Owner | Exact next step | Blocks training? | Parallelize now? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | Source trail-gap four-case source contract | done-for-now | Source batch protects hole-space safe, stored body still kills, print-to-hole boundary kills, and hole-to-print boundary kills. | SOURCE | Keep `scenarios/environment/source_trail_gap_batch.json` in the source regression set and rerun it after trail, PrintManager, or runner edits. | indirect | no |
| S2 | Natural multi-step trail-gap source contract | done-for-now | `source_trail_gap_natural_multistep_hole_crossing.json` has JS/Python common trace parity through `source-trail-gap-canary`; it proves one taped PrintManager hole crossing where p0 stays alive. | SOURCE | Keep it as a separate loop check after gap edits; add broader natural variants only when they isolate a new rule. | indirect | no |
| S3 | Normal trail exact-threshold and multi-step cadence | partial | `source_trail_batch.json` covers normal point insertion and below-radius no-point. `tests/test_source_env.py` now pins exact-radius no-draw and radius-plus-epsilon draw behavior in the scalar source env. | SOURCE | Add a multi-step cadence fixture only if it isolates a new source rule; rerun `source_trail_batch.json` after trail edits. | indirect | yes |
| S4 | Old-body `old:true` death metadata | partial | Separate metadata batch covers one seeded old opponent body with `age_ms=2000`; broader boundary coverage is not claimed. | SOURCE | Add a young/old boundary pair only if broader metadata coverage becomes needed. | indirect | later |
| S5 | Natural lifecycle/spawn/RNG source claim | partial | Python direct parity now exists for 28 pinned lifecycle fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p`. `tests/test_source_lifecycle_runner.py` compares Python events, `randomCalls`, and snapshots against `tools/reference_oracle/lifecycle_oracle.js` for the current 2P/3P/4P focused lifecycle slice: warmup/start, heading retry, terminal/warmdown/next-round, 3P/4P spawn order/RNG labels, focused 4P all-dead and survivor next-round paths, focused present/absent first-round, survivor-scoring, and next-round paths, 2P/3P match ends, 3P all-dead continuation, 3P survivor round-end and next-round continuation, tie-at-max continuation, and all-present 3P multi-round match end. | SOURCE | Add broader 4P match lifecycle, broader present/non-present variants, vector lifecycle, trainer/replay/final observation, or broader present/alive leave cases only when each isolates a new source rule. | yes | yes |
| S5a | Temporary executable source harness | partial | `CurvyTronSourceEnv` remains the scalar executable spec/oracle harness. It is checked against JS lifecycle oracle cases for 2P warmup/start, 2P wall death/match end, focused 3P/4P reset/spawn order, one promoted movement trace, one mid-round 2P leave case, and one long 111-step 1v1/no-bonus wall-round-done rollout. Public vector bridge coverage now includes two strict 1v1/no-bonus checks for that long fixture: terminal-step parity from the source penultimate frame, and reset-to-terminal parity using the source fixture random tape/warmup policy. Direct fast-runtime multiplayer coverage now includes seeded 3P/4P no-bonus wall-scoring/order canaries only; it is not public multiplayer reset, warmup, replay, or observation coverage. This is strong narrow evidence, not broad full-CurvyTron parity. Missing: broader multiplayer public env coverage, broader present/alive leave, non-present continuation after start, broad bonus behavior, broader public-vector lifecycle beyond the named long 1v1 proof, and moving more source semantics into the fast runtime. | SOURCE/VECTOR | Keep `CurvyTronSourceEnv` as the source-backed oracle, port missing source rules into the one fast runtime, and compare broader rollouts before promoting trainer or vector claims. | yes | yes |
| S6 | Bonus config, spawn, catch, stack, expiry | partial | JS oracle fixtures and `CurvyTronSourceEnv` now verify the narrow active `BonusSelfSmall` catch/no-catch/death-order slice: seeded map bonus catch after movement, strict-overlap tangent no-catch, and same-tick wall death no-catch. The wall-death proof keeps `bonusCount=1`, emits no `bonus:clear`/`bonus:stack`, leaves p0 radius at `0.6`, and awards p1 the round. Python filters JS's non-important death `point` event, so this is not full hidden point-event parity. `source_bonus_spawn_type_position_rng_step.json` has JS/Python source-env parity for the natural one-type spawn/type RNG proof: five labeled draws, one `BonusSelfSmall` at `(23.94, 64.06)`, `bonusCount=1`, and `bonus:pop` before zero-elapsed source position events. `source_bonus_default_weights_type_rng_step.json` adds one default multi-type weight/type RNG proof selecting `BonusAllColor`. `source_bonus_spawn_game_world_retry_step.json` adds one rejected game-world candidate followed by one accepted retry at `(68.072, 19.928)`. `source_bonus_self_small_expiry_restore_step.json` adds one timed expiry proof: after catch, the `7500` ms timeout restores p0 radius to `0.6`, emits `bonus:stack remove`, leaves `bonusCount=0`, and does not emit a second `bonus:clear`. `source_bonus_game_clear_immediate_step.json` adds one forced `BonusGameClear` proof: seeded catch after safe movement, one preexisting main-world body cleared, event order `bonus:clear` then `clear`, `worldActive=true`, `worldBodyCount=0`, no `bonus:stack`, no avatar property change, and no active avatar bonuses. | SOURCE | Keep these as the first bonus proofs; next add caps, broader stack math/expiry ordering, borderless, speed/radius/inverse/color, other bonus types, vector/runtime support, and broader death interactions as separate fixtures. | indirect | yes |
| S7 | Random stream policy for source features | partial | PrintManager random call order and one random cadence case are pinned. The current lifecycle slice proves source-chronological 2P spawn, heading retry, PrintManager start/stop RNG labels, focused 3P first-round spawn RNG labels, and one 3P non-present avatar skip. The minimal bonus spawn fixture pins the source order for bonus start delay, next delay, one-type selection, and x/y position; the retry fixture adds one bonus position retry pair. Broader random policy is not vector/reset-backed. | SOURCE | Turn the verified call labels into one row-local RNG stream contract with tape/state/cursor rules for spawn, PrintManager, and bonuses. | indirect | yes |
| S8 | Natural spawn positions/headings | partial | Python direct parity for the current pinned lifecycle slice pins reverse 2P spawn order, x/y RNG labels, accepted first-attempt headings, one rejected heading attempt followed by an accepted retry, no immediate world bodies after spawn, next-round 2P spawn order, focused 3P first-round spawn order, focused 4P first-round spawn order, and one 3P non-present spawn skip. | SOURCE | Add broader spawn rejection or lifecycle cases only when they isolate a new source rule. | yes | later |
| S9 | Borderless follow-up corner/body/island edges | partial | Borderless wrap, PrintManager wrap, destination-body skip, and exact-edge/corner-axis are promoted. | SOURCE | Add next-frame second-axis wrap or island-boundary body lookup only if another spec depends on it. | indirect | later |
| S10 | Multiplayer scoring/order stress beyond current canaries | partial | Normal-wall 3P/4P and head-head/order canaries exist. | SOURCE | Add one 3P order stress only if it isolates a new score/death rule; otherwise keep current acceptance batches scoped to their named claims. | indirect | later |
| S11 | Wire event and replayed socket payload spec | open | Network source map is source-read; common trace is not a wire replay. | SOURCE/WIRE | Add `wire_event_single_tick` with compressed position/angle and one death+score payload. | no | later |
| V1 | Vector support for fourth trail-gap source case | done-for-now | Source trail-gap fixtures are included in the narrow mixed vector claim; exact command output belongs beside the comparator command. | VECTOR | Keep `source_trail_gap_batch.json` in the mixed vector run and rerun after trail-gap, event-row, or seeding edits. | indirect | no |
| V2 | Vector acceptance commands aligned with claimed fixtures | done-for-now | Focused vector commands are regression guardrails, not a fidelity headline. | VECTOR/TEST | Keep unsupported cases reported separately; update exact command output only when the claimed fixture set changes. | indirect | no |
| V3 | Natural trail-gap scalar vector arrays | done-for-now | Natural source fixture S2 now has scalar full-trace vector parity with row-local random tape arrays. Speed defaults still own only the forced trail-gap cases. | VECTOR | Keep the natural vector comparator separate; promote to B>1 speed only after batch-row multi-step support is explicit. | indirect | no |
| V4 | Broader wall and borderless vector support | partial | Vector covers normal-wall death, simple borderless wrap, same-frame wall draw replay packing, borderless destination-body skip then next-frame kill, borderless PrintManager wrap toggle, and collision-order batch support as source-pinned bridge tests. Planck's long-wall terminal-step public-vector bridge test is source-pinned from the penultimate frame only. | VECTOR | Keep these bridge cases as regression guards; add exact-edge/corner and broader lifecycle cases only from promoted source fixtures. Do not widen this to bonuses/3P/4P/full CurvyTron. | indirect | yes |
| V5 | Event row schema beyond current narrow types | partial | Fixed event rows cover position, point, die, score:round, score, round:end, and property in supported fixtures. | VECTOR | Add `angle`, `clear`, `borderless`, bonus events, and overflow policy only when source fixtures require them. | indirect | later |
| V6 | Row-local RNG arrays | partial | Current vector rows have row-local random tape values, length, cursor, exhaustion, and draw-count metadata; selected-row reset snapshots terminal cursor/count first, then copies template tape metadata for reset rows while skipped rows stay untouched. The current lifecycle source slice proves source-chronological RNG labels for the promoted 2P core fixtures, focused 3P fixtures, and focused 2P max-score match-end fixture. The vector seeder rejects natural lifecycle fixtures honestly and records RNG metadata: call index, site, avatar, value, at-ms, expected call count, and capacity pressure. Optimized/vector lifecycle RNG parity is not done. | VECTOR | Move the row-local RNG contract toward production reset: add seed/state metadata, random-call logs for spawn/PrintManager/bonuses, and compare vector rows against all promoted lifecycle source fixtures only after real reset/spawn arrays exist. | yes | yes |
| V7 | Event overflow and truncation policy | done-for-now | Public-step horizon truncation and overflow truncation have landed in the strict 1v1/no-bonus env. Overflow rows map to `truncated=true` with explicit terminal reasons, terminal metadata, final arrays, and replay-visible handoff for the narrow public slice. | VECTOR/OBS | Keep this scoped to strict 1v1/no-bonus; broaden capacity policies only when promoted source/vector rows need them. | indirect | no |
| V8 | Vector runtime API extraction | partial | `vector_runtime.step_many` is the supported fixture-backed source-ordered CPU transition kernel. Benchmark normal calls route through it; private diagnostic timer paths stay benchmark-only after safe timer cleanup, and the dead duplicate benchmark body was removed. Runtime terminal rows now mark optional `done`, `terminated`, `reset_pending`, `terminal_reason`, `draw`, and `winner` arrays. This proves the runtime boundary for promoted fixture rows only; it is not trainer integration, broad lifecycle, or production speed evidence. | VECTOR | Keep adding fast runtime behavior one promoted source contract at a time, using `CurvyTronSourceEnv` as the temporary witness until the runtime matches the source contract. | yes | yes |
| B1 | B>1 trail-gap default | done-for-now | Current B>1 defaults include the four forced trail-gap fixtures; exact outputs belong with the speed acceptance commands. | BATCH/VECTOR | Rerun `uv run pytest tests/test_benchmark_vector_batch_rows.py -q` and the smoke matrix after vector fixture-set changes. | indirect | no |
| B2 | B>1 delayed-start timer rows | partial | Batch rows now have a focused B>1 delayed-start pre-step timer helper with tests; speed defaults still exclude delayed-start. | BATCH/RESET | Keep the focused delayed-start helper tied to the delayed-start timer contract; promote the behavior only after production reset/timer arrays exist and the speed default fixture set is deliberately changed. | yes | yes |
| B3 | Mixed player-count batching policy | open | Current benchmarks group P=1, P=2, P=3 separately. | BATCH | Decide padded mixed-P batch versus separate fixed-P workers; write shape and mask tests before speed claims. | yes | yes |
| B4 | Debug-event/no-event hot path policy | partial | Benchmarks can compare debug-event and no-event modes; training policy is not chosen. | BATCH/REPLAY | Decide whether training hot path carries no events, sampled events, or debug events by ref; update replay/event refs accordingly. | yes | yes |
| B5 | Batch overflow and capacity limits | done-for-now | Horizon truncation and overflow truncation are implemented in the strict public 1v1/no-bonus env. Body and event capacity breaches enter done/truncated/replay through the narrow terminal metadata handoff. | BATCH/OBS | Keep this scoped to strict 1v1/no-bonus; add broader capacity cases with promoted source/vector rows before timing. | indirect | no |
| B6 | Stable benchmark thresholds and buckets | partial | Local scout numbers exist, but they are not stable gates. | BENCH | Record non-gating baseline commands and require pass/fail correctness before timing; avoid hard speed thresholds until CI hardware is known. | no | yes |
| R1 | Reset initializes source-like row state and timers | partial | `src/curvyzero/env/vector_reset.py` now has production-facing `reset_arrays(target, reset_template, reset_mask, *, reset_seed, reset_source, snapshot_array_names=None)`. It validates required row arrays, snapshots selected terminal rows, copies selected reset-template rows, increments `episode_id`, stamps reset metadata, clears terminal/event/timer-fired fields where present, preserves skipped rows, and returns reset metadata plus terminal snapshot. It does not natural-spawn, schedule timers, autoreset, preserve final obs, write replay, or expose a trainer API. | RESET | Build natural reset/spawn templates, lifecycle timer scheduling, seed generation/history, autoreset, final-observation policy, and replay policy on top of the reset boundary. | yes | yes |
| R2 | Pre-step timer advancement and event rows | partial | Comparator delayed-start support exists, batch rows now have a focused B>1 delayed-start pre-step timer helper with tests, and scalar source-env `advance_timers` now drains every due timer up to the target time, including newly scheduled due timers, with a zero-delay loop guard. This is still not a production vector timer API. | RESET/VECTOR | Promote the helper into the production reset/timer path: accept `timer_advance_ms[B]`; skip inactive/done/overflow rows; fire due `timer_*[B,T]` slots in `timer_seq` order; emit point/property timer events before movement events; preserve the scalar drain semantics and loop guard. | yes | yes |
| R3 | Autoreset preserves terminal transition first | partial | `final_transition_mask(done, truncated=None)` identifies final rows, comparator-local reset helpers preserve terminal snapshots before row reset, and actor bridge has debug-only internal autoreset after replay staging. `src/curvyzero/env/vector_autoreset.py` has `plan_autoreset_rows(...)` and `apply_autoreset_rows(...)`: final observation/reward arrays are staged first, then selected rows are reset through `vector_reset.reset_arrays(...)`. The strict public `VectorTrainerEnv1v1NoBonus` uses the same staging idea before selected-row reset/spawn/warmup, exposes reset/step control-wrapper info, and the live-step replay recorder consumes returned final arrays with terminal metadata, seed/reset metadata across autoreset, and the terminal barrier replay policy. The source-fixture reset path now proves the long 1v1 reset-to-terminal rollout. | RESET/REPLAY | Add row-local full RNG state/history/ref while keeping terminal outputs unchanged and proving next state has a new `episode_id`; broaden beyond the source-fixture reset hook only with source-backed reset claims. | yes | yes |
| R4 | Horizon and overflow truncation | done-for-now | The strict public `VectorTrainerEnv1v1NoBonus` step path now increments `episode_step` for active rows and marks rows at `max_ticks` as `done=true`, `terminated=false`, `truncated=true`, `terminal_reason=timeout_truncated`, with zero reward and final-array/autoreset staging. Overflow truncation uses the same narrow final-array/replay-visible terminal metadata policy. | RESET/OBS | Keep this scoped to strict 1v1/no-bonus; add broader lifecycle truncation separately. | indirect | no |
| R5 | Row-local seed history | partial | Replay metadata asks for seed; vector RNG history remains thin. The strict public env and live-step recorder now preserve `episode_id`, reset seed, and reset source across autoreset into recorder chunks. Full RNG state/history/ref is not captured yet. | RESET/REPLAY | Store env row id and RNG state/ref when available, and reject missing RNG history once that history exists. | yes | yes |
| R6 | Warmup/warmdown lifecycle in reset/autoreset | partial | Warmup, next-round, heading retry, focused 3P spawn-order, focused 3P warmup/PrintManager start, focused 4P first-round spawn order, focused 4P all-present all-dead warmdown/next-round, focused 4P survivor warmdown/next-round, focused 3P present/absent first-round behavior, one focused 3P present/absent survivor-scoring path, one focused 3P present/absent warmdown/next-round path, one focused 2P max-score match-end path, one focused all-present 3P `max_score: 2` match-end path, one focused all-present 3P multi-round match-end path, one focused 3P tie-at-max continuation, one focused 3P all-dead next-round path, one focused 3P survivor scoring through `round:end`, and one focused 3P survivor warmdown/next-round path have Python direct parity for the 28 promoted lifecycle fixtures. `run_warmup_start_step_1v1_no_bonus_rows(...)` now composes strict 1v1/no-bonus reset/spawn/warmup timer advancement and runtime stepping; a focused test proves wall-death terminal state plus vector trainer final-observation/reward handoff into autoreset planning. The strict public 1v1 vector env composes this slice and now covers horizon/overflow truncation, terminal metadata, control-wrapper info, and reset-to-terminal parity for the long source fixture. Warmdown/next-round, seed generation/history, broad bonuses, and broad 3P/4P lifecycle are not ready. | RESET/SOURCE | Add wall-death round-end/warmdown next. Keep seed history, bonuses, and broad 3P/4P as separate claims. | yes | yes |
| O1 | Final trainer observation schema | partial | `curvyzero_egocentric_rays/v0` is pinned with hashes, ray angles, scalar order, and flat LightZero shape. The scalar helper still covers current toy/grid `EnvState`; the new vector helper `observe_vector_1v1_egocentric_rays_v0(...)` raycasts against vector body circles and builds pinned `float32[106]` observations plus `float32[B,2,106]` final observation arrays for the strict 1v1/no-bonus slice. The strict public 1v1 vector env returns those arrays directly, and the live-step recorder packs them into replay-v0 chunks. | OBS | Broaden source-state fixtures later for trail gap, borderless wrap, same-frame death, and normal-wall terminal; keep replay terminal refs separate. | yes | yes |
| O2 | Observation fixture manifests from trusted states | partial | `obs_empty_arena_geometry_v0` is analytic only. `obs_source_movement_empty_multistep_v0` references a promoted source movement fixture and distills trusted expected frames into empty-occupancy observation states; it is not browser pixel fidelity or full source observation fidelity. | OBS | Add the next manifest only when it is tied to a represented trusted state snapshot without stealing priority from lifecycle/spawn/RNG. Good later targets remain trail gap, borderless, same-frame death, and normal-wall terminal. | yes | yes |
| O3 | Perspective and identity leak checks | partial | Analytic empty-arena tests cover symmetric p0/p1 perspective and scoped no-absolute-position leak checks. The source movement canary adds p0/p1 non-wall perspective symmetry for trusted source movement frames while intentionally leaving wall geometry unchecked. | OBS | Add broader p0/p1 permutation checks so observations do not leak stable seat, color, or absolute shortcuts unless the schema says so. | yes | yes |
| O4 | Debug observation schema hash stability | partial | Debug packer metadata emits schema ids and hashes in actor bridge sample. | OBS/REPLAY | Promote a deterministic hash test for debug obs, action space, debug reward, and selected-fixture rules hash. | yes | yes |
| O5 | Sparse round-outcome reward | partial | Trainer observation helpers emit compact `curvyzero_sparse_round_outcome/v0` reward info. The vector handoff derives sparse final reward maps from `terminal_reason`/`winner`/`draw` for the strict 1v1/no-bonus slice, including zero reward for horizon and overflow truncation. The live-step recorder packs these rewards for replay-v0, while current vector debug reward remains separate. | REWARD | Harden broader refs and schema-hash compatibility checks for replay rows. | yes | yes |
| O6 | Standard terminal info fields | partial | Toy-v0 now returns `final_observation` for terminal rows and refuses hidden post-terminal stepping until reset; contract helper pins trainer reset/step info keys and LightZero `eval_episode_return`. The local no-train LightZero-shaped smoke tests terminal reason, winners, losers, timeout/truncation, schema ids/hashes, `final_reward_map`, `eval_episode_return`, final observation, no hidden autoreset, and optional local-to-`BaseEnvTimestep` conversion. Real DI-engine imports are absent locally, so real info retention remains unverified. | OBS/REWARD | Verify the same terminal fields through real LightZero/DI-engine `BaseEnv` glue only after the runtime is available; later wire replay rows. | yes | yes |
| O7 | Legal action mask contract | partial | Debug packer emits `[B,P,3]`; contract helper pins canonical mask order left/straight/right, and the first trainer observation helper returns env `bool[3]` plus LightZero `int8[3]`. Focused tests cover live rows, strict left/right masks, terminal/dead all-false rows, and order. The local no-train LightZero-shaped smoke tests live `[1,1,1]`, terminal/truncated `[0,0,0]` masks, and preserves the mask through the optional timestep conversion boundary. Batch wrapper mask tests are still missing, and real DI-engine wrapper checks are skipped until imports exist. | OBS | Add batch mask tests and real LightZero/DI-engine wrapper mask checks only after reset/lifecycle/RNG/replay blockers stay explicit. | yes | yes |
| RP1 | Replay chunk writer schema | partial | Local `src/curvyzero/training/debug_actor_loop_replay.py` writes/reads one `.npz` debug chunk. Separate `src/curvyzero/training/replay_chunk_v0.py` writes/reads the 1v1/no-bonus replay-v0 shape with episode/reset/final arrays. Actor bridge sample-only output can write both the debug chunk and a replay-v0 chunk for P=2. `src/curvyzero/training/vector_env_replay_recorder.py` now records live strict public env step batches into replay-v0 using returned trainer observations/rewards, terminal metadata, terminal barrier replay policy, replay metadata defaults from env info, control-wrapper info, and final arrays. Replay row metadata and an optional strict replay/profile manifest exist for this slice. Actor-bridge replay-v0 remains debug/sample-only. The current replay-v0 schema is 1v1; 3P/4P needs a generalized schema with opponent policy ids/actions and the full wrapper action sidecar. | REPLAY | Add row-local seed/RNG history, broader event/state refs, opponent policy/action metadata, manifest hardening, and compaction to the live-step path before broader production training claims. | yes | yes |
| RP2 | Replay reader compatibility checks | partial | Debug replay and replay-v0 readers validate shape, dtype, schema hashes, and expected compatibility metadata. Focused evidence lives in `tests/test_debug_actor_loop_replay.py`, `tests/test_replay_chunk_v0.py`, `tests/test_trainer_replay_v0_builder.py`, `tests/test_vector_env_replay_recorder.py`, and `tests/test_benchmark_vector_actor_loop_bridge.py`; actor bridge sample output round-trips through both readers, and the live-step recorder builds chunks from returned vector env batches. | REPLAY | Add compatibility fixtures for terminal refs, seed/RNG history, and final trainer observation/reward schema changes as those contracts harden. | yes | yes |
| RP3 | Terminal and next observation policy | partial | Actor bridge has debug-only internal autoreset after replay staging. Replay-v0 sample output still carries debug actor-bridge arrays, but the strict vector path now builds real trainer `final_observation` and `final_reward_map` arrays and can stage them before reset mutation. The strict public 1v1 vector env returns pre-reset terminal final arrays while optionally returning post-reset observations for autoreset rows, and the live-step recorder uses those final arrays with terminal metadata and a terminal barrier replay policy. See `docs/working/environment/replay_terminal_seed_contract_2026-05-09.md`. | REPLAY/RESET | Add broader source refs and seed/RNG history beyond the current strict recorder path. | yes | yes |
| RP4 | Event/state/trace refs instead of hot JSON rows | partial | Contracts list refs; batch rows still local/debug. | REPLAY | Store event ranges and optional refs in chunks; keep source/common trace outside policy observations. | yes | yes |
| RP5 | Replay artifact layout and compaction | open | Modal/vector docs propose layouts, but replay writer does not own one. | REPLAY/MODAL | Pick local and Modal artifact paths for replay shards, summaries, complete markers, and manifest finalization. | yes | yes |
| RP6 | Search metadata beside replay | open | Actor bridge has fake action weights/root values; no real policy metadata contract. | REPLAY/POLICY | Add optional action weights, root value, policy id, opponent policy ids, model checkpoint id, and priority fields without replacing core env fields. | yes | later |
| M1 | Policy ego-row mapping | done-for-now | Pure helper `policy_row_mapping.py` maps `obs[B,P,...]` plus live/legal masks to compact/padded policy rows with env row ids and player ids, then maps selected trainer action ids back to a wrapper action map (`joint_action[B,P]` sidecar) with dead/padded rows masked to no-op. This is the repo-native multiplayer self-play shape: compact live ego rows, policy/search, rehydrate the wrapper action map, then call trainer env step. Focused tests cover compact rows, padding, legal-mask validation, compact selected actions for padded mappings, all-no-op empty mappings, and actor-bridge mapping shape. | POLICY | Reuse the helper in the LightZero adapter; keep unsupported policy moves out of fidelity claims. | yes | no |
| M2 | Replace synthetic feedback in actor bridge | partial | Actor bridge step 0 uses fixture moves; later steps still use synthetic feedback, but selected synthetic trainer action ids now pass through the policy-row mapping helper and source-move adapter before re-entering the wrapper action map (`joint_action[B,P]` sidecar). | POLICY/BATCH | Replace the synthetic policy/search stand-in with the real policy/MCTS boundary; keep unsupported policy moves out of fidelity claims. | yes | yes |
| M3 | Dummy model API in real observation shape | open | Modal Mctx uses synthetic/debug observations and toy model weights. | POLICY/MCTS | Build deterministic dummy representation/prediction/dynamics functions against O1/O4 shapes before using a checkpoint. | yes | yes |
| M4 | Mctx boundary with real masks and root rows | partial | Modal L4 can run Mctx on debug/fixture-root rows; still boundary evidence. | POLICY/MCTS | Run Mctx with real legal masks, fixed root row ids, and schema metadata; report setup, H2D, compile, steady timing separately. | yes | yes |
| M5 | Real self-play loop, local CPU first | open | Current local actor loop is fixture-reset blocks, debug packer, synthetic search, in-memory staging. | POLICY/MCTS | Build one local CPU loop with reset/autoreset, obs/reward, policy rows, env step, and replay chunks before Modal/GPU claims. | yes | yes |
| M6 | Wrapper joint-action search policy decision | done-for-now | Decision note: v0 keeps independent ego policy rows through `policy_row_mapping.py` and defers full wrapper joint-action MCTS because `3^P` grows fast. LightZero/MuZero should start as one ego decision row at a time; the wrapper fills non-ego players with explicit versioned opponent policies, advances held source controls over the elapsed-ms server-frame window, and logs the full wrapper action map as replay sidecar. Actor bridge tests now pin the independent row-mapping contract while the search remains synthetic. | POLICY/MCTS | Revisit full wrapper joint search only after the LightZero adapter, generalized 3P/4P replay schema, and real MCTS boundary exist. | yes | later |
| G1 | Modal CPU fixture-array equivalence job | open | Modal CPU smoke runs source benchmark and synthetic vector profile, not fixture equivalence. | MODAL | Add a coarse Modal CPU function/wrapper for `compare_vector_arrays_to_fidelity.py` over scenario shards and artifact refs. | no | yes |
| G2 | Modal GPU Mctx with real observation packer | partial | Modal L4 Mctx runs exist for synthetic and debug fixture-root rows. | MODAL/GPU | After O1/O4, run a tiny L4 profile using real observation schema or stable debug schema, with exact artifact refs. | no | yes |
| G3 | Tensor env step smoke | open | No JAX/PyTorch source-like env transition exists. | GPU/VECTOR | Add CPU tensor smoke first; only request GPU when the same code keeps rollout work on device long enough to measure. | no | later |
| G4 | Modal artifact manifest and finalizer | partial | Proposed layout exists; smoke writes coarse artifacts. | MODAL | Add manifest validation and single-finalizer rule for environment vectorization/replay runs. | no | yes |
| G5 | No per-step Modal guardrail | partial | Docs say no per-step Modal calls; no test enforces it. | MODAL/TEST | Add code review checklist or static test that training/env hot-loop modules do not import Modal wrappers. | yes | yes |
| G6 | GPU cost and hardware labels | partial | Runbook records L4 hardware and cost notes. | MODAL/GPU | Require every Modal/GPU result to record app id, GPU type, package versions, setup/compile/H2D/steady timings, and billing warning. | no | yes |
| T1 | Source-claim acceptance commands | partial | Existing source batches remain regression guardrails; the 20 pinned lifecycle/spawn/RNG fixtures including `source_lifecycle_spawn_rng_4p_next_round`, `source_lifecycle_survivor_score_4p_next_round`, `source_lifecycle_present_absent_3p_survivor_score_round_end`, and `source_lifecycle_multi_round_match_end_3p` now have a focused parity command through `tests/test_source_lifecycle_runner.py`. | TEST | Keep source batches as separate commands; add new focused commands only when a new source claim lands. | indirect | yes |
| T2 | Focused vector comparator gate | done-for-now | Focused vector and mixed comparator commands are regression guardrails for claimed vector fixtures. | TEST/VECTOR | Run the vector comparator acceptance command after vector or fixture changes; record exact output beside the claimed fixture set, not as a dashboard. | indirect | no |
| T3 | Repository regression footer | done-for-now | Repository tests are hygiene after code changes, not source-fidelity claims. | TEST | Rerun repository tests after code changes only when needed, and keep exact output outside headline source-claim docs. | indirect | no |
| T4 | Static cleanup | done-for-now | Lint is static hygiene, not behavior evidence. | TEST | Rerun lint after code changes only when needed, and keep exact output outside headline source-claim docs. | no | no |
| T5 | Benchmark smoke tests stay correctness-first | partial | Tests cover batch rows, obs/reward packing indirectly, actor bridge sample metadata, policy-row mapping shape, sample-only debug replay chunk round-trip, and sample-only replay-v0 chunk round-trip. | BENCH | Keep unit tests small; separate timing scout commands from behavior correctness checks. | no | yes |
| T6 | Benchmarks with debug/no-event split | partial | Batch rows and actor bridge support both modes; quick smokes are non-gating. | BENCH | Rerun quick `B=32` batch rows and `B=16 rollout_steps=2` actor bridge after fixture-set or event-row changes; record numbers as non-gating acceptance output. | no | yes |
| T7 | Replay/observation/reward tests in CI | partial | Focused debug replay tests cover writer/reader metadata validation and absent-policy rejection. Replay-v0 tests cover 1v1/no-bonus arrays, episode/reset/final arrays, and compatibility rejection. Actor bridge sample-only replay output round-trips through both readers. Live-step recorder tests cover nonterminal chunks and terminal autoreset final arrays from returned vector env batches. Broader production replay metadata tests are still missing. | TEST | Run `uv run pytest tests/test_debug_actor_loop_replay.py tests/test_replay_chunk_v0.py tests/test_trainer_replay_v0_builder.py tests/test_vector_env_replay_recorder.py tests/test_benchmark_vector_actor_loop_bridge.py -q`; add targeted tests as O/RP specs land and keep them separate from slow Modal/GPU jobs. | yes | yes |
| T8 | Modal/GPU smoke not in default pytest | partial | Modal commands allocate remote resources. | TEST/MODAL | Keep Modal/GPU as explicit runbook commands with cost notes, not default local CI. | no | yes |

## Training Readiness Gates

Do not call source-faithful training ready until these are all true:

1. Promoted source claims have oracle/probe evidence and Python parity for the
   trail, trail-gap, PrintManager, body, wall/border, kinematics, and
   collision-order batches.
2. Vector comparator matches every fixture claimed by the vector path, with
   unsupported cases reported separately.
3. Reset, pre-step timers, terminal final observations, and autoreset are tested.
4. Observation, action mask, reward, done/truncated, and terminal info schemas
   have ids and hashes.
5. Production replay writer/reader is wired to actor bridge chunks and rejects
   mismatched rules/schema hashes for the final trainer observation and reward
   schemas.
6. Policy/search rows map cleanly to and from env rows, with opponent policy
   ids/actions retained when non-ego actions are filled by a wrapper policy.
7. LightZero adapter reset/step returns the expected observation dict, legal
   mask, scalar reward, done/truncated, and terminal info without importing
   source-fidelity or vector debug actor paths.
8. Modal/GPU jobs are coarse artifact jobs only, never per-step calls.
9. Repository tests and lint are clean as hygiene footnotes after the latest
   claimed fixture set changes.
