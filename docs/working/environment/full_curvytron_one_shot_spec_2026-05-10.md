# CurvyTron One-Shot Env Spec - 2026-05-10

Status: docs-only execution list for the remaining CurvyTron environment work.
Goal: harden one fast, source-faithful CurvyTron runtime:
`VectorMultiplayerEnv`. Strict 1v1 scalar/ray support is proof/profiling
infrastructure, and visual LightZero input is planned without ALE.

Optimizer visual tensor handoff:
[optimizer_visual_tensor_handoff_2026-05-10.md](optimizer_visual_tensor_handoff_2026-05-10.md).

## Boundary

CurvyTron is not an ALE environment. ALE means Arcade Learning Environment: the
Atari emulator/API used for real Atari ROM Pong. Use ALE only for official
Atari Pong control runs. CurvyTron must use repo-owned CurvyTron state,
rendering, wrappers, and replay.

The intended runtime under hardening is `VectorMultiplayerEnv`. The
current useful proof/profiling side path is scalar/ray single-ego:

```text
float32[106] observation
int8[3] LightZero action_mask
to_play = -1
actions: 0 left, 1 straight, 2 right
```

Visual CurvyTron remains required later. That means CurvyTron pixels from a
source-faithful renderer, shaped for LightZero stacked frames. It does not mean
Atari, ROMs, ALE, Pong rewards, or Atari action meanings.

Native CurvyTron semantics are held player control state advanced over
elapsed-ms frames. `step`, `decision_ms`, and `joint_action` are wrapper and
replay terms; `decision_ms` is a wrapper decision window, not a native tick.
Restricted wrappers are temporary proof/profile configs only. The
reconstruction path is source-default CurvyTron behavior in
`VectorMultiplayerEnv`.

## Optimizer Boundary

Environment owns visual truth:

- source-fidelity level: source-faithful renderer versus debug occupancy smoke;
- visual schema meaning, ids, hashes, shape, dtype, range, and channel order;
- player perspective: full arena, ego-centered, per-player row, or another
  named view;
- frame-stack ownership as a contract field;
- reset and final-observation policy;
- metadata fields needed for replay, audit, and promotion;
- source/browser comparison plan;
- promotion gates that say when a tensor is more than debug/profiling data.

Optimizer owns visual plumbing and speed choices after Environment names the
tensor contract:

- debug visual smoke and profiler surfaces;
- LightZero adapter plumbing and config/import smokes;
- batching, memory layout, CPU/GPU/Modal/JAX/PyTorch alternatives;
- render, stack, env step, policy/search, replay, and reset cost breakdowns;
- bottleneck reads and implementation recommendations.

Optimizer may help reconstruct faster visual tensor code, but it must not
redefine CurvyTron pixel truth. Environment should provide contracts and
truth-defining helpers only where needed; adapter implementation stays an
Optimizer/plumbing task once the contract is named.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Priority List

1. Harden `VectorMultiplayerEnv` as the intended runtime.
2. Preserve the strict scalar/ray 1v1 no-bonus proof as a proof/profiling
   boundary only.
3. Promote no-bonus 2P/3P/4P public lifecycle from metadata-only to
   source-backed natural reset, lifecycle, replay, observation, reward, and mask
   parity.
4. Keep row-local RNG provenance minimal: enough seed/source/cursor/ref data to
   debug and replay a row, not a heavyweight history system.
5. Add a generalized multiplayer replay path before any 3P/4P self-play claim.
6. Add learned 3P/4P observations or keep them explicitly debug-only.
7. Promote bonuses one source rule at a time, then broaden fast runtime beyond
   the current optional-array scouts.
8. Define the visual tensor truth contract and source/browser promotion gates.
9. Let Optimizer wire visual smoke/profilers and LightZero adapter plumbing only
   against named CurvyTron contracts.
10. Use proof artifacts and focused acceptance commands, not pass-count
   dashboards or rows/sec alone, as the claim gate.

## What Exists

Core source proof:

- Source mechanics are pinned for movement, normal wall, borderless, body
  collisions, collision order, PrintManager, normal trail cadence, forced trail
  gaps, one natural trail-gap case, old-body metadata, focused lifecycle, and
  narrow bonus facts.
- Main docs: `coverage_tracker.md`, `source_feature_inventory.md`,
  `full_fidelity_spec_matrix_2026-05-09.md`, and `reorientation_packet.md`.
- Source fixtures live under `scenarios/environment/`.
- JS/source proof tools live under `tools/reference_oracle/` and
  `tools/run_fidelity_batch.py`.

Strict scalar/ray 1v1 no-bonus proof/profiling path:

- `src/curvyzero/env/vector_runtime.py::step_many` is the supported
  fixture-backed, source-ordered CPU transition kernel.
- `src/curvyzero/env/vector_trainer_env.py::VectorTrainerEnv1v1NoBonus`
  exposes the narrow public B-row 1v1 no-bonus trainer env.
- `src/curvyzero/env/vector_trainer_observation.py` builds
  `curvyzero_egocentric_rays/v0` `float32[106]` observations, masks, final
  observations, and sparse final rewards.
- `src/curvyzero/training/vector_env_replay_recorder.py` records live strict
  1v1 no-bonus env batches into replay-v0 chunks.
- Tests: `tests/test_vector_trainer_env.py`,
  `tests/test_vector_trainer_observation.py`,
  `tests/test_vector_env_replay_recorder.py`, `tests/test_vector_autoreset.py`,
  and `tests/test_replay_chunk_v0.py`.

Intended multiplayer no-bonus runtime under hardening:

- `src/curvyzero/env/vector_multiplayer_env.py::VectorMultiplayerEnv`
  is the intended public 2P/3P/4P runtime under hardening. Today it is
  still a metadata-only surface.
- It has debug metadata observations under
  `curvyzero_debug_metadata_only/v0`; it does not claim learned observations.
- Public seeded 3P/4P wall canaries, 4P fixture-tape reset/spawn,
  3P present/absent reset, present/absent scoring, warmdown continuation, and
  focused 3P match/tie/multi-round metadata checks exist.
- Tests: `tests/test_vector_multiplayer_env.py`,
  `tests/test_vector_lifecycle.py`, `tests/test_vector_runtime.py`, and
  `tests/test_multiplayer_replay_contract.py`.

Lifecycle:

- Pinned lifecycle fixtures cover focused 2P/3P/4P spawn, warmup,
  next-round, survivor scoring, present/absent, match-end, tie-at-max, and one
  multi-round path.
- Tests: `tests/test_lifecycle_oracle.py`,
  `tests/test_source_lifecycle_runner.py`, `tests/test_source_env.py`,
  `tests/test_vector_lifecycle.py`, and `tests/test_vector_multiplayer_env.py`.

Bonuses:

- Narrow source proofs exist for `BonusSelfSmall` catch/no-catch/death-order,
  one-type spawn RNG, default multi-type type RNG, spawn retry against game
  world, one expiry/restore path, and forced `BonusGameClear`.
- Fast-runtime optional-array bonus slices now cover forced `BonusSelfSmall`
  catch/no-catch/death-order, the 7500 ms expiry/restore path from
  `source_bonus_self_small_expiry_restore_step.json`, and forced
  `BonusGameClear` immediate clear. Keep this optional-array only, with no
  natural spawn, public bonus environment, replay, or broad bonus-system claim.
- Tests: `tests/test_source_env.py` and `tests/test_env_scenarios.py`.
- Source files: `third_party/curvytron-reference/src/server/manager/BonusManager.js`,
  `third_party/curvytron-reference/src/server/model/BonusStack.js`, and
  `third_party/curvytron-reference/src/server/model/Bonus/*.js`.

LightZero scalar/ray bridge:

- `src/curvyzero/training/curvyzero_lightzero_smoke.py` provides a local
  fallback single-ego LightZero-shaped smoke around the toy `CurvyTronEnv`
  without requiring DI-engine/LightZero imports.
- `src/curvyzero/training/curvyzero_lightzero_env.py` registers
  `curvyzero_v0_lightzero`, reuses the local smoke semantics, and returns real
  `BaseEnvTimestep` rows only when DI-engine is installed.
- `src/curvyzero/training/curvyzero_lightzero_runtime_probe.py` is the
  no-train config/import/env-factory/reset/step probe for installed runtimes,
  including Modal images that provide DI-engine/LightZero.
- Keep local fallback, Modal installed runtime, and real required-mode
  DI-engine/LightZero runs labeled separately. These are adapter smokes, not
  real training over the source-faithful env.
- Tests: `tests/test_curvyzero_lightzero_smoke.py`,
  `tests/test_curvyzero_lightzero_env.py`, and
  `tests/test_curvyzero_lightzero_runtime_probe.py`.

Visual smoke only:

- `src/curvyzero/training/curvytron_visual_observation.py` contains
  `curvyzero_debug_occupancy_gray64/v0`: a debug occupancy renderer and
  four-frame stack helper.
- It is `uint8`, marks body centers/avatar positions, and explicitly does not
  claim source visual fidelity.
- Treat this as an Optimizer/profiler plumbing surface. Environment owns the
  debug-only label, schema caveat, and promotion gate.
- Tests and timing: `tests/test_curvytron_visual_observation.py` and
  `scripts/benchmark_curvytron_visual_observation.py`.

## Missing Spec And Gap List

### 1. Source Fidelity Promotion

Missing:

- Longer JS/Python/source-env comparisons for lifecycle and bonus paths.
- Broader 4P match lifecycle and broader present/non-present variants only when
  a new source rule is isolated.
- Broader emitted-trail collision paths after the current point/body canaries.
- A single source claim id per promoted fast-runtime behavior.

Likely files:

- `src/curvyzero/env/source_env.py`
- `src/curvyzero/env/trace_compare.py`
- `tools/reference_oracle/scenario_runner.js`
- `tools/reference_oracle/lifecycle_oracle.js`
- `scripts/run_environment_fidelity_matrix.py`
- `scenarios/environment/source_*.json`

Likely tests:

- `tests/test_source_env.py`
- `tests/test_env_scenarios.py`
- `tests/test_source_lifecycle_runner.py`
- `tests/test_lifecycle_oracle.py`
- `tests/test_run_environment_fidelity_matrix.py`

Acceptance shape:

- Add one fixture per source rule.
- Compare JS and Python common traces.
- Record first mismatch and source claim id.
- Promote to fast runtime only after the source proof is named.

### 2. Fast Scalar/Ray 1v1 No-Bonus

Missing:

- Row-local full RNG state/history/ref.
- Replay compaction and broader source refs.
- Harder reset/autoreset stress with many B rows.
- Whole actor-loop timing with env, observation, policy/search, replay, and
  autoreset separated.

Likely files:

- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_reset.py`
- `src/curvyzero/env/vector_autoreset.py`
- `src/curvyzero/env/vector_trainer_env.py`
- `src/curvyzero/env/vector_trainer_observation.py`
- `src/curvyzero/training/vector_env_replay_recorder.py`
- `src/curvyzero/training/replay_chunk_v0.py`
- `scripts/benchmark_vector_trainer_actor_loop_profile.py`
- `scripts/benchmark_vector_actor_loop_bridge.py`

Likely tests:

- `tests/test_vector_runtime.py`
- `tests/test_vector_reset.py`
- `tests/test_vector_autoreset.py`
- `tests/test_vector_trainer_env.py`
- `tests/test_vector_trainer_observation.py`
- `tests/test_vector_env_replay_recorder.py`
- `tests/test_benchmark_vector_trainer_actor_loop_profile.py`

Acceptance shape:

- Strict 1v1 no-bonus reset-to-terminal still matches the long source fixture.
- Final observation/reward are captured before reset.
- RNG/ref fields survive autoreset.
- Timing reports name `env_impl_id`, ruleset, source claim, feature flags,
  `decision_ms`, capacities, reset source, and included timed components.

### 3. Multiplayer 2P/3P/4P No-Bonus

Missing:

- Natural public reset parity from seed/history, not only fixture tape.
- Public warmdown, next-round, match-end, and match-mode episode policy.
- General reset/autoreset/final-observation policy for 3P/4P.
- Per-player reward vector tests for live, dead, absent, survivor win, draw,
  timeout, truncation, and match end.
- Legal mask tests for live, dead, absent, terminal, and reset-pending players.
- A learned 3P/4P observation schema, or a deliberate continued
  metadata-only stance.

Likely files:

- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_lifecycle.py`
- `src/curvyzero/env/vector_spawn.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_reset.py`
- `src/curvyzero/training/multiplayer_replay_contract.py`
- `src/curvyzero/training/policy_row_mapping.py`

Likely tests:

- `tests/test_vector_multiplayer_env.py`
- `tests/test_vector_lifecycle.py`
- `tests/test_vector_spawn.py`
- `tests/test_vector_runtime.py`
- `tests/test_multiplayer_replay_contract.py`
- `tests/test_policy_row_mapping.py`

Acceptance shape:

- Every public 3P/4P claim says whether it is seeded fixture state,
  fixture-tape reset, natural reset, warmdown bridge, replay, or learned
  observation.
- `VectorMultiplayerEnv` remains metadata-only until a real learned
  observation schema is added.
- Public env info carries player count, present/alive masks, score vectors,
  death order, round/match terminal flags, reset seed/source, RNG cursor/ref,
  action sidecar, and final observation policy.

### 4. Replay, RNG Provenance, And Reproducibility

Missing:

- Minimal row-local RNG provenance for spawn, PrintManager, and bonuses: reset
  seed/source, cursor/draw counts, and a future ref/hash when raw history is
  needed for debugging.
- 3P/4P replay writer and reader.
- Opponent policy sidecar with policy id/version/seed and supplied actions.
- Replay compaction plus source/evidence refs.
- Replay manifest hardening for strict schema hashes and rules hashes.

Likely files:

- `src/curvyzero/training/replay_chunk_v0.py`
- `src/curvyzero/training/trainer_replay_v0_builder.py`
- `src/curvyzero/training/vector_env_replay_recorder.py`
- `src/curvyzero/training/multiplayer_replay_contract.py`
- `src/curvyzero/training/policy_row_mapping.py`
- `src/curvyzero/env/vector_reset.py`
- `src/curvyzero/env/vector_multiplayer_env.py`

Likely tests:

- `tests/test_replay_chunk_v0.py`
- `tests/test_trainer_replay_v0_builder.py`
- `tests/test_vector_env_replay_recorder.py`
- `tests/test_multiplayer_replay_contract.py`
- `tests/test_policy_row_mapping.py`
- `tests/test_vector_multiplayer_env.py`

Acceptance shape:

- Replay can rebuild terminal transitions without guessing.
- Reset seed alone is not treated as a full reproducibility record.
- RNG provenance is debug/replay evidence. Training still uses source-like
  randomness, and later may add extra controlled variation.
- Terminal barrier policy is explicit.
- 1v1 replay-v0 and future 3P/4P replay ids stay separate.

Future extra training noise belongs behind explicit config knobs and must be
separate from source-fidelity proof runs. Useful future knobs include reset seed
variation, spawn jitter, arena size jitter, speed/turn/radius jitter, decision
cadence jitter, action-repeat jitter, bonus probability schedules, opponent
policy mixtures, visual noise, scale/color jitter, and curriculum schedules.

### 5. Bonuses

Missing:

- Cap behavior.
- Bonus-world retry.
- Natural `BonusGameClear` selection/probability.
- Speed, slow, inverse, straight-angle, borderless, color, invincible, printing,
  and radius effects beyond the current narrow proofs.
- Stack math with multiple active bonuses.
- Same-frame expiry ordering with other timers.
- Death interactions for non-`BonusSelfSmall` types.
- Broad/natural/public fast-runtime and public-env support beyond the current
  optional-array scouts.

Likely files:

- `src/curvyzero/env/source_env.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_lifecycle.py`
- `src/curvyzero/env/vector_reset.py`
- `third_party/curvytron-reference/src/server/manager/BonusManager.js`
- `third_party/curvytron-reference/src/server/model/BonusStack.js`
- `third_party/curvytron-reference/src/server/model/GameBonusStack.js`
- `third_party/curvytron-reference/src/server/model/Bonus/*.js`
- `scenarios/environment/source_bonus_*.json`

Likely tests:

- `tests/test_source_env.py`
- `tests/test_env_scenarios.py`
- Future: `tests/test_vector_runtime.py` bonus slices.

Acceptance shape:

- One source rule per fixture.
- Forced catch/effect fixtures before natural random spawn variants.
- Vector support only after source fixtures prove exact state/event order.
- Current priority: do not pull bonuses into the public multiplayer env before
  replay/autoreset/final-observation behavior is stronger. The fast-runtime
  optional-array scout now covers forced active `BonusSelfSmall` catch,
  expiry/restore, and forced `BonusGameClear`; the remaining bonus work is
  natural spawn/caps, public-env bonus contracts, replay claims, and the other
  source bonus effects.

### 6. Visual Tensor Truth Contract

Missing:

- A named truth contract for source-faithful pixels versus debug occupancy.
- A source-faithful renderer definition from fast CurvyTron state arrays.
- A visual schema id for real learned input, separate from
  `curvyzero_debug_occupancy_gray64/v0`.
- `float32[1,64,64]` normalized grayscale frame contract.
- LightZero stacked input contract: default `float32[4,64,64]`.
- Frame provenance: renderer id/hash, state tick, elapsed ms, player count,
  ego id, crop/scale policy, color policy, and terminal-frame policy.
- Final visual observation before reset.
- Pixel/array tests for stable shape, value range, nonblank frames, deterministic
  output, terminal frame, and replay sidecar reconstruction.
- A clear decision on full-arena view versus ego-centered crop. First default:
  full arena, grayscale, LightZero-owned stacking.

Environment-owned contract and proof files:

- Future: `src/curvyzero/env/vector_visual_observation.py` or equivalent
  truth-defining renderer helper.
- `docs/working/environment/optimizer_visual_tensor_handoff_2026-05-10.md`
- `docs/working/environment/coverage_tracker.md`
- Source/browser comparison fixtures or artifacts when created.

Optimizer-owned plumbing/profiler files once the contract exists:

- `src/curvyzero/training/curvytron_visual_observation.py`
- `scripts/benchmark_curvytron_visual_observation.py`
- `src/curvyzero/training/curvyzero_lightzero_env.py`
- `src/curvyzero/training/curvyzero_lightzero_smoke.py`
- LightZero/Modal files that wire config/import/profiling.

Likely tests:

- Environment: future `tests/test_vector_visual_observation.py`, plus
  source/browser comparison tests or artifact validators when defined.
- Optimizer/plumbing: `tests/test_curvytron_visual_observation.py`,
  `tests/test_curvyzero_lightzero_env.py`, and
  `tests/test_curvyzero_lightzero_runtime_probe.py`.

Acceptance shape:

- Debug occupancy remains labeled debug-only.
- Real visual renderer truth consumes source-faithful state arrays, not ALE.
- Source/browser comparison plan says what "pixel faithful" means before
  promotion.
- Contract says whether frame stacking is owned by LightZero, the env wrapper,
  or a separate pre-stacked schema.
- Visual replay records enough provenance to regenerate frames from state.

### 7. LightZero Adapter Plumbing

Missing:

- Adapter plumbing over the fast source-faithful env, not the toy
  `CurvyTronEnv`, after Environment names the contract.
- Single-ego opponent policy contract with policy id/version/seed.
- Visual adapter config for conv-style MuZero once the visual tensor contract
  exists.
- Correct `done = terminated or truncated` with separate flags in info.
- Clear no-hidden-autoreset policy for LightZero env manager use.
- Required-mode installed-runtime coverage must stay explicit when configs
  change; local fallback smoke alone is not enough.

Environment-owned contract/helper files:

- `src/curvyzero/training/policy_row_mapping.py` when it defines row truth,
  opponent-policy metadata, or replay sidecar meaning.
- Contract docs that define observation/action/reward/reset/final-observation
  semantics.

Optimizer-owned adapter/plumbing files:

- `src/curvyzero/training/curvyzero_lightzero_env.py`
- `src/curvyzero/training/curvyzero_lightzero_smoke.py`
- `src/curvyzero/training/curvyzero_lightzero_runtime_probe.py`
- Future visual config files under the training or infra path that already owns
  LightZero runs.

Likely tests:

- Optimizer/plumbing: `tests/test_curvyzero_lightzero_smoke.py`,
  `tests/test_curvyzero_lightzero_env.py`, and
  `tests/test_curvyzero_lightzero_runtime_probe.py`.
- Environment: contract tests for policy-row metadata, replay sidecars, visual
  schema ids, and terminal/final-observation meaning.
- Installed-runtime smoke in a Modal/LightZero image when configs change.

Acceptance shape:

- The wrapper returns `observation`, `action_mask`, `to_play=-1`, optional
  `timestep`, scalar reward, done, and audit info.
- Non-ego actions come from named opponent policies.
- Full wrapper action maps are replay sidecars.
- No full wrapper joint-action MCTS in v0; branching is `3^P`.
- No ALE dependency or Atari naming in CurvyTron configs.

### 8. Testing And Fidelity Proof

Missing:

- A single "claim promotion" checklist used by source, fast runtime, replay,
  visual, and LightZero work.
- Visual proof artifacts with state refs and frame hashes.
- Long multiplayer rollout comparison beyond current focused fixtures.
- Whole-loop performance reports tied to named source claims.

Likely files:

- `scripts/run_environment_fidelity_matrix.py`
- `scripts/compare_vector_arrays_to_fidelity.py`
- `scripts/check_environment_doc_status.py`
- `scripts/benchmark_vector_batch_rows.py`
- `scripts/benchmark_vector_trainer_actor_loop_profile.py`
- `scripts/benchmark_curvytron_visual_observation.py`
- `docs/working/environment/coverage_tracker.md`
- `docs/working/environment/active_lanes.md`

Likely tests:

- `tests/test_compare_vector_arrays_to_fidelity.py`
- `tests/test_run_environment_fidelity_matrix.py`
- `tests/test_check_environment_doc_status.py`
- The focused tests named in each chunk above.

Acceptance shape:

- Every implementation PR names the source claim, public contract id,
  observation/reward/action schema ids, and unsupported boundaries.
- Every visual claim includes frame schema id, renderer id/hash, state source,
  shape, dtype, range, frame stack policy, and terminal-frame policy.
- Every timing claim names the timed surface and whether reset, replay,
  observation, policy/search, and model calls are included.

## Parallel Implementation Chunks

Chunk A: Source claims.

- Owner scope: fixtures, JS oracle, source env parity, docs.
- Files: `scenarios/environment/source_*.json`, `tools/reference_oracle/*`,
  `src/curvyzero/env/source_env.py`, `tests/test_source_env.py`.
- Output: one named source rule per fixture and a common-trace comparison.

Chunk B: Fast scalar/ray proof wrapper.

- Owner scope: strict 1v1 no-bonus proof wrapper, final observations, RNG refs,
  replay-v0 hardening, timing.
- Files: `vector_runtime.py`, `vector_reset.py`, `vector_autoreset.py`,
  `vector_trainer_env.py`, `vector_trainer_observation.py`,
  `vector_env_replay_recorder.py`.
- Output: focused strict 1v1 source bridge plus stronger reproducibility
  metadata.

Chunk C: Public multiplayer no-bonus.

- Owner scope: `VectorMultiplayerEnv` lifecycle metadata, masks, rewards,
  public warmdown/match policy, future learned observation boundary.
- Files: `vector_multiplayer_env.py`, `vector_lifecycle.py`, `vector_spawn.py`,
  `multiplayer_replay_contract.py`, `policy_row_mapping.py`.
- Output: 2P/3P/4P public claims that say exactly whether they are
  metadata-only, replay-capable, or learned-observation capable.

Chunk D: Multiplayer replay and ego-row policy.

- Owner scope: generalized 3P/4P replay schema, opponent policy sidecars,
  row mapping, terminal barrier policy.
- Files: `multiplayer_replay_contract.py`, `policy_row_mapping.py`,
  `trainer_replay_v0_builder.py`, `replay_chunk_v0.py`.
- Output: replay rows that can audit single-ego decisions with full wrapper
  action maps.

Chunk E: Bonuses.

- Owner scope: source bonus fixtures first, then fast runtime/public-env support
  beyond the optional-array scouts.
- Files: `source_env.py`, `vector_runtime.py`, bonus fixtures, bonus tests.
- Output: one promoted bonus rule at a time. Optional-array scouts remain
  narrow; no broad bonus claim.

Chunk F: Visual tensor truth contract.

- Owner scope: Environment.
- Work: define source-faithful versus debug truth level, schema id/hash, shape,
  dtype, range, perspective, frame-stack owner, final observation policy,
  metadata, and source/browser comparison gate.
- Files: future truth-defining renderer helper if needed, contract docs,
  `coverage_tracker.md`, and source/browser proof artifacts.
- Output: a named visual tensor contract Optimizer can implement and profile.

Chunk G: Visual smoke/profiler and LightZero adapter plumbing.

- Owner scope: Optimizer.
- Work: debug visual smoke, profiler, batching, Modal/CPU/GPU choices,
  LightZero config/import smokes, and adapter plumbing after contracts exist.
- Files: `curvyzero_lightzero_env.py`, `curvyzero_lightzero_smoke.py`,
  `curvyzero_lightzero_runtime_probe.py`,
  `scripts/benchmark_curvytron_visual_observation.py`, and relevant infra files.
- Output: LightZero-compatible CurvyTron rows and timing reports that name
  truth level and include no ALE dependency.

Chunk H: Proof and performance.

- Owner scope: split. Environment owns promotion gates and claim guardrails;
  Optimizer owns timing and bottleneck reads.
- Files: matrix runner, vector comparator, benchmarks, coverage docs.
- Output: reports that name the exact surface timed or proven.

## Do Not Do

- Do not route or label CurvyTron as ALE.
- Do not reuse Atari action meanings or Pong rewards.
- Do not promote debug occupancy frames as source visual fidelity.
- Do not reuse the strict 1v1 ray observation schema for 3P/4P.
- Do not claim natural public multiplayer reset from fixture-tape reset.
- Do not hide autoreset behind a terminal step.
- Do not start with full wrapper joint-action MCTS.
- Do not treat seed alone as replay-grade RNG state.
- Do not use rows/sec as a fidelity claim.

## Immediate One-Shot Order

1. Add row-local RNG history/ref contract and carry it through strict 1v1
   reset/autoreset/replay.
2. Turn `VectorMultiplayerEnv` natural reset, lifecycle, replay, and
   observation parity into source-backed public tests for the pinned lifecycle
   fixtures.
3. Add 3P/4P replay writer skeleton guarded by
   `multiplayer_replay_contract.py`; keep observations debug-only until a real
   schema exists.
4. Promote one bonus runtime rule only after adding its source fixture.
5. Environment defines the visual tensor contract: truth level, schema, shape,
   dtype, range, perspective, stack owner, final observation policy, metadata,
   source/browser comparison, and promotion gate.
6. Optimizer wires visual smoke/profilers and LightZero adapter config against
   that named contract, with LightZero-owned frame stacking to `(4,64,64)` only
   when the contract says so.
7. Run focused proof commands per changed lane and update `coverage_tracker.md`
   with the claim, caveat, and next hole.
