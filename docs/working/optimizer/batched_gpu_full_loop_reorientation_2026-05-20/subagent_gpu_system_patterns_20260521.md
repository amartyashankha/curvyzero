# Subagent GPU System Patterns

Date: 2026-05-21

Scope: local docs/code inspection only. No browsing, no code edits, no live
training runs, no trainer/tournament defaults touched.

## Web/Literature Follow-Up

This section adds a small official-source web pass from the optimizer main
thread. It does not change the local conclusions.

Sources checked:

- Sample Factory architecture docs:
  <https://www.samplefactory.dev/06-architecture/overview/>
- Sample Factory batched sampling docs:
  <https://www.samplefactory.dev/07-advanced-topics/batched-non-batched/>
- CuLE NeurIPS abstract:
  <https://papers.nips.cc/paper/2020/hash/e4d78a6b4d93e1d79241f7b282fa3413-Abstract.html>
- EnvPool arXiv abstract:
  <https://arxiv.org/abs/2206.10558>
- Brax arXiv abstract:
  <https://arxiv.org/abs/2106.13281>
- NVLabs CuLE repo notes:
  <https://github.com/NVlabs/cule>

Plain read from those sources:

- Sample Factory explicitly separates rollout workers, inference workers,
  batcher, and learner, and uses shared-memory buffers to avoid serializing
  observations between components. That supports our current suspicion that
  scalar Python/NumPy timestep churn is a real wall.
- Sample Factory's batched mode assumes observations are already one large
  tensor, ideally already on GPU for GPU envs. That matches the device-stack
  direction and argues against a host-only ring buffer as the final answer.
- CuLE's headline trick is not merely "render on GPU." It runs thousands of
  Atari games in parallel and renders frames directly on GPU to avoid CPU/GPU
  bandwidth bottlenecks. The CuLE repo example uses `--num-ales 1200`, which is
  the same broad shape as our "large batch alive across the hot loop" target.
- EnvPool attacks environment execution as its own engine, with high
  compatibility to existing RL libraries. That is the CPU-side analogue of our
  possible compact actor/fan-in lane.
- Brax keeps the environment and learning code together in JAX on accelerators.
  That is the clean device-resident endpoint, but it is a bigger rewrite than
  the current stock LightZero-compatible profile lane.

Updated conclusion:

```text
The next high-upside lane is not another isolated render micro-kernel.
It is either:
1. keep the observation stack/policy input as a device tensor until inference,
   or
2. keep many CPU actors parallel and move compact/shared buffers into batched
   GPU inference/render work.
```

For CurvyTron right now, the smallest honest probe is still profile-only:
device/uint8 stack plus explicit GPU normalization/model-handoff timing. If
that probe merely moves the `~45ms` host stack bucket into a later
materialization bucket, we have not won.

## Bottom Line

The current batched GPU observation work is a real profile lane, but it is not
yet the architecture used by proven high-throughput GPU RL systems. It batches
CurvyTron rows through env state, two player views, direct GPU render, and a
trainer surface stack, then crosses back to host and scalarizes into
LightZero-shaped Python/NumPy timesteps.

That is why it can beat CPU-oracle subprocess controls at some widths but still
does not yield a 5-10x full-loop win. The current C512 read in local docs is:

```text
best real batched GPU C512:    ~1439.84 steps/s
C512 zero-observation ceiling: ~1805.22 steps/s
observation-only upside:       ~1.25x
```

The hard conclusion is that another render-kernel pass cannot explain a 5-10x
result under this stock LightZero-shaped loop. A large win requires preserving
large batches across env step, observation, policy/search, replay, and learner
or preserving actor parallelism while batching the GPU-heavy work centrally.

## Pattern 1: Scalar GPU Is The Wrong Mental Model

The docs repeatedly separate scalar GPU rendering from batched GPU rendering.
The scalar `jax_gpu` trainer backend renders one env row at a time, copies back
to NumPy, and was slower than `cpu_oracle` in the stock trainer. That path pays
GPU launch/copy/synchronization overhead without enough work per call.

The useful local shape is instead:

```text
VectorMultiplayerEnv[B,2]
-> compact state
-> batched GPU direct_gray64/simple_symbols render
-> [B,2,4,64,64]
-> scalar LightZero timesteps only at the outside boundary
```

This is the right direction, but still not the final system pattern. It batches
only the observation surface. Code confirms the current bridge sorts scalar
LightZero env ids, loops them into a `[B,2]` joint action, steps once, then
returns dicts keyed by scalar env id. That is a bridge, not a device-resident
RL loop.

## Pattern 2: The Env Manager Boundary Is The Critical Architecture Line

Stock LightZero expects a vector env manager, but in the trusted CurvyTron lane
each wrapper is effectively a scalar ego stream: LightZero chooses one ego
action, and the env wrapper supplies the fixed/frozen opponent action
internally. `collector_env_num` creates many scalar envs; it does not
automatically create one large CurvyTron render batch.

The new profile manager changes that by mapping two scalar LightZero env ids to
one physical CurvyTron row. This is why the batched manager is meaningful. But
it remains profile-only and stock-boundary-compatible:

```text
stock train_muzero
-> patched create_env_manager
-> SourceStateMultiplayerTrainerSurface(B,2)
-> renderer_backed_profile stack
-> batched renderer or zero renderer
-> BatchedLightZeroScalarActionBridge
-> BatchedLightZeroStockEnvManagerAdapter
-> stock collector sees scalar env ids again
```

This design is a smart compatibility experiment. It is not yet the design a
proven GPU RL system would choose if starting from scratch, because it
intentionally reconstitutes scalar timesteps after the batched surface.

## Pattern 3: Policy/Search Batching Matters As Much As Observation Batching

Known Zero-style systems in the local notes separate actors, inference/search,
replay, learner, checkpoint, and eval. The important pattern is not just "the
env is fast." It is that many roots or actor requests feed batched model/search
work, and the learner does not block every collection step in one synchronous
hot loop.

Stock LightZero does batch policy/search over ready roots inside
`policy.collect_mode.forward`, but it is still synchronous inside collection.
There is no separate actor fleet, no centralized inference service, and no
concurrent learner. At high widths the local rows already show non-render walls:
policy forward, MCTS/search, manager step, scalar timestep construction, and
normal-death live-root collapse.

MCTX/JAX in the local docs is the clean batched-search primitive if CurvyTron
eventually owns a JAX model/search stack. Its relevant shape is root batching:

```text
obs_roots[R,4,64,64]
invalid_actions[R,3]
row_mask[R]
-> jitted representation/prediction/search
-> action[R], action_weights[R,3], root_value[R]
-> scatter to joint_action[B,2]
```

But MCTX is not a trainer. Using it for real would imply owning model,
learner, replay, checkpoint, and actor freshness semantics. It is not a drop-in
optimization for the trusted stock LightZero lane.

## Pattern 4: Host Overhead Is Now First-Class

The local code and docs agree on the current CPU/GPU crossing:

```text
CPU VectorMultiplayerEnv state
-> CPU compact/pack
-> H2D copy
-> JAX render and block_until_ready
-> D2H readback
-> CPU transpose/row-major pack
-> CPU float32 stack update
-> Python BaseEnvTimestep/info payload
-> LightZero uploads obs to GPU again for policy/search
```

This is why "GPU render" is not "GPU RL." The render can be fast and the loop
can still pay sync, copy, stack, scalar object, and policy re-upload costs.
The zero-observation lane is useful precisely because it preserves most of this
stock-loop boundary while removing pixels. At C512 that floor is still only
about `1.8k steps/s`, far below a 5-10x target.

Payload format is another host-side clue. The uint8 note says B512 direct-GPU
still pickled about `67.1MB` per step because the LightZero policy payload is
`1024 * 4 * 64 * 64 * float32`. A uint8 payload could cut observation bytes by
about 4x, but it requires an explicit collector/learner/RND model contract; it
cannot be a hidden env flip.

## What Proven Systems Do Differently

Using only the local architecture research notes, the proven patterns are:

- End-to-end GPU env stacks, like the Isaac Gym/Brax/CuLE family described in
  local docs, keep env state, observation, reward/reset, and policy tensors on
  accelerator. They avoid per-step CPU/GPU transfers and Python synchronization.
- High-throughput CPU/vector systems, like EnvPool/Sample Factory/Puffer-style
  notes, keep simulation highly parallel and move compact buffers through shared
  memory/static buffers while batching inference.
- Zero-style systems, like the local AlphaZero/OpenSpiel/SEED/MuZero notes,
  split actors, inference/search, replay, learner, checkpoint, and eval. They
  batch neural inference/search across many roots or actor requests.

CurvyTron is currently between these patterns. It is no longer purely scalar,
but it is also not device-resident and not a true actor/inference/replay
topology. That middle position explains the measured partial wins.

## What Would Need To Change For A Real 5-10x

The current best full-loop GPU lane is around `1.4k steps/s`; the local
zero-observation ceiling is around `1.8k steps/s`. A 5x result over `1.4k`
means roughly `7.2k steps/s`; a 10x result means roughly `14.4k steps/s`.
That requires architecture, not only render cleanup.

Most plausible paths:

1. **Hybrid actor plus batched GPU observation.** Keep subprocess-style actor
   parallelism for CurvyTron stepping, but send compact state to a central GPU
   observation service. This tests whether actor parallelism plus render
   batching can beat the one-process zero-observation ceiling.
2. **Reduce scalar LightZero payload churn.** Keep larger contiguous arrays
   alive longer, reduce per-env dict/object construction, and consider a named
   uint8 observation contract with explicit cast/scale in collector, learner,
   replay, and RND.
3. **Make search/root batching a first-class profile axis.** Measure average
   root batch size, policy/search time, MCTS time, and GPU utilization across
   C256/C512/C768 and sim2/sim4/sim8. If sim or width moves the wall to search,
   renderer work is not the main lane.
4. **Device-resident env/search rewrite.** The cleanest 5-10x architecture is
   a JAX/Torch-owned CurvyTron state transition, reward/reset, observation,
   stack/latest-frame extraction, and possibly search. This is the biggest
   ownership change and not a quick stock LightZero patch.
5. **Separate RND as its own workload.** RND meter is only about `10-12%` in the
   current C512 rows, but aggressive cadence can dominate. It must be scheduled
   and profiled separately from renderer claims.

Near-term critical rule: keep Coach/default training on stock
LightZero/frozen-opponent CPU-oracle until the batched lane has stable semantic
gates and a matched full-loop speed win that survives normal death, RND meter,
backend identity checks, and no hidden fallback.

## Exact Files And Sections Read

- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/README.md`
  - "Plain Current Truth"
  - "What Could Go Wrong With Batching"
  - "Current Gates"
  - "Next Optimization Priority"
  - "Current Speed Read"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md`
  - "Full Loop Shape"
  - "Where A 10x Win Could Come From"
  - "2026-05-21 Amdahl Check"
  - "2026-05-21 Gate Update And Priority Shift"
  - "External Architecture Research Update"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/orchestration.md`
  - "Current Priority"
  - "Active Wave: Vector/Full-Loop Bridge"
  - "Active Wave: External GPU Architecture Research"
  - "Separate Axis: RND Cadence"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/host_overhead_map.md`
  - "Buckets To Measure"
  - "Current Suspicion"
  - "Instrumentation Gaps From Code Review"
  - "Profile Keys Added"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_gpu_architecture_research.md`
  - "Bottom Line"
  - "Local Context To Preserve"
  - "External Patterns"
  - "Architecture Read For CurvyTron"
  - "Concrete Next Experiments"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_host_overhead_code_audit.md`
  - "Bottom Line"
  - "Current Stock-Boundary Dataflow"
  - "Candidate Render Boundary"
  - "Trainer Surface Dataflow"
  - "Scalar LightZero Boundary"
  - "Zero Observation Lane"
  - "CPU/GPU Crossings"
  - "Already Batched Versus Still Scalar"
  - "Host Sync And Copy Suspects"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_amdahl_next_priorities.md`
  - "Bottom Line"
  - "Existing Throughput Anchors"
  - "Amdahl Read"
  - "Timer Clues"
  - "What 5-10x Requires"
  - "Priority Order"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_amdahl_sanity_20260521.md`
  - "Bottom Line"
  - "Trusted Throughput Anchors"
  - "Amdahl Read"
  - "CPU-Oracle Versus Batched GPU"
  - "What A 10x Win Would Require"
  - "Recommended Priority"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/hybrid_actor_batched_observation_prototype_plan.md`
  - "Plain Goal"
  - "Why This Is Different"
  - "First Version"
  - "Pass Criteria"
  - "Relationship To Current Grid"
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/uint8_payload_design_note.md`
  - "Plain Finding"
  - "Why It Is Not A Simple Flip"
  - "Practical Read"
- `docs/working/training/curvytron_architecture_research_2026-05-12/stock_lightzero_dataflow.md`
  - "Stock CurvyTron Entry"
  - "Stock LightZero Runtime Objects"
  - "Env Manager Seam"
  - "Collector Seam"
  - "Policy/Search Seam"
  - "GameSegment Seam"
  - "MuZeroGameBuffer Seam"
  - "LightZero Runtime Assumptions Exposed Here"
- `docs/working/optimizer/architecture_reexploration_2026-05-12/lightzero_stock_dataflow.md`
  - "Plain Read"
  - "One Stock Iteration"
  - "Where Things Live"
  - "CurvyTron Env Step"
  - "Synchronous Vs Parallel"
  - "GPU And CPU Boundary"
  - "Biggest Architecture Limits"
  - "Optimizer Implications"
- `docs/working/optimizer/architecture_reexploration_2026-05-12/large_scale_zero_architectures.md`
  - "Plain Answer"
  - "Pieces That Are Usually Separated"
  - "Which Bottleneck Moves First?"
  - "CurvyTron Experiments To Run"
  - "Bottom Line"
- `docs/working/optimizer/architecture_reexploration_2026-05-12/mctx_jax_search.md`
  - "Plain Verdict"
  - "What MCTX Would Replace"
  - "What MCTX Would Not Replace"
  - "Required Data Shapes"
  - "GPU Batching Model"
  - "Risks And Costs"
- `src/curvyzero/training/source_state_batched_observation_mock_collector.py`
  - `BatchedLightZeroScalarActionBridge`
  - `BatchedLightZeroProfileEnvManager`
  - `BatchedLightZeroStockEnvManagerAdapter`
- `src/curvyzero/training/multiplayer_source_state_trainer_surface.py`
  - `_RendererBackedSourceStateGray64Stack4`
  - `SourceStateMultiplayerTrainerSurface`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  - `ZeroObservationRenderer`
  - `curvyzero_batched_profile` env-manager hook registration/fail-closed path
