# MCTX Comparator Status, 2026-05-23b

Scope: read-only scout pass over the MCTX/JAX lane. I inspected the working
optimizer notes, `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`, the
MCTX legality tests, and current MCTX/JAX public docs. No live training runs,
Modal jobs, checkpoints, or benchmark artifacts were launched or modified.

## Short Answer

MCTX is already a credible accelerator-native search comparator for this repo.
It is not yet a next-main training move.

The repo has moved past "can MCTX run?" It now has profile-only code that can
consume real renderer-backed compact visual roots, run a tiny pure-JAX model
through `mctx.gumbel_muzero_policy`, validate legal compact outputs, and report
a comparable `compact_search_service_profile` row that is explicitly marked
`profile_only`, `not_lightzero_ctree`, and `not_train_muzero`.

The smallest useful next MCTX comparator is a same-denominator real-root sidecar
row against the current direct CTree/service-tax grid: current compact visual
roots, `R=B*2`, `A=3`, `H=64`, sim16 and sim32, compile excluded, selected
actions fed to the next compact env step, and replay-index proof on. Its job is
to measure the all-device ceiling and boundary taxes. It should not be sold as
CurvyTron learning evidence.

## 1. What Has Already Been Tried?

### Dependency and Synthetic Runtime

- `docs/experiments/2026-05-09-modal-mctx-dependency-smoke.md` records CPU and
  GPU Modal smokes using pinned `mctx==0.0.6` and `jax/jaxlib==0.7.0`. The GPU
  smoke ran on a Modal L4, saw JAX backend `gpu`, and completed one tiny
  `gumbel_muzero_policy` call.
- `docs/experiments/2026-05-09-modal-mctx-synthetic-benchmark.md` records flat
  synthetic MCTX profiles and a small batch/simulation sweep. The useful larger
  L4 synthetic profile was `B=64`, sim16, `H=64`, max depth 16, with about
  `12.1k` decisions/sec and `193k` simulations/sec. This was search/runtime
  evidence only.
- The benchmark then gained CurvyTron-shaped modes: synthetic `[B,P,9]` debug
  observations, fixture-seeded CPU debug-packer output, and actor-bridge sample
  output. These proved shape, mask, host-setup, H2D, search, and D2H timing
  boundaries, but still used synthetic JAX dynamics.

### Real Compact Visual Root Path

`src/curvyzero/infra/modal/mctx_synthetic_benchmark.py` now goes beyond the
original synthetic flat smoke:

- `curvytron_visual_root` builds synthetic `[B,2,4,64,64]` visual-policy stacks.
- `curvytron_hybrid_compact_visual_sample` builds a renderer-backed
  `HybridCompactBatch`, validates `CompactRootBatchV1`, and runs MCTX on real
  `[B,2,4,64,64]` compact visual roots.
- The compact visual mode supports host root tensors and a resident GPU stack
  source via `compact_visual_observation_source=resident_gpu`.
- It times host setup, H2D, compile-plus-first-run, warm steady search, D2H
  action/action-weight output, one-step replay edge, and optional closed compact
  loop buckets.
- It can build `CompactSearchResultV1` and
  `CompactReplayIndexRowsV1` from MCTX output when root values are available.
- The closed-loop mode can feed MCTX-selected actions into the next hybrid env
  step and measure root build, search, D2H, joint action build, env step, and
  replay-index proof.
- The benchmark intentionally uses a tiny pure-JAX representation/prediction/
  recurrent model. It does not call the current PyTorch LightZero model.

The latest working-memory read in `world_model.md` and
`current_state_audit_20260523.md` records profile-only H100 compact visual
closed-loop rows:

```text
curvytron_hybrid_compact_visual_sample, H100, B512/P2/body1024,
closed_loop_steps=24, native_actor_buffer=true,
compact_root_copy_observation=false, compact_visual_resident_sync=false

sim16: 27.6k active roots/sec, search 0.149s, env_step 0.647s
sim32: 22.2k active roots/sec, search 0.362s, env_step 0.650s
```

That is strong architecture evidence. It is not LightZero-equivalent because
the model/search semantics are toy JAX/MCTX.

### Guardrails Already In Tests

`tests/test_mctx_synthetic_benchmark_legality.py` covers the key local MCTX
guardrails:

- active roots only: legal selected actions, finite/action-weight row sums, and
  zero illegal action-weight mass;
- failure on illegal selected action or illegal visit mass;
- resident compact visual latest-frame shape and `uint8` dtype;
- row-major `[env_row, player]` root ordering for resident compact visual mode;
- closed compact timing breakdown separation between mechanics and handoff;
- profile row labels: `profile_only`, `not_lightzero_ctree`,
  `not_train_muzero`;
- root-value extraction from `search_tree.node_values[:,0]` for replay payload
  profiling.

This means the current MCTX lane has shape and legality safety rails. It still
does not have LightZero semantic parity, learner parity, or training proof.

## 2. Smallest Real-Root Comparator That Would Teach Us Something Now

Use the existing `curvytron_hybrid_compact_visual_sample` path. Do not add a new
trainer path. Do not bridge PyTorch into JAX.

Run a tiny same-denominator comparator grid:

```text
Backend: mctx_hybrid_compact_visual_search_service
Claim: profile-only sidecar, not LightZero CTree, not train_muzero

Input:
  current HybridCompactBatch -> CompactRootBatchV1
  real compact visual roots [B,2,4,64,64] -> [R,4,64,64]
  B=512, P=2, R=1024
  A=3 invalid-action masks
  active/inactive root masks preserved

MCTX:
  tiny pure-JAX visual encoder
  vector hidden H=64
  mctx.gumbel_muzero_policy
  sim16 and sim32
  max_depth == sim count for the comparator rows

Loop:
  compile and warmup reported separately
  closed_loop_steps around 24 is enough for the first comparator
  selected actions drive the next env step
  compact replay-index proof enabled
  compare host-stack input vs resident-GPU-stack input only if the first row is clean
```

Measure these buckets in the same report row:

- root/visual setup and root sidecar build;
- H2D or resident-stack handoff;
- warmed search only;
- D2H action-only;
- D2H action weights plus root value;
- compact search-result validation;
- replay-index proof;
- env step after selected-action feedback;
- total active roots/sec and slowest bucket.

Decision rule:

```text
Keep MCTX as a serious side comparator if warmed setup + search + required D2H
is at least 2x faster than direct CTree on the same B/P/sim denominator, with
legal outputs, no surprise recompiles, and replay-index proof green.

Demote it to architecture reference only if visual setup/H2D/env step dominates,
if shapes recompile, if root values/payload extraction are unstable, or if it
requires PyTorch host callbacks inside the JAX recurrent function.
```

This comparator would teach one useful thing now: whether the all-device search
ceiling still matters once real compact visual roots, selected-action feedback,
and replay-index proof are included. It would not teach whether the current
PyTorch/LightZero policy can be replaced.

## 3. What Would Make MCTX A Trap Here?

The trap is treating a clean JAX search body as a drop-in optimization for a
PyTorch/LightZero training system.

Specific failure modes:

- **Framework bridge tax.** MCTX wants pure JAX `RootFnOutput` and
  `recurrent_fn` work inside JIT. Calling the current PyTorch LightZero model
  from inside that recurrent function would reintroduce host callbacks, syncs,
  or CPU copies.
- **Wrong semantic claim.** `gumbel_muzero_policy` output is not automatically
  the same target distribution as LightZero CTree visit-count targets. Values,
  reward transforms, root noise, temperature, and backup rules can drift.
- **Independent root approximation.** MCTX expands one action per root.
  CurvyTron physically steps two players simultaneously. Independent per-seat
  `A=3` roots are the smallest useful comparator, but a joint `A=9` control is
  needed before claiming multiplayer semantic adequacy.
- **Dynamic-shape recompilation.** JAX JIT specializes on static shapes and
  static arguments. Live-root counts must be padded/masked; changing root
  capacity, sim count, hidden shape, or action shape creates new compiled
  profiles.
- **Tree memory.** MCTX stores embeddings in the tree. Vector `H=64` is the
  right first probe; spatial embeddings can make `[R,N,...]` tree memory the
  wall before compute.
- **Replay/RND gap.** MCTX returns search outputs. It does not provide the
  repo's LightZero collector, compact replay chunks, RND latest-frame cadence,
  learner sample parity, checkpoints, eval, GIF, or tournament compatibility.
- **Amdahl whiplash.** The current profile-only MCTX rows already show that once
  search gets fast, `env_step_sec` can become the slowest bucket. Faster search
  alone is not a 5x training path unless compact env/observation/replay
  ownership improves too.
- **Dependency and ops lane.** MCTX/JAX is pinned in Modal images, not a normal
  local runtime dependency. Promoting it means owning JAX/JAXLIB/MCTX install,
  compile cache, CUDA compatibility, artifacts, and versioned model state.

External anchor: the official MCTX README describes MCTX as JAX-native,
JIT-compatible, batched search with `muzero_policy` and
`gumbel_muzero_policy`, requiring root outputs and a recurrent function:
<https://github.com/google-deepmind/mctx>. JAX's own docs note that changing
static arguments or shapes can trigger recompilation:
<https://docs.jax.dev/en/latest/jit-compilation.html>.

## 4. Next-Main Move Or Side Comparator?

Side comparator.

The next-main optimizer move should stay behind the compact search/dataflow
contract:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> learner/replay adapter
```

Within that boundary, the practical main lane is still a trainer-compatible
fixed-shape search backend or compact slab that can beat direct CTree on the
same denominator while preserving replay/RND/player/sample gates.

MCTX/JAX should run beside that lane as the clean accelerator-native ceiling:

- it tells us what a closed, fixed-shape, batched search body can do;
- it keeps pressure on Torch/CTree designs to remove per-simulation CPU/list
  traffic;
- it provides a cold kill signal if the real-root all-device ceiling is not
  enough after setup and D2H;
- it becomes a main move only after an explicit framework decision: JAX-native
  model/search/replay/learner ownership, or a carefully versioned sidecar model
  with its own parity gates.

Plain verdict: run the real-root MCTX comparator because it is cheap and sharp.
Do not route Coach or `train_muzero` through it. Treat every MCTX row as
profile-only until it has a real model strategy, compact replay/RND parity,
stock-vs-candidate smoke, and learner/sample compatibility.

## Local Sources Read

- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_jax_mctx_spike_critique_20260521.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_mctx_gpu_search_research_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_fixed_shape_search_designs_20260523.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/compact_search_replay_service_contract_20260522.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/current_state_audit_20260523.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/world_model.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/next_phase_optimizer_synthesis_20260523.md`
- `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/strategy_reorientation_20260523b.md`
- `docs/experiments/2026-05-09-modal-mctx-dependency-smoke.md`
- `docs/experiments/2026-05-09-modal-mctx-synthetic-benchmark.md`
- `docs/working/optimizer/architecture_reexploration_2026-05-12/mctx_visual_root_benchmark_plan.md`
- `src/curvyzero/infra/modal/mctx_dependency_smoke.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `tests/test_mctx_synthetic_benchmark_legality.py`
