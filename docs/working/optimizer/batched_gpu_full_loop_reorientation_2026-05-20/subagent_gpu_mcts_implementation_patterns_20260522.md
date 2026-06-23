# GPU / Parallel MCTS Implementation Patterns

Date: 2026-05-22

Scope: research note for CurvyTron optimizer search/replay architecture. This
is doc-only. No live training runs, trainer defaults, Modal state, or production
code were changed.

## Bottom Line

GPU MCTS is likely one real blocker, but not the only blocker. The current
CurvyTron wall is better described as:

```text
stock LightZero collect/search/replay topology
  = scalar/object env rows
  + CPU/list CTree boundary
  + Python per-simulation search loop
  + GPU recurrent inference results copied back to CPU each sim
  + replay/RND/learner object materialization after search
```

The strongest external pattern is not "put MCTS on CUDA" in isolation. It is:

```text
many active roots / games / positions
-> compact array tree state
-> batched neural leaf/recurrent inference
-> few host synchronization points
-> replay rows materialized only at a coarse compatibility edge
```

That means the next serious optimization should attack the LightZero CTree/list
boundary and compact replay boundary together. A GPU-resident search can help a
lot only if it keeps enough roots/leaves alive and avoids many tiny
kernel-launch / CPU-sync / atomic-contention traps.

## Local Context Read

Local docs already say the denominator moved away from rendering:

- `current_hot_path_bottleneck_map_20260522.md`: current direct path still
  pays root CPU prep, CTree CPU/list APIs, per-simulation recurrent-output D2H,
  replay/learner/RND object lanes, and scalar env-manager fanout.
- `search_boundary_next_wave_20260522.md`: profile-only H100 ladder has
  `direct_ctree_gpu_latent` around `6145 roots/sec` at sim16 versus stock
  facade around `2094 roots/sec`; dense eager Torch is fast at sim8 but falls
  behind at sim16.
- `array_native_ctree_opportunity_20260522.md`: LightZero CTree is already C++,
  but its public Cython/Python boundary is nested-list shaped.
- `subagent_compiled_search_architecture_20260522.md`: the next plausible
  speedups come from array-native CTree or fixed-shape compiled dense search,
  not just another renderer pass.

Important caveat from the local notes: the profile-only search ladder looks much
better than the full `train_muzero` denominator. Direct CTree GPU-latent is
valuable evidence, but a full-loop win still has to include collector, replay,
learner, and RND.

## Practical Options For Us

| Option | What it means | Expected upside | Main engineering risk | Fit now |
| --- | --- | ---: | --- | --- |
| Keep LightZero CTree, reduce boundary | Keep stock CTree semantics but replace root prep, traverse/backprop, and output with dense fixed-`A=3` arrays where possible. Keep PyTorch recurrent inference. | `1.2x-1.6x` over current direct search if per-sim list/D2H churn is large; maybe `>2x` over stock facade. Full-loop likely smaller unless replay/env lanes also shrink. | Cython/C++ extension maintenance; preserving LightZero semantics; still CPU-owned tree and per-sim model synchronization. | Best near-term conservative lane. |
| C++/CUDA extension | Own a CurvyTron-specific fixed-shape MCTS extension with C++ host orchestration and optional CUDA kernels for PUCT/select/backup/tree arrays. | Could remove Python from inner loop and support GPU-resident tree state. If done well, plausible `2x-4x` search-side over current direct. | Hardest semantic/debug surface; atomics, locks, memory layout, kernel launch overhead, and model-call integration can erase the win. | Worth only after microbench falsifiers show CTree boundary is not enough. |
| Triton kernels / compiled Torch | Keep search as dense arrays in PyTorch/Triton with fixed `R=B*2`, `A=3`, fixed sim count, static buffers, masks instead of dynamic shapes. Fuse PUCT/select/backprop fragments. | Good for answering "can GPU-resident fixed-shape search beat CTree?" Maybe `1.2x-2x` over direct at sim16 if launch overhead is tamed. | Eager Torch already failed sim16 scaling; Triton/custom kernels need careful fusion to avoid many tiny kernels and graph breaks. | Best profile-only GPU falsifier before a full CUDA extension. |
| JAX/MCTX | Scratch or alternate lane using JAX-native MCTX MuZero/Gumbel MuZero search, pure JAX recurrent function, JIT, batch-first roots. | Cleanest accelerator-native design; strong reference for resident search. Could be big if env/search/model/replay all stay JAX/device-side. | Not a patch to PyTorch LightZero. PyTorch model inside JAX recreates host boundary; serious path implies JAX model/replay/checkpoint ownership or shadow conversion. | Useful as tiny viability toy, not train-facing patch. |
| CPU vectorized / compiled sidecar | Build fixed-`A=3` C++/Rust/Numba CPU tree sidecar with dense arrays and batched model calls; maybe many CPU threads across roots, not inside one root. | Lower semantic risk than CUDA; may beat LightZero list boundary and be easier to debug. Could deliver `1.2x-2x` search-side if current Python/list overhead dominates. | Still model GPU sync per sim; CPU capacity alone already failed a local CPU64 falsifier; could just rebuild CTree. | Good fallback if GPU kernels are launch-bound. |
| Batched search service | MiniZero/KataGo-style actor/search service: many env rows/seats alive, one evaluator/search worker, compact arrays in/out, replay writer at edge. | Only option here that plausibly changes the full-loop denominator by `3x+`, because it attacks batch fill, search, and replay materialization topology. | Larger architecture change; requires new trust gates for replay targets, self-play freshness, RND, checkpoints, and training parity. | The likely medium-term architecture if full-loop speed is the goal. |

## What The Literature Says

### Parallelization Modes

Classic MCTS parallelization separates into root, leaf, and tree
parallelization. Chaslot, Winands, and van den Herik describe these modes and
call out that effective tree parallelization needs local mutex handling and
virtual loss. The practical read for CurvyTron is:

- Root parallelism is easy and communication-light, but combines independent
  trees only at the root. This maps to many CurvyTron row/seat roots, but does
  not make one tree deeper faster.
- Leaf parallelism batches rollout/evaluation from selected leaves. This is the
  most natural MuZero/PUCT pattern because neural evaluation is expensive and
  batchable.
- Tree parallelism shares one tree across workers. It can improve search but
  introduces contention, locks/atomics, and virtual-loss semantics.

For MuZero-like search, the high-value primitive is usually batched leaf /
recurrent inference across many roots, not massive contention inside one tiny
`A=3` root.

### Batched Neural MCTS

Cazenave's Batch MCTS paper states the key systems fact directly: batching
network inference on GPU is much faster than sequential single-state inference.
It proposes using tree statistics plus a table of evaluated states, and discusses
virtual loss / virtual mean style handling while leaf evaluations are pending.

MiniZero implements the same broad architecture in a production-ish training
system: each self-play worker maintains multiple MCTS instances, selects leaves
from them, then evaluates leaves by batch GPU inference. That maps well to
CurvyTron's natural root batch:

```text
rows[B] x seats[2] -> roots[R=B*2] -> batched recurrent inference
```

KataGo's analysis engine is also a warning: it gets speed from analyzing many
positions in parallel and exploiting cross-position batching. A single scalar
root with tiny requests does not keep a modern GPU busy.

### LightZero Is Already Partly Optimized

LightZero's docs say MuZero CTree uses C++ for `batch_traverse` and
`batch_backpropagate`, and that batched root search parallelizes model
inference. So "rewrite MCTS in C++" is not a diagnosis by itself. The remaining
CurvyTron problem is the CTree ABI around those C++ calls:

```text
policy_logits.tolist()
legal_actions: List[List[int]]
roots.prepare(...)
batch_traverse(...) -> Python lists
recurrent_inference(...)
reward/value/policy -> detach().cpu().numpy().tolist()
batch_backpropagate(...)
roots.get_distributions() -> List[List[int]]
```

An array-native CTree boundary is therefore more targeted than a broad rewrite.

### GPU MCTS Papers / Repos: What Transfers

The tensor MuZero MCTS paper by Balaz and Tarabek proposes processing a batch of
observations on a single GPU with tensor operations, explicitly because MuZero
must run one MCTS per observation in training. Their data layout uses tensors
for node/state/action statistics, including `Q`, `R`, and policy arrays. This is
the closest published shape to a dense Torch/Triton CurvyTron spike.

The `mcts_numba_cuda` repo is a useful lower-level CUDA reference: it combines
leaf/root/tree-level parallelism, uses CUDA block/thread grouping, reductions for
sum/max/argmax, and emphasizes no atomics/mutexes plus few host-device
transfers. The transfer to CurvyTron is mostly architectural: avoid shared-tree
contention when possible, use block-level reductions, and minimize D2H.

Older GPU MCTS work is split. Rocki and Suda's CUDA poster uses block
parallelism and independent searches to fit SIMD hardware, while also calling
out CPU sequential tree management as a weakness. Buzer and Cazenave's GPU Monte
Carlo Search paper gets very large speedups for playout-friendly nested Monte
Carlo search, but it also explains why generic Monte Carlo game code historically
did not map easily to GPUs: iterative control, small per-core cache, and slow
global memory latency. The exact speedups do not transfer directly to MuZero
PUCT, but the warnings do.

MCTX is the cleanest modern implementation reference: JAX-native, JIT-friendly,
batch-first AlphaZero/MuZero/Gumbel MuZero search. It is a design target, not a
drop-in LightZero patch.

## Why Naive GPU MCTS Often Fails

1. **The search is sequential by design.** Each simulation's PUCT choice depends
   on visit/value statistics from earlier simulations. Parallel workers either
   duplicate work, need virtual loss/unobserved-sample accounting, or contend on
   shared nodes.

2. **Small action spaces do not saturate GPUs.** CurvyTron has `A=3`. A naive
   "one root, one kernel per selection/backprop" implementation has too little
   arithmetic and too much launch overhead.

3. **Atomic contention can dominate shared-tree CUDA.** Tree parallelism updates
   visit counts and value sums along common upper paths. Atomic adds and CAS
   around hot root edges can serialize the workload. Virtual loss reduces
   duplicate selection but adds more writes and semantic knobs.

4. **Irregular memory layout hurts coalescing.** Pointer-heavy nodes and dynamic
   child lists make poor GPU data structures. The GPU-friendly shape is
   structure-of-arrays:

   ```text
   visits[R, max_nodes, A]
   value_sum[R, max_nodes, A]
   priors[R, max_nodes, A]
   rewards[R, max_nodes, A]
   child_index[R, max_nodes, A]
   parent_index[R, max_nodes]
   action_from_parent[R, max_nodes]
   latent_pool[R, max_nodes, H]
   ```

5. **Host synchronization kills overlap.** If every simulation does
   `recurrent -> detach().cpu().numpy() -> CTree -> next recurrent`, the GPU
   cannot hide latency. CurvyTron's current direct CTree path still has exactly
   this recurrent-output D2H before backprop.

6. **Dynamic shapes break compiler/capture tools.** CUDA graphs and static
   compiler wins require stable memory addresses, stable shapes, and limited
   dynamic control flow. `nonzero`, variable legal-action lists, variable active
   roots, Python exceptions, `.item()`, and allocation inside the loop all fight
   this.

7. **Full-loop Amdahl limits arrive quickly.** Even a perfect search microbench
   does not remove env scalarization, replay target building, RND latest-frame
   extraction, learner batch materialization, checkpoint/eval, or telemetry
   syncs.

## Recommended First Experiments

These should remain profile-only and should not touch live training runs.

1. **No-model CTree ABI microbench.**
   Feed synthetic `reward[N]`, `value[N]`, `policy[N,3]`, `legal_mask[N,3]`
   through current LightZero CTree root/traverse/backprop/output versus a flat
   array prototype. Sweep `N`, sim8, sim16, sim32. This isolates list/root API
   overhead from model cost.

2. **Resident recurrent-output falsifier.**
   Replace recurrent inference with precomputed resident tensors in the current
   direct CTree GPU-latent loop. If throughput remains far below dense search
   ceilings, CTree/list/control is the wall. If it jumps, recurrent launch/D2H
   is the wall.

3. **Fixed-shape dense GPU MCTS spike.**
   For `R=B*2`, `A=3`, sim16, allocate all tree arrays once on device. Use masks
   for live roots and legal actions. Avoid `nonzero`, `.item()`, per-sim
   allocation, and D2H until final actions/visits. Test eager, `torch.compile`,
   and one or two Triton kernels for PUCT argmax/backprop.

4. **Array-native CTree API sketch before implementation.**
   Prototype only the Python-facing ABI first:

   ```text
   prepare(values[N], policy_logits[N,3], legal_mask[N,3], noise[N,3])
   traverse() -> latent_index[N], action[N]
   backprop(reward[N], value[N], policy_logits[N,3])
   output() -> visits[N,3], root_value[N], selected_action[N]
   ```

   The useful patch must reduce per-simulation list/API overhead. Root/output
   cleanup alone is probably too small.

5. **Compact replay writer dry run.**
   In a non-live profile, bypass stock game-segment materialization and write
   compact arrays:

   ```text
   obs_ref, action, reward, done, to_play, visit_counts[3], root_value, mask[3]
   ```

   This tests whether search wins are immediately swallowed by replay/target
   object work.

6. **MCTX/JAX toy only as architecture probe.**
   Keep it tiny: fake CurvyTron-like simultaneous-action state, `R=B*2`, vector
   latent, `gumbel_muzero_policy`, sim8/16. Success means "resident batch/search
   is plausible", not "replace LightZero tomorrow".

## Decision Guidance

Near-term recommendation:

```text
Keep LightZero CTree semantics, but attack the array boundary first.
```

Reason: this is the smallest path that matches current evidence. It keeps
LightZero PUCT/target semantics close while removing the proven nested-list and
per-simulation boundary. It should be validated against full-loop, not just
profile-only roots/sec.

Medium-term recommendation:

```text
Build a profile-only batched search service shape.
```

Reason: external systems that scale do not merely optimize one MCTS call. They
keep many roots alive and batch leaf/model work. For CurvyTron, this is likely
the first architecture that can make GPU search a full-loop win rather than a
search microbench win.

Deprioritize for now:

```text
"Just add CPUs" and "rewrite all MCTS in CUDA" without a boundary falsifier.
```

Reason: local CPU64 already failed as a simple capacity fix, and naive CUDA MCTS
can be slower if it introduces atomics, tiny kernels, dynamic allocation, or
host syncs. CUDA is attractive only after fixed-shape array search has shown
where the cycles are.

## Is GPU MCTS The Blocker?

Yes, but with a qualifier.

The current blocker is not "we lack CUDA MCTS." It is that the current
LightZero-compatible topology makes search a CPU/list/Python synchronization
problem even when the model and latent tensors are on GPU. GPU MCTS is one way
out, but the real blocker is the boundary:

```text
GPU model tensors <-> CPU CTree/list APIs <-> Python simulation loop
```

If we only replace CTree with naive GPU kernels, the blocker may move to launch
overhead, atomics, and replay materialization. If we replace the boundary with
batch-first fixed arrays and compact replay writes, GPU MCTS becomes a plausible
major contributor to a larger speedup.

Practical answer:

- For `1.3x-2x` full-loop: array-native CTree plus compact collect/replay edge
  is the likely best bet.
- For `3x+` full-loop: we probably need a MiniZero/KataGo-style batched search
  service or JAX/MCTX-like resident lane, plus replay/RND cleanup.
- For `5x-10x`: search alone is not enough; the full actor/search/replay/learner
  topology must become array-native.

## Source Links

- Chaslot, Winands, van den Herik, "Parallel Monte-Carlo tree search":
  <https://cris.maastrichtuniversity.nl/en/publications/parallel-monte-carlo-tree-search/>
- Liu et al., "Watch the Unobserved: A Simple Approach to Parallelizing Monte
  Carlo Tree Search": <https://arxiv.org/abs/1810.11755>
- Cazenave, "Batch Monte Carlo Tree Search":
  <https://ludii.games/citations/ARXIV2021-1.pdf>
- Balaz and Tarabek, "Tensor Implementation of Monte-Carlo Tree Search for
  Model-Based Reinforcement Learning":
  <https://www.mdpi.com/2076-3417/13/3/1406>
- Source repo for that tensor MuZero implementation:
  <https://github.com/marrekb/MuZero>
- DeepMind MCTX, JAX-native MCTS:
  <https://github.com/google-deepmind/mctx>
- MiniZero architecture and batched GPU inferencing:
  <https://github.com/rlglab/minizero>
- KataGo analysis engine cross-position batching:
  <https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md>
- LightZero tree-search docs:
  <https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html>
- MCTS-NC Numba CUDA repo:
  <https://github.com/pklesk/mcts_numba_cuda>
- Rocki and Suda, "Accelerating Parallel Monte Carlo Tree Search using CUDA":
  <https://developer.download.nvidia.com/GTC/PDF/GTC2012/Posters/P0227_AcceleratingParallel_KamilRocki.pdf>
- Buzer and Cazenave, "GPU for Monte Carlo Search":
  <https://www.lamsade.dauphine.fr/~cazenave/papers/MonteCarloGPU_LION.pdf>
- PyTorch CUDA Graphs blog:
  <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>
