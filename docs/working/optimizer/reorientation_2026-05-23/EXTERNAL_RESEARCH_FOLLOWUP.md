# External Research Follow-Up

Date: 2026-06-02

Scope: research/implementation critique only. No source code, live Coach runs,
Modal state, checkpoints, evals, tournaments, or trainer defaults were changed.

Current application: PufferLib/EnvPool/Sample Factory support the
fixed-buffer/small-reference lesson. They do not justify a CurvyTron port, a
GPU mechanics rewrite, H200/B200, or multi-GPU before the same-work H100
accepted-fast-path repeatability work points there.

## Short Read

External fast-RL systems point at the same next move as the local audit:
CurvyZero's likely multiplier is not another scalar renderer pass or "put MCTS
on GPU" as a slogan. It is moving the hot loop to compact fixed-shape ownership:

```text
compact env rows
-> resident/frame-stack observation arrays
-> batched search/inference service
-> compact action/visit/value arrays
-> replay refs and learner materialization at the edge
```

The trap is speed-currency mixing. Puffer-style env throughput, MCTX roots/sec,
and Coach learner iterations/hour are different denominators until a matched
loop ties them together.

## Patterns To Act On

### 1. Static Slabs Beat Hot Object Forests

Pattern:

- PufferLib-style systems allocate stable contiguous memory for observations,
  actions, rewards, terminals, activations, and transfer buffers.
- Sample Factory sends buffer ids through shared memory rather than serializing
  observation payloads between components.
- EnvPool exposes vector/async env ids and fixed batch-shaped outputs.

CurvyZero next step:

- Treat `HybridCompactBatch`, `CompactRootBatchV1`,
  `CompactSearchResultV1`, and `CompactReplayIndexRowsV1` as the hot facts.
- Keep `BaseEnvTimestep`, `PolicyRowRecordV0`, GameSegment-like rows, Python
  action dicts, and full observation materialization out of collection/search.
- Require row/player/policy ids on every compact row so async chunks cannot
  silently reorder terminal or RND fields.

Trap:

- A slab that immediately re-expands into LightZero objects is just a prettier
  copy tax.
- Compatibility wrappers are fine at validation/sample edges, not in the
  env-search-replay inner loop.

Tiny toy benchmark:

```text
B=512, P=2, T=512 synthetic steps.
Compare:
  A: build scalar timestep dicts plus policy rows every step
  B: write SoA arrays and materialize only every 64th sample batch
Measure wall time, allocations, bytes copied, and row/player identity checksum.
Kill criterion: less than 25 percent improvement means object fanout is not the
current wall in this denominator.
```

### 2. Batch Roots/Leaves, Not Just Environments

Pattern:

- OpenSpiel's fast AlphaZero path separates actors, MCTS/evaluator, learner,
  cache, and batched GPU inference.
- MiniZero/KataGo-style systems keep many games/positions in flight so GPU
  neural evaluation sees real batch fill.
- Sample Factory double-buffering overlaps CPU env work with inference waits.

CurvyZero next step:

- Build the compact search service around `R = B * P` row-seat roots and fixed
  `A = 3`.
- Return only selected action plus stable identity on the env-critical path.
- V1 must split selected-action return from replay-payload flush so the env can
  advance before visit counts/policy, root value, and debug materialization
  become replay-visible.

Trap:

- More collectors can make dashboards look faster while replay attachment,
  terminal final observation, or RND latest-frame identity drifts.
- Do not accidentally convert simultaneous two-seat decisions into sequential
  player decisions or joint `A=9` search unless that is an explicit algorithm
  change.

Tiny toy benchmark:

```text
Synthetic recurrent model: tiny conv or MLP, fixed R in {64,128,256,512,1024},
sim in {4,8,16}.
Compare:
  A: one scalar root search at a time
  B: batched roots, CPU tree, batched recurrent inference
  C: batched roots, fake tree/service-tax recurrent calls only
Measure roots/sec, GPU utilization, per-sim sync count, and action checksum.
The useful output is the batch size where recurrent calls stop being tiny.
```

### 3. Fixed-Shape Search Before CUDA Graphs Or Compile

Pattern:

- MCTX is fast because tree arrays, recurrent function, masks, and loop shape
  are JAX-native and batch-first.
- CUDA graphs/PyTorch graph capture need stable shapes, long-lived tensors, and
  replayed kernels with the same memory addresses.
- Dense GPU MCTS with small `A=3` can lose if it becomes many tiny launches,
  `.item()` gates, `nonzero` compaction, or per-depth allocation.

CurvyZero next step:

- Make the next dense Torch search probe fully padded:
  fixed `R`, fixed `A=3`, fixed `num_simulations`, fixed `N`, live-root masks,
  legal masks, and preallocated path/tree tensors.
- Only after that, test `torch.compile(mode="reduce-overhead")` or CUDA graph
  capture.
- Keep LightZero CTree as semantic oracle while comparing visits/action/value.

Trap:

- `torch -> numpy -> jax -> numpy -> torch` is not GPU search. It is the old
  boundary wearing a faster-looking hat.
- Eager Torch tree polishing without shape stability is likely 1.x cleanup, not
  the 5x path.

Tiny toy benchmark:

```text
Use fake priors/value/reward tensors and no real env.
Compare:
  A: dynamic active-root compaction and allocation each sim
  B: padded fixed-shape eager Torch
  C: padded fixed-shape compiled/captured Torch
  D: CPU CTree/list baseline
Measure sim/sec, kernel launches, syncs, allocations, and max visit/action
delta versus a deterministic CPU oracle.
Proceed only if B or C beats D by at least 1.5x at sim16.
```

### 4. GPU Rendering Helps Only When It Removes A Boundary

Pattern:

- Isaac Gym, CuLE, and PixelBrax win by keeping simulation, rendering or
  observation/reward preprocessing, and policy inputs resident on the device.
- GPU rendering by itself is not the win if frames come back to CPU and then
  immediately return to GPU for search/learner input.

CurvyZero next step:

- Keep the GPU observation lane focused on resident latest-frame/stack input
  for search/RND, not just faster standalone image production.
- Measure CPU oracle, GPU observation, and no-observation-refresh rows inside
  the same compact closed loop.
- Do not spend the next phase on scalar GPU render calls.

Trap:

- Renderer wins can disappear in the Coach loop if LightZero still scalarizes
  envs, stacks on CPU, or rebuilds root payloads per player.
- GPU obs can also hide row-order bugs: view-major versus row-major ordering,
  final observation after autoreset, or RND latest-frame mismatch.

Tiny toy benchmark:

```text
B=512, T=256, sim fixed.
Compare:
  A: CPU dirty-cache render + CPU stack + H2D root input
  B: GPU render + GPU stack + only selected action D2H
  C: no observation refresh, reuse previous stack
Measure closed-loop steps/sec and exact checksums for row/player/latest-frame.
If B is close to C, observation is solved enough; if A and B are close, search
or materialization is still the wall.
```

### 5. Replay Must Own Compact Indices First

Pattern:

- High-throughput RL systems separate rollout, batcher/replay, learner, and
  logging/checkpoint roles.
- Replay should ingest compact identity plus targets, then expand only when the
  learner actually samples.

CurvyZero next step:

- Make compact rows sample-visible only after action, reward, done,
  visit-policy/counts, root value, final-observation sidecar, RND sidecar, and
  policy/search version are attached.
- Add an offline comparator that rejects unmatched speed currencies and
  unmatched stock/candidate profile rows.

Trap:

- "Replay sampling is not the wall" does not mean replay ownership is safe.
  The costly part may be hot-path construction and identity drift, not the
  sample call itself.
- Autoreset can make the learner see the wrong next/final observation unless
  terminal sidecars are committed before reset reuse.

Tiny toy benchmark:

```text
Generate compact search results for non-prefix env ids, shuffled players, and
forced terminal rows.
Compare:
  A: materialize full learner payload every collect step
  B: store compact refs, materialize only sampled groups
Measure commit time, sample time, bytes copied, and parity of final obs/RND
frame/action/visit rows against scalar oracle.
```

## Priority Order

1. Run the matched full-loop stock-vs-candidate profile gate first. Do not infer
   Coach speed from compact roots/sec.
2. Build the compact closed-loop falsifier rows: observation off, mechanics
   no-op, scalar materialization on, replay materialization on, precomputed
   recurrent outputs, service-tax search, mock search.
3. Tighten the fixed-shape dense Torch search probe before judging GPU MCTS.
4. Promote GPU observation only if it stays resident through search/RND input
   and wins in the closed loop.
5. Keep MCTX/JAX as a ceiling/prototype until model/search/replay semantics are
   explicitly accepted as an algorithm change or proven close enough.

## Sources Checked

- PufferLib docs: https://puffer.ai/docs.html
- EnvPool paper: https://papers.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html
- EnvPool Python interface: https://envpool.readthedocs.io/en/latest/content/python_interface.html
- Sample Factory architecture: https://www.samplefactory.dev/06-architecture/overview/
- Sample Factory double-buffering: https://www.samplefactory.dev/07-advanced-topics/double-buffered/
- MCTX README: https://github.com/google-deepmind/mctx
- OpenSpiel AlphaZero docs: https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- NVIDIA Isaac Gym overview: https://developer.nvidia.com/blog/introducing-isaac-gym-rl-for-robotics/
- NVIDIA CuLE page: https://research.nvidia.com/publication/2019-07_gpu-accelerated-atari-emulation-reinforcement-learning
- PixelBrax arXiv: https://arxiv.org/abs/2502.00021
- PyTorch CUDA graphs: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
