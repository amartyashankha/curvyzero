# External Fast RL Patterns, 2026-05-23

Status: research-only optimizer note. Network was available; web sources were
checked on 2026-05-23. No live runs, Modal state, checkpoints, evals, GIFs, or
trainer defaults were touched.

Local grounding:

- `current_state_audit_20260523.md`: trusted Coach lane is stock LightZero;
  compact/MCTX/search-service work is profile-only.
- Latest same-denominator local read: compact Torch is roughly direct speed to
  about `1.4x` faster than direct CTree in one no-noise row; service-tax probes
  show `1.7x-2.5x` boundary headroom; current eager Torch tree/recurrent loop
  is not a 5x/10x win by itself.

## Short Answer

Most production-ish AlphaZero/MuZero systems do **not** put the full tree on
GPU. They keep tree ownership/control in CPU or C++ and batch neural inference
on GPU. True accelerator-native search exists, but it usually requires an
array-native model/search stack, not a PyTorch model bridged through CPU objects.

The practical CurvyTron direction is:

```text
fixed-shape compact env/state buffers
-> CompactRootBatchV1 [B,P,4,64,64] + [B,P,3] masks
-> one search service returning action/visits/value arrays
-> compact replay/RND/final-observation rows
-> LightZero scalar objects only at validation or learner-materialization edges
```

The realistic 5x-10x path is not "GPU MCTS" alone. It is removing whole hot
contracts: scalar env timesteps, Python action dicts/lists, per-simulation
GPU->CPU tree sync, replay object materialization, and one-batch synchronous
search cadence.

## 1. Do Systems Put MCTS On GPU?

| System | What actually happens | CurvyTron read |
| --- | --- | --- |
| LightZero CTree | Tree `batch_traverse` and `batch_backpropagate` are C++, but each simulation builds Torch tensors, runs `model.recurrent_inference`, detaches outputs to CPU NumPy, converts to Python lists, then calls C++ backprop. Source: [LightZero CTree docs/source](https://opendilab.github.io/LightZero/_modules/lzero/mcts/tree_search/mcts_ctree.html). | This validates current diagnosis: the middle is C++, but the boundary is CPU/list/Python. Rewriting "CTree in C++" is not enough. |
| EfficientZero | Ray architecture with self-play actors, replay, CPU rollout/context workers, GPU batch workers, and learner. MCTS atomic work is C++; neural inference is Python/PyTorch; Cython bridges them. Source: [supplement](https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material), [repo](https://github.com/YeWR/EfficientZero). | Strong evidence for CPU/C++ tree plus batched GPU inference as a conservative trainer shape. It also exposes the per-simulation sync ceiling. |
| MiniZero | Server + self-play workers + optimizer + storage. Each self-play worker maintains multiple MCTS instances, collects leaf nodes across games, then runs batch GPU inference. Source: [MiniZero README](https://github.com/rlglab/minizero). | Copy the many-roots/leaf-batching pattern, not the board-game API assumptions. |
| OpenSpiel AlphaZero | Python version has no inference batching and CPU-only. C++/LibTorch version uses threads, shared cache, batched inference, and GPU inference/training. Source: [OpenSpiel AlphaZero docs](https://openspiel.readthedocs.io/en/latest/alpha_zero.html). | Actor/search/learner/replay split is good; scalar game-state API is not the speed path. |
| KataGo analysis engine | Many positions/games can be analyzed in parallel; tuning exposes MCTS threads per position and analysis threads for positions in flight; GPU benefit comes from cross-position NN batching and cache. Source: [KataGo Analysis Engine](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md). | Useful analogy for a local search service: keep many row-seat roots in flight so the model/search service sees real batch fill. |
| MCTX/JAX | Search is JAX-native, batched, JIT-compatible, and accelerator-friendly. Tree arrays are dense `[B,N]` and `[B,N,A]`; recurrent function is inside the JAX search loop. Sources: [MCTX README](https://github.com/google-deepmind/mctx), [tree arrays](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py), [search loop](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/search.py), [policies](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/policies.py). | This is the clean device-resident reference. It is not a drop-in LightZero backend unless the recurrent model and inputs are JAX/device-native. |
| GPU-MCTS research | There are full/tensor GPU MCTS papers, but DNN-MCTS papers still often report CPU tree + accelerator DNN batching; one recent adaptive CPU/GPU DNN-MCTS paper reports `1.5x-3x`, not 10x from tree parallelism alone. Sources: [tensor GPU MCTS](https://www.mdpi.com/2076-3417/13/3/1406), [adaptive DNN-MCTS](https://arxiv.org/abs/2310.05313). | Custom full-GPU search is plausible only after fixed shape and compact ownership are proven. It is a high-risk backend, not the first trainer-facing move. |

Answer: mainstream systems mostly batch NN inference while tree control stays
CPU/C++/threaded. MCTX is the important exception because it changes the whole
model/search contract to arrays on the accelerator.

## 2. Patterns That Avoid Sync And Object Churn

1. **Static SoA buffers, not object forests.** PufferLib allocates contiguous
   memory up front and does not create/reallocate tensors in the hot loop; env
   observations/actions/rewards/terminals are big contiguous chunks across many
   instances. Sources: [PufferLib docs](https://puffer.ai/docs.html),
   [older vectorization docs](https://pufferai.github.io/build/html/rst/landing.html).

2. **Chunked vector envs and async transfer.** PufferLib chunks env instances
   into buffers, uses rollout workers on CUDA streams, pinned memory, and async
   transfer. EnvPool gives the CPU-side version: C++ batched env pool, async/sync
   APIs, C++ developer API, native batched pixels, and much faster throughput
   than Python subprocess vector envs. Sources: [EnvPool repo](https://github.com/sail-sg/envpool),
   [EnvPool paper](https://arxiv.org/abs/2206.10558).

3. **One compact search service call.** The service boundary should consume:

   ```text
   obs_uint8[B,P,4,64,64]
   legal_mask[B,P,3]
   row_id[B], player_id[P or B,P]
   optional recurrent/embedding/state sidecars
   ```

   and return:

   ```text
   action[B,P]
   visit_counts[B,P,3]
   root_value[B,P]
   policy_version/search_config ids
   ```

   No LightZero `BaseEnvTimestep`, per-env dicts, `roots.get_distributions()`
   fanout, or full GameSegment materialization in collect.

4. **Keep compiled loops closed.** MCTX works because `RootFnOutput`,
   `recurrent_fn`, tree arrays, invalid masks, expansion, and backup stay inside
   JAX. The local trap is `torch -> numpy -> jax -> numpy -> torch`; that gives
   the appearance of GPU search while preserving the bad boundary.

5. **Batch across roots/leaves, not just across env step.** MiniZero, KataGo,
   OpenSpiel C++ and EfficientZero all use some variant of many actors/positions
   in flight, shared/cache/batch inference, and queued replay/learner work. This
   is how they avoid one tiny recurrent call per root starving the GPU.

6. **Defer rich replay objects.** Store compact rows with identity:

   ```text
   root_obs_ref, legal_mask, selected_action, visit_policy, root_value,
   reward, done, final_observation_ref, row_id, player_id,
   policy_version, search_version, rnd_frame_ref
   ```

   Materialize LightZero-style learner batches only at sampler/validation edges.

## 3. What Translates To CurvyTron/LightZero

Clean translations:

- Puffer-style contiguous buffers map directly to `HybridCompactBatch` /
  `CompactRootBatchV1`: `[B,P,4,64,64]`, `[B,P,3]`, rewards, done/final,
  row/player ids, autoreset sidecars.
- MCTX maps cleanly as an **architecture ceiling** and profile-only sidecar:
  dense tree arrays, fixed `A=3`, invalid masks, recurrent function, compact
  `action_weights`. It only becomes trainer-facing if the model/search boundary
  is device-native and replay parity is proven.
- LightZero CTree remains the compatibility comparator. It is useful behind
  `CompactSearchServiceV1` as the "real semantics" backend while array-native or
  MCTX-style backends are developed.
- MiniZero/KataGo actor batching maps to `B * P` row-seat roots and eventually a
  many-producer search service. CurvyTron has natural fixed action size and many
  simultaneous roots.
- Gumbel MuZero is worth testing because `A=3` and local overhead makes lower
  simulation count valuable. Source: [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO).

Traps:

- "Just use MCTX" while keeping a PyTorch LightZero recurrent model outside the
  JAX loop. That only moves the sync point.
- "Just more C++." LightZero/EfficientZero already use C++ for tree atomic work;
  the costly part is the boundary around it.
- "Just renderer/GPU observations." Local closed-loop evidence says observation
  handoff matters, but renderer-only work is bounded unless it also removes
  stack ownership, root packaging, and action/result materialization.
- Full PufferLib trainer replacement. PufferLib is PPO/V-trace-like, not MuZero
  with search targets. Steal its buffer contract, not its learner.
- Gym/PettingZoo compatibility as the optimized path. Good for testing, bad for
  removing scalar wrapper churn.
- More actors/CPUs without ordered compact identities. Throughput can rise while
  replay row attachment, terminal final observation, learner seat, or RND frame
  silently drift.

## 4. 5x-10x Upside Versus 1.3x-2x

Likely only `1.3x-2x`:

- Array-native CPU CTree/list removal while recurrent outputs still return to
  CPU each simulation.
- Pinned host transfer, dtype tweaks, `.tolist()` avoidance, or smaller
  per-step allocations alone.
- Replay-row deferral alone.
- Renderer-only optimization without resident stack/search input ownership.
- More CPU workers around the current scalar LightZero collector.
- Dense eager Torch search polishing without compile/static loop residency.

Plausible `2x-3x`:

- One compact service denominator where direct CTree, service-tax, mock, and any
  candidate backend all consume/return the same compact arrays.
- Device-resident observation stack feeding search input, with host copies only
  for parity sampling or final action commit.
- Compact replay owner that prevents hot GameSegment/materialization fanout.
- Low-sim Gumbel search if policy quality survives at `sim4-sim8`.

Realistic `5x-10x` only as an architecture change:

```text
many compact env/root producers
-> fixed-shape search/inference service
-> model/search loop device-native or at least queue-batched with no per-sim D2H
-> compact replay/RND owner
-> learner materialization at coarse boundaries
```

Two viable versions:

1. **Conservative 5x candidate:** CPU/C++ compact env + compact replay + many
   row-seat roots in flight + service-batched recurrent inference/tree work. This
   is MiniZero/KataGo/EfficientZero-shaped. It can keep LightZero semantics
   longer, but must stop rebuilding scalar collect objects.

2. **Radical 5x-10x candidate:** MCTX/JAX or equivalent fixed-shape GPU search
   with resident observations/embeddings and compact replay. This is the cleanest
   accelerator-native target, but it requires a JAX-native recurrent model or a
   custom Torch/Triton-style search body. A PyTorch-JAX bridge is a trap.

Decision rule for CurvyTron:

```text
If a change only shortens one timer inside the existing scalar LightZero loop,
expect 1.3x-2x.

If it removes a whole host/object/device boundary and keeps row/player/replay
identity compact through the next env step, expect 2x+.

If it keeps many roots in flight and prevents per-simulation CPU/GPU sync,
then 5x+ becomes credible.
```

## Source Links

- MCTX: [repo](https://github.com/google-deepmind/mctx), [base types](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/base.py), [tree arrays](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py), [search loop](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/search.py), [policies](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/policies.py)
- LightZero: [repo](https://github.com/opendilab/LightZero), [CTree source docs](https://opendilab.github.io/LightZero/_modules/lzero/mcts/tree_search/mcts_ctree.html)
- EfficientZero: [supplement](https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material), [repo](https://github.com/YeWR/EfficientZero)
- MiniZero: [repo/README](https://github.com/rlglab/minizero)
- OpenSpiel AlphaZero: [docs](https://openspiel.readthedocs.io/en/latest/alpha_zero.html)
- KataGo: [parallel analysis engine](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)
- PufferLib: [current docs](https://puffer.ai/docs.html), [vectorization/emulation docs](https://pufferai.github.io/build/html/rst/landing.html)
- EnvPool: [repo](https://github.com/sail-sg/envpool), [paper](https://arxiv.org/abs/2206.10558)
- Gumbel MuZero: [OpenReview](https://openreview.net/forum?id=bERaNdoegnO)
- Tensor GPU MCTS: [Applied Sciences](https://www.mdpi.com/2076-3417/13/3/1406)
- DNN-MCTS adaptive parallelism: [arXiv](https://arxiv.org/abs/2310.05313)
