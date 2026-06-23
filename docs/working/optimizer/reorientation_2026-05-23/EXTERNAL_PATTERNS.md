# External Patterns

Date: 2026-05-23

Purpose: record what fast RL/MuZero-style systems suggest, without pretending
they are drop-in fixes.

## Shared Pattern

Fast systems tend to avoid making Python objects the owner of hot training
data.

The common shape is:

- many environments write into fixed arrays;
- policy/search consumes arrays in large batches;
- replay stores compact references first;
- learner expands only what it needs;
- slow logging/eval/checkpoint work is outside the hot loop;
- host/device sync happens at explicit coarse points.

That is the same idea we are calling compact ownership.

## What This Means For CurvyTron

The stock LightZero path is still the semantic control path. It is trusted
because it owns the real training logic.

The optimizer path should not fork the whole algorithm casually. It should first
try to move the hot data boundary:

```text
env arrays -> observation arrays -> search arrays -> action arrays
          -> compact replay refs -> learner payloads
```

Scalar LightZero rows can still exist, but they should not be rebuilt in the
inner loop if we are trying for a large speedup.

## Useful Ideas To Copy

- Puffer-style slab ownership: fixed arrays for env state, action, reward,
  done, observation, and stats.
- Isaac-style vectorized simulation: keep mechanics/observations/rewards close
  to the accelerator-facing tensor surface and cross host/device boundaries at
  coarse, explicit points.
- EnvPool/Sample-Factory-style actor flow: keep collection moving and batch
  work before crossing expensive boundaries.
- MiniZero/AlphaZero-style self-play/search separation: many roots in flight,
  with search work batched and replay decoupled.
- MCTX/JAX-style dense search: fast if the model/search/checkpoint path is
  array-native, but not a drop-in if semantics differ.
- EfficientZero/Ray-style distributed roles: actor, replay, learner, and
  checkpoint ownership are explicit.

## 2026-05-26 Research Refresh

The external pattern is consistent with our latest confounder audit:

- MCTX is the clean "all array, all compiled" example. It runs batched
  MuZero/Gumbel MuZero search in JAX and is meant to use JIT/accelerators. That
  explains why it can be a useful ceiling. It also explains why a
  PyTorch-to-NumPy-to-JAX bridge is not the real win.
  https://github.com/google-deepmind/mctx
- PufferLib is the clean "static memory and explicit rollout buffers" example.
  Its docs emphasize single contiguous allocation, no hot-loop tensor
  reallocation, CUDA graph replay, pinned async transfers, and environment
  batches as buffers. That maps directly to our compact ownership goal.
  https://puffer.ai/docs.html
- OpenSpiel's AlphaZero docs draw the same boundary in a simpler way: the
  Python path is easy but CPU/non-batched; the faster C++ path uses threads,
  shared cache, batched inference, and GPU inference/training.
  https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- LightZero CTree is useful and trusted, but its documented Python boundary
  still detaches tensors to CPU/NumPy around tree work. That is the boundary we
  are trying to escape in the compact lane.
  https://opendilab.github.io/LightZero/_modules/lzero/mcts/tree_search/mcts_ctree.html
- CUDA graphs are relevant only after the shapes and allocations are stable.
  They should be treated as a follow-up optimization, not the first correctness
  gate.
  https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html

Practical read: the next big move is not another small helper rewrite. It is a
fixed denominator plus one of two architecture probes:

- fixed-root-tape search backend comparison, to see whether a dense compiled
  search owner or MCTX-style owner is worth promoting; or
- compact actor/env ownership, if no-search rows prove search is no longer the
  wall.

## 2026-06-07 Read After Fixed-SoA H100 Rejections

The latest CurvyTron rows reinforce the external pattern instead of weakening
it. Fixed-SoA exact, locality, and slot-candidate paths can be made
proof-clean, but they do not move enough ownership:

- fixed-SoA exact/locality stayed below OPT-104;
- slot-candidate fixed SoA ran on H100 at `10348.87 env/s`, only `0.816x`
  OPT-104 and `0.653x` the columnar r2 support row;
- the row still spent broad wall in owner/sample/train/search surfaces, and
  fixed-SoA successor-index work alone was about `4.080s`.

This matches Puffer/Isaac-style systems: the speed comes from fixed buffers,
static allocation, vectorized env/mechanics, and explicit host/device
boundaries as a single ownership design. It does not come from repeatedly
rewriting one learner gather path while the rest of the loop remains
Python/object/ownership fragmented.

## Practical Ranking For This Repo

| Rank | Option | Why |
| ---: | --- | --- |
| 1 | Flat CTree / CPU-C++ batched evaluator API | Best semantic promotion path if it preserves LightZero-like search while removing Python/list boundaries. Needs parity proof first. |
| 2 | Vectorized env / compact actor ownership | Biggest whole-loop lever after stock object flow is removed. Needs terminal/autoreset/RND/replay identity tests. |
| 3 | Custom dense Torch search owner | Fast local experiment using the PyTorch model and current compact contracts. Risk: Python simulation loop and imperfect PUCT parity. |
| 4 | MCTX/JAX | Best theoretical array-native search body, but profile-only until model/checkpoint and Gumbel-vs-CTree semantics are accepted or proven close. |
| 5 | CUDA graph / compile | Useful after fixed shapes and stable allocations; not a topology fix by itself. |

First proof gate for all search-backend work is the fixed-root-tape comparator.
First proof gate for whole-loop work is the frozen-denominator compact-owned
trainer candidate with honest `calls_train_muzero=false`.

## What Not To Copy Blindly

- Do not replace LightZero search with MCTX just because it is faster unless we
  accept the algorithm change or prove the distributions are close enough.
- Do not scale actor count blindly if the learner/RND/replay path cannot digest
  the data.
- Do not confuse more self-play samples per wall clock with better learning per
  wall clock until tournament/eval results confirm it.
- Do not optimize only render now that render is no longer the whole wall.

## Implication

If the goal is `5x-10x`, small local patches are unlikely to be enough. The
candidate big move is a compact-owned actor/search/replay path with a clear
adapter back to the trusted learner, or a deliberate migration to a different
array-native training stack.
