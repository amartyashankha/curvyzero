# Confounder Audit

Date: 2026-05-26

Purpose: keep the optimizer lane honest before the next large change.

## Plain Answer

Recent compact Torch helper patches did not prove a speedup. The main risk is
not that the numbers are useless; it is that total wall-clock can move for
reasons unrelated to the code change.

## Current Confounders

1. Action trajectory changes.

   `search_feedback` lets the search result drive the next environment step.
   Tiny search changes can change selected actions, survival length, collision
   pattern, trail occupancy, and therefore env/render/observation cost. If
   `env_action_checksum_total` or `env_trajectory_checksum_total` changes, wall
   time is not a clean speed comparison.

   Use `scripted_random` only as a search-cost side panel. It keeps applied
   environment actions and trajectory checksums comparable while search still
   runs, but replay commits are dropped because the applied actions no longer
   match the search-selected actions. That means it is not a full learner
   denominator.

2. GPU warmup and compile/cache state.

   `torch.compile`, JAX jit, CUDA allocator caches, and Modal worker cold state
   can all move the first part of a run. Rows need enough warmup and should not
   mix first-compile time with measured time unless that is explicitly the
   thing being tested.

3. Bucket drift.

   A patch can lower `probe_sec` while `actor_sec`, `observation_sec`, sample,
   or learner move in the opposite direction. This means total wall is the
   final read, but bucket-level reads need repeated/matched rows before a claim.

4. Search semantics.

   Compact Torch PUCT-like search, LightZero CTree, fixed-shape first-legal,
   and MCTX/Gumbel do not choose the same actions. MCTX can be a speed
   architecture signal without being a drop-in training backend yet.

   The smallest honest MCTX gate is a fixed-root-tape comparator: same
   `CompactRootBatchV1` sequence, same checkpoint, same masks, same
   simulations, root noise off, enough warmup, then compare actions, visit
   policy, root values, legality, and transfer/timing bytes. If it does not
   match closely, MCTX is an explicit algorithm-change candidate, not a CTree
   replacement.

5. Speed currency mixing.

   Profile-only rows with `calls_train_muzero=false` are useful for optimizer
   decisions. They are not Coach training throughput claims.

6. Root noise and stochastic settings.

   The profile manifest can inherit LightZero root noise unless it is forced
   off. Root noise can change selected actions and later trajectory. Every
   timing row must report root noise and stochasticity state before it is used
   as evidence.

7. Summary bucket naming.

   Labels like `h2d_sec` can include input preparation or synchronization,
   depending on the path. Prefer byte counters and path-specific telemetry over
   the short summary label when deciding whether a host/device boundary is
   really gone.

## External Pattern Notes

- MCTX is designed as JAX-native batched MCTS/MuZero/Gumbel MuZero search. Its
  README says search operates on batches and supports JIT compilation so it can
  use accelerators.
  https://github.com/google-deepmind/mctx

- OpenSpiel's AlphaZero docs say the faster C++ path supports batched
  inference, shared cache, threads, and GPU inference/training, while the
  simpler Python path does not batch inference and stays CPU-bound.
  https://openspiel.readthedocs.io/en/latest/alpha_zero.html

- PufferLib's performance notes emphasize static memory, no repeated tensor
  allocation, CUDA graph replay, vectorized rollout buffers, asynchronous
  pinned transfers, and fused small kernels.
  https://puffer.ai/docs.html

- Treant's MCTS architecture docs describe a batched evaluator bridge: search
  threads queue leaf evaluations, a collector batches them, and the GPU handles
  the neural network batch.
  https://mcts.dev/docs/concepts/architecture/

- NVIDIA CUDA graph guidance is a warning for this lane: graph/capture wins
  require stable allocations and stable addresses. Fresh tensors, dynamic
  shapes, or hidden allocation in the captured path can erase the benefit.
  https://docs.nvidia.com/dl-cuda-graph/latest/cuda-graph-basics/constraints.html

## What To Investigate Next

Run rows that answer one question at a time:

1. Fixed-root-tape gate.

   Save a deterministic sequence of root batches, then run compact Torch,
   direct/CTree if available, MCTX/JAX shadow, and no-search/fixed-shape over
   the same roots. This removes trajectory confounding completely.

   Minimal local seam:

   - capture at
     `src/curvyzero/training/compact_rollout_slab.py::CompactRolloutSlab.step`,
     immediately after `build_compact_root_batch_v1(...)` and before
     `search_service.run(...)` or `run_action_step(...)`;
   - replay through the common
     `src/curvyzero/training/compact_search_service.py::CompactSearchServiceV1.run`
     contract;
   - concrete replay candidates are `CompactTorchSearchServiceV1.run`,
     `MctxCompactSearchServiceV1.run`,
     `_LightZeroCollectForwardCompactSearchService.run`, and
     `FixedShapeBatchedSearchOwnerV0.run`.

   Required metrics:

   - action match fraction;
   - visit-policy L1 distance;
   - root-value absolute difference;
   - legality failures;
   - active-root counts;
   - model/search/readback/input-prep timings;
   - observation/action-mask H2D bytes and replay/action D2H bytes.

   Implementation status:

   - root-tape capture and replay are wired into the compact slab profile path;
   - compare mode is profile-only, `calls_train_muzero=false`, and
     `touches_live_runs=false`;
   - compare mode now fails closed on hidden resident host fallback, one-service
     comparisons, service/root identity mismatch, or comparator exceptions;
   - MCTX is locally wired as an independent sidecar service label for the next
     fixed-root tape row;
   - the corrected MCTX sidecar row ran remotely and showed large same-root
     semantic deltas versus compact Torch, so MCTX is not a parity backend yet;
   - direct CTree remains deferred until the model identity is clean enough for
     a backend-only comparison;
   - resident root-tape compare remains intentionally unsupported until a real
     device-to-host root snapshot is wired.

2. Controlled-action profile.

   Same compact denominator, `scripted_random`, same seeds/checksums. This
   isolates backend overhead from policy-induced trajectory changes.

3. Search-feedback baseline.

   Same denominator, `search_feedback`. This captures real closed-loop behavior
   but must be read with action and trajectory checksums.

4. Search-service-only row.

   Exercise compact search cost without actor/env/observation refresh so we can
   decide whether MCTX/custom search is still the right next move.

5. Long-warmup repeat.

   Repeat the current best strict row with more warmup or duplicate rows before
   claiming small percentage differences.

6. MCTX real-model gate.

   Only compare MCTX as a candidate if the row uses the real checkpoint shadow
   model, reports backend coverage, and is labeled as MCTX/Gumbel semantics.

7. No-search floor and no-refresh ceiling.

   Keep these as ceilings only. They answer how much wall remains after search
   or observation refresh is artificially removed; they do not define a legal
   training setup.

## Current Recommendation

Do not keep chasing tiny compact Torch helper rewrites. They have been tested
and mostly falsified. The next credible speedup is either:

- a real batched/JIT search backend behind the compact contract, proven on a
  fixed-root tape before any training claim; or
- a larger actor/env/observation ownership change that removes host/object
  boundaries and is validated against matching action/trajectory checksums.

Before implementing either, run a small confounder-controlled grid so the next
patch has a clean target.
