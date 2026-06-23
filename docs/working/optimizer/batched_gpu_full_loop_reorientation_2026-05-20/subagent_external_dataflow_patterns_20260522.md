# External Dataflow Patterns For The Current Compact Loop

Date: 2026-05-22

Status: research-only optimizer note. No source code, live runs, checkpoints,
Modal jobs, evals, GIFs, or tournaments were touched.

## Short Read

The refreshed external read agrees with the latest local denominator: the next
large win is not another isolated renderer patch and not "make CTree C++" in
the abstract. The live question is ownership:

```text
compact env state
-> observation/stack/root batch
-> batched search
-> selected action readback
-> replay/search payload commit
-> learner/RND sample edge
```

After the root-value extraction fix, the bottleneck read should shift away from
"can we cheaply see the search result?" and toward "can the next tick consume
the result without rebuilding or rereading the world?" Selected actions can come
back early; replay-valid policy/root-value payloads still have to be committed
somewhere before the row is sample-visible.

Fast RL and self-play systems keep many roots/envs alive, batch model/search
work, keep hot payloads in contiguous arrays, and defer scalar framework objects
to validation or logging edges. PufferLib phrases this as static contiguous
memory, CUDA graph-friendly work, chunked env buffers, pinned async transfers,
and no reallocation in the hot loop ([PufferLib docs](https://puffer.ai/docs.html)).
MCTX phrases the search side as JAX-native, JIT-compiled, batched MCTS over
root arrays ([MCTX README](https://github.com/google-deepmind/mctx)).

For CurvyTron now: selected action readback is allowed because it is tiny and
needed by the CPU env step. Reading back full observations or per-simulation
reward/value/policy payloads just to rebuild the next GPU input is the pattern
to attack.

## Local Docs Reviewed

Most relevant local notes:

- `README.md` in this packet: trusted Coach-facing training is still stock
  LightZero; compact GPU/MCTX/CTree lanes are profile-only.
- `current_hot_path_bottleneck_map_20260522.md`: direct CTree gave a real
  `~1.28x-1.31x` stock-loop profile win, but the remaining wall is the
  LightZero collect/search topology boundary.
- `subagent_full_iteration_dataflow_20260522.md`: compact closed-loop rows have
  moved the wall to next-search-input preparation, mostly observation/stack/root
  handoff rather than raw CurvyTron mechanics or raw GPU drawing.
- `puffer_style_contiguous_buffer_attach_audit_20260522.md`: the pre-scalar
  `HybridCompactBatch` sidecar is the right attach point; scalar LightZero
  timestep materialization is already too late for a speed path.
- `subagent_action_only_hypothesis_critique_20260522.md`: action-only readback
  is only meaningful if visit-policy/root-value payloads are deferred and later
  committed replay-valid, not deleted.
- `subagent_device_resident_observation_boundary_20260522.md`: current repeated
  compact MCTX flow still double-bounces GPU render to host stack and back to
  device for search.
- Older framework notes under `architecture_reexploration_2026-05-12/`:
  EfficientZero/Ray, MCTX/JAX, MiniZero, and large-scale Zero-style systems all
  point toward actor/search/replay/learner separation with batched inference.
- Training literature notes under `docs/working/training/...` mostly warn about
  replay target semantics, `to_play`, action masks, stale checkpoints, and
  simultaneous-action row identity. Those constraints must survive any compact
  speed lane.

I found no local note where Sampled MuZero is a direct implementation target
for CurvyTron's current `A=3` discrete action problem. It is still useful as a
negative example: sample actions when enumeration is infeasible, not when the
action space is already tiny.

## Patterns That Apply To Us Now

1. **Puffer-style contiguous ownership.** CurvyTron's current compact sidecar
   already has the right shape: `[B,P,4,64,64]` observations, `[B,P,3]` masks,
   rewards, dones, row/player ids, terminal/final-observation/autoreset facts,
   and search/replay sidecars. Keep extending that owner instead of rebuilding
   per-env dicts and `BaseEnvTimestep` objects.

2. **MCTX-style device search shape.** MCTX's core types are exactly the
   target ABI: `RootFnOutput` has prior logits `[B,A]`, value `[B]`, embedding
   `[B,...]`; the recurrent function returns reward/discount/prior/value; and
   `PolicyOutput` returns action `[B]` plus action weights `[B,A]`
   ([MCTX base types](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/base.py)).
   The tree stores dense arrays over `[B,N]` and `[B,N,A]`, including invalid
   action masks and embeddings ([MCTX tree](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py)).

3. **Batch roots/leaves, not one scalar decision.** MiniZero explicitly keeps
   multiple MCTS instances per self-play worker and evaluates leaves through
   batched GPU inference ([MiniZero README](https://github.com/rlglab/minizero)).
   KataGo's analysis engine makes the same batching point for many positions
   in flight ([KataGo analysis engine](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)).

4. **Use stock LightZero as a semantic control, not the hot owner.** LightZero
   already uses C++ for `batch_traverse` and `batch_backpropagate`, and batch
   search helps parallel model inference ([LightZero tree docs](https://opendilab.github.io/LightZero/api_doc/mcts/tree_search/index.html)).
   Our remaining wall is around that CTree core: CPU/list root prep,
   per-simulation host-visible payloads, Python control, dict outputs, replay
   objects, and RND side paths.

5. **Separate actor/search/replay/learner roles.** OpenSpiel's AlphaZero docs
   contrast a Python implementation without inference batching against a C++
   path with threads, shared cache, batched inference, and GPU support
   ([OpenSpiel AlphaZero](https://openspiel.readthedocs.io/en/latest/alpha_zero.html)).
   EfficientZero's supplement uses self-play data workers, CPU context workers,
   GPU batch workers, replay, queues, and a learner to keep CPU/GPU stages busy
   ([EfficientZero supplement](https://openreview.net/attachment?id=OKrNPg3xR3T&name=supplementary_material)).

6. **Small action readback is fine.** EnvPool's async API splits action send
   from result receive and tracks env ids in batches ([EnvPool README](https://github.com/sail-sg/envpool)).
   For us, reading selected actions `[B,P]` back to CPU before env step is not
   the sin; forcing full visual stacks or recurrent search payloads through CPU
   just to feed the next device consumer is.

7. **Centralized service patterns apply, but as search services.** SEED RL
   centralizes inference/training on the learner to use accelerators efficiently
   ([SEED RL paper page](https://research.google/pubs/seed-rl-scalable-and-efficient-deep-rl-with-accelerated-central-inference/)).
   MuZero needs more than inference service: the central owner must understand
   search, root identity, visit policy, root value, and replay commit.

## Patterns That Likely Do Not Apply

- **Sampled MuZero as the next speed fix.** Sampled MuZero targets
  high-dimensional or continuous action spaces where full action enumeration is
  infeasible ([PMLR paper](https://proceedings.mlr.press/v139/hubert21a.html)).
  CurvyTron currently has `A=3`; our problem is dataflow/control overhead, not
  action branching.

- **"Just use MCTX" as a LightZero patch.** MCTX only pays off when the model,
  recurrent function, tree, and masks stay JAX/compiled. Bridging PyTorch
  LightZero into JAX would recreate the same host boundary in fancier clothes.

- **More CTree C++ as the main thesis.** LightZero CTree is already C++ in the
  center. The expensive part is the surrounding ABI and object topology.

- **Full GPU CurvyTron env rewrite as the immediate next step.** Isaac/Brax-like
  same-device env/search/learner is a valid long-term architecture, but current
  profile evidence points at a smaller falsifier first: remove the render/stack
  double bounce and keep the CPU env with tiny action readback.

- **Action-only throughput as replay throughput.** If action-only skips
  visit-policy/root-value extraction and replay-index construction, it is a
  ceiling probe, not a valid collection loop.

- **Turn-based scalar trajectory assumptions.** CurvyTron must keep `[row,
  player]` provenance and commit one joint action per physical row. Per-seat
  independent `A=3` search is compatible with current semantics; centralized
  `A=9` joint search is a separate algorithmic control.

- **Go-like transposition/NN caches as the main bet.** Cross-position batching
  applies. Repeated-state caching is less obviously valuable for visual
  CurvyTron source-state trajectories.

## Five Concrete Experiments

1. **Device-resident observation stack for repeated compact MCTX.**
   Feed MCTX from `renderer.last_output_device` plus a resident JAX FIFO stack
   instead of `GPU render -> np.asarray -> host stack -> jax.device_put`.
   Keep host `HybridCompactBatch` for masks, ids, final-observation validation,
   and replay-index proof. Pass if closed-loop roots/sec improves by `1.2x+`
   with zero stack/replay identity drift and `obs_h2d_bytes=0`.

2. **Selected-action-first, replay-valid deferred payload.**
   Read only selected actions before CPU env step, but drain action weights and
   root values from the same search output before committing replay indices.
   Compare full baseline, action-only ceiling, immediate payload drain, and
   one-step deferred replay drain. Keep only if the replay-valid drain remains
   materially closer to action-only than to the full baseline on total wall.

3. **Precomputed recurrent-output split inside the compact service.**
   In the same compact service denominator, compare real recurrent inference
   against resident synthetic reward/value/policy tensors, and compare current
   list CTree against flat-A3 where available. This tells us whether the next
   wall is model launch/batching, CTree/list ABI, or surrounding control.

4. **Compact replay/RND tensor ring with learner stub.**
   Store `CompactReplayIndexRowsV1` plus search arrays during collection,
   materialize learner-shaped tensors only at sample time, and slice RND latest
   frames from the resident/latest stack before CPU materialization. Measure
   replay-index write, sample materialization, learner H2D/batch construction,
   RND train/estimate/metrics, and terminal final-observation correctness.

5. **Producer/consumer compact search service mock.**
   Run multiple CPU compact env batches feeding one fixed-shape search worker
   through bounded queues:

   ```text
   env actors -> compact root queue -> search service -> selected actions
              -> compact replay queue -> learner/replay stub
   ```

   Start with mock legal visit policies, then direct CTree, then MCTX. Report
   queue wait, GPU/search utilization, action round-trip latency, replay commit
   cost, checkpoint/model id carried per chunk, and stale-data distribution.
   This prices whether topology separation itself buys headroom before we build
   a full service.

## Recommendation

The next serious optimizer lane should be:

```text
resident/contiguous compact batch ownership
-> selected-action-first CPU env step
-> replay-valid delayed search payload commit
-> compact replay/RND/learner sample edge
```

Keep stock LightZero as the semantic oracle and adapter. Do not treat it as the
hot owner for the 5-10x lane unless fresh same-denominator profiles prove the
compact path has no real ceiling.
