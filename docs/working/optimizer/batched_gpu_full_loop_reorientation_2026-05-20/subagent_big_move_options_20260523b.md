# Big Architecture Move Options

Date: 2026-05-23

Role: parallel architecture-options critique agent. Scope is documentation
only. I did not touch source code, live training runs, Modal state, checkpoints,
evals, tournaments, GIFs, or volumes.

## Short Read

Do not promote the current eager compact Torch search body. It usefully proved
the compact service boundary, but the latest read is that the remaining hot work
is the Python/Torch tree plus recurrent loop, and eager compact search is not a
clean whole-loop breakthrough versus direct CTree.

The profile signal still says there is real headroom:

```text
direct_ctree_gpu_latent = practical LightZero/CTree comparator
service_tax_probe      = real model work, fake-ish search/control ceiling
mock_search_service    = fake-search/dataflow ceiling
```

On the common compact profile rows, service-tax/mock are far above direct CTree,
especially at higher simulation counts. That argues for an architecture move at
the search/dataflow boundary, not more wrapper polishing.

Balanced recommendation:

1. First try a real compiled/fused fixed-shape MCTS backend behind
   `CompactSearchServiceV1`.
2. In parallel, harden the contiguous compact slab so the loop has one owner for
   row/player/search/replay identity.
3. Keep fixed-`A=3` array-native CTree as the conservative compatibility bridge.
4. Use JAX/MCTX as the clean accelerator-native comparator, not trainer advice.
5. Move replay/RND payloads under compact ownership before any trainer claim.
6. Defer many-producer service architecture until the compact contracts are
   boring.
7. Keep stock tuning as the baseline/control, not the main bet.

## Ranking Matrix

Rank is a balanced priority: expected whole-loop upside, implementation risk,
and validation ease together.

| Rank | Move | Expected whole-loop upside | Implementation risk | Validation ease | Read |
| ---: | --- | --- | --- | --- | --- |
| 1 | Compiled/fused fixed-shape MCTS | High: plausible `1.5-3x` on search-heavy compact rows if it reaches service-tax territory | High | Medium | Best near-term shot at turning the search ceiling into a real backend. |
| 2 | Puffer-style contiguous buffer/vector slab | Medium-high: `1.2-2x` alone; enables larger wins | Medium | Easy-medium | Removes scalar/object fanout and gives the loop a clean compact owner. |
| 3 | Array-native fixed-`A=3` CTree API | Medium: likely `1.1-1.6x` whole-loop, more in no-model CTree | Medium | Easy-medium | Conservative CTree-compatible bridge; capped but clean. |
| 4 | JAX/MCTX comparator | High as architecture evidence; trainer upside only after framework migration | High | Easy as sidecar, hard as production | Cleanest all-device search reference; do not bridge PyTorch into JAX hot loops. |
| 5 | Compact replay/RND ownership | Medium: `1.05-1.3x` alone; required for any bigger win | Medium-high | Medium-hard | Prevents replay/RND/materialization from reintroducing the sync wall. |
| 6 | Many-producer batched search service | Very high endpoint: possible `2-6x+` if GPU batch fill is real | Very high | Hard | Real 5x architecture, but only after identity/replay contracts are stable. |
| 7 | Do nothing except stock tuning | Low: `0-1.2x` unless a missed config knob exists | Low | Easy | Necessary control; insufficient if service-tax/mock headroom remains. |

## 1. Compiled/Fused Fixed-Shape MCTS

What changes:

- Keep the existing compact contract:
  `CompactRootBatchV1 -> CompactSearchServiceV1 -> CompactSearchResultV1`.
- Replace the eager Python/Torch tree body with a fixed-shape compiled body.
- Fix the first denominator: `B=512`, `P=2`, roots `R=1024`, `A=3`,
  `sim=16/32`, padded active roots, `root_noise_weight=0.0`.
- Preallocate tree arrays, path scratch, min/max stats, recurrent outputs, and
  visit buffers.
- Compile or capture the selection, expansion, backup, and output assembly
  path. Start with Torch compile/CUDA graphs; switch to Triton/custom CUDA if
  graph breaks dominate.
- Return only the required compact result arrays:
  `selected_action`, `visit_policy`, `raw_visit_counts`, `root_value`, and
  diagnostics.

Why it could be faster:

- It directly attacks the measured wall: repeated Python/Torch tree control and
  recurrent-loop overhead.
- It avoids per-simulation CPU/list CTree crossings and repeated tiny host
  synchronization.
- The service-tax/mock gap says a real backend that keeps model work but removes
  the search/control tax can plausibly move whole-loop rows, especially at
  `sim32`.

What could go wrong:

- Compiled helpers graph-break back to eager and the row is only relabeled.
- Correctness silently depends on all actions being legal.
- Active-root order, legal-mask polarity, root value backup, or latent-slot
  reuse diverges from direct CTree.
- It wins `sim16` but regresses `sim32`.
- It only wins by omitting required readback into `CompactSearchResultV1`.

Smallest falsifying experiment:

```text
H100 profile-only common denominator:
  B512/P2/A3, host_uint8 roots, sim16 and sim32, root_noise_weight=0.0
  direct_ctree_gpu_latent
  service_tax_probe
  mock_search_service
  compiled_fixed_shape_mcts

Required gates:
  active-root order preserved
  mixed masks and single-legal masks pass
  zero illegal selected actions and zero illegal visit mass
  closed compact loop action-feedback proof passes
  CompactReplayIndexRowsV1 materializes to trusted rows
```

Kill or demote if warmed `compiled_fixed_shape_mcts` is not at least `1.25x`
faster than direct CTree at `sim16`, or if it loses to direct CTree at `sim32`,
after compile warmup and required readback are included.

## 2. Puffer-Style Contiguous Buffer / Vector Slab

What changes:

- Make one profile-only compact rollout owner around the existing manager:
  `HybridCompactBatch -> CompactRolloutSlab -> search service -> next_joint_action`.
- Store hot data in row-major arrays:
  `obs_stack[B,P,4,64,64]`, `legal_mask[B,P,3]`, `reward[B,P]`,
  `done[B]`, `joint_action[B,P]`, row/player ids, terminal/final-observation
  masks, and RND latest-frame handles.
- Push scalar LightZero objects, dicts, `GameSegment` materialization, and
  target-row construction to validation/sample edges.
- Let selected actions from search become the next env `joint_action` directly.

Why it could be faster:

- It removes actor payload merge, scalar object fanout, per-env dict creation,
  and repeated conversion between compact arrays and public LightZero surfaces.
- Existing zero-observation native actor-buffer rows already showed the actor
  payload/merge boundary can be priced and removed.
- It gives compiled MCTS, array-native CTree, MCTX, replay, and RND the same
  memory layout instead of each option inventing its own bridge.

What could go wrong:

- The slab becomes a second replay system with subtly different semantics.
- Terminal final observations are captured after autoreset.
- Row/player ids drift between root, action, next transition, and replay.
- RND sees a different latest frame than the policy/search row.
- The profile wins only on zero observations and disappears with real render
  state and terminal rows.

Smallest falsifying experiment:

```text
Local/profile-only compact slab smoke:
  B512/P2/A3, steps enough for live and terminal rows
  direct CTree compact service first
  service-selected actions drive next_joint_action
  previous search commits into CompactReplayIndexRowsV1 when next batch arrives
  materialized sample rows match the trusted immediate path
```

Then run two same-shape profiles:

```text
old compact manager + direct CTree
compact slab owner + direct CTree
```

Kill as a speed lane if the slab cannot beat the old compact manager by at
least `20%` on zero-observation rows and `10%` on renderer-backed compact rows
while passing action-feedback, terminal, perspective, and replay checks.

## 3. Array-Native / Fixed-`A=3` CTree API

What changes:

- Vendor or fork the CTree wrapper enough to expose fixed-`A=3` typed-array
  APIs.
- Start from the existing flat path:
  `batch_backpropagate_flat_a3(rewards[B], values[B], policy_logits[B,3])`.
- Extend only if the measured row demands it: flat root prepare, flat traverse
  outputs, and flat distributions/values.
- Preserve CTree semantics and use direct CTree as the oracle.

Why it could be faster:

- It removes Python nested lists, `.tolist()`, and per-row `vector<vector<float>>`
  construction in the hot search loop.
- The no-model CTree gate already showed a real local win:
  about `2.0x` on all-legal fixed `A=3`, and about `1.75x` on mixed `2-of-3`
  masks.
- It is the cleanest compatibility bridge because it does not require changing
  model framework, replay targets, or MuZero search semantics.

What could go wrong:

- The no-model win does not transfer once recurrent inference and output
  extraction are included.
- CTree tie-breaking makes exact parity tests misleading unless fixtures use
  single-legal or clear-preference cases.
- A vendored CTree fork becomes maintenance debt.
- Backprop flattening alone is capped because stock traverse and output
  extraction remain list/object shaped.

Smallest falsifying experiment:

```text
Integrate fixed-A3 backprop into direct_ctree_gpu_latent compact service.
Run H100 B512/P2/A3 sim16 and sim32:
  stock direct_ctree_gpu_latent
  flat_a3_ctree_gpu_latent

Required gates:
  exact forced-mask and single-legal parity
  clear-preference action parity
  no illegal actions
  root values/logits within tolerance
  compact replay/index rows match trusted rows
```

Kill as a big architecture lane if the integrated flat-A3 row does not beat
direct CTree by at least `15%` at `sim16`, or if it regresses `sim32`. Keep it as
a semantic control if it is clean but capped.

## 4. JAX/MCTX Comparator

What changes:

- Build a sidecar comparator that consumes real compact roots:
  `obs_roots[R,4,64,64]`, `invalid_actions[R,3]`, row/player ids.
- Use a tiny pure-JAX representation/prediction/recurrent model and
  `mctx.gumbel_muzero_policy`.
- Keep tree arrays, embeddings, priors, values, visits, rewards, and backups
  inside JAX/JIT.
- Return `action[R]`, `action_weights[R,3]`, and root summaries through the
  compact result shape.
- Do not call the PyTorch LightZero model inside JAX. That would recreate the
  host boundary.

Why it could be faster:

- MCTX is the clean reference for fixed-shape accelerator-resident MuZero search.
- It removes CPU CTree traversal/backprop, Python list APIs, per-simulation
  recurrent-output readback, and scalar root surfaces.
- Existing MCTX profile-only rows already show much higher active-root rates
  than direct CTree-shaped search, but with a toy JAX model.

What could go wrong:

- Framework ownership is a rewrite: PyTorch LightZero weights do not naturally
  live inside jitted MCTX.
- Independent per-seat `A=3` search may be the wrong multiplayer semantics;
  joint `A=9` search is a useful control but changes the branching problem.
- Tree memory can explode if embeddings are spatial instead of vector latent.
- Fixed-shape padding or changing masks can trigger recompiles.
- It proves search mechanics but not replay, learner, RND, checkpoints, evals,
  or tournament compatibility.

Smallest falsifying experiment:

```text
Profile-only MCTX real-root sidecar:
  B512/P2 -> R=1024 roots
  obs uint8/float32 compact roots from the current profile manager
  A=3 invalid-action masks
  hidden H=64 vector latent
  sim16 and sim32
  compile time separate from warmed steady timing
```

Measure visual/root setup, H2D, search, D2H for action only, then D2H for
action weights/root value.

Kill as a near-term architecture lane if warmed
`setup + search + required D2H` is not at least `2x` faster than direct CTree on
the same root/simulation shape, or if the plan requires PyTorch host callbacks
inside the JAX recurrent function. Keep it as an external comparator even if it
is not trainer-facing.

## 5. Compact Replay / RND Ownership

What changes:

- Make compact replay/RND the owner of hot collect records:
  observation handles, action, visit policy, raw visits, root value, reward,
  done, terminal final-observation marker, row/player ids, and RND latest-frame
  references.
- Materialize full LightZero target rows only at validation and learner sample
  edges.
- Split the service sync policy:
  action now, replay payload later.
- RND consumes latest policy frames from the same compact records, not a
  separately reconstructed stack.

Why it could be faster:

- Replay/RND payloads do not have to gate the next env step.
- It removes premature observation and next-observation materialization from
  the collect hot path.
- It prevents a faster search backend from handing the win back through scalar
  target rows, public sample objects, or per-step RND extraction.

What could go wrong:

- Off-by-one transition bugs: search at record `k` trains against record `k+1`
  incorrectly.
- Payload flush attaches visits/root values to the wrong row.
- Learner samples a row before delayed payload is committed.
- Terminal rows use post-autoreset observations or reset RND frames.
- Player 0/player 1 perspective swaps are invisible until learning degrades.
- Positive RND changes the objective and masks a plumbing bug.

Smallest falsifying experiment:

```text
Closed compact replay/RND meter smoke:
  two or three records
  mixed live + terminal/autoreset rows
  both players
  non-prefix active roots
  non-identity policy_env_id
  real compact search result
  rnd_meter_v0 with zero reward weight

Assert:
  selected search action becomes next joint_action
  CompactReplayIndexRowsV1 materializes to trusted immediate rows
  public MuZeroGameBuffer.sample parity holds when lzero is available
  RND predictor changes
  RND target hash stays fixed
  target rewards stay unchanged in meter mode
  latest-frame order matches env_row/player and terminal final frame
```

Kill as a speed lane if compact ownership cannot match trusted replay/sample/RND
rows exactly. If it matches but gives little immediate speed, keep it as a
mandatory platform layer for any compiled or service backend.

## 6. Many-Producer Batched Search Service

What changes:

- Run multiple compact env/slab producers feeding one ordered search/inference
  service.
- The service batches initial and recurrent inference, search work, and payload
  flushes across producers.
- CPU env stepping, GPU search, replay commit, and learner sampling become
  independently busy instead of one synchronous collect/search tick.
- Results return with sequence ids so selected actions and payloads attach to
  the correct producer row/player/record.

Why it could be faster:

- It is the path external systems point toward: many positions in flight,
  batched neural evaluation, compact replay, and coarse host synchronization.
- It can fill GPU batches even when one actor batch underfills recurrent search.
- It can overlap CPU env mechanics with GPU search and delayed replay payload
  flushes.

What could go wrong:

- Request ordering bugs attach search results to the wrong env row.
- Policy versioning becomes ambiguous.
- Queue latency erases batching gains.
- Backpressure changes exploration distribution or replay order.
- Debuggability drops sharply if the compact single-batch contract is not
  already boring.

Smallest falsifying experiment:

```text
Single-process local service:
  2-4 compact slab producers
  direct CTree or compiled fixed-shape backend
  ordered request ids
  action-only response before env step
  payload flush before sample visibility

Compare:
  synchronous compact direct CTree loop
  multi-producer batched service loop
```

Kill for now if it cannot beat synchronous direct CTree by at least `2x` on a
search-heavy `sim32` profile while preserving deterministic request/replay
ordering, or if queue/flush time consumes the search gain. Do not start here
before the compact slab and replay identity gates are solid.

## 7. Do Nothing Except Stock Tuning Baseline

What changes:

- Keep Coach on stock LightZero `train_muzero`.
- Tune only low-risk knobs: simulation count, actor count, batch size,
  checkpoint/eval sidecars, RND meter/cadence, root noise, temperature, and
  profile-only direct CTree controls.
- Keep compact search, service-tax, mock, MCTX, and slab work as optimizer
  evidence only.

Why it could be faster:

- It avoids semantic risk and may recover easy waste from sidecars, bad actor
  count, checkpoint/eval overhead, or stale defaults.
- It keeps the only trusted training denominator stable while architecture
  probes mature.
- It provides the control row needed to judge every big move.

What could go wrong:

- It leaves the largest measured headroom untouched.
- It optimizes a stale denominator while the real wall is search/dataflow.
- It can overfit no-death, no-RND, or profile-only rows.
- CPU or actor-count tuning can get slower and still look like activity.

Smallest falsifying experiment:

```text
Two matched stock full-loop tuning sweeps:
  same seed/config family
  sidecars disabled or matched
  fallback counts explicit
  trainer entrypoint confirmed
  no eval/GIF/checkpoint noise unless intentionally included
```

Demote stock-only tuning as the main optimizer plan if it cannot produce a
stable `10-15%` full-loop improvement while service-tax/mock remain at least
`1.7x` above direct CTree on the compact same-shape denominator.

## Recommended Next Slice

Do not choose one giant rewrite yet. Run three small falsifiers in parallel:

1. Compiled/fused fixed-shape MCTS at `B512/P2/A3/sim16+sim32`, behind
   `CompactSearchServiceV1`, with compact replay proof on.
2. Compact slab owner using direct CTree first, proving selected actions drive
   the next env step and replay/RND identities survive.
3. Flat fixed-`A=3` CTree integrated into the direct compact service, to see
   whether the no-model CTree win survives real model/search rows.

Use JAX/MCTX as the external comparator for what the accelerator-native endpoint
should look like. Use stock tuning as the control. Do not tell Coach to launch
any compact, fused, MCTX, service-tax, mock, slab, or replay-owner lane until
the compact sampler/RND/player-perspective gates and a matched stock-vs-candidate
full-loop smoke pass.
