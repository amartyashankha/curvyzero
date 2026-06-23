# Late Architecture Reorientation, 2026-05-22

Scope: read-only review of the active optimizer docs in this folder. No live
Coach runs, source code, or shared training state were touched.

Docs reviewed: `README.md`, `orchestration.md`, `task_board.md`,
`experiment_log.md`, `current_hot_path_bottleneck_map_20260522.md`,
`search_boundary_next_fix_strategy_20260522.md`, `gpu_mcts_current_flow_explainer_20260522.md`,
`world_model.md`, and the late subagent notes on env-step, device-resident
observation, closed-loop contracts, post-search hot path, and next architecture.

## Plain Verdict

The current true bottleneck is no longer raw MCTS/search. Search is strategically
important, but the latest closed compact rows say it is not the next wall in the
measured loop.

The hot wall is the repeated closed compact edge:

```text
selected MCTX/direct actions
-> compact actor/env step
-> production state to compact render state
-> persistent renderer / latest-frame handoff
-> host stack update or root-batch rebuild
-> CompactRootBatchV1 / CompactReplayIndexRowsV1 edge
-> next search
```

`env_step_sec` is currently a misleading label for this wall. The late timing
split says actual game runtime can be small, while render-state write,
observation/renderer/stack update, and production-to-compact handoff dominate
the bucket. The representative late H100 row after the render-state filter was
about `15,906` closed-loop roots/sec with env `73.4%` and search `5.3%`;
within that, actor render-state write was visible (`0.368s`), but
observation/stack (`0.988s`) and production-to-compact (`0.517s`) were larger.

Native actor buffer and live-prefix trimming are good support wins, not the
architecture break. Native actor buffer moved B512/sim16 from about `5.79k` to
`6.82k` active roots/sec and B1024/sim16 from about `6.25k` to `8.92k`. The
live-prefix trim then produced stronger rows, including B1024/sim16/loop16 at
about `15.26k` active roots/sec. But the repeated loop still spends roughly
`71-82%` of wall in `env_step_sec`, while search is usually single-digit
percent.

So the next big question is state/observation residency and compact ownership,
not another narrow search-kernel patch.

## Already-Falsified Main Lanes

- CPU scaling: `gpu-h100-cpu64` made the current search rows slower. More CPU is
  not the fix for this boundary.
- Flat-A3 CTree as the main path: useful ABI evidence, but matched full-loop
  rows did not move (`direct` about `516.55` steps/sec, flat-A3 about `509.69`).
- Direct CTree/output-fast as a 5-10x lane: real and worth keeping as a
  baseline, but matched stock-loop gains are about `1.28x-1.31x`, not a new
  architecture.
- Output assembly: already mostly solved in the direct hook. More polishing
  there cannot move the current denominator much.
- Dense eager Torch search polish: good sim8 signal, but it fails the practical
  sim16 gate against `direct_ctree_gpu_latent`.
- Raw MCTX roots/sec as a training claim: MCTX/JAX proves 10x-class
  search-boundary headroom on real compact visual roots, but repeated
  search/env/replay loops collapse to the few-thousand to low-teens-thousand
  roots/sec regime because the surrounding edge is hot.
- Renderer-kernel-only work: persistent GPU rendering is valuable, but the next
  wall is the whole handoff/stack/root-batch path, not just device draw time.
- Device-latest as previously tested: it cut H2D but regressed when the host
  stack still had to be maintained. A useful resident path must replace the hot
  stack handoff, not add a second stack beside it.
- In-process actor sharding: after live-prefix trim, B1024/sim16/loop16 was
  fastest at `actor_count=1` (`16.42k`) and slower at `4` (`13.15k`) and `16`
  (`11.92k`). This does not rule out real subprocess/native actors, but it rules
  out more in-process shards as the next main move.
- More replay machinery in the current shape: compact replay/index rows are
  cheap enough already. The proof path wrote `61,440` rows with public
  LightZero output bytes `0`; replay/index row cost is not the obvious wall.

## Next 3 Highest-Upside Moves

1. Split `env_step_sec` until it stops hiding the wall.

   Add profile-only timing inside `VectorMultiplayerEnv.step`,
   `_batch`/`_public_info`, compact render-state conversion, and
   observation update. Separate at least:

   ```text
   actor physics/runtime
   body collision scan slots
   public packaging / info / action mask
   production_to_compact render-state conversion
   renderer delta pack / H2D / persistent update / D2H
   host stack shift/latest update
   CompactRootBatch and CompactReplayIndexRows builders
   residual sync / Python packing
   ```

   This should be the first move because `env_step_sec` is now 70%+ of wall. A
   blind patch inside that bucket risks optimizing the wrong half of the
   pipeline.

2. Prototype a device-resident compact observation stack for MCTX.

   Use the persistent renderer's `last_output_device` to maintain a JAX resident
   FIFO stack shaped like `[B,2,4,64,64]`, and feed MCTX directly from that stack.
   Keep host `HybridCompactBatch` / `CompactRootBatchV1` for masks, identity,
   replay validation, and sampled parity, but make the measured search input
   avoid:

   ```text
   GPU render -> host uint8 stack -> jax.device_put(obs_host)
   ```

   Required telemetry: `obs_h2d_bytes=0`, full-frame D2H count, host-stack-update
   count, sampled host/device stack parity, and reset/final-observation guard.
   A `1.2x-1.6x` closed-loop win would already be strong. If it barely moves,
   the remaining P0 is actor/env packaging rather than observation transfer.

3. Add a lower-copy compact actor/env step path with strict sampled validation.

   The optimized lane should write only the compact sidecars the next loop needs:
   reward, done/final/autoreset flags, legal mask, row/player identity, render
   deltas/state, and replay-index fields. Stock/public `BaseEnvTimestep`-style
   packaging, giant info dicts, and target-row materialization should stay as
   validation or sampler-edge adapters.

   A plausible shape:

   ```text
   step_compact_into(preallocated_buffers)
   -> renderer consumes compact/native render buffers
   -> MCTX consumes resident or compact stack
   -> fast CompactReplayIndexRowsV1 builder
   -> strict builder runs periodically or on adversarial rows
   ```

   Include hostile tests for terminal/final reward mismatch, non-prefix active
   roots, omitted inactive rows, and RND latest-frame equivalence. If the new
   split shows no-death collision/public packaging is the real cost, add a
   profile-only no-death/package-skip canary under this same lane.

## Likely Rabbit Holes

- More search-only MCTX, dense Torch, or flat-A3 tables without the repeated
  env/observation/replay edge in the denominator.
- Advertising MCTX/JAX toy-model rows as Coach/training speed. They are
  architecture evidence, not learning proof.
- More CPU count, C768/C2048 width chasing, or in-process actor-count sweeps
  before fixing the compact handoff.
- Renderer kernel polish that still reads frames back to host and rebuilds the
  host stack every loop.
- Exact neutral tie parity in CTree/MCTS as a blocker for falsifiers. Forced
  action, mask, clear-preference, and statistical gates are the useful tests.
- Moving strict target-row materialization into the hot path. Keep index rows hot
  and materialize observations at validation/sampler edges.
- Calling a path "resident" unless telemetry proves no full-frame D2H,
  `.cpu().numpy()`, or scalar LightZero timestep materialization happened inside
  the measured loop.

## Recommendation

Keep `direct_ctree_gpu_latent` and MCTX search rows as baselines and guardrails,
but move the main implementation pressure to the closed compact edge. The next
real architecture bet is:

```text
compact state owner
-> lower-copy/render-resident observation stack
-> device-resident or array-native search
-> compact replay/RND/target edge
-> stock LightZero adapters only for validation/migration
```

The immediate blocker is not proving that search can be fast. It can. The
blocker is proving that the next search call can be fed without repeatedly
rebuilding and moving the same compact observation/root batch through Python and
host memory.
