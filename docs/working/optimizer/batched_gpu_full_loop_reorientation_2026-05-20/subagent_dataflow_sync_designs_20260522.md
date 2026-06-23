# Compact MCTX Dataflow, Sync Points, And Designs - 2026-05-22

Status: docs-only subagent note. I reviewed the current working docs and
`src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`. I did not launch live
runs, touch production trainer code, modify checkpoints, or change trainer
defaults.

## Scope

This note is about the profile-only compact visual plus MCTX loop, not live
Coach training.

Current denominator used as the mental model:

```text
H100, B1024/P2, loop24, no-death compact loop,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack,
no root observation copy,
explicit resident-stack sync off,
exact delta pack,
closed-loop replay-index on for replay-valid rows.
```

Short read:

```text
The old "payload extraction is huge" read was mostly self-inflicted.
_extract_mctx_root_values now reads root node values directly from the MCTX
search tree before falling back to search_tree.summary().

After that fix, root-value extraction is small. Replay-index construction is
also small. The main wall is again the next-search-input handoff:
CPU compact/render state -> delta/compose H2D -> GPU latest frame
-> resident stack/root input -> MCTX search.
```

## Current Objects And Owners

Assume:

```text
B = 1024 env rows
P = 2 players
R = 2048 search roots
A = 3 actions
stack = [4,64,64]
```

| Object | Owner now | Residency | Approx size at B1024/P2 | Read |
| --- | --- | --- | ---: | --- |
| CurvyTron mechanics state | vector actor/env | CPU NumPy | large, trail/body dependent | CPU is authoritative. |
| Borrowed render state | actor `env.state` | CPU NumPy references | large if copied | Current profile borrows for `actor_count=1` no-death. |
| Compact scalar sidecars | `HybridCompactBatch` / manager | CPU NumPy | mostly KiB | Rewards, done, masks, ids. |
| Latest rendered frame | persistent JAX renderer | GPU/JAX | about `8 MiB` uint8 | `[B,2,1,64,64]`. |
| Resident visual stack | benchmark loop | GPU/JAX | about `32 MiB` uint8 | `[B,2,4,64,64]`. |
| Root observation host field | `CompactRootBatchV1` | CPU NumPy/view | `32 MiB` if copied | Current denominator uses no-copy. |
| Invalid/legal mask | root sidecar | CPU then GPU | about `6 KiB` bool | Small, but can still sync. |
| MCTX search state | JAX/MCTX | GPU/JAX | sim/hidden dependent | Search-internal. |
| Selected actions | MCTX output | GPU then CPU | about `8 KiB` int32 | Required before CPU env step. |
| Visit policy | MCTX output | GPU then CPU | about `24 KiB` float32 | Required for replay target, not env step. |
| Root values | MCTX search tree | GPU then CPU | about `8 KiB` float32 | Now extracted from root node values cheaply. |
| Replay index rows | compact replay bridge | CPU NumPy | small, usually `<128 KiB` | Current replay-valid rows say this is not the wall. |

Large byte movements are the stack, latest frame, render/trail state, and
renderer delta/compose payloads. The action, mask, policy, and value arrays are
small by bytes but can be expensive if they force ordering at the wrong time.

## Full Iteration Dataflow

One closed-loop iteration starts with a CPU `HybridCompactBatch`, GPU resident
stack, and previous search output. It ends with the next `HybridCompactBatch`,
updated GPU stack, search result payload, and optional replay-index rows.

1. **CPU joint action exists.**
   Previous selected actions are scattered into `joint_action[B,2] int16`.
   This is about `4 KiB` and lives on CPU.

2. **CPU env step runs.**
   `HybridBatchedObservationProfileManager.step(joint_action)` calls the
   in-process actor and `VectorMultiplayerEnv.step(...)`. True mechanics are
   CPU/NumPy. In current rows, mechanics are small compared with the inclusive
   `env_step_sec` bucket.

3. **CPU compact sidecars are written.**
   The manager/actor writes reward, done, action masks, row ids, player ids,
   active-root masks, and small metadata. With `native_actor_buffer=True`, this
   is direct parent-buffer writing rather than returning large payload objects.

4. **Render state is exposed.**
   With `borrow_single_actor_render_state=True`, the renderer borrows the
   single actor's render state instead of copying large visual-trail arrays into
   parent render buffers. This is profile-only and no-death-only unless a
   terminal snapshot path exists.

5. **Renderer builds next latest frame.**
   The persistent JAX renderer still performs CPU compact-state/delta work:

   ```text
   CPU production/borrowed state
   -> CPU compact render state
   -> CPU exact delta pack
   -> H2D delta/compose state
   -> GPU persistent framebuffer update
   -> GPU compose latest frame [B,2,1,64,64]
   ```

   Raw GPU drawing is already small. The larger costs are ownership, packing,
   H2D, public packaging, and synchronization around the draw.

6. **GPU resident stack updates.**
   The benchmark appends `renderer.last_output_device` to the resident stack:

   ```text
   device_stack = concat(device_stack[:, :, 1:], latest_device)
   ```

   This rewrites or allocates a `32 MiB` uint8 stack on device. In the current
   denominator, the explicit `device_stack.block_until_ready()` is off.

7. **CPU compact root batch is built.**
   `build_compact_root_batch_v1(...)` builds `CompactRootBatchV1` with root
   identity, legal mask, active mask, reward/done sidecars, and observation
   metadata. `copy_observation=False` avoids copying the `32 MiB` observation
   stack. In resident mode, the hot search input is the GPU stack, not the host
   root observation.

8. **Search input is readied.**
   Resident mode reshapes the GPU stack to `[R,4,64,64]`. The invalid-action
   mask `[R,3]` is still put on the JAX device each loop.

   Current code still calls `block_until_ready()` on `loop_obs` and
   `loop_invalid` before measuring search. With explicit resident-stack sync
   off, this block or the search call may become the first real wait for the
   renderer/stack chain.

9. **JAX/MCTX search runs.**
   `run_search(...)` normalizes the uint8 visual stack, runs the tiny visual
   encoder, builds MCTX roots, and calls `mctx.gumbel_muzero_policy(...)`.
   Replay-valid rows block on `action_weights`; action-only diagnostic rows
   block on `action`.

10. **Search outputs become CPU-visible.**
    Replay-valid rows read:

    ```text
    action[R]
    action_weights[R,3]
    root_values[R]
    ```

    The important fix is that `_extract_mctx_root_values(...)` now tries
    direct fields like `search_tree.node_values[:, 0]` before using
    `search_tree.summary()`. That changed root extraction from a large
    accidental materialization path into a small payload read.

11. **Search result is validated.**
    `CompactSearchResultV1` validates active-root selected actions, legal visit
    policies, root values, root ordering, and metadata. This is CPU work over
    small arrays.

12. **Replay-index rows are built.**
    `CompactReplayIndexRowsV1` records action, policy target, root value,
    reward/done/final sidecars, and row/player identity. It does not copy the
    full observation stack. Latest replay-valid rows show this is small.

13. **Loop carries state forward.**
    The next iteration starts with:

    ```text
    CPU: next HybridCompactBatch and env state
    GPU: updated persistent renderer layer, latest frame, resident stack
    CPU/GPU boundary: small mask/action/search payloads
    ```

## Expected Sizes Per Step

These are rough payload sizes for the current denominator:

| Payload | Shape | Size | Move |
| --- | --- | ---: | --- |
| selected actions | `[2048] int32` | `8 KiB` | GPU to CPU, mandatory. |
| joint action | `[1024,2] int16` | `4 KiB` | CPU write. |
| legal/invalid mask | `[2048,3] bool` | `6 KiB` | CPU to GPU today. |
| visit policy | `[2048,3] float32` | `24 KiB` | GPU to CPU for replay. |
| root values | `[2048] float32` | `8 KiB` | GPU to CPU for replay. |
| latest frame | `[1024,2,1,64,64] uint8` | `8 MiB` | Keep GPU-resident. |
| full visual stack | `[1024,2,4,64,64] uint8` | `32 MiB` | Keep GPU-resident. |
| full visual stack float32 | same | `128 MiB` | Avoid on hot path. |
| host stack shift | roughly `3/4` stack | `24 MiB` | Avoid in resident mode. |
| root observation copy | `[2048,4,64,64] uint8` | `32 MiB` | Current no-copy avoids it. |
| visual trail pos | `[1024,4096,2] float32` | `32 MiB` | Avoid full-row copies. |
| visual trail radius | `[1024,4096] float32` | `16 MiB` | Avoid full-row copies. |
| visual trail owner | `[1024,4096] int32` | `16 MiB` | Avoid full-row copies. |

The bytes say the selected-action readback is not the big target. The target is
the repeated reconstruction and handoff of the next observation/search input.

## Synchronization That Must Happen

These barriers are semantic in the current CPU-env architecture:

1. **Selected action before CPU env step.**
   The CPU env cannot advance until it has `joint_action[B,2]`.

2. **Post-step sidecars before root/search.**
   Legal masks, active-root masks, reward/done sidecars, row ids, and player ids
   must match the same state as the observation being searched.

3. **Renderer-to-search dependency.**
   Search must consume the frame produced by the current renderer update. This
   requires a device dependency, but not necessarily a host-visible block at the
   renderer boundary.

4. **Terminal final-observation before autoreset.**
   The current denominator is no-death. A production-like variant must snapshot
   final observations before reset mutates the row.

5. **Replay commit before learner visibility.**
   Action, visit policy, root value, reward, done, final flags, row ids, and
   player ids must all be present before a transition is sample-visible.

6. **Validation and summary reads.**
   Parity checks, sampled host mirrors, logs, and profile summaries can sync,
   but they should be labeled as validation/profile overhead.

## Synchronization That Can Be Delayed Or Sampled

These are candidates, not guaranteed wins:

1. **Resident-stack block.**
   Keep explicit resident-stack sync off and judge total loop wall. The wait may
   move into search input or search; roots/sec is the only score.

2. **Renderer output host readback.**
   Do not D2H `[B,2,1,64,64]` just to feed search. Read host frames only for
   sampled parity, debug, or terminal/final-observation paths.

3. **Host stack maintenance.**
   With scalar LightZero materialization off and MCTX consuming GPU stacks, host
   stack update should be a sampled mirror, not the hot source of truth.

4. **Root observation materialization.**
   Keep `CompactRootBatchV1` as sidecars and validation metadata in the hot
   row. Do not copy `[R,4,64,64]` each loop.

5. **Invalid-mask H2D readiness.**
   The mask is tiny. It may be moved earlier, kept persistent, or allowed to
   become ready when search consumes it.

6. **Visit policy and root value readback.**
   These are now small after direct root extraction. They are not needed before
   CPU env step, but they are needed before replay commit. Chunking can still
   help if it avoids bad synchronization; serial deferred flushing is not a win
   by itself.

7. **RND hashes and metric scalars.**
   Keep them off the per-step action-critical cadence. They are not in the
   current MCTX denominator but will reappear in trainer-like rows.

8. **Thread overlap canaries.**
   Keep them diagnostic. The latest read says overlap hid some wait but inflated
   env/render work through contention.

## Ten Architecture Designs And Critiques

1. **Polish the current compact loop.**
   Keep no-copy roots, direct root-value extraction, exact delta pack, resident
   stack, and sampled validation. Critique: useful for preventing regressions,
   but unlikely to deliver another large win by itself.

2. **Make resident GPU stack the first-class search input.**
   Host roots carry sidecars; MCTX consumes the resident stack. Critique: this
   is already the right denominator, but stale-stack bugs are easy unless row
   order and sampled pixel parity are strict.

3. **Persistent compact render-state owner.**
   Stop rebuilding compact state from production arrays every step. Let the
   actor or renderer own the compact render layout. Critique: this hits the
   current wall, but terminal/autoreset snapshots must be designed first.

4. **Actor-emitted compact deltas.**
   Actor/env emits changed trail segments, reset masks, player compose fields,
   and bonus compose fields directly. Critique: highest leverage on delta pack
   and H2D, but cursor/reset bugs would silently poison observations.

5. **Persistent device mask and sidecar buffers.**
   Keep legal masks and small root sidecars in stable device buffers, updating
   only changed slices. Critique: byte savings are tiny; value is reducing sync
   and allocation noise. Kill it if total roots/sec does not move.

6. **Chunked replay payload owner.**
   Read selected actions immediately, but commit visit policies/root values in
   ordered chunks before replay rows become sample-visible. Critique: after the
   direct root extraction fix this is lower priority than render/input handoff,
   but it is still the right shape for a future replay service.

7. **Fast compact replay-index builder with sampled strict checks.**
   Use the current strict builder as oracle, but add a hot builder that assumes
   already-validated shapes and copies only needed arrays. Critique: replay
   index is small now, so this should not be P0 unless the strict builder grows
   again in trainer-like rows.

8. **Terminal-safe borrowed render state.**
   Keep the no-copy/borrowed-state win while adding a final-observation snapshot
   or copied fallback for terminal rows. Critique: required before the no-death
   profile can generalize, but it may reduce the borrowed-state win.

9. **Array-native or MCTX search service for sim32 pressure.**
   At sim32, search is a large wall again. A service-owned array search can keep
   tree/model outputs compact and resident. Critique: it does not solve the
   env/render handoff, and toy MCTX is not a Coach-training semantic proof.

10. **Replacement compact trainer topology.**
    Long-term shape:

    ```text
    compact actors
    -> resident observation/root buffers
    -> batched search service
    -> compact replay/RND/sampler
    -> learner tensors
    ```

    Critique: this is the only credible multi-x training architecture, but it
    is not a hidden LightZero config change. It needs separate parity gates for
    death, reset, replay targets, RND, checkpoints, and evaluation.

## Recommended Next Read

The next P0 should be an ownership falsifier on the same denominator:

```text
Can we remove or sharply reduce CPU compact-state rebuild, exact delta pack,
and renderer H2D/update handoff without changing observation semantics?
```

Use total loop roots/sec as the pass/fail metric. Bucket movement alone is not
enough, because delayed synchronization can make one timer prettier while the
same wait returns in search or residual.
