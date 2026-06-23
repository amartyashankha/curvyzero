# Resident GPU Architecture Options, 2026-05-22

Scope: design note only. I did not touch live Coach training runs, Modal
volumes, checkpoints, evals, GIFs, tournaments, or source code. This note is
for the compact visual/MCTS optimizer lane after the borrowed actor render-state
and resident GPU stack work.

## Baseline Read

The current profile-only best shape is already the right kind of evidence:

```text
CPU CurvyTron actor mechanics
-> borrowed actor render state
-> persistent GPU policy framebuffer
-> resident GPU stack
-> compact root/search
-> compact search result / replay sidecars
```

The useful current anchor is roughly `50k` roots/sec at sim16 and `40k+`
roots/sec at sim32 in borrowed/resident rows, with the user-provided best read
around `50.4k` sim16 and `43.3k` sim32. Nearby docs show the same regime:
borrowed resident sim16 in the `48.6k-51.8k` band, a no-refresh ceiling around
`61.9k`, and lazy resident-stack sync helping one sim16 row but regressing a
sim32 row. So explicit resident-stack sync is not the main next wall.

The hard correction is:

```text
Raw GPU draw is already small. The remaining wall is ownership:
CPU packing, compact/delta construction, H2D of sidecars, root/search-input
materialization, search boundary work, result/readback validation, and residual
object/control flow.
```

All Amdahl ceilings below are rough per-current-profile-loop ceilings, not
Coach training speed claims. I use:

```text
S_max = 1 / (1 - f_removed)
```

where `f_removed` is the fraction of current wall that the option could delete
if implemented perfectly. If an option targets 10% of wall, its infinite-speed
ceiling is only `1.11x`. This is why several ideas that sound powerful are
still not enough on their own.

## Option 1: Compact Native Render-State Owner

What changes:

- Make the actor/env own a renderer-native compact state, not production-shaped
  arrays that must be reinterpreted every decision.
- The current borrowed mode avoids the parent visual-trail copy, but the
  persistent renderer still calls production-to-compact and delta-pack logic.
- The new owner would expose stable compact arrays for player pose, avatar
  color, bonus state, trail cursor/event buffers, reset generation, and row
  validity directly to the renderer.

Resident data:

- CPU-resident compact render state in actor-owned contiguous buffers.
- GPU-resident trail framebuffer and latest frame remain as today.
- Optional pinned host compact buffers for delta/compose H2D.

Host syncs/copies that disappear:

- Parent render-state copy is already gone in borrowed mode, but this removes
  the remaining production-state dict/array adaptation path.
- Repeated `np.asarray` validation and shape conversion around production
  state can become debug-only.
- Renderer production-to-compact work becomes a cheap pointer/view handoff or a
  generation check.

Expected risk:

- Medium-high. The current borrowed canary fails closed on terminal rows because
  post-reset borrowed state is not enough for final-observation semantics.
- The compact owner must preserve visual-trail cursor, break-before,
  same-owner continuity, avatar color reset, bonus/head ordering, and
  terminal snapshot ordering.

Amdahl ceiling:

- If the remaining compact/render ownership work is about `10-20%` of the
  borrowed sim16 wall, perfect deletion gives about `1.11x-1.25x`.
- At sim32, search is hotter, so the same change may only be `1.05x-1.15x`.
- This is still a good P0 because it attacks the largest non-search ownership
  bucket left after borrowing.

First cheap falsifier:

- Profile-only no-behavior-change variant that precomputes or reuses the
  compact render-state view while keeping real actor stepping and real search.
- Judge only total roots/sec and exclusive `production_to_compact` /
  `persistent_delta_pack` / residual movement. Kill the rewrite if a no-op
  compact owner improves total by less than about `8-10%`.

## Option 2: Actor Emits Compact Deltas Directly

What changes:

- Instead of rebuilding renderer deltas by scanning compact state after the
  actor step, the actor emits the new trail/head/bonus deltas at the moment the
  runtime writes visual events.
- The persistent renderer consumes a per-step delta stream:

```text
row, player/view, segment start/end, radius, owner, break flag, reset generation
bonus/head compose sidecars
```

Resident data:

- Actor-owned CPU delta ring, preferably preallocated and pinned.
- Renderer-owned GPU trail layer.
- Renderer-owned previous cursor/owner state can either stay on CPU as a
  validation mirror or move to device once the delta stream is trusted.

Host syncs/copies that disappear:

- Host-side `persistent_delta_state` row/slot loops.
- Many previous-cursor comparisons and owner-position scans in the renderer.
- Some compact-state H2D if only deltas and small compose sidecars are copied.

Expected risk:

- High. This changes the ownership contract at a semantic boundary.
- The scary cases are terminal/final frame, autoreset, trail truncation,
  owner-color change, skipped rows, and any browser-line continuity rule.

Amdahl ceiling:

- If delta pack plus associated H2D is `8-18%` of wall, the ceiling is
  `1.09x-1.22x`.
- Combined with Option 1, a realistic ceiling is closer to `1.25x-1.45x` on
  sim16, but less at sim32 unless search is also changed.

First cheap falsifier:

- Add an instrumentation-only delta-count path over saved profile states:
  measure how much time is spent deriving deltas versus applying them.
- Stronger falsifier: a profile-only "actor emitted delta" toy path for
  no-terminal rows that replays deltas into the existing renderer and compares
  sampled frames. Kill it if total roots/sec does not move, even if the delta
  timer collapses.

## Option 3: GPU-Resident Stack And Root Builder

What changes:

- Treat the resident GPU stack as the search observation source of truth.
- Build root/search input from a device stack handle plus small sidecars,
  instead of building a host `CompactRootBatchV1` whose observation is
  validated/materialized every decision.
- Keep a sampled host mirror for parity, terminal rows, and debug.

Resident data:

- Latest policy frame and `[B, P, 4, 64, 64]` FIFO stack on GPU.
- Root observation, active mask, legal mask, `to_play`, row/player ids, and
  reward/done sidecars as device arrays where the search implementation can
  consume them.
- Host `CompactRootBatchV1` becomes a sampled validation/adaptation object, not
  the hot input owner.

Host syncs/copies that disappear:

- Full stack D2H for search input.
- Root observation copy/validation in the hot cadence.
- Action-mask H2D when masks are already staged on device or copied in one
  batched sidecar transfer.
- Immediate stack `block_until_ready()` if the search call is the dependency.

Expected risk:

- Medium-high. Stale-stack bugs are easy here.
- Must prove FIFO order, row/player reshape, first-frame warmup, terminal final
  observation, autoreset, and partial reset behavior.

Amdahl ceiling:

- Current rows already use resident stack, so this is not another `1.4x`
  image-bounce win.
- The remaining root/sidecar/validation slice is probably `5-20%`, giving
  `1.05x-1.25x` alone.
- It becomes more valuable as soon as search/replay also stay device-resident.

First cheap falsifier:

- Run a profile-only sidecar-only root builder: search consumes the resident
  device stack, while the host root object is built only every K steps and for
  terminal rows.
- Require sampled stack parity and compare total roots/sec. If deleting host
  root observation work gives less than about `8%`, do not make this the next
  main lane by itself.

## Option 4: Device-Resident Renderer Compose State

What changes:

- Keep more than the trail framebuffer on device: previous cursor/owner state,
  avatar colors, bonus compose state, head pose, and reset generation can become
  resident device arrays updated by small deltas.
- Fuse or sequence update+compose without host-visible synchronization until
  search consumes the latest frame.

Resident data:

- GPU trail layer, previous-owner/cursor state, avatar colors, compose sidecars,
  latest frame, and resident stack.
- CPU keeps authoritative game mechanics and a sampled validation mirror.

Host syncs/copies that disappear:

- Compose-state H2D for fields that do not need to be resent every step.
- Some renderer update/compose blocking if the next search dependency is enough.
- Potentially repeated CPU copies of avatar/bonus/head arrays.

Expected risk:

- Medium. Easier than a GPU env rewrite, harder than a packing optimization.
- Device and CPU mirrors can diverge silently unless reset generations and
  sampled checksums are strict.

Amdahl ceiling:

- Raw draw is only a few milliseconds, so this is not a raw-render win.
- If compose H2D/update sync is `5-12%`, ceiling is `1.05x-1.14x`.
- Bigger only if measurement proves compose-state movement is hiding inside the
  residual.

First cheap falsifier:

- Add a profile-only "preloaded compose state" row with real actor step and real
  search, then compare total roots/sec to current borrowed/resident.
- If the gain is under `5-8%`, keep device compose residency as cleanup, not P0.

## Option 5: Vectorized CPU Packing: C++ / Numba / JAX-CPU

What changes:

- Keep CPU authoritative state, but rewrite the remaining hot packing paths in
  a lower-overhead fixed-shape implementation:
  preallocated NumPy, Numba, C++/pybind, Cython, or possibly JAX-CPU.
- Target production-to-compact, delta pack, root sidecar validation/copy, and
  joint-action/result assembly.

Resident data:

- Host compact/delta/root sidecar buffers, ideally preallocated and optionally
  pinned.
- GPU renderer/search state remains as today.

Host syncs/copies that disappear:

- None by ownership. This mainly removes Python loop/allocation overhead.
- It can also reduce H2D overhead indirectly if it writes directly into pinned
  transfer buffers.

Expected risk:

- Low to medium for Numba/preallocated NumPy; medium for C++ due build,
  portability, and semantic test burden.
- JAX-CPU may look elegant but can add compilation and array ownership friction
  without deleting any required boundary.

Amdahl ceiling:

- If Python packing is `5-15%`, ceiling is `1.05x-1.18x`.
- A perfect C++ packer cannot beat the ownership wall if the same bytes and
  syncs still happen.

First cheap falsifier:

- Microbench saved actor states through current pack/delta/root sidecar code
  versus a no-op/preallocated upper bound.
- Only build C++/Numba if the no-op upper bound predicts at least `10%` total
  closed-loop speedup. Otherwise this is polish.

Critical note:

- "Rewrite the packer in C++" sounds serious, but it is not automatically a big
  architecture move. The current wall is not just Python instructions; it is
  ownership and synchronization.

## Option 6: Device-Resident Search Boundary

What changes:

- Replace the LightZero CTree/list/CPU boundary or the current partial compact
  search edge with a fixed-shape device-resident search service.
- The search service owns priors, values, visits, rewards, recurrent state,
  path indices, and legal masks as arrays.
- MCTX/JAX is the cleanest existing shape; a CUDA/Triton/C++ extension is
  possible but should copy the same fixed-array contract, not the old list API.

Resident data:

- Tree arrays, hidden states, recurrent outputs, policy/value/reward arrays,
  legal masks, selected actions, visit policies, and root values.
- Only selected actions need immediate host visibility while env mechanics stay
  CPU.

Host syncs/copies that disappear:

- Per-simulation reward/value/policy D2H.
- Per-simulation traverse action H2D.
- Root policy/value `.cpu().numpy().tolist()` setup.
- Visit/root-value readback can be delayed to replay chunks instead of every
  decision.

Expected risk:

- Very high if connected to real training. The semantics surface includes legal
  masks, Dirichlet/Gumbel behavior, terminal rows, value transforms, root value,
  visit distributions, action selection, and replay targets.
- PyTorch model plus JAX search can recreate the wall if model outputs cross
  frameworks every simulation.

Amdahl ceiling:

- In the current borrowed/resident loop, search is roughly `15-20%` at sim16
  and can be around `30%` at sim32. Infinite search gives only about
  `1.18x-1.25x` at sim16 and `1.43x` at sim32.
- The search-boundary microbench ceilings are huge, but closed-loop speed only
  gets huge if env/obs/replay ownership changes too.

First cheap falsifier:

- Run a closed-loop no-search/mock-search-service ceiling with the same actor,
  render, resident stack, action readback, and replay sidecars.
- If deleting real search does not push the closed loop far beyond current
  `50k/40k` rows, a search rewrite alone cannot be the next big win.

Critical note:

- This is still one of the only plausible 5x ingredients, but not alone. The
  falsifier must be closed-loop, not isolated fresh-boundary roots/sec.

## Option 7: Chunked Search Result / Replay / RND Residency

What changes:

- Read back only selected actions every decision.
- Keep visit policy, root value, search tree summaries, replay index rows, and
  RND latest-frame inputs resident or in compact chunk buffers.
- Flush to CPU/learner-facing storage every K steps, with sampled validation.

Resident data:

- Action weights, root values, compact replay rows, latest frames for RND, and
  target-builder sidecars.
- CPU stores only chunk commits and sampled mirrors.

Host syncs/copies that disappear:

- Per-step action-weight/root-value D2H.
- Per-step replay-index validation/array construction when not needed for the
  next CPU action.
- RND latest-frame extraction from host observations, if RND is included.

Expected risk:

- Medium. Replay/RND target ordering is training-critical.
- The architecture must preserve final observation, reward mutation, row/player
  identity, and action history exactly enough for learner targets.

Amdahl ceiling:

- Existing no-replay rows suggest replay-index alone is small in some
  denominators, often single-digit to about `10%`.
- Ceiling is probably `1.03x-1.15x` unless root-value/action-weight readback and
  validation residual are larger than currently labeled.

First cheap falsifier:

- Profile three rows: current full result readback, action-only readback, and
  no replay rows. Same seed, same closed loop.
- If action-only is not materially faster, defer resident replay until after the
  env/search ownership changes.

Critical note:

- "Resident replay" sounds like a system rewrite, but if replay is only a few
  percent in the current row, it will not push through the wall by itself.

## Option 8: Larger Roots / Bigger Batch Aggregation

What changes:

- Increase B or aggregate multiple actor chunks into a larger search/root batch
  to amortize fixed costs and improve GPU occupancy.
- Could be B1536/B2048, multiple actor groups per search service call, or
  delayed action commit for a larger root batch.

Resident data:

- Larger GPU stack, root arrays, tree arrays, replay sidecars, and optional
  per-actor compact buffers.

Host syncs/copies that disappear:

- None. This amortizes fixed overhead; it does not remove the boundary.

Expected risk:

- Medium. Memory grows quickly with stack and tree state.
- Bigger B can worsen actor/observation handoff, terminal handling, cache
  behavior, and per-step latency.

Amdahl ceiling:

- The current B2048 borrowed sim16 row reportedly did not materially beat B1024
  (`~52.3k` versus the `~51k` band).
- Near-term ceiling is probably `1.00x-1.10x`, with regression risk at sim32.

First cheap falsifier:

- Already mostly falsified by B2048 borrowed/resident rows.
- If retesting, do a small B768/B1536/B2048 sweep with identical loop length and
  report total roots/sec plus observation/search/residual fractions.

Critical note:

- Bigger batch is tempting because it makes GPU numbers look better. The
  present wall is not underfilled draw kernels; it is closed-loop ownership.

## Option 9: Subprocess / Actor Parallelism And Overlap

What changes:

- Run multiple actor subprocesses producing compact deltas or compact root
  sidecars while a central GPU service searches the previous chunk.
- Use shared memory or pinned buffers; keep scalar Python objects out of IPC.

Resident data:

- Per-actor compact host buffers, central GPU renderer/search state, and chunked
  replay buffers.
- Optionally one GPU stack per actor chunk, or a central stack indexed by row.

Host syncs/copies that disappear:

- None by default. This hides work through overlap and escapes some GIL/process
  scheduling limits.
- It only helps if IPC/merge cost is lower than the CPU work it overlaps.

Expected risk:

- High. Actor-count grids have already shown that more actors are not a free
  win when merge/copy costs dominate.
- Terminal/final-observation ordering across processes is easy to get wrong.
- Modal/process overhead can swamp the gain.

Amdahl ceiling:

- If env/observation/handoff is `45-55%` and search is `15-30%`, perfect overlap
  of search under CPU work gives only about `1.15x-1.30x`.
- If subprocesses also halve CPU handoff, combined ceiling could be
  `1.3x-1.6x`, but only after compact buffers eliminate IPC object churn.

First cheap falsifier:

- Build a profile-only two-process producer that emits preallocated compact
  deltas into shared memory while the parent runs the existing renderer/search.
- Include a no-render/no-search IPC-only row. Kill this lane if IPC+merge costs
  eat more than half of the overlapped work.

Critical note:

- Subprocess parallelism is relevant only after the payload is compact. Multiple
  processes moving Python objects would make the current disease distributed.

## Option 10: Native Vector Buffer Or GPU Env Rewrite

What changes:

- Larger architectural move: CurvyTron state, actions, rewards, dones, legal
  masks, visual deltas, and replay rows live in static contiguous buffers.
- Could be a C++/Rust/Numba native vector env, a JAX env, or eventually a CUDA
  env. The important part is direct buffer ownership, not the language badge.

Resident data:

- Native host or device env state buffers.
- Legal masks, rewards, done flags, visual deltas, and action outputs in
  contiguous arrays.
- With a GPU env, selected actions can remain device-side and the CPU action
  barrier disappears.

Host syncs/copies that disappear:

- Python env object fanout.
- Scalar row/player materialization.
- Many compact sidecar copies.
- With a GPU env, selected-action D2H before CPU env step.

Expected risk:

- Very high. This is a semantic rewrite of physics, collision, bonuses, visual
  trail event writes, terminal/autoreset, and final-observation ordering.
- A pure GPU env rewrite is especially risky because actual game mechanics are
  only a small fraction of current `env_step_sec`; the larger cost is packaging
  and ownership around mechanics.

Amdahl ceiling:

- If it accelerates only mechanics, and mechanics are about `8-11%` of
  `env_step_sec`, whole-loop ceiling may be only `1.05x-1.12x`.
- If it removes most env/observation/sidecar ownership, it can target
  `35-55%` of wall, giving `1.54x-2.22x`.
- Full `5x+` needs this combined with device-resident search and chunked replay,
  not as an isolated env rewrite.

First cheap falsifier:

- Replace real env stepping with a deterministic prerecorded/no-op state tape
  while keeping real renderer, resident stack, search, action readback, and
  replay sidecars.
- If no-op env mechanics barely moves total roots/sec, do not start with a GPU
  env. If no-op env plus no-pack compact sidecars moves a lot, the native buffer
  rewrite has a real target.

Critical note:

- "GPU env" sounds like the endgame, but the current evidence says "native
  compact buffer ownership" is the valuable idea. GPU physics alone is probably
  a distraction right now.

## Ranking

Most likely to move the current borrowed/resident wall soon:

1. Compact native render-state owner plus actor-emitted deltas.
2. Sidecar-only GPU-resident root builder with sampled host mirror.
3. Closed-loop device-resident search boundary falsifier.
4. Chunked result/replay readback after action-only correctness is proven.

Useful but likely bounded:

5. Vectorized/C++/Numba packing, only if no-op pack shows a double-digit total
   ceiling.
6. Renderer compose-state residency, only if telemetry exposes compose/H2D as a
   real slice.

Probably not the next wall by itself:

7. Bigger batches.
8. More subprocess actors without compact shared buffers.
9. GPU env mechanics alone.
10. More raw draw-kernel tuning.

## Concrete Combined Architecture

The architecture that plausibly breaks out of the `50k`-ish profile ceiling is
not one option. It is this stack:

```text
actor-owned compact state and emitted visual deltas
-> persistent GPU renderer and resident stack
-> device/search-native root builder
-> device-resident fixed-shape search
-> action-only per-step readback
-> chunked replay/RND/target commit
-> sampled host mirrors for parity and terminal/final-observation rows
```

The first combined falsifier should be intentionally small:

```text
current borrowed/resident row
vs sidecar-only root + action-only readback + sampled host mirror
vs mock/no-search ceiling with the same env/render/replay edge
```

If that ceiling is only modestly above the current row, the next bottleneck is
still env/render ownership. If it jumps, the search/replay boundary is the next
P0. Either way, do not judge by isolated fresh-boundary roots/sec; judge by the
closed compact loop.
