# CurvyTron Compact Visual + MCTX GPU Sync Model

Date: 2026-05-22

Scope: optimizer subagent note. I did not touch live Coach training runs,
checkpoints, evals, GIFs, tournaments, or Modal volumes. I read local optimizer
docs/source and official/external references. I tried a tiny local JAX device
check, but this uv environment does not have `jax` installed, so I did not run
a local transfer microbench.

## Short Read

GPU work helps when the same device-owned data feeds several steps of work:

```text
device latest frame -> device visual stack -> batched MCTX search
```

GPU work does not help much when every step still does:

```text
CPU production state -> CPU packing -> many small device_puts
-> block_until_ready -> GPU work -> host readback -> CPU objects
```

The current compact visual + MCTX profile is already past the first obvious
mistake. The full observation stack is mostly device-resident, raw GPU drawing
is small, direct root-value extraction fixed an accidental MCTX materialization
wall, and replay-index construction is small in replay-valid rows.

The current Amdahl wall is the next-search-input boundary:

```text
CPU env mechanics and sidecars
-> compact/render-state ownership
-> delta pack
-> host-to-device payloads
-> resident stack/root input readiness
-> MCTX search
```

At lower simulation counts, the handoff dominates. At higher simulation counts,
MCTX search itself becomes a large bucket too. That means the next useful work
is not "GPU everything" as a slogan. It is fewer host-visible boundaries and
fewer forced waits between the CPU env, GPU renderer, resident stack, and MCTX.

## External Anchors

The external guidance is consistent:

- JAX dispatch is asynchronous. A `jax.Array` is a future; host inspection,
  NumPy conversion, printing, or `block_until_ready()` forces Python to wait.
  See JAX asynchronous dispatch:
  <https://docs.jax.dev/en/latest/async_dispatch.html>.
- CUDA best practices say to minimize host/device transfers, keep intermediate
  data on device, batch many small transfers into larger ones, and use pinned
  memory for high-bandwidth/async copies when appropriate:
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>.
- MCTX is the clean search-side shape: JAX-native, JIT-able, batched MCTS over
  root arrays:
  <https://github.com/google-deepmind/mctx>.
- PufferLib's performance docs emphasize vectorized env execution, pinned CPU
  observation buffers when offloading, batch sizing, zero-copy options, and
  bottleneck identification:
  <https://pufferai-pufferlib.mintlify.app/advanced/performance-tuning>.
- TorchRL's vectorized-env docs make the normal RL systems point: env stepping
  can be CPU-heavy, so parallel/vectorized environment execution is a standard
  tool:
  <https://docs.pytorch.org/rl/stable/reference/envs_vectorized.html>.

## Current Local Flow

Current profile-only loop, simplified:

```text
1. CPU has selected action from previous search.
2. CPU CurvyTron env steps B rows with P=2 players.
3. CPU writes rewards, done flags, masks, ids, and render/public sidecars.
4. Persistent JAX renderer builds compact render state and exact deltas on CPU.
5. Renderer device_puts delta/compose payloads.
6. GPU updates persistent framebuffer and produces latest [B,P,1,64,64].
7. Resident GPU stack appends latest frame into [B,P,4,64,64].
8. Compact root sidecars are built on CPU.
9. MCTX consumes resident stack plus invalid masks.
10. CPU reads selected actions for the next env step.
11. Replay-valid rows also read visit policy and root value, validate, and
    build compact replay-index rows.
```

Important current code boundaries:

- `mctx_synthetic_benchmark.py` builds roots, transfers invalid masks, calls
  MCTX, reads selected actions/policy/root values, then steps the compact visual
  manager.
- `source_state_hybrid_observation_profile.py` owns the CPU env step, compact
  sidecars, observation update, compact batch build, and pre-scalar probe.
- `source_state_batched_observation_boundary_profile.py` owns the persistent
  renderer path:

  ```text
  production_to_compact -> persistent_delta_pack -> H2D
  -> persistent_update -> device_render -> optional D2H
  ```

- `source_state_gpu_render_benchmark.py::_copy_state_to_device(...)` now has an
  optional no-block mode, but normal copies can still force a wait.

## Current Measured Read

Use these as profile-only compact-loop facts, not live Coach training speed:

- Direct MCTX root-node extraction changed root-value extraction from a large
  accidental wall into a small read. That is a real keep.
- Replay-valid loop96 rows show replay-index row construction is small relative
  to total loop wall.
- Async H2D deferral gave only a small win, around `1.05x-1.06x` in paired
  loop96 rows. Keep it as an opt-in profile flag, not the main thesis.
- `compact_batch_build_sec` and `batched_stack_probe_wall_sec` are tiny in the
  latest timed row. They are not the hidden wall.
- Raw device render is tiny compared with the enclosing renderer/observation
  handoff. The expensive part is not drawing one frame; it is ownership,
  packing, H2D/update waits, public packaging, and search-input readiness.

Plain Amdahl read:

```text
If we only speed raw drawing, total speed barely moves.
If we remove repeated CPU packing/H2D/waits at the observation/search boundary,
we can still move total wall.
If sim count rises, search becomes hot enough that search architecture and
batching matter alongside observation handoff.
```

## Tensor Residency Model

Assume `B=1024`, `P=2`, `R=B*P=2048`, `A=3`, stack `[4,64,64]`.

### Worth Keeping Resident

These should stay device-resident in the hot loop:

| Tensor/state | Shape/size | Why |
| --- | ---: | --- |
| Latest policy frame | `[B,P,1,64,64] uint8`, about `8 MiB` | It feeds the next stack frame. Reading it to host just to send it back is wasted. |
| Policy visual stack | `[B,P,4,64,64] uint8`, about `32 MiB` | This is the actual MCTX observation input. Keep it as the search source of truth. |
| Persistent framebuffer/trail layer | renderer-owned device arrays | The whole point of the GPU renderer is avoiding full redraw and full frame traffic. |
| MCTX tree/model state | JAX arrays | MCTX only pays off when search arrays and recurrent/model work stay in the compiled/device world. |
| Optional staged visit policy/root value chunks | `[R,A]`, `[R]` | Small by bytes, but keeping them ordered/chunked can avoid bad per-step payload waits later. |

These should stay CPU-resident for now:

| State | Why |
| --- | --- |
| CurvyTron mechanics state | CPU is still authoritative. Rewriting physics first is too big and not yet proven to be the wall. |
| Reward/done/legal/id sidecars | They are small and tied to CPU env semantics. Copy as packed batches when search needs them. |
| Selected actions after search | CPU env needs them before the next step. This readback is small and semantically required. |

### Cheap To Copy

These are fine to copy when batched and not accidentally forcing unrelated
device work to finish:

| Payload | Approx size | Read |
| --- | ---: | --- |
| Selected actions `[R] int32` | `8 KiB` | Mandatory before CPU env step. |
| Joint actions `[B,P] int16` | `4 KiB` | CPU write. |
| Invalid/legal mask `[R,A] bool` | `6 KiB` | Cheap by bytes; still watch sync/allocation. |
| Visit policy `[R,A] float32` | `24 KiB` | Needed for replay, not for env step. |
| Root values `[R] float32` | `8 KiB` | Needed for replay. Direct root-node extraction makes this cheap. |
| Scalar metrics/checksums | tiny | Safe only outside the per-step critical path or sampled. |

Cheap by bytes does not mean free. A tiny `np.asarray(device_array)` can still
force all earlier queued GPU work to finish.

### Expensive Or Poisonous To Copy

Avoid these in the hot loop:

| Payload | Approx size | Why |
| --- | ---: | --- |
| Full policy stack `[R,4,64,64] uint8` | `32 MiB` | It is the hot search input. Keep resident. |
| Full stack float32 | `128 MiB` | Avoid unless the model really needs this resident format. Normalize on device. |
| Latest frame D2H every step | `8 MiB` | Only useful for sampled parity/debug/final-observation cases. |
| Visual trail full arrays | tens of MiB | Copying/scanning full trail state every step defeats incremental render. |
| Per-root Python timestep/info objects | thousands of objects | Object churn kills batching and makes GPU wait for Python. |

## Safe Sync Points

These are semantically real or safely amortized:

1. **Selected-action readback before CPU env step.** CPU cannot step without
   actions. This is a small read and should remain explicit.
2. **Device dependency from renderer to search.** Search must see the latest
   frame. This should be a device-side dependency, not a host readback.
3. **Replay commit boundary.** Visit policy, root value, action, reward, done,
   mask, row id, player id, and final-observation facts must all exist before a
   replay row is sample-visible. This can be chunked.
4. **Terminal final-observation capture.** Terminal rows need exact ordering
   before autoreset mutates the row. It is acceptable to use a slower fallback
   for terminal rows.
5. **Warmup/end-of-run profiling barriers.** Fine for honest measurement, as
   long as they are labeled as profiling overhead.
6. **Sampled parity checks.** Read back frames or stacks every K steps, first N
   steps, and terminal rows. Do not do this every step on the hot path.

## Likely Poison Sync Points

These should be treated as suspect unless a same-denominator profile proves
they are harmless:

1. **`block_until_ready()` immediately after renderer update/compose.** If the
   next MCTX search consumes the frame, search can be the dependency.
2. **`block_until_ready()` on the resident stack before search.** It may only
   move wait time; judge total roots/sec.
3. **Many small `device_put`s for delta/compose state.** CUDA guidance is clear:
   batch small transfers. Many tiny arrays can cost more in launch/sync/driver
   overhead than their byte size suggests.
4. **Host inspection of device arrays inside the step loop.** In JAX this turns
   a future into a hard wait.
5. **Full frame or full stack host mirrors every step.** Useful for validation,
   bad as the source of truth.
6. **Fallback to `search_tree.summary()` for root values.** This was the old
   accidental materialization wall. Treat fallback use as a profile regression.
7. **Thread overlap that touches Python-heavy env/render code.** The previous
   overlap canary hid wait in one bucket but inflated other work. It is a
   diagnostic, not a training recommendation.

## When GPU Helps

GPU helps when:

- root count is large enough (`B*P` roots) to amortize launch overhead;
- shapes are stable enough for compiled/JIT paths;
- observation stays as `uint8` until device-side normalization;
- renderer output feeds resident stack without D2H;
- MCTX search reads resident tensors directly;
- only selected actions return to CPU on the action-critical cadence;
- replay/search payloads commit in batches, not one scalar object at a time.

GPU helps less when:

- CPU env waits after every tiny device operation;
- we copy full stacks or frames just to rebuild Python objects;
- legal masks, frame stacks, and search payloads are all separate sync points;
- the model/search is too small per launch;
- Python loops own per-root/per-simulation control.

## Three Concrete Architecture Tests

### 1. Packed Device Sidecar + Resident Stack Search Input

Goal: make the search input one resident stack plus one packed sidecar transfer.

Test shape:

```text
current resident stack baseline
vs
device stack + packed sidecar struct:
  invalid_mask, active_root_mask, row, player, reward, done
```

Rules:

- no full root observation copy;
- no host latest-frame readback;
- one packed H2D sidecar if possible;
- sampled host root/stack parity only every K steps;
- profile loop96 sim16 and sim32.

Pass condition: total roots/sec improves meaningfully, not just bucket
reshuffling. This is a smaller, safer test than a full GPU env rewrite.

### 2. Direct Compact Delta Owner

Goal: remove repeated production-state-to-compact and delta-pack work.

Test shape:

```text
actor/env emits compact visual deltas:
  row, segment start/end, radius, owner, break flag,
  head/bonus compose fields, reset generation
renderer consumes deltas directly
```

Rules:

- profile-only first;
- no-death first, then terminal fallback;
- compare sampled frames against current reference path;
- keep row/player identity checks;
- report production_to_compact, persistent_delta_pack, H2D array count/bytes,
  and total roots/sec.

Pass condition: total loop wall moves by at least the Amdahl-predicted share of
those buckets. If only the named timer collapses but total roots/sec does not
move, stop.

### 3. Search-Service Boundary With Action-First / Replay-Valid Commit

Goal: separate the tiny action-critical read from the larger replay-valid
payload commit without lying about training semantics.

Test shape:

```text
MCTX search output:
  read selected actions immediately for CPU env
  keep or chunk visit policy/root value until replay commit
  commit replay rows only when payload is complete and validated
```

Rules:

- action-only remains a ceiling, not a valid row;
- replay-visible rows must have action, visit policy, root value, reward, done,
  final-observation flags, row/player ids;
- include RND/latest-frame sidecar only at the compact replay/sample edge;
- run a multi-record parity canary against the immediate replay-valid path.

Pass condition: replay-valid total wall moves closer to action-only without
changing replay contents. If deferred flush simply pays the same cost later,
do not promote it.

## Final Recommendation

Do not chase raw renderer kernels in isolation right now. The credible next
speed path is ownership:

```text
CPU env keeps mechanics and small sidecars.
GPU owns latest frame, visual stack, framebuffer, and MCTX tensors.
The only per-step host readback is selected action.
Replay/RND/search payloads commit in compact chunks with strict parity gates.
```

This is the path that matches JAX/CUDA guidance and the current profile facts.
It is also the cleanest way to test bigger changes without touching live Coach
training.
