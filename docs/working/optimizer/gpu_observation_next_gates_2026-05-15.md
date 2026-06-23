# GPU Observation Next Gates

Date: 2026-05-15

Status: active optimizer gate list.

## Current Truth

Production training and tournaments still use:

```text
browser_lines + simple_symbols -> cpu_oracle -> [4,64,64] stack
```

The isolated H100 GPU renderer now matches the CPU oracle on the checked
real-env smoke rows after same-owner connectivity, cursor masking, avatar-color
reference, and owner-priority fixes. That is a useful renderer proof, not a
trainer proof.

Do not switch stock training to GPU observations until the gates below pass.

## Gate 1: Adversarial Render Parity

Build a frozen corpus of hard production states and compare GPU output against
the actual CPU oracle, not a reconstructed fake oracle.

Expanded red-team list:
[GPU renderer adversarial gates](gpu_renderer_adversarial_gates_2026-05-15.md).

Required cases:

- same-owner trail points separated by opponent slots;
- `break_before` suppressing a segment;
- owner trail crossings where lower-luma player should still win by owner draw
  order;
- stale active bits beyond `visual_trail_write_cursor`;
- cursor wrap or near-full visual trail buffers;
- swapped, duplicated, and high-index `avatar_color` rows;
- live head over trail, bonus over trail, and head/bonus overlap;
- all simple-symbol bonus types, including edge clipping;
- controlled-player `0` and `1`;
- reset frame, terminal frame, and post-reset first frame.

Pass condition for the debug/reference row: exact `[1,64,64]` policy-frame
parity for every checked case.

Pass condition for the aggressive learned-observation row: same shape, dtype,
player view, draw order, object presence, and stack/reset behavior, with only
bounded tiny grayscale drift allowed. The current float32 tolerance is for
one-to-two luma edge differences, not for missing objects, wrong symbols, wrong
seat, or reset/final-observation mistakes.

Current progress: an isolated `adversarial_fixture` benchmark now covers the
geometry/color/bonus subset above and passed exact H100 parity for controlled
players `0`, `1`, and `2` on a 4-row, 3-player corpus. That closes the first
small renderer-parity slice; it does not close stack/reset/final-observation
contract parity.

## Gate 2: Trainer-Visible Contract Parity

The renderer frame is only one field. A GPU observation backend must preserve
the whole env step contract:

- `obs` stack contents and channel order;
- `reward`;
- `done` / `terminated`;
- `truncated`;
- `info` and `final_observation` behavior;
- legal/reset behavior;
- controlled-player perspective;
- LightZero `to_play=-1`; the controlled seat is encoded by the selected
  observation slice/action mask and metadata, not by `to_play`;
- stochasticity, bonus, and death timing;
- frame-stack reset behavior.

Pass condition: a one-step gauntlet compares every trainer-visible field across
CPU and GPU for hand-built states plus sampled real rollouts. Use exact
comparison for `float64` debug rows and bounded-difference comparison for
`float32` speed rows. Shape/order/perspective/reset/RND mistakes are hard
failures in both modes.

## Gate 3: Profile-Only Batched Boundary

Do not keep polishing scalar `policy_observation_backend=jax_gpu` as a
production candidate. It renders one env at a time and pays launch/copy overhead
on every step.

Current status: a profile-only batched observation facade now exists. It should
stay outside live training runs and stock LightZero defaults while it measures
the boundary below.

Next systems measurement:

1. Own many CPU envs in one parent process.
2. Step them normally.
3. Gather compact render state.
4. Call one batched GPU renderer.
5. Update `[4,64,64]` stacks.
6. Return LightZero-shaped observations.
7. Measure pack, host-to-device, render, readback, stack update, reset, and
   final observation costs.

This is profile-only work. It should not touch live training runs or stock
LightZero defaults.

## Gate 4: Amdahl Read After The Boundary

Only after Gate 3 should we claim a full-loop speedup. The current exact GPU
renderer is correctness-promising but not clearly faster:

- B64/S1024 one-view H100: about `212ms`;
- B256/S1024 one-view H100: about `735ms`;
- B64/S256 one-view H100: about `59ms`.

The narrow render-only speed is not enough. We need wall-clock impact after
batching, stack update, policy/search, replay, reset, and host synchronization
are included.

2026-05-15 ordered-compact update: the benchmark-only
`owner_ordered_compact` trail composition drops the high-resolution priority
buffer by CPU-packing compact trails in production draw order, then overwriting
on GPU. It passed exact checked parity on the adversarial fixture and the
real-env B64/S1024 H100 row. The real-env row improved from `208.9ms` to
`135.5ms` device render and from `214.0ms` to `140.4ms` for H2D + render +
readback (`~1.5x`). Follow-up rows also passed exact checked parity: B256/S1024
improved `729.6ms -> 391.4ms` device (`~1.9x`), and B64/S256
controlled-player-1 improved `54.2ms -> 36.0ms` device (`~1.5x`). This does not
change production defaults; it only upgrades the next GPU renderer candidate. A
longer B64/S2048 row also improved `412.6ms -> 265.8ms` device (`~1.6x`). All
of these are one-view isolated renderer rows; trainer value still depends on a
batched row/view boundary.

Scalar two-view H100 profile, S1024: fused exact two-view rendering is much
better than rendering player 0 and player 1 separately (`~1.8x` total speedup),
but both priority-buffer and owner-ordered compact fused B1 rows are still about
`28-29ms` per env step including compacting, host-to-device copy, render, and
readback. Plain read: do not promote scalar GPU. The next useful test is many
rows and both views in one batched boundary.

2026-05-15 two-view batched renderer update: the normal H100 benchmark now has
`--render-views both`, explicit `output_order=view_major`, corrected work
counters, and separate setup timings. Matched real-env S1024 rows with
readback:

- B64 priority buffer: exact checked parity, `489.99ms` device render,
  `494.18ms` H2D + render + readback.
- B64 owner-ordered compact: exact checked parity, `248.13ms` device render,
  `251.51ms` H2D + render + readback (`~1.96x`).
- B256 priority buffer: exact checked parity, `1840.06ms` device render,
  `1844.97ms` H2D + render + readback.
- B256 owner-ordered compact: exact checked parity, `1136.28ms` device render,
  `1142.06ms` H2D + render + readback (`~1.62x`).

Setup timing is reported but not included in those H2D/render/readback rows.
Owner-ordered packing cost in this one-shot benchmark was `5.7ms` at B64 and
`19.6ms` at B256. The next profile must charge packing, stack update, reset,
final observation, and row-major reordering inside a real batched observation
boundary. Do not call these rows a full-loop speedup.

Boundary plan:
[batched GPU observation boundary plan](batched_gpu_observation_boundary_plan_2026-05-15.md).

2026-05-20 boundary check update: the sidecar now has explicit parity modes.
`geometry_dtype=float64` defaults to `parity_mode=exact`; `float32` defaults to
`parity_mode=tolerant`, with bounded mismatch reporting. This lets the speed
candidate verify real steps without disabling parity or failing on a single
one-luma edge pixel.

2026-05-15 boundary result: the sidecar now charges env step,
production-to-compact, owner-ordered packing, H2D, device render, readback,
view-major-to-row-major conversion, and stack update. `geometry_dtype=float32`
is the aggressive GPU candidate now. It failed exact CPU-oracle parity on
B64/S1024 at a later checked step by one luma (`100` vs `101`) at one edge
pixel; that is acceptable for a learned policy observation. `float64` remains
the exact-parity reference/debug mode.

H100 boundary rows:

- B64/S1024 float32: tiny one-luma mismatch in prior deeper check, `255ms`
  candidate observation versus `654ms` CPU reference render+stack.
- B64/S1024 float64: exact reset + 4 checked steps, `379ms` candidate observation
  versus `1.09s` CPU reference render+stack (`~2.9x`).
- B128/S1024 float64: exact reset + 1 checked step, `713ms` candidate observation
  versus `1.43s` CPU reference render+stack (`~2.0x`).
- B256/S1024 float32: speed row with reset-only parity, `1.14s` candidate
  observation versus `2.48s` CPU reference render+stack.
- B256/S1024 float64: exact reset + 2 checked steps, `1.38s` candidate observation
  versus `2.79s` CPU reference render+stack (`~2.0x`).

Float32 is about `1.5x` faster than float64 at B64/S1024 and about `1.2x`
faster at B256/S1024. Use float32 for speed/prototype rows; use float64 only
when debugging exact parity.

Timeout/autoreset row:

- B64/S1024 x64, `max_ticks=5`: exact reset + 4 checked steps +
  terminal/final-observation/autoreset stack parity, `376ms` median candidate
  observation and `920ms` p95. The p95 is higher because terminal steps pay
  final-observation copy plus reset render/stack.

Read: float32 GPU geometry is worth making the aggressive default **inside the
future batched GPU backend**. It is not the same as the current scalar
`policy_observation_backend=jax_gpu`, which is measured slower than CPU in the
stock trainer. The sidecar still reads frames back to host and does not include
policy/search, replay, learner, or natural-death/reward trainer semantics.

## Current Ranked Options

1. **Batched GPU observation boundary with owner-ordered compact render.**
   Highest ceiling if it can batch many rows/views and avoid scalar env CUDA.
   Current isolated evidence says owner-ordered compact is the better exact
   render kernel candidate than the priority-buffer variant. Current boundary
   evidence says float32 is the aggressive default candidate and x64 is the
   exact CPU-oracle parity reference. Risk: host readback, trainer integration,
   and full-loop value.
2. **CPU dirty-block oracle hardening.** Best near-term safety path because it
   stays inside the trusted backend. Risk: invalidation bugs around overlaps,
   reset, and bonus/head order.
   Current progress: simple-symbol RGB stamping now writes only the local symbol
   crop instead of allocating/zeroing a full `704x704` scratch frame, and dirty
   cache fallback reasons are recorded even when detailed timing is off.
3. **Collect/search fanout.** Important whole-loop experiment: if MCTS/search
   dominates after render cleanup, scaling collectors may matter more than
   renderer kernels.
4. **Torch/CuPy/CUDA renderer bakeoff.** Worth testing behind the same fixed
   adversarial corpus after JAX parity is pinned. Risk: another parity surface.
5. **Larger observations.** Signal experiment, not a speed path. Keep separate
   from the 64x64 optimization lane.

## Kill Criteria

Stop a GPU promotion if any of these happens:

- one unexplained trainer-visible parity mismatch;
- hidden CPU fallback;
- required host readback inside the intended hot loop;
- CUDA/subprocess topology issue;
- speedup disappears after reset/final-observation handling;
- semantics differ between trainer, tournament, eval, and GIF inspection.
