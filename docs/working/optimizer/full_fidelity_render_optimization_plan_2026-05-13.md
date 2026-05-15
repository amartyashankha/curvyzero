# Policy Observation Render Optimization Plan

Date: 2026-05-13
Updated: 2026-05-15

Scope: optimizer-owned render speed work for stock CurvyTron LightZero.

## Current Target

The current policy-observation target is `browser_lines + simple_symbols`.
CPU is the production oracle today; the intended speed target is a future
batched GPU implementation of the same surface.

That means:

- browser-like line trail geometry from source state;
- active bonuses drawn as `simple_symbols`;
- live heads drawn after bonuses;
- BT.601 luma and exact 11x11 area downsample to `uint8[1,64,64]`;
- FIFO stack to the stock LightZero `[4,64,64]` input contract.

CPU `browser_lines + simple_symbols` is the production backend and parity oracle
today. It is not the destination once a batched GPU path is proven in the
trainer.

`browser_sprites` are for artifacts, GIF/eval/reference views, and browser
fidelity work. `body_circles_fast` is historical ablation/control only.

## Working Rule

Make the target observation faster without changing what the policy sees.

The GPU path must match the CPU `browser_lines + simple_symbols` oracle for the
same real source-state rows. If the output is gray64, compare gray64 bytes. If
an intermediate RGB cache is introduced, compare that too.

Do not use browser golden-frame proof as the optimizer gate. Environment
Reconstruction owns browser-pixel claims.

## Current Read

- Full redraw is still the main render risk for long trajectories.
- GPU only helps if render output stays near policy/search or if transfer cost
  is clearly beaten.
- The next useful GPU work is real-state, policy-target parity:
  `browser_lines + simple_symbols`, not original sprite parity.
- End-to-end LightZero profiles decide whether render is still the bottleneck
  after collection, search, learner, replay, checkpoint, and artifact costs.

## Historical Probe

The 2026-05-13 `block_704_gray64` rows compared a synthetic GPU renderer to
production CPU `browser_lines`. Read these as geometry and transfer history, not
as current policy-target proof:

- `B=1`, `trail_slots=64`, no bonuses: exact byte parity on the tiny oracle.
- `B=1`, `trail_slots=64`, 8 bonuses: 51 of 4096 pixels differed, max diff 90.
- `B=16`, `trail_slots=64`, no bonuses: about 4.74ms device render and 3.34ms
  host-to-device copy on L4, without output readback.
- `B=2`, `trail_slots=256`, no bonuses: exact byte parity on the two-row oracle.
- `B=16`, `trail_slots=500`, no bonuses: about 46.7ms device render and 3.2ms
  host-to-device copy on L4.

Plain read: line/head geometry looked plausible, transfer was already visible,
and bonus handling was not the current target. These rows do not justify
replacing the current CPU oracle with `browser_sprites`, `body_circles_fast`, or
scalar GPU trainer calls.

## Next Gates

1. Lock a CPU oracle for `browser_lines + simple_symbols` on real source-state
   rows, with explicit metadata for trail and bonus modes.
2. Feed the GPU benchmark from real vector runtime state arrays.
3. Match GPU gray64 output to the CPU oracle for line, symbol, head, overlap,
   clipping, and downsample cases.
4. Measure transfer, render, readback, and device-resident handoff separately.
5. Run a stock LightZero full-loop profile before making a training-speed claim.
6. Keep `browser_sprites` in the artifact/reference lane unless explicitly
   measuring browser fidelity.

## Avoid

- Do not call CPU `browser_lines + simple_symbols` GPU rendering or the final
  speed destination; it is the production oracle today.
- Do not use `browser_sprites` as the policy-observation target.
- Do not use `body_circles_fast` as the recommendation target.
- Do not claim fidelity from synthetic GPU probes.
- Do not optimize the old custom two-seat trainer path.
