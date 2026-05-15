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

Pass condition: exact `[1,64,64]` policy-frame parity for every checked case.

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
CPU and GPU for hand-built states plus sampled real rollouts.

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

## Current Ranked Options

1. **Batched GPU observation boundary.** Highest ceiling if it can batch many
   rows/views and avoid scalar env CUDA. Risk: parity and host readback.
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
