# Batched GPU Observation Boundary Plan

Date: 2026-05-15

Status: profile-only optimizer plan. Do not wire into trainers, tournaments,
checkpoints, eval, Modal Volumes, or live runs until the batched backend exists.

## Current Result

The profile-only sidecar now exists:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
```

The important result is simple:

- `geometry_dtype=float32` is the aggressive default for the GPU candidate. It
  is faster and the observed mismatch is tiny: one luma at one 64x64 edge pixel
  in the B64/S1024 later-step check. This is acceptable for a learned policy
  observation unless a future test shows missing objects, wrong ordering, or
  unstable symbols.
- `geometry_dtype=float64` is the exact-parity reference/debug mode. Keep it for
  contract tests and diagnosis, not as the default speed candidate.
- The sidecar can now force timeout terminals with `max_ticks`, capture the
  terminal stack as `final_observation`, explicitly autoreset terminal rows,
  render the reset frame, and charge those costs separately.

Plain default: use `geometry_dtype=float32` for the aggressive GPU observation
candidate. Use `float64` only when the question is exact CPU-oracle parity.

Important blocker: this does **not** mean setting production
`policy_observation_backend=jax_gpu` as-is. The wired trainer backend named
`jax_gpu` is scalar and slow. The fast path measured here is batched and still
profile-only.

## Plain Goal

The isolated H100 renderer can now render both player views with exact checked
CPU parity, but that is still only a renderer proof. The next useful proof is a
profile-only boundary that measures the real observation work around the
renderer:

```text
env state -> compact render state -> GPU render both views -> host frames
-> row-major player stacks -> final/reset handling
```

This boundary should answer one question:

> If the renderer itself is faster on GPU, does the whole observation boundary
> still win after packing, copies, view ordering, stack update, reset, and final
> observation are charged?

## Fixed First Experiment

Use a separate Modal sidecar, not trainer code:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
```

First configs:

- `batch_size`: `64`, then `256`;
- `player_count`: `2`;
- `render_mode`: `browser_lines`;
- `bonus_render_mode`: `simple_symbols`;
- `render_surface`: `block_704_gray64`;
- `trail_composition`: `owner_ordered_compact`;
- `render_views`: `both`;
- `geometry_dtype`: `float32` for the aggressive candidate; `float64` only for
  exact CPU-oracle parity/debug rows;
- `frame_size`: `704`;
- `target_size`: `64`;
- `transfer_output`: `true`;
- `death_mode`: no-death throughput first, tiny terminal row second.
- `max_ticks`: default `2000`; set a small value such as `3` or `5` only for
  timeout/autoreset profile rows.

## Shape Contract

The fused GPU renderer returns view-major frames:

```text
gpu[0:B]   = player 0 views for rows 0..B-1
gpu[B:2B]  = player 1 views for rows 0..B-1
```

The boundary must convert that to row-major before stack update:

```text
row_major[2*r + 0] = gpu[r]
row_major[2*r + 1] = gpu[B + r]
```

The trainer-like stack target for this profile is:

```text
uint8 raw frames:     [B, 2, 1, 64, 64]
float32 stacks:       [B, 2, 4, 64, 64]
```

The CPU profile facade should own this row-major contract first. The GPU sidecar
should compare against it exactly.

## Timing Buckets

Report medians and p95 for steady iterations. Keep compile/warmup separate.

- `env_step_sec`
- `production_to_compact_sec`
- `owner_ordered_pack_sec`
- `host_to_device_sec`
- `device_render_sec`
- `device_to_host_sec`
- `view_major_to_row_major_sec`
- `stack_sec`
- `reset_sec`
- `final_obs_sec`
- `candidate_total_observation_sec`
- `candidate_total_step_plus_observation_sec`
- `cpu_reference_render_stack_sec`

`cpu_reference_render_stack_sec` is parity cost, not candidate cost. Do not fold
setup, stack, reset, or final observation into render time.

## Parity Gates

Hard fail on:

- any shape, dtype, row order, player view, or controlled-player perspective
  mismatch;
- any missing object, wrong symbol, wrong draw order, or hidden CPU fallback;
- any stack mismatch after reset or step that is larger than the selected
  parity mode allows;
- player-view order mismatch;
- controlled-player perspective mismatch;
- reset history leak;
- terminal `final_observation` not copied before reset/autoreset;
- hidden CPU fallback;
- trainer/tournament/default change.

Use `geometry_dtype=float64` with `parity_mode=exact` for debug/reference rows.
Use `geometry_dtype=float32` with `parity_mode=tolerant` for the aggressive
learned-observation speed candidate. Tolerant mode is only for bounded
one-to-two-luma edge drift; it is not permission to accept semantic changes.
Mismatch samples should include logical row, player view, channel, `y`, `x`,
GPU value, CPU value, absolute difference, mismatch count, and max diff.

## Timing Read

Use the isolated H100 S1024 two-view rows only as a historical baseline:

- B64 owner-ordered compact: `251.51ms` H2D + render + readback, plus `5.7ms`
  one-shot owner-ordered packing outside render timing.
- B256 owner-ordered compact: `1142.06ms` H2D + render + readback, plus
  `19.6ms` one-shot owner-ordered packing outside render timing.

Actual boundary rows now exist. H100, S1024, no-death, both player views,
owner-ordered compact:

| Batch | Geometry | Parity | Candidate Observation | CPU Reference Render+Stack | Read |
| --- | --- | --- | ---: | ---: | --- |
| B64 | float32 | tiny one-luma mismatch in prior deeper check | `255ms` | `654ms` | aggressive default candidate |
| B64 | float64 | exact reset + 4 checked steps | `379ms` | `1.09s` | `~2.9x` exact-boundary win |
| B128 | float64 | exact reset + 1 checked step | `713ms` | `1.43s` | `~2.0x` exact-boundary win |
| B256 | float32 | parity disabled after reset in speed row | `1.14s` | `2.48s` | aggressive default candidate, needs a checked row |
| B256 | float64 | exact reset + 2 checked steps | `1.38s` | `2.79s` | `~2.0x` exact-boundary win |

2026-05-20 tooling update: the sidecar now exposes `parity_mode`,
`parity_max_abs_diff`, and `parity_max_mismatch_fraction`. Default `auto`
selects exact parity for float64 and tolerant parity for float32. This means
future float32 speed rows should run with `verify_steps > 0` instead of
turning step verification off.

Float32 versus float64 speed read:

- B64/S1024 candidate observation: `255ms` versus `379ms`, so float32 is about
  `1.5x` faster than float64.
- B256/S1024 candidate observation: `1.14s` versus `1.38s`, so float32 is about
  `1.2x` faster than float64.

The boundary is still render-dominated. At B64 x64, device render was about
`364ms` of `374ms`; at B256 x64, device render was about `1.35s` of `1.38s`.
Packing, H2D, readback, row-major conversion, and stack update are currently
small. Amdahl read: if this lane becomes production-worthy, renderer kernel
work remains the highest-leverage optimization inside the observation boundary.

Timeout/autoreset gate, H100, S1024, `geometry_dtype=float64`:

| Batch | max ticks | Parity | Median Candidate Observation | Candidate p95 | Read |
| --- | ---: | --- | ---: | ---: | --- |
| B16 | `3` | exact reset + 4 checked steps + terminal/autoreset stack | `21.7ms` | `86.6ms` | tiny terminal smoke; 2 terminal steps / 32 terminal rows |
| B64 | `5` | exact reset + 4 checked steps + terminal/autoreset stack | `376ms` | `920ms` | terminal steps pay final-observation copy plus reset render/stack |

Plain read: timeout terminal/autoreset ordering is no longer completely
unmodeled in the sidecar. This is still not a natural-death/reward trainer
proof, and it still reads frames back to host before stack update.

## Precision Finding

The B64/S1024 `float32` boundary failed exact parity at logical row `28`, player
view `0`, pixel `(y=46, x=14)`: GPU `100`, CPU `101`. A read-only critique
confirmed the likely cause: CPU line geometry uses float64 in
`vector_visual_observation.py`, while the compact GPU path had truncated trail,
head, and bonus geometry to float32. A one-source-pixel coverage difference in
an 11x11 downsample block can move the final rounded luma by one.

The `float64` geometry variant fixes the observed failure. Current optimizer
decision: that level of mismatch is acceptable for the aggressive policy
observation candidate. Do not silently accept larger mismatches, missing
objects, wrong player perspective, wrong draw order, or symbol collisions.

## Explicit Non-Goals

- Do not confuse the scalar trainer backend with the batched candidate. Scalar
  `policy_observation_backend=jax_gpu` is still measured slow; making that the
  default would be a regression.
- Do not touch stock LightZero trainer, tournaments, eval loading, replay,
  checkpoints, or Modal Volumes.
- Do not touch live Modal jobs or training runs.
- Do not promote scalar `policy_observation_backend=jax_gpu`.
- Do not fallback to body circles, adjacent-slot trails, browser sprites, or a
  global controlled-player closure.
