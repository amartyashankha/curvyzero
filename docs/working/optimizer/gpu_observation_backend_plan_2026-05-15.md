# GPU Observation Backend Plan

Date: 2026-05-15

Purpose: keep the current optimizer lane clear. This is about making the
policy observation renderer faster, not changing CurvyTron game rules, training
claims, or live runs.

## Plain Current State

The policy observation contract is still:

```text
browser_lines + simple_symbols -> [4,64,64] grayscale stack
```

`browser_lines` is the semantic trail surface. It does not mean CPU-only. The
intended fast backend is a future batched GPU observation backend, not the
current scalar `policy_observation_backend=jax_gpu` canary. Any fast backend
must produce the same controlled-player view that the CPU oracle produces.

The current GPU prototype is:

```text
src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py
--render-surface block_704_gray64
--bonus-render-mode simple_symbols
```

That prototype does not write Modal Volumes and does not touch trainers. It
copies compact source-state arrays to a JAX CUDA function, computes the final
64x64 frame from 704-style 11x11 blocks, evaluates simple bonus symbols at the
same source-canvas sample positions, and can compare against the CPU oracle. It
does not materialize a full 704x704 image, but it still follows the same
source-canvas-then-downsample contract.

## What Is Proven

2026-05-15 real-env smoke:

```text
state_source=real_env_rollout
compute=gpu-l4-t4
batch_size=8
trail_slots=64
real_env_steps=32
controlled_player=0
bonus_count=1
render_surface=block_704_gray64
bonus_render_mode=simple_symbols
```

Result:

- JAX backend was `gpu`.
- Device render median was about `2.78ms` for 8 frames.
- End-to-end with host copies and readback was about `7.32ms` for 8 frames.
- CPU oracle for the one checked row was about `93ms`.
- Parity was not byte-exact: `8/4096` pixels differed.
- Every differing pixel was off by exactly `1` gray value.

Plain read: the lab GPU renderer can consume real CurvyTron state and is much
faster than the CPU oracle in isolation, but that is not yet a trusted trainer
backend. Follow-up fidelity review found a semantic browser line gap: the old
JAX path connected to the immediately previous trail slot, while the CPU oracle
connects to the previous active same-owner visual-trail point. Overlapping caps
can hide the error in aggregate parity rows, so the earlier "tiny rounding
only" read was no longer sufficient.

Follow-up real-env rows:

| GPU | B | trail slots | real env steps | controlled player | device render | end-to-end with readback | parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| L4 | 64 | 64 | 64 | 0 | `74.18ms` | `78.75ms` | 6 pixels off, max diff 1 |
| H100 | 64 | 64 | 64 | 0 | `7.87ms` | `10.88ms` | 6 pixels off, max diff 1 |
| H100 | 64 | 256 | 128 | 1 | `28.74ms` | `31.92ms` | 25 pixels off, max diff 2 |

Plain read: H100 matters a lot for the fused 704-style GPU renderer once the
batch has real work. L4 is still usable for development, but H100 is the better
hardware for a serious GPU-render observation backend. The longer-trail row
also confirms cost grows with trail history, so long-survival policies still
need either a future batched GPU backend or dirty/incremental thinking.

## What Is Not Proven

- The final fast GPU renderer is not wired into the stock LightZero trainer.
  A scalar `policy_observation_backend=jax_gpu` canary is wired and runs, but it
  is slower than CPU in the full trainer. Follow-up component profiles show the
  scalar GPU cost is dominated by static trail capacity: two-view H100 render
  is about `8.57ms` at 256 slots, `20.77ms` at 1024, `39.43ms` at 2048, and
  `72.20ms` at the production default 4096. Treat the full-trainer `~80ms`
  observation number primarily as a capacity/shape problem, with integration
  overheads as secondary suspects.
- The output is still copied back to host NumPy in the benchmark. A trainer
  integration may lose speed if it round-trips through CPU every step.
- Stock LightZero currently sees many scalar env instances. The GPU renderer is
  most attractive when it renders a batch of rows at once. A naive per-env
  `B=1` GPU call could erase the win with launch/copy overhead, especially in
  subprocess env managers.
- The isolated GPU benchmark has not yet produced a `[B, P, 4, 64, 64]` stack
  in the same object that the trainer consumes. A separate profile-only batched
  observation facade now exists for boundary measurement, but it is not a stock
  LightZero trainer backend and should not touch live runs.
- Historical caution: old JAX `browser_lines` semantics were not
  oracle-equivalent because segment connection used blindly adjacent slots. The
  lab renderer now carries previous active same-owner visual-trail records for
  the first H100 parity smokes, but the trainer path must prove the same before
  any GPU backend is used for training.
- The training Modal image can install JAX CUDA for the scalar canary, but that
  is not the same as having a production batched observation backend.
- The larger observation question (`96x96` or `128x128`) is separate and not a
  reason to delay the 64x64 GPU backend.

## Research Critique Update

The latest external/framework pass changes the order of work, not the target.
The target is still `browser_lines + simple_symbols`, but the current scalar
JAX path should be treated as a diagnostic canary only.

Plain read:

- CPU oracle remains the semantic anchor. A fast backend that changes trail
  connectivity, player perspective, bonus placement, or stack order is a
  different observation, not an optimization.
- GPU work is only compelling if it is batched and exact enough. Scalar JAX
  inside each env step pays launch/readback overhead and runs into subprocess
  CUDA trouble.
- The first GPU proof is an adversarial renderer parity test, especially
  interleaved owners, inactive holes, reset/cursor cases, and `break_before`.
- The next architecture proof is a one-batch renderer bakeoff, not a training
  run: CPU oracle versus JAX, PyTorch tensor/scatter, and possibly CuPy/CUDA on
  identical fixed states at B64/B256/B1024.
- The final proof is pipeline-level: render, stack, normalize, policy-forward
  stub, and replay copy in a mock collector shape. Isolated kernel speed is not
  enough if the result immediately round-trips through host memory.
- OpenGL/EGL/Skia-style renderers are useful as reference/parity tools, but
  they are probably not the trainer hot path unless they can batch cleanly and
  feed tensors without painful readback.
- EnvPool/Sample Factory/Isaac Gym point to the real lesson: large throughput
  comes from batching, compiled env work, and avoiding per-step host/device
  churn. It does not require every branchy env detail to be on GPU immediately.

Concrete next experiments:

1. Build a tiny adversarial state batch where raw slot adjacency is deliberately
   wrong: owner sequence like `[0, 1, 0]`, inactive holes, segment breaks, and a
   reset/cursor case. Fail any GPU renderer that does not match the CPU oracle.
2. Run a renderer bakeoff on that same fixed state family at B64/B256/B1024:
   CPU oracle, JAX with explicit previous-owner metadata, PyTorch tensor
   renderer, and CuPy/CUDA only if dependency/setup cost is acceptable.
3. Run a mock LightZero collector pipeline for the best candidate so the timing
   includes render, stack update, normalization, policy-forward stub, and replay
   copy. This decides whether the backend helps the loop, not just the kernel.

## Current GPU Lab Verdict

After the same-owner trail fix, benchmark harness cleanup, and owner-priority
composition fix, the H100 lab renderer finally matches the CPU oracle on the
current real-env smoke rows:

- B64/S1024/real-env512 one-view rows run at about `212ms` end-to-end for 64
  frames, or about `300` frames/sec, and have exact parity on the checked rows.
- The scalar fused two-view JIT is about `1.8x` faster than two separate scalar
  JAX renders for S1024.
- The scalar fused output matches both separate JAX and the CPU oracle exactly
  on the checked row for both views.
- The owner-priority fix made the B64 path slower than the previous
  non-equivalent renderer, but it removed the `max_abs_diff 18-23` mismatch.

Plain decision: still do not wire this into the trainer yet. The lab renderer
now passes the first real-env parity smoke, but it needs broader adversarial
parity and a real batched-boundary profile before it can replace CPU oracle in
training.

## Architecture Decision Tree

Keep `cpu_oracle` as the default unless the candidate path preserves the exact
policy observation contract:

```text
browser_lines + simple_symbols -> controlled-player [4,64,64]
```

Decision tree:

1. Need a trustworthy training run tonight?
   Use `cpu_oracle`. It is the reliable backend and already preserves the
   trainer/tournament observation surface.

2. Need to understand why full-trainer `jax_gpu` is slow?
   Keep the scalar backend, but treat it as a diagnostic lane only. First stop
   rendering dead capacity. Measure active visible trail slots per step, then
   compare fixed safe capacities, active-prefix compaction, and bucketed JIT
   shapes. Only after that chase secondary integration hazards such as Modal
   benchmark imports, unfused player views, JAX memory behavior, and broad
   observation timing scopes.

3. Need real speedup while staying close to stock LightZero?
   Do not jump straight to a service. First make the JAX browser-line renderer
   oracle-equivalent, including previous active same-owner connectivity. Then
   build the smallest semantics-preserving shape reduction: compact each row to
   its active trail prefix, pad to a bucket such as 256/512/1024/2048/4096, and
   render the bucketed shape. If that gets scalar H100 near CPU cost, then
   parent-side microbatching can amortize launch/readback and produce the first
   real stock-LightZero speedup.

4. Need the eventual production shape for long games?
   Use the profile-only batched observation facade to measure this boundary
   first. Eventual production likely moves observation ownership to a single GPU
   owner process/service or vector-env facade with fixed device buffers, frame
   stacks, reset/final-observation commands, and batched render calls. The
   policy should eventually consume the rendered stack without per-step host
   readback. This is the right architecture for long-survival games, but it is
   no longer a tiny stock LightZero change.

5. Should rendering happen inside `env.step`?
   Only for CPU oracle and scalar diagnostics. It is the simplest correctness
   boundary, because each returned LightZero timestep already contains the
   correct post-step observation and terminal final-observation context. It is
   also the worst GPU boundary: one small render per env step, poor batching,
   awkward JAX/PyTorch memory sharing, and subprocess CUDA failure risk.

6. Should rendering happen in a separate Modal function per step?
   No for training hot paths. Remote function overhead and serialization would
   dominate 64x64 observations. A long-lived in-process or colocated service can
   be sensible; per-step Modal RPC is a benchmark/control-plane tool, not the
   trainer architecture.

7. Should we skip GPU entirely?
   For short trajectories and low simulation counts, maybe. Amdahl says the
   full stock loop also pays MCTS, policy forward, replay, learner, collection,
   and reset/autoreset costs. If CPU render is a small fraction of wall, GPU
   render cannot move the whole loop much. For long trajectories with many
   trail slots, render grows with history and becomes worth isolating; the
   H100 batched rows are specifically promising for that future regime.

Smallest viable speed path: do not reduce physical env trail capacity unless a
separate gameplay/fidelity decision approves it. Instead render only active
visible trail data, padded to a small set of fixed JIT buckets, then profile the
stock full loop. If bucketed scalar still cannot beat `cpu_oracle`, build
parent-side microbatched render. Do not build a service before proving the
capacity issue is solved.

## Immediate Tasks

- [x] Add a real-env rollout state source to the GPU benchmark.
- [x] Add controlled-player self/other luma to the GPU benchmark.
- [x] Add mismatch samples when parity is not exact.
- [x] Decide whether the `max_abs_diff=1` parity gap should be fixed exactly or
  accepted behind a documented tolerance gate. Current decision: accept the
  observed one-luma edge-pixel float32 mismatch for the aggressive GPU
  candidate. Keep `geometry_dtype=float64` only as the exact-parity reference
  and diagnosis mode.
- [x] Fix and prove JAX `browser_lines` previous-point semantics: connect each
  visual trail point to the previous active same-owner point, respecting
  `break_before`, inactive slots, cursor/prefix compaction, owner grouping, and
  wrap/reset cases. Current best candidate is the owner-ordered compact
  two-view boundary, still lab-only.
- [x] Run both controlled-player views through the real-env benchmark.
- [x] Run at least one larger batch and one longer-trail real-env GPU row.
- [ ] Extract or wrap the renderer behind a separate batched backend name, for
  example `policy_observation_backend=jax_gpu_batched`. Keep `jax_gpu`
  documented as the scalar canary unless/until it is replaced. Before trainer
  promotion, the boundary must pass a full-loop canary, not only the
  profile-only sidecar.
- [ ] Add a fail-fast backend flag. No hidden fallback from `jax_gpu` to CPU.
- [x] Add a first timeout/autoreset profile row to the batched boundary. Current
  result: B64/S1024 x64 with `max_ticks=5` passes exact step,
  final-observation, and autoreset stack parity; terminal p95 is about `920ms`.
- [ ] Decide the trainer integration shape: batched render at an env-manager or
  collector boundary is likely better than a scalar per-env GPU call.
- [x] Profile the stock LightZero trainer with CPU oracle versus the scalar GPU
  backend.
- [ ] Split scalar full-trainer observation timing into state packing,
  host-to-device, render, readback, normalization/stack, and LightZero dict-copy
  buckets before drawing conclusions from the `~80ms` aggregate.
- [ ] Add active trail slot distribution telemetry for policy-observation rows
  at reset, short episodes, and long-survival/no-death profiles.
- [ ] Prototype render-only active-prefix compaction with fixed JIT buckets,
  preserving chronological draw order, owner grouping, break-before flags, and
  inactive-slot masking.
- [ ] Replace the scalar hook with a batched integration shape before treating
  GPU observation as a training default.

## Side Lane: Larger Observations

Keep production at `[4,64,64]` until a profile says otherwise.

The first sensible larger-observation experiment is `96x96`, because LightZero
has a stock branch for 96-sized Atari observations. `128x128` is probably a
model-code experiment, not a simple config change.

The question to answer later is not "can the renderer draw 96"; it is whether
the full loop stays fast enough after larger observations increase IPC, replay,
root representation, model memory, and possibly search-side latent cost.

First renderer-only H100 rows at B64/trail_slots128/real_env_steps128:

| output | device render | end-to-end with readback |
| --- | ---: | ---: |
| `64x64` | `14.80ms` | `18.35ms` |
| `96x96` | `15.62ms` | `18.89ms` |
| `128x128` | `26.33ms` | `29.47ms` |

Plain read: renderer-only economics make `96x96` plausible enough to profile
later. They do not justify changing the production model contract yet.

## Scalar-vs-Batched Integration Read

Do not wire GPU rendering into each scalar stock env just to say it is hooked
up.

Evidence:

- Local CPU dirty-cache env-only profile, `browser_lines + simple_symbols`,
  no-death, blank/noop opponent:
  - 128 steps: `0.389s` wall, `0.285s` render, about `2.23ms` render/step.
  - 500 steps: `1.365s` wall, `1.043s` render, about `2.09ms` render/step.
  - This render number includes the two scalar player-perspective frames that
    the wrapper keeps.
- H100 GPU B1 real-env row, one controlled-player view, trail_slots128:
  - device render `1.32ms`, but end-to-end with host copies/readback `3.84ms`.
  - This is for one view, not both player-perspective frames.
- H100 GPU B64 real-env row, one controlled-player view, trail_slots64:
  - end-to-end `10.88ms` for 64 frames.

Follow-up scalar component profile: the one-env, two-player scalar GPU shape is
capacity dominated on H100:

| trail slots | two-view render |
| ---: | ---: |
| 256 | `8.57ms` |
| 1024 | `20.77ms` |
| 2048 | `39.43ms` |
| 4096 | `72.20ms` |

Plain read: the full-trainer scalar canary is slow mostly because production
defaults ask the GPU to scan 4096 trail slots every step. The first architecture
fix is not a service; it is separating env storage capacity from render work.
Static smaller capacity is tempting but risky if it changes collision or trail
history semantics. Active-prefix compaction and bucketed JIT shapes are safer:
they preserve full env state while rendering only active visible trail data,
padded to fixed shapes that avoid JIT churn. Parent microbatching is still the
likely production speed path after this, especially for many scalar LightZero
envs, but it should consume compacted/bucketed rows rather than raw 4096-slot
state.

## Full-Loop Context

Fresh stock LightZero profiles on 2026-05-15 prove the current loop, but they do
not prove the final batched GPU renderer integration.

The measured full-loop path was:

```text
lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode profile
--env-variant source_state_fixed_opponent
--source-state-trail-render-mode browser_lines
--source-state-bonus-render-mode simple_symbols
--num-simulations 8
```

On L4/T4+40CPU, C8 collected 4,096 env steps in `36.60s`
(`111.9` steps/s). C32 collected 16,384 env steps in `50.34s`
(`325.5` steps/s). C32 is therefore a much better throughput shape than C8 in
this current code state, but the wall still includes large collector,
policy/MCTS, replay, and learner buckets.

Env-only long-rollout Amdahl is different: local no-death profiles from 100 to
2,000 steps show render is about `77%` of a single-env rollout wall. If render
became free, that narrow env-only loop could improve by about `4.3x-4.5x`; if
render became 10x faster, env-only rollout speed would improve about
`3.2x-3.3x`.

Plain read: render remains worth optimizing for long-survival policies, but the
current stock LightZero full loop will not get a 3x win from a scalar renderer
swap alone. The useful GPU direction is batched observation rendering plus fresh
full-loop profiling, not a per-env B1 GPU call.

Detailed table:
[full-loop Amdahl reorientation](full_loop_amdahl_reorientation_2026-05-15.md).

## Scalar GPU Full-Loop Canary

The scalar `jax_gpu` path is now wired through the canonical stock LightZero
launcher and reaches `lzero.entry.train_muzero`. It is not a speedup.

Matched H100/base/C1/sim2/no-death profiles:

| backend | steps | wall | steps/s | observation mean | MCTS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cpu_oracle` | 64 | `11.65s` | `5.49` | `8.64ms` | `0.37s` |
| `jax_gpu` | 64 | `19.82s` | `3.23` | `79.44ms` | `0.27s` |
| `cpu_oracle` | 512 | `15.54s` | `32.94` | `4.42ms` | `2.50s` |
| `jax_gpu` | 512 | `63.73s` | `8.03` | `80.31ms` | `2.55s` |

The subprocess scalar canary failed before collection with JAX CUDA
initialization errors inside env workers.

Plain read: GPU math is not the problem. The scalar hook pays per-step
JAX/copy overhead, uses about `62GB` on H100 in base mode, and is unsafe in the
subprocess env-manager shape. Keep `cpu_oracle` for stock training until there
is a batched GPU observation boundary.

Full catalog:
[GPU observation full-loop canary](gpu_observation_full_loop_canary_2026-05-15.md).

## Validation

Latest local validation after the real-env benchmark instrumentation and
contract wording cleanup:

```text
uv run ruff check src/curvyzero/env/observation_surface_contract.py \
  src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py
uv run pytest tests/test_env_contract.py \
  tests/test_vector_visual_observation.py::test_source_state_gray64_renders_optional_bonus_geometry -q
uv run pytest tests/test_source_state_visual_survival_learner_seat_regression.py \
  tests/test_multiplayer_source_state_trainer_surface.py \
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif -q
```

Result: ruff passed, `9 passed`, and `17 passed`.
