# Optimizer Active Working Memory

Date: 2026-05-14

Purpose: short live state for the optimizer lane. Read this before older
optimizer notes if the docs disagree.

## Role

Optimizer owns speed, profiling, setup advice, and renderer/search performance
experiments. Coach owns learning claims and which checkpoints are good. Do not
touch live overnight training runs except read-only.

## Current Trusted Path

Training/profiling source of truth is stock LightZero:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
```

Current runs storage default is the fresh Modal Volume `curvyzero-runs-v2`.
The shared source of truth is
`src/curvyzero/contracts/curvytron.py::DEFAULT_CURVYTRON_RUNS_VOLUME_NAME`.
The old `curvyzero-runs` volume still exists only as historical/read-only
material unless a separate exact-delete decision is made.

2026-05-15 strict contract: the target policy observation surface is
`browser_lines + simple_symbols` everywhere that trains, rates, or runs
policies. The source of truth is
`src/curvyzero/env/observation_surface_contract.py`. `browser_lines` names the
semantic trail surface; it does not mean CPU-only. GPU rendering must implement
that same surface through a separate backend flag/path, not by changing the
policy mode. CPU `browser_lines + simple_symbols` is the production backend and
parity oracle today. The intended replacement is a future batched GPU backend,
not scalar `jax_gpu`. The only deliberate policy-observation approximation is
replacing original bonus sprite art with the 12 designed simple symbols; trail
geometry, heads, controlled-player view, grayscale, and downsample semantics
should match the CPU oracle. The controlled-player/seat contract is Coach-owned
and summarized in
`docs/working/training/leaderboard_to_training_2026-05-13/policy_observation_perspective_contract_2026-05-15.md`;
Optimizer only implements faster backends for that requested view. Current
policy paths reject
`body_circles_fast`, `browser_sprites`, `fast_gray64_direct`, and any
`allow_legacy_policy_surface` escape hatch.

The CPU parity oracle is `browser_lines + simple_symbols`:

```text
source-state RGB 704-style canvas
-> simple-symbol bonuses
-> BT.601 luma
-> 11x11 area average
-> uint8[1,64,64]
-> [4,64,64] stack for LightZero
```

This is CPU-reference fidelity, not browser-pixel fidelity. The optimizer target
is to make a future batched GPU backend match this oracle closely enough, then
use that backend for training/profile commands.

Training/profile commands should be explicit about the semantic surface, or
import the shared constants:

```text
--source-state-trail-render-mode browser_lines
--source-state-bonus-render-mode simple_symbols
```

The old `auto`/browser-sprite default was removed from the current policy
surface. Raw RGB/GIF artifacts may still use richer visual paths for display,
but policy observations and tournaments should name `simple_symbols`.

Backend truth: `compute=gpu-*` puts LightZero model/search/learner on GPU. It
does **not** by itself move the CurvyTron observation renderer to GPU. The
explicit backend flag is `policy_observation_backend`, with values
`cpu_oracle` and `jax_gpu`. The scalar `jax_gpu` canary is wired through the
canonical trainer and reaches `lzero.entry.train_muzero`, but it is currently
slower than CPU because it renders one env step at a time and copies frames back
to NumPy. Do not overload `source_state_trail_render_mode` to mean GPU.

2026-05-15 GPU lab update: the isolated H100 `block_704_gray64` renderer now
matches the CPU oracle exactly on the checked real-env smoke rows after the
same-owner, cursor, avatar-color reference, and owner-priority fixes. The latest
S1024 rows are still expensive: B64 is about `212ms` for one view and B256 is
about `735ms` for one view. The shorter S256/B64 row is about `59ms`. This
means the GPU path has become a correctness candidate, not a speed recommendation
yet. A profile-only batched observation facade now exists; use it for boundary
measurement while stock training remains on `cpu_oracle`. Gate list:
[GPU observation next gates](gpu_observation_next_gates_2026-05-15.md).
First adversarial renderer slice now passes too: `adversarial_fixture` on H100,
3 players, 12 symbols, non-identity/duplicate/high `avatar_color`, controlled
players `0/1/2`, exact `0`-mismatch parity against CPU oracle. This is still
frame parity, not trainer-contract parity.
CPU oracle cleanup also landed: simple-symbol RGB stamping now touches only the
local symbol crop instead of allocating a full-frame scratch canvas, and dirty
cache fallback reasons are recorded outside full timing profiles. This keeps the
trusted backend faster and easier to debug while the GPU path remains gated.
Tournament contract cleanup also landed: policy evaluation still selects
`observation[0, seat]` and `action_mask[0, seat]`, but LightZero `to_play` is
now `-1` to match the stock controlled-player training env. The seat is encoded
by the selected view/action mask and metadata, not by `to_play`.
Frozen checkpoint opponent inference has the same rule now: the opponent policy
gets its own controlled-player visual/action-mask slice, while LightZero
`to_play` is `[-1]`. Do not reintroduce player-id `to_play` for current
non-board-game source-state rows.

Latest critique synthesis:
[parallel critique synthesis](parallel_critique_synthesis_2026-05-15.md).

## Current Speed Read

- Fresh stock LightZero full-loop rows on 2026-05-15 used the current trusted
  surface, `browser_lines + simple_symbols`, through
  `lzero.entry.train_muzero`. They are the current speed reference. C8/sim8
  collected 4,096 env steps in `36.60s` (`111.9` steps/s). C32/sim8 collected
  16,384 env steps in `50.34s` (`325.5` steps/s). See
  [full-loop Amdahl reorientation](full_loop_amdahl_reorientation_2026-05-15.md).
- Fresh env-only no-death profiles from 100, 200, 500, 1000, and 2000 steps
  say render is stable at about `77%` of a single-env rollout wall for the
  current `browser_lines + simple_symbols` surface. If render became free, that
  narrow loop could improve about `4.3x-4.5x`; if render became 10x faster, it
  would improve about `3.2x-3.3x`.
- Amdahl read: render still matters a lot for long single-env trajectories, but
  current stock LightZero full-loop wall is less renderer-pure because env
  workers run in parallel and collector/policy/MCTS/replay/learner work remains
  on the critical path. C32/sim8 spent `37.78s` in collector collect, with
  `22.43s` policy collect and `14.55s` MCTS/search buckets, inside a `50.34s`
  wall. Do not promise a 3x full-loop win from renderer work alone until the
  batched GPU backend is actually wired and profiled.
- Bigger GPUs are not automatically better. H100 helped more when search sims
  were heavier, but the sim8 full-loop rows did not justify H100 by default.
- Fresh stock fast-path grid, batch32/no-death/source512: C64/L4/sim8 fast V8
  measured `591.6` env steps/sec, versus a matched C64/L4/sim8 browser
  reference at `491.4` env steps/sec. C384/L4/sim8 fast V8 measured `946.1`
  env steps/sec and then wider L4 rows dropped. C256/H100/sim8 measured
  `1081.9` env steps/sec, and C768/H100/sim8 measured `1204.0`, the fastest
  speed-only row so far. This does not make C768 a learning default; it is a
  speed/aggression probe. See
  [stock fast-path scaling grid](stock_fast_path_scaling_grid_2026-05-14.md).
- Superseded Coach-facing recommendation:
  [fast stock recommendation](coach_handoff_fast_stock_recommendation_2026-05-14.md).
  Plain read: keep its speed numbers as historical evidence, but do not copy
  its `body_circles_fast` launch commands into new runs.
- Scalar GPU observation canary, H100/base/C1/sim2/no-death:
  `cpu_oracle` at 512 steps was `15.54s` wall (`32.94` steps/s) with
  `4.42ms` observation/step; `jax_gpu` was `63.73s` wall (`8.03` steps/s) with
  `80.31ms` observation/step. The scalar GPU hook is a proof gate, not the
  production optimization. Full catalog:
  [GPU observation full-loop canary](gpu_observation_full_loop_canary_2026-05-15.md).
- Operational risk: the profile Modal container warned that `/runs` volume inode
  usage was about `97.7%`. Clean or route future artifacts before serious long
  runs.

## Active Plates

1. **Bonus symbol renderer**
   Implemented and wired as the current policy bonus encoding. The production
   lane is CPU `cpu_oracle` `browser_lines + simple_symbols`; the future GPU
   lane must preserve the same 3 outer shapes x 4 inner marks, high-contrast
   luma by shape, row-specific asymmetric marks, and minimum 7x7 footprint.
   Browser sprites are display/reference artifacts, not the default policy
   surface.
   Current artifact:
   `artifacts/local/curvytron_render_profiles/bonus_simple_symbols_actual_v8_20260514.png`.
   Independent visual and numeric critiques agree that the direct fast path has
   no class collisions in the tested center/offset/radius/edge/remap sweeps.
   V8 keeps thick horizontal/vertical marks and gives the three X variants
   different asymmetry. The raw 7x7 nearest-pair floor is pinned at L1 `>=1300`
   and mismatch pixels `>=10`. Caveat:
   `simple_symbols` must be tested in the same gray64 canvas/downsample path
   used by the future batched GPU backend and CPU oracle before it is treated
   as production equivalent.
   Trail overlay is tested: all 12 symbols overwrite both direct
   `visual_trail_*` and fallback `body_*` trail pixels while staying distinct.
   Do not alpha-blend the symbol body unless a future test proves it helps;
   blending makes the same bonus vary with whatever trail is underneath.
   Current design note:
   [bonus symbol render plan](bonus_symbol_render_plan_2026-05-14.md).

2. **GPU browser-lines renderer**
   The lab GPU renderer is promising but not a trainer backend. After the
   same-owner trail and owner-priority fixes, first H100 smoke rows match the
   CPU oracle exactly on checked rows; adversarial parity and a real batched
   LightZero boundary profile are still required for the current
   `simple_symbols` target. Current note:
   [GPU render parity gap](gpu_render_parity_gap_2026-05-13.md).
   Current implementation plan:
   [GPU observation backend plan](gpu_observation_backend_plan_2026-05-15.md).
   Correction as of 2026-05-15: the next practical GPU target is **not**
   original browser sprites. It is faithful GPU `browser_lines` trail/head
   geometry plus `simple_symbols` bonus encoding. This keeps the important
   browser-line trail surface while avoiding the harder RGBA sprite parity
   problem. This was implied by the GPU-render and simple-symbol docs but was
   not promoted into the active implementation list; treat that as an optimizer
   miss, not a settled decision against the lane.
   Fresh isolated benchmark: `block_704_gray64` still outputs final `64x64`;
   it just computes each output pixel from 11x11 high-resolution sample
   positions to match the CPU oracle. On 2026-05-15, `browser_lines +
   simple_symbols` hit exact checked parity on synthetic CPU-oracle rows. H100
   B64/S64 measured `9.85ms` device render versus L4 `81.7ms`; H100 B64/S256
   measured `31.0ms` versus L4 `291.0ms`. Host-to-device copy stayed around
   `3ms`. Plain read: GPU render is real, but this was pre owner-priority fix
   smoke evidence; LightZero boundary integration remains a blocker.

   Fresh real-env rollout smoke, 2026-05-15: L4, B8, trail_slots64,
   real_env_steps32, controlled_player0, one active simple-symbol bonus. JAX
   used GPU. Device render median was about `2.78ms` for 8 frames, end-to-end
   with host copies/readback was about `7.32ms`, and the CPU oracle for the one
   checked row took about `93ms`. Parity was `4088/4096` exact; the remaining 8
   pixels were all off by one gray value. Plain read: real-state GPU rendering
   works and is fast in isolation; the next questions are tiny rounding parity,
   both-seat coverage, and trainer backend integration.

   Follow-up real-env rows: L4 B64/S64 took about `74.18ms` device and
   `78.75ms` end-to-end; H100 B64/S64 took about `7.87ms` device and
   `10.88ms` end-to-end; H100 B64/S256 with controlled_player1 took about
   `28.74ms` device and `31.92ms` end-to-end. The checked rows still had only
   tiny parity gaps (`6` pixels max diff `1`, then `25` pixels max diff `2` for
   the longer trail row). Plain read: H100 is the serious hardware target for
   the GPU renderer, and the remaining blocker is integration/fidelity, not raw
   GPU math.

   Integration guardrail: do not put a B1 JAX call inside every scalar stock
   env. The stock LightZero scalar canary is now measured and is slower:
   `jax_gpu` was about `4.1x` slower than `cpu_oracle` at 512 steps. The
   promising shape is batched GPU rendering, not scalar GPU round trips.
   Stronger critique, 2026-05-15: parity now outranks batching. The old GPU
   `browser_lines` benchmark connected each raw trail slot to `slot-1`; the
   current lab renderer now carries the previous active same-owner point. It
   also carries a high-resolution owner-priority buffer so overlapping trails
   follow CPU owner draw order instead of slot order or max luma. Fresh H100
   B64/S1024 real-env rows now match the CPU oracle exactly on the checked
   rows, at about `212ms` end-to-end for 64 frames. Scalar fused two-view
   rendering also matches CPU exactly on the checked row and is about `1.8x`
   faster than two separate scalar JAX renders. Keep `cpu_oracle` as the
   production backend until adversarial parity and a real batched boundary
   profile are complete.

3. **Full-loop scaling**
   Keep measuring stock LightZero with clear buckets. Relevant knobs are
   collector width, search sims, learner batch, CPU count, GPU class, telemetry
   stride, and trajectory length. Do not infer learning quality from speed
   profiles. Current C32/sim8 is much faster than C8/sim8 by throughput, but
   the next recommendation needs C64/C96 and sim16 rows before choosing a final
   Coach config.

4. **Moving environment**
   Environment Reconstruction is still changing source-state/render details.
   Every speed number must name code state, render mode, bonus mode, warmup,
   trajectory length, and whether search/learner are included.

## To Do

- [x] Harvest sprite inventory and practical symbol-design subagents.
- [x] Write down the 12-bonus symbol plan and Amdahl implication.
- [x] Harvest the running signature-probe subagent and fold results into the
  symbol plan.
- [x] Add an explicit opt-in `simple_symbols` bonus render mode for the direct
  fast path.
- [x] Gate the fast path with centered, offset, radius, remap, metadata, and
  render/wrapper tests.
- [x] Rerun local env-only profile after symbol implementation.
- [x] Harvest the independent actual-symbol visual and numeric critique agents.
- [x] Add explicit stock training flag/metadata:
  `source_state_bonus_render_mode`.
- [x] Collect the fresh waited stock full-loop profile pair.
- [x] Harvest the active wider-collector stock fast-path grid and write the
  Coach-facing recommendation. Current grid:
  [stock fast-path scaling grid](stock_fast_path_scaling_grid_2026-05-14.md).
- [x] Feed real env rollout rows into the isolated GPU render benchmark and
  verify controlled-player GPU output against the CPU oracle.
- [x] Run fresh stock LightZero full-loop C8/C32 profiles on the current
  `browser_lines + simple_symbols` surface and write the current Amdahl read:
  [full-loop Amdahl reorientation](full_loop_amdahl_reorientation_2026-05-15.md).
- [x] Wire an explicit scalar `policy_observation_backend=jax_gpu` canary into
  the canonical stock LightZero trainer and compare it to `cpu_oracle`.
- [ ] Build the batched faithful-GPU-plus-simple-symbols lane:
  CPU reference target is `browser_lines + simple_symbols`; the scalar trainer
  hook proves plumbing but is too slow, so the next useful implementation is a
  batched env-manager/collector observation boundary or render service.
- [ ] Before GPU batching work, fix/prove GPU `browser_lines` semantics against
  adversarial interleaved-owner, inactive-hole, reset/cursor, and `break_before`
  fixtures. No training recommendation should use GPU observation before this.
- [ ] Run the next stock full-loop matrix before final Coach run settings:
  C32/C64/C96 at sim8 on L4/T4+40CPU; C64 sim16 on L4/T4 and H100; one
  telemetry-light row to estimate profiling overhead.

## Operating Pattern

- Main thread is for planning, delegation, orchestration, and short synthesis.
- Use subagents for bounded code reading, toy probes, literature/research, and
  critique.
- Keep docs current before details fall out of context.
- Do not change live training runs. New renderer experiments must be opt-in.
