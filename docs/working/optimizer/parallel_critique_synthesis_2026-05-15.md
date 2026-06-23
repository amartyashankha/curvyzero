# Parallel Critique Synthesis

Date: 2026-05-15

Status: active optimizer working note.

## Current Plain Truth

The production policy observation surface is:

```text
browser_lines + simple_symbols -> cpu_oracle -> [4,64,64] stack
```

`browser_lines` is the semantic trail surface. It is not a CPU/GPU hardware
claim. `cpu_oracle` is the current trusted backend. GPU rendering is still a
lab/profiling lane until it passes the trainer-visible contract, not only a
single-frame parity test.

The source-state training env now rejects `body_circles_fast` as a
trainer-facing trail mode. Lower-level visual-observation tests may still keep
that renderer as a historical/control surface, but it must not leak into new
trainer, tournament, or Coach recommendations.

## What Changed In This Round

- A profile-only batched observation facade now exists:
  [source_state_batched_observation_profile.py](/Users/shankha/curvy/src/curvyzero/training/source_state_batched_observation_profile.py).
  It owns a local vector-env batch, updates `[B,4,64,64]` stacks from the CPU
  oracle, and exposes timing slots for `pack`, `render`, `readback`, `stack`,
  `reset`, and `final_obs`. It does not call `train_muzero`, does not touch live
  runs, and does not change trainer defaults.
- The trainer-facing source-state env no longer has the direct
  `body_circles_fast` gray64 branch. Attempts to pass
  `source_state_trail_render_mode=body_circles_fast` fail early.
- Frozen checkpoint opponent inference now uses LightZero `to_play=[-1]`.
  The opponent still receives the correct per-player visual/action-mask slice;
  `player_id` is routing metadata, not the non-board-game `to_play` value.
- The whole-loop critique now points at a missing measurement layer:
  per-step `info` construction, `BaseEnvTimestep` payload size, subprocess IPC,
  reset/final-observation copying, root-batch fragmentation, replay object
  churn, and artifact tax can hide outside current render/MCTS buckets.
- The GPU-render critique found the next concrete render experiment:
  owner-ordered compact rendering with no priority buffer. The current exact
  renderer is slower because it maintains a high-resolution owner-priority
  buffer. If compact trails are already in CPU draw order, overwrite should
  preserve parity with less memory traffic.
- First H100 check confirms the idea is worth pursuing. On `real_env_rollout`,
  B64/S1024, controlled player 0, `owner_ordered_compact` matched the CPU
  oracle exactly on checked rows and cut device render from `208.9ms` to
  `135.5ms` (`1.54x`). End-to-end with host copies/readback went from
  `214.0ms` to `140.4ms` (`1.52x`). This is still isolated renderer evidence,
  not trainer promotion.
- Follow-up rows held up: B256/S1024 improved `729.6ms -> 391.4ms` device
  (`1.86x`) and `734.4ms -> 395.5ms` end-to-end (`1.86x`), while B64/S256
  improved `54.2ms -> 36.0ms` device (`1.51x`) and `57.2ms -> 40.3ms`
  end-to-end (`1.42x`). All checked rows had exact CPU-oracle parity.
- The same H100 benchmark now has a two-view mode. It returns frames in
  `view_major` order: all player-0 rows, then all player-1 rows. That is fine
  for renderer timing, but a future trainer boundary must reorder into
  `[batch, player, channel, height, width]` before stack updates. B64/S1024
  two-view rows improved from `494.18ms` to `251.51ms` H2D + render + readback
  (`~1.96x`) with owner-ordered compact. B256/S1024 improved from `1844.97ms`
  to `1142.06ms` (`~1.62x`). Exact checked CPU-oracle parity passed.
- The phrase "end-to-end" in old renderer rows means H2D + render + optional
  readback only. It does not include env stepping, production-state conversion,
  owner-ordered packing, stack update, reset, final observation, policy/search,
  replay, learner, checkpointing, or eval.
- The new profile-only batched boundary sidecar closes that measurement gap for
  observation work. It found a tiny float32 parity gap: one luma at one
  B64/S1024 edge pixel in the checked failure sample. Current decision:
  `geometry_dtype=float32` is acceptable as the aggressive GPU candidate;
  `geometry_dtype=float64` is the exact-parity reference/debug mode. H100
  boundary rows: B64/S1024 float32 `255ms` candidate observation vs `654ms` CPU
  reference render+stack; B64/S1024 float64 `379ms` vs `1.09s`; B128 float64
  `713ms` vs `1.43s`; B256 float32 `1.14s` vs `2.48s`; B256 float64 `1.38s`
  vs `2.79s`. This is still lab-only and host-readback based, but it is now an
  honest observation-boundary result.
- The boundary sidecar now has a timeout/autoreset gate. B64/S1024 x64 with
  `max_ticks=5` passed exact step, terminal final-observation, and autoreset
  stack parity. Median observation stayed about `376ms`; p95 was about `920ms`
  because terminal steps include final-observation copy plus reset render/stack.

## Highest-Value Critique Threads

1. **Info/IPC falsifier.**
   Add a profile-only `full|minimal` info-payload mode or equivalent counters.
   If wall/collector drops by at least `15%`, render is not the next best
   default target.

2. **Owner-ordered GPU renderer.**
   Prototype a benchmark-only `block_704_gray64_ordered_overwrite` path. Pass
   requires exact adversarial parity and at least `1.25x` speedup over the
   current exact priority-buffer renderer at H100 B64/S1024.
   Current status: pass on the first real-env B64/S1024 H100 row; wider B256
   and shorter S256 rows also passed exact checked parity and improved
   `1.4x-1.9x`.

3. **Tile-sparse renderer.**
   Bigger and riskier. Use Triton/CuPy/Pallas/CUDA only after the ordered
   overwrite path is tested. Target is exact 11x11 sampling only for blocks that
   intersect candidate trail segments.

4. **Trainer-visible GPU contract parity.**
   Frame parity is not enough. Reset stack, terminal `final_observation`,
   controlled-player perspective, action masks, `to_play=-1`, reward/done/info,
   and tournament checkpoint loading must match before promotion.

5. **Batched two-view boundary.**
   The next useful GPU proof is not another scalar renderer row. It is a
   profile-only boundary that renders both player views, reorders from
   `view_major` to row-major, updates `[B,2,4,64,64]` stacks, copies terminal
   final observations, handles reset rows, and reports every component cost.
   Current plan:
   [batched GPU observation boundary plan](batched_gpu_observation_boundary_plan_2026-05-15.md).
   Current status: implemented. Float32 is the aggressive candidate and has
   only a tiny observed one-luma edge mismatch; float64 is the exact reference.

6. **Metadata identity drift.**
   First pass fixed: checkpoint specs, pair/game specs, rating context, rating
   pool hash, and rating roster now carry `policy_observation_backend`, and
   tournament/rating reject lab `jax_gpu` by default. Remaining audit work is to
   verify checkpoint payload extraction and downstream public leaderboard
   consumers preserve this field everywhere it matters.

7. **Collector/search scaling.**
   Wider collector rows help, but root batching and policy/search boundaries may
   fragment. H100 is only worth revisiting when search pressure or batch width
   can actually feed it.

## Immediate Gates

- Keep live training runs read-only.
- Keep stock training/tournament recommendations on `cpu_oracle` until the GPU
  backend clears the contract gates.
- Use the profile-only facade to measure the batched boundary before touching
  stock LightZero defaults.
- Update docs immediately when a claim changes; old speed rows are useful only
  if they name exactly what was timed.

## Verification In This Round

- `uv run ruff check` on the touched source-state env, batched facade, visual
  observation, and tournament files passed.
- `uv run pytest tests/test_source_state_batched_observation_profile_cpu.py
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py
  tests/test_multiplayer_source_state_trainer_surface.py::test_body_circles_fast_is_rejected_by_current_trainer_surface
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif -q`
  passed: `55 passed`.
- `uv run pytest tests/test_vector_visual_observation.py::test_cpu_oracle_simple_bonus_symbols_rgb_avoids_full_canvas_scratch
  tests/test_curvytron_two_seat_render_mode.py -q` passed:
  `32 passed, 1 skipped`.
- `uv run pytest tests/test_source_state_gpu_render_benchmark_cpu.py
  tests/test_source_state_visual_survival_learner_seat_regression.py -q`
  passed: `11 passed, 1 skipped`.
- `uv run pytest tests/test_lightzero_checkpoint_opponent_provider.py -q`
  passed: `3 passed`.
- `uv run pytest tests/test_source_state_visual_survival_learner_seat_regression.py -q`
  passed: `5 passed`.
- `uv run pytest tests/test_curvytron_checkpoint_tournament.py::test_tournament_render_contract_pins_policy_surface_and_full_gif
  tests/test_curvytron_checkpoint_tournament.py::test_tournament_rejects_legacy_policy_surface
  tests/test_curvytron_checkpoint_tournament.py::test_tournament_rejects_lab_policy_observation_backend
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_spec_reads_policy_render_mode_from_observation_contract
  tests/test_curvytron_checkpoint_tournament.py::test_rating_context_hash_changes_for_evaluator_not_roster -q`
  passed: `5 passed`.
- `uv run pytest tests/test_source_state_gpu_render_benchmark_cpu.py -q`
  passed after adding fused two-view parity coverage: `11 passed, 4 skipped`.
- H100 Modal isolated renderer rows:
  - adversarial B4/S10 priority buffer: exact parity, `0.784ms` device,
    `5.231ms` end-to-end.
  - adversarial B4/S10 owner-ordered compact: exact parity, `0.705ms` device,
    `3.253ms` end-to-end.
  - real-env B64/S1024 priority buffer: exact parity, `208.9ms` device,
    `214.0ms` end-to-end.
  - real-env B64/S1024 owner-ordered compact: exact parity, `135.5ms` device,
    `140.4ms` end-to-end.
  - real-env B256/S1024 priority buffer: exact parity, `729.6ms` device,
    `734.4ms` end-to-end.
  - real-env B256/S1024 owner-ordered compact: exact parity, `391.4ms` device,
    `395.5ms` end-to-end.
  - real-env B64/S256 controlled player 1 priority buffer: exact parity,
    `54.2ms` device, `57.2ms` end-to-end.
  - real-env B64/S256 controlled player 1 owner-ordered compact: exact parity,
    `36.0ms` device, `40.3ms` end-to-end.
  - real-env B64/S2048 priority buffer: exact parity, `412.6ms` device,
    `416.6ms` end-to-end.
  - real-env B64/S2048 owner-ordered compact: exact parity, `265.8ms` device,
    `269.7ms` end-to-end.
- H100 Modal isolated two-view renderer rows, real-env S1024, readback included,
  explicit `output_order=view_major`:
  - B64 priority buffer: exact parity, `489.99ms` device,
    `494.18ms` H2D + render + readback.
  - B64 owner-ordered compact: exact parity, `248.13ms` device,
    `251.51ms` H2D + render + readback. Setup reported separately:
    `5.7ms` owner-ordered packing, not included in render timing.
  - B256 priority buffer: exact parity, `1840.06ms` device,
    `1844.97ms` H2D + render + readback.
  - B256 owner-ordered compact: exact parity, `1136.28ms` device,
    `1142.06ms` H2D + render + readback. Setup reported separately:
    `19.6ms` owner-ordered packing, not included in render timing.
- Scalar one-env/two-view component profiles, H100, S1024:
  - priority buffer fused two-view: exact CPU parity, `24.69ms` fused render,
    `28.35ms` full compact + H2D + render + readback.
  - owner-ordered compact fused two-view: exact CPU parity, `23.68ms` fused
    render, `29.21ms` full compact + H2D + render + readback.
  - Both compositions show the same important fact: fusing both controlled
    views is about `1.75x-1.81x` faster than rendering each view separately in
    the scalar B1 shape.

## Important Caveat

The GPU rows above are isolated one-view render rows. The real trainer/frozen
opponent path often needs both controlled-player views, then stack update,
terminal `final_observation`, policy/search, replay, and learner work. The next
claim should be about the batched observation boundary, not just the one-view
kernel. The scalar two-view profile says the B1 shape is still too launch/copy
heavy; the ordered-compact win becomes meaningful when rows/views are batched.
