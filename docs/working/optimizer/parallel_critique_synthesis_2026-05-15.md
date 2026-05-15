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

## Highest-Value Critique Threads

1. **Info/IPC falsifier.**
   Add a profile-only `full|minimal` info-payload mode or equivalent counters.
   If wall/collector drops by at least `15%`, render is not the next best
   default target.

2. **Owner-ordered GPU renderer.**
   Prototype a benchmark-only `block_704_gray64_ordered_overwrite` path. Pass
   requires exact adversarial parity and at least `1.25x` speedup over the
   current exact priority-buffer renderer at H100 B64/S1024.

3. **Tile-sparse renderer.**
   Bigger and riskier. Use Triton/CuPy/Pallas/CUDA only after the ordered
   overwrite path is tested. Target is exact 11x11 sampling only for blocks that
   intersect candidate trail segments.

4. **Trainer-visible GPU contract parity.**
   Frame parity is not enough. Reset stack, terminal `final_observation`,
   controlled-player perspective, action masks, `to_play=-1`, reward/done/info,
   and tournament checkpoint loading must match before promotion.

5. **Metadata identity drift.**
   The current audit still flags backend identity as under-specified in
   checkpoint/tournament/rating metadata. `cpu_oracle` and scalar `jax_gpu`
   should not collapse into the same rating identity if scalar GPU is ever used
   for a lab checkpoint.

6. **Collector/search scaling.**
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
