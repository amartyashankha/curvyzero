# Current Optimizer Plate Map

Date: 2026-05-13
Updated: 2026-05-15

Purpose: one plain page to keep optimizer docs pointed at the current target.

## Main Lane

The active training lane is stock LightZero:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode train
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--opponent-use-cuda=false
```

## Policy Observation Target

Current target for policy observations:

```text
source-state geometry
-> browser_lines trails
-> simple_symbols bonuses
-> live heads
-> BT.601 luma
-> exact 11x11 area average
-> uint8[1,64,64]
-> FIFO stack [4,64,64]
```

Current reliable implementation: CPU `cpu_oracle`.

Measured scalar `jax_gpu` is a canary only: it reaches stock `train_muzero`,
but it is slower than CPU and fails in subprocess workers. Preferred future
implementation is batched GPU rendering for the same surface, at an
env-manager/collector boundary or render service.

`browser_sprites` are for artifacts, reference screenshots, GIF/eval views, and
browser-fidelity work. `body_circles_fast` and `fast_gray64_direct` are
historical/prototype renderer names, not current training or tournament policy
surfaces.

## Active Plates

1. **GPU policy renderer**
   Build and prove a batched GPU `browser_lines + simple_symbols` observation
   backend for stock LightZero policy observations. Do not scale the scalar
   `jax_gpu` hook as if it were production.

2. **CPU oracle**
   Keep CPU `browser_lines + simple_symbols` as the reliable training backend
   until the batched GPU path exists. Record resolved trail mode, bonus mode,
   and `policy_observation_backend` in metadata.

3. **Full-loop Amdahl**
   Render matters in long-survival/no-death profiles, but stock LightZero also
   spends time in collection, search, frozen-opponent inference, replay,
   learner, checkpoints, and artifact I/O. Fresh stock profiles showed C1
   `10.8` env steps/sec, C32 `153.6`, C64 `408.4`, C96 `487.3`, and C64 H100
   `321.0`. Use full-loop profiles before calling render the bottleneck.

4. **Artifact/reference lane**
   Keep `browser_lines + browser_sprites` for visual reference and
   browser-fidelity artifacts. Do not let that lane redefine policy
   observations.

5. **Historical controls**
   The 212-run integration review found same-checkpoint `browser_lines` versus
   `body_circles_fast` results basically tied: 58 matched pairs, median
   browser-minus-fast delta `0.0`, and nearly even signs. That is useful
   history, not the current target.

## Do Not Drift Back To

- Treating scalar `jax_gpu` as the recommendation destination.
- `browser_sprites` as the policy-observation target.
- `body_circles_fast` as the recommendation target.
- `fast_gray64_direct` as a stock-path recommendation; that name belongs to the
  old custom two-seat adapter.
- Old `--mode two-seat-selfplay` learning conclusions as evidence about current
  stock LightZero.
- Browser-pixel claims without a browser golden-frame harness.

## Immediate Checks

- Keep optimizer advice scoped to stock `--mode train`.
- Keep train/eval metadata explicit for trail render mode and bonus render mode.
- Compare GPU output against CPU `browser_lines + simple_symbols` on real rows.
- Run a fresh stock full-loop profile before making throughput claims.
