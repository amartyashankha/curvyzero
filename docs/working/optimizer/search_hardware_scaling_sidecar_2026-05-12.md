# Search And Hardware Scaling Sidecar

Date: 2026-05-12

Scope: optimizer sidecar for the canonical two-seat path:
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
This is a speed/profiling note, not a learning-quality recommendation.

## Current Read

Render is still the bottleneck after the perspective reuse and visual-trail
cache work. The current wait-mode evidence is:

```text
B16/L4/sim16:
  visual_stack_update_sec sum 136.2s
  policy_search_sec       sum  14.0s
  elapsed                 198.6s
```

That puts visual stack update at roughly `10x` policy/search in the named
collect buckets. The pending `B64/H100/B128` wait-mode results should be read
as scaling evidence only after their render/search split is available.

Practical consequence: do not spend the next optimizer cycle on MCTS internals
unless render drops enough that policy/search becomes a first-order bucket.
The trigger for deeper MCTS work is a steady-state two-seat profile where
`policy_search_sec` is consistently at least `25-30%` of named collect time, or
where raising `num_simulations`/batch width makes wall time grow mainly through
`policy_search_sec` rather than `visual_stack_update_sec`.

## Hardware Read

H100 can be useful for faster sweeps, but the current two-seat launcher only
routes `cpu`, `gpu-l4-t4`, and `gpu-h100-cpu40`. The stock path exposes
`gpu-h100x2-cpu40`, but `--mode two-seat-selfplay` rejects it today. So
multi-GPU is not a meaningful next two-seat knob without code changes.

Even single H100 should be treated as an accelerator test, not a default. If
render remains CPU-side and dominant, H100 mostly helps the smaller model/search
slice and may leave most of the wall clock untouched. H100 starts making sense
when the L4 profile shows either high `policy_search_sec`, high model inference
time, or underfilled-but-growing batched search where bigger root batches can
actually feed the GPU.

## Matrix Knobs

Keep the next matrix small and denominator-clean:

- `compute`: `gpu-l4-t4` first, `gpu-h100-cpu40` as a paired accelerator check.
- `--batch-size`: `16`, `32`, `64`, then `128` only if render/search fractions
  still improve. In two-seat mode, this is env row width; fresh policy rows are
  up to `B * 2`, reduced by action-repeat reuse and dead rows. It is not an
  independent learner minibatch control in this sidecar.
- `num_simulations`: `16` for control, `32` as the first stronger-search check,
  `50` only after render is no longer drowning the readout.
- `two-seat-collect-steps-per-iteration`: keep fixed while comparing hardware,
  usually `16` or `32` for profiling and `64` for training-shaped smoke.
- `two-seat-death-mode`: `profile_no_death` only for long-trail stress. Use
  normal death mode for training-shaped speed estimates.
- `two-seat-trail-render-mode`: keep `browser_lines` as product/default.
  `body_circles_fast` is a render comparison knob only because it changes
  training-visible pixels.
- Learner/replay knobs such as `two-seat-learner-sample-size`,
  `two-seat-updates-per-iteration`, and `two-seat-max-replay-rows` are not MCTS
  root batching knobs. Vary them only in a separate learner/replay lane.

Do not mix reward, exploration, action-repeat, observation-noise, or eval
changes into this matrix. They change data distribution and learning semantics,
which makes hardware/search attribution muddy.

## Command Templates

Attached profiling run:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode two-seat-selfplay \
  --compute gpu-l4-t4 \
  --run-id opt-search-hw-sidecar-20260512 \
  --attempt-id b16-l4-sim16-nodeath-wait \
  --batch-size 16 \
  --num-simulations 16 \
  --max-train-iter 10 \
  --two-seat-collect-steps-per-iteration 32 \
  --two-seat-updates-per-iteration 0 \
  --two-seat-death-mode profile_no_death \
  --two-seat-trail-render-mode browser_lines \
  --save-ckpt-after-iter 1000 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --wait-for-train
```

Hardware/search sweep cells should change only these fields:

```text
--compute gpu-l4-t4 | gpu-h100-cpu40
--batch-size 16 | 32 | 64 | 128
--num-simulations 16 | 32 | 50
--attempt-id b{B}-{compute}-sim{S}-nodeath-wait
```

## What To Record

For each cell, report:

- elapsed wall time;
- `visual_stack_update_sec`, `policy_search_sec`, `env_step_sec`,
  `replay_row_build_sec`, and learner time if updates are enabled;
- `policy_search_call_count`, `policy_search_row_count`, and policy rows per
  call;
- model device and any GPU utilization samples available;
- completed steps/games and whether no-death profiling was enabled;
- render mode, surface/schema, batch width, simulations, collect steps, and
  whether background eval/GIF/checkpoints were suppressed.

## Boundary

This sidecar can say which hardware/search shape is faster per measured
denominator. It must not say the policy learned better. Learning claims need
Coach-owned eval, checkpoint comparisons, and normal-death runs. In particular,
`profile_no_death`, `body_circles_fast`, sparse/no checkpoints, and disabled
background eval/GIF are profiling controls, not product training results.
