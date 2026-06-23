# Full Loop Profile V8

Date: 2026-05-14

Purpose: measure whether the V8 fast observation path speeds up the real stock
LightZero loop, not just the local env renderer.

2026-05-15 supersession: this is historical/control speed evidence. Do not
copy the `body_circles_fast + simple_symbols` row into fresh production
launches. Current production policy observations are CPU `cpu_oracle`
`browser_lines + simple_symbols`; GPU rendering remains lab/profiling-only
until trainer-visible contract parity passes.

## Scope

Both rows used the trusted stock path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
--mode profile
--env-variant source_state_fixed_opponent
```

Shared settings:

- compute: `gpu-l4-t4-cpu40`
- collector envs: `8`
- search: `num_simulations=8`
- learner batch: `256`
- profile stop: `12` learner train calls
- no eval/GIF/checkpoint commit
- no-death profile: `disable_death_for_profile=true`,
  `opponent_runtime_mode=blank_canvas_noop`, `opponent_death_mode=immortal`
- source horizon: `source_max_steps=512`

The only intended visual difference:

| row | trail mode | bonus mode |
| --- | --- | --- |
| browser reference | `browser_lines` | `browser_sprites` |
| fast observation | `body_circles_fast` | `simple_symbols` |

## Result

| row | wall s | env steps | steps/s | collector s | MCTS s | policy forward s | learner s | replay sample s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| browser reference | 36.547 | 4096 | 112.1 | 26.145 | 11.860 | 16.332 | 2.664 | 2.260 |
| fast observation | 31.362 | 4096 | 130.6 | 21.416 | 11.447 | 15.696 | 2.546 | 2.278 |

Plain read: the fast observation row was about `1.17x` faster end to end in
this profile. This is useful, but it is not a giant whole-loop win.

## Env Timing

The model-facing observation got much faster:

| row | mean observation ms/step | mean vector ms/step | mean env step before info ms |
| --- | ---: | ---: | ---: |
| browser reference | 6.02 | 1.47 | 7.54 |
| fast observation | 0.57 | 1.46 | 2.09 |

Do not subtract the env timing sums directly from wall time. These env timing
rows come from subprocess collectors, so sampled per-env sums can overlap in
wall-clock time. The safe wall-clock comparison is the top table.

## Amdahl Read

The V8 fast path removes most of the model-observation render cost, but the
full loop still spends a lot of wall time in collection, policy/model forward,
MCTS, replay, and learner work. In this exact C8/sim8/no-death profile:

- env observation mean improved about `10.5x`;
- full-loop wall improved about `1.17x`;
- MCTS/model timings changed only slightly;
- GPU utilization sampling was noisy and low, so this is not proof that larger
  GPUs help.

The next useful profile is a wider collector row, such as C32, because C8 may
not show the regime Coach actually wants for large self-play collection.

## Caveats

- This is a speed/profile row, not a learning-quality claim.
- It used no-death and blank-canvas opponent to force long trajectories.
- It used one source tick per policy action. That is the current trusted
  cadence, but it makes policy/search work heavier per physical game time than
  older hidden-repeat comparisons.
- Rich eval/GIF visual artifacts should keep `browser_lines + browser_sprites`.
