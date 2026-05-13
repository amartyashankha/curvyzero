# Second Wave Profile Tensor

Date: 2026-05-12

Purpose: follow up on the first function-readback wave without becoming ad hoc.
This tensor keeps the trusted stock LightZero profile path and changes only one
or two meaningful axes per row.

Manifest:

```text
artifacts/local/curvytron_optimizer_profile_manifests/opt-stock-frozen-profile-second-wave-20260512b.json
```

Note: `20260512a` proved the tensor shape but most launches were rejected
because generated run IDs exceeded the repo's 96-character run-id limit. The
`20260512b` manifest shortens row tags and is the one to use.

Common defaults:

```text
mode=profile
env_variant=source_state_fixed_opponent
batch_size=16
source_max_steps=256
stop_after_learner_train_calls=10
eval/GIF/checkpoint effectively off
readback=parent modal run --detach + child --profile-spawn + FunctionCall.get()
```

Rows:

| row | question | compute | C | sims | death | render | opponent | reward |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| S01 | does C128 still help? | `gpu-l4-t4-cpu40` | 128 | 8 | normal | browser | frozen checkpoint | sparse |
| S02 | does C160 still help or bend? | `gpu-l4-t4-cpu40` | 160 | 8 | normal | browser | frozen checkpoint | sparse |
| S03 | does sim16 still stay cheap at C96? | `gpu-l4-t4-cpu40` | 96 | 16 | normal | browser | frozen checkpoint | sparse |
| S04 | long browser cost at C32 | `gpu-l4-t4-cpu40` | 32 | 8 | nodeath | browser | frozen checkpoint | sparse |
| S05 | long fast-render cost at C32 | `gpu-l4-t4-cpu40` | 32 | 8 | nodeath | fast | frozen checkpoint | sparse |
| S06 | fast render on short normal runs | `gpu-l4-t4-cpu40` | 32 | 8 | normal | fast | frozen checkpoint | sparse |
| S07 | frozen-opponent cost at C32 | `gpu-l4-t4-cpu40` | 32 | 8 | normal | browser | fixed straight | sparse |
| S08 | frozen-opponent cost at C96 | `gpu-l4-t4-cpu40` | 96 | 8 | normal | browser | fixed straight | sparse |
| S09 | is L4 helping at sim8? | `cpu64` | 32 | 8 | normal | browser | frozen checkpoint | sparse |
| S10 | dense reward overhead at C96 | `gpu-l4-t4-cpu40` | 96 | 8 | normal | browser | frozen checkpoint | dense survival |

Expected use:

- Run S01-S10 as one structured wave if capacity is fine.
- Compare S01/S02 to first-wave C64/C96.
- Compare S04/S05 to quantify long-trajectory render tradeoff at a wider
  collector count.
- Compare S07/S08 to frozen-checkpoint rows S06/S01 or first-wave rows 06/08 to
  isolate opponent-policy overhead.
- Do not treat fixed-straight rows as learning recommendations. They are a
  timing lens only.

## Results

Result files:

```text
artifacts/local/curvytron_optimizer_profile_results/opt-stock-frozen-profile-second-wave-20260512b/
```

| row | C | sims | death | render | reward | steps | wall | steps/s | collect | MCTS | learner | obs | opp | GPU max |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S01 | 128 | 8 | normal | browser_lines | sparse_outcome | 1176 | 28.82 | 40.80 | 11.28 | 0.76 | 1.83 | 15.06 | 608.75 | 0.0 |
| S02 | 160 | 8 | normal | browser_lines | sparse_outcome | 1475 | 33.70 | 43.77 | 12.69 | 0.61 | 2.02 | 13.62 | 1065.04 | 0.0 |
| S03 | 96 | 16 | normal | browser_lines | sparse_outcome | 911 | 22.82 | 39.92 | 8.65 | 0.74 | 1.75 | 8.13 | 355.58 | 0.0 |
| S04 | 32 | 8 | nodeath | browser_lines | sparse_outcome | 8192 | 90.09 | 90.93 | 81.16 | 6.83 | 1.75 | 14.80 | 42.27 | 0.0 |
| S05 | 32 | 8 | nodeath | body_circles_fast | sparse_outcome | 8192 | 51.34 | 159.57 | 42.34 | 7.01 | 1.82 | 4.35 | 35.23 | 0.0 |
| S06 | 32 | 8 | normal | body_circles_fast | sparse_outcome | 316 | 13.35 | 23.67 | 4.30 | 0.33 | 2.08 | 1.35 | 34.82 | 0.0 |
| S07 | 32 | 8 | normal | browser_lines | sparse_outcome | 428 | 14.77 | 28.97 | 5.27 | 0.73 | 2.13 | 3.10 | 0.00 | 0.0 |
| S08 | 96 | 8 | normal | browser_lines | sparse_outcome | 1248 | 21.34 | 58.50 | 6.96 | 0.87 | 2.01 | 18.16 | 0.00 | 0.0 |
| S09 | 32 | 8 | normal | browser_lines | sparse_outcome | 316 | 18.59 | 17.00 | 5.36 | 0.67 | 3.99 | 2.12 | 42.28 |  |
| S10 | 96 | 8 | normal | browser_lines | dense_survival_plus_outcome | 925 | 24.77 | 37.34 | 8.95 | 0.50 | 2.10 | 7.93 | 355.06 | 0.0 |

Read:

- C96 is close to the current practical width knee. First-wave C96 was about
  41.0 steps/s; C128 was 40.8; C160 was 43.8. C160 is not a dramatic win for
  the added process count.
- Sim16 at C96 is almost the same throughput as sim8 at C96. Search is still
  not the first bottleneck in these stock fixed-opponent profiles.
- Long no-death trajectories make fast render matter. At C32, fast render is
  159.6 steps/s versus 90.9 for browser render. That is about 1.75x total.
- Short normal trajectories do not care as much about render. C32 fast render is
  23.7 steps/s versus 20.9 for C32 browser in the first wave.
- Fixed-straight opponent is much faster than frozen-checkpoint opponent:
  C32 improves from about 20.9 to 29.0 steps/s, and C96 improves from about
  41.0 to 58.5 steps/s. This is a control-lane cost, not a final self-play
  conclusion.
- CPU64 is viable but slower than L4+CPU40 at the same C32/sim8 shape.
- Dense reward bookkeeping is a small cost at C96, not a dominant bottleneck.
