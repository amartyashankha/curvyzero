# Pong Stock64 Signal Comparison - 2026-05-11

Question: why did `s122` look like the only improving run at first, while
`s114`, `s120`, `s121`, and `s142` looked flat?

Short answer: the early read was too soon. After a longer wait, multiple
stock64 rows show survival learning. `s122` is still the strongest run, but it
is no longer the only positive row. The clean lesson is that survival signal
can appear late, and score can lag behind survival.

## Runs Checked

All five runs use the installed LightZero `0.2.0` stock64 Atari Pong surface:
`zoo.atari.config.atari_muzero_config`, `PongNoFrameskip-v4`, observation
shape `(4, 64, 64)`, 8 collectors, 3 evaluators, `num_simulations=50`, batch
size 256, sparse stock reward, no reward shaping, and checkpoint cadence
override `1000`.

| run | seed | compute | cap | latest checkpoint/progress seen | late eval seeds | survival read |
| --- | ---: | --- | ---: | --- | ---: | --- |
| `s114` | 114 | L4/T4 CPU40 | 50k | completed; latest `iteration_13159`; `attempt.json` present | 8 | `761.25 -> 792 -> 1612.12` at `0/10000/13000` |
| `s120` | 120 | L4/T4 CPU40 | 50k | completed; latest `iteration_14012`; `attempt.json` present | 8 | `761.25 -> 776.5 -> 961.375` at `0/11000/14000` |
| `s121` | 121 | L4/T4 CPU40 | 65k | running; latest checked eval `iteration_17000` | 8 | `853.25 -> 772.5 -> 1579.38` at `0/11000/17000` |
| `s122` | 122 | H100 CPU40 | 100k | running; latest checked eval `iteration_26000` | 16 | `761.5 -> 1378.06 -> 1591.69 -> 1977.62` at `0/12000/20000/26000` |
| `s142` | 142 | H100 CPU40 | 100k | running; latest checked eval `iteration_15000` | 16 | `761.5 -> 761.5 -> 839.938 -> 938.375` at `0/7000/12000/15000` |
| `s113` exact | 113 | L4/T4 CPU40 | 200k | running; latest checked eval `iteration_20000` | 8 | `761.25 -> 833.75 -> 917.125` at `0/10000/20000` |
| `s123` exact | 123 | H100 CPU40 | 200k | running; latest checked eval `iteration_20000` | 8 | `761.25 -> 851.25 -> 1145.12` at `0/10000/20000` |

Refs checked:

- Monitor doc: `docs/working/lightzero_pong_replication_monitor_2026-05-11.md`.
- Progress refs under each run's `train/progress/latest.json`.
- Checkpoint refs under each run's `train/lightzero_exp/ckpt`.
- Eval refs under each run's `eval/...` roots, including the `...20260511c`
  stock-only eval panels.

## What Is Actually Different

Known differences:

- Seed differs across all rows.
- Hardware differs: `s114/s120/s121` are L4/T4 CPU40; `s122/s142` are H100
  CPU40.
- Run cap differs: `s114/s120` use 50k, `s121` uses 65k, `s122/s142` use
  100k, and `s113/s123` exact controls use 200k.
- Elapsed training at the latest progress poll differs. This matters because
  several rows looked flat before they reached the later checkpoints that
  moved.
- Late eval seed count differs. `s122` and `s142` have 16-seed late reads;
  the other rows in this note use 8-seed late reads. This is enough for a
  direction check, but higher-confidence claims should use larger seed panels.

Not meaningful config differences found:

- The saved `total_config.py` for `s122` and `s142` matches on the important
  stock64 training settings, apart from `exp_name` and `seed`.
- The `attempt.json` files for completed `s114` and `s120` show the same
  module, LightZero version, no reward shaping, same checkpoint cadence, same
  compute class, and same 50k cap, with seed as the main difference.
- The monitor's launch commands show `s121` differs from the L4/T4 50k rows
  mainly by seed and 65k cap, and `s122/s142` differ mainly by seed, H100, and
  100k cap.

## What We Can Infer

We can say stock64 visual Pong replication has a real learning signal across
multiple rows. `s122` is strongest, but `s114`, `s120`, `s121`, `s142`, `s113`,
and `s123` all improved survival on later checkpoints.

We can say this is survival learning first, not solved Pong. Even when score is
still poor, the policy survives longer. That is the correct early signal.

We can say H100 helped later checkpoints arrive sooner in wall-clock time. It
does not prove H100 caused the learning.

## What We Cannot Infer

We cannot say H100 caused the learning. L4/T4 rows also improved after longer
training.

We cannot say the 100k cap caused the learning. The shorter L4/T4 rows also
eventually moved.

We cannot keep calling `s120` or `s121` failed. Later evals reversed that
claim.

We cannot use score alone for this decision. Survival steps moved before score
became good; score would have hidden the early `s122` signal.

## Current Best Explanation

The safest explanation is: same stock64 setup, sparse reward, slow early
learning, and real seed/checkpoint variance. `s122` learned fastest and
strongest, but the broader pattern says the setup works once runs mature.

The next clean tests are higher-cap evals for strong checkpoints and continued
later checkpoints for weaker rows. `s122` is already near the `2048` eval cap,
so a longer cap is needed to see whether survival keeps improving.
