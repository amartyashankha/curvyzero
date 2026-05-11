# 2026-05-09 Modal LightZero Pong 8192 Sim25 Parallel 512

## Question

Does raising the selected official Atari Pong checkpoint eval cap from `256`
to `512` steps reveal hidden progress in the `8192/sim25` LightZero run?

## Setup

Scope: installed `LightZero==0.2.0` official Atari Pong only.

- Modal app: `ap-TmYBGeAyofKbT0wKfTMH4J`
- Run: `lz-visual-pong-8192-sim25-s0`
- Attempt: `train-8192-sim25-b64-env4-auto`
- Eval wrapper: `curvyzero.infra.modal.lightzero_pong_eval_smoke`
- Selected iterations: `0,100,500,900,932`
- `max_eval_steps=512`
- `num_simulations=25`
- Model fallback disabled
- `run_stock_evaluator` enabled

Do not include `ckpt_best` as quality evidence for this run. Earlier state diff
showed it is reset-looking, not a learned best checkpoint.

## Command

Parallel eval through `curvyzero.infra.modal.lightzero_pong_eval_smoke` with
the following surface:

- parallel checkpoint eval
- selected iterations `0,100,500,900,932`
- `max_eval_steps=512`
- `num_simulations=25`
- no model fallback
- `run_stock_evaluator`

## Results

Manifest:

```text
training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve512_parallel_sim25/manifest_custom_steps512_seed0_20260509T224529Z.json
sha256 f4221798b8124ae19626a37e669ce2b1813751e1672dc03014960866c4ed2b22
```

| Checkpoint | Action histogram | Return | Steps | Stock/manual match | Fallback | Positive rewards | Negative rewards |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `iteration_0` | `{3: 512}` | `-13` | `512` | true | false | `0` | `13` |
| `iteration_100` | `{0: 512}` | `-13` | `512` | true | false | `0` | `13` |
| `iteration_500` | `{5: 512}` | `-13` | `512` | true | false | `0` | `13` |
| `iteration_900` | `{0: 512}` | `-13` | `512` | true | false | `0` | `13` |
| `iteration_932` | `{1: 512}` | `-13` | `512` | true | false | `0` | `13` |

Every row had no positive Pong rewards and `13` negative Pong rewards over
`512` steps.

## Interpretation

Increasing the eval cap from `256` to `512` did not reveal hidden progress.
Every periodic checkpoint is still hard action collapse: one action for the
whole eval window, no fallback, stock/manual match true, no positive rewards,
and the same `-13` return.

Plain read: this is stronger negative evidence for the selected periodic curve.
It does not prove LightZero cannot learn Pong. It says this off-recipe
`8192/sim25` run did not produce credible periodic-checkpoint learning under a
longer capped eval.

## Artifacts

- Manifest:
  `training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_curve512_parallel_sim25/manifest_custom_steps512_seed0_20260509T224529Z.json`

## Follow-ups

- Keep `ckpt_best` out of quality summaries until the best-save path is
  explained.
- Do not rerun the same selected periodic curve at small cap changes. The next
  useful official Atari step is exact stock reproduction or a clearly bounded
  larger-recipe run with explicit update and checkpoint accounting.
