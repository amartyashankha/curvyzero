# LightZero Official Atari Next Run Plan - 2026-05-09

Scope: official LightZero Atari Pong only. This is the stock
`PongNoFrameskip-v4` visual MuZero lane, not custom dummy Pong and not
CurvyTron. No training was run for this plan.

## Decision

Run one L4/T4 calibration rung that is closer to official LightZero without
jumping to full official cost:

| Knob | Current GPU2048 control | Next rung | Official reference |
| --- | ---: | ---: | ---: |
| max env steps | `2048` | `4096` | `500000` |
| collector envs | `1` | `2` | `8` |
| evaluator envs | `1` | `1` | `3` |
| simulations | `2` | `10` | `50` |
| batch size | `8` | `32` | `256` |
| update per collect | `1` | `2` | auto replay-ratio / much larger |
| game segment length | `16` | `64` | `400` |
| episode cap | `256` | `512` | uncapped by our wrapper |

This is deliberately not a `num_simulations=50` run. Going from `2` to `50`
is a `25x` search-cost jump before we know whether the wrapper can turn extra
search into a useful eval curve. The sane ladder is `10 -> 25 -> 50`, where
each step requires a visible curve signal and acceptable runtime.

## Required Tiny Wrapper Change

The current train wrapper validation cap rejects this command. Before launch,
make a cap-only wrapper edit, with no training-logic change:

- raise train GPU timeout from `12m` to `30m`;
- allow `max_env_step=4096`;
- allow `max_train_iter=32`;
- allow `collector_env_num=2`;
- allow `num_simulations=10`;
- allow `batch_size=32`;
- allow `update_per_collect=2`;
- allow `max_episode_steps=512`;
- allow `game_segment_length=64`.

Keep `evaluator_env_num=1` for this rung to avoid multiplying eval work. If
this rung works, evaluator env count can move later.

## Train Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_tiny_train_smoke --compute gpu-l4-t4 --mode train --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2
```

Expected checkpoint shape: because GPU2048 stopped at `iteration_8`, this
4096-step rung should roughly reach `iteration_16` if LightZero's collection
cadence stays similar. `max_train_iter=32` is just a ceiling.

## Eval Curve

Primary eval is no-fallback, same policy path, `num_simulations=10`, 256 real
ALE steps for comparability with the GPU1024/GPU2048 readouts. Score these
checkpoints if present:

| Eval point | Purpose |
| --- | --- |
| `iteration_0` | run-local baseline |
| `iteration_4` | early movement |
| `iteration_8` | direct comparison with GPU2048's final checkpoint index |
| `iteration_16` or max checkpoint | final rung read |

Command template after train returns the checkpoint refs:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_eval_smoke --checkpoint-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/checkpoints/lightzero/iteration_16.pth.tar --run-id lz-visual-pong-4096-sim10-s0 --attempt-id train-4096-sim10-b32-env2 --output-ref training/lightzero-official-visual-pong/lz-visual-pong-4096-sim10-s0/attempts/train-4096-sim10-b32-env2/eval/iteration_16_sim10_eval256/lightzero_visual_pong_eval_iteration16_sim10_eval256.json --max-env-step 4096 --max-train-iter 32 --collector-env-num 2 --evaluator-env-num 1 --num-simulations 10 --batch-size 32 --update-per-collect 2 --max-episode-steps 512 --game-segment-length 64 --max-eval-steps 256 --step-detail-limit 8 --no-allow-model-fallback
```

If the `iteration_16` eval is promising, run one final 512-step eval for the
same checkpoint with `--max-eval-steps 512`. Do not run 512-step evals for the
whole curve unless the 256-step curve moves.

## Expected Time And Cost

GPU2048 with `2` simulations, one env, `batch_size=8`, and `update_per_collect=1`
took about `84s` remote elapsed on an L4.

This rung doubles env steps and increases search from `2` to `10`, plus modest
batch/update/env overhead. Expect roughly `12-25 minutes` wall time if the
wrapper and subprocess envs behave, with a hard planning budget of `30 minutes`.

Modal pricing checked on 2026-05-09 lists L4 at `$0.000222/sec` and T4 at
`$0.000164/sec`, plus CPU/memory metering:

- L4 GPU-only envelope for `12-25m`: about `$0.16-$0.33`;
- T4 GPU-only envelope for `12-25m`: about `$0.12-$0.25`;
- all-in practical planning envelope: `$0.25-$0.60`;
- stop the rung family if a single sim10 train approaches `$1` or exceeds
  `30m` without producing the expected checkpoint curve.

Pricing source: https://modal.com/pricing

## Stop Criteria

Stop immediately and do not climb the ladder if any of these happen:

- CUDA is unavailable or the runtime GPU is not L4/T4;
- wrapper validation reports config problems;
- checkpoint mirroring misses `iteration_8`;
- train exceeds `30m`;
- any curve eval uses model fallback;
- all curve evals are flat at or below the GPU2048 result: return `-6`, no
  positive rewards, and action support collapsed to three or fewer actions.

## Signal That Counts

Weak signal:

- final 256-step no-fallback eval improves over run-local `iteration_0` by at
  least two return points; or
- `iteration_8` or final gets at least one `+1` Pong reward and uses at least
  four actions.

Useful signal:

- curve is directionally better from `iteration_0 -> iteration_8 -> final`;
- final 256-step return is at least `-3`, matching or beating the prior
  GPU1024 small signal;
- action support stays broad rather than collapsing to `{0,1,2}`;
- strict no-fallback eval remains clean.

Only after useful signal should the next rung be `4096` or `8192` steps with
`num_simulations=25`. Save `num_simulations=50` for the point where sim25 gives
a real curve and the runtime envelope is understood.
