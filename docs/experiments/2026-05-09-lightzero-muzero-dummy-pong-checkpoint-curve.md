# 2026-05-09 LightZero MuZero Dummy Pong Checkpoint Curve

## Question

Do saved checkpoints improve over training in the same trusted post-deep-seed-fix
LightZero MuZero dummy Pong run?

## Source Run

- Run: `lz-dpong-20260509T154530Z-b049f29edb64`
- Attempt: `attempt-20260509T154530Z-ca60e4962603`
- Train Modal URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-zYhiKPXQKKijsOXbN29aiK`
- Prior single-checkpoint scorecard Modal URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-GdYm0zsa8ifnsch1rUU9Vh`
- Volume: `curvyzero-runs`

Available LightZero checkpoints in the run:

| Label | Volume ref | SHA-256 |
| --- | --- | --- |
| `iter0` | `training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar` | `4b20241909346a52334d25d2fa4adc91349a5cc7314bf8c8dd7ce9bd8fae493e` |
| `iter16` | `training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar` | `e2dd80f8a08b15d750f5c8c643051b8e11e63eb5ce44a5ab71fee9fecaf88ee8` |
| `best` | `training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/ckpt_best.pth.tar` | `066d8fc3a1e6385c566935bbd8934a8cb8fa40662c50957217140772ee81e0aa` |

## Eval

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints "lightzero:iter0=ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar,lightzero:iter16=ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar,lightzero:best=ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/ckpt_best.pth.tar" --episodes 16 --seed 1701 --split-id dummy_pong_post_deep_seed_fix_checkpoint_curve_heldout_v0 --split-role heldout --eval-id mcts-scoreboard-post-deep-seed-fix-checkpoint-curve-1024x16-all-ckpts-e16 --max-env-step 1024 --num-simulations 8 --run-id lz-dpong-20260509T154530Z-b049f29edb64 --attempt-id attempt-20260509T154530Z-ca60e4962603
```

- Eval Modal URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-ZWrn3qUKYYReTw8WCCqRsT`
- Summary:
  `training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/eval/mcts-scoreboard-post-deep-seed-fix-checkpoint-curve-1024x16-all-ckpts-e16/summary.json`
- Episodes:
  `training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/attempts/attempt-20260509T154530Z-ca60e4962603/eval/mcts-scoreboard-post-deep-seed-fix-checkpoint-curve-1024x16-all-ckpts-e16/episodes.jsonl`
- Episodes per seating: `16`
- Total episodes: `528`
- Held-out seed: `1701`
- Split: `dummy_pong_post_deep_seed_fix_checkpoint_curve_heldout_v0`
- Seating: paired seats
- Max env step: `1024`
- MCTS simulations: `8`
- Adapter: `MuZeroPolicy.eval_mode.forward`, `to_play=[-1]`,
  `ready_env_id=[0]`, all-ones action mask.
- Strict load: all three checkpoints loaded with `strict=true`, no missing keys,
  no unexpected keys. Variant: `res_connection_in_dynamics_true`.

Player0-only was not run: the paired rows were already decisive and the
multi-checkpoint paired scorecard had done the requested held-out comparison.

## Baseline Sanity

Action histograms are `[up, stay, down]`.

| Row | Episodes | Wins | Truncs | Mean / median / p90 steps | Shaped return | Actions |
| --- | ---: | --- | ---: | --- | --- | --- |
| `random_uniform_vs_random_uniform` | 16 | `random_uniform:16` | 0 | 16.25 / 8 / 35.5 | `random_uniform:0.00397` | `random_uniform:[161,166,193]` |
| `random_uniform_vs_track_ball` | 32 | `random_uniform:0`, `track_ball:32` | 0 | 25.875 / 19 / 52 | `random_uniform:-0.98737`, `track_ball:1.0` | `random_uniform:[281,264,283]`, `track_ball:[296,234,298]` |
| `random_uniform_vs_lagged_track_ball_1` | 32 | `random_uniform:17`, `lagged_track_ball_1:15` | 0 | 12.469 / 8 / 19 | `random_uniform:0.06517`, `lagged_track_ball_1:-0.05908` | `random_uniform:[123,145,131]`, `lagged_track_ball_1:[131,106,162]` |
| `lagged_track_ball_1_vs_track_ball` | 32 | `lagged_track_ball_1:0`, `track_ball:28` | 4 | 141.875 / 19 / 924.6 | `lagged_track_ball_1:-0.86823`, `track_ball:0.875` | `lagged_track_ball_1:[189,4137,214]`, `track_ball:[193,4135,212]` |
| `track_ball_vs_track_ball` | 16 | `track_ball:0` | 16 | 1024 / 1024 / 1024 | `track_ball:0.0` | `track_ball:[622,31568,578]` |

## Checkpoint Rows

| Checkpoint | Opponent | Episodes | LZ wins | Opp wins | Truncs | Mean / median / p90 steps | LZ shaped | LZ actions |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `iter0` | `random_uniform` | 32 | 15 | 17 | 0 | 11.094 / 8 / 19 | -0.05959 | `[339,16,0]` |
| `iter0` | `lagged_track_ball_1` | 32 | 15 | 15 | 2 | 74.938 / 8 / 19 | 0.00267 | `[2327,71,0]` |
| `iter0` | `track_ball` | 32 | 0 | 27 | 5 | 176.719 / 19 / 1024 | -0.83559 | `[5457,198,0]` |
| `iter16` | `random_uniform` | 32 | 19 | 13 | 0 | 15.219 / 8 / 30 | 0.19009 | `[170,317,0]` |
| `iter16` | `lagged_track_ball_1` | 32 | 12 | 19 | 1 | 43.531 / 8 / 19 | -0.21526 | `[822,571,0]` |
| `iter16` | `track_ball` | 32 | 0 | 28 | 4 | 143.938 / 19 / 925.7 | -0.86722 | `[2978,1628,0]` |
| `best` | `random_uniform` | 32 | 15 | 17 | 0 | 10.406 / 8 / 8 | -0.05975 | `[237,96,0]` |
| `best` | `lagged_track_ball_1` | 32 | 14 | 18 | 0 | 12.812 / 8 / 19 | -0.12129 | `[298,112,0]` |
| `best` | `track_ball` | 32 | 0 | 27 | 5 | 178.094 / 24.5 / 1024 | -0.83492 | `[3655,2044,0]` |

Checkpoint-vs-checkpoint rows were also emitted by the multi-checkpoint wrapper:
`iter0` beat `iter16` 16-12 with 4 truncations, `best` beat `iter0` 16-12
with 4 truncations, and `iter16` tied `best` 14-14 with 4 truncations. These
rows are secondary; the requested read is checkpoint-vs-baseline.

## Read

No, checkpoints do not consistently improve over training in this trusted run.

`iter16` is better than `iter0` against `random_uniform` by wins, shaped return,
and p90 survival (`19-13` instead of `15-17`; shaped `0.190` instead of
`-0.060`; p90 `30` instead of `19`). But that gain does not transfer to the
scripted opponents. Against `lagged_track_ball_1`, `iter16` falls from an
even `15-15` with 2 truncations to `12-19` with 1 truncation, and mean survival
drops from `74.94` to `43.53`. Against `track_ball`, neither checkpoint wins,
and `iter16` has lower mean survival and worse shaped return than `iter0`.

`ckpt_best` is not a rescue checkpoint. It loses to `random_uniform` and
`lagged_track_ball_1`, gets 0 wins against `track_ball`, and still never chooses
action index 2. All three checkpoints have zero `down` actions in every
checkpoint-vs-baseline row, so the remaining issue still looks like policy
quality or wiring, not checkpoint discovery or strict-load failure.

No pytest.
