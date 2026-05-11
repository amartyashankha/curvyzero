# 2026-05-09 LightZero Dummy Pong MCTS Scorecard

## Question

Can the 512/8 LightZero MuZero dummy Pong checkpoint run full episode eval
through LightZero MCTS/eval-mode, outside the training loop?

## Source

- Algorithm: LightZero MuZero.
- Checkpoint: 512/8 `iteration_8.pth.tar`.
- Eval mode: LightZero MCTS/eval-mode.
- Episodes: 16 per seating.
- `num_simulations`: 8.
- Strict full model load: OK.
- Strict-load variant: `res_connection_in_dynamics_true`.
- Modal URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-Ou59sqrdljB295FFBpyIUP`.

Artifacts:

```text
eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/summary.json
eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-20260509T150000Z-20260509T150243Z/episodes.jsonl
```

## Rows

| Row | LZ wins | Opp wins | Mean survival | LZ shaped | LZ reward | LZ actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lightzero_iter8_vs_lagged_track_ball_1` | 13 | 15 | 25.09 | -0.0397 | -0.0625 | `[801,2,0]` |
| `lightzero_iter8_vs_random_uniform` | 17 | 15 | 13.84 | 0.0953 | 0.0625 | `[443,0,0]` |
| `lightzero_iter8_vs_track_ball` | 0 | 30 | 25.66 | -0.8618 | -0.9375 | `[816,5,0]` |

## Read

MCTS eval-mode is no longer just a loader smoke. Full episode eval works.

The checkpoint still does not look good. Combined LightZero action histogram is
`[2060,7,0]`, so it is effectively up-only and never chooses down in this
scorecard. The next blocker is policy quality/training signal, not checkpoint
loading.

Bug-sweep items are investigation leads, not proven root causes:

- Direct policy-head argmax can collapse weak/tied logits to action `0`, but
  MCTS also mostly chooses up, so this only explains the control path.
- Horizon/config mismatch risk remains: training used `max_env_step=512`,
  checkpoint scoring can default to `64`, and independent eval currently uses
  `PongConfig()` default `max_steps=120`.
- `DummyPongLightZeroEnv.random_action()` reseeds every call, so helper calls
  can repeat inside an episode.
- `timestep` compatibility is wrapper-local, not in the base env observation.

Next: the fixed-horizon rerun and longer `iteration_64` scorecard below make
the remaining question trainer/eval mismatch or objective/wiring, not
checkpoint loading, horizon mismatch, or simply-too-short training.

No pytest.

## Corrected 512-step Rerun

The config bug was fixed and this rerun matches the checkpoint training horizon:
`PongConfig(max_steps=lightzero_max_env_step)` with `max_env_step=512`.

Command:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt --checkpoints lightzero:iter8_maxstep512=ref:training/lightzero-dummy-pong/lz-dpong-20260509T144635Z-eb5a0ed35de0/checkpoints/lightzero/iteration_8.pth.tar --episodes 16 --seed 303 --eval-id mcts-scoreboard-512x8-iter8-maxstep512 --max-env-step 512 --num-simulations 8
```

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-zUwRanuyB0OHCA8NdpOHVQ`

Artifacts:

```text
eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-maxstep512-20260509T151544Z/summary.json
eval/lightzero-dummy-pong/mcts-scoreboard-512x8-iter8-maxstep512-20260509T151544Z/episodes.jsonl
```

Config checks from `summary.json`:

- `config.max_steps`: 512
- `lightzero_eval_config.max_env_step`: 512
- `lightzero_eval_config.num_simulations`: 8

Rows:

| Row | Kind | Episodes | Mean reward | Wins | Mean steps | P90 steps | Trunc rate | Actions |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `lagged_track_ball_1_vs_lagged_track_ball_1` | baseline | 16 | `{"lagged_track_ball_1":0.0}` | `{"lagged_track_ball_1":13}` | 105.25 | 512.0 | 0.1875 | `{"lagged_track_ball_1":[136,3090,142]}` |
| `lagged_track_ball_1_vs_track_ball` | baseline | 32 | `{"lagged_track_ball_1":-0.6875,"track_ball":0.6875}` | `{"lagged_track_ball_1":0,"track_ball":22}` | 172.03 | 512.0 | 0.3125 | `{"lagged_track_ball_1":[197,5118,190],"track_ball":[196,5119,190]}` |
| `lightzero_iter8_maxstep512_vs_lagged_track_ball_1` | MCTS | 32 | `{"lagged_track_ball_1":0.0,"lightzero_iter8_maxstep512":0.0}` | `{"lagged_track_ball_1":14,"lightzero_iter8_maxstep512":14}` | 74.44 | 462.7 | 0.125 | `{"lagged_track_ball_1":[146,2087,149],"lightzero_iter8_maxstep512":[2373,9,0]}` |
| `lightzero_iter8_maxstep512_vs_random_uniform` | MCTS | 32 | `{"lightzero_iter8_maxstep512":-0.125,"random_uniform":0.125}` | `{"lightzero_iter8_maxstep512":14,"random_uniform":18}` | 12.81 | 19.0 | 0.0 | `{"lightzero_iter8_maxstep512":[407,3,0],"random_uniform":[135,135,140]}` |
| `lightzero_iter8_maxstep512_vs_track_ball` | MCTS | 32 | `{"lightzero_iter8_maxstep512":-0.90625,"track_ball":0.90625}` | `{"lightzero_iter8_maxstep512":0,"track_ball":29}` | 64.88 | 41.0 | 0.09375 | `{"lightzero_iter8_maxstep512":[2069,7,0],"track_ball":[215,1608,253]}` |
| `random_uniform_vs_lagged_track_ball_1` | baseline | 32 | `{"lagged_track_ball_1":0.1875,"random_uniform":-0.1875}` | `{"lagged_track_ball_1":19,"random_uniform":13}` | 16.59 | 41.0 | 0.0 | `{"lagged_track_ball_1":[139,234,158],"random_uniform":[172,180,179]}` |
| `random_uniform_vs_random_uniform` | baseline | 16 | `{"random_uniform":0.0}` | `{"random_uniform":16}` | 16.94 | 30.0 | 0.0 | `{"random_uniform":[168,177,197]}` |
| `random_uniform_vs_track_ball` | baseline | 32 | `{"random_uniform":-1.0,"track_ball":1.0}` | `{"random_uniform":0,"track_ball":32}` | 24.50 | 50.9 | 0.0 | `{"random_uniform":[263,263,258],"track_ball":[240,325,219]}` |
| `track_ball_vs_track_ball` | baseline | 16 | `{"track_ball":0.0}` | `{"track_ball":0}` | 512.00 | 512.0 | 1.0 | `{"track_ball":[472,15382,530]}` |

Read: corrected MCTS eval still shows the trained policy is effectively
up-only. Across the three LightZero rows, its aggregate action histogram is
`[4849,19,0]`: 99.61% action 0, 0.39% action 1, and 0 down actions. Matching
the 512-step env horizon does not remove the collapse.

## Longer Train And Post-Train Scorecard

Long CPU train succeeded after the scaled wrapper caps were intentionally
changed to `8192/64`.

Modal URL:
`https://modal.com/apps/modal-labs/shankha-dev/ap-Uj3XVYgEnzr9oSVan3NILH`

Run:
`lz-dpong-20260509T151212Z-b95b61de2eb0`

Attempt:
`attempt-20260509T151212Z-8b9db08f8fcb`

Train summary:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/train/summary.json
```

`iteration_64` checkpoint:

```text
training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/checkpoints/lightzero/iteration_64.pth.tar
sha256 11a0cc80f797ce8e63150e0a6018efc163b7858bed9efd92b77dda8cadaf95e4
```

Trainer-side telemetry: 578 episodes, 535 wins, 43 losses, no timeouts,
survival mean 18.18, median/p90 19/19, max 52, score mean 0.8512, shaped mean
0.8513, player_0 actions `[9539,800,170]`.

Post-train independent MCTS scorecard:

```text
modal_url: https://modal.com/apps/modal-labs/shankha-dev/ap-G8BlfW9uUBtT7jTKxgtx0U
summary: training/lightzero-dummy-pong/lz-dpong-20260509T151212Z-b95b61de2eb0/attempts/attempt-20260509T151212Z-8b9db08f8fcb/eval/mcts-scoreboard-4096x64-iter64-maxstep4096/summary.json
checkpoint: iteration_64
max_env_step: 4096
num_simulations: 8
```

Rows:

| Row | LZ wins | Opp wins | Mean steps | LZ shaped | LZ actions |
| --- | ---: | ---: | ---: | ---: | --- |
| vs `random_uniform` | 13 | 19 | 15.91 | -0.1862 | `[290,219,0]` |
| vs `lagged_track_ball_1` | 11 | 19 | 266.25 | -0.2492 | `[6475,2045,0]` |
| vs `track_ball` | 0 | 31 | 144.34 | -0.9668 | `[3335,1284,0]` |

Read: longer training moved from almost pure up to up+stay, but still zero down
and does not beat random or scripted baselines. This points away from
simply-too-short and toward a trainer/eval mismatch or remaining
objective/wiring issue.
