# 2026-05-09 LightZero dummy Pong contact-pressure modest rung

## Question

After the 2-iteration contact-pressure smoke failed held-out quality, does one
modest same-curriculum LightZero MuZero rung improve the scoreable
`lagged_track_ball_1` contact-pressure target without action collapse?

## Setup

- Algorithm: LightZero MuZero.
- Environment: project-owned custom dummy Pong, not stock Atari Pong.
- Feature mode: `tabular_ego`.
- Reset profile: `pong_reset_profile=contact_pressure`.
- Training pressure agent: `pong_reset_pressure_agent=ego`.
- Held-out scorecard pressure agent: `pong_reset_pressure_agent=player_0` with
  `--no-paired-seats`.
- Opponent for training: `lagged_track_ball_1`.
- Reward: unchanged sparse env reward only, ego score `+1`, opponent score
  `-1`, non-score step `0`.
- Horizon/settings: fixed 64-step Pong episode horizon and 64-step LightZero
  eval horizon.
- Budget: one bounded 8-iteration diagnostic request. LightZero wrote
  `iteration_0`, `iteration_3`, and `ckpt_best`; no further rung was launched.
- No pytest.

## Commands

Train:

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 37 \
  --opponent-policy lagged_track_ball_1 \
  --ego-agent player_0 \
  --max-env-step 64 \
  --pong-episode-max-steps 64 \
  --pong-reset-profile contact_pressure \
  --pong-reset-pressure-agent ego \
  --max-train-iter 8 \
  --collector-env-num 2 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 2 \
  --num-simulations 2 \
  --batch-size 16 \
  --update-per-collect 1 \
  --n-episode 2 \
  --game-segment-length 32 \
  --td-steps 64 \
  --num-unroll-steps 5 \
  --discount-factor 1.0 \
  --reward-support-min -1 \
  --reward-support-max 1 \
  --reward-support-delta 1 \
  --value-support-min -1 \
  --value-support-max 1 \
  --value-support-delta 0.01
```

Matching contact-pressure/player0 MCTS scorecard:

```sh
uv run --extra modal python -B -m modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt \
  --checkpoints lightzero:iter0=ref:training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/checkpoints/lightzero/iteration_0.pth.tar,lightzero:iter3=ref:training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/checkpoints/lightzero/iteration_3.pth.tar,lightzero:best=ref:training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/checkpoints/lightzero/ckpt_best.pth.tar \
  --episodes 16 \
  --seed 71 \
  --split-id dummy_pong_contact_pressure_modest_rung \
  --split-role monitor \
  --run-id lz-dpong-20260509T175407Z-77159cc3a6b4 \
  --attempt-id attempt-20260509T175407Z-8105d62c1e00 \
  --eval-id mcts-scoreboard-contact-pressure-modest-rung \
  --max-env-step 64 \
  --num-simulations 2 \
  --feature-mode tabular_ego \
  --pong-reset-profile contact_pressure \
  --pong-reset-pressure-agent player_0 \
  --no-paired-seats
```

The first scorecard launch `ap-V2BeTtlf7StHzM0mbd2vtc` failed during Modal
packaging because a local `scripts/__pycache__/*.pyc` changed during the build.
It produced no eval artifacts. The rerun used `python -B` and completed.

## Results

- Train app: `ap-Zr829nRQJqi3WqnTUEwHwr`.
- Contact-pressure MCTS scorecard app: `ap-r5iWQT58qLeLGLIDQ4kDUM`.
- Train run:
  `lz-dpong-20260509T175407Z-77159cc3a6b4`,
  attempt `attempt-20260509T175407Z-8105d62c1e00`.
- Train summary:
  `training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/attempts/attempt-20260509T175407Z-8105d62c1e00/train/summary.json`.
- Eval summary:
  `training/lightzero-dummy-pong/lz-dpong-20260509T175407Z-77159cc3a6b4/attempts/attempt-20260509T175407Z-8105d62c1e00/eval/mcts-scoreboard-contact-pressure-modest-rung/summary.json`.
- Local fetched summaries:
  `artifacts/local/contact-pressure-modest-rung-2026-05-09/train-summary.json`
  and
  `artifacts/local/contact-pressure-modest-rung-2026-05-09/contact-pressure-summary.json`.

Trainer-side telemetry over 12 episodes:

| Episodes | Wins | Losses | Timeouts | Survival mean/median/p90 | Shaped mean | Raw score mean | Learner actions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 12 | 4 | 6 | 2 | 20.0 / 15.0 / 60.3 | -0.1335 | -0.1667 | up=135 stay=74 down=31 |

The trainer-side action histogram includes all three actions.

Held-out contact-pressure/player0 MCTS scoreable-target rows versus
`lagged_track_ball_1`:

| Checkpoint | Wins/losses/trunc | Survival mean/median/p90 | Shaped mean | Raw score mean | Actions | All three actions? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `iteration_0` | 4/10/2 | 15.875 / 5.0 / 45.0 | -0.3433 | -0.3750 | up=170 stay=84 down=0 | no |
| `iteration_3` | 1/11/4 | 21.75 / 5.0 / 64.0 | -0.5879 | -0.6250 | up=235 stay=113 down=0 | no |
| `ckpt_best` | 7/6/3 | 24.625 / 15.0 / 64.0 | 0.0981 | 0.0625 | up=271 stay=123 down=0 | no |

Held-out baseline sanity rows:

| Checkpoint | Opponent | Wins/losses/trunc | Survival mean/median/p90 | Shaped mean | Raw score mean | Actions |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `iteration_0` | `random_uniform` | 9/7/0 | 14.5625 / 15.0 / 26.5 | 0.1616 | 0.1250 | up=147 stay=86 down=0 |
| `iteration_3` | `random_uniform` | 7/9/0 | 16.25 / 15.5 / 27.0 | -0.0620 | -0.1250 | up=163 stay=97 down=0 |
| `ckpt_best` | `random_uniform` | 6/10/0 | 15.625 / 10.0 / 38.0 | -0.1953 | -0.2500 | up=176 stay=74 down=0 |
| `iteration_0` | `track_ball` | 0/12/4 | 28.9375 / 26.0 / 64.0 | -0.6489 | -0.7500 | up=292 stay=171 down=0 |
| `iteration_3` | `track_ball` | 0/10/6 | 33.6875 / 26.0 / 64.0 | -0.5493 | -0.6250 | up=361 stay=178 down=0 |
| `ckpt_best` | `track_ball` | 0/14/2 | 21.5 / 26.0 / 45.5 | -0.7695 | -0.8750 | up=221 stay=123 down=0 |

Baseline policy sanity under the same contact-pressure eval stayed plausible:
`track_ball` self-play truncated 16/16 at 64 steps, and
`lagged_track_ball_1` self-play used all three actions
`up=37 stay=1376 down=39`.

## Interpretation

This rung is a bounded diagnostic failure, not a campaign seed.

There is a small `ckpt_best` improvement on the intended scoreable target
relative to `iteration_0` versus `lagged_track_ball_1`: raw score moved from
`-0.3750` to `0.0625`, shaped return from `-0.3433` to `0.0981`, and survival
mean from `15.875` to `24.625`. However, that is not enough to pass the gate:
every held-out learned checkpoint row still has `down=0`, and the final
checkpoint `iteration_3` is worse than initialization on the scoreable target.

The normal default-reset scorecard was intentionally skipped. The requested
stop condition triggered on the matching held-out contact-pressure scorecard:
action collapse persisted, final held-out quality collapsed, and the modest
curriculum did not produce a robust improvement. Launching the default-reset
eval would add cost without changing the go/no-go decision.

## Decision

Stop this contact-pressure rung. Do not launch a campaign or a longer
same-curriculum follow-up from this result. Do not use default `track_ball` as a
scoreable contact-pressure target; the earlier scoreability probe found
`track_ball` scoreability at 0/64.

The next useful work is not more length on this rung. It is a bug/objective
investigation into why LightZero MCTS eval remains unable to use all three
actions on custom dummy Pong, despite trainer-side exploration seeing all three
actions.
