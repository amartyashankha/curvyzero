# 2026-05-09 LightZero dummy Pong contact-pressure sim16 target gate

## Question

Before doing more custom dummy Pong training, does a higher train-time MCTS
budget make the contact-pressure root target sane enough to trust?

The gate is simple: on known scoreable contact-pressure states where the sparse
oracle wins with `down`, MCTS root visits at 16/25 simulations should give mass
to `down`, preferably top mass. No more blind custom Pong runs.

## Support Scale

I inspected the existing wrapper surface before launching any follow-up after
the gate. The train wrapper exposes support-range knobs:

- `--reward-support-min/max/delta`
- `--value-support-min/max/delta`

It does not expose or log an actual compiled `policy.model.support_scale`,
`reward_support_size`, or `value_support_size` knob. Existing notes warn that
pinned `LightZero==0.2.0` may still compile `support_scale=300` even when the
summary records small support ranges.

I did not add a support-scale patch here. That would need at least train config
logging and likely checkpoint reconstruction/eval loader handling for changed
support-head sizes. For this run, support scale remains open.

## Commands

This tiny training run was launched before the later "root target first" update.
It used train-time `--num-simulations 16`.

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --seed 38 \
  --opponent-policy lagged_track_ball_1 \
  --ego-agent player_0 \
  --max-env-step 64 \
  --pong-episode-max-steps 64 \
  --pong-reset-profile contact_pressure \
  --pong-reset-pressure-agent ego \
  --max-train-iter 4 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 16 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-episode 1 \
  --game-segment-length 32 \
  --td-steps 64 \
  --num-unroll-steps 5 \
  --discount-factor 1.0 \
  --reward-support-min -1 \
  --reward-support-max 1 \
  --reward-support-delta 1 \
  --value-support-min -1 \
  --value-support-max 1 \
  --value-support-delta 0.01 \
  --run-id lz-dpong-contact-sim16-s38 \
  --attempt-id train-64x4-sim16-contact
```

Root-target gate:

```sh
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_contact_pressure_oracle \
  --checkpoint-refs iteration_0=training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/checkpoints/lightzero/iteration_0.pth.tar,iteration_4=training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/checkpoints/lightzero/iteration_4.pth.tar,ckpt_best=training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/checkpoints/lightzero/ckpt_best.pth.tar \
  --state-seeds 20260510,20260515,20260523 \
  --num-simulations 16,25 \
  --run-id lz-dpong-contact-sim16-s38 \
  --attempt-id train-64x4-sim16-contact \
  --eval-id root-target-oracle-sim16-train \
  --max-env-step 64 \
  --feature-mode tabular_ego \
  --seed 71
```

Matching scorecard:

```sh
uv run --extra modal python -B -m modal run \
  -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt \
  --checkpoints lightzero:iter0=ref:training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/checkpoints/lightzero/iteration_0.pth.tar,lightzero:iter4=ref:training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/checkpoints/lightzero/iteration_4.pth.tar,lightzero:best=ref:training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/checkpoints/lightzero/ckpt_best.pth.tar \
  --episodes 16 \
  --seed 72 \
  --split-id dummy_pong_contact_pressure_sim16_target_gate \
  --split-role monitor \
  --run-id lz-dpong-contact-sim16-s38 \
  --attempt-id train-64x4-sim16-contact \
  --eval-id mcts-scoreboard-contact-sim16-target-gate \
  --max-env-step 64 \
  --num-simulations 16 \
  --feature-mode tabular_ego \
  --pong-reset-profile contact_pressure \
  --pong-reset-pressure-agent player_0 \
  --no-paired-seats \
  --baseline-policies lagged_track_ball_1,random_uniform,track_ball
```

No pytest.

## Artifacts

- Train app: `ap-WpLvodol7X6NWITDONUqMR`
- Root-target app: `ap-rlRYRhrssXjnfiRAQPWtA6`
- Scorecard app: `ap-Ro6e2vFoBj6ZRkT8qizWst`
- Train summary:
  `training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/attempts/train-64x4-sim16-contact/train/summary.json`
- Root-target JSON:
  `training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/attempts/train-64x4-sim16-contact/eval/root-target-oracle-sim16-train/contact_pressure_state_action_oracle.json`
- Scorecard summary:
  `training/lightzero-dummy-pong/lz-dpong-contact-sim16-s38/attempts/train-64x4-sim16-contact/eval/mcts-scoreboard-contact-sim16-target-gate/summary.json`
- Local fetched summaries:
  `artifacts/local/train-contact-sim16-s38-summary.json`,
  `artifacts/local/root-target-oracle-sim16-train.json`,
  `artifacts/local/mcts-scoreboard-contact-sim16-target-gate-summary.json`

## Root-Target Gate

All three audited states were sparse-score `down` wins:

| State | Oracle score by action |
| --- | --- |
| `player_0-seed-20260510` | `up=-1`, `stay=0`, `down=1` |
| `player_0-seed-20260515` | `up=-1`, `stay=0`, `down=1` |
| `player_0-seed-20260523` | `up=-1`, `stay=0`, `down=1` |

MCTS root visits are `up/stay/down`:

| Checkpoint | Sims | Root visits on the three states | Read |
| --- | ---: | --- | --- |
| `iteration_0` | 16 | `[5,5,6]`, `[5,5,6]`, `[5,5,6]` | `down` top |
| `iteration_0` | 25 | `[8,8,9]`, `[8,8,9]`, `[8,8,9]` | `down` top |
| `iteration_4` | 16 | `[6,5,5]`, `[6,5,5]`, `[6,5,5]` | `up` top |
| `iteration_4` | 25 | `[9,8,8]`, `[9,8,8]`, `[9,8,8]` | `up` top |
| `ckpt_best` | 16 | `[5,5,6]`, `[5,5,6]`, `[5,5,6]` | `down` top |
| `ckpt_best` | 25 | `[8,8,9]`, `[8,8,9]`, `[8,9,8]` | `down` top on 2/3 |

This is not clean enough for a longer follow-up. Initialization has the desired
target, but four train iterations move the final checkpoint to an all-up target
on the same known down-winning states. `ckpt_best` partly preserves the useful
target, but even it is unstable at 25 sims.

## Train Telemetry

Trainer-side env telemetry over 9 episodes:

| Wins | Losses | Timeouts | Survival mean/median/p90 | Shaped mean | Raw score mean | Learner actions |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 6 | 1 | 23.56 / 16.0 / 60.8 | -0.3819 | -0.4444 | up=58 stay=32 down=122 |

The train row used all three actions and especially `down`, but that is not
enough. The root-target probe shows why selected/executed action telemetry
alone is not a safe gate.

## Matching Scorecard

Contact-pressure/player0, no paired seats, scorecard MCTS sims `16`:

| Checkpoint vs opponent | Wins | Raw score mean | Shaped mean | Survival mean/median/p90 | Actions up/stay/down |
| --- | ---: | ---: | ---: | --- | --- |
| `iter0` vs `lagged_track_ball_1` | 3/16 | -0.5000 | -0.4761 | 13.88 / 5.0 / 39.5 | 0 / 0 / 222 |
| `iter4` vs `lagged_track_ball_1` | 3/16 | -0.3750 | -0.3550 | 21.50 / 5.0 / 64.0 | 344 / 0 / 0 |
| `best` vs `lagged_track_ball_1` | 6/16 | -0.2500 | -0.1421 | 19.75 / 16.0 / 37.5 | 110 / 117 / 89 |
| `iter0` vs `random_uniform` | 6/16 | -0.2500 | -0.2183 | 11.25 / 5.0 / 21.0 | 0 / 0 / 180 |
| `iter4` vs `random_uniform` | 2/16 | -0.7500 | -0.6953 | 10.38 / 5.0 / 27.0 | 166 / 0 / 0 |
| `best` vs `random_uniform` | 10/16 | 0.2500 | 0.2852 | 14.12 / 15.0 / 16.0 | 72 / 75 / 79 |
| `iter0` vs `track_ball` | 0/16 | -0.8125 | -0.7524 | 19.69 / 5.0 / 64.0 | 0 / 0 / 315 |
| `iter4` vs `track_ball` | 0/16 | -0.5625 | -0.5205 | 33.38 / 26.5 / 64.0 | 534 / 0 / 0 |
| `best` vs `track_ball` | 0/16 | -0.9375 | -0.7646 | 26.12 / 26.5 / 37.5 | 161 / 128 / 129 |

## Decision

Do not launch a longer custom Pong training run from this result.

The useful path is either:

- implement persistent target telemetry for collect decisions, including root
  visit counts/child visits and selected action; or
- run preflight root-target probes at 16/25 sims before any new train, and only
  train when the target generator keeps oracle-winning actions alive.

Support-scale calibration is still open because the existing run wrapper does
not expose/log the actual compiled `policy.model.support_scale`.
