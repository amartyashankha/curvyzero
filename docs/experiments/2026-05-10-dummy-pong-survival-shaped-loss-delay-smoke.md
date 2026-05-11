# Dummy Pong Survival-Shaped Loss-Delay Smoke - 2026-05-10

Purpose: launch the smallest clearly labeled shaped-objective Pong training
path where survival/loss-delay is part of the training return, not just eval
telemetry.

This is custom dummy Pong only. It is not stock LightZero Atari Pong, not the
LightZero dummy Pong MuZero sparse-reward control, and not comparable as stock
reward. In this note, survival means steps survived. The fraction is only
`steps_survived / max_steps`.

## Command

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.dummy_pong_survival_curriculum_train_attempt \
  --run-id pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0 \
  --attempt-id survival-shaped-loss-delay-alpha0.5-smoke8192-s0 \
  --epochs 8 \
  --games-per-epoch 8 \
  --eval-games 4 \
  --max-steps 120 \
  --survival-weight 0.5 \
  --truncation-bonus 0.0 \
  --reward-mode loss_delay \
  --seed 0
```

Training rollout upper bound: `8 * 8 * 120 = 7680` env steps, chosen as the
cheap first smoke near an 8192-step budget.

## Reward

True score is logged separately:

```text
win:      +1.0
loss:     -1.0
timeout:   0.0
```

The shaped training return for this ablation is:

```text
win:      +1.0
loss:     -1.0 + 0.5 * (episode_steps / max_steps)
timeout:   0.0
```

Warning: do not promote or compare this run using `mean_training_return` as if
it were stock sparse reward. Read true score, wins/losses/timeouts, steps
survived, action histograms, and shaped training return together.

## Result

The Modal run completed and committed artifacts.

- Modal app: `ap-att7Gn5sZMB5uYkVoCSF1F`
- Run id: `pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0`
- Attempt id: `survival-shaped-loss-delay-alpha0.5-smoke8192-s0`
- Summary:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0/attempts/survival-shaped-loss-delay-alpha0.5-smoke8192-s0/train/summary.json`
- Rows:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0/attempts/survival-shaped-loss-delay-alpha0.5-smoke8192-s0/train/survival_shaped_rows.jsonl`
- Checkpoint:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-smoke8192-s0/checkpoints/iteration-000008/checkpoint.npz`
- Checkpoint sha256:
  `53e1ef80704b6792178f9094598806b5b89640c2cf750e24ae4afea5dfbb8082`

Final eval highlights from the returned summary:

| Opponent | Episodes | Wins | Losses | Timeouts | Mean score | Mean steps | Mean training return | Action note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `random_uniform` | 8 | 5 | 3 | 0 | 0.25 | 20.375 | 0.273958 | collapsed greedy eval to `down` |
| `weak_track_ball` | 8 | 1 | 6 | 1 | -0.625 | 34.375 | -0.548438 | collapsed greedy eval to `down` |
| `track_ball` | 8 | 0 | 7 | 1 | -0.875 | 34.375 | -0.794271 | collapsed greedy eval to `down` |

Steps-survived read:

- Raw score did not materially improve. Compared with the epoch-1 eval,
  `random_uniform` stayed at `0.25`, `weak_track_ball` moved from `-0.75` to
  `-0.625`, and `track_ball` stayed at `-0.875`.
- Steps survived did improve in the final eval: `random_uniform` rose from
  `12.125` to `20.375`, `weak_track_ball` from `17.625` to `34.375`, and
  `track_ball` from `27.5` to `34.375`.
- The secondary survival fraction rose with those steps:
  `0.101042 -> 0.169792` versus
  `random_uniform`, `0.146875 -> 0.286458` versus `weak_track_ball`, and
  `0.229167 -> 0.286458` versus `track_ball`.
- Shaped eval return improved only where the policy survived longer or lost
  slightly less badly: `random_uniform` stayed `0.273958`,
  `weak_track_ball` improved from `-0.686458` to `-0.548438`, and
  `track_ball` improved from `-0.822917` to `-0.794271`.
- The last epoch's training rollout itself was still poor: raw score `-1.0`,
  `23.125` mean steps, `0.192708` survival fraction, and shaped training
  return `-0.903646`.
- Final greedy eval action histograms are fully collapsed to `down`:
  `163/0/0`, `275/0/0`, and `275/0/0` for
  `down/stay/up` against the three opponents.

Read: plumbing pass and shaped-objective smoke only. The loss-delay objective
nudged steps survived up, but there is no credible raw Pong skill improvement
because score barely moved and the final greedy policy collapsed to action
`down`. It proves the run path can train on a visible loss-delay survival
return and write durable artifacts. It does not prove policy quality.

## Larger 24-Epoch Same-Lane Run

Command:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.dummy_pong_survival_curriculum_train_attempt \
  --run-id pong-survival-shaped-loss-delay-alpha0.5-epochs24-s0 \
  --attempt-id survival-shaped-loss-delay-alpha0.5-epochs24-s0 \
  --epochs 24 \
  --games-per-epoch 16 \
  --eval-games 8 \
  --max-steps 120 \
  --survival-weight 0.5 \
  --truncation-bonus 0.0 \
  --reward-mode loss_delay \
  --seed 0
```

Training rollout upper bound: `24 * 16 * 120 = 46080` env steps. This is the
next slightly larger shaped-objective dummy Pong lane only, still separate from
stock Atari Pong and sparse-reward dummy Pong.

Result:

- Modal app: `ap-f5ftgocWh7HFoEPdwhEdFi`
- Run id: `pong-survival-shaped-loss-delay-alpha0.5-epochs24-s0`
- Attempt id: `survival-shaped-loss-delay-alpha0.5-epochs24-s0`
- Summary:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-epochs24-s0/attempts/survival-shaped-loss-delay-alpha0.5-epochs24-s0/train/summary.json`
- Rows:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-epochs24-s0/attempts/survival-shaped-loss-delay-alpha0.5-epochs24-s0/train/survival_shaped_rows.jsonl`
- Checkpoint:
  `training/dummy-pong-survival-shaped/pong-survival-shaped-loss-delay-alpha0.5-epochs24-s0/checkpoints/iteration-000024/checkpoint.npz`
- Checkpoint sha256:
  `387f47c799e76230d9caf2ccbe716f87a5b44167cfe5b159e359fb338116e9c2`

Final eval highlights from the returned summary:

| Opponent | Episodes | Wins | Losses | Timeouts | Mean score | Mean steps | Mean survival fraction | Mean training return | Action note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `random_uniform` | 16 | 10 | 6 | 0 | 0.25 | 14.875 | 0.123958 | 0.282552 | collapsed greedy eval to `up` |
| `weak_track_ball` | 16 | 3 | 13 | 0 | -0.625 | 19.6875 | 0.164063 | -0.554948 | collapsed greedy eval to `up` |
| `track_ball` | 16 | 0 | 14 | 2 | -0.875 | 27.5 | 0.229167 | -0.822917 | collapsed greedy eval to `up` |

Last epoch train rollout:

- `16` episodes, `0` wins, `15` losses, `1` timeout.
- Mean score return `-0.9375`.
- Mean steps `26.0`, survival fraction `0.216667`.
- Mean shaped training return `-0.860417`.
- Sampled train actions were not fully collapsed: `down=66`, `stay=107`,
  `up=243`.

Read: larger plumbing run completed, but it is not a quality improvement.
Compared with the 8-epoch smoke, final raw score is unchanged against all three
opponents, and final greedy eval still collapses to one action, now `up`
instead of `down`. Final steps survived are lower than the smoke against
`random_uniform` and `weak_track_ball`, and lower against `track_ball` if using
the smoke's final eval mean steps. Keep this as shaped-objective evidence and
artifact lineage, not a Pong-skill claim.
