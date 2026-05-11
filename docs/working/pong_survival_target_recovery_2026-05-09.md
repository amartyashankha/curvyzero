# Pong Survival Target Recovery - 2026-05-09

Status: historical diagnostic. Do not use this as the current Pong plan. The
active spine is `docs/working/pong_selfplay_training_plan_2026-05-09.md` plus
`docs/working/pong_training_critique_wave_2026-05-09.md`; the self-play lane is
under critique after gen2 lost to its parent and won 0 games against
`track_ball`.

Current correction: the checkpoint scoreboard now reports survival and shaped
loss-delay telemetry. Future Pong eval summaries must not collapse to only
`0/N wins`.

Scope: reward-shaping recovery note for dummy Pong. No web was needed; this
uses local docs and code only.

## Finding

Keep `PongEnv.rewards`, eval summaries, and checkpoint scoreboards score-delta
only. The existing docs are explicit that rally length, survival time, paddle
hits, and distance-to-ball should not become v0 reward or eval metrics.

The viable small recovery path is not an environment reward change. It is a
separate training target for short-lookahead action relabeling:

```text
score_return = sum(score_delta rewards over the candidate rollout)
loss_delay_bonus = alpha * normalized_steps_until_loss
training_target_return = score_return + loss_delay_bonus only when score_return < 0
```

This gives partial credit when every candidate loses, preferring the action
that delays the loss. It should not give bonus on wins or truncations, and it
should not be reported as eval reward.

I did not find an existing Pong loss-delay or survival-shaped target. The
closest implementation is the current score-delta short-lookahead relabeler.

## Implementation Status

This is now implemented as a training-label option, not an environment reward
change.

- `scripts/build_dummy_pong_lookahead_replay.py` accepts
  `--loss-delay-alpha`.
- `src/curvyzero/training/dummy_pong_lookahead_replay.py` ranks candidate
  actions by `score_delta_return + alpha * steps_run / lookahead_steps`, but
  only when `score_delta_return < 0`.
- Replay rows write both raw score return and shaped training return:
  `target_score_delta_return`, `target_loss_delay_bonus`, `target_return`, and
  `target_return_kind`.
- The shaped target uses
  `dummy_pong_score_delta_loss_delay_short_lookahead_target_v0`.
- `PongEnv.rewards` and the checkpoint scoreboard were not changed.

Scoreboard telemetry update:

- `src/curvyzero/training/dummy_pong_eval.py` now stores `max_steps` per
  episode and computes `truncation_rate`, `median_steps`, `p90_steps`,
  `std_steps`, `survival_steps`, `score_return_stats_by_policy`, and
  `shaped_loss_delay_return_stats_by_policy`.
- `scripts/run_dummy_pong_checkpoint_scoreboard.py` now preserves those fields
  in `scoreboard_rows`.
- The shaped loss-delay readout uses the same simple rule already documented:
  win `+1.0`, loss `-1.0 + 0.5 * episode_steps / max_steps`, timeout `0.0`.
- This is eval telemetry and selection context. It is not a change to
  `PongEnv.rewards`.
- Smoke record:
  `docs/experiments/2026-05-09-dummy-pong-scoreboard-telemetry-patch-smoke.md`.

Metadata smoke:

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 1 \
  --seed 9 \
  --max-steps 40 \
  --lookahead-steps 16 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy track_ball \
  --loss-delay-alpha 0.05 \
  --output-dir artifacts/local/dummy-pong-loss-delay-metadata-smoke-2026-05-09
```

Result: 49 rows. The shaped target schema and target policy id were present,
and summary stats now separate `target_return_stats` from
`target_score_delta_return_stats`.

First training smoke:

- Replay: 251 rows.
- Targets different from `track_ball`: 0.
- Training validation accuracy: about 0.860.
- Scoreboard against `track_ball`: learned checkpoint won 0/32,
  `track_ball` won 27/32, and 5/32 episodes truncated.

Interpretation: the code path works, but `loss-delay-alpha 0.05` with
`track_ball` tie-break did not create a new action signal. It mostly asks the
learner to copy `track_ball`, so it should not be scaled as-is.

## Prior Local Research

- `docs/research/pong_reward_design.md` says dummy Pong v0 reward is score
  delta only and warns that rewarding both players for longer rallies can teach
  stalling or timeout farming.
- `docs/research/observation_reward_design.md` marks survival tick bonus as
  non-default because it can reward stalling/timeouts, and keeps timeout
  telemetry separate from terminal payoff.
- `docs/working/pong_training_plan.md` keeps Pong scoreboards honest:
  learned-vs-`random_uniform`, learned-vs-`track_ball`, win/loss/truncation,
  and mean score reward stay the progress metrics.
- `docs/working/pong_angle_learning_next_steps_2026-05-09.md` says contact and
  angle probes are debug signals; policy progress still needs score-delta
  return or pressure on a fresh recorded multi-start scoreboard.
- `docs/experiments/2026-05-09-dummy-pong-lookahead-angle-tie-g32-h32.md`
  is the relevant negative result: one-step score-delta labels plus
  `angle_control` tie-break produced many non-`track_ball` targets but no wins
  against `track_ball`.

## Existing Hooks

- `src/curvyzero/training/dummy_pong.py` already keeps environment reward pure:
  scoring step gives `+1/-1`, non-scoring step gives `0`, and max-step timeout
  sets `truncated` without a reward.
- `src/curvyzero/training/dummy_pong_eval.py` records `steps`, `truncated`,
  `winner`, score rewards, survival-step stats, and shaped loss-delay stats in
  eval rows/summaries. This is enough to keep eval honest while observing
  longevity.
- `src/curvyzero/training/dummy_pong_lookahead_replay.py` is the smallest
  implementation hook. `_evaluate_candidate` already returns
  `score_delta_return`, `terminated`, `truncated`, `winner`, and `steps_run`.
  `_lookahead_row` already stores candidate terminal metadata and chooses a
  target action from candidate returns.
- `src/curvyzero/training/dummy_pong_value_train.py` backs up score-delta
  returns from `reward_after_step`; leave this alone unless a separate shaped
  value-target experiment is explicitly created.
- `src/curvyzero/training/dummy_survival.py` has the survival/longevity pattern
  for the solo toy: terminal reward is still sparse, and mean/max steps are
  telemetry. It is useful as a reminder to log longevity separately.

## Smallest Implementation Path

1. Add optional CLI flag `--loss-delay-alpha`. Done.
2. Thread it into `build_dummy_pong_lookahead_replay`. Done.
3. Rank losing candidates by loss-delayed training return. Done.
4. Preserve raw score return and shaped training return in replay rows. Done.
5. Use a separate loss-delay target schema id. Done.
6. Train with the existing imitation trainer and score with the existing
   scoreboard. Done once; result was negative.

## Historical Commands

The existing score-delta lookahead path can be rerun for diagnostics:

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 4 \
  --seed 0 \
  --max-steps 120 \
  --lookahead-steps 32 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy track_ball \
  --output-dir artifacts/local/dummy-pong-score-delta-lookahead-current-smoke
```

The current scoreboard remains the honest eval command:

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 16 \
  --seed 17 \
  --split-id dummy_pong_monitor \
  --split-role monitor \
  --checkpoint candidate=artifacts/local/some-pong-policy/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-candidate-scoreboard
```

## Verification Commands

The implementation was checked with:

```sh
uv run python -m py_compile \
  src/curvyzero/training/dummy_pong_lookahead_replay.py \
  scripts/build_dummy_pong_lookahead_replay.py
```

The first training smoke used this replay command:

```sh
uv run python scripts/build_dummy_pong_lookahead_replay.py \
  --games-per-seat 4 \
  --seed 11 \
  --max-steps 120 \
  --lookahead-steps 32 \
  --collector-policy random_uniform \
  --include-ties \
  --tie-break-policy track_ball \
  --loss-delay-alpha 0.05 \
  --output-dir artifacts/local/dummy-pong-loss-delay-lookahead-smoke-2026-05-09
```

```sh
uv run python scripts/train_dummy_pong_imitation.py \
  --replay-path artifacts/local/dummy-pong-loss-delay-lookahead-smoke-2026-05-09 \
  --output-dir artifacts/local/dummy-pong-loss-delay-policy-smoke-2026-05-09 \
  --seed 0 \
  --epochs 80 \
  --learning-rate 0.5 \
  --validation-fraction 0.2
```

```sh
uv run python scripts/run_dummy_pong_checkpoint_scoreboard.py \
  --episodes 16 \
  --seed 17 \
  --split-id dummy_pong_loss_delay_smoke \
  --split-role monitor \
  --checkpoint loss_delay=artifacts/local/dummy-pong-loss-delay-policy-smoke-2026-05-09/checkpoint.npz \
  --output-dir artifacts/local/dummy-pong-loss-delay-scoreboard-smoke-2026-05-09
```

## Guardrails

- Do not change `PongEnv.rewards`.
- Do not add survival reward to eval summaries or selection records.
- Do not reward longer truncations. A timeout is a guardrail metric, not a win.
- Do not call the target "reward" in artifacts except when referring to raw
  `reward_after_step`; use `training_return` or `target_return`.
- Treat success as better scoreboard behavior: fewer quick losses to
  `track_ball`, more truncation pressure without random-opponent collapse, or
  actual score wins.
- Do not scale alpha `0.05` with `track_ball` tie-break as-is; it produced no
  non-`track_ball` target actions.
