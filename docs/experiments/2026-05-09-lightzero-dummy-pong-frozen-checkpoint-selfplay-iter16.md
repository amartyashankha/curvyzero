# 2026-05-09 LightZero Dummy Pong Frozen-Checkpoint Self-Play Iter16

## Question

Can the frozen-checkpoint opponent lane run against a later parent checkpoint,
`iteration_16.pth.tar`, with the MCTS adapter and tiny bounded train settings?

## Train Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode train \
  --run-id frozen-selfplay-iter16-20260509 \
  --attempt-id attempt-mcts-opp-iter16-256x4 \
  --opponent-policy lightzero_mcts_checkpoint \
  --opponent-checkpoint ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar \
  --opponent-checkpoint-label post_deep_seed_iter16 \
  --opponent-checkpoint-adapter mcts_eval_mode \
  --opponent-checkpoint-num-simulations 2 \
  --max-env-step 256 \
  --max-train-iter 4 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 4 \
  --batch-size 8 \
  --update-per-collect 1 \
  --n-episode 1 \
  --game-segment-length 50
```

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-FqRXg04AsokQ93JeXAokfC
```

Result:

```text
ok: true
status: completed
problems: []
called_train_muzero: true
training_iterations: [0]
checkpoint_iterations: [0, 4]
train_result.elapsed_sec: 13.843257
final_rewards: [1.0, -1.0, 1.0, -1.0]
```

Artifacts:

```text
summary: training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/attempts/attempt-mcts-opp-iter16-256x4/train/summary.json
episodes: training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/attempts/attempt-mcts-opp-iter16-256x4/train/episodes.jsonl
training_signals: training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/attempts/attempt-mcts-opp-iter16-256x4/train/lightzero_training_signals.json
lightzero_artifacts: training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/attempts/attempt-mcts-opp-iter16-256x4/train/lightzero_artifacts_manifest.json
```

## Trainer-Side Scorecard

| Metric | Value |
| --- | ---: |
| Episodes | 9 |
| Wins / losses / timeouts | 6 / 3 / 0 |
| Mean survival steps | 12.88888888888889 |
| Median survival steps | 8.0 |
| P90 survival steps | 21.200000000000003 |
| Max survival steps | 30.0 |
| Mean score return | 0.3333333333333333 |
| Mean shaped loss-delay return | 0.3409288194444444 |
| Unique seeds | 7 |

Action counts:

| Agent | Control | Up | Stay | Down |
| --- | --- | ---: | ---: | ---: |
| `player_0` | live learner | 19 | 19 | 78 |
| `player_1` | frozen checkpoint | 49 | 67 | 0 |

Control and checkpoint identity:

```text
learner_control_kinds: [live]
learner_policy_kinds: [lightzero_train_muzero]
opponent_control_kinds: [frozen_checkpoint]
opponent_policy_ids: [lightzero_mcts_checkpoint]
opponent_is_frozen_checkpoint: true
```

Frozen opponent:

```text
label: post_deep_seed_iter16
adapter: mcts_eval_mode
adapter_schema_id: curvyzero_lightzero_dummy_pong_mcts_eval_mode/v0
num_simulations: 2
source_ref: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar
sha256: e2dd80f8a08b15d750f5c8c643051b8e11e63eb5ce44a5ab71fee9fecaf88ee8
state_key: model
```

New learner checkpoints mirrored by the smoke:

```text
training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/checkpoints/lightzero/ckpt_best.pth.tar
training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/checkpoints/lightzero/iteration_4.pth.tar
```

## Independent MCTS Scorecard

Primary command with the new learner checkpoint and the frozen parent
checkpoint:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_scoreboard_attempt \
  --checkpoints lightzero:iter4_frozen_iter16_smoke=ref:training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/checkpoints/lightzero/iteration_4.pth.tar,lightzero:parent_iter16=ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_16.pth.tar \
  --episodes 1 \
  --seed 23 \
  --run-id frozen-selfplay-iter16-20260509 \
  --attempt-id attempt-mcts-opp-iter16-256x4 \
  --eval-id mcts-scoreboard-iter4-vs-parent-iter16-e1 \
  --max-env-step 256 \
  --num-simulations 2 \
  --paired-seats
```

Modal URL:

```text
https://modal.com/apps/modal-labs/shankha-dev/ap-D9E9dTwVOB11XQq2Jhh3Ct
```

Artifacts:

```text
summary: training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/attempts/attempt-mcts-opp-iter16-256x4/eval/mcts-scoreboard-iter4-vs-parent-iter16-e1/summary.json
episodes: training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/attempts/attempt-mcts-opp-iter16-256x4/eval/mcts-scoreboard-iter4-vs-parent-iter16-e1/episodes.jsonl
total_episodes: 23
episodes_per_match: 1
```

Selected independent rows:

| Pair | Episodes | Wins | Mean steps | Mean shaped return | Action histograms `[up, stay, down]` |
| --- | ---: | --- | ---: | --- | --- |
| `iter4` vs `parent_iter16` | 2 | `iter4`: 1, `parent_iter16`: 1 | 8.0 | `iter4`: 0.0078125, `parent_iter16`: 0.0078125 | `iter4`: [6, 10, 0], `parent_iter16`: [6, 10, 0] |
| `iter4` vs `random_uniform` | 2 | `iter4`: 0, `random_uniform`: 2 | 13.5 | `iter4`: -0.9736328125, `random_uniform`: 1.0 | `iter4`: [6, 21, 0], `random_uniform`: [10, 8, 9] |
| `iter4` vs `lagged_track_ball_1` | 2 | `iter4`: 1, `lagged_track_ball_1`: 1 | 8.0 | `iter4`: 0.0078125, `lagged_track_ball_1`: 0.0078125 | `iter4`: [4, 12, 0], `lagged_track_ball_1`: [6, 2, 8] |

A first one-checkpoint scorecard also ran before the paired-parent rerun:

```text
summary: training/lightzero-dummy-pong/frozen-selfplay-iter16-20260509/attempts/attempt-mcts-opp-iter16-256x4/eval/mcts-scoreboard-iter16-frozen-smoke-iter4-e1/summary.json
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-kIf5UCC3A69KKIhi5gDwC3
```

## Read

The run proves the later frozen checkpoint can be used as an env-owned MCTS
opponent inside the bounded train attempt, and the independent MCTS scorecard
can compare the new learner checkpoint against random, lagged, and the frozen
parent checkpoint.

This is not a quality win. The independent scorecard is tiny, and both
LightZero checkpoints still show no `down` actions in the selected MCTS rows.

## Verification

No code was changed. No `py_compile` was needed. No pytest was run.
