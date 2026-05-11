# 2026-05-09 LightZero Dummy Pong Frozen-Checkpoint Self-Play Smoke

## Question

Can the existing LightZero dummy Pong train attempt actually collect and train
with an env-owned frozen LightZero checkpoint opponent, while reporting survival
steps, shaped score, action counts, wins, checkpoint refs, and whether the
learner/opponent are live, scripted, or frozen?

## Code Change

Small telemetry-only patch:

```text
src/curvyzero/training/lightzero_dummy_pong_env.py
```

Terminal env rows now include:

```text
learner_control_kind: live
opponent_control_kind: scripted or frozen_checkpoint
```

The env-side scorecard now includes learner/opponent control kinds, opponent
policy ids, `opponent_is_frozen_checkpoint`, and frozen checkpoint refs.

## Verification

No pytest was run.

Compile command:

```sh
uv run python -m py_compile \
  src/curvyzero/training/lightzero_dummy_pong_env.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py \
  src/curvyzero/training/lightzero_dummy_pong_policy.py
```

Result: passed.

## Modal Smoke

Command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode train \
  --run-id frozen-selfplay-smoke-20260509 \
  --attempt-id attempt-mcts-opp-iter0-smoke-2 \
  --opponent-policy lightzero_mcts_checkpoint \
  --opponent-checkpoint ref:training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar \
  --opponent-checkpoint-label post_deep_seed_iter0 \
  --opponent-checkpoint-adapter mcts_eval_mode \
  --opponent-checkpoint-num-simulations 2 \
  --max-env-step 128 \
  --max-train-iter 2 \
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
https://modal.com/apps/modal-labs/shankha-dev/ap-CnPtTlNv3qLWBTnNnB7V62
```

Result:

```text
ok: true
status: completed
problems: []
called_train_muzero: true
run_id: frozen-selfplay-smoke-20260509
attempt_id: attempt-mcts-opp-iter0-smoke-2
```

Artifacts:

```text
summary: training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/attempts/attempt-mcts-opp-iter0-smoke-2/train/summary.json
episodes: training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/attempts/attempt-mcts-opp-iter0-smoke-2/train/episodes.jsonl
training_signals: training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/attempts/attempt-mcts-opp-iter0-smoke-2/train/lightzero_training_signals.json
lightzero_artifacts: training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/attempts/attempt-mcts-opp-iter0-smoke-2/train/lightzero_artifacts_manifest.json
```

## Scorecard

Trainer-side env scorecard:

| Metric | Value |
| --- | ---: |
| Episodes | 5 |
| Wins / losses / timeouts | 4 / 1 / 0 |
| Mean survival steps | 10.2 |
| Median survival steps | 8.0 |
| P90 survival steps | 14.6 |
| Max survival steps | 19.0 |
| Mean score return | 0.6 |
| Mean shaped loss-delay return | 0.60625 |

Action counts:

| Agent | Up | Stay | Down |
| --- | ---: | ---: | ---: |
| `player_0` live learner | 11 | 16 | 24 |
| `player_1` frozen opponent | 51 | 0 | 0 |

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
label: post_deep_seed_iter0
adapter: mcts_eval_mode
adapter_schema_id: curvyzero_lightzero_dummy_pong_mcts_eval_mode/v0
num_simulations: 2
source_ref: training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar
path: /runs/training/lightzero-dummy-pong/lz-dpong-20260509T154530Z-b049f29edb64/checkpoints/lightzero/iteration_0.pth.tar
sha256: 4b20241909346a52334d25d2fa4adc91349a5cc7314bf8c8dd7ce9bd8fae493e
state_key: model
```

New learner checkpoints mirrored by the smoke:

```text
training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/checkpoints/lightzero/ckpt_best.pth.tar
training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/checkpoints/lightzero/iteration_0.pth.tar
training/lightzero-dummy-pong/frozen-selfplay-smoke-20260509/checkpoints/lightzero/iteration_2.pth.tar
```

## Read

Frozen-checkpoint self-play is ready to run now in the limited honest sense:
LightZero still controls one live ego learner, and the env supplies the
simultaneous opponent action from a frozen checkpoint policy loaded in-container.
This is not full current-policy learner-vs-learner simultaneous self-play.

The chosen checkpoint is poor. The frozen opponent selected `up` every recorded
step, so this smoke proves plumbing and telemetry, not opponent strength.
