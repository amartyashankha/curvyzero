# 2026-05-09 Modal LightZero Official Sparse Example Sanity

Date: 2026-05-09.

Purpose: prove a cheap official LightZero sparse/delayed terminal-reward path
can run on Modal before using heavier Atari/Pong examples.

## Choice

Inspected `/tmp/lightzero-src/zoo/board_games` and the existing Modal wrappers
under `src/curvyzero/infra/modal/`.

Chosen example:

```text
zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config
lzero.entry.train_muzero
```

Reason: TicTacToe is the cheapest official board-game MuZero example found. It
uses `play_with_bot_mode` with terminal game outcome reward. The official config
sets `td_steps=9` and `discount_factor=1`, matching final-outcome targets.

The tiny shape follows
`/tmp/lightzero-src/lzero/mcts/tests/config/tictactoe_muzero_bot_mode_config_for_test.py`:

```text
max_env_step: 64
max_train_iter: 4
collector_env_num: 1
n_episode: 1
evaluator_env_num: 1
n_evaluator_episode: 1
num_simulations: 5
batch_size: 4
update_per_collect: 10
eval_freq: 1
cuda: false
battle_mode: play_with_bot_mode
bot_action_type: v0
game_segment_length: 5
td_steps: 9
num_unroll_steps: 3
discount_factor: 1
reward/value support: (-10, 11, 1)
```

## Commands

Compile only:

```sh
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_tictactoe_tiny_train_smoke.py
```

Remote dry smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke \
  --mode dry \
  --run-id official-tictactoe-sanity-20260509 \
  --attempt-id attempt-dry-smoke-2
```

Remote train smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke \
  --mode train \
  --run-id official-tictactoe-sanity-20260509 \
  --attempt-id attempt-train-smoke-1
```

No pytest was run.

## Dry Result

First dry run exposed a package/source mismatch: the Modal-installed
`LightZero==0.2.0` TicTacToe config did not include `reward_support_range` or
`value_support_range`, while `/tmp/lightzero-src` did. The wrapper now adds both
support ranges explicitly.

Passing dry run:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-eN3rc8G8MLeqbNt5auG2ow
ok: true
status: completed
problems: []
```

## Train Result

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-LuBUWp8htU26CROO2YjiDD
ok: true
status: completed
train_result.ok: true
return_type: MuZeroPolicy
remote_elapsed_sec: 28.020082
train elapsed_sec: 10.397254
problems: []
```

Key signals:

```text
training_iterations: [0]
checkpoint_iterations: [0, 10]
final_rewards: [1.0]
eval_episode_return_mean: 1.0
envstep_count: 4.0
total_loss_avg: 16.57254
policy_loss_avg: 4.394449
reward_loss_avg: 9.133568
value_loss_avg: 12.17809
target_reward_avg: -0.25
target_value_avg: -0.5
```

Artifacts:

```text
summary: training/lightzero-official-tictactoe/official-tictactoe-sanity-20260509/attempts/attempt-train-smoke-1/train/summary.json
training_signals: training/lightzero-official-tictactoe/official-tictactoe-sanity-20260509/attempts/attempt-train-smoke-1/train/lightzero_training_signals.json
lightzero_artifacts: training/lightzero-official-tictactoe/official-tictactoe-sanity-20260509/attempts/attempt-train-smoke-1/train/lightzero_artifacts_manifest.json
checkpoint_manifest: training/lightzero-official-tictactoe/official-tictactoe-sanity-20260509/checkpoints/lightzero/manifest.json
ckpt_best: training/lightzero-official-tictactoe/official-tictactoe-sanity-20260509/checkpoints/lightzero/ckpt_best.pth.tar
iteration_0: training/lightzero-official-tictactoe/official-tictactoe-sanity-20260509/checkpoints/lightzero/iteration_0.pth.tar
iteration_10: training/lightzero-official-tictactoe/official-tictactoe-sanity-20260509/checkpoints/lightzero/iteration_10.pth.tar
```

## Interpretation

The official sparse/delayed TicTacToe MuZero bot-mode path runs on Modal CPU,
returns a `MuZeroPolicy`, emits learner/evaluator signals, writes LightZero
artifacts, and mirrors checkpoints into `curvyzero-runs`.

This is not a quality proof. It only proves that the cheap official
sparse/delayed reward path can execute end to end.
