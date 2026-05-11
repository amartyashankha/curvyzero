# 2026-05-09 Modal LightZero Official Connect4 Sparse Smoke

Date: 2026-05-09.

Purpose: check whether a slightly larger official sparse/delayed terminal
LightZero board-game example can run cheaply on Modal after the official
TicTacToe MuZero bot-mode smoke passed.

## Decision

Feasible, with one narrow packaging caveat.

Inspected:

```text
/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py
src/curvyzero/infra/modal/lightzero_tictactoe_tiny_train_smoke.py
```

The official Connect4 config is a sparse board-game MuZero bot-mode example:

```text
battle_mode: play_with_bot_mode
bot_action_type: rule
observation_shape: (3, 6, 7)
action_space_size: 7
game_segment_length: 21
td_steps: 21
discount_factor: 1
reward/value support: (-300, 301, 1)
```

`LightZero==0.2.0` on Modal did not include
`zoo.board_games.connect4`, so the wrapper mounts only the local official
`/tmp/lightzero-src/zoo` snapshot ahead of the pip package on `PYTHONPATH`.
The trainer entrypoint remains stock `lzero.entry.train_muzero` from the
Modal-installed LightZero package.

Added:

```text
src/curvyzero/infra/modal/lightzero_connect4_tiny_train_smoke.py
```

Tiny caps:

```text
max_env_step: 64
max_train_iter: 4
collector_env_num: 1
evaluator_env_num: 1
n_evaluator_episode: 1
n_episode: 1
num_simulations: 5
batch_size: 4
update_per_collect: 10
eval_freq: 1
cuda: false
num_unroll_steps: 3
```

## Commands

Compile:

```sh
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_connect4_tiny_train_smoke.py
```

Remote dry smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_connect4_tiny_train_smoke \
  --mode dry \
  --run-id official-connect4-sanity-20260509 \
  --attempt-id attempt-dry-smoke-2
```

Remote train smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_connect4_tiny_train_smoke \
  --mode train \
  --run-id official-connect4-sanity-20260509 \
  --attempt-id attempt-train-smoke-1
```

No pytest was run.

## Results

Initial dry attempt:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-Miu1w4f2LA7F5WmwfUJDmm
status: failed
blocker: ModuleNotFoundError: No module named 'zoo.board_games.connect4'
```

Passing dry attempt after mounting `/tmp/lightzero-src/zoo`:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-Es7xaAr1O1rQvVMnkfH1iw
ok: true
status: completed
remote_elapsed_sec: 17.901028
problems: []
```

Tiny train attempt:

```text
Modal URL: https://modal.com/apps/modal-labs/shankha-dev/ap-e3OU2fHc4ig3KHnH38JVYZ
ok: true
status: completed
train_result.ok: true
return_type: MuZeroPolicy
remote_elapsed_sec: 28.19186
train elapsed_sec: 8.978004
problems: []
```

Key train signals:

```text
training_iterations: [0]
checkpoint_iterations: [0, 10]
final_rewards: [-1.0]
eval_episode_return_mean: -1.0
envstep_count: 4.0
episode_count: 1.0
total_loss_avg: 30.459156
policy_loss_avg: 4.864775
reward_loss_avg: 19.195786
value_loss_avg: 25.594379
target_reward_avg: -0.1875
target_value_avg: -0.625
```

Mirrored artifacts:

```text
summary: training/lightzero-official-connect4/official-connect4-sanity-20260509/attempts/attempt-train-smoke-1/train/summary.json
training_signals: training/lightzero-official-connect4/official-connect4-sanity-20260509/attempts/attempt-train-smoke-1/train/lightzero_training_signals.json
lightzero_artifacts: training/lightzero-official-connect4/official-connect4-sanity-20260509/attempts/attempt-train-smoke-1/train/lightzero_artifacts_manifest.json
checkpoint_manifest: training/lightzero-official-connect4/official-connect4-sanity-20260509/checkpoints/lightzero/manifest.json
ckpt_best: training/lightzero-official-connect4/official-connect4-sanity-20260509/checkpoints/lightzero/ckpt_best.pth.tar
iteration_0: training/lightzero-official-connect4/official-connect4-sanity-20260509/checkpoints/lightzero/iteration_0.pth.tar
iteration_10: training/lightzero-official-connect4/official-connect4-sanity-20260509/checkpoints/lightzero/iteration_10.pth.tar
```

## Pong Comparison

The useful overlap with Pong is narrow:

- Both can exercise sparse/delayed reward handling and final-outcome targets.
- Connect4 is closer to board-game/self-play or bot-play patterns: legal action
  masks, discrete turns, and short terminal episodes.
- Connect4 observation shape is compact structured state `(3, 6, 7)`.

Do not treat this as a Pong substitute. Pong has rich visual/pixel
observations, temporal motion cues, Atari/ALE/ROM dependencies in the stock
LightZero path, and more continuous-ish paddle/ball dynamics. This Connect4
smoke only says the official sparse board-game MuZero path can execute cheaply
on Modal CPU.

## Interpretation

Connect4 is a useful second official sparse sanity check. It extends the
TicTacToe result to a larger board, larger support range, and longer
final-outcome target horizon while staying cheap.

It does not prove Pong training viability or policy quality. The right Pong
gate remains the explicit Pong env/ROM path and then a tiny real Pong trainer
smoke once that environment is available.
