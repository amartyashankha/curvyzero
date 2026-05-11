# LightZero Self-Play Architecture Audit - 2026-05-09

## Short Answer

LightZero uses "self-play" in two different ways depending on the environment.

For Atari, CartPole, and our current LightZero dummy Pong adapter, the simple
meaning is: the current policy repeatedly plays the environment, the collector
saves those episodes, and the learner trains from that replay. That is policy
rollout data. It is not two live learners playing each other.

For board games such as Gomoku, LightZero has a true two-player self-play mode.
The same learner policy can make moves for both sides of the game, and the env
tracks whose turn it is with `to_play`. That is different from board-game
`play_with_bot_mode`, where player 1 is the learner and player 2 is a scripted
or search bot.

## What LightZero Does

The MuZero trainer builds collector envs, evaluator envs, one policy, one
collector, one replay buffer, and one learner. In the main loop it asks the
collector for new episodes, pushes those game segments into replay, samples
training data, and updates the policy.

Source anchors:

- `/tmp/lightzero-src/lzero/entry/train_muzero.py:73` creates collector and
  evaluator envs.
- `/tmp/lightzero-src/lzero/entry/train_muzero.py:94` creates one policy with
  learn, collect, and eval modes.
- `/tmp/lightzero-src/lzero/entry/train_muzero.py:111` creates the MuZero
  collector.
- `/tmp/lightzero-src/lzero/entry/train_muzero.py:187` collects new data.
- `/tmp/lightzero-src/lzero/entry/train_muzero.py:193` pushes collected game
  segments into replay.
- `/tmp/lightzero-src/lzero/entry/train_muzero.py:201` samples replay for
  learning.
- `/tmp/lightzero-src/lzero/entry/train_muzero.py:215` trains the learner.

The collector does the policy-env loop. It builds policy input from ready env
observations, calls the policy, steps the env with the selected actions, and
stores actions, rewards, observations, `action_mask`, and `to_play` into a game
segment.

Source anchors:

- `/tmp/lightzero-src/lzero/worker/muzero_collector.py:308` defines episode
  collection.
- `/tmp/lightzero-src/lzero/worker/muzero_collector.py:414` builds policy input.
- `/tmp/lightzero-src/lzero/worker/muzero_collector.py:426` calls policy forward.
- `/tmp/lightzero-src/lzero/worker/muzero_collector.py:454` steps the env.
- `/tmp/lightzero-src/lzero/worker/muzero_collector.py:481` appends transition
  data to the game segment.

## Single-Agent Style Envs

CartPole and Atari expose one learner action per env step. They return
`to_play=-1`, meaning there is no alternating player perspective for MCTS.

Source anchors:

- `/tmp/lightzero-src/zoo/classic_control/cartpole/config/cartpole_muzero_config.py:45`
  marks CartPole as `env_type='not_board_games'`.
- `/tmp/lightzero-src/zoo/classic_control/cartpole/envs/cartpole_lightzero_env.py:97`
  returns `to_play=-1` on reset.
- `/tmp/lightzero-src/zoo/classic_control/cartpole/envs/cartpole_lightzero_env.py:130`
  steps the Gym env with the learner action.
- `/tmp/lightzero-src/zoo/classic_control/cartpole/envs/cartpole_lightzero_env.py:147`
  returns `to_play=-1` after a step.
- `/tmp/lightzero-src/zoo/atari/config/atari_muzero_config.py:64` marks Atari as
  `env_type='not_board_games'`.
- `/tmp/lightzero-src/zoo/atari/envs/atari_lightzero_env.py:178` steps the Atari
  env with the learner action.
- `/tmp/lightzero-src/zoo/atari/envs/atari_lightzero_env.py:209` returns
  `to_play=-1`.

Plain language: for these envs, "self-play" is better read as "the current
agent generates its own training data by playing the environment."

## Board Games

Gomoku has explicit modes:

- `self_play_mode`: the env applies one move, swaps the current player, and
  reports the next `to_play`. The same learner policy can be used for both
  sides over the course of the game.
- `play_with_bot_mode`: the learner plays player 1, then the env immediately
  asks a bot for player 2's move.
- `eval_mode`: similar to bot mode, except it is for evaluation or human play.

Source anchors:

- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:64` sets default
  `battle_mode='self_play_mode'`.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:126` allows
  `self_play_mode`, `play_with_bot_mode`, and `eval_mode`.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:150` defines two
  players.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:212` sets
  `to_play` to the current player in self-play mode.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:241` branches on
  battle mode.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:242` uses one
  learner action in self-play mode.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:251` starts
  bot mode.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:263` gets the
  bot action.
- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:337` swaps the
  current player after a move.
- `/tmp/lightzero-src/zoo/board_games/gomoku/config/gomoku_muzero_sp_mode_config.py:25`
  configures `self_play_mode`.
- `/tmp/lightzero-src/zoo/board_games/gomoku/config/gomoku_muzero_bot_mode_config.py:27`
  configures `play_with_bot_mode`.

Plain language: board-game self-play is real two-sided play. Bot mode is not.

## Our Current LightZero Dummy Pong Setup

Our LightZero dummy Pong wrapper is single-ego. LightZero controls one paddle.
The wrapper supplies the other paddle action from `opponent_policy`.

Source anchors:

- `src/curvyzero/training/lightzero_dummy_pong_env.py:3` says LightZero controls
  one ego paddle while the wrapper supplies the opponent action.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:72` defaults the opponent
  to `random_uniform`.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:150` resets one episode.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:161` steps with one
  LightZero action.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:169` asks the opponent
  policy for the other paddle action.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:176` builds the joint
  action for both paddles.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:184` steps the underlying
  Pong env.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:281` creates scripted or
  checkpoint-backed opponent policies.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:288` allows frozen
  LightZero checkpoint opponent modes.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:365` returns the ego
  observation to LightZero.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:368` returns `to_play=-1`.
- `src/curvyzero/training/lightzero_dummy_pong_env.py:424` records the opponent
  policy id in telemetry.
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:240`
  patches the policy env type to `not_board_games`.
- `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:276`
  registers the custom env and MuZero policy.
- `src/curvyzero/infra/modal/lightzero_dummy_pong_train_attempt.py:55` defaults
  training to `opponent_policy='random_uniform'`.

Answers:

- Are we generating data from repeated play? Yes. LightZero collects repeated
  episodes from the current learner policy interacting with the dummy Pong env.
- Are we playing against scripted opponents? Yes by default. Current common
  opponents are `random_uniform`, `track_ball`, and `lagged_track_ball_1`.
- Do we currently have true learner-vs-learner self-play? No.
- Do we currently have frozen-checkpoint self-play? We have support for a frozen
  checkpoint as the opponent, but it is a configured opponent mode, not yet an
  automated self-play training loop or league.

There is also a separate older NumPy self-play replay path. It can run the same
behavior policy on both Pong seats and write one replay row per ego player per
step. That is useful scaffolding, but it is not LightZero MuZero self-play.

Source anchors:

- `src/curvyzero/training/dummy_pong_selfplay_replay.py:70` builds a dummy Pong
  self-play replay dataset.
- `src/curvyzero/training/dummy_pong_selfplay_replay.py:117` says the same
  behavior policy controls both seats.
- `src/curvyzero/training/dummy_pong_selfplay_replay.py:148` records the policy
  used by each agent.
- `src/curvyzero/training/dummy_pong_selfplay_replay.py:207` creates joint
  actions for both agents.
- `src/curvyzero/training/dummy_pong_selfplay_train.py:163` labels that trainer
  as a tiny NumPy policy/value update, not MuZero search.
- `src/curvyzero/training/dummy_pong_selfplay_train.py:241` says it does not
  prove MuZero search.

## What Needs To Change

For the next practical self-play step, do frozen-checkpoint opponent training:

1. Train or select a LightZero checkpoint.
2. Start a new LightZero training run where the learner controls one paddle and
   the opponent is that frozen checkpoint.
3. Pair seats in evaluation so a checkpoint is tested as both player 0 and
   player 1.
4. Write checkpoint lineage into telemetry: parent checkpoint, opponent
   checkpoint, adapter, state key, feature mode, seed set, and scorecard refs.
5. Gate promotion with held-out scorecards against scripted opponents and older
   checkpoints.

For true live learner-vs-learner Pong, the current single-ego wrapper is not
enough. We need either:

- a collector/env path that asks a policy for both paddles before stepping the
  simultaneous Pong env; or
- a turn/perspective wrapper with clean `to_play` semantics, if we deliberately
  turn the problem into a board-game-like interface.

For later CurvyTron multiplayer, the likely shape is joint-action self-play:
multiple agents, policy identities per seat, opponent/checkpoint pools, paired
or randomized seats, and replay rows that preserve each ego perspective.

## Backlog Pointer

The backlog now points here from the frozen-checkpoint self-play lane. Keep the
short rule simple: current LightZero dummy Pong is learner-vs-scripted unless
the opponent is explicitly configured as a frozen checkpoint. Full live
learner-vs-learner self-play is not present yet.
