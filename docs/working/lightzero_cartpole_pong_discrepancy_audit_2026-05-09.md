# LightZero Sparse-Reward Discrepancy Audit - 2026-05-09

Goal: explain why official LightZero examples run while our dummy Pong trains
but does not become a reliable checkpoint policy. CartPole is only a control:
it proves Modal plus LightZero plus checkpoint plumbing can run. It is not the
closest learning analogue for dummy Pong.

Short answer: the closest simple official pattern is not CartPole. It is
LightZero's board-game `play_with_bot_mode` family: TicTacToe, Connect4, and
Gomoku. Those examples also use sparse/delayed terminal outcomes and sometimes
hide a bot move inside one agent-facing environment step. But they pair that
with large final-outcome targets (`td_steps` to game length, `discount_factor=1`),
`board_games` policy mode, legal-action masks, bigger collection/update volume,
and a turn-based state contract. Our dummy Pong borrowed much more from the
CartPole/Atari single-agent template, while having board-game-like terminal
reward and an opponent hidden inside the env step.

## Closest Official Patterns

### 1. Board Games, Play With Bot: Closest Simple Analogue

This is the nearest official match to current dummy Pong.

Why it matches:

- It is sparse terminal reward. Connect4 documents `0` on nonterminal steps and
  terminal `+1/-1` in bot mode
  (`/tmp/lightzero-src/zoo/board_games/connect4/envs/connect4_env.py:26-28`).
- It hides the opponent inside the env step. Connect4 first applies the agent
  move, then the bot move, then flips the bot reward back to the agent
  perspective (`/tmp/lightzero-src/zoo/board_games/connect4/envs/connect4_env.py:240-265`).
- It sets `to_play=-1` in bot mode because MCTS should see a single-agent
  surface, not alternating players
  (`/tmp/lightzero-src/zoo/board_games/connect4/envs/connect4_env.py:245-263`).
- It trains with much larger sparse-reward settings than our Pong smokes:
  Connect4 bot mode uses `collector_env_num=8`, `n_episode=8`,
  `num_simulations=50`, `update_per_collect=50`, `batch_size=256`,
  and `max_env_step=5e5`
  (`/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py:6-13`).
- Most importantly, it makes the value target reach the final outcome:
  `td_steps=int(6*7/2)` and `discount_factor=1`
  (`/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py:40-55`).
  TicTacToe does the same with `td_steps=9`, `discount_factor=1`
  (`/tmp/lightzero-src/zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py:45-60`).

Why it is still not identical:

- Board games are turn-based. Dummy Pong is simultaneous: our wrapper asks
  LightZero for one ego action, samples/scripts the opponent action, then steps
  both together (`src/curvyzero/training/lightzero_dummy_pong_env.py:168-220`).
- Board games have shrinking legal-action masks. Dummy Pong always marks all
  three actions legal (`src/curvyzero/training/lightzero_dummy_pong_env.py:368-377`).
- Board-game observations encode ownership and legal moves on a board. Dummy
  Pong uses a custom 10-float ego feature vector
  (`src/curvyzero/training/lightzero_dummy_pong_features.py:14-32`).

Practical read: if we want a simple official template for sparse dummy Pong,
start from board-game bot mode ideas, not from CartPole.

### 2. Board Games, Self Play: Closest Future Self-Play Pattern

Gomoku self-play is close to a future two-sided dummy Pong lane, but not to the
current single-ego-vs-scripted-opponent wrapper.

Official self-play keeps delayed terminal outcomes, but exposes alternating
players via `to_play=current_player` and uses full-game outcome targets. Gomoku
self-play sets `battle_mode='self_play_mode'`, `env_type='board_games'`,
`action_type='varied_action_space'`, `game_segment_length=board_size*board_size`,
`td_steps=board_size*board_size`, and `discount_factor=1`
(`/tmp/lightzero-src/zoo/board_games/gomoku/config/gomoku_muzero_sp_mode_config.py:22-67`).
Its env reward is terminal-only in practice: `_player_step` returns
`float(winner == current_player)`, so nonterminal and draw positions are `0`
(`/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py:316-360`).

Practical read: use this pattern if dummy Pong becomes true LightZero self-play.
For the current hidden-opponent wrapper, bot mode is closer.

### 3. Official Atari Pong/Breakout: Closest Theme, Not Closest Simple Setup

Official Atari Pong is obviously closest in name and score structure, but it is
not a tiny/simple example. It assumes image observations, ALE wrappers, clipped
rewards, large budgets, and heavy update volume.

PongNoFrameskip config uses 8 collector envs, 8 episodes per collect, 50 MCTS
simulations, `update_per_collect=1000`, `batch_size=256`,
`max_env_step=1e6`, `game_segment_length=400`, 4x96x96 frame-stack
observations, and action size 6
(`/tmp/lightzero-src/lzero/agent/config/muzero/gym_pongnoframeskip_v4.py:6-69`).
Breakout is similar, with action size 4 and explicit collect/eval episode caps
(`/tmp/lightzero-src/lzero/agent/config/muzero/gym_breakoutnoframeskip_v4.py:6-71`).
The Atari env forwards the ALE reward, accumulates return, always exposes a
full action mask, and sets `to_play=-1`
(`/tmp/lightzero-src/zoo/atari/envs/atari_lightzero_env.py:169-209`).

Practical read: Atari says sparse/delayed arcade score needs much more
experience and update pressure than our tiny dummy Pong smokes. It does not say
CartPole-scale knobs should work.

### 4. Atari Segment Configs: Closest Long-Horizon Replay Pattern

The segment configs are relevant because they are LightZero's official response
to longer Atari trajectories: train from fixed segments instead of treating a
whole episode like a tiny control task.

`atari_unizero_segment_config.py` uses `game_segment_length=20`,
`num_segments=8`, `num_unroll_steps=10`, `num_simulations=50`, replay ratio
`0.25`, and reanalysis settings
(`/tmp/lightzero-src/zoo/atari/config/atari_unizero_segment_config.py:10-34`,
`/tmp/lightzero-src/zoo/atari/config/atari_unizero_segment_config.py:78-103`).
The multitask MuZero segment config similarly uses `game_segment_length=20`,
`update_per_collect=80`, `td_steps=5`, `num_segments`, and delayed training
start after 2000 env steps
(`/tmp/lightzero-src/zoo/atari/config/atari_muzero_multitask_segment_ddp_config.py:95-158`).

Practical read: this is useful for dummy Pong once we care about long sequences,
but it is heavier than the simple board-game bot pattern.

### 5. LunarLander: Useful Vector-Control Reference, Not Sparse-Reward Reference

LunarLander is closer than CartPole in horizon and control difficulty, but it is
not a clean sparse terminal-reward analogue. The wrapper just forwards Gymnasium
LunarLander rewards and accumulates them
(`/tmp/lightzero-src/zoo/box2d/lunarlander/envs/lunarlander_env.py:141-169`),
and the config is a large vector MLP task: 8 observation dims, 4 actions,
`update_per_collect=200`, `batch_size=256`, `max_env_step=5e6`,
`game_segment_length=200`
(`/tmp/lightzero-src/lzero/agent/config/muzero/gym_lunarlander_v2.py:6-60`).

Practical read: use it as a "non-image MLP can train longer tasks" reference,
not as the sparse-Pong template.

### 6. CartPole: Control Only

CartPole proves our Modal lane can import official LightZero, run `train_muzero`,
return a policy, emit evaluator metrics, and write checkpoints
(`src/curvyzero/infra/modal/lightzero_cartpole_tiny_train_smoke.py:1-147`,
`docs/experiments/2026-05-09-modal-lightzero-official-example-sanity.md:68-130`).
It is dense-reward classic control: the wrapper forwards the Gym reward every
step and accumulates `eval_episode_return`
(`/tmp/lightzero-src/zoo/classic_control/cartpole/envs/cartpole_lightzero_env.py:130-140`).
Official CartPole also uses roomy defaults: `collector_env_num=8`, `n_episode=8`,
`update_per_collect=100`, `batch_size=256`, and `max_env_step=1e5`
(`/tmp/lightzero-src/zoo/classic_control/cartpole/config/cartpole_muzero_config.py:6-12`).

Practical read: CartPole passing rules out infrastructure breakage. It does not
validate our sparse Pong objective.

## Ranked Discrepancies For Dummy Pong

### 1. The Biggest Missing Pattern Is Final-Outcome Targeting

Official sparse board games explicitly set large `td_steps` and
`discount_factor=1` "to make sure the value target is the final outcome"
(`/tmp/lightzero-src/zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py:56-60`,
`/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py:51-54`,
`/tmp/lightzero-src/zoo/board_games/gomoku/config/gomoku_muzero_sp_mode_config.py:58-61`).

Our dummy Pong config patch sets `env_type='not_board_games'`,
`n_episode`, `game_segment_length`, `num_simulations`, `batch_size`, and
`update_per_collect`, but it does not patch `td_steps` or `discount_factor`
(`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:279-297`).
That leaves MuZero defaults of `td_steps=5` and `discount_factor=0.997`
(`/tmp/lightzero-src/lzero/policy/muzero.py:168-170`).

Simple read: Pong's only meaningful reward may arrive dozens of steps later,
but its value target is still using a short, discounted control-task default.
This is the first knob to align with official sparse examples.

### 2. Sparse Reward Volume Is Far Below Official Sparse Examples

Dummy Pong's default lane is intentionally tiny: `max_env_step=64`,
`max_train_iter=2`, `batch_size=8`, `update_per_collect=1`, `n_episode=1`
(`src/curvyzero/infra/modal/lightzero_dummy_pong_tiny_train_smoke.py:51-52`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:37-45`).
Even scaled attempts are still small compared with official sparse configs.

Official bot games use `update_per_collect=50`, `batch_size=256`, and 8 to 32
episodes per collect
(`/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py:6-13`,
`/tmp/lightzero-src/zoo/board_games/gomoku/config/gomoku_muzero_sp_mode_config.py:6-13`).
Official Atari Pong uses `update_per_collect=1000`, `batch_size=256`, and
`max_env_step=1e6`
(`/tmp/lightzero-src/lzero/agent/config/muzero/gym_pongnoframeskip_v4.py:6-12`).

Simple read: CartPole can show motion on tiny budgets because every survival
step is reward. Sparse Pong needs many more terminal examples.

### 3. Current Pong Is Board-Game-Like In Reward But Not In Policy Mode

Dummy Pong reward is exactly sparse terminal score: `0.0` until someone scores,
then `+1/-1`; timeout is `0.0`
(`src/curvyzero/training/dummy_pong.py:156-166`). The LightZero wrapper passes
that raw score reward through as the training reward
(`src/curvyzero/training/lightzero_dummy_pong_env.py:191-220`).
`shaped_loss_delay_return` is only logged after done; it is not the learned
reward (`src/curvyzero/training/lightzero_dummy_pong_env.py:398-405`,
`src/curvyzero/training/lightzero_dummy_pong_env.py:443-446`,
`src/curvyzero/training/lightzero_dummy_pong_env.py:709-719`).

But the config tells LightZero this is `not_board_games` with fixed action
space, not a board-game sparse-outcome task
(`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:279-297`).
That may be fine for all-actions-legal Pong, but it means we are not borrowing
the most important board-game sparse settings unless we add them explicitly.

Simple read: Pong's objective resembles board games; its config currently
resembles generic single-agent control.

### 4. Hidden Opponent Is Official, But Our Opponent Is Simultaneous

The official bot-mode pattern proves "opponent inside env step" is legitimate.
Connect4 bot mode performs agent then bot moves in one agent-facing step and
returns the result from the agent perspective
(`/tmp/lightzero-src/zoo/board_games/connect4/envs/connect4_env.py:240-265`).

Our Pong wrapper similarly supplies the opponent action internally, but the
underlying game is simultaneous. LightZero chooses ego action, the wrapper
chooses opponent action, and `PongEnv.step(joint_action)` advances both players
at once (`src/curvyzero/training/lightzero_dummy_pong_env.py:168-220`).
Opponents include `random_uniform`, `track_ball`, `lagged_track_ball_1`, and
checkpoint opponents (`src/curvyzero/training/lightzero_dummy_pong_env.py:288-366`).

Simple read: the hidden opponent is not disqualifying. The simultaneous dynamic
makes credit assignment harder than turn-based bot games.

### 5. Eval Collapse Is A Real Measurement Failure Mode

LightZero collect mode adds Dirichlet root noise and samples from visit counts
(`/tmp/lightzero-src/lzero/policy/muzero.py:746-783`). Eval mode removes root
noise and picks deterministic argmax over visits
(`/tmp/lightzero-src/lzero/policy/muzero.py:905-920`). The helper uses
`np.argmax`, so ties choose the lowest index
(`/tmp/lightzero-src/lzero/policy/utils.py:648-653`).

Our debug docs show this is happening. The lagged-opponent checkpoint had all
24 debug rows choose `stay`; every row had visits `[2,3,3]`, so `stay` beat
`down` only by tie order
(`docs/working/lightzero_pong_eval_action_collapse_debug_2026-05-09.md:255-282`).
Other runs collapsed to all `up` or all `down`, so this is weak-root behavior,
not one fixed action-map inversion
(`docs/working/lightzero_pong_eval_action_collapse_debug_2026-05-09.md:25-54`).

Simple read: deterministic MCTS eval is often exposing that the policy is not
confident, rather than proving it learned a stable Pong rule.

### 6. Horizon Semantics Were Fixed, But They Are Not The Root Cause

LightZero's `max_env_step` is a training-budget stop condition
(`/tmp/lightzero-src/lzero/entry/train_muzero.py:24-31`,
`/tmp/lightzero-src/lzero/entry/train_muzero.py:220-234`). Dummy Pong originally
let that knob double as episode horizon. The current patch splits this with
`pong_episode_max_steps`, though it still defaults to `max_env_step` when absent
(`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:139-148`,
`src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py:329-355`).

The horizon-fixed probe used train budget `1024` and Pong horizon `120` and
confirmed the split, but independent MCTS eval still collapsed to all `up`
(`[570, 0, 0]`)
(`docs/experiments/2026-05-09-lightzero-dummy-pong-horizon-fixed-probe.md:32-44`,
`docs/experiments/2026-05-09-lightzero-dummy-pong-horizon-fixed-probe.md:124-133`).

Simple read: fixing horizon was necessary hygiene. It did not supply the sparse
terminal learning pattern.

### 7. Observation And Action Scale Are Plausible, But Untested Against Official Patterns

Dummy Pong uses a 10-float ego observation and three actions (`up`, `stay`,
`down`) (`src/curvyzero/training/dummy_pong.py:17-19`,
`src/curvyzero/training/lightzero_dummy_pong_features.py:14-32`). Baselines use
the same action ids and emit all three actions
(`src/curvyzero/training/dummy_pong_eval.py:93-164`).

Official board games use spatial ownership planes and legal-action masks.
Official Atari uses frame stacks and arcade wrappers. Dummy Pong is a custom
middle ground: tiny vector features for a sparse/delayed, opponent-mediated
game.

Simple read: action ids look low suspicion; representation quality and root
separation are higher suspicion.

## Bottom Line

CartPole passing means the LightZero lane runs. It does not mean dummy Pong
should learn under CartPole-like settings.

The closest simple official examples are TicTacToe/Connect4/Gomoku bot mode:
sparse terminal outcome, hidden bot inside env step, `to_play=-1`, large
collector/update volume, `td_steps` long enough to see the final outcome, and
`discount_factor=1`. Official Atari Pong/Breakout support the same scale lesson
from the arcade side: delayed score learning is expensive and uses much bigger
budgets or segment machinery. LunarLander is not a sparse-reward guide.

## Practical Next Diagnostics

1. Add an official-sparse config probe for dummy Pong: keep
   `pong_episode_max_steps=120`, set `td_steps=120`, `discount_factor=1`,
   `game_segment_length=120`, and compare to the current defaults.

2. Run the same probe with official board-game-ish volume before judging policy:
   at least `n_episode>=8`, `batch_size=256` if feasible, and
   `update_per_collect` closer to `50` than `1`.

3. Keep the existing first-N MCTS debug rows for every checkpoint: observation,
   policy logits, visit counts, selected action, and whether ball-vs-paddle
   geometry calls for `up`, `stay`, or `down`.

4. Add a same-observation collect-vs-eval diagnostic:
   `collect_mode.forward(..., epsilon=0)` versus `eval_mode.forward(...)`.
   This isolates MCTS noise/sampling from deterministic eval tie behavior.

5. Run a labeled reward ablation: raw sparse score reward versus a temporary
   training-only shaped reward, with held-out evaluation still on raw score.
   Pass/fail is action diversity plus wins, not shaped score alone.

6. If pursuing true self-play, make a separate lane patterned after Gomoku
   self-play. Do not mix that diagnosis with the current hidden scripted
   opponent lane.
