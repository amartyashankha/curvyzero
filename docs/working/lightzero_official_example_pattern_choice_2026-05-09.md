# LightZero Official Example Pattern Choice - 2026-05-09

Goal: choose the closest official LightZero example pattern for the next serious
dummy Pong run. This is source inspection only; no pytest was run.

## Decision

Use a CartPole/LunarLander-style `train_muzero` episode collector as the base,
with two narrow Atari/board-game borrowings:

- borrow Atari's `update_per_collect=None` + `replay_ratio` pattern, or at
  least raise `update_per_collect` far above the current smoke value of `1`;
- borrow board-game-sized value/reward support ranges, because dummy Pong's raw
  reward/value scale is much closer to terminal `-1/0/+1` games than to the
  default `(-300, 300)` MuZero support.

Do not switch the next run to `train_muzero_segment` yet. Segment collection is
an official Atari pattern, but it is mostly solving long image-game collection
and reanalysis pressure. Our dummy Pong adapter is a short/medium-horizon,
single-ego, low-dimensional custom env with sparse terminal reward. We need one
clean episode-collector run with the right update budget, horizon separation,
and support scale before adding the segment collector's extra moving parts.

## Sources Read

- CartPole: `/tmp/lightzero-src/zoo/classic_control/cartpole/config/cartpole_muzero_config.py`
- Atari Pong/Breakout family: `/tmp/lightzero-src/zoo/atari/config/atari_muzero_config.py`,
  `/tmp/lightzero-src/zoo/atari/config/atari_muzero_segment_config.py`,
  `/tmp/lightzero-src/lzero/agent/config/muzero/gym_pongnoframeskip_v4.py`,
  `/tmp/lightzero-src/lzero/agent/config/muzero/gym_breakoutnoframeskip_v4.py`
- LunarLander: `/tmp/lightzero-src/zoo/box2d/lunarlander/config/lunarlander_disc_muzero_config.py`
- Board-game bot/self-play: `/tmp/lightzero-src/zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py`,
  `/tmp/lightzero-src/zoo/board_games/tictactoe/config/tictactoe_muzero_sp_mode_config.py`,
  `/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_bot_mode_config.py`,
  `/tmp/lightzero-src/zoo/board_games/connect4/config/connect4_muzero_sp_mode_config.py`
- Custom env docs: `/tmp/lightzero-src/docs/source/tutorials/envs/customize_envs.md`
- Our adapter/config surface: [lightzero_dummy_pong_config_import_smoke.py](/Users/shankha/curvy/src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py),
  [lightzero_dummy_pong_env.py](/Users/shankha/curvy/src/curvyzero/training/lightzero_dummy_pong_env.py)

## Setting Comparison

| Surface | CartPole | Atari Pong/Breakout | Atari segment | LunarLander | Board games | Our dummy Pong |
| --- | --- | --- | --- | --- | --- | --- |
| Entry | `train_muzero` | `train_muzero` | `train_muzero_segment` | `train_muzero` | `train_muzero` | `train_muzero` |
| Collector unit | `n_episode=8` | `n_episode=8` | `num_segments=8` | `n_episode=8` | `n_episode=8` | currently `n_episode=1` default |
| Model | MLP, obs `4`, actions `2` | Conv, stack `4x64x64`, Atari actions | Conv, stack `4x64x64` | MLP, obs `8`, actions `4` | Small conv/image board | MLP, obs `10` or `135`, actions `3` |
| `env_type` | `not_board_games` | `not_board_games` | `not_board_games` | `not_board_games` | `board_games` | `not_board_games` |
| `action_type` | default fixed | default fixed | default fixed | default fixed | `varied_action_space` | explicit `fixed_action_space` |
| `to_play` | `-1` | `-1` | `-1` | `-1` | self-play: player id; bot/eval: `-1` | `-1` |
| `random_collect_episode_num` | default `0` | `0` | `0` | default `0` | default `0` | `0` |
| Epsilon | default disabled | zoo config default disabled; agent Pong disabled, Breakout enabled | default disabled | default disabled | default disabled | disabled |
| Updates | `update_per_collect=100` | `update_per_collect=None`, `replay_ratio=0.25` | same | `update_per_collect=200` | `50` | current serious attempts used `1` |
| `game_segment_length` | `50` | `400` | `20` | `200` | terminal horizon-ish, e.g. `5/9` TicTacToe | `50` current default |
| `num_unroll_steps` | default `5` | default `5` | local var says `5`; default applies | default `5` | `3` TicTacToe | default `5` |
| `td_steps` | default `5` | default `5` | explicit `5` | default `5` | terminal horizon, e.g. `9` | default `5` |
| Supports | default `(-300, 301, 1)` | default `(-300, 301, 1)` | default `(-300, 301, 1)` | default `(-300, 301, 1)` | often `(-10, 11, 1)` or wider Connect4 | currently inherited default unless patched |
| Eval | `n_evaluator_episode=evaluator_env_num`, `eval_freq=100` | eval every `2e3`/`5e3` | eval every `5e3` | eval every `1e3` | eval every `2e3` | patched `eval_freq=1` in smoke surface |

Important source details:

- LightZero custom env docs require a dict observation with `observation`,
  `action_mask`, and `to_play`; non-board envs should use `to_play=-1` and
  all-ones masks for discrete actions.
- Our adapter already matches that non-board contract: fixed three-action
  space, all legal actions, `to_play=-1`, and reward space `[-1, 1]`.
- LightZero's default MuZero config says `update_per_collect=None` computes
  updates from collected transitions times `replay_ratio`; otherwise it uses
  the explicit `update_per_collect`.

## Why Not Pure Atari Segment

Atari segment is tempting because the name is Pong and the collector is
designed for long-running games. But official Atari segment assumes image
observations, `num_segments`, `game_segment_length=20`, delayed training start,
augmentation, optional buffer reanalysis, and auto replay-ratio updates. That
is a lot to change at once.

For dummy Pong, the biggest known failure is not that episodes are too long to
collect. It is that final checkpoints collapse under independent MCTS eval
after under-updated, sparse-reward training. The least confusing next move is:
keep the episode collector, fix the update/reward-support/horizon surfaces, and
only then try segment collection if collection length becomes the bottleneck.

## Small Config Recipe

Recommended next serious run, keeping the current custom env:

```python
env=dict(
    collector_env_num=4,
    evaluator_env_num=3,
    n_evaluator_episode=3,
)
policy=dict(
    cuda=False,
    env_type="not_board_games",
    action_type="fixed_action_space",
    model=dict(
        model_type="mlp",
        observation_shape=10,          # or 135 for raster_flat, but do not mix in same run
        action_space_size=3,
        self_supervised_learning_loss=True,
        discrete_action_encoding_type="one_hot",
        norm_type="BN",
        reward_support_range=(-5., 6., 1.),
        value_support_range=(-5., 6., 1.),
    ),
    collector_env_num=4,              # 8 if Modal wall time is acceptable
    evaluator_env_num=3,
    n_episode=4,                      # match collector_env_num
    num_simulations=25,               # CartPole scale first; 50 later
    batch_size=128,
    update_per_collect=None,
    replay_ratio=0.25,
    game_segment_length=120,          # match explicit Pong horizon for sparse terminal reward
    td_steps=10,                      # larger than official non-board default, still not board-game full horizon
    num_unroll_steps=5,
    random_collect_episode_num=0,
    eps=dict(
        eps_greedy_exploration_in_collect=True,
        type="linear",
        start=0.20,
        end=0.02,
        decay=20_000,
    ),
    fixed_temperature_value=0.5,
    eval_freq=100,
)
```

Run-level knobs:

- decouple `max_env_step` from `pong_episode_max_steps`;
- set `pong_episode_max_steps=120` or another explicit horizon, then keep it
  fixed while scaling `max_env_step`;
- record action histograms for collector, evaluator, and independent scorecard;
- keep independent MCTS scorecards as the quality gate.

If this recipe is too heavy for a first serious run, the safe smaller variant is
`collector_env_num=2`, `n_episode=2`, `batch_size=64`, `num_simulations=10`, but
keep `update_per_collect=None`, explicit support ranges, explicit horizon, and
the action histogram checks.

## Risks

- `update_per_collect=None` may create many learner updates per collect. Watch
  wall time and learner iteration count; if it gets silly, pin
  `update_per_collect=50` before changing anything else.
- Narrow supports are scale-correct for raw dummy Pong rewards, but they are a
  deliberate deviation from CartPole/Atari defaults. If any shaped reward is
  later moved into the actual env reward, revisit the support range.
- Epsilon exploration is not copied from the zoo Atari Pong config; it is a
  targeted response to observed action collapse. Treat it as an audited
  intervention, not proof that the official examples require epsilon.
- Longer `td_steps` helps sparse terminal credit assignment but increases
  bootstrap dependence. Compare `td_steps=5` vs `10` if the run improves
  survival but remains value-noisy.
- The board-game pattern is not otherwise appropriate yet. We should not set
  `env_type="board_games"` or `action_type="varied_action_space"` unless the env
  truly exposes alternating player turns and changing legal moves to LightZero.
- Segment collection remains a follow-up, not rejected forever. Try it after an
  episode-collector run shows sensible action coverage but collection throughput
  or long-horizon replay becomes the limiting factor.

## Bottom Line

Closest official pattern: CartPole/LunarLander custom-env MLP with the episode
collector. Best next pattern: a small hybrid that keeps that base, borrows
Atari's replay-ratio update accounting, and borrows board-game-scale supports.
Atari segment should wait until the simpler trainer has had one fair, well
instrumented run.

## Redirect: Closest Non-CartPole Sparse Sanity

If CartPole is excluded, the cheapest official sanity lane to replicate on
Modal is TicTacToe MuZero `play_with_bot_mode`, not Atari segment.

Reason:

- TicTacToe bot mode is a sparse/delayed reward game: most steps are zero, and
  the outcome reward arrives at the end.
- It matches our current dummy Pong abstraction better than self-play: one
  learner move is followed by an environment-owned opponent/bot move, and
  LightZero sees `to_play=-1` in bot/eval mode.
- It is tiny: 3x3 board, action space 9, small conv model, supports
  `(-10, 11, 1)`, `game_segment_length=5`, `td_steps=9`, `num_unroll_steps=3`,
  `discount_factor=1`.
- LightZero has a test-sized official config at
  `/tmp/lightzero-src/lzero/mcts/tests/config/tictactoe_muzero_bot_mode_config_for_test.py`
  with `num_simulations=5`, `update_per_collect=10`, and `batch_size=4`.

Recommended Modal smoke shape:

```text
official_example: zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config
patch_source: copy the tiny values from lzero/mcts/tests/config/tictactoe_muzero_bot_mode_config_for_test.py
entry: lzero.entry.train_muzero
mode: progression
seed: 0
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
env.battle_mode: play_with_bot_mode
env.bot_action_type: v0
policy.env_type: board_games
policy.action_type: varied_action_space
policy.game_segment_length: 5
policy.td_steps: 9
policy.num_unroll_steps: 3
policy.discount_factor: 1
policy.model.reward_support_range: (-10., 11., 1.)
policy.model.value_support_range: (-10., 11., 1.)
```

Command shape, assuming a CartPole-smoke-style Modal wrapper named
`curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke`:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_tictactoe_tiny_train_smoke \
  --mode progression \
  --source-module zoo.board_games.tictactoe.config.tictactoe_muzero_bot_mode_config \
  --seed 0 \
  --max-env-step 64 \
  --max-train-iter 4 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-episode 1 \
  --num-simulations 5 \
  --batch-size 4 \
  --update-per-collect 10 \
  --eval-freq 1 \
  --run-id official-tictactoe-bot-sparse-sanity-20260509
```

Connect4 is the second-choice official sparse sanity. It is still reasonable,
but it starts from a larger board and the stock config uses `num_simulations=50`
and `batch_size=256`; patching it down would be less directly official than
TicTacToe's existing test-sized config. Gomoku is third: even its reduced board
config uses 32 collectors and a 36-action board, so it is useful after
TicTacToe, not before. Atari segment is last for this specific sanity check:
it is official Pong, but it pulls in ALE/image wrappers/conv training and tests
long image-game segment mechanics more than sparse delayed reward mechanics.
