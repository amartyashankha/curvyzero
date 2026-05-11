# LightZero Source Setup Audit - 2026-05-09

Scope: source/research lane only. I inspected our dummy Pong LightZero adapter
and config patching, plus a local official LightZero clone at
`/tmp/lightzero-src` (`de740552`). No pytest and no code edits beyond this doc.

## Executive Ranking

Top suspects for the no-learning/no-down behavior, ranked by directness and
testability:

1. **Exploration bootstrap is too weak.**
   Our patched config inherits LightZero defaults:
   `random_collect_episode_num=0`,
   `eps.eps_greedy_exploration_in_collect=False`, and
   `fixed_temperature_value=0.25`. With only 2-8 MCTS simulations in many runs,
   collect action sampling is nearly deterministic from a sparse visit-count
   distribution. Official CartPole also leaves random/eps off, but it uses 25
   simulations, 8 collector envs, 8 episodes, and 100 updates per collect.
   We cut all of those at once.

2. **`n_episode`, `batch_size`, and `update_per_collect` are too small/noisy.**
   Official episode collector configs collect `n_episode=8` and train at much
   higher reuse (`update_per_collect=50` or `100`). Our default is one
   collector env, `n_episode=1`, `batch_size=8`, `update_per_collect=1`.
   For dummy Pong's short terminal episodes, this can train on one narrow
   trajectory distribution at a time and may reinforce the first action prior
   instead of discovering down.

3. **`game_segment_length` / horizon wiring is a real footgun.**
   The current patched config does not set `policy.game_segment_length`, so it
   keeps CartPole's `50` while env `max_steps` can be 64, 512, 1024, or 4096.
   Episode collector will still save final partial segments, so this is not an
   immediate crash bug. But it changes padding/replay boundaries and previously
   we already found horizon mismatches between training/eval. For dummy Pong,
   make `game_segment_length` explicit and aligned to the intended horizon or a
   deliberate short segment size.

4. **`train_muzero` vs `train_muzero_segment` is a plausible scaling issue, not
   the first bug.**
   Official CartPole and board-game examples use `train_muzero` with episode
   collection. Official Atari Pong uses `train_muzero_segment`, `num_segments=8`,
   `game_segment_length=20`, auto updates via `replay_ratio=0.25`, and delayed
   training after env steps. Our dummy env is closer to CartPole/control than
   Atari pixels, so `train_muzero` is defensible. But if episodes are long or
   we want fixed chunk collection, a segment run is the clean A/B after fixing
   exploration/data volume.

5. **`to_play`, `env_type`, and `battle_mode` look mostly correct for the
   current wrapper.**
   Official Connect4/TicTacToe bot modes set `env_type='board_games'`,
   `action_type='varied_action_space'`, `battle_mode='play_with_bot_mode'`,
   and force `to_play=-1` because the opponent is folded into the env step.
   Our wrapper is also ego-vs-scripted-opponent, so `to_play=-1` and
   `env_type='not_board_games'` are coherent. Do not flip to board-game mode
   unless we expose true alternating self-play with legal-action masks and
   player-perspective rewards.

6. **`action_type=fixed_action_space` is low suspicion for a 3-action Pong env.**
   MuZero's default is `fixed_action_space`; official board games use
   `varied_action_space` because legal moves shrink with board occupancy.
   Dummy Pong always has exactly three legal actions and an all-ones mask, so
   fixed action space is the right contract. Changing this is unlikely to fix
   no-down and could complicate target-policy handling.

## Source Comparisons

Our patched dummy Pong config starts from official CartPole:

- Official CartPole:
  `/tmp/lightzero-src/zoo/classic_control/cartpole/config/cartpole_muzero_config.py`
  uses `collector_env_num=8`, `n_episode=8`, `num_simulations=25`,
  `update_per_collect=100`, `batch_size=256`, `game_segment_length=50`,
  `env_type='not_board_games'`, and `train_muzero`.
- Our patch:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
  changes env/model/action size and caps collection, but does not set
  `random_collect_episode_num`, `eps`, `fixed_temperature_value`,
  `game_segment_length`, `td_steps`, `num_unroll_steps`, reward/value supports,
  or `action_type`.

Official Atari Pong is materially different:

- `/tmp/lightzero-src/zoo/atari/config/atari_muzero_segment_config.py`
  uses `train_muzero_segment`, `num_segments=8`, `game_segment_length=20`,
  `random_collect_episode_num=0`, `replay_ratio=0.25`,
  `update_per_collect=None`, `num_unroll_steps=5`, `td_steps=5`,
  `train_start_after_envsteps=2000`, `num_simulations=50`, and a conv model.
- This is a chunked replay setup, not just CartPole with a Pong env id.

Official board-game bot/self-play examples are not the right target contract:

- TicTacToe and Connect4 MuZero bot/self-play configs set
  `env_type='board_games'`, `action_type='varied_action_space'`,
  large terminal-outcome `td_steps`, and `discount_factor=1`.
- In official Connect4 `play_with_bot_mode`, the env explicitly sets
  `to_play=-1` after the bot step because alternation is hidden from MCTS.
  That supports our `to_play=-1` choice for ego-vs-scripted dummy Pong.

LightZero entry/collector details that matter:

- `train_muzero` uses `MuZeroCollector`, which collects complete episodes based
  on `policy.n_episode`.
- `train_muzero_segment` uses `MuZeroSegmentCollector`, which returns after
  `policy.num_segments` segments and requires `num_segments == env_num`.
- Collection temperature comes from `visit_count_temperature`; when manual
  decay is false, it is the fixed value, default `0.25`.
- If eps-greedy collection is enabled, LightZero first takes deterministic
  visit argmax and then randomly overrides with epsilon. If disabled, it samples
  from visit counts with the collect temperature.

## Concrete Recommendations

Run only two discriminating next runs before broad scaling:

1. **Exploration/data-volume episode run.**
   Keep `train_muzero`, fixed action space, `env_type='not_board_games'`,
   `to_play=-1`, and the same opponent. Patch/run with:
   `collector_env_num=2`, `evaluator_env_num=1`, `n_episode=8`,
   `batch_size=32`, `update_per_collect=8`, `num_simulations=16`,
   `random_collect_episode_num=8`,
   `eps.eps_greedy_exploration_in_collect=True`,
   `eps.start=0.5`, `eps.end=0.05`, `eps.decay=2000`,
   `fixed_temperature_value=1.0` for early collect, and explicit
   `game_segment_length=max_steps` or `min(max_steps, 50)` with that choice
   recorded. Score the final checkpoint against the same random opponent and
   check whether action `2` appears in collector telemetry and independent
   MCTS.

2. **Segment-collector A/B only if run 1 still has zero down.**
   Keep the same exploration knobs and model, but use `train_muzero_segment`
   with `num_segments=collector_env_num`, `game_segment_length=20` or `50`,
   `update_per_collect=None`, and `replay_ratio=0.25`, mirroring Atari's
   collection geometry without switching to Atari's conv/pixel assumptions.
   This isolates whether complete-episode collection plus tiny short episodes
   is starving replay diversity.

Do not spend the next run on `varied_action_space`, board-game mode, or
`to_play=1/2`. Those are useful only as diagnostics after the two runs above,
because official LightZero source says fixed/not-board/`to_play=-1` is a
coherent contract for a single-agent discrete env with the opponent hidden
inside `step()`.

## Config Hygiene To Add Before The Runs

- Surface and record these fields in `patched_surface`: `action_type`,
  `game_segment_length`, `td_steps`, `num_unroll_steps`,
  `random_collect_episode_num`, `eps`, `fixed_temperature_value`,
  `reward_support_range`, and `value_support_range`.
- Assert `n_episode >= collector_env_num` for `train_muzero`.
- Assert `num_segments == collector_env_num` for any `train_muzero_segment`
  path.
- Assert independent eval `PongConfig.max_steps` equals the checkpoint's train
  `max_env_step`.
- Keep reward/value supports broad enough for LightZero defaults, but consider
  a later support-range ablation only after exploration/data-volume is fixed;
  the default `(-300, 301, 1)` is oversized for `[-1, 1]` terminal Pong reward,
  but it is not the top no-down suspect.
