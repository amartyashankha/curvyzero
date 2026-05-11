# LightZero Pong Live Self-Play Feasibility - 2026-05-09

Role: bounded live/current-policy self-play feasibility audit. No pytest run. No
code changes made.

## Short Answer

Do not try to turn simultaneous Pong into LightZero board-game
`self_play_mode` yet.

The recommended path now is:

1. Keep the existing single-ego LightZero dummy Pong trainer as the stable
   learner lane.
2. Use the already-supported frozen LightZero checkpoint opponent mode as the
   next self-play bridge, with clear telemetry saying it is frozen-checkpoint
   self-play, not live current-policy self-play.
3. Put true current-policy two-paddle support behind a small joint-action
   collector/shim interface: the collector asks the current policy for one row
   per live paddle, maps those rows back into `joint_action`, steps Pong once,
   and stores one training row per ego perspective.

That keeps the env honest: Pong remains simultaneous, LightZero remains
single-action until we have a real joint-action boundary, and CurvyTron can
reuse the same row-mapping shape later.

## What LightZero Board-Game Self-Play Actually Does

LightZero board games such as Gomoku and TicTacToe model self-play as
alternating scalar decisions. In `self_play_mode`, the env returns
`obs["to_play"] = current_player`; the collector calls the same policy for one
action; the env applies one move, swaps `current_player`, and returns the next
`to_play`.

Relevant anchors:

- `/tmp/lightzero-src/zoo/board_games/gomoku/envs/gomoku_env.py`
  defines `battle_mode='self_play_mode'`, tracks `players = [1, 2]`, returns
  `to_play=current_player`, and swaps the player after `_player_step`.
- `/tmp/lightzero-src/zoo/board_games/tictactoe/envs/tictactoe_env.py` follows
  the same pattern.
- `/tmp/lightzero-src/lzero/worker/muzero_collector.py` keeps one
  `action_mask` and one `to_play` per env id, calls policy once per ready env,
  then appends one scalar action to the game segment.
- `/tmp/lightzero-src/lzero/worker/alphazero_collector.py` has reward shaping
  that uses alternating `to_play` to assign terminal value signs.

That pattern is clean for games where exactly one player acts at a state
transition. It is not a natural fit for Pong, where both paddles act before one
physics tick resolves.

## Current Curvy Dummy Pong Shape

`src/curvyzero/training/lightzero_dummy_pong_env.py` is intentionally
single-ego:

- LightZero supplies one action for `ego_agent`, default `player_0`.
- The wrapper asks an opponent policy for the other paddle.
- The wrapper builds `joint_action = {ego_agent: ego_action,
  opponent_agent: opponent_action}` and steps `PongEnv` once.
- It returns `to_play = -1`, so LightZero treats this as a non-board-game env.

The code already supports frozen checkpoint opponents:

- `opponent_policy = "lightzero_policy_head_checkpoint"`
- `opponent_policy = "lightzero_mcts_checkpoint"`
- checkpoint adapters live in
  `src/curvyzero/training/lightzero_dummy_pong_policy.py`

The separate NumPy replay path,
`src/curvyzero/training/dummy_pong_selfplay_replay.py`, already demonstrates
the desired data shape for simultaneous self-play: same behavior policy controls
both seats, each environment step emits one replay row per ego agent, and the
row records `policy_by_agent`, `joint_action`, rewards, and terminal returns.
That is useful scaffolding, but it is not LightZero MuZero collection.

## Option Assessment

### Alternating `to_play` wrapper

Not recommended now.

A fake alternating wrapper would have to collect paddle A's action, expose the
same physical state as a pending-action state for paddle B, then step the real
Pong tick only after paddle B acts. That creates artificial half-steps,
awkward reward assignment, duplicate observations, and a dynamics model that
learns wrapper bookkeeping instead of Pong physics. It also leans on
board-game reward conventions that assume one actor caused the transition.

This is the fastest way to make the stack clever in a bad way.

### Env wrapper asks one policy for both actions

Tempting, but do not put this inside `DummyPongLightZeroEnv`.

The env currently receives only a scalar learner action in `step(action)`. It
does not own the live LightZero collect-mode policy object, and it should not:
that would create a circular dependency between env and policy, make env
workers responsible for model state, and blur who owns exploration,
temperature, MCTS, `ready_env_id`, recurrent/cache reset, and device placement.

This is acceptable only for frozen checkpoint opponents because the checkpoint
adapter is deliberately a passive env-side opponent, loaded once and labelled
as frozen.

### Frozen checkpoint league

Recommended now, with honest labels.

This is already the least messy bridge from scripted opponents toward
self-play pressure. The learner remains one LightZero policy in the standard
collector. The opponent is an older checkpoint loaded through the existing
opponent-policy protocol. Runs can rotate seats, record checkpoint lineage, and
promote checkpoints only when scorecards beat scripted and older opponents.

It is not true live current-policy self-play. It is still the right immediate
training pressure while the CurvyTron reconstruction is not ready.

### Custom joint-action collector

Recommended later, as the true live path.

For true current-policy self-play, the policy/collector boundary needs to be:

```text
ready Pong states
-> build policy rows: one row per live paddle
-> current policy.forward(rows)
-> map selected row actions back to joint_action[env, player]
-> env.step(joint_action)
-> append one ego-perspective training row/segment per live paddle
```

This shape matches `src/curvyzero/training/policy_row_mapping.py`, which already
maps multiplayer arrays to compact policy rows and maps selected row actions
back to a joint action tensor. It also matches what CurvyTron will need more
than board-game `to_play` does.

## Exact Interface Gap

The missing interface is not "a different `to_play` value." The missing
interface is a joint-action collection boundary.

Current LightZero collector contract:

```text
obs_by_env[env_id] -> policy.forward(...) -> action_by_env[env_id]
env.step(action_by_env)
game_segment.append(action, next_obs, reward, action_mask, to_play)
```

Needed for simultaneous current-policy Pong:

```text
obs_by_env_player[env_id, player_id]
-> policy.forward(policy_rows)
-> action_by_env_player[env_id, player_id]
-> env.step(joint_action_by_env)
-> append ego row/segment for each player whose policy row was real
```

Concretely, there is currently no clean place where the standard
`MuZeroCollector` can ask the same live collect-mode policy for a second
observation from the same env tick, receive two actions, step once, and store
two ego-perspective transitions tied to the same physics transition. Without
that, any "live both paddles" implementation either hides policy inference in
the env or serializes a simultaneous game into fake turns.

## Recommended Path Now

Use frozen-checkpoint opponent training as the immediate next step, and keep
the future live path isolated behind a joint-action collector design.

For the next practical dummy Pong run, prefer:

```text
opponent_policy = lightzero_mcts_checkpoint
opponent_checkpoint_adapter = mcts_eval_mode
opponent_checkpoint_num_simulations = 2 or 4
ego_agent = player_0 for one run, then swapped/paired evaluation
```

Report it as:

```text
single live learner vs frozen LightZero checkpoint opponent
```

Do not call it live current-policy self-play.

## What Not To Do Yet

- Do not fake Pong as board-game `self_play_mode` with alternating `to_play`
  unless the explicit experiment is "serialized Pong wrapper," not real Pong.
- Do not let `DummyPongLightZeroEnv` reach back into the live learner policy to
  choose the opponent action.
- Do not build a full league service before one frozen-checkpoint opponent run
  has clean scorecards and lineage.
- Do not add an exported cheap policy format yet. Use the existing checkpoint
  adapters first; optimize only after env-step cost is the bottleneck.
- Do not generalize the LightZero dummy Pong env around CurvyTron assumptions
  until the joint-action row contract is settled.

## Implementation Ladder

1. **Now: frozen-checkpoint bridge.** Run or document one bounded learner-vs
   frozen-checkpoint Pong training lane using the existing env config fields.
   Require telemetry for learner checkpoint, opponent checkpoint, adapter,
   seat, seeds, action histograms, survival, wins/losses, and scorecard refs.

2. **Next: no-training joint-action spike.** Add a tiny collector-side prototype
   outside the standard LightZero trainer that uses one current policy adapter
   to act for both Pong seats, builds two policy rows per tick, maps back to
   `joint_action`, and writes replay/telemetry in the existing
   `dummy_pong_selfplay_replay.py` row style. This proves the boundary without
   disturbing the trainer.

3. **Later: real custom collector.** If the spike is clean, implement a
   LightZero-compatible joint-action collector or actor loop that owns the live
   policy, batching, MCTS/exploration kwargs, and per-ego segment storage. This
   is the first place to claim true current-policy self-play for simultaneous
   Pong, and it should be designed to reuse `policy_row_mapping.py` for
   CurvyTron.

## Smallest Useful Step Before CurvyTron Reconstruction

The smallest useful step is not full live self-play. It is a bounded
frozen-checkpoint opponent run plus a written joint-action collector contract.

That gives us immediate learning pressure beyond scripted opponents while
keeping the live current-policy design clean enough to carry into CurvyTron.
