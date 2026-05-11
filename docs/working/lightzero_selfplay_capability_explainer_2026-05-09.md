# LightZero Self-Play Capability Explainer - 2026-05-09

Ownership: this note only.

## Short Answer

Yes, LightZero/MuZero supports self-play.

But that sentence hides three different meanings:

1. For single-agent environments like Atari or CartPole, "self-play" usually
   means the current policy plays the environment, writes its own replay, and
   learns from it. There is no opponent.
2. For turn-based board games like TicTacToe, Gomoku, and Connect4, LightZero
   has real two-sided self-play. One move is chosen at a time, the environment
   tracks whose turn it is, and the same learner policy can play both sides.
3. For simultaneous multiplayer games like CurvyTron, stock LightZero does not
   naturally expose "all players choose actions at once" as a native
   multi-agent training shape. A wrapper or custom collector is needed.

So the plain answer for CurvyTron is:

```text
MuZero as an idea can be used for CurvyTron self-play.
LightZero as a stock trainer is not a native CurvyTron simultaneous self-play engine.
For first use with LightZero, a single-ego wrapper is the clean path.
```

## What LightZero Means By Environment Shape

LightZero's custom-env docs say every environment observation should include:

```text
observation
action_mask
to_play
```

For non-board-game environments, `to_play` is `-1`. That is the single-player
case: one policy action goes into one environment step.

For board games, `to_play` is a player id. That lets MCTS know whose turn it is.
This is useful for alternating games like Gomoku, where exactly one player moves
at a time.

This distinction matters. `to_play` is not the same thing as a simultaneous
multi-agent action dictionary.

## What Board-Game Self-Play Does

In LightZero Gomoku self-play mode, the environment works like this:

```text
state says player 1 to play
policy chooses one move
env places player 1's stone
env switches current player to player 2

state says player 2 to play
policy chooses one move
env places player 2's stone
env switches current player to player 1
```

That is real self-play because both sides are driven by the learning system over
the course of the same game. It is also turn-based. The game never needs both
players' actions for the same tick.

LightZero also has board-game bot modes. In bot mode the learner makes one move,
then the environment immediately asks a bot for the other move. That is useful,
but it is not the same as self-play between two learner-controlled sides.

## Why CurvyTron Is Different

CurvyTron is simultaneous. On each game tick, the real game wants a full joint
action:

```text
player_0 action
player_1 action
...
player_N action
-> one physics/collision/scoring update
```

It is wrong to pretend this is a turn-based board game:

```text
player_0 acts -> fake partial update -> player_1 acts -> fake partial update
```

That changes timing. In CurvyTron, movement, trail creation, wall collision,
body collision, deaths, and scoring are all tied to the shared tick. Local
source notes also record that avatar update order can affect collision evidence,
so hidden half-turns are especially dangerous.

The clean CurvyTron step shape is:

```text
collect every live player's intended action
apply all actions to the same real tick
record the full joint action and terminal evidence
```

OpenSpiel's API is a useful contrast here: it explicitly distinguishes
`apply_action(action)` for turn-based games from `apply_actions(actions)` for
simultaneous-move games. That is the shape CurvyTron semantically wants.

## Does Stock LightZero Handle That Directly?

Not in the normal stock path.

The LightZero MuZero collector batches many environment rows, but each ready
environment row produces one selected action and then calls the environment
manager's `step(actions)` with one action per env row. That is parallel
single-agent or alternating-player collection, not a native `[env, player]`
simultaneous multi-agent loop.

LightZero board-game support handles multiple players through `to_play`: one
current player acts, then the env changes the player. That fits Gomoku. It does
not fit CurvyTron ticks, because CurvyTron needs all player actions before the
tick resolves.

Could we force it?

Yes, but each option has a cost:

- Encode the whole joint action as one giant action. For `P` players and `A`
  actions each, the action count becomes `A ** P`. This gets ugly fast.
- Write a custom LightZero collector/policy path that asks for every live
  player's action before stepping the env. That is possible engineering, but it
  is no longer "stock LightZero handles it directly."
- Use a single-ego wrapper. This keeps stock LightZero mostly intact and lets
  the wrapper provide the missing opponent actions.

## Is A Single-Ego Wrapper Necessary?

For stock LightZero plus CurvyTron: practically, yes.

It is not a deep theoretical requirement of MuZero. It is an integration choice.
MuZero can be reformulated for simultaneous games, and other frameworks can
represent simultaneous joint actions directly. But if we want to use LightZero's
existing trainer without forking its collection loop, the single-ego wrapper is
the honest shape.

The wrapper works like this:

```text
LightZero chooses ego action
wrapper chooses opponent action or actions
real CurvyTron env steps once with the full joint action
wrapper returns ego observation, ego reward, done, and audit info
```

This is not cheating. It is like training against a named computer opponent.
The important rule is that the opponent policy must be explicit:

```text
random_uniform
wall_avoidance_v0
frozen_checkpoint:<id>
current_policy_snapshot:<train_iter>
```

The replay and scorecards must record the full joint action, opponent id,
opponent checkpoint, ego seat, reset seed, terminal observation, and winner.
The learner may only see the ego observation, but the audit trail must preserve
what every player actually did.

## What Counts As Self-Play For CurvyTron?

Use precise labels:

| Label | Meaning | Fair to call self-play? |
| --- | --- | --- |
| `ego_vs_random` | LightZero controls one seat; random policy controls others. | No. It is learner-vs-random. |
| `ego_vs_scripted` | LightZero controls one seat; heuristic controls others. | No. It is learner-vs-scripted. |
| `ego_vs_frozen_checkpoint` | Current learner plays against an older frozen policy. | Yes, league/checkpoint self-play style, but say "frozen checkpoint." |
| `ego_vs_current_snapshot` | Opponent uses a snapshot of the same learner. | Yes, shared-policy self-play style, but log the snapshot rule. |
| `all_players_policy_only` | Same policy chooses every player's action before each tick. | Yes, direct simultaneous policy self-play. Not stock LightZero unless wrapped/customized. |
| `all_players_with_mcts` | Search or policy improvement is run for every player before each tick. | Yes, but much harder; likely custom, not v0 LightZero. |

For v0, the safest LightZero path is:

```text
single ego -> scripted opponents -> frozen checkpoint opponents -> current-policy snapshot opponents
```

Do not call the first two "self-play." They are useful gates before self-play.

## For What Kinds Of Games Is LightZero A Good Fit?

LightZero is strongest for:

- single-agent discrete-control environments, including Atari-style games;
- classic-control/Gym-like environments after wrapping into LightZero's
  observation/action-mask/to-play format;
- turn-based board games with legal action masks and clear current-player
  state;
- MuZero-family experiments where LightZero's standard replay, targets, model,
  and trainer are acceptable.

LightZero is weaker for:

- simultaneous multi-agent games with a real player axis;
- n-player general-sum games where every player needs its own value semantics;
- CurvyTron-style training where replay must preserve joint actions, final
  observations, seat identities, opponent identities, and trace hashes as first
  class data;
- custom search formulations where opponent actions are sampled, modeled, or
  searched separately from the ego action.

That does not make LightZero useless. It means LightZero should be treated as a
contained MuZero control lane, not as the semantic owner of CurvyTron's
multi-agent environment.

## Bottom Line

LightZero supports self-play for board games in the normal AlphaZero/MuZero
sense: one player moves, then the other player moves, and `to_play` tells the
search whose turn it is.

CurvyTron is not that. CurvyTron is simultaneous. The real environment needs a
joint action every tick.

Therefore:

```text
Use a single-ego wrapper if using stock LightZero.
Record opponent actions and identities honestly.
Call it self-play only when the opponents are policy checkpoints/current-policy snapshots, not random scripts.
Do native simultaneous all-player self-play only with a custom collector or a repo-owned training loop.
```

## Sources Checked

Local sources:

- `docs/working/lightzero_self_play_architecture_audit_2026-05-09.md`
- `docs/working/lightzero_single_ego_selfplay_design_2026-05-09.md`
- `docs/working/single_ego_wrapper_explanation_2026-05-09.md`
- `docs/research/lightzero_feature_fit_for_curvyzero.md`
- `docs/decisions/0003-multiplayer-selfplay-v0-formulation.md`
- `docs/research/multiplayer_selfplay_muzero.md`
- `docs/research/curvytron_source_map/facts_index.md`

Primary upstream sources:

- LightZero docs, custom environments:
  https://opendilab.github.io/LightZero/tutorials/envs/customize_envs.html
- LightZero docs, overview/API index:
  https://opendilab.github.io/LightZero/
- LightZero upstream Gomoku env:
  https://github.com/opendilab/LightZero/blob/main/zoo/board_games/gomoku/envs/gomoku_env.py
- LightZero upstream Gomoku MuZero self-play config:
  https://github.com/opendilab/LightZero/blob/main/zoo/board_games/gomoku/config/gomoku_muzero_sp_mode_config.py
- LightZero upstream MuZero collector:
  https://github.com/opendilab/LightZero/blob/main/lzero/worker/muzero_collector.py
- MuZero paper, Nature:
  https://www.nature.com/articles/s41586-020-03051-4
- OpenSpiel simultaneous action API:
  https://openspiel.readthedocs.io/en/latest/api_reference/state_apply_action.html
