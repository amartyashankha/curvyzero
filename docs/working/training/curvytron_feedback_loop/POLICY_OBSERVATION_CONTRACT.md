# Policy Observation Contract

Last updated: 2026-05-16.

This contract applies to both training and tournament evaluation.

## Plain-English Rule

The policy should always see a controlled-player view:

- the player controlled by that policy is rendered as `self`;
- the other player is rendered as `other`;
- physical seat/player id may change, but the visual convention does not.

This is different from fixed-player-zero observation. We may randomize whether
the learner controls physical player 0 or player 1, but the policy view should
still use controlled-player `self`/`other` palette semantics on the global
board.

Uniform perspective does not mean camera-centering, mirroring, rotation, or
translation. The current policy tensor keeps the global board coordinates and
changes the player-owned encoding so the controlled player is `self`.

## Current Training Behavior

- Default learner physical role: `random_per_episode`.
- On each reset, training chooses whether the learner controls player 0 or
  player 1.
- `_lightzero_observation()` returns the stack for `ego_player_index`.
- `_update_stack()` renders that stack with a player-perspective palette for
  `ego_player_index`.
- Frozen opponent policies receive a batched observation where each physical
  player slot contains that slot's controlled-player view.

## Current Tournament Behavior

- Tournament pair scheduling may randomize which checkpoint controls physical
  seat 0 or seat 1.
- During a game, policy at physical seat `N` receives `observation[0, N]` and
  controls physical player `N`.
- `SourceStateGray64Stack4` renders `observation[0, N]` as player `N`'s
  controlled-player view.

## Required Invariants

- Policy perspective must be `controlled_player_view` in training metadata,
  checkpoint metadata, tournament game summaries, and render contracts.
- Training may randomize physical learner seat, but must not expose a raw
  player-zero/player-one color convention to the single policy.
- Tournament may randomize physical seating, but each checkpoint must receive
  the controlled-player view for the seat it controls.
- Checkpoint loading must fail closed when policy observation surface/backend
  metadata is missing, internally contradictory, or incompatible.
- Tests must cover both fixed physical seats and randomized physical seats.

## Proven Locally

- Same source state, player 0 view and player 1 view differ by player-owned
  `self`/`other` encoding, not by a spatial transform.
- Neutral bonus rendering is seat-invariant for the active gray64 policy tensor.
- The batched two-seat renderer matches the direct controlled-player renderer
  for both seats.
- Trainer fixed-player-0, fixed-player-1, and random-per-episode resets hand
  LightZero the selected controlled-player stack.
- Tournament game execution passes seat `N` the `observation[0, N]` stack and
  writes the controlled-player perspective contract into game summaries.
- Tournament checkpoint loading now requires compatible trail mode, bonus mode,
  backend, contract id, and perspective schema, and rejects contradictory nested
  `observation_contract` metadata.
- Trainer frozen-opponent provider loading now validates the same policy
  observation sidecar/metadata contract before loading the checkpoint as an
  opponent.
- Trainer step telemetry records `learner_seat_mode`, learner/opponent player
  indices and ids, and controlled-player observation perspective fields.
- Training-candidate refresh validates policy-observation metadata before
  copying a leaderboard checkpoint into a trainer assignment, then writes a
  clean sidecar next to the copied control-volume checkpoint.

## Remaining Gaps

- `raw_observation(player_perspective=True)` is a raw RGB artifact accessor and
  currently does not prove the model-facing policy tensor. Treat it as outside
  this contract until fixed or renamed.
- Blank-canvas metadata still has stale player-id wording in some places; the
  behavior is seat-relative, but docs/telemetry should stop implying a fixed
  player.

## Why This Matters

If training sees controlled-player `self`/`other` semantics but tournament eval
sees raw player colors, tournament rank is invalid. If training sees different
raw conventions for player 0 and player 1, the policy is learning two avoidable
visual tasks. The intended contract is role randomization with one stable
model-facing visual language.
