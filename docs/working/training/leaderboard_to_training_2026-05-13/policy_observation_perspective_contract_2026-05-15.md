# Policy Observation Perspective Contract - 2026-05-15

Plain rule: Optimizer does not choose the player perspective. Coach/training
chooses which physical player a policy row controls. The renderer only emits
the requested controlled-player view for the current policy observation surface.

## Terms

- Physical player / seat: the real CurvyTron slot in the environment. Today
  this is player 0 or player 1.
- Controlled player / ego / learner: the physical player whose action the
  policy row will choose.
- Opponent player: the other physical player in the fixed-opponent lane.
- Policy observation surface: `browser_lines + simple_symbols`.
- Observation backend: `cpu_oracle` is current reliable training backend.
  Scalar `jax_gpu` is an experimental canary and currently too slow. Future
  batched GPU backend work must not change the observation contract.
- Controlled-player view: the controlled player is encoded as SELF and other
  players as OTHER before the policy sees the tensor.

## Training Contract

- Fresh real training uses `learner_seat_mode=random_per_episode`.
- `fixed_player_0` and `fixed_player_1` are diagnostics only.
- Old `ego_player_index` config is removed and should be rejected.
- At reset, the training env chooses `ego_player_index` from
  `learner_seat_mode` and sets `opponent_player_index = 1 - ego_player_index`.
- The learner observation is one `float32 [4,64,64]` stack for the chosen
  `ego_player_index`.
- The frozen opponent, if present, must receive its own opponent-player view.
- Joint action, action mask, reward, terminal info, and telemetry must all use
  the same selected physical player.
- LightZero still sees the normal single-agent shape: one observation, one
  action id, one scalar reward, and `to_play=-1`.

## Tournament Contract

- Tournament seating is separate from trainer `learner_seat_mode`.
- Tournament games use balanced/random physical seating for ratings.
- Seat 0 receives `observation[0,0]` and controls player 0.
- Seat 1 receives `observation[0,1]` and controls player 1.
- Do not feed both policies the player-0 view in tournament; that would mix the
  wrong visual owner with the seat-1 actuator.

## Optimizer Contract

- The policy surface source of truth is
  `src/curvyzero/env/observation_surface_contract.py`.
- The perspective schema id is
  `curvyzero_policy_observation_controlled_player_perspective/v1`.
- Optimizer may replace the backend with GPU/JAX, cache the render, or improve
  speed, but the output must stay the same controlled-player view.
- GPU parity tests must check both player 0 and player 1 views.
- Any docs or metrics should say `controlled-player view`, not imply that
  Optimizer independently chooses a "player-perspective grayscale" policy.

## Metadata To Preserve

Every policy artifact should make these visible when practical:

- `policy_observation_contract_id`
- `player_perspective_schema_id` or
  `policy_observation_perspective_schema_id`
- `source_state_player_perspective=true`
- `controlled_player_id` / `ego_player_index` for training rows
- `policy_player` or seat id for tournament/multiplayer rows
- `policy_trail_render_mode=browser_lines`
- `policy_bonus_render_mode=simple_symbols`
- `policy_observation_backend`

## Current Evidence

- 2026-05-16 fork check: current source-state training is not fixed to player
  0. `DEFAULT_LEARNER_SEAT_MODE` is `random_per_episode`; each reset chooses
  player 0 or player 1 deterministically from the episode seed/reset index, the
  learner action is routed to that physical player, and reward/action mask/info
  use the same selected player.
- 2026-05-16 fork check: tournament eval uses `balanced_random` seating and
  per-seat controlled-player observations. Seat 0 gets `observation[0,0]` and
  acts as player 0; seat 1 gets `observation[0,1]` and acts as player 1.
- 2026-05-16 cleanup: actual tournament policy loading now fails closed when a
  checkpoint lacks explicit `policy_trail_render_mode`,
  `policy_bonus_render_mode`, or `policy_observation_backend` in checkpoint
  payload/sidecar/run metadata. Fresh eval must not silently assume the current
  surface for a checkpoint whose training surface is unknown.
- `player_perspective_audit_2026-05-15.md`: training-seat audit and fixed-seat
  regression plan.
- `random_seat_manifest_wiring_2026-05-15.md`: manifest contract for
  `learner_seat_mode=random_per_episode`.
- `tournament_eval_seat_perspective_audit_2026-05-15.md`: tournament per-seat
  observation audit.
- `render_path_decision_2026-05-15.md`: policy surface and GPU-backend target.
