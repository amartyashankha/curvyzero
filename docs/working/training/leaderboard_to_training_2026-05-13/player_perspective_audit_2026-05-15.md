# Player Perspective Audit - 2026-05-15

## Bottom line

Current implementation has moved past the original finding. Normal fresh Coach
training still uses the `source_state_fixed_opponent` LightZero lane, but the
learner seat is now controlled by `learner_seat_mode`. Fresh manifests default
to `random_per_episode`; `fixed_player_0` and `fixed_player_1` are explicit
diagnostics only. Old `ego_player_index` config is rejected.

Historical finding that caused the change:
the earlier real18/v2 lane effectively trained the learner as seat 0 /
`player_0` only. That invalidated it as final evidence and is the reason the
restart gate now requires random-seat tests.

The current env selects `ego_player_index` internally from
`learner_seat_mode`, sets `opponent_player_index = 1 - ego_player_index`, and
records the selected seat in reset/step metadata.

The shared wording now lives in
`policy_observation_perspective_contract_2026-05-15.md`: Coach/training owns
the learner-seat choice, and the observation backend emits the selected
controlled-player view.

Modal normal training now passes `learner_seat_mode` through the trainer config
and tonight18 manifest rows. Poller/eval does not inherit this trainer field;
tournament eval has its own balanced-seat contract.

## Perspective normalization

For the normal single-ego lane, learner observations are controlled-player
views. The model stack is updated in `_update_stack`; the renderer creates
player 0 and player 1 views, copies `raw_frames[self.ego_player_index]` into the
learner stack, and copies `raw_frames[self.opponent_player_index]` into the
frozen-opponent stack. This path must stay seat-aware in tests; shape `[4,64,64]`
alone cannot prove the right seat was selected.

The public metadata now names the controlled-player perspective schema for the
model observation. The raw render path still ignores
`raw_observation(player_perspective=True)` and returns the same RGB frame
(lines 1059-1074); tests assert this equality at
`tests/test_curvyzero_source_state_visual_survival_lightzero_env.py` lines
258-285 and 398-415.

There is a separate two-seat stack that is correctly player-perspective:
`SourceStateGray64Stack4` keeps `[B, P, 4, 64, 64]` observations and renders a
last frame for each player using
`render_source_state_canvas_gray64_player_perspectives` or the fast direct
player-perspective renderer (lines 347-535 of
`src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`). Tests
verify player 0 and player 1 get different self/other frames in
`tests/test_curvytron_two_seat_render_mode.py` lines 112-164 and 166-235.
That path is not the normal real18/v2 `train_muzero` lane: its result payload
explicitly says it does not call LightZero `train_muzero`, does not use the
LightZero collector, and `true_lightzero_current_policy_self_play_training` is
false (lines 1664-1688 of
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`).

## Seat-1 Training Coverage Today

Normal fresh Coach lane: yes, through `learner_seat_mode=random_per_episode`.
The fixed-opponent lane remains single-agent LightZero, not two-seat self-play:
one learner observation, one learner action, one scalar reward. The seat is
environment state selected at reset and recorded in metadata.

Two-seat smoke/custom lane: yes, in a limited local lane. It can collect rows
for both player ids when both seats are controlled by the current policy, and
with a frozen opponent it can make the current policy control the other seat.
`DEFAULT_FROZEN_OPPONENT_PLAYER_ID = 1`, and the configured value is validated
as 0 or 1 (lines 133 and 301-303 of
`src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`). The
current-policy live mask removes only the frozen slot, so if
`frozen_opponent_player_id=0`, current policy rows would be seat 1
(lines 1413-1421). Replay rows carry `player_id`, `to_play`, and
`learner_controlled=True` (lines 2387-2416). Tests cover replay stratification
containing both `player_0` and `player_1` rows and terminal reward targets for
both seats at `tests/test_curvytron_two_seat_render_mode.py` lines 1163-1218
and 1237-1309. The Modal two-seat payload forwards
`frozen_opponent_player_id`; current tests show the default/frozen case with
player 1 as frozen at `tests/test_curvytron_live_checkpoint_eval_plumbing.py`
lines 3894 and 3921.

## Implemented Shape For Normal Training

Seat assignment lives inside
`CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv` while preserving the
single-agent LightZero contract:

1. Accept `learner_seat_mode` values `fixed_player_0`, `fixed_player_1`, and
   `random_per_episode`. Reject old `ego_player_index` config. Keep the
   LightZero observation shape `[4,64,64]`, action space `3`, scalar reward,
   and `to_play=-1`.
2. On every reset choose `self.ego_player_index` and set
   `self.opponent_player_index = 1 - self.ego_player_index`; update
   `ego_player_id`/`opponent_player_id` before building reset info.
3. Build `joint_action` by slot, not by literal order:
   start with a `[1,2]` array and assign the learner action into
   `joint_action[0, ego]` and opponent action into `joint_action[0, opponent]`.
4. Make the learner stack explicitly player-perspective for the chosen ego.
   Reuse the same renderer contract as `SourceStateGray64Stack4`: for two
   players, render both perspectives and copy `raw_frames[ego]` into the model
   stack and `raw_frames[opponent]` into `_opponent_stack`. Do not rely on the
   current seat-0/global-luma coincidence.
5. Keep frozen checkpoint opponents as metadata-only providers. Pass
   `opponent_mask` for the actual opponent slot and pass
   `observation[0, opponent]` as that opponent's own player-perspective stack.
6. Add explicit telemetry fields:
   `learner_seat_assignment_schema_id`, `ego_player_index`,
   `opponent_player_index`, `source_state_player_perspective=True`,
   `acting_player_id`, `controlled_player_id`, and `reward_player_id`.

This keeps LightZero clean: the learner still sees one observation, chooses one
action id, and receives one scalar reward. Seat assignment stays environment
metadata and reset-time state, not a change to LightZero's policy API.

## Break risks

- Replay/telemetry consumers may assume all learner rows are `player_0`.
  Current step info and telemetry use names like `requested_ego_action`,
  `executed_ego_action`, and `fixed_opponent_action`; downstream summaries count
  `acting_player_id` but the fixed-opponent lane has only emitted player 0 so
  far (Modal action observability around lines 8930, 9033, and 9453-9454).
- Frozen checkpoint opponents are currently tested with
  `opponent_mask == [[False, True]]` and action selected for player 1
  (`tests/test_curvyzero_source_state_visual_survival_lightzero_env.py` lines
  674-693). Seat randomization must also support `[[True, False]]`.
- The frozen provider gets only a `[4,64,64]` observation and a `player_id`
  (lines 63-99 of `lightzero_checkpoint_opponent_provider.py`); action ids are
  local left/straight/right and should not be flipped by seat, but the provider
  must receive the opponent's own self/other perspective.
- `blank_canvas_noop` currently means "player 1 is inert/hidden": death
  immunity, disabled mask, scrubbing, and render hiding all use
  `self.opponent_player_index`, which is good if that field becomes dynamic,
  but the documented claim strings literally say `player_1` (for example
  `curvyzero_source_state_visual_survival_lightzero_env.py` lines 1114-1162 and
  2091-2118). Those claims must become seat-relative.
- Reward assignment must follow the sampled ego seat. The reward code already
  takes `player_index`, but all normal-lane evidence has only exercised
  `player_0`; bonus pickup and terminal outcome maps need seat-1 tests.
- Raw render semantics will be confusing unless named carefully:
  `raw_observation(player_perspective=True)` currently returns the same world
  RGB frame. Either leave raw RGB as world/human view and expose only model
  tensors as player-perspective, or introduce a new named raw perspective
  accessor and update tests/metadata.

## Regression Tests

Required tests:

- old `ego_player_index` config rejects with a clear message;
- `learner_seat_mode=fixed_player_1` reset succeeds;
- for `fixed_player_1`, one step with action `0` emits
  `joint_action == {"player_0": 1, "player_1": 0}` for fixed-straight
  opponent, action mask comes from player 1, and `reward_player_id` is
  `"player_1"`.
- A deterministic random-seat reset mode alternates/samples both seats under a
  fixed seed and records the chosen seat in reset and step info.
- An asymmetric visual state proves learner obs for seat 1 equals the
  controlled-player-1 renderer output and differs from seat 0.
- Frozen opponent with learner seat 1 calls the provider with
  `opponent_mask == [[True, False]]`, sends the player-0 perspective stack, and
  maps the provider action to `joint_action["player_0"]`.
- `blank_canvas_noop` with learner seat 1 hides/scrubs/disables player 0, not
  player 1, and learner wall death/reward still works.
- Sparse, survival+bonus, and terminal-outcome reward variants are tested with
  `learner_seat_mode=fixed_player_1`, including same-step bonus pickup and
  final reward maps.
- Modal config/readiness tests expose and preserve the new seat mode fields,
  and action-observability summaries accept both `acting_player_id` values.

Recommendation: do not relaunch real training until these focused tests pass
and the manifest default remains `random_per_episode`.

## Test patch

Changed test/docs files:

- `tests/test_source_state_visual_survival_learner_seat_regression.py`: focused
  planned-fix regression coverage for fixed learner seat 1, seat-index
  validation, seat-relative joint-action/reward/control metadata, frozen
  opponent provider routing for opponent player 0, blank-canvas no-op scrubbing
  of the actual opponent seat, and deterministic random learner-seat sampling
  under dynamic reset seeds.
- `docs/working/training/leaderboard_to_training_2026-05-13/player_perspective_audit_2026-05-15.md`:
  this test patch note.
