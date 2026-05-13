# Controls Fidelity Audit - 2026-05-13

Status: audit note, not a completion claim.
Owner surface: Environment.

## Source Control Model

- Source CurvyTron stores real-time per-player control state, then advances
  server frames by elapsed milliseconds. It does not expose native trainer
  action ids, `step(joint_action)`, or simultaneous discrete decisions.
- Browser input reduces to source moves: left-only `-1`, right-only `1`,
  neither `0` on the wire, both/opposite keys `0` on the wire. The client emits
  `player:move`; the server immediately calls `avatar.updateAngularVelocity`.
- A control change affects the next `Game.update(step_ms)` after the server
  receives it. Deterministic traces may stage all planned move changes before an
  elapsed-ms source frame, but that staging is test harness language.
- Source movement turns first, then moves, using elapsed milliseconds. Source
  frames are about 16.67 ms from `BaseGame.framerate = (1 / 60) * 1000`.

Primary evidence:
`docs/research/curvytron_source_map/movement_controls.md`,
`third_party/curvytron-reference/src/client/model/PlayerInput.js`,
`third_party/curvytron-reference/src/client/controller/GameController.js`,
`third_party/curvytron-reference/src/server/controller/GameController.js`, and
`third_party/curvytron-reference/src/shared/model/BaseAvatar.js`.

## CurvyZero Wrapper Abstraction

- CurvyZero trainer action ids are wrapper ids: `0` left, `1` straight,
  `2` right. They map to native source moves `-1`, `0`, `1`.
- `VectorMultiplayerEnv.step(actions)` is a wrapper decision over `int[B,P]`
  player action ids. Its `joint_action`/`player_action` fields are replay and
  wrapper metadata, not native CurvyTron API facts.
- When `decision_source_frames` is set, the wrapper derives `decision_ms`,
  holds the selected native control value for that many source-sized internal
  frames, and stops early on death/overflow.
- When `decision_source_frames` is not set, `VectorMultiplayerEnv` can still
  step one larger `decision_ms` physics update. That path is labeled by
  `source_frame_decision=False` and should not be used as source-frame controls
  fidelity evidence.
- Current contract labels are mostly clear: `native_control_model_id`,
  `trainer_control_wrapper_id`, `source_frame_decision`,
  `decision_source_frames`, `source_physics_step_ms`, `source_moves`, and
  `native_control_value` are exposed in public info/sidecars. The risky names
  are `step` and `joint_action`; keep describing them as wrapper/replay terms.

Primary evidence:
`src/curvyzero/env/trainer_contract.py`,
`src/curvyzero/env/vector_multiplayer_env.py`,
`src/curvyzero/env/vector_runtime.py`, and
`docs/design/environment/training_interface_contract.md`.

## Currently Tested

- `tests/test_controls_source_input_fidelity.py` loads the original JS
  `PlayerInput.js`, client `GameController.onMove`, and server
  `GameController.js`. It now proves keyboard left/right reduction,
  release-to-wire-0, opposite-key neutralization, restore-on-opposite-release,
  no duplicate emits, and synchronous server delivery into
  `avatar.updateAngularVelocity` for `-1`, `0`, and `1`.
- Source movement scenarios compare source-shaped Python movement with JS oracle
  traces for straight, left, right, two-player turning, multistep, varied
  elapsed-ms, and movement event order.
- `CurvyTronSourceEnv.step(joint_actions, elapsed_ms)` is explicitly documented
  as wrapper language over source move factors, not native trainer action ids.
- `vector_runtime.advance_player_movement` tests source math: only live rows
  update, turn before move, elapsed-ms distance, and inverse sign flipping.
- `tests/test_controls_vector_fidelity.py` proves the 2P public runtime control
  matrix for every player and action id, held-control parity between one
  `decision_source_frames=N` step and `N` one-frame steps, release-to-straight,
  bad live action rejection, absent/dead inactive noops, and terminal-padding
  noop behavior under `decision_source_frames`.
- `tests/test_controls_multiplayer_vector_fidelity.py` extends the public
  runtime control proof to P=3 and P=4 for every player and every action id,
  plus a 4P held-control parity case.
- `tests/test_lightzero_source_state_wrapper_product_fidelity.py` proves the
  LightZero-facing source-state wrapper preserves scalar joint-action decoding,
  native control sidecars, held source frames, raw RGB -> gray64 stack, terminal
  final observation, rewards, masks, and terminal facts for one 2P trace.
- Other `VectorMultiplayerEnv` tests record `source_moves`, action sidecars,
  masks, and native control values in several 2P/3P/4P fixtures.
- A 2P public collision canary now proves `decision_source_frames` source-frame
  substeps prevent a large-decision tunneling case and report substeps/elapsed
  source physics time.
- `tests/test_2p_product_path_fidelity.py` proves one direct 2P product-route
  terminal trace stops a held decision early on wall death.
- Source-state visual and turn-commit wrapper tests prove some 2P action
  propagation into `joint_action` and `native_control_value`, but they are not a
  full controls-fidelity proof.

## Missing Tests

- No real browser DOM/EventEmitter integration or actual Socket.IO transport
  proof exists. The current JS tests use small shims around original modules.
- Touch and gamepad input reduction are not covered.
- Full multiplayer trainer replay arrays are still unsupported from the
  LightZero-facing wrapper. Current wrapper proof covers timestep observations
  and underlying public sidecars, not replay arrays.
- Normal public terminal rows still require reset before another step; current
  terminal-padding proof uses a controlled padded-row fixture, not a relaxed
  post-terminal public stepping contract.
- 3P/4P public controls now have one-frame trajectory parity and one 4P
  held-control parity proof, but not broad lifecycle/replay/trainer propagation.

## Test Cases Added

1. `test_js_player_input_reduction_and_release`
   Added in `tests/test_controls_source_input_fidelity.py`.

2. `test_js_player_move_reaches_server_update_angular_velocity`
   Added in `tests/test_controls_source_input_fidelity.py`.

3. `test_public_vector_2p_one_source_frame_control_change_matrix`
   Added in `tests/test_controls_vector_fidelity.py`.

4. `test_public_vector_2p_held_control_over_decision_source_frames`
   Added in `tests/test_controls_vector_fidelity.py`.

5. `test_public_vector_2p_release_to_straight_stops_turning`
   Added in `tests/test_controls_vector_fidelity.py`.

6. `test_public_vector_invalid_and_inactive_controls_are_contractual`
   Added in `tests/test_controls_vector_fidelity.py`.

7. `test_public_vector_terminal_padding_noop_under_decision_source_frames`
   Added in `tests/test_controls_vector_fidelity.py`.

## Exact Next Test Cases

1. `test_touch_and_gamepad_input_reduction_if_training_or_browser_eval_needs_it`
   Source keyboard behavior is now pinned. Touch/gamepad behavior should be
   pinned only if a future route depends on it.

2. `test_wrapper_replay_array_contract_when_replay_arrays_exist`
   The current wrapper proof intentionally stops at timestep observations and
   underlying public sidecars. Add replay-array assertions when that surface is
   promoted.
