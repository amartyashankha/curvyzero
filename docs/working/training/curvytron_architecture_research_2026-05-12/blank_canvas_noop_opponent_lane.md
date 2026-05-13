# Blank Canvas No-Op Opponent Lane

Date: 2026-05-13

Scope: focused design note for a one-player CurvyTron wall-avoidance sanity
lane that keeps the fixed-opponent LightZero interface shape but removes real
opponent dynamics. Status update: `blank_canvas_noop` is now implemented and
tiny stock `train_muzero` canaries passed for `body_circles_fast` and
`browser_lines`; keep the earlier design notes below as rationale.

## Findings

Faraday's follow-up confirmed the important negative result: current
`opponent_death_mode=immortal` is passive death immunity only. Player 1 still
moves, can continue out of bounds, writes normal collidable `body_*` trail
points, writes visual trail points, and player 0 can still die on player 1's
trail. Treat death immunity as insufficient for this lane.

The vector runtime stores collision trails and visual trails separately.
Collision bodies live in `body_active`, `body_pos`, `body_radius`,
`body_owner`, `body_num`, `body_insert_tick`, `body_insert_kind`,
`body_write_cursor`, and per-player `body_count`. Visual/browser-line trail
points live in `visual_trail_active`, `visual_trail_pos`,
`visual_trail_radius`, `visual_trail_owner`, `visual_trail_break_before`,
`visual_trail_write_cursor`, and `has_visual_trail_last`.

`step_many` currently advances each live player in reverse player order. For a
printing player it appends visual trail points, appends collision body points
when the draw cursor crosses the radius threshold, checks wall/body death, and
updates the print manager. Body collision scans all active `body_*` slots except
the player's own too-young trail points, so an opponent can only kill the
learner through active `body_*` records owned by the opponent.

Rendering is state-array driven. The gray64 source-state renderer draws active
`body_*` circles and then live heads for players where `present && alive`.
The canvas-like/browser-lines path prefers `visual_trail_*` records when
available, falls back to `body_*`, and also draws heads from `present && alive`.
Therefore suppressing opponent collision bodies is not enough for a visually
blank lane: the opponent head will still render unless it is hidden, absent, or
the renderer gets a mask.

The current fixed-opponent wrapper always constructs a two-player
`VectorMultiplayerEnv`, samples/chooses player 1's action, and steps a joint
action `[[player_0_action, player_1_action]]`. That is the right place to keep
the LightZero/two-player interface shape while making player 1 fake.

Outcome pressure is separate from opponent physics. The public env reward is
sparse winner/loss on terminated rows. The current dense wrapper reward still
adds that sparse outcome term. A blank lane should use an explicit
survival-only or survival-plus-bonus reward variant with terminal outcome terms
zeroed, because a fake player 1 should not make "beat/loss versus opponent" a
training target.

## Existing Knobs

Closest existing approximations:

- `opponent_policy_kind=fixed_straight`: simple, deterministic opponent action,
  but player 1 still moves, dies, renders, and leaves trails.
- `opponent_death_mode=immortal`: current dirty-tree implementation maps to
  `death_immunity_player_ids=(1,)`, so player 1 cannot die from walls/bodies.
  It does not stop movement, trail writes, rendering, bonus catches, or weird
  out-of-bounds state after suppressed wall deaths.
- `death_mode=profile_no_death` / `disable_death_for_profile=True`: disables
  death globally and is profile-only; it also makes the learner unable to die,
  so it is the wrong lane.
- `source_state_trail_render_mode=body_circles_fast|browser_lines`: changes
  visual rendering only. It does not remove collision bodies or opponent heads.
- Public present/absent support exists for multiplayer lifecycle fixtures, but
  reset requires at least two present players and the fixed-opponent wrapper is
  built around a 2P joint-action shape.

## Minimal Clean Implementation

Add a wrapper-level diagnostic mode, not a trainer hot-path rewrite:

```text
opponent_runtime_mode = normal | blank_canvas_noop
```

For `blank_canvas_noop`, keep `VectorMultiplayerEnv(player_count=2)` and keep
the LightZero action/metadata shape as a fixed-opponent env. The clean contract
is: player 1 remains a real slot for array shapes and lifecycle accounting, but
is physically inert, hidden from observation, unable to write trail state,
unable to catch bonuses, and ignored by training reward.

Preferred implementation after read-only code review:

- add a `disabled_player_mask` to the vector step input;
- pass it through `VectorMultiplayerEnv.step(...)`;
- in the vector runtime, exclude disabled players from the per-player live mask
  used for movement, trail writes, wall/body collision, print-manager updates,
  and bonus catch checks;
- keep player 1 public `present/alive` for shape and lifecycle;
- render through a wrapper-owned state view or render mask that hides player 1
  from observations/GIFs.

This is cleaner than relying only on post-step cleanup. A small scrubber is
still useful as a guard, but the disabled-player mask should prevent most
side effects from happening in the first place.

After reset and after every physical step, apply a narrowly scoped player-1
scrubber as a defense:

- keep the wrapper interface two-player and continue emitting a player 1 action
  field, preferably `NO_OPPONENT_ACTION` or fixed straight with metadata saying
  it is ignored;
- keep `present[0, 1]` and `alive[0, 1]` true if possible, so the underlying
  2P lifecycle does not see a one-live-player row before the learner dies;
- prevent movement by freezing player 1 speed and turn rate at zero;
- set player 1 radius to zero or otherwise exclude its head from physics that
  could catch bonuses;
- use `death_immunity_player_ids=(1,)` as a guard, but not as the main feature;
- set `printing[0, 1] = False` and `print_manager_active[0, 1] = False`;
- clear player 1 draw/visible-trail cursors;
- clear all active `body_*` slots and `visual_trail_*` slots whose owner is 1;
- hide player 1 from source-state visual rendering, preferably by rendering
  through a wrapper-owned temporary state view or render mask rather than
  mutating public lifecycle `present` just for pixels;
- use a no-outcome training reward variant for this lane.

Rejected shortcut: making player 1 absent/dead after reset is smaller, but it
mixes the fake-opponent idea into lifecycle and scoring. In particular,
runtime death scoring uses `player_count - alive_count` at frame start, so a
pre-dead player 1 can distort score metadata when player 0 later dies. It may
still be acceptable for a tiny local probe, but it is not the clean matrix lane.

The cleanest low-blast-radius version is to make the source-state survival
wrapper own this as a diagnostic state scrubber plus renderer visibility
metadata. Avoid adding generic trail modes to `vector_runtime` until the lane
graduates, because true no-trail source semantics would need hooks in warmup
PrintManager start, normal point insertion, important point insertion, visual
trail insertion, body collision, and renderer heads.

Recommended acceptance checks before any matrix row:

- reset observation contains only the learner head and no player 1 trail/head
  pixels;
- after N straight steps, player 1 position/trail counters/body counts remain
  unchanged or zero;
- forcing player 1 into a wall or body does not emit `death_player_ids`, does
  not terminate, and does not add death/body points;
- forcing player 0 into a wall still terminates normally;
- player 0's terminal training reward comes only from the survival/no-outcome
  reward contract, not from player 1 winning;
- placing player 0 on a synthetic player 1 body is impossible in the lane
  because the scrubber removes owner-1 bodies before collision/render;
- with bonuses enabled, player 1 cannot catch or alter bonuses.

## Recommendation

Do not use current `opponent_death_mode=immortal` as the blank-canvas lane. It
is useful as a separate immortal-opponent canary, but it still has a moving,
rendered, trail-capable opponent.

For the first wall-avoidance sanity lane, implement
`opponent_runtime_mode=blank_canvas_noop` only in
`CurvyZeroSourceStateVisualSurvivalLightZeroLocalEnv`, plus the minimal
`disabled_player_mask` hook in the vector env/runtime, focused tests, and a
no-outcome reward variant. Keep training launcher changes limited to config
plumbing and metadata. Defer any broad `opponent_trail_mode=none` until there
is evidence we need a real no-trail opponent that still appears/moves.

Main risks:

- hiding player 1 by mutating `present/alive` can perturb lifecycle, bonus, and
  observation metadata unless the wrapper labels the mode clearly;
- clearing owner-1 body slots without compacting arrays can leave confusing
  write cursors, so tests should check active slots and capacity behavior;
- if player 1 remains `present && alive`, current renderers will draw its head,
  so blank visuals require either state hiding or a renderer mask.
- leaving the current dense-plus-outcome reward in place would preserve
  opponent-outcome pressure even if the opponent cannot move or write trails.
- reward support sizing for the no-outcome reward must be checked before a high
  cap run. A huge value/reward support head would make the clean lane too heavy
  or impossible even if the reward logic is semantically correct.
