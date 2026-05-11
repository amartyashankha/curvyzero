# Multiplayer Fidelity Canaries

Status: verified narrow source-fidelity canaries. The 3P and 4P canaries below
pass through the JS/Python common-trace batch with event comparison enabled,
including the harder 4P same-frame terminal draw. This verifies only the named
normal-wall scoring, death-order, and terminal-draw behavior, not body
collisions, trails, bonuses, browser messages, or replay payloads.

Purpose: check the multiplayer rules that 1v1 wall/border cases cannot prove:
map size, reverse update order, same-frame scoring, accumulated death count,
survivor scoring, death order, and round end.

Keep these canaries in the fidelity lane. The public training interface can wrap
or simplify multiplayer later, but it should not decide whether the source rules
match.

Use forced state, disabled bonuses, `step_ms = 100`, radius `0.6`, speed `16`.
One straight tick moves `1.6` units. Headings are radians: `0` east,
`1.5707963268` south, `3.1415926536` west, `4.7123889804` north.

Apply all moves with `avatar.updateAngularVelocity(move)` before
`game.update(step_ms)`. Use `move = 0` unless stated. Player order is `P1`,
`P2`, `P3`, `P4`; the JS server updates in reverse order.

## 2P-CANARY-001: same-frame wall draw

Status: covered by the current wall/border event slice if mapped to
`source_normal_wall_same_frame_draw_step`; keep this doc as the readable intent.

- Players: 2. Expected map size: `88`.
- Initial state:
  - `P1`: `(86.9, 30.0)`, heading `0`, alive.
  - `P2`: `(1.1, 58.0)`, heading `3.1415926536`, alive.
- Script:
  - Tick 1: `P1=0`, `P2=0`, `step_ms=100`.
- Checks:
  - `P1` reaches about `(88.5, 30.0)` and dies on east wall.
  - `P2` reaches about `(-0.5, 58.0)` and dies on west wall.
  - Both deaths use frame-start death count `0`.
  - Death event order is `P2`, then `P1`.
  - Round ends with no round winner and final scores `P1=0`, `P2=0`.
- Testable now with headless JS:
  - Map size, motion, wall collision, death order, round end, score events.
- Later:
  - Browser/server message payloads and pixel rendering.

## 3P-CANARY-001: two die, one survives

Status: verified narrow only. Scenario:
`source_normal_wall_3p_two_die_one_survivor_step`.

- Players: 3. Expected map size: `95`.
- Initial state:
  - `P1`: `(47.5, 47.5)`, heading `0`, alive.
  - `P2`: `(93.9, 30.0)`, heading `0`, alive.
  - `P3`: `(1.1, 65.0)`, heading `3.1415926536`, alive.
- Script:
  - Tick 1: `P1=0`, `P2=0`, `P3=0`, `step_ms=100`.
- Checks:
  - `P1` reaches about `(49.1, 47.5)` and survives.
  - `P2` dies on east wall; `P3` dies on west wall.
  - Same-frame deaths both use frame-start death count `0`.
  - Death event order is `P3`, then `P2`.
  - Round winner is `P1`; final scores are `P1=2`, `P2=0`, `P3=0`.
- Testable now with headless JS:
  - 3-player map size, reverse update order, same-frame deaths, survivor bonus.
- Later:
  - Add a separate 3-player prior-death scoring canary if the JS oracle needs a
    focused non-terminal score trace.

## 4P-CANARY-001: ordered deaths and survivor score

Status: verified narrow only. Scenario:
`source_normal_wall_4p_ordered_deaths_survivor_score`.

- Players: 4. Expected map size: `101`.
- Initial state:
  - `P1`: `(50.5, 50.5)`, heading `0`, alive.
  - `P2`: `(99.9, 20.0)`, heading `0`, alive.
  - `P3`: `(98.3, 40.0)`, heading `0`, alive.
  - `P4`: `(96.7, 60.0)`, heading `0`, alive.
- Script:
  - Tick 1: all players `0`, `step_ms=100`.
  - Tick 2: all living players `0`, `step_ms=100`.
  - Tick 3: all living players `0`, `step_ms=100`.
- Checks:
  - Tick 1: `P2` dies, gets round score `0`.
  - Tick 2: `P3` dies, gets round score `1`.
  - Tick 3: `P4` dies, gets round score `2`.
  - `P1` survives and receives final survivor score `3`.
  - Final scores are `P1=3`, `P2=0`, `P3=1`, `P4=2`.
  - Death order is `P2`, `P3`, `P4`.
- Testable now with headless JS:
  - 4-player map size, accumulated death count, terminal survivor scoring.

## 4P-CANARY-002: prior deaths then terminal draw

Status: verified narrow only. Scenario:
`source_normal_wall_4p_two_prior_then_same_frame_terminal_draw`.

- Players: 4. Expected map size: `101`.
- Initial state:
  - `P1`: `(96.15, 20.0)`, heading `0`, alive.
  - `P2`: `(96.15, 40.0)`, heading `0`, alive.
  - `P3`: `(97.75, 60.0)`, heading `0`, alive.
  - `P4`: `(99.35, 80.0)`, heading `0`, alive.
- Script:
  - Tick 1: all players `0`, `step_ms=100`.
  - Tick 2: all living players `0`, `step_ms=100`.
  - Tick 3: all living players `0`, `step_ms=100`.
- Checks:
  - Tick 1: `P4` dies, gets round score `0`.
  - Tick 2: `P3` dies, gets round score `1`.
  - Tick 3: `P2` and `P1` die in the same terminal frame and both get round
    score `2`.
  - No player survives; round ends with no round winner.
  - Final scores are `P1=2`, `P2=2`, `P3=1`, `P4=0`.
  - Death order is `P4`, `P3`, `P2`, `P1`.
- Testable now with headless JS:
  - 4-player map size, accumulated death count, same-frame score capture,
    no-survivor terminal draw, and reverse update ordering.

## Verification

Command:

```sh
uv run python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-batch-terminal-draw-final
```

Result: `3` pass, `0` fail, `0` blocked; `diff_mode: common-trace`;
`first_mismatch: null` for
`source_normal_wall_3p_two_die_one_survivor_step`,
`source_normal_wall_4p_ordered_deaths_survivor_score`, and
`source_normal_wall_4p_two_prior_then_same_frame_terminal_draw`.
