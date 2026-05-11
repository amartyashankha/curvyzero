# 2026-05-08 Source Kinematics First Match

## Question

Can the Python scenario runner match the JS reference for the first forced
movement step without changing the toy-v0 runner?

## Result

Yes, for source movement kinematics only.

The Python scenario CLI now has:

```sh
uv run python -m curvyzero.env.scenarios scenarios/environment/forced_two_player_turn_step.json --runner source-kinematics --compact
```

This mode is deliberately narrow:

- scenario: `forced_two_player_turn_step`
- players: 2
- inputs: forced positions, forced headings, source moves `-1`, `0`, `1`
- timing: fixed `step_ms`
- speed: 16 source units/sec
- turn base: `2.8 / 1000` rad/ms

It does not implement collisions, trail behavior, bonuses, round state, or the
full JS game loop. The payload labels this as source fidelity only for forced
movement kinematics.

## Matched Values

For the first step at `1000 / 60` ms:

- player `p0`: angle `-0.046667`, position `[20.266376, 39.98756]`
- player `p1`: angle `3.188259`, position `[59.733624, 39.98756]`

These match the JS scenario runner's rounded common-trace values.

## Checks

```sh
uv run --extra dev pytest tests/test_env_scenarios.py
uv run --extra dev ruff check src/curvyzero/env/scenarios.py tests/test_env_scenarios.py
uv run python tools/fidelity_diff.py /private/tmp/curvy-js-source-kinematics.json /private/tmp/curvy-python-source-kinematics.json --json --common-trace
```

The common-trace diff returned `match: true`.

## Blockers

No blocker for first-step kinematics. Full source fidelity remains blocked on
collisions, trails, bonuses, and broader game state.
