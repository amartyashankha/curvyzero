# 2026-05-08 Common Trace Diff

## Question

Can the local loop compare JS and Python through the common trace instead of
stopping on runner metadata?

## Result

Yes. `uv run python tools/run_fidelity_loop.py --common-trace` now reaches the
first real game-field mismatch.

- Scenario: `scenarios/environment/forced_two_player_turn_step.json`
- First mismatch: `$.steps[0].players[0].angle`
- JS value: `-0.046667`
- Python toy-v0 value: about `-0.08`

## Why

The diff is doing its job. Python toy-v0 turns by a fixed per-tick amount and
uses toy time handling. The source path turns and moves from elapsed
milliseconds, source angular velocity, and source speed.

## Next Fix

Add a Python source-fidelity kinematics path for turn, time, and speed. Do this
before matching collisions, trail printing, or trail gaps.
