# 2026-05-08 Python Scenario Trace And Diff

## Question

Can the Python toy-v0 environment read the first shared environment scenario and
emit a JSON trace that can be compared with a tiny first-mismatch diff tool?

## Result

Yes, for plumbing only.

- Scenario: `scenarios/environment/forced_two_player_turn_step.json`
- Runner: `curvyzero.env.scenarios`
- Diff: `tools/fidelity_diff.py`

The Python runner reads the first forced 2-player scenario, maps source move
values `-1`, `0`, and `1` to toy-v0 actions, forces the starting positions and
headings, and emits a toy-v0 trace payload.

This is not a source-fidelity claim. The output is labeled `source_fidelity:
false` and `toy_v0_behavior: true`.

## Checks

```sh
uv run python -m curvyzero.env.scenarios scenarios/environment/forced_two_player_turn_step.json --compact
uv run --extra dev pytest tests/test_env_scenarios.py tests/test_env_tracing.py
uv run --extra dev ruff check src/curvyzero/env/scenarios.py tools/fidelity_diff.py tests/test_env_scenarios.py
```

## Blockers

None for this toy-v0 loop. Matching the JS oracle is still future work.
