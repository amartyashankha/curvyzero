# 2026-05-08 Local Fidelity Loop

## Question

Can one local command run the JS scenario runner, the Python scenario runner,
and the fidelity diff for a single scenario?

## Result

Yes. `tools/run_fidelity_loop.py` runs the two trace producers and then runs
`tools/fidelity_diff.py`.

The loop now has `--python-runner`, which passes the scenario runner mode through
to `python -m curvyzero.env.scenarios` as `--runner <mode>`. Omitting it keeps
the Python scenario runner default unchanged: toy-v0.

Default scenario:

- `scenarios/environment/forced_two_player_turn_step.json`

Default outputs:

- `artifacts/local/fidelity/<scenario_id>/js.json`
- `artifacts/local/fidelity/<scenario_id>/python.json`
- `artifacts/local/fidelity/<scenario_id>/diff.json`
- `artifacts/local/fidelity/<scenario_id>/summary.json`

The default diff mode is common-trace. Use `--raw-diff` only to debug raw JS and
Python runner output. A mismatch is recorded in
`summary.json` as `status: "mismatch"` with a `first_mismatch` object. The
command returns success for a completed mismatch unless `--fail-on-mismatch` is
passed.

## Command

```sh
uv run python tools/run_fidelity_loop.py
```

Expected current result: the loop completes and records a mismatch, because the
Python runner is still the `curvyzero-v0` toy environment rather than a
source-fidelity implementation.

Source-kinematics check:

```sh
uv run python tools/run_fidelity_loop.py scenarios/environment/forced_two_player_turn_step.json --python-runner source-kinematics --artifact-root /private/tmp/curvy-single-default-common-trace
```

Exact result on 2026-05-08:

- exit code: `0`
- scenario: `forced_two_player_turn_step`
- artifact dir: `/private/tmp/curvy-single-default-common-trace/forced_two_player_turn_step`
- Python command included `--runner source-kinematics --compact`
- diff mode: `common-trace`
- status: `match`
- diff status: `pass`
- match: `true`
- first mismatch: `null`

Batch check:

```sh
uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --artifact-root /private/tmp/curvy-source-kinematics-batch
```

Exact result on 2026-05-08:

- exit code: `0`
- diff mode: `common-trace`
- pass: `4`
- fail: `0`
- blocked: `0`

## Checks

```sh
uv run --extra dev pytest tests/test_run_fidelity_loop.py
uv run --extra dev ruff check tools/run_fidelity_loop.py tests/test_run_fidelity_loop.py
```

## Blockers

No blocker for local loop automation or first-step source-kinematics matching.
Full source fidelity remains blocked on collisions, trails, bonuses, and broader
game state.
