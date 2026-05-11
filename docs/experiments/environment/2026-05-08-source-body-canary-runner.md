# 2026-05-08 Source Body Canary Runner

## Question

Can the Python source-body canary runner match the JS oracle for the promoted
body/trail canary fixtures?

## Scenario

- Batch: `scenarios/environment/source_body_canary_batch.json`
- Fixtures:
  - `source_body_opponent_tangent_safe_step.json`
  - `source_body_opponent_overlap_kills_step.json`
  - `source_body_own_delta3_safe_step.json`
  - `source_body_own_delta4_kills_step.json`
  - `source_body_same_frame_point_kills_step.json`
  - `source_body_same_frame_point_control_safe_step.json`
- Scope: narrow body/trail canaries only. This verifies exact-tangent safety,
  immediate opponent-body overlap death, own delta `3` safety, own delta `4`
  death, same-frame point materialization kill, and the same-frame control safe
  case. It does not cover deterministic print-manager holes, broader trail
  storage, bonuses, browser messages, or replay payloads.

## Commands And Results

```sh
uv run --extra dev pytest
```

Original result at this checkpoint: `84 passed in 1.12s`. Current full-suite
counts are tracked in the environment handoff because the suite has grown.

```sh
uv run --extra dev ruff check .
```

Result: `All checks passed!`.

```sh
uv run python tools/run_fidelity_batch.py scenarios/environment/source_body_canary_batch.json --python-runner source-body-canary --fail-on-mismatch --artifact-root /private/tmp/curvy-source-body-canary-final
```

Original result for the first two opponent-body cases: `2` pass, `0` fail, `0`
blocked; `diff_mode: common-trace`.

Later checkpoint: the same batch grew to four stored-body canaries and passed
with `4` pass, `0` fail, `0` blocked using `source-body-canary`.

Current checkpoint: the batch now contains six body/trail canaries and passes
with `6` pass, `0` fail, `0` blocked using `source-body-canary`.

## Interpretation

The source-body-canary runner is now verified for the six promoted body/trail
cases. Keep this checkpoint narrow: deterministic print-manager fixtures are now
covered by the separate `source-print-manager-canary` checkpoint, not by this
body canary runner.
