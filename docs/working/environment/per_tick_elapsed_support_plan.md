# Per-Tick Elapsed Support Plan

Status: implemented locally
Date: 2026-05-09

This lane promoted `source_kinematics_varied_elapsed_multistep.json` after the
source-kinematics runner learned per-step elapsed milliseconds. The fixture was
already JS-pinned and proves that
`10 + 20 + 15 + 21.666666666666668` ms does not trace the same path as four
fixed `16.666666666666668` ms ticks, even though total elapsed time matches.

## Result

- `run_source_kinematics_scenario()` now passes one elapsed-ms value per source
  input frame into `_trace_source_kinematics()`.
- Fixed policies still repeat one scalar for all ticks.
- Per-step policies read `time_policy.step_ms_sequence` and check
  `total_step_ms` when it is present.
- Common trace projects per-step Python `step_ms` after the reset frame is
  dropped.
- Source `angle` events are projected for JS and Python common traces.
- `source_kinematics_batch.json` now includes the varied elapsed-ms fixture.

## Completed Implementation Checklist

1. In `source_runners.py`, keep the existing scalar helper for other promoted
   source runners. Add a narrow source-kinematics helper that returns one
   elapsed-ms value per source input frame.
2. For `time_policy.kind == "fixed"`, repeat `time_policy.step_ms` for
   `len(scenario.raw_action_script)` ticks so existing fixtures remain
   byte-for-byte equivalent.
3. For `time_policy.kind == "per-step"`, read
   `time_policy.step_ms_sequence`; reject bools/non-numbers and require length
   to equal `len(scenario.raw_action_script)`. If `total_step_ms` is present,
   compare it to `sum(step_ms_sequence)` with a tiny float tolerance.
4. Change only `run_source_kinematics_scenario()` and
   `_trace_source_kinematics()` to pass and consume that sequence. In the trace
   loop, zip each source input frame with its matching elapsed-ms value.
5. Preserve movement event order from the varied fixture. When events are
   included, emit an `angle` event before the same player's `position` event
   only for non-zero moves, while retaining the existing reverse player update
   order.
6. In `trace_compare.py`, project Python `step_ms` per common-trace step:
   fixed policies keep the scalar; per-step policies use
   `time_policy.step_ms_sequence[step_index]` after the reset frame has been
   dropped.
7. In `trace_compare.py`, project source `angle` events for both JS and Python
   common trace as `{event, player_id, angle}` so the varied fixture can pass
   with `include_events: true`.
8. Convert the strict xfail in `tests/test_env_scenarios.py` into the normal
   source-kinematics multistep oracle test, then append
   `source_kinematics_varied_elapsed_multistep.json` to
   `scenarios/environment/source_kinematics_batch.json`.
9. Add focused `tests/test_trace_compare.py` cases for Python per-step
   `time_policy.step_ms_sequence` projection and source `angle` event
   projection.

## Minimal File Set

- `src/curvyzero/fidelity/source_runners.py`
- `src/curvyzero/env/trace_compare.py`
- `tests/test_env_scenarios.py`
- `tests/test_trace_compare.py`
- `scenarios/environment/source_kinematics_batch.json`

No scenario-schema change is needed while fixtures provide
`time_policy.step_ms_sequence`. Add schema support later only if new fixtures
need to derive the sequence from `steps[].step_ms`.

## Acceptance Commands

```bash
uv run pytest tests/test_env_scenarios.py -k "source_kinematics"
uv run pytest tests/test_trace_compare.py
uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --fail-on-mismatch --artifact-root /private/tmp/curvy-source-kinematics-regression
```

Latest local claim: `source_kinematics_batch.json` protects the fixed-step and
varied elapsed-ms movement fixtures through JS/Python common-trace parity.

## Risks

- Off-by-one timing: Python traces include a reset frame, but common trace drops
  it. The first sequence value must map to the first action frame.
- Shared-runner blast radius: many source runners still use the scalar helper,
  so keep the per-step helper scoped to source kinematics until each broader
  runner has a fixture that needs variable elapsed time.
- Silent fixed-step regression: rerun the current seven-case kinematics batch
  after movement runner edits.
- Fixture drift: keep `time_policy.step_ms_sequence`, `steps[].step_ms`, and
  `comparison.expected.frames[].step_ms` aligned when regenerating the JS pin.
