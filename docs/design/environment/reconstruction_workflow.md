# Environment Reconstruction Workflow

Status: canonical workflow

Use one narrow path until it is reliable:

```text
source map -> probe -> scenario -> common trace -> diff -> implement -> test
```

## Workflow

1. Source map

   Read `docs/research/curvytron_source_map/README.md` and the relevant subsystem
   notes. Mark each fact as `source-mined`, `probe-backed`, `python-matched`,
   `pending`, or `deferred`.

2. Probe

   Use the headless JS runner to call original CurvyTron objects directly. The JS
   probe emits raw source trace and events. It should not copy physics rules into
   new JS logic.

3. Scenario

   Add or update one small scenario JSON. It defines setup, player starts, moves,
   time policy, target ruleset, provenance, and tolerances. It does not define
   artifact paths.

4. Common trace

   Run JS and Python from the same scenario. Normalize both raw outputs into
   `curvyzero_common_trace/v1`. Compare only fields that belong in the common
   trace.

5. Diff

   Diff common traces, not raw traces. The diff returns:

   - `pass`: fields match.
   - `fail`: fields differ; report the first mismatch.
   - `blocked`: the comparison is invalid, unsupported, or cannot normalize.

6. Implement

   Fix one real mismatch. If Python should intentionally differ, label it as
   `v0-choice` or `source-inspired` instead of forcing source behavior.

7. Test

   Add or update the smallest regression test or fixture. Promote the scenario
   only when provenance, target ruleset, tolerance, and last status are clear.

## Responsibilities

| Piece | Owns | Does not own |
| --- | --- | --- |
| Source map | Source files, facts, evidence status, open questions. | Python implementation policy. |
| Scenario | Input setup, actions, time, target, tolerance. | Artifact paths or runner internals. |
| JS runner | Raw source trace and source events. | Python behavior or diff policy. |
| Python runner | Raw Python trace for one selected target. | Source evidence status. |
| Normalizer | Common field names, frame alignment, reset-frame handling. | Rule fixes. |
| Differ | Pass/fail/blocked and first mismatch. | Trace production. |
| Local loop | Command order and local artifacts. | Physics semantics. |
| Modal wrapper | Batch execution and remote artifacts later. | Per-tick work or schema design. |

## Canonical Files

- Source facts: `docs/research/curvytron_source_map/README.md` and the source-map
  subsystem notes.
- Target rules: `docs/design/rulesets.md` and
  `docs/design/deterministic_environment.md`.
- Workflow and loop contract: this file and
  `docs/design/environment/trace_loop_contract.md`.
- Trace shape: `docs/design/environment/trace_schema.md`.
- Scenario shape: `docs/design/environment/scenario_schema.md`, after it is
  aligned to one write format.
- Comparison policy: `docs/design/environment/fidelity_comparison.md`.
- Current code path: `tools/run_fidelity_loop.py`,
  `tools/reference_oracle/scenario_runner.js`,
  `src/curvyzero/env/scenarios.py`, `src/curvyzero/env/trace_compare.py`, and
  `tools/fidelity_diff.py`.

## Support Notes

- Handoffs: useful starting context, not contracts.
- Experiment logs: evidence of what happened in one run.
- Research critiques: reasoning and risks, not active schema definitions.
- Working notes: scratch space.
- Modal/browser docs: future support unless a local scenario batch requires them.

## Current Simplifications

- Common-trace diff is the meaningful result.
- Raw traces are debug artifacts.
- Local single-scenario and local batch loops come before Modal.
- State and events come before browser pixels.
- Movement comes before collision, trails, scoring, randomness, bonuses, and
  browser protocol checks.
