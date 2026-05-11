# 2026-05-08 Environment Reconstruction Reorientation

Status: short critique

## Read

- `docs/research/curvytron_source_map/README.md`
- `docs/design/environment/README.md`
- `docs/research/environment/README.md`
- `docs/design/environment/trace_loop_contract.md`
- `docs/design/environment/scenario_schema.md`
- `docs/design/environment/trace_schema.md`
- `docs/design/environment/fidelity_checklist.md`
- `docs/design/environment/fidelity_comparison.md`
- `docs/design/environment/reference_oracle.md`
- `docs/design/environment/modal_fidelity_jobs.md`
- `docs/handoffs/2026-05-08-environment-fidelity-handoff.md`
- `tools/run_fidelity_loop.py`
- `tools/reference_oracle/scenario_runner.js`
- `src/curvyzero/env/scenarios.py`
- `src/curvyzero/env/trace_compare.py`
- `tools/fidelity_diff.py`

## Short Answer

The core architecture is not too complicated:

```text
source map -> probe -> scenario -> common trace -> diff -> implement -> test
```

The overcomplication is around the core. There are too many schema names, old
aliases, artifact layouts, raw-diff paths, and future Modal/browser plans for the
stage we are in.

Keep the local loop as the main path. Make common-trace comparison the default.
Automate one mechanic at a time.

## What Is Good

- The source map is the right front door for reverse engineering.
- The headless JS runner is the right oracle. It uses source objects and avoids
  the old browser/server build.
- The Python runner is honest about `toy-v0` versus source fidelity.
- The common trace is the right comparison layer.
- `pass`, `fail`, and `blocked` are useful. A broken or unsupported comparison is
  not the same as a rule mismatch.
- Modal is correctly scoped as batch-level only.

## What Is Too Complicated

- Scenario shape was split between older `environment-scenario-v0` fields and
  newer `environment_scenario/v1` fields. The current loop contract now treats
  the accepted v0 shape as the write shape until a real migration happens.
- Raw diff used to be a normal path, even though raw JS and raw Python traces
  should not match. Common-trace diff is now the default.
- Local artifacts and planned Modal artifacts use different layouts.
- The source-kinematics Python path was hardcoded to one scenario; it has now
  been generalized across the current forced movement fixtures.
- Several docs redefine similar contracts. Readers have to guess which page wins.
- Modal, browser hosting, screenshots, videos, and protocol checks are described
  before the state loop is broad enough to need them.

## Are Responsibilities Clear?

Mostly, but they need one canonical statement.

- Source map: records source facts and evidence status.
- Scenario: names one setup, time policy, actions, target ruleset, and tolerances.
- JS runner: emits raw reference state and events.
- Python runner: emits raw Python state for the selected target.
- Normalizer: projects raw traces into `curvyzero_common_trace/v1`.
- Differ: compares common traces and reports first mismatch.
- Loop runner: owns command order and local artifacts.
- Modal wrapper: later runs the same loop over batches.

Do not let runners own diff policy. Do not let Modal own trace fields. Do not let
experiment notes redefine schemas.

## Automate Next

1. Make common-trace diff the default in the local loop. Keep raw diff as an
   explicit debug option.
2. Write `js/common_trace.json` and `python/common_trace.json` as real artifacts
   before diffing.
3. Done: generalize `--python-runner source-kinematics` across the current
   forced movement fixtures.
4. Done: add a local batch command that runs selected scenario JSON files and
   writes one compact pass/fail/blocked summary.
5. Add a scenario schema check or migration helper. New files should use one shape;
   old aliases should be read-only compatibility.
6. Promote the movement suite first: straight, left, right, mixed, and elapsed-ms
   checks.
7. After movement is boring, add wall death, wall scoring, borderless wrap if in
   scope, same-frame deaths, and 3/4-player canaries.

## Canonical Vs Support

Canonical now:

- Source facts: `docs/research/curvytron_source_map/README.md` and subsystem pages
  under that folder as they are added.
- Target rules: `docs/design/rulesets.md` and
  `docs/design/deterministic_environment.md`.
- Workflow: `docs/design/environment/reconstruction_workflow.md` and
  `docs/design/environment/trace_loop_contract.md`.
- Common trace: `docs/design/environment/trace_schema.md`.
- Current tools: `tools/run_fidelity_loop.py`,
  `tools/reference_oracle/scenario_runner.js`,
  `src/curvyzero/env/scenarios.py`, `src/curvyzero/env/trace_compare.py`, and
  `tools/fidelity_diff.py`.

Canonical but needs cleanup:

- `docs/design/environment/scenario_schema.md`. It still describes the older
  fixture shape. Pick one write shape and call old fields compatibility.

Support notes:

- `docs/handoffs/*`
- `docs/experiments/environment/*`
- `docs/research/environment/*critique*.md`
- `docs/working/*`
- Modal/browser hosting notes until the local state loop needs remote batching.

Support notes can explain why choices were made. They should not be treated as
the active contract.
