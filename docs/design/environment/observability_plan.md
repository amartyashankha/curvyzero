# Environment Observability Plan

Status: Draft

This page defines the small observability layer for environment reverse
engineering. The goal is to make JS/Python mismatches easy to understand without
turning the fidelity lane into a dashboard or browser-tooling project.

## Why It Matters

Reverse engineering is mostly a loop of asking why two trusted-looking runs
diverged. Good observability shortens that loop:

- It shows the first behavior difference, not just that a batch failed.
- It preserves enough context to explain the difference without reopening huge
  raw traces.
- It keeps source-fidelity facts separate from training-facing environment
  APIs.
- It makes each fix reviewable: scenario, raw artifacts, common projection,
  diff, and summary all point at the same moment in time.

The observability bar is practical, not ornamental. A developer should be able
to inspect one failed scenario and decide which source fact, runner projection,
or Python rule needs attention.

## Current Artifacts

The current local fidelity loop already emits useful debugging evidence:

- Scenario JSON under `scenarios/environment/` defines provenance, initial
  state, time policy, moves, comparison mode, and tolerances.
- `tools/reference_oracle/scenario_runner.js` writes the raw JS/source trace.
- Python source-fidelity runners write raw Python trace artifacts for the same
  scenario.
- Normalization projects raw outputs into `curvyzero_common_trace/v1` before the
  default diff.
- `tools/fidelity_diff.py` reports the first mismatch in common-trace mode.
- Per-scenario artifact folders contain `js.json`, `js.common_trace.json`,
  `js.timeline.txt`, `js.stderr.txt`, `python.json`,
  `python.common_trace.json`, `python.timeline.txt`, `python.stderr.txt`,
  `diff.json`, `diff.stderr.txt`, and `summary.json`.
- Batch runs add one compact root `summary.json` with counts and failed-scenario
  first-mismatch snippets.

These are enough for the current reconstruction loop. Keep raw artifacts out of
git; promote only curated fixtures, small examples, and dated experiment notes.

## Near-Term Improvements

Add only low-cost artifacts that make a failed scenario faster to read.

Done:

- Common trace files are explicit: `js.common_trace.json` and
  `python.common_trace.json` sit beside the raw runner outputs.
- Compact common-trace timelines are explicit: `js.timeline.txt` and
  `python.timeline.txt` summarize steps, step time, player state, body counters,
  and projected events when present.

Next useful target: compact mismatch context. Keep it bounded to the first
mismatch and one nearby step; do not grow this into a replay viewer.

1. Expand first-mismatch context.
   - Keep the existing field path, left value, right value, and message.
   - Include the step index, player id or event index when applicable, previous
     common-trace step, current JS step, and current Python step.
   - Keep this bounded to one or two nearby steps so `diff.json` stays small.

2. Add small state tables.
   - For failed scenarios, write a compact table-shaped JSON or markdown summary
     for the last matching step, first mismatching step, and maybe the next step.
   - Include player id, x, y, angle, alive, printing, score, round score, and
     other fields only when the scenario compares them.
   - Prefer this over long pasted trace excerpts in docs.

3. Record runner and projection fingerprints.
   - Include scenario id, ruleset id, source target, source commit, Python
     runner id, common-trace schema, comparison mode, and tool versions when
     available.
   - Do not block useful local runs if a version field is unavailable; mark it
     unknown and keep moving.

4. Consider lightweight raster later.
   - Add tiny local raster or frame summaries only when state traces are trusted
     and a visual question remains.
   - Use raster artifacts for inspection, observation work, or pixel-adjacent
     checks, not as the first source of truth for rule fidelity.

## Non-Goals For Now

- No production code changes just to satisfy this plan.
- No dashboards, web viewers, browser replay UI, or notebook-first workflow.
- No Modal-first observability layer; local artifacts should stay readable
  before remote batches matter.
- No screenshot or video diff as a substitute for state and event traces.
- No broad logging framework in `env.step()` or training hot loops.
- No large committed artifacts from `artifacts/`, `runs/`, `tmp/`, `logs/`,
  `checkpoints/`, `replay/`, or `videos/`.

## Suggested Artifact Layout

Keep the current layout and add derived files only when the producer already has
the data in memory:

```text
<artifact_root>/<scenario_id>/
  js.json
  js.common_trace.json
  js.timeline.txt
  js.stderr.txt
  python.json
  python.common_trace.json
  python.timeline.txt
  python.stderr.txt
  diff.json
  diff.context.json
  state_table.md
  summary.json
```

`diff.context.json` and `state_table.md` are optional. A passing scenario does
not need all of them. A failed scenario should have enough context to explain
the first mismatch without opening raw traces first.

## Implementation Order

1. Done: materialize `js.common_trace.json` and `python.common_trace.json` from
   the existing normalizer path.
2. Done: add compact `js.timeline.txt` and `python.timeline.txt` from those
   common traces.
3. Add bounded first-mismatch context to `diff.json` or a sidecar
   `diff.context.json`.
4. Add small state tables for failed scenarios.
5. Revisit lightweight raster artifacts only after the state/event loop is
   boring and observation work needs it.

Stop at the first improvement that makes the current mismatch easy to diagnose.
The point is faster reconstruction, not a larger observability product.
