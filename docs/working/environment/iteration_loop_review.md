# Environment Iteration Loop Review

Status: concise working note for the environment-fidelity lane.

## Loop

Use this loop until it stops producing useful evidence:

```text
source-read -> probe/oracle -> scenario -> common trace -> Python canary runner -> batch -> docs
```

Keep each turn small. Read the relevant source first, prove the behavior with a
headless oracle or probe, encode one forced scenario, compare JS and Python through
the common trace, then promote only the useful slice into a batch and document the
result.

## Current Proven Slices

- Common-trace plumbing, sidecars, and batch execution are the right proof
  surface for promoted source slices.
- Source-backed core slices exist for movement, selected wall/border behavior,
  body collisions, collision order, PrintManager behavior, trail cadence,
  forced trail gaps, one natural taped trail-gap case, selected bonus proofs,
  and 28 pinned lifecycle fixtures.
- `CurvyTronSourceEnv` and the JS oracle are proof tools while source rules move
  into one fast faithful runtime: `VectorMultiplayerEnv`.
- The strict public 1v1/no-bonus bridge has landed for the named long
  reset-to-terminal wall-round-done fixture. This is proof/profiling only, not
  the final target.
- Current multiplayer fast-runtime evidence includes direct seeded 3P/4P
  no-bonus wall-scoring/order canaries, no-bonus N-player reset/warmup helpers,
  one narrow 3P warmdown/next-round helper, and a metadata-only public
  `VectorMultiplayerEnv` stepping surface. It is not natural public
  reset, learned observation, replay, or broad lifecycle coverage.
- Current focused validation should stay attached to the command that produced
  it. Do not use pass counts as a dashboard or fidelity headline.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

## Testing Posture

Improve testing by adding the smallest scenario that isolates the next source fact.
Prefer forced state, deterministic inputs, and common-trace diff over broad
simulation snapshots. Add batch coverage when two or more related fixtures protect a
mechanic. Keep Modal at coarse scenario or batch granularity; local trace loops
should carry the daily iteration load.

Avoid overengineering by resisting generalized replay, browser hosting, pixel diff,
or production environment rewrites until the next mechanic has source evidence and a
passing local canary.

Performance planning exists as a deferred lane in
[environment performance and vectorization](../../research/environment/performance_vectorization_plan.md):
use it to preserve API/vectorization options, not to pull optimization into the
current fidelity loop. The current toy-v0 smoke is about 33.5k steps/s; record it
as simplified-environment evidence, not a reason to optimize. Production
optimization backend rewrites remain deferred until source-fidelity fixtures and
the single-env contract are stable enough to compare backends.

## Blind Spots

- Broader opponent trail behavior, trail print cadence, trail holes, and
  delayed/exact-zero print-manager edges.
- Bonuses beyond the current narrow source proofs, broad round lifecycle beyond
  the pinned fixtures and strict 1v1 bridge, and replay/server message payloads.
- Numeric tolerance policy for longer elapsed-ms traces.
- Observation fidelity and trainer-facing reward/interface assumptions.
- Browser build/runtime fidelity for the original app.
- Modal JS oracle and full batch artifact layout are still future work.

## Next Measurement Improvements

- Add trail cadence and trail-gap fixtures using the verified print-manager
  toggle behavior as the base.
- Add an opponent-trail collision canary after trail gaps are understood.
- Add broader trail, bonus, and observation checks only after those mechanics
  have source evidence.
- Use `js.common_trace.json`, `python.common_trace.json`, `js.timeline.txt`, and
  `python.timeline.txt` when reading local mismatches.
- Add bounded first-mismatch context after the compact timeline sidecars.
- Record per-fixture provenance: source file/line, probe command, runner, trace
  schema version, and last batch status.
- Add compact batch summaries that report pass/fail/blocked, first mismatch, and
  fixture coverage by mechanic.
- Keep event comparison opt-in and named by contract so state-only fixtures stay
  simple.
- Keep benchmark work focused on scaffolding and measurement manifests for likely
  hot spots: trail and collision writes, observation generation, and Python
  dict/object overhead.
- Keep any optimization backend rewrite or vectorization TODO behind the
  source-fidelity loop; current local batches are evidence checks, not final
  performance proof.

## Detail Notes

Source-read means inspect the original CurvyTron implementation or the compact
source-map index before changing Python behavior. Probe/oracle means use the
headless JS path to force state and observe one behavior directly. Scenario means a
small JSON fixture with explicit player state, inputs, timing, and comparison
expectations. Common trace is the stable comparison layer; raw runner output is a
debug artifact. Python canary runner means the narrow source runner for one mechanic,
not the toy-v0 environment. Batch means related fixtures promoted together after the
single-scenario loop passes. Docs means writing the result down before moving on.

## TODO Checklist

- [x] Add same-frame point materialization fixtures and Python source-body
      canary support.
- [x] Add deterministic print-manager fixtures and Python/common-trace canary
      support.
- [ ] Add broader trail, bonus, and observation checks after print-manager.
- [x] Materialize explicit common trace artifacts as `js.common_trace.json` and
      `python.common_trace.json`.
- [x] Add compact timeline sidecars as `js.timeline.txt` and
      `python.timeline.txt`.
- [ ] Add bounded first-mismatch context.
- [ ] Link each promoted scenario to its source-read/probe note.
- [ ] Add a standard short run summary format for local batches.
- [ ] Decide when a Python canary graduates from narrow runner to shared source
      helper.
- [ ] Define tolerances for multi-step float drift before adding longer traces.
- [ ] Keep Modal batch design aligned with local artifact shape, but do not block
      local evidence on Modal.
- [ ] Revisit observation and trainer-interface checks after source state/events are
      less blind.
