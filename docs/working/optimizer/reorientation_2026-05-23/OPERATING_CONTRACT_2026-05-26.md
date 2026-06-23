# Optimizer Operating Contract, 2026-05-26

Purpose: stop the optimizer lane from drifting into side patches, stale speed
claims, or repeated explanations.

Status: archive. Superseded by `goal.md`, `NEXT_MOVES.md`,
`ORCHESTRATION.md`, and `OPERATING_PATTERNS.md` for OPT-132. Keep this file for
historical guardrails only; do not use its current target as active work.

## Single Objective

Speed up the real CurvyTron training path by removing avoidable host handoff and
scalar/object churn from the hot loop, while preserving the training-visible
observation/search/replay contract.

The current implementation target is now:

```text
promote the resident compact path from optimizer sidecar proof to an
end-to-end, Coach-comparable candidate profile
```

The resident renderer-to-search handoff itself has passed. The remaining gap is
that stock Coach training still enters through `lzero.entry.train_muzero`,
while the best resident compact path enters through the optimizer sidecar.

## Current Critical Path

The next main-thread work must move this path:

```text
persistent renderer device frame
-> resident observation stack/handle
-> compact Torch search
-> durable compact replay/sample path with policy-version lineage
-> learner/RND consumption
-> split compact-owned loop entrypoint
-> matched candidate-vs-stock profile result
```

Done means the whole chain has a clear denominator and a fail-closed result.
A contract, dataclass, flag, test stub, or local helper is not done unless the
next consumer in the chain uses it.

Side work is allowed only if it fixes a failing gate on that path or runs in a
subagent without blocking the main thread.

## Progress Gates

Do not claim progress unless all relevant gates are named.

1. End-to-end path proof
   - The intended code path must be active.
   - Telemetry must show the resident/compact path was used.
   - Hidden fallback is failure, not partial success.

2. Correctness proof
   - Shape, dtype, device, row order, player order, stack freshness, and terminal
     handling must be checked.
   - If the fast observation differs from the trusted path, describe the
     difference in plain language and say whether it is training-visible.

3. Matched speed proof
   - Compare against a named baseline from the same repo state.
   - State the speed currency in the first sentence.
   - Report the major buckets, not just a single wall-clock number.

## Stop Conditions

Stop and reorient before more edits if any of these happen:

- two consecutive patches do not directly move the current critical path;
- a speed claim cannot be tied to a manifest, result artifact, and config;
- docs, code, and run settings disagree about the active path;
- a path silently falls back instead of failing closed;
- a profile excludes a bucket but is being discussed as a full-loop result.

## Main Thread Rule

The main thread owns the critical path.

Before editing, name the artifact being advanced. After editing, update:

- `CURRENT_STATE.md` for what is true now;
- `TODO.md` for the next concrete task;
- `MEASUREMENT_LEDGER.md` if a run produced a number;
- this contract if the active objective or gates change.

## Subagent Rule

Subagents are scouts, not drivers.

Use them for bounded work:

- read a specific set of files and report stale assumptions;
- critique a concrete design for hidden fallback;
- run a named profile grid and write exact results;
- check whether telemetry proves the intended path ran.

Every subagent prompt must include:

- one narrow question;
- exact files or commands when known;
- the expected output shape;
- where the answer should be recorded or how it will change the next action.

If a subagent result does not change the next action, summarize it and move on.

## Docs Rule

Active docs must stay small and decisive:

- `CURRENT_STATE.md`: the one source of truth for what is true now;
- `TODO.md`: ordered next tasks;
- `BOTTLENECK_MODEL.md`: current Amdahl read;
- `MEASUREMENT_LEDGER.md`: run names, configs, numbers, conclusions;
- `RESIDENT_OBSERVATION_SEARCH_CONTRACT_2026-05-26.md`: current data contract;
- this file: operating rules.

Older docs are historical evidence. They must not override the active contract.
Files with old names like `current_state`, `next_moves`, or `coach_handoff`
outside the active reorientation folder are evidence only unless
`CURRENT_STATE.md` explicitly promotes them.

Every doc update should answer:

- what changed;
- what evidence supports it;
- what to do next.
