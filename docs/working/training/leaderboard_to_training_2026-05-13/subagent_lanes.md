# Subagent Lanes

## Lane Map

| Lane | Question | Output |
| --- | --- | --- |
| Trainer contract | How does trainer consume immutable opponent assignments? | implemented/missing map, tests, wiring plan |
| Tournament/intake | Does new checkpoint intake work as an online Elo feeder? | V0 vs target state, gaps, tests |
| Public leaderboard | How do rating snapshots become training-safe leaderboard snapshots? | publisher/pointer contract |
| Assignment selector | How are champions/recent/anchors/scripted sentinels selected? | deterministic strategy and audit schema |
| One-frame evaluator | Does tournament game execution match current train cadence? | parity tests and launch gate |
| Scheduler/testing critique | Are tournament tests checking product behavior, not just code paths? | fairness map, missing deterministic tests, launch gates |
| Seeded roster | How to include scripted/hand-coded policies and anchors? | roster schema options and recommendation |
| Optimizer/speed | Which speed settings are safe to use? | semantics-preserving recommendations |
| Docs critique | Is the current documentation useful and non-confusing? | structure fixes and missing context |

## Delegation Rules

- Give each lane exact paths to read.
- Ask for implemented vs designed, not generic summaries.
- Require caveats and next tests.
- Main thread reconciles conflicts.
- Do not let subagents approve launches.

## Return Template

Each subagent should return:

```text
Summary:
Implemented:
Designed-only:
Missing tests:
Risks:
Recommended next gate:
Files read:
```

## Current Priority Lanes

1. Public leaderboard snapshot/pointer contract.
2. Assignment selector and audit contract.
3. Trainer `--opponent-assignment-ref` plumbing.
4. One-frame tournament evaluator validation.
5. Scheduler fairness and repair tests.
6. Seeded roster/scripted policy representation.
7. Intake continuation/idempotency.
8. Optimizer-safe settings for next manifest.

## Latest Critique Notes

- Website lane: the tournament browser still does too much work in the main
  page route. Next cleanup should make `/` a fast shell and lazy-load rankings,
  checkpoint panels, battle panels, and GIF samples through cached JSON routes.
- Refactor lane: safest extractions are checkpoint discovery, intake-service
  pure logic, rating artifact I/O, and browser read-model/cache code. Avoid
  moving Modal app/image/Volume globals until behavior is locked.
- Active-pool lane: the top-100 rule belongs in rating/scheduling, not only in
  the website or public leaderboard. Retired rows are unscheduled history, not
  deleted checkpoints.

## Active Follow-Ups

- Website performance: inspect the tournament browser route and propose the
  smallest fast-shell/lazy-load/cache patch. Do not edit yet.
- Subscriber/intake: verify the simple product contract:
  watch training Volume paths, enqueue new checkpoints, continue ratings, keep
  full history, and let scheduling exclude retired rows.
- Refactor critique: identify safe extractions that shrink the Modal file
  without moving app/image/Volume globals or changing behavior.
