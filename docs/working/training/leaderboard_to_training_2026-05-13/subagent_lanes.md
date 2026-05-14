# Subagent Lanes

## Lane Map

| Lane | Question | Output |
| --- | --- | --- |
| Trainer contract | How does trainer consume immutable opponent assignments? | implemented/missing map, tests, wiring plan |
| Tournament/intake | Does new checkpoint intake work as an online Elo feeder? | V0 vs target state, gaps, tests |
| Public leaderboard | How do rating snapshots become training-safe leaderboard snapshots? | publisher/pointer contract |
| Assignment selector | How are champions/recent/anchors/scripted sentinels selected? | deterministic strategy and audit schema |
| One-frame evaluator | Does tournament game execution match current train cadence? | parity tests and launch gate |
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
5. Seeded roster/scripted policy representation.
6. Intake continuation/idempotency.
7. Optimizer-safe settings for next manifest.
