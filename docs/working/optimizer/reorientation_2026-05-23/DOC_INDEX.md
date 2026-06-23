# Optimizer Doc Index

Date: 2026-06-02

Purpose: keep the working-memory surface small.

## Canonical Read Order

| Doc | Role |
| --- | --- |
| `goal.md` | Main goal, current truth, active task, and stop rules. |
| `CURRENT_STATE.md` | Short fact sheet with latest valid evidence. |
| `TASK_BOARD.md` | One active task. |
| `ORCHESTRATION.md` | Exact next sequence when implementation spans more than one command. |
| `MEASUREMENT_LEDGER.md` | Rows that decide the current speed work. |
| `OPERATING_PATTERNS.md` | Stable rules for avoiding drift. |

## Optional Helpers

| Doc | Role |
| --- | --- |
| `NEXT_MOVES.md` | Optional short helper; it does not override the active plan. |
| `FOLLOWUPS.md` | Open follow-ups only. |
| `SUBAGENT_DELEGATION.md` | Current sidecar questions. |

## Treat Everything Else As Archive

Older dated docs are evidence only. They do not override the active docs above.

If an old document contains a fact that matters now, copy the fact into
`CURRENT_STATE.md` or `MEASUREMENT_LEDGER.md` in plain language. Do not make the
next agent reread the whole archive.
