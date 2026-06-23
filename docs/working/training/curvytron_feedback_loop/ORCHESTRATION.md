# Orchestration

Last updated: 2026-05-16.

## Main Thread Contract

Use the main thread for:

- deciding the next question;
- delegating independent work;
- integrating subagent reports;
- updating docs;
- running small local simulations;
- deciding when code changes are justified.

Do not use the main thread for broad code spelunking if a subagent can do it.

## Active Subagents

| Agent | Lane | Current Ask | Next Follow-Up |
| --- | --- | --- | --- |
| Kepler | current implementation truth | Map current scheduler, rating, intake, tests, and exposed knobs. | Ask for exact `adaptive_v0` defaults and where caller knobs still leak. |
| Fermat | scheduler research | Compare scheduling families and rating systems. | Ask for a short recommendation matrix. |
| Erdos | simulations | Design simulation worlds, schedulers, and metrics. | Ask for runnable toy harness shape and first expected outputs. |
| Chandrasekhar | docs and phase gates | Propose compact doc organization. | Ask whether this hub doc set is enough or missing an operator view. |

## Follow-Up Rhythm

After each subagent report:

1. Extract facts.
2. Mark what changed in `CURRENT_RESEARCH_PHASE.md`.
3. Add or adjust tests in `SCHEDULER_SIMULATION_PLAN.md`.
4. Add concrete results to `EXPERIMENT_LOG.md`.
5. Update the task board if the work changed priorities.

## Main Thread Checkpoint

Before starting another broad lane, answer these plainly:

- What changed since the last checkpoint?
- Which subagent reports are still pending?
- Which findings have been written into docs?
- Which experiment or proof is the current blocker?
- Which lane can be closed?
- Which task is safe to delegate next?

## Current Delegation Plan

Near term:

- Main: create docs and run the first toy simulation.
- Kepler: verify current code behavior and defaults.
- Fermat: research rating/scheduling theory and recommend simple V1 options.
- Erdos: turn the simulation design into concrete probes.
- Chandrasekhar: keep the documentation surface small.

Next wave:

- One subagent should red-team the chosen scheduler.
- One subagent should inspect website implications only after scheduler shape is
  clearer.
- One subagent should inspect refactor cuts only after current code truth is
  known.

## Operating Reminders

- Use simple words.
- Prefer bounded experiments over vague arguments.
- Keep docs current while learning, not after the fact.
- Treat all-pairs as an audit tool, not the default service shape.
- Do not confuse ranking quality with loop wiring.
- Do not confuse website symptoms with durable tournament truth.
- Do not launch large Modal work until the local scheduler plan is coherent.
