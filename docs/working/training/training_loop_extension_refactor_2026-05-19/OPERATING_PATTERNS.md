# Operating Patterns For This Refactor

Last updated: 2026-05-19

This doc is about how to operate on this task. It is separate from `ORCHESTRATION.md`, which tracks the current work and subagents.

## Durable State

- Use this folder as the durable state for the refactor.
- Update docs before switching lanes.
- When interrupted, re-read `CURRENT_PHASE.md`, `TASK_BOARD.md`, and `ORCHESTRATION.md` before acting.

## Parallel Work

- Use subagents for bounded audits and critiques.
- Keep subagents read-only until a write scope is explicit.
- Do not duplicate the same investigation across agents unless the point is critique from different angles.
- Integrate every useful subagent result into `FINDINGS_LOG.md`; do not leave important conclusions only in chat.

## Refactor Discipline

- Prefer small, behavior-preserving extractions.
- Remove stale defaults and hidden fallbacks when they are in the touched path and the intended replacement is clear.
- Avoid broad compatibility layers in this research repo unless they protect a real running experiment.
- Pin important contracts with tests before depending on them.

## Patch Order

1. Observe the current contract.
2. Write or identify the focused test gate.
3. Extract the pure contract or wrapper.
4. Remove or ledger any fallback in the touched path.
5. Run the local gate.
6. Only then consider a Modal launch for a named remote/runtime question.

## Trainer Patch Gate

Before touching the Modal trainer, name the patch type:

- pure extraction;
- side-effect wrapper;
- temporary compatibility shim;
- documented exception.

Any temporary compatibility shim needs an owner, protected run or caller, expiry/deletion criteria, and test coverage.

## Compatibility Ledger

Compatibility is not a habit. It is a ledger.

Every alias, old default, fallback, or private helper preserved in a touched path needs:

- keep or delete;
- reason;
- protected run/caller if kept;
- expiry/deletion criteria;
- test coverage.

Unlisted fallback means delete when touched.

## Subagent Integration Gate

Do not start another broad scout/critic batch until prior findings have been reduced into:

- `FINDINGS_LOG.md`;
- accepted/rejected decisions;
- task-board changes.

Subagent reports should answer: required decisions, first safe patch, tests that must exist, and fallbacks to delete.

## Research Discipline

- Separate observations from decisions.
- Quantify when possible.
- Use file/line references for claims about code.
- Ask "what would make this false?" before declaring something done.

## Waiting And Context

- If a task requires waiting, use that time for non-overlapping work.
- If there is no useful parallel work, use a long sleep instead of token-dripping.
- Keep the main context clean: targeted reads, summaries, and docs over giant file dumps.
