# Operating Patterns

## Role

The main thread is the coach. It should plan, delegate, read returned findings,
update docs, and decide the next safe move. It should not disappear into a
large edit without tests.

## Work Rhythm

1. Re-read this directory before resuming.
2. Check the dirty worktree before editing.
3. Start independent audits in parallel.
4. Keep local source edits small and test-backed.
5. Update docs when facts change.
6. Report in simple language.

## Parallelism Rule

If tasks can run independently, start them together. If one lane fails, stop or
discard only the work that depends on that lane. Do not make unrelated lanes
wait for a speculative one.

## Self-Critique Questions

- Am I moving code before tests describe the current contract?
- Am I cleaning the trusted `--mode train` path or accidentally reviving the old
  custom path?
- Am I drifting into environment work when the current job is trainer code?
- Am I hiding training logic inside the environment?
- Am I confusing checkpoint discovery, status display, resume selection, and
  GIF/eval scheduling?
- Am I accidentally letting one policy action cover multiple granular game
  steps when the trusted lane should act every step?
- Did a returned subagent result change the todo list or source of truth?
- Are the names understandable to a tired human reading a dashboard or file
  tree?

## Subagent Rules

- Give each agent one bounded lane.
- Ask for exact file/function refs.
- Ask for facts first, hypotheses second, proposed patches third.
- Ask agents to write one concise doc when useful.
- Send follow-ups when assumptions change.
- Do not let two workers edit the same files in parallel.
