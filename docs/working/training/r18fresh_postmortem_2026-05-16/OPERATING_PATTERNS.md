# Operating Patterns

This file is durable. It is not the tactical task board. Keep it focused on how
we work so future agents do not repeat the same failure modes.

## Core Rhythm

- Write important facts down immediately. Do not rely on thread memory.
- Keep a current doc set and make the start-here path obvious.
- Separate durable operating habits from tactical orchestration.
- Use simple names for claims. Avoid terms that hide what is actually proven.
- Do not turn storage names into conceptual claims. For example,
  `round-000006` is a tournament game-batch directory, not a training round.
- Keep visible launch names short and structured: batch, row, active axes,
  recipe. Put exact settings in manifest fields, not in a swollen run ID.
- Treat old docs as archive unless a current doc explicitly promotes them.

## Parallel Work

- Keep the main thread for integration, direct verification, doc synthesis, and
  user-facing status.
- Delegate bounded questions to subagents: run metrics, tournament behavior,
  coverage gaps, cleanup inventories, and independent critiques.
- Give each subagent a narrow scope, explicit write permissions, and a concrete
  requested output.
- Use follow-ups when a subagent result is incomplete or ambiguous.
- Close completed agent lanes so the pool does not fill with stale workers.
- For optimizer work, keep bounded sidecars running while the main thread ships:
  one speed/implementation critic, one correctness critic, one strategy critic,
  and one fake-progress critic. Do not wait on them unless their answer blocks
  the next local step.
- Fold useful sidecar findings into the active board or goal doc, then keep
  moving. Sidecars are there to sharpen the main path, not to create a second
  research project.
- If a blocker appears, ask whether it blocks the whole goal or only one path.
  If only one path is blocked, run a smaller honest proof or alternate lane in
  parallel while debugging the blocked path.
- Keep an asset registry. Parallel lanes are not useful if nobody knows which
  apps, Volumes, Dicts, Queues, arenas, ratings, run prefixes, and docs are
  current.
- Do not spawn agents just to feel busy. If the agent pool is full, close stale
  lanes or narrow the question.

## Delegation Packet

Every delegated lane should state:

- Question:
- Scope:
- Read/write permission:
- Expected artifact:
- Stop condition:
- Integration target:

The integration target is usually `FINDINGS.md`, `SIGNALS.md`,
`ORCHESTRATION.md`, or `TODO.md`. A subagent answer that does not update the
main board has not actually reduced system uncertainty.

## Proof Standards

- Do not claim the full loop works from one adjacent artifact.
- Do not infer live behavior from patched defaults. Always read the real Modal
  Dict and volume artifact that the deployed scheduler is using.
- Do not infer catch-up from "a tournament game batch is running." First compare
  intake checkpoint count, latest rating checkpoint count, and active game-batch
  checkpoint count. If intake is ahead and the active batch is not larger than
  latest rating, the batch may be alive but still rating the old pool.
- After any "defaults are fixed" patch, immediately prove both sides:
  deployed status reads the intended IDs, and running jobs consume the intended
  assignment hashes. A patched constant is not live proof.
- Do not use raw Volume spelunking as the first-line operator path when a
  compact status tool exists. Use the tool first; only fall back to raw
  artifacts for deep forensics.
- If a status tool says something was skipped, it must also say why. Expose the
  compact skip decision in the status output before doing manual artifact digs.
- Do not skip a tournament game batch from a stale control/progress file alone.
  Before recovery skips anything, check real game output activity. If the output
  scan fails, surface that as a blocker instead of silently discarding active
  work.
- Recovery checks must count the output shape the scheduler actually writes.
  With one Modal worker per game, fresh `games/*/summary.json` files are real
  progress even if shard summaries are absent.
- If you add a status tool, immediately write its exact command, fields, and
  limits into the current truth docs before continuing.
- Full-loop proof means:
  `trainer checkpoint -> intake/subscriber -> tournament rating -> leaderboard
  publication -> assignment refresh -> trainer env rows loading those opponents`.
- Partial proof must be labeled as partial. For example, "48 of 136 trainers
  have latest generation-4 assignment applied" is real progress, but it is not
  the same as "the whole loop is stable."
- Prefer immutable Volume files and explicit assignment hashes over dashboards.
- Distinguish internal rating rows from trainer-facing active rows.
- Distinguish latest checkpoint quality from best-so-far checkpoint quality.
- Quantify survival, reward, tournament duration, rank, Elo, games, opponents,
  checkpoint count, and missingness separately.
- For each batch, require a lineage view that joins checkpoint write, intake,
  tournament scheduling, tournament result, trainer-facing export, assignment
  SHA, trainer apply, provider load, and post-apply learning metrics.
- Treat dashboards as operator aids, not truth. If a dashboard disagrees with
  Volume artifacts, debug the dashboard after preserving the artifact truth.

## Analysis Standards

- Use fair matched comparisons before drawing setting conclusions.
- Latest-only comparisons are operational health, not primary learning quality.
- Reward metrics are comparable within a reward variant; survival is the
  cross-reward metric.
- Tournament Elo is a relative head-to-head signal, not raw survival duration.
- When data is missing or confounded, say that plainly.
- Separate these questions every time:
  "Did the plumbing move data?" and "Did the policies improve?"
- Latest, best-so-far, and tournament-selected checkpoints are different
  objects. Do not substitute one for another without saying so.

## Sleep Tickets

Long sleeps are valid only when the expected next artifact needs wall-clock
time. Before sleeping, write down:

- Sleep reason:
- Expected artifact:
- Wake command or read path:
- Success predicate:
- Failure predicate:
- Parallel lane while sleeping:
- Owner:

## Cleanup Standards

- Preserve current useful live lanes until their data has been extracted.
- Do not leave stale apps, arenas, or docs as the first thing a new agent sees.
- Do not keep compatibility shims or hidden fallbacks in the current path just
  because old artifacts exist. Put old-artifact handling in explicit migration
  or archive tooling.
- Before killing a run batch, write down what signal was extracted and what
  still needs to be preserved.

## Self-Check

Before every substantial action, ask:

- Am I proving the actual loop or only a nearby piece?
- Am I about to wait when another useful lane can run in parallel?
- Did I update the current docs with the latest fact?
- Is this a real blocker or a quality preference?
- Are old/stale docs going to mislead the next agent?
- Are the names clear to a tired human looking at a dashboard?
- Are policy identity and runtime flags separated in the name? Example:
  rank1 is the policy, immortality is an `imm` axis.
- Is this action improving observability, learning signal, or cleanup, or am I
  just moving noise around?
