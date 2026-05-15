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
| Modal/debug tooling | Can we summarize live Modal artifacts without hand-running many commands? | one-command status bundle and mismatch warnings |
| Coarse trainer refresh | How can a running trainer safely consume a newer assignment? | Dict/Volume contract, collector-boundary hook, tests, race-condition list |
| Refresh red team | What can go wrong in Dict/Volume/env-manager/replay concurrency? | prioritized failure modes and invariants |

## Delegation Rules

- Give each lane exact paths to read.
- Ask each lane to use the plain-language glossary in `README.md` for repair,
  continuation, validation, assignment, and refresh terms.
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

1. Coarse trainer refresh: run-control Dict pointer -> immutable assignment ->
   future collection batch.
2. Refresh red team: race conditions, split-brain subprocess envs, stale loaded
   opponents, and replay metadata.
3. Public leaderboard snapshot/pointer contract.
4. Assignment selector and audit contract.
5. Trainer `--opponent-assignment-ref` plumbing.
6. One-frame tournament evaluator validation.
7. Scheduler fairness and repair tests.
8. Seeded roster/scripted policy representation.
9. Intake continuation/idempotency.
10. Optimizer-safe settings for next manifest.
11. Modal/debug tooling for manifest/config/latest/progress summaries.

## Latest Critique Notes

- Mutable refresh lane: use a bounded control-plane poll, not env-step or
  episode polling. Coach materializes immutable assignments; trainer reads only
  the pending assignment pointer and applies before `Collector.collect`.
- The highest refresh risks are partial Volume/Dict writes, stale generations,
  multiple attempts racing on one run id, subprocess env split brain, mixed
  replay batches, and reused slot names keeping an old frozen opponent object.
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

## Refactor Direction

Keep the decorated Modal functions as thin adapters. Move pure logic first.

Safest extractions:

- Intake/discovery contract into `checkpoint_intake_service.py` plus a focused
  checkpoint-discovery helper. Keep Dict/Queue writes and `.spawn(...)` calls in
  the Modal file.
- Rating artifact I/O into a `rating_artifact_store` helper. Keep rating math
  and scheduler algorithms where they are until the top-100 active-pool behavior
  has more production evidence.
- Public leaderboard publication planning into a small helper. Modal should
  read the rating snapshot, write Volume files, commit, then move the Dict
  pointer.

Do not refactor GIF execution, policy loading, Modal app/image/Volume globals,
or first-class scripted tournament players yet.

## Website Direction

The tournament browser should stop doing heavy reads in the `/` route. The
smallest path is a fast shell plus cached JSON panels:

- `/` renders selectors, progress shell, and empty panel containers.
- standings, checkpoint battles, battle detail, and GIF samples load through
  JSON routes only when selected.
- cache keys should include cheap file-stat tokens, not broad Volume reloads.
- battle detail should read only the requested game/GIF page plus one extra row,
  not walk all shard summaries just to open a panel.
- copy the GIF browser's safe reload/read pattern before increasing web
  concurrency;
- version GIF URLs when using long immutable cache headers;
- progress polling should detect a changed snapshot token, not only a progress
  completion flag.

## Subscriber Direction

Implemented now:

- `intake-seed` writes the service manifest.
- `tournament-submit` / `intake-submit` only add checkpoint/run candidates.
- `intake-tick` rescans live run IDs or run prefixes; explicit refs stay frozen.
- `intake-drain` claims before spawning, repairs missing queue events from
  queued refs, blocks existing rating runs unless `continue_from_latest=True`,
  and spawns the rating loop.

Still missing proof or behavior:

- Remote proof that a live watch finds later checkpoints on `curvyzero-runs`.
- Remote proof for queue-loss repair, stale-claim repair, and continuation from
  existing `latest.json`.
- Remote proof that child game/rating workers survive launch lifetime: use
  `modal run --detach`, a deployed scheduled path that keeps the work alive, or
  wait for spawned work to finish. Check `latest.json` and game summaries, not
  only scheduling/progress files.
- Full active-key reconciliation by scanning durable Volume manifests. Explicit
  tournament/rating calls can now repair one missing Dict manifest from Volume.
- Remote proof that the rating config uses the full live watch pool when the
  manifest has already discovered many checkpoint players.

Simple durable contract:

- Volume manifest is truth.
- Modal Dict active keys and Modal Queue events are cache/wakeup state.
- Continuing ratings must use the full `seen_checkpoint_refs` pool.
- Background child workers need a live or detached parent; non-detached
  `modal run` can stop them when the local entrypoint exits.
