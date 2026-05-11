# Coach Iteration Process Review - 2026-05-09

Purpose: learn from the coach lane's repeated mistakes and set a simpler loop
for the next Pong/LightZero/CurvyTron iterations.

This note synthesizes local working docs and sanitized local Codex state
metadata. It does not copy private raw logs.

## Sources Reviewed

Main docs:

- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/training_coach_reorientation_2026-05-09.md`
- `docs/working/training_coach_self_critique_2026-05-09.md`
- `docs/working/training_process_critique_2026-05-09.md`
- `docs/working/coach_optimizer_reorientation_2026-05-09.md`
- `docs/working/pong_two_lane_worldview_2026-05-09.md`
- `docs/working/lightzero_reproduction_parity_plan_2026-05-09.md`
- `docs/working/training_lessons_for_curvytron_2026-05-09.md`
- `docs/working/shared_training_reporting_contract_2026-05-09.md`

Local state check:

- Codex state metadata only: thousands of total threads, hundreds in this repo,
  and many spawn edges. This confirms a wide, parallel, memory-pressure-heavy
  process.
- Codex log database: counted only; not used as a content source.
- Recent local instruction history was treated as process signal only. No raw
  user/session text is copied here.

## What We Repeatedly Got Wrong

1. We mixed lanes.
   Official Atari Pong, custom dummy Pong, repo-native PPO, Modal plumbing, Mctx
   probes, CEM baselines, and supervised MLP baselines were often summarized as
   one story. They are different tools answering different questions.

2. We over-claimed plumbing.
   A job running on Modal, a checkpoint loading, or a scorecard executing is an
   infrastructure pass. It is not evidence that a policy learned.

3. We trusted weak or partial metrics.
   Win/loss alone hid useful early signal. Trainer-side telemetry hid held-out
   failures. Fixed-seed evals and `ckpt_best` sometimes sounded stronger than
   they were.

4. We let checkpoint archaeology become the main lane.
   The checkpoint work found real bugs and useful facts, but it kept pulling the
   main thread away from the bigger decision: which lane deserves the next run?

5. We scaled before proving the target was scoreable.
   Several runs asked training to solve a target before we had shown that the
   target, search settings, support scale, and eval setup could expose progress.

6. We used unclear names and abstractions.
   Terms like "MuZero-shaped", "visual Pong", "self-play", and "progress" were
   too loose unless the report said exactly which lane, which game, which
   checkpoint, and which eval.

7. We did not promote worker findings cleanly enough.
   Subagents/workers produced useful probes, critiques, and run results, but the
   main state docs did not always absorb the decision. Old questions returned.

## What Is Working Now

- Lane separation is clearer:
  official Atari Pong is the stock LightZero/ALE control; custom dummy Pong is
  the project-owned diagnostic bridge; repo-native PPO is the simultaneous
  `[B, P]` CurvyTron architecture probe.

- Modal artifacts are becoming real:
  runs, attempts, checkpoint paths, hashes, manifests, and eval outputs are
  showing up in durable places often enough to support repeatable scorecards.

- Independent eval is now treated as mandatory:
  strict load, no fallback, action histograms, raw return, nonzero rewards,
  survival, truncation, and baseline comparisons are the right direction.

- The coach is more honest about non-claims:
  tiny smokes, CEM/MLP baselines, Mctx search tests, and Modal plumbing are being
  labeled as support work instead of training success.

- The target-quality lesson is concrete:
  MuZero trains toward MCTS root visits, not just the action executed during
  collection. Root-target sidecars and fixed-state probes are now first-class.

## Iteration Loop From Here

Use this loop for every next run or doc change:

1. Name the lane.
   Choose one: official Atari Pong, custom dummy Pong, repo-native PPO, Modal
   tooling, Mctx/search, docs/process.

2. State the decision.
   Write one sentence: "This will decide whether..." If there is no decision,
   do not run it.

3. Prove the target is scoreable.
   Before scaling, show that the eval can detect progress and that the target
   labels make sense. For MuZero custom games, root visits must put mass on
   known good actions at normal-enough simulation counts.

4. Run the smallest useful job.
   Use smoke settings for mechanics only. Use normal-enough settings for
   learning claims. Keep these labels separate.

5. Eval independently as soon as checkpoints exist.
   Do not wait for a full training job if checkpoints are already in the
   Volume. Run strict/no-fallback eval on the same cap and same reported schema.

6. Report a scoreboard, not a story.
   Include lane, run id, attempt id, checkpoint, eval cap, strict-load status,
   action histogram, raw return, nonzero rewards, survival, truncation, and the
   continue/stop decision.

7. Update the state docs once.
   Promote the decision into `training_state_index`, the experiment backlog, or
   the relevant lane note. Do not create another front door unless the old one
   is retired or linked.

## How To Use Subagents Better

- Keep the main thread for orchestration:
  lane choice, stop rules, decision summaries, and state-doc updates.

- Give workers narrow jobs:
  one run, one eval, one source audit, one discrepancy table, one doc cleanup, or
  one red-team critique.

- Require worker outputs to end with:
  pass/fail, artifact refs, what changed, what did not change, and the next
  blocked question.

- Use workers while long jobs run:
  eval fresh checkpoints, verify artifact roots, inspect config fidelity,
  compare official-vs-custom assumptions, or update the backlog.

- Kill or ignore stale workers:
  if a worker is no longer tied to an active decision, it should not consume the
  main thread.

## How To Use Docs And Tooling Better

- One current truth doc:
  `training_state_index_2026-05-09.md` should stay the top map for the coach
  lane.

- One run ledger:
  the experiment backlog should say active, stopped, or pending for each lane.

- One report shape:
  use `shared_training_reporting_contract_2026-05-09.md` for both LightZero and
  repo-native reports. Missing fields should be explicit, not omitted.

- Sidecars over prose:
  target visits, action histograms, checkpoint refs, support scale, and eval
  rows should be JSON/JSONL artifacts first, then summarized in docs.

- Tooling should reduce waiting:
  live eval helpers, checkpoint discovery, artifact-root verification, and
  summary table generation are higher value than more manual checkpoint
  archaeology.

## Top Process Rules

1. Do not mix official Atari Pong and custom dummy Pong results.
2. Do not call a run learning progress unless independent eval improves.
3. Do not scale a target until it is scoreable and its labels are sane.
4. Do not use win/loss alone; always report survival and action histograms.
5. Do not let checkpoint analysis become the main lane.
6. Do not treat Modal plumbing as training evidence.
7. Do not trust `ckpt_best` without auditing what it loaded and how it was
   selected.
8. Use subagents for probes and evals; keep the main thread on decisions.
9. Promote every real decision into the state docs once.
10. Say plainly what is not claimed.

## Immediate Next Shape

For the coach lane, the next useful work should be:

- official Atari Pong: bounded reproduction run and strict independent eval
  curve, with normal-enough settings and no fallback;
- custom dummy Pong: fixed-state root-target parity probe before more scale;
- repo-native PPO: keep the simultaneous `[B, P]` actor-loop shape visible with
  scorecards and profiling, but do not claim quality yet;
- docs/tooling: keep state index, backlog, and report contract current while
  workers handle narrow probes.

Simple reset:

```text
lane -> decision -> scoreable target -> smallest useful run -> independent eval
-> scoreboard -> stop/continue -> state-doc update
```
