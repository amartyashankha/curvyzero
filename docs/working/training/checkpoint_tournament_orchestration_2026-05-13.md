# Checkpoint Tournament Orchestration, 2026-05-13

## Purpose

This is the planning/delegation/orchestration lane for the CurvyTron checkpoint
tournament work. It is deliberately separate from implementation notes,
validation logs, and long research docs.

North star: build an ongoing checkpoint rating and inspection system that can
absorb every useful checkpoint without all-pairs compute. The output should help
the coach identify strong policies, inspect representative GIFs, and eventually
feed a public leaderboard that future training can sample for frozen opponents.

## Operating Pattern

Main thread owns decisions, sequencing, launches, and final synthesis. Subagents
own bounded research, critique, and audit lanes. The main thread should not use
subagents as a substitute for reading the active ledger before acting.

Before any launch or code edit:

- Read this file, `checkpoint_tournament_active_threads_2026-05-13.md`, and
  `checkpoint_tournament_todo_2026-05-13.md`.
- Check the latest validation/evidence docs for completed runs and known
  failures.
- Confirm checkpoint discovery is broad and fresh.
- Decide whether the work belongs in the main thread or a subagent lane.
- Keep code changes out of this docs lane unless the user explicitly asks for
  implementation.

After any launch or decision:

- Update the active-thread ledger first.
- Record evidence, not vibes: arena id, rating run id, game count, battle count,
  failure count, website/GIF checks, and known limitations.
- Keep old launch commands marked historical when the target has moved.

## Main Thread Vs Subagents

Keep in the main thread:

- deciding the next launch;
- editing orchestration, todo, active-thread, and source-of-truth docs;
- making small scoped implementation changes when requested;
- running final tests, Modal smoke checks, deploys, and launch commands;
- reconciling conflicting advice from subagents.

Delegate to subagents:

- scheduler/Elo critique and simulation design;
- website scale and UX/API contract review;
- Modal ops failure-mode review;
- checkpoint discovery audits;
- refactor boundary critiques;
- docs synthesis when multiple notes disagree.

Do not delegate:

- final launch approval;
- public leaderboard readiness calls;
- edits that cross code ownership boundaries without a main-thread plan.

## Active Lanes

| Lane | State | Main-thread action |
| --- | --- | --- |
| Orchestration | active | keep this doc, active threads, and todo aligned |
| Scheduler/adaptive Elo | active | enforce bounded placement and breadth before rank claims |
| Checkpoint discovery | active guardrail | use broad `train/lightzero_exp*/ckpt/iteration_*.pth.tar` discovery |
| Website sanity | active | verify indexed/paged read paths before larger GIF-heavy runs |
| Modal intake/ops | V0 batch launcher | keep Dict/Queue for coordination and Volume artifacts as truth |
| Validation | active | record exact run ids, counts, failure counts, and website checks |
| Public leaderboard | future lane | do not expose as trusted until evidence breadth/status fields exist |
| Refactor | opportunistic | only small cuts that preserve the public Modal facade |

## Current Scheduler State

The current target is all-checkpoint adaptive Elo, not latest-only all-pairs.
`adaptive_v0` is batch-wave scheduling over immutable battle artifacts:

- placement/low-coverage checkpoints get scheduled before ordinary near-rating
  or random bridge work;
- after placement coverage, spend extra battles with a smooth bias toward
  higher Elo/rank policies, especially top-10/top-20 candidates, because public
  leaderboard and training consumers care most about reliable top policies;
- do not make that bias a hard cutoff or starve low-Elo policies before they
  have sufficient games and distinct opponents;
- placement is still the first gate. A new or undercovered checkpoint must get
  enough distinct-opponent evidence before top-band polish can consume the
  round budget;
- live website progress must read shard summaries while a round is running, not
  the stale round-start progress artifact;
- live website progress must also handle per-game workers. A run with
  `games_per_shard=1` writes game summaries but no shard summaries, so the
  cheap website path uses pair-directory estimates unless an explicit diagnostic
  exact count is requested;
- the website should avoid direct large scans in normal request handlers. Prefer
  small cached artifacts plus background refresh jobs, then page/lazy-load the
  growing views;
- pair specs carry scheduler metadata such as `pair_key` and
  `schedule_reason`;
- pair history and scheduler state are persisted between rounds;
- rating context and roster identity must reject unsafe evidence reuse;
- Volume artifacts are durable truth; Modal Dict/Queue are coordination tools.

Current evidence from the expanded probe:

- `arena-curvytron-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`
  / `elo-top20runs-allckpts-placement-gpp21-gifs5-step8000-20260513a`;
- 424 checkpoints, 212 placement battles, 4452 games;
- `failed_game_count=0`;
- zero zero-game checkpoints;
- checked website paths served rankings, checkpoint drilldown, battle detail,
  and GIF samples.

Important limitation: one placement battle per checkpoint is not leaderboard
quality. It proves plumbing and coverage, not stable policy strength.

## Next Launch Plan

Next launch should be chosen from two options, not improvised:

1. Website/detail sanity lane.
   - Preferred if the next work is scale safety.
   - Add or verify lighter battle-detail/GIF sample artifacts and paged game
     rows.
   - Confirm checkpoint drilldown and battle detail do not scan huge global
     indexes or live shard/game files in normal request paths.

2. Modest adaptive breadth wave.
   - Preferred if the next work is rating evidence.
   - Use broad checkpoint discovery or an explicit broad-discovery manifest.
   - Keep `games_per_pair=21`, GIFs capped or off unless this is explicitly a
     visual inspection wave.
   - Schedule enough placement/anchor/bridge pairs to increase distinct
     opponents, not merely total games.

Do not launch a large all-checkpoint public leaderboard wave until website
read paths, status fields, and evidence breadth are ready.

## Website Sanity Lane

Website sanity is now a first-class lane because the product value is inspection,
not just a finished JSON artifact.

Required checks before scaling:

- tournaments list loads the intended active run and hides broken zero-checkpoint
  attempts;
- rankings show status/freshness and do not overstate provisional Elo;
- checkpoint drilldown uses per-checkpoint indexes;
- battle detail returns summary and GIF samples before any large game table;
- `/gif?ref=...` serves sampled GIFs with correct content type;
- refresh/reload failures are visible enough for operators to avoid stale data.

Current known gap: battle detail and large game/GIF views still need stronger
paging/index work before very large GIF-heavy runs.

## Public Leaderboard Future Lane

The leaderboard is a future training contract, not just a scoreboard. Training
loops may later sample frozen opponents from it, so public rows must carry enough
evidence for safe consumption.

Minimum readiness shape:

- rank/Elo plus `provisional` or `active` status;
- games, distinct opponents, outside-lineage opponents, failure rate, draw or
  timeout rate, and freshness;
- discovery provenance for the checkpoint ref;
- scheduler/evaluator context id;
- clear warning when evidence comes from one-opponent or low-breadth placement.

Current gate: target at least 20 distinct opponents per checkpoint before
calling a row leaderboard-active. Use bounded adaptive waves to reach breadth;
do not fall back to full N^2 all-pairs unless explicitly requested as a stress
test.

## Checkpoint Discovery Footgun

Never build tournament manifests from `train/lightzero_exp/ckpt` alone.
DI-engine can create timestamped experiment directories after restart, such as:

```text
train/lightzero_exp_260513_123802/ckpt
```

Tournament discovery must scan:

```text
train/lightzero_exp*/ckpt/iteration_*.pth.tar
```

This applies to:

- latest-checkpoint visual canaries;
- all-checkpoint adaptive runs;
- explicit manifests passed into intake;
- future public leaderboard rows;
- any frozen-opponent training consumer that samples from leaderboard output.

If a manifest was built from fixed-path trainer status, treat it as stale until
rebuilt or verified with broad discovery.

## Guardrails

- Do not edit code in the docs/orchestration lane.
- Do not describe plumbing canaries as policy-strength evidence.
- Do not treat GIF-off smokes as website/GIF validation.
- Do not allow a checkpoint with zero games or one opponent to look trusted.
- Do not launch full all-pairs by habit.
- Do not move `run_checkpoint_game`, policy loading, GIF writing, or public
  Modal wrappers during cleanup unless a separate implementation plan says so.
