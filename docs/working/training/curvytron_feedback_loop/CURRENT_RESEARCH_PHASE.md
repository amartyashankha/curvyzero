# Current Research Phase

Last updated: 2026-05-16.

## Purpose

We are not changing tournament production behavior yet. We are figuring out how
the tournament should choose games when many checkpoints arrive.

Plain goal:

1. New checkpoints arrive from training runs.
2. The tournament gives them enough useful games quickly.
3. Strong policies rise into the top pool.
4. Weak policies stop consuming much work.
5. The trainer can trust the public leaderboard as an opponent source.

The hard part is game selection. Full all-pairs is simple but grows too fast.
For `N` checkpoints it needs `N * (N - 1) / 2` battles. At 21 games per battle,
that becomes millions of games once we include many checkpoints from many runs.

## Current Truth

- The current scheduler has an `adaptive_v0` lane.
- Live intake defaults have been moved away from unbounded all-pairs.
- The normal active pool limit is intended to be 100.
- New rows should get broad placement evidence before they look trusted.
- The current code still exposes too many engine-room controls in places.
- The two biggest code files are still too large and need staged cleanup.
- Deployed status/control tooling is the operator truth. Modal Volume artifacts
  are durable backing evidence for debugging. Dicts and Queues are coordination
  tools.

## North Star

Build a bounded online tournament service:

```text
checkpoint appears
-> intake records it
-> scheduler chooses useful games
-> Modal runs games in parallel
-> ratings update
-> public leaderboard publishes top usable policies
-> trainer samples from that leaderboard
```

The service should decide the scheduler details. Normal CZ26 live callers
should use bounded adaptive scheduling and should not need to choose all-pairs,
random, pair counts, or raw internal game-batch knobs.

## What We Need To Learn

- How many opponents does a new checkpoint need before it can be trusted?
- Which opponents should it play first: top policies, nearby policies, anchors,
  same-run neighbors, random bridges, or a mixture?
- How do we avoid a lucky early top policy becoming the opponent for everyone?
- How do we avoid dropping strong new policies during a large burst?
- How do we detect non-transitive policies that beat the leader but lose to
  weaker-looking policies?
- How much top-100 quality do we lose under bounded scheduling versus all-pairs?
- How should the scheduler behave when 20, 100, 500, or 1000 new policies arrive?
- What should the trainer-facing leaderboard hide, mark, or publish while rows
  are still under-tested?

## External Grounding

These are references, not instructions to copy blindly:

- Elo: ratings infer relative strength from results against rated opponents, and
  the K-factor controls how much ratings move after over- or under-performing
  expected score. This supports keeping batch Elo as a simple baseline.
  Source: https://en.wikipedia.org/wiki/Elo_rating_system
- Glicko: rating reliability is explicit through rating deviation (RD), and
  rating periods can update many players in parallel. This maps well to Modal
  batches and supports adding uncertainty beside Elo.
  Source: https://www.glicko.net/glicko/glicko.pdf
- TrueSkill: tracks uncertainty, models draws, and generalizes beyond two
  players/teams. This is useful later, but not required for the first V1
  scheduler.
  Source: https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system-2/
- Swiss pairings: useful ideas are no early repeats, similar-score/rating
  pairings, balance constraints, and transparent explanations. CurvyTron differs
  because games are cheap and parallel, so we can use stronger active placement.
  Source: https://handbook.fide.com/chapter/C0401Till2026

## What We Might Learn

- Current `adaptive_v0` may be good enough with a few caps and clearer metrics.
- We may need a two-stage placement: broad anchors first, then local refinement.
- We may need explicit uncertainty, not just Elo and game count.
- We may need a provisional mini-pool for large bursts so new strong policies
  play each other before top-100 truncation.
- We may need separate wave intents: breadth, top refinement, audit, and intake.
- We may find all-pairs is still useful as a small audit, but not as the normal
  service contract.

## Active Lanes

| Lane | Owner | Status | Output |
| --- | --- | --- | --- |
| Current code truth | Kepler | reported | What scheduler/rating/intake does today. |
| Scheduling research | Fermat | reported | Candidate strategies and failure modes. |
| Simulation design | Erdos | reported once | Simulation plan and metrics. |
| Docs/orchestration | Chandrasekhar | reported once | Doc structure and phase gates. |
| Multi-wave burst critique | Zeno | reported | Wave math and coverage risk. |
| Simulation gates | Arendt | reported | False-drop metrics and pass/fail gates. |
| Main thread | Codex | active | Integrate notes, run toy sims, keep docs current. |

## Follow-Up Register

This map starts rough and should change as the research teaches us more.

| ID | Source | Owner | Status | Question | Next Action | What It Could Change |
| --- | --- | --- | --- | --- | --- | --- |
| FU-001 | user | main | active | Does current `adaptive_v0` keep large bursts bounded? | Run multi-wave current-code coverage probe. | Add hard max, coverage warnings, or preflight fail. |
| FU-002 | Erdos | main/Erdos | active | Which scheduler family wins in toy sims? | Compare random, top-anchor, spread-anchor, binary ladder, current-code probes. | Pick first V1 scheduler candidate. |
| FU-003 | Fermat | Fermat | reported | What does Elo theory support? | Fold recommendation matrix into candidate scheduler doc. | Add uncertainty/RD and match-quality fields. |
| FU-004 | Chandrasekhar | main | active | Are docs becoming another archive swamp? | Keep this hub current and do not create extra dated folders. | Archive old notes instead of duplicating them. |
| FU-005 | user | main/Zeno | active | Can we safely truncate to top 100? | Simulate burst arrivals, false drops, and grace-pool designs. | Add provisional protection or grace windows. |
| FU-006 | user | Arendt | active | What pass/fail gates prove the scheduler is sane? | Return simple metric gates for simulations. | Make scheduler changes test-driven. |
| FU-007 | EXP-003 | main | active | Is a flat 300-pair wave too slow for burst placement? | Simulate burst-aware budgets and provisional grace pool. | Add burst mode or visible under-placement status. |
| FU-008 | EXP-004 | main | active | Does the local online loop protect the top-100 contract? | Keep the new online simulation tests in the focused gate. | Prevent scheduler changes from silently reintroducing tail churn. |
| FU-009 | Einstein | main | active-18-proven | Where does Modal still bottleneck after uncapping hot workers? | Next probe should aggregate reload/commit timings; completed progress now reads `latest.json` first. | Decide whether to remove double per-game summary writes or reduce commits. |

## What We Might Learn And What Changes

| Possible Lesson | What We Do Next |
| --- | --- |
| Current `adaptive_v0` is good enough with bounded waves. | Keep it as V1 spine; add readouts, caps, and tests. |
| Current `adaptive_v0` is too slow to touch every new checkpoint. | Add burst-specific coverage mode or larger first-touch wave. |
| Current `adaptive_v0` touches everyone but too slowly reaches 20 opponents. | Keep rows provisional longer, increase burst placement budget, or schedule new-vs-new provisional pools. |
| Established-first placement is the bottleneck. | Add a burst mode that mixes established anchors with new-vs-new placement so each pair can cover two new rows when safe. |
| Top-anchor placement protects strong new policies. | Keep top challenges, but cap anchor appearances. |
| Top-anchor placement creates narrow evidence. | Use spread anchors and random bridges before status promotion. |
| Binary ladder works in clean worlds but fails under noise. | Keep it as a diagnostic, not the default scheduler. |
| Random coverage beats clever placement under heavy noise. | Increase random bridge / breadth quota. |
| Non-transitive worlds break scalar Elo. | Add top-band cross-style audits and non-transitivity reports. |
| Top-100 truncation drops strong new policies too early. | Add provisional grace pool and delay retirement until evidence gates pass. |
| Rating uncertainty gives cleaner decisions. | Add Glicko/RD-like confidence and conservative score. |
| Existing docs or UI hide scheduler risk. | Publish budget expansion, coverage, anchor concentration, and status reasons. |
| Modal still feels slow after uncapping workers. | Measure Volume reload/commit/scans separately; do not guess from website symptoms. |

## Current Gates

We should not promote a new scheduler until these are true:

- It passes local toy simulations for transitive, noisy, burst, lineage-heavy,
  non-transitive, timeout-heavy, and top-100-boundary worlds.
- It reports budget expansion before spawning Modal work.
- It tracks distinct opponents and outside-lineage opponents.
- It keeps top-100 recall high under a fixed game budget.
- It avoids one anchor receiving most placement games.
- It can explain every scheduled pair with a simple reason.
- It updates rankings as games arrive, without waiting for a perfect full round.

## Latest Local Gate

Added on 2026-05-16:

- `tests/test_curvytron_tournament_online_simulation.py`

This file now proves three basic service behaviors without Modal:

- weak new entrants can get placement games, lose, fall below rank 100, and be
  excluded from later scheduling;
- a clone/draw swarm has zero Elo movement and deterministically publishes 100
  active rows plus a retired tail;
- `1000 established + 50 new` stays bounded, touches every new row, and avoids
  old rank-101+ rows.

This does not prove real CurvyTron games or Modal throughput. It is the cheap
gate before spending remote work.

## Latest Remote Gate

Added on 2026-05-16:

- `curvy-scale-probe-18latest-gamefanout-20260516a`
- `elo-scale-probe-18latest-gamefanout-20260516a`

This run proved the current game-level Modal fanout path at small live scale:

- 18 latest checkpoints;
- 153 battles;
- 3,213 games;
- `games_per_shard=1`;
- `save_gif=false`;
- ratings written;
- failed games 0.

The important bug found during validation was not the games. The bug was status:
completed progress could scan every game summary even though `latest.json`
already proved the round was complete. The fix is to read the completed snapshot
first, then only scan summaries while a round is still live.
