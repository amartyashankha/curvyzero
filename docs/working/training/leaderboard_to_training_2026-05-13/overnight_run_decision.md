# Overnight Run Decision Surface

## Inputs

- Survival/leaderboard synthesis:
  `docs/working/training/curvytron_architecture_research_2026-05-12/fair_comparison_212_run_investigation_2026-05-13.md`
- Tournament/intake state:
  `docs/working/training/checkpoint_tournament_intake_runbook_2026-05-13.md`
- Trainer cadence state:
  `docs/working/training/lightzero_train_refactor_2026-05-13/current_source_of_truth.md`
- Optimizer lane:
  `docs/working/optimizer/README.md`

## Decision Split

There are four different goals. Do not collapse them.

| Goal | Primary evidence | Recommended use |
| --- | --- | --- |
| Improve survival objective | survival eval curves | choose training objective/knobs |
| Top leaderboard | latest-212 Elo | choose champions/opponents |
| Generate frozen opponents | leaderboard + diversity | build assignment snapshot |
| Improve throughput | optimizer docs/profiles | choose render/batch/collector only if semantics stay fixed |

## Current Recommended Training Defaults

```text
objective: survival_plus_bonus_no_outcome
sim: 8
batch: 32
collector: 32 default, 64 probe
render: paired browser/fast for key cells
survival stochasticity: medium + light core, heavy bounded
mix repeat: repM anchor + repH upside, rep0 control
```

## Candidate Blocks

| Block | Rows | Purpose |
| --- | ---: | --- |
| Survival medium blank | high | leaderboard center and stable survival signal |
| Survival light blank | medium | strongest survival-gain point estimate |
| Survival heavy blank | low-medium | top-tail/stress lane |
| Survival steady blank | low | control |
| Mix blank-containing `repH` | medium | leaderboard-upside/champion pressure |
| Mix blank-containing `repM` | medium | stable mix anchor |
| Collector64 sentinels | low | throughput/progress probe |
| Sim16 sentinels | very low | only if needed for search sensitivity |
| Batch64 sentinels | avoid | current evidence negative |

## Frozen Opponent Pool Candidates

Use leaderboard winners plus diversity, not raw survival winners only.

Candidate types:

- top survival medium/blank checkpoints;
- top survival heavy/blank checkpoints;
- latest-212 rank 1 mix2 `r50-blank25-scr25 repH`;
- strong mix3 blank-containing `repH` rows;
- a small passive survival set as dirty/control opponents;
- optional scripted/blank sentinels outside the leaderboard.

## Optimizer Interaction

Optimizer speed settings are orthogonal until they change:

- observation semantics;
- decision cadence;
- reward semantics;
- tournament/eval contract;
- checkpoint compatibility.

Use optimizer recommendations only after checking these invariants. Faster
browser render can justify more browser rows, but not if it changes the input
contract.

## Launch Preconditions

For plain static-manifest overnight:

- manifest uses current one-frame cadence;
- no stale `decision_ms`;
- checkpoint refs are exact immutable `iteration_N.pth.tar`;
- broad checkpoint discovery used for frozen refs;
- run names encode key knobs.

For leaderboard-derived overnight:

- public leaderboard snapshot exists or manually curated equivalent exists;
- assignment snapshot is immutable and hashable;
- trainer can consume assignment ref or static manifest mirrors assignment;
- audit records source leaderboard/rating rows.

## Current Practical Recommendation

If launching soon before full leaderboard-to-training wiring:

1. Use a static manifest that mirrors the intended assignment snapshot.
2. Include leaderboard-derived checkpoint opponents explicitly.
3. Include scripted/blank/passive entries only if they are already supported by
   current trainer/env flags.
4. Do not wait for full online Elo unless the launch goal is specifically to
   test that infrastructure.
