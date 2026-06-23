# Tournament Seeding And Anchors

## Purpose

A clean future tournament should not blindly include every available artifact or
only the latest checkpoints. The seed roster should be explicit: which policies
are official competitors, which are anchors, and which are diagnostics.

## Recommended Roster Classes

| Class | Examples | Official leaderboard? | Training assignment? | Notes |
| --- | --- | --- | --- | --- |
| Current champions | latest-212 top rows | Yes | Yes | Best source of current game-strength opponents. |
| Survival champions | high survival eval rows | Yes | Yes | Good objective-progress anchors. |
| Historical anchors | prior known-good checkpoints | Yes | Yes | Calibrate drift across cohorts. |
| Failure sentinels | collapse/weak checkpoints | Maybe | Maybe | Useful for diagnostics, not promotion. |
| Scripted sentinels | wall-avoidant, fixed-straight | No for official checkpoint Elo | Yes | Training-pressure sentinels are immortal by contract. |
| Blank/no-op control | `blank_canvas_noop` | Diagnostic only | Yes | Immortal inert baseline; useful survival pressure. |
| Immortal checkpoint slice | death-immunity overlay around a frozen checkpoint | No for official Elo | Yes, if explicit and small | Use duplicated mixture entries; keep total immortal pressure bounded. |

## Near-Term Seeding Plan

For the next clean checkpoint-only leaderboard:

1. Include top latest-212 leaderboard checkpoints.
2. Include top survival-eval checkpoints.
3. Include a few anchors from older/preserved cohorts.
4. Include failure/collapse sentinels only if labeled diagnostic.
5. Do not include immortal wrappers or scripted sentinels in official checkpoint
   Elo.
6. Use scripted/blank sentinels only in training assignments or diagnostic pools,
   with explicit `opponent_immortal=true`.

## May 14 Live Tournament Seed Decision

Use the latest-212 public leaderboard as seed material:

```text
tournaments/curvytron/leaderboards/curvytron-latest212-smoke-20260513/snapshots/latest212-smoke-20260513.json
```

Take the top 100 active rows by rank and extract exact `checkpoint_ref` values.

Important caveat:

- This old leaderboard is evidence for choosing seed players, not the new public
  one-frame leaderboard truth.
- The 100 checkpoint players must be re-rated under the current one-frame
  tournament settings before publication.
- Do not use live Dict pointers, browser state, provisional snapshots, CSV
  projections, or materialized tiny assignment files as the 100-player source.

## Candidate Neural Seeds

From the latest-212 leaderboard:

- rank 1: `mix2 r50-blank25-scr25 repH sim8/c32/b32`;
- strong survival blank-medium/blank-heavy rows;
- strong survival passive-light/passive-medium rows as dirty-control candidates;
- strong mix3 blank-containing `repH` rows.

From survival eval:

- survival `blank light` rows with high survival;
- survival `blank medium` rows with strong leaderboard center;
- mix `blank100` and `r50-blank50` rows as non-survival-family anchors.

## Official Pool vs Diagnostic Pool

Use two pools.

### Official Score Pool

Participants must be fair under first-death scoring.

Allowed now:

- neural checkpoints with exact immutable refs.

Excluded now:

- immortal wrappers;
- scripted/blank sentinel rows unless explicitly labeled as diagnostic;
- hidden runtime modifiers.

### Diagnostic Pressure Pool

Participants or entries may be unfair because the goal is pressure or inspection.

Allowed:

- immortal blank/no-op sentinels;
- immortal fixed-straight or wall-avoidant sentinels;
- small explicit immortal slices of frozen checkpoints;
- failure sentinels.

These should not be merged into the official Elo pool without explicit context
labels.

## Seeding Metadata

Each seed row should record:

- participant id;
- participant kind (`checkpoint`, `scripted_policy`, `diagnostic_control`);
- label;
- source run id, if neural;
- exact checkpoint ref, if neural;
- iteration;
- source evidence (`leaderboard_top`, `survival_top`, `anchor`, `diagnostic`);
- evaluator context;
- training eligibility;
- leaderboard eligibility.

## Open Design Questions

- Should the first one-frame public leaderboard be checkpoint-only?
- Should scripted wall-avoidant sentinels remain training-only forever, or get a
  clearly separate diagnostic rating pool?
- Should immortal controls be excluded from all official score pools?
- How many old anchors are needed to stabilize rating drift?
- Should top leaderboard checkpoints be chosen by rank, by diversity, or by both?
