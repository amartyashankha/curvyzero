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
| Scripted policies | wall-avoidant, fixed-straight | Not yet | Yes | Tournament needs general player specs first. |
| Blank/no-op control | `blank_canvas_noop` | Diagnostic only | Yes | Good survival-pressure baseline. |
| Passive/immortal control | fixed-straight + immortal | Diagnostic only | Yes, small share | Dirty control; not source-faithful. |
| Invincible wrapper | death-immunity modifier around another opponent | No for official Elo | Yes, if explicit | It changes scoring semantics. |

## Near-Term Seeding Plan

For the next clean checkpoint-only leaderboard:

1. Include top latest-212 leaderboard checkpoints.
2. Include top survival-eval checkpoints.
3. Include a few anchors from older/preserved cohorts.
4. Include failure/collapse sentinels only if labeled diagnostic.
5. Do not include invincible wrappers in official Elo.
6. Do not include scripted policies until the tournament runner supports
   non-checkpoint players.

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

- neural checkpoints with exact immutable refs;
- possibly non-invincible scripted policies after general player specs exist.

Excluded now:

- invincible wrappers;
- passive immortal rows unless explicitly labeled as diagnostic;
- hidden runtime modifiers.

### Diagnostic Pressure Pool

Participants or entries may be unfair because the goal is pressure or inspection.

Allowed:

- blank/no-op;
- passive immortal;
- invincible variants;
- scripted wall-avoidant policies;
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
- Should scripted wall-avoidant policies be rated in the same Elo pool as neural checkpoints?
- Should passive immortal controls be excluded from all official score pools?
- How many old anchors are needed to stabilize rating drift?
- Should top leaderboard checkpoints be chosen by rank, by diversity, or by both?
