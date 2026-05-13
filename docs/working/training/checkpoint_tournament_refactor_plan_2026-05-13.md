# Checkpoint Tournament Refactor Plan, 2026-05-13

## Purpose

The tournament lane is becoming too dense. We need clearer module boundaries
before adaptive Elo, website refresh, and all-checkpoint discovery keep adding
more branches.

This is not a big rewrite plan. The rule is small safe cuts, tested after each
cut.

## Current Problem

Two files carry too much:

- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py`
- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`

The pure helper file mixes schemas, artifact refs, spec normalization, game
specs, scoring, GIF helpers, scheduler helpers, pair history, and rating math.

The Modal file mixes deployment image setup, volume helpers, discovery, game
workers, rating orchestration, progress/reduce, website APIs, HTML, CSS, JS, and
cache helpers.

This makes it easy to add one more string or helper in the wrong place and hard
to see the real data flow.

## Layer Contract

Keep these layers separate:

1. **Domain contracts**
   - schema ids
   - constants
   - artifact refs
   - safe ids
   - checkpoint specs
   - pair/game/rating specs

2. **Tournament scoring**
   - game tally
   - pair summary
   - standings
   - Elo snapshot
   - pair history

3. **Adaptive scheduler**
   - pool hash
   - pair key
   - adaptive pair selection
   - scheduler state payload rules

4. **Checkpoint discovery**
   - scan `train/lightzero_exp*/ckpt/iteration_*.pth.tar`
   - latest/all/iteration selection
   - no trainer/poller side effects

5. **Modal execution**
   - Modal image/app/function declarations
   - worker functions
   - orchestration functions
   - volume commits/reloads

6. **Artifact IO**
   - read/write JSON
   - battle index writing
   - progress writing
   - reduce/rebuild helpers

7. **Website API**
   - API payload builders
   - pagination
   - freshness/cache reads
   - no HTML strings here

8. **Website rendering**
   - HTML page render
   - CSS/JS constants
   - small helpers for escaping and table rows

## Data Flow

```text
checkpoint discovery
  -> rating spec
  -> adaptive scheduler / pair specs
  -> game or shard specs
  -> game summaries / shard summaries
  -> pair summaries
  -> battle index
  -> pair history + scheduler state
  -> rating snapshot
  -> progress/latest/provisional artifacts
  -> website APIs
  -> one-page UI
```

Durable truth:

- checkpoint refs in `curvyzero-runs`
- tournament artifacts in `curvyzero-curvytron-tournaments`
- immutable game/shard/battle summaries
- derived rating snapshots and indexes that can be rebuilt

Not truth:

- website cache
- scheduler priority
- Modal function return order
- provisional snapshot if final latest exists

## Safe Refactor Order

### Cut 1: Name The Contracts

Move no code yet. Add named helpers/constants where strings are repeated.

Candidates:

- checkpoint scan glob
- schedule reason names
- rating artifact filenames
- web cache key names

Status: chosen as the next cleanup cut. The first critique pass from Arendt,
Mill, Hilbert, Lorentz, Pauli, and Gibbs all points the same way: name the
implicit contracts before extracting modules.

Include in this cut if it stays small:

- checkpoint selection values: `latest`, `all`, `iteration`;
- checkpoint scan glob;
- adaptive schedule reason names;
- progress phase names that cross Python/JS boundaries;
- orphan artifact refs such as rating-run results and provisional latest.

Do not include in this cut:

- moving Modal decorators;
- moving `run_checkpoint_game`;
- changing Elo math;
- changing website behavior;
- changing checkpoint discovery semantics.

### Cut 2: Extract Discovery

Move checkpoint discovery helpers from the Modal app into a pure module, likely:

`src/curvyzero/tournament/checkpoint_discovery.py`

Keep the Modal function as a thin wrapper that calls the pure helper.

Tests already target local tmp paths, so this is a good first extraction.

### Cut 3: Extract Rating Artifacts

Move pair-history and scheduler-state payload writing into a pure-ish artifact
module, but keep actual Modal volume commits in the Modal app.

Candidate module:

`src/curvyzero/tournament/rating_artifacts.py`

### Cut 4: Extract Website Payload Builders

Move API payload builders away from HTML rendering.

Candidate module:

`src/curvyzero/tournament/web_payloads.py`

### Cut 5: Extract Static Website Assets

Move CSS and JS strings to separate constants or files copied into the Modal
image.

Do this only after tests cover rendering and API behavior well enough.

## What Not To Do Yet

- Do not split the Modal app into many Modal apps.
- Do not move Modal decorators into many modules unless deploy stays simple.
- Do not rewrite the website design.
- Do not replace Volume artifacts with Queue/Dict.
- Do not change scoring math during refactor.

## Acceptance Checks

Every refactor cut must keep:

- focused tournament tests green;
- `all_pairs` and `random` behavior unchanged;
- adaptive scheduler tests green;
- discovery timestamped-dir tests green;
- Modal app import/compile green.

After any Modal wrapper extraction, run a tiny remote discover or rating smoke.

## Critique Round 1 Notes

The sub-agent critiques agreed on these plain contracts:

- Durable truth is Volume artifacts: immutable game/shard/battle summaries,
  rating snapshots, pair history, and scheduler state.
- `pair_history.json` is evidence-derived truth. `scheduler_state.json` is
  scheduling telemetry. Scheduler intent must never count an unfinished battle
  as played.
- Keep one Modal app. The Modal file should later become thin wrappers, but the
  app/function names should stay stable.
- Website request paths should read small indexes and snapshots. They should
  not scan every game summary after the system scales.
- The tournament may be adaptive, but the coach-facing UI must show uncertainty,
  opponent count, failures, and freshness next to rank.

## Open Questions

- Should discovery live in `curvyzero.tournament` or `curvyzero.infra.modal`?
  Current answer: pure path scanning belongs outside Modal; Modal wrapper belongs
  in infra.
- Should website renderer become templates?
  Current answer: not yet. First split payload builders from HTML.
- Should pair history be a separate domain module?
  Current answer: yes eventually, but leave it near Elo until the scheduler
  contract is stable.
