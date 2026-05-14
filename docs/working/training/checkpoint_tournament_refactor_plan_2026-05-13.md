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

Cleanup wave 2 has been delegated while Cut 1 is being finished:

- Arendt: pure/domain module boundaries.
- Mill: data contracts, artifact refs, Volume paths, and context mixing risk.
- Pauli: validation and semantic-risk review of the current Cut 1 diff.
- Hilbert: Modal ops, Volume/cache, and thin-wrapper cut order.
- Lorentz: website/API/cache/paging cut order.
- Gibbs: adaptive Elo/scheduler architecture and coach-misleading risks.

Returned notes:

- `checkpoint_tournament_refactor_domain_arendt_2026-05-13.md`
- `checkpoint_tournament_refactor_contracts_mill_2026-05-13.md`
- `checkpoint_tournament_refactor_modal_hilbert_2026-05-13.md`
- `checkpoint_tournament_refactor_website_lorentz_2026-05-13.md`
- `checkpoint_tournament_refactor_scheduler_gibbs_2026-05-13.md`

Durable consensus:

- Keep the existing public files as facades first. Move behavior behind them
  only in small slices.
- Durable truth stays in Volume artifacts. Caches, queues, and live scheduler
  priority are not truth.
- Split `pool_hash` into two ideas before serious adaptive reuse:
  checkpoint roster identity and evaluator/rating context identity.
- Website normal paths should read small committed artifacts. Shard/game scans
  are recovery paths, not the product path.
- The first extraction should be contracts, then scoring/specs/reducer/scheduler
  only after each cut has focused tests.
- The first website index cut has landed locally. Per-checkpoint battle indexes
  live at
  `tournaments/curvytron/<tournament_id>/checkpoints/<checkpoint_id>/battle_index.json`.

Status update: Cut 1 is landed and focused tests pass.

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

### Cut 1A: Extract The Pure Contract Surface

Status: landed locally and tested.

Added:

```text
src/curvyzero/tournament/curvytron/contracts.py
src/curvyzero/tournament/curvytron/__init__.py
```

The old public module,
`src/curvyzero/tournament/curvytron_checkpoint_tournament.py`, still re-exports
the moved names. Existing Modal code and tests still import
`curvytron_checkpoint_tournament as arena`.

Moved in this cut:

- schema ids;
- selection names;
- artifact filenames;
- default tournament/rating constants;
- safe id/hash helpers;
- checkpoint id and pair key helpers;
- tournament/rating/game artifact ref helpers;
- artifact ref validation.

Not moved:

- `rating_context_hash`, because it depends on rating spec normalization;
- pair/game/rating spec builders;
- scoring;
- policy loading and game execution.

Validation:

- compile check passed;
- facade compatibility check passed for representative `arena.*` names;
- focused tournament tests passed after the later roster guard:
  `90 passed, 10 skipped`.

### Cut 2: Extract Discovery

Move checkpoint discovery helpers from the Modal app into a pure module, likely:

`src/curvyzero/tournament/checkpoint_discovery.py`

Keep the Modal function as a thin wrapper that calls the pure helper.

Tests already target local tmp paths, so this is a good first extraction.

Updated ordering note: extraction should start with domain contracts if the
current file remains confusing after Cut 1. Discovery extraction is still a good
early cut, but it should reuse the named checkpoint selection and scan-glob
contracts rather than inventing another contract surface.

### Cut 2A: Add Rating Context Identity

This is a small behavior guard before serious adaptive Elo reuse.

Add a pure helper:

`rating_context_hash(spec)`

It should cover evaluator semantics, not checkpoint roster:

- policy mode;
- max steps;
- decision timing and source-frame settings;
- number of simulations;
- natural bonus setting and env/reward compatibility fields;
- scoring formula/version;
- observation/render contract fields that affect policy inputs.

Use it to prevent `pair_history.json` and `scheduler_state.json` from being
reused across different tournament/evaluator meanings. Roster expansion should
not by itself erase old pair evidence, but evaluator context drift must not
silently mix evidence.

Status: landed locally and tested. Previous snapshots, pair history, and
scheduler state now reject changed evaluator context.

Follow-up guard also landed:

- rating artifacts now include `checkpoint_roster`;
- roster expansion can still carry old pair evidence;
- if the same checkpoint id points at a different checkpoint ref, previous
  pair history, scheduler state, and previous snapshots are rejected.

### Cut 2B: Per-Checkpoint Battle Indexes

Status: landed locally and tested.

The website now has a bounded checkpoint-drilldown path:

```text
tournaments/curvytron/<tournament_id>/checkpoints/<checkpoint_id>/battle_index.json
```

The global `battle_index.json` still exists. The per-checkpoint file is read
first when a checkpoint is selected. Global-index filtering and live shard scans
remain fallbacks for old artifacts and recovery cases.

Follow-up tightening also landed:

- checkpoint index files carry their own `ref`;
- checkpoint index rows preserve `pair_key`, `schedule_reason`,
  `schedule_priority`, `scheduled_round_index`, and `schedule`;
- checkpoint-specific index reads still filter rows defensively;
- checkpoint drilldown no longer live-scans shard summaries when the
  checkpoint index is present.

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
