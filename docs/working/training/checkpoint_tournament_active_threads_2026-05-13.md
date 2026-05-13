# Checkpoint Tournament Active Threads, 2026-05-13

## Purpose

Keep the active plays visible. This is the short ledger; detailed thinking lives
in the goal, orchestration, scheduling, critique, and validation docs.

## Current Main Target

Build all-checkpoint adaptive Elo for CurvyTron.

Plain version: every useful checkpoint can enter the pool, but we do not run
all-pairs. New checkpoints get placement games, strong ones move up, uncertain
or important matchups get more games, and the coach can watch progress on the
website.

## Active Threads

| Thread | Owner | Status | Next Gate |
| --- | --- | --- | --- |
| Orchestration | main | active | keep this ledger and orchestration doc current |
| Rating research | Gibbs / Arendt | first pass done | turn critique items into tests and scheduler rules |
| Code architecture | Pauli / main | helper and artifact wiring landed | adaptive remote smoke |
| Checkpoint discovery | Arendt / Pauli / main | footgun guard landed | use `checkpoint_selection=all` in adaptive runbook |
| Modal ops | Hilbert | first pass done | add commit/reload retry only if needed; keep Volume as truth |
| Website scale | Lorentz | first pass done | add paged/indexed click paths before very large runs |
| Validation | main | focused tests passing | run focused tests after each patch; smoke remote only after wiring |
| Product/coach view | main + critique agents | active | ensure rankings show status, evidence, and freshness |
| Refactor architecture | Arendt / Mill / Hilbert / Lorentz / main | first critique pass returned | make Cut 1: name implicit contracts, then rerun focused tests |

## Current Evidence

- Pure `adaptive_v0` scheduler helpers exist and are tested.
- Pair specs can carry `pair_key`, `schedule_reason`, and `schedule`.
- Pair history is keyed by canonical sorted checkpoint ids.
- Rating rounds now write `pair_history.json` and `scheduler_state.json`.
- Discovery scans `train/lightzero_exp*/ckpt/iteration_*.pth.tar`.
- Discovery supports `latest`, `iteration`, and `all`.
- Focused local test result: `78 passed, 10 skipped`.
- Remote discovery smoke passed: `found_count=3`, `missing_count=0`, and two
  of the three returned refs were under timestamped `lightzero_exp_*` dirs.
- Refactor critique lane returned a consistent answer: keep the single Modal
  app entrypoint, keep public wrappers stable, and first promote implicit
  strings/paths into named contracts before moving code.
- The two largest risk files remain
  `src/curvyzero/tournament/curvytron_checkpoint_tournament.py` and
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`.

## Current Not Done

- Website does not yet have per-checkpoint battle indexes for huge runs.
- Tiny adaptive rating smoke has not run yet.
- Cleanup refactor Cut 1 has not landed yet.
- No large all-checkpoint adaptive run should launch yet.

## Keep Spinning

- Keep critique lanes alive when design changes.
- Keep code changes small and test them in parallel.
- Keep docs updated immediately after evidence changes.
- Do not let the trainer footgun consume this lane; record it and protect
  tournament discovery, but leave trainer repair to the coach/optimizer lane.
- Refactor only in safe slices. Each cut should preserve the Modal app entrypoint
  and keep focused tests green.
- First cleanup slice should be boring: name contract strings and paths only.
  Do not move `run_checkpoint_game` or website route registration yet.
