# Checkpoint Tournament Goal And Plan, 2026-05-13

## Goal

Build a small CurvyTron checkpoint tournament lane. It should compare policies
from saved checkpoints by making them play the real two-player game against each
other.

The first score is simple: who dies first.

The next score is a small Elo ladder. That is not the first thing to trust. The
first thing to trust is the raw battle runner.

## Product Shape

- One Modal app for the whole tournament lane.
- One CPU function that runs one game.
- One function that runs many games for one pair.
- One function that runs many pairs for a tournament.
- One basic website that lists tournaments, battles, scores, and sample GIFs.
- Artifacts live in the shared `curvyzero-runs` Volume under
  `tournaments/curvytron/...`.

## Product Contract

- A game has exactly two checkpoint policies and one shared two-player env.
- The default score is from env terminal info: first dead player loses.
- A battle is N independent seeded games for the same two checkpoints.
- A tournament is a set of checkpoint pairs and battle jobs.
- Game summaries are immutable facts.
- Battle summaries and standings can be recomputed from game summaries.
- Rating/Elo snapshots are derived state and can be thrown away.

## Implementation Plan

1. Add pure tournament helpers under `src/curvyzero/tournament/`. Done.
2. Add one Modal app module under `src/curvyzero/infra/modal/`. Done.
3. Reuse existing checkpoint loading and LightZero policy action code where it is
   safer than writing our own. Done for V0.
4. Use `VectorMultiplayerEnv` directly for real checkpoint-vs-checkpoint games.
   Done.
5. Use the existing source-state render path for model observations and GIFs.
   Done.
6. Add focused tests that do not need Modal or a real checkpoint. Done.
7. Deploy the Modal app. Done once; redeploy after every code change.
8. Run a tiny remote game/pair with real checkpoint refs. Pending.
9. Check the website can list the tournament and serve GIF/JSON artifacts.
   Pending.
10. Add Elo loop after raw battle artifacts are reliable. Planned, not V0.

## Next Remote Smoke

Use two existing checkpoint refs and run:

`uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode pair --tournament-id arena-smoke-YYYYMMDD --checkpoint-refs <ref-a>,<ref-b> --games-per-pair 1 --max-steps 16 --num-simulations 1 --wait`

Expected result:

- one tournament marker
- one battle summary
- one game summary
- one GIF if `--save-gif true`
- website lists the tournament after refresh/reload

## Operating Notes

- Keep this separate from training and live GIF plumbing.
- Do not create a Modal app per battle.
- Do not write shared aggregate files from game workers.
- Game workers write only their own immutable game summary and optional GIF.
- Pair and tournament functions write aggregate summaries.
- Use Modal `.map(..., order_outputs=False)` for parallel work when waiting for
  all child results is fine.
- Use `.spawn()` from the local entrypoint so long jobs can run detached.
- Avoid Volume reloads in hot request paths unless the user asks for freshness.

## Done For V0

- A user can pass two checkpoint refs and run N games.
- The pair summary says seat 0 wins, seat 1 wins, draws, failures.
- At least one GIF can be saved for human inspection.
- The website can list tournament rows and link to JSON/GIFs.

## Not Done Yet

- Real checkpoint smoke has not been proven in this compacted context.
- Website is basic. It is useful but not the final tournament dashboard.
- Huge 300x300x50 round robin may need sharding to avoid too many tiny commits
  and too many files.
- Elo loop is designed but not implemented.
