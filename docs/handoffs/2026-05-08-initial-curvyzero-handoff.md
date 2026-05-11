# 2026-05-08 Initial CurvyZero Handoff

## Current State

- Repository initialized in `/Users/shankha/curvy`.
- CurvyTron reference repo cloned locally at `third_party/curvytron-reference` and gitignored; provenance tracked in `docs/sources/curvytron_reference.md`.
- Main code uses `src/curvyzero/` hierarchy.
- Initial deterministic simulator exists at `src/curvyzero/env/`.
- Initial Modal entry points exist at `src/curvyzero/infra/modal/smoke.py`.
- Documentation hierarchy is established under `docs/`.
- Research notes cover CurvyTron source mining, Modal patterns, performance/vectorization, baselines, MuZero, Mctx, LightZero, multiplayer self-play, observations/rewards, hierarchy, and user-message memory review.
- Local tests and Ruff pass.
- Modal remote pytest, CPU benchmark, and GPU smoke have run successfully.

## Accepted/Proposed Decisions

- ADR-0001: use an investigation-first ML repo with CurvyTron as reference.
- ADR-0002: keep Modal Queues/Dicts/network primitives out of per-step/MCTS hot loops.
- ADR-0003: use ego-perspective shared-policy self-play for v0.
- ADR-0004: spike JAX/Mctx first, keep LightZero contained as a fallback.

All ADRs are currently `Proposed`; promote to `Accepted` once the user wants the direction frozen.

## Verification

Local:

```sh
uv run --extra dev pytest
uv run --extra dev ruff check .
```

Latest result:

- `pytest`: 9 passed.
- `ruff`: all checks passed.

Modal:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind tests
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind benchmark --episodes 25 --max-steps 500
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind gpu
```

Recorded results:

- Remote tests: Modal app run `ap-88LM0Y5FYWAmDc4omylcFt`, `4 passed in 0.47s`.
- CPU benchmark: Modal app run `ap-HhifGZ9XakXcGBhs0A98e8`, `575` steps, about `8192` steps/sec.
- GPU smoke: Modal app run `ap-xak6TKbcJ2C3Pgr90fE8XW`, `NVIDIA L4, 23034 MiB, 580.95.05`.

## Key Source-Derived CurvyTron Facts

- Server targets 60 Hz but integrates elapsed milliseconds.
- Warmup/warmdown are `3000`/`5000` ms.
- Trail printing starts `3000` ms after `game:start`.
- Reference map size formula is `round(sqrt(80^2 + ((players - 1) * 80^2 / 5)))`.
- Avatar velocity is `16`, angular base is `2.8 / 1000` rad/ms, radius is `0.6`, and self-collision latency is `3`.
- Trail gaps are distance-based: print `60`, hole `5`.
- Collision is strict circle overlap after movement, not swept geometry.
- Same-frame deaths share frame-start score.

## Important Caveats

- Current `curvyzero-v0` simulator is deliberately simplified and does not yet implement source-faithful timing, collisions, trail gaps, scoring, or bonuses.
- The first environment tests are scaffolding, not proof of rule fidelity.
- Modal should be used early for smokes and benchmarks, but not as a microservice boundary inside the game/search loop.
- Research notes are evidence, not current truth. Promote stable conclusions into `docs/design/` or `docs/decisions/`.

## Next Actions

1. Implement a source-aware `curvyzero-v0` rules config and decide which reference constants should become actual simulator defaults.
2. Replace the placeholder observation vector with ego-centric ray observations and add schema hashes.
3. Add deterministic golden tests for source-mined collision/scoring/trail-gap cases.
4. Add `step_many`/`reset_many` only after single-env semantics are better pinned.
5. Add Modal Volume-backed artifact paths before larger benchmark/training runs.
6. Implement the synthetic Mctx benchmark as the first GPU search experiment.

