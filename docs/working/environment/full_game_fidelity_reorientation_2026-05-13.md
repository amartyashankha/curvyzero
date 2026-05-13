# Full-Game Fidelity Reorientation - 2026-05-13

Status: working memory, not a completion claim.
Owner surface: Environment.

This note is the current simple map for full-game CurvyTron fidelity work. The
target is `VectorMultiplayerEnv`. Older trainer-only, no-bonus, scalar, or
fixture-only paths can stay useful as proof surfaces, but they are not the
destination.

Process memory:
[operating_patterns_2026-05-13.md](operating_patterns_2026-05-13.md) and
[orchestration_patterns_2026-05-13.md](orchestration_patterns_2026-05-13.md).
Concise proof strategy:
[fidelity_testing_strategy_2026-05-13.md](fidelity_testing_strategy_2026-05-13.md).
Controls fidelity audit:
[controls_fidelity_audit_2026-05-13.md](controls_fidelity_audit_2026-05-13.md).

## Current Read

- We are reorienting around one source-faithful multiplayer runtime:
  `VectorMultiplayerEnv`.
- The current 2P source-state visual gate is useful, but it is not
  trainer/replay proof. It checks the source-state image path, not the whole
  product trace that a trainer, replay writer, and final-observation reader
  need to survive.
- 3P/4P coverage is mostly metadata-backed or fixture-backed today. Treat it as
  narrow lifecycle evidence, not broad multiplayer fidelity.
- The main risk is no longer naming the runtime. The main risk is proving the
  whole 2P path end to end, then widening the same source-backed discipline to
  multi-body, lifecycle, bonus, replay, final-observation, trainer-observation,
  and render-mode boundaries.
- Bonus probability semantics are now reconciled for source-default type
  selection: JS `BaseBonus.getProbability()` makes every non-`BonusGameClear`
  default bonus effectively weight `1`. Remaining bonus work is broader
  spawn/retry/RNG and long-run public stress, not the basic default weights.

## Render-Mode Boundary

- `browser_lines` is the default source-state visual render mode for the
  product image path.
- `body_circles_fast` is an approximate RGB-to-gray path. It is useful for
  speed and profiling, but it is not the default fidelity lane.
- `fast_gray64_direct` is an approximate direct-gray path in self-play plumbing.
  It should stay labeled as an approximation, not as the source-state visual
  gate.

These modes are source-state/native rendering boundaries. None of them proves
real browser canvas pixel parity unless a separate browser/canvas harness says
so.

## Prioritized Missing Feature Map

1. End-to-end 2P product trace.
   Direct `VectorMultiplayerEnv` proof now covers one 2P trace through raw RGB
   -> gray64, seeded clear, terminal wall death, final masks, and metadata
   replay. LightZero-facing wrapper proof now covers scalar joint-action
   decoding, raw RGB -> gray64 stack, held source frames, terminal final
   observation, rewards, masks, and native sidecars. Remaining work is
   warmdown/match breadth and replay arrays beyond metadata.
2. Multi-body hit owner order.
   Runtime now follows source-style hit owner lookup through corner islands and
   newest-first body scan. Stress tests now cover 4P newest-owner overlap, 4P
   corner island order, 3P own-body latency, and 4P two-victim metadata.
   Remaining work is raw JS fixtures and product/replay/debug-event propagation
   for those exact cases.
3. Broader lifecycle.
   Extend beyond the narrow warmup/warmdown/leave/metadata slices into ordinary
   and edge lifecycle rows, including 3P/4P behavior.
4. Bonus stack/death stress.
   Default probability/type selection and one public natural-spawn boundary are
   now pinned. Keep focused bonus support, but add source-backed overlap,
   expiry, clear, borderless, invincibility, wall/body death, and same-tick
   interaction cases.
5. Replay and final observations.
   Promote replay/final rows beyond metadata-only audit facts so bonus,
   lifecycle, RNG, terminal, and observation state are preserved without
   overclaiming browser replay.
6. Trainer observation surfaces.
   Prove the trainer gets the intended source-state visual frames, gray64
   stacks, final observations, masks, rewards, and sidecar facts through the
   actual wrapper path.
7. Render-mode boundaries.
   Keep default, approximate, and self-play render paths named separately.
   Avoid using an approximate render-mode pass as evidence for the product
   source-state visual gate.

## Working Rule

Call a gap closed only when there is source/original evidence and a matching
`VectorMultiplayerEnv` product-route proof. A source-state visual pass, a
metadata replay row, or a fixture-only 3P/4P case is useful evidence, but each
one covers only the surface it actually exercises.
