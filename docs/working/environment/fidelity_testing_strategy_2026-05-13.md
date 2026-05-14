# Fidelity Testing Strategy - 2026-05-13

Status: concise proof strategy, not a completion claim.
Owner surface: Environment.

Use a proof ladder. A gap closes only for the surface actually tested.

## Proof Ladder

1. JS/source evidence: first establish the original CurvyTron behavior.
2. `CurvyTronSourceEnv`: mirror that behavior in the source-shaped Python
   oracle.
3. `vector_runtime`: prove the fast kernel preserves the same state transition.
4. `VectorMultiplayerEnv` product route: prove reset/step/lifecycle/reward
   through the public runtime, not only a kernel fixture.
5. Source-state visual renderer: prove the intended product image path, such
   as source-state RGB -> gray64 stack, separately from state transition.
6. Trainer, replay, and final observations: prove the wrapper-facing surfaces
   that training and analysis actually consume.
7. Downstream LightZero parity: after repo-owned target rows are stable, compare
   real `MuZeroGameBuffer` sampled reward, value, policy, action, mask,
   observation, and `to_play`. This is not primary environment-fidelity proof.

## Closure Rule

Visual, metadata, speed, and Modal checks are useful, but none of them proves
full environment fidelity by itself. A source-state visual pass closes only the
visual surface tested. A metadata row closes only the metadata field tested. A
speed or Modal route check closes only reachability/performance evidence unless
it also asserts source-backed behavior on the product route.

## Current P0 Order

The short queue lives in
[current_queue_2026-05-13.md](current_queue_2026-05-13.md). Current top proof
targets are:

1. Broader multiplayer lifecycle and presence/leave parity beyond the promoted
   2P/3P/4P focused rows.
2. Controls tail proof: touch/gamepad input and real browser/Socket.IO evidence
   only if product/eval routes need it. This is not a trainer blocker.
3. Bonus breadth: retry/RNG stress, timer/random ordering, public
   metadata/replay, and stack/death combinations beyond the focused
   `BonusSelfFast` and 4P `BonusEnemySlow` terminal cases.
4. Collision breadth beyond the promoted hit-owner stress shapes.
5. Renderer/fast-path boundaries, continuous-trail and bonus-sprite regression,
   trainer/replay/final-observation propagation, and later browser/canvas pixel
   parity.
