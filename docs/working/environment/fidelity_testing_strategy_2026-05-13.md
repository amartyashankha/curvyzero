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

1. Controls source-frame fidelity.
2. End-to-end 2P product route through `VectorMultiplayerEnv`.
3. Bonus probability and source-default type selection.
4. Hit-owner ordering.
5. Wider multiplayer, including 3P/4P lifecycle, bonus stack/death stress,
   replay/final observations, trainer observation propagation, and later
   browser/canvas pixel parity.
