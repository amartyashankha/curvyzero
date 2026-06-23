# Policy Perspective Uniformity Investigation

Last updated: 2026-05-16.

## Question

We need to prove what the raw policy observation means when a policy controls
physical player 0 versus physical player 1.

The current user-facing concern is sharper than just "do we randomize seats."
The raw tensor handed to a policy must speak one stable language:
`self`, `other`, and neutral objects. The policy should not have to infer that
"green means me when I am player 0 but red means me when I am player 1," or any
similar physical-seat convention.

The intended invariant is simple:

- both physical seats should see the same observation language;
- the controlled player should be rendered as self;
- the opponent should be rendered as other;
- bonuses and neutral game objects should not depend on seat;
- changing seats should not require the policy to learn a different color
  convention;
- tournament evaluation and training must use the same rule.

This may be only a trail-color swap over the same global board. If any other
seat-dependent transform exists, it must be explicit and tested.

## Non-Goals

- Do not require the board to be camera-centered or mirrored unless the engine
  already does that. A global board is acceptable.
- Do not conflate seat randomization with perspective uniformity. Seat
  randomization chooses which physical player the learner controls; perspective
  uniformity controls what the raw policy tensor means.
- Do not allow tournament eval to silently reinterpret a checkpoint that was
  trained on a different observation surface.

## What To Inspect

- Game engine state: whether player ids affect rules, bonuses, deaths, scoring,
  or action semantics beyond controlling a different physical player.
- Rendering: whether `controlled_player=0` and `controlled_player=1` differ only
  by self/other palette on player-specific objects.
- Training env: what tensor reaches LightZero for each `learner_seat_mode`.
- Tournament eval: what tensor each checkpoint receives at physical seat 0 or 1.
- Checkpoint metadata: whether each checkpoint records enough observation-surface
  data for tournament loading to fail closed on incompatible observations.

## Proof We Need

- A fixed game state rendered from player 0 view and player 1 view should have:
  same shape, dtype, and neutral object pixels.
- Self/other trail pixels should swap roles exactly as documented.
- The batched two-seat renderer should match the direct controlled-player
  renderer for both seats.
- Training fixed-player-0 and fixed-player-1 observations should match the same
  renderer contract.
- Training random-seat mode should produce observations matching the selected
  controlled player, not a hard-coded physical seat.
- Tournament seat 0 and seat 1 policy inputs should match the same renderer
  contract.
- If a checkpoint lacks compatible observation metadata, tournament eval should
  refuse to run it rather than silently guessing.

## Findings

The model-facing gray64 policy tensor uses controlled-player view. That is not
the same as fixed physical player-zero view. It is also not camera-centered:
the current policy tensor keeps the global board and swaps player-owned
`self`/`other` encoding by controlled player.

Audits and focused tests now prove:

- the active renderer applies no rotation, reflection, crop, translation, or
  rule change for player 0 versus player 1 views;
- player-owned trails/heads swap `self`/`other` encoding by controlled player;
- bonuses are neutral in the gray64 policy tensor;
- trainer fixed-seat and random-seat paths use the selected controlled-player
  stack;
- tournament eval feeds seat `N` the stack at `observation[0, N]`;
- tournament policy loading fails closed on missing or incompatible policy
  surface/backend/contract/perspective metadata.

Remaining follow-up gaps are listed in `POLICY_OBSERVATION_CONTRACT.md`.

## Delegation Plan

- Engine/render audit: inspect source-state rendering and identify all
  seat-dependent pixel changes.
- Trainer audit: trace reset, learner seat selection, stack update, and exact
  tensor handed to LightZero.
- Tournament audit: trace checkpoint metadata validation, per-seat stack output,
  policy call input, and action writeback.
- Test-design audit: propose the smallest robust raw-observation tests and one
  synthetic end-to-end proof.

## Immediate Local Plan

- Add or confirm low-level renderer parity tests.
- Add or confirm trainer fixed-seat and random-seat parity tests.
- Add or confirm tournament policy-input parity tests.
- Tighten checkpoint observation metadata validation so incompatible policy
  surfaces fail closed.
- Fold proven facts into `POLICY_OBSERVATION_CONTRACT.md` and leave any
  remaining unknowns visible here.

## Status

Focused proof complete for the gray64 model-facing tensor. Keep this document
open for the follow-up gaps: raw RGB accessor semantics, frozen-opponent
provider metadata validation, trainer telemetry fields, blank-canvas metadata
wording, and any future GPU renderer parity work.
