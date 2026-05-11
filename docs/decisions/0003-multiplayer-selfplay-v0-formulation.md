# ADR-0003: Use Ego-Perspective Shared-Policy Self-Play For v0

Date: 2026-05-08
Status: Proposed

## Context

CurvyTron can be multiplayer and is not automatically a simple two-player zero-sum board game. AlphaZero and MuZero provide useful search/self-play patterns, but they do not directly solve n-player general-sum training without formulation choices.

## Decision

For v0, train from an ego perspective with a shared policy/value model and scalar centered rank payoff. Start 1v1. When scaling beyond 1v1, rotate the ego seat and use policy-only or checkpoint-pool opponents before considering all-player search. Do not start with joint-action search or vector-valued general-sum backups.

## Evidence

- `docs/research/multiplayer_selfplay_muzero.md`
- `docs/research/baseline_learnability.md`
- `curvytron_muzero_modal_handoff.md`

## Consequences

- The observation encoder should support ego-relative views.
- Replay records need to identify ego player, opponent policy/checkpoint, ruleset, and reward convention.
- MCTS can initially search only the ego player's action while opponents are sampled or policy-controlled.
- The system can still grow to checkpoint pools or league training later.

## Reversal Conditions

- If 3+ player training fails due to non-stationarity or strategic cycling, consider richer opponent populations, league training, or general-sum search/value formulations.
- If joint-action search becomes tractable for small player counts and clearly improves learning, introduce it behind a separate experiment gate.

