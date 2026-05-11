# ADR-0001: Investigation-First Repository Structure

Date: 2026-05-08
Status: Proposed

## Context

The handoff recommends a fresh ML repository with the CurvyTron browser game kept as a reference dependency. The user also asked for aggressive parallel research, critique, and documentation because the early structure decisions are important and easy to forget.

## Decision

Use this workspace as the main CurvyZero ML/investigation repository. Keep original CurvyTron code under `third_party/curvytron-reference` for rule mining, provenance, and possible demo work. Put stable implementation code at the repo root, and put uncertainty into `docs/research`, `docs/experiments`, and `docs/working` until it deserves promotion.

## Evidence

- The CurvyTron browser repo is useful as source material but not shaped like a fast training simulator.
- The simulator, training loop, Modal jobs, and experiments need different dependencies and performance constraints than the web game.
- The first weeks are dominated by reversible decisions and evidence gathering, so the docs tree must support both concise summaries and messy lower-level notes.

## Consequences

- Engineers can work on the simulator without dragging browser/server assumptions into the hot loop.
- Research notes can be written quickly without pretending they are final.
- The project must periodically promote or prune working notes, or the docs tree will become stale.

## Reversal Conditions

- If the CurvyTron source proves easy to isolate into a deterministic headless simulator with strong throughput, reconsider using it more directly.
- If a separate repo boundary becomes necessary for licensing, release, or deployment, split `third_party/curvytron-reference` or the ML package out later.

## Links

- `curvytron_muzero_modal_handoff.md`
- `docs/research/`

