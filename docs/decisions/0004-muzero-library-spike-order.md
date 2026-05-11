# ADR-0004: Spike Mctx First, Keep LightZero Contained

Date: 2026-05-08
Status: Proposed

## Context

CurvyZero needs MuZero-style search eventually, but the simulator API should not be owned by a training library. Research compared JAX/Mctx and PyTorch/LightZero as early candidates.

## Decision

Run the first MuZero/search spike with JAX/Mctx, starting with a synthetic `gumbel_muzero_policy` benchmark on Modal. Keep LightZero as a contained PyTorch fallback spike with an ego-agent wrapper and clear rejection criteria. Do not let either framework define the core simulator interface.

## Evidence

- `docs/research/mctx_integration.md`
- `docs/research/lightzero_integration.md`
- `docs/research/muzero_architecture_deep_dive.md`
- `docs/research/performance_vectorization.md`

## Consequences

- The first MCTS benchmark should use fixed action count `A=3`, fixed batch sizes, fixed simulation counts, and compile-time separated from steady-state runtime.
- The real simulator stays outside the Mctx recurrent function; Mctx searches learned latent dynamics.
- LightZero integration should prove value through measured wall-clock performance, not convenience alone.

## Reversal Conditions

- If Mctx integration is too slow, brittle, or costly on Modal, promote the LightZero/PyTorch spike or another library into contention.
- If the baseline path suggests policy-only training is enough for a while, defer MuZero entirely without changing the simulator API.

