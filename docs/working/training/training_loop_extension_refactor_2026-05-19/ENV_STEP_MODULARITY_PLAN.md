# Env Step Modularity Plan

Last updated: 2026-05-19

## Why This Exists

The source-state env is the place where policy perspective, opponent action, game stepping, reward, observation, and telemetry meet. It needs clearer internal boundaries before we add more training behavior.

## Known Locations

- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/vector_visual_observation.py`
- `src/curvyzero/env/vector_multiplayer_env.py`

## Target Shape

Keep the public env API stable, but split internal helpers for:

- learner-seat selection,
- policy-perspective normalization,
- opponent action execution,
- reward computation,
- observation rendering,
- telemetry extraction.

## Next Investigation Tasks

- Map reset and step flow.
- Identify where player zero/player one perspective is chosen.
- Identify whether observation color/perspective normalization is centralized.
- Identify where opponent immortality is applied.
- Identify focused tests for observation and reward parity.

## Done Criteria

- Reset and step are easier to read.
- Policy perspective has one documented contract.
- Reward and observation helpers can be tested without a full training run.
- Tournament eval and training can share the same observation contract intentionally.
