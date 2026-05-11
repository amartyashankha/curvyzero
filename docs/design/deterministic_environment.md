# Deterministic Environment

Status: Draft

This page describes the intended simulator contract. Details should be updated as CurvyTron reference behavior is inspected and v0 rules are chosen.

## Scope For v0

- `v0-choice`: 1v1 only.
- `v0-choice`: one round equals one episode.
- `v0-choice`: no bonuses or powerups.
- `v0-choice`: solid trails unless and until source-derived trail gaps are implemented.
- `v0-choice`: fixed rectangular arena, per-tick speed, turn rate, trail width, and tick duration.
- `wrapper-choice`: a trainer `step` accepts one public action id per live player, then
  maps those ids to held source-style controls. Native CurvyTron uses control state over
  elapsed-ms server frames, not a source-level `joint_action`.
- `wrapper-choice`: strict public env info records `native_control_model_id`,
  `trainer_control_wrapper_id`, and `decision_ms` so replay/profile consumers know
  the wrapper cadence.
- `v0-choice`: terminal reward from round outcome.
- `v0-choice`: readable Python reference implementation first, with fixed-shape NumPy-friendly state.
- `v0-choice`: occupancy-grid collision backend first unless source mining or tests force a different choice.

## API Shape

```python
env = CurvyTronEnv(config)
obs = env.reset(seed=123)
result = env.step({"player_0": 0, "player_1": 1})
```

This is the CurvyZero trainer-wrapper API, not the native CurvyTron source API.
`step` must be deterministic for the same seed, initial state, config, and wrapper action
sequence. It is a fixed decision wrapper over source-style control state, not native
discrete simultaneous actions.

## Code Location

The implementation lives under `src/curvyzero/env/` so the simulator can be a stable project subsystem rather than a top-level one-off package.

## Rule Provenance Labels

Every important rule should be labeled in implementation docs:

- `source-derived` - copied or matched from CurvyTron source or observed gameplay.
- `v0-choice` - deliberately invented for the first training environment.
- `unresolved` - known gap that must not be treated as final.

## Source-Derived CurvyTron v1 Facts

These facts were promoted from `docs/research/curvytron_reference_notes.md` after local
inspection of `third_party/curvytron-reference`.

| Area | Source-derived behavior |
| --- | --- |
| Timing | Server loop targets 60 Hz, but physics uses elapsed wall-clock milliseconds per frame. Round warmup is 3000 ms and warmdown is 5000 ms. After `game:start`, trail printing starts another 3000 ms later. |
| Arena | Map size is `round(sqrt(80^2 + ((players - 1) * 80^2 / 5)))`. |
| Movement | Base velocity is 16 units/s. Angular velocity base is `2.8 / 1000` radians/ms. Position integration is continuous: `x += cos(angle) * velocity / 1000 * step_ms`, `y += sin(angle) * velocity / 1000 * step_ms`. |
| Avatar body | Base radius is 0.6. Velocity is clamped to at least 8 units/s. Self-collision ignores bodies until the point number delta is greater than trail latency 3. |
| Controls | Default client keys are left/right arrows. Left-only maps to `-1`, right-only maps to `1`, neither or both maps to `0`; inverse controls flip the sign. |
| Trail gaps | Trail and hole lengths are distance-based. Base print distance is 60 and base hole distance is 5, with randomized multipliers in the reference `PrintManager`. |
| Collision | Server update order is motion, border check, trail/body collision, then print-manager and bonus catch. Body collision is strict circle overlap: distance must be less than the sum of radii. Equal distance is safe. Borderless mode wraps; normal border collision kills. |
| Scoring | Death score is based on the death count captured at frame start, so same-frame deaths share the same round score. The last alive avatar gets `max(players - 1, 1)` extra round score. A game winner must be a unique leader at or above max score. |
| Bonuses | Default `bonusRate` is 0. Base bonus radius is 3, default duration is 5000 ms, spawn cap is 20, and base pop time is 3000 ms adjusted by bonus rate. |

`CurvyTronConfig.reference_defaults` exposes a subset of these constants for tests and
future source-fidelity work. The current `curvyzero-v0` simulator does not consume that
metadata directly, and changing it should not change trajectories or the rules hash.

## Current v0 Deviations

- `curvyzero-v0` uses a 64x64 grid by default rather than the source map-size formula.
- `curvyzero-v0` uses speed `1.0` per physics tick and turn rate `0.08` radians per tick,
  not source elapsed-millisecond integration.
- `curvyzero-v0` rasterizes occupied cells and line segments, not source circular body
  islands or strict endpoint-circle collision.
- `curvyzero-v0` starts with occupied spawn cells and solid trails; it has no warmup,
  warmdown, delayed print start, distance-based holes, borderless wrap, scoring ladder,
  or bonuses.

## Early Golden Tests

- Same seed produces same initial state.
- Same action sequence produces same trajectory and terminal result.
- Wall collision.
- Self trail collision.
- Opponent trail collision.
- Head/head or simultaneous collision.
- Tie reward behavior.
- Vectorized stepping equals individual stepping.

## Performance Constraints

- Keep state arrays fixed-shape where practical.
- Separate rule semantics from collision backend details.
- Design `reset_many` and `step_many` after the single-env reference is correct.
- Keep hot-loop code friendly to future Numba acceleration: simple numeric arrays, explicit loops where needed, minimal Python object churn.
- Defer JAX-native, PyTorch tensor, and native-extension environments until benchmarks show the reference path is insufficient.

## Open Questions

- Should `curvyzero-v0` keep the explicit straight action, or model the source client more
  literally as left/right buttons where neither or both resolves to no turn?
- Should a future source-fidelity ruleset use reference elapsed-millisecond integration or a
  deterministic fixed `1000 / 60` ms step?
- Should source-fidelity collision reproduce endpoint-circle island lookup exactly, or use a
  deterministic swept approximation as a separate ruleset?
- How closely should v0 match original CurvyTron v1 versus CurvyTron 2 public rules?
