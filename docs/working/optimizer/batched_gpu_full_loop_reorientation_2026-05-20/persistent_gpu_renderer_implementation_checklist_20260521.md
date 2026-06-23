# Persistent GPU Renderer Implementation Checklist

Date: 2026-05-21

Scope: docs-only next-step checklist for the real profile-only persistent
policy-space GPU renderer. Do not treat this as trainer, tournament, or Coach
launch advice.

## Backend Label

- Add the real renderer behind an explicit profile-only backend label, for
  example `jax_gpu_persistent_policy_framebuffer_profile`.
- Keep this label separate from `cpu_oracle`, scalar `jax_gpu`,
  `block_704_gray64`, and `direct_gray64`.
- Report the selected label in every profile artifact, together with the
  profile run id and renderer telemetry.
- If the implementation changes observation semantics beyond the current
  policy-space surface, give it a new surface label instead of reusing the
  persistent-framebuffer label.

## Default Rules

- Do not flip trainer defaults.
- Do not flip tournament defaults.
- Do not change checkpoint metadata or loader expectations.
- Do not promote scalar `policy_observation_backend=jax_gpu`.
- Keep live training/runs untouched; this lane is profile-only until the full
  evidence gate passes.

## Expected Evidence

- Modal/profile-only rows comparing persistent update rendering against the
  current stateless dynamic GPU redraw ladder.
- End-to-end profile numbers, not renderer-only numbers: wall time, renderer
  update time, readback, stack update, scalar materialization if present, dirty
  row count, rebuild row count, fallback count, active trail count, and payload
  bytes when measured.
- A direct comparison against the current CPU dirty-cache shape and the current
  dynamic full-redraw GPU shape for the same no-death/normal-death profile
  pattern.
- Explicit proof that the row/player order, stack order, terminal
  `final_observation`, partial autoreset, and RND latest-frame extraction
  contracts were exercised.
- Clear artifact language that any speedup is profile-only until stock
  LightZero search/replay/learner/RND/checkpoint/eval paths are separately
  measured.

## Minimal Tests

- Unit/parity tests against a stateless reference on asymmetric rows:
  diagonal trails, overlapping owners, radius changes, break-before gaps,
  bonus overlap, head overlap, reset, cursor regression, prefix mutation, game
  clear, and map-size change.
- Row/player perspective tests for both controlled players, including
  row-major request order and partial row/player requests.
- Terminal/autoreset tests that verify `final_observation` is rendered from
  the pre-autoreset state and reset observations from the post-autoreset state.
- Draw-order tests: background, trails, bonuses, heads; bonus symbols remain
  distinguishable at policy resolution and heads draw last.
- Telemetry tests that fail if rebuild/fallback counters, backend label, or
  hidden-fallback flags are missing from the profile output.

## Fallback And Fail-Closed Rules

- Never silently fall back to CPU rendering or stateless GPU redraw.
- On reset, cursor regression, prefix mutation, map-size change, palette or
  avatar-color change, game clear, unsupported bonus mode, or unsupported
  request shape, either rebuild affected row caches with explicit telemetry or
  fail the profile row.
- If fallback is intentionally allowed for one profile experiment, record the
  reason, affected rows, count, and time cost; do not mix that row with clean
  speed claims.
- If semantic parity fails for row/player view, terminal observation, stack
  order, draw order, or RND latest-frame extraction, stop the lane and keep the
  renderer out of trainer/tournament recommendations.
- If the profile artifact cannot prove the backend label and no-hidden-fallback
  status, treat the row as invalid.

## Promotion Gate

Recommend this renderer to Coach only after it is faster end-to-end in the
profile boundary, passes the semantic tests above, reports complete telemetry,
and has an explicit metadata plan that tournament/checkpoint paths can honor.
