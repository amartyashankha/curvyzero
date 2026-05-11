# Observation Fidelity Plan

Status: working plan for post-state-parity
Date: 2026-05-09

This plan defines how to test observations after source gameplay state is
trusted enough to observe. It is deliberately separate from source gameplay
parity: the original CurvyTron source has browser pixels, canvas rendering, and
wire state, but it does not define learned observations.
It also does not define trainer `step()`, fixed action ids, or `joint_action`;
those are CurvyZero wrapper/schema/replay abstractions over real-time control
state and elapsed-millisecond server frames.

Use this after a scenario already has JS/Python state evidence. If state parity
fails, fix the source-fidelity lane first. An observation failure should mean
the chosen CurvyZero observation schema encoded a trusted state incorrectly, not
that the source rules are still unknown.

Design anchors:

- [observability_plan.md](../../design/environment/observability_plan.md)
- [observation_fidelity.md](../../design/environment/observation_fidelity.md)
- [training_interface_contract.md](../../design/environment/training_interface_contract.md)
- [source_feature_inventory.md](source_feature_inventory.md)
- [fidelity_metrics_options.md](../../research/environment/fidelity_metrics_options.md)

## Boundary

Keep four surfaces separate:

| Surface | Meaning | Compare against | Main failure meaning |
| --- | --- | --- | --- |
| Source gameplay state | JS/Python common trace, events, outcomes, and rules. | CurvyTron JS oracle and Python source-fidelity runner. | Gameplay reconstruction is wrong or unsupported. |
| State observation | Privileged debug or oracle projection of Python state. | The same trusted Python state snapshot. | Debug projection, serialization, or schema drift. |
| Ray/raster observation | Trainer-facing semantic observation derived from simulator state. | Trusted Python state and occupancy semantics, not screenshots. | Observation schema, perspective, channel, mask, or normalization bug. |
| Browser pixels | Render/browser/demo image evidence. | Pinned browser or renderer frame from a state-passing scenario. | Render, viewport, palette, antialiasing, timing, or browser protocol issue. |

Do not promote browser pixels as the first learned observation. Do not use a
ray/raster mismatch to rewrite gameplay rules unless the underlying state
fixture also fails.

## Top Observation Gates

1. State prerequisite: each observation fixture references a scenario and tick
   whose JS/Python common-trace state status is `pass`.
2. Schema gate: observation schema id, schema hash, shape, dtype, channel order,
   scalar order, action order, bounds, and missing-value policy are exact.
3. Purity gate: `observe(ego)` is deterministic, does not mutate state, and
   repeated observation calls produce byte-stable output for the same state.
4. Perspective gate: ego transforms, player order, color, and seat permutations
   do not leak unintended identity or absolute-coordinate shortcuts.
5. Content gate: known movement, trail-gap, borderless, and collision-death
   states produce the expected semantic ray/raster channels and masks.
6. Replay gate: replay rows carry rules, trainer action, reward, observation
   schema hashes, ego id, perspective transform id, legal mask, wrapper
   joint-action metadata, and terminal observation policy.
7. Pixel gate: browser screenshots are compared only after state and semantic
   observation checks pass for the same scenario and frame.

## Observation Families

### State Observation

The current flat global vector is useful as `oracle_global_debug` only. It can
expose absolute positions, player order, alive flags, and tick fraction because
its job is debugging and deterministic inspection, not learned policy input.

Tests should compare:

- Shape, dtype, field order, and schema hash exactly.
- Player ids, alive flags, terminal flags, score fields, and action masks
  exactly.
- Position, heading, elapsed time, and normalized scalar values with the same
  numeric tolerances used by the source state fixture unless the observation
  schema defines a tighter tolerance.
- Canonical JSON or array-byte hash after documented float rounding or
  quantization.

State observations are allowed to be privileged, but they must be labeled as
debug/oracle artifacts in replay and checkpoint metadata.

### Ray Observation

Target schema: `curvyzero_egocentric_rays/v0`. The exact first trainer
contract, including hashes, ray angles, scalar order, legal mask, reward, and
info fields, is pinned in
`trainer_observation_reward_contract_v0_2026-05-09.md`. The older research name
`curvyzero-observe-v0-rays` is a legacy alias and should not be emitted in new
trainer rows.

Expected shape from the current design:

```text
rays        float32[24, 4]
scalars     float32[10]
action_mask bool[3]
```

Ray checks should use simulator-native geometry or occupancy, not rendered
pixels. Exact checks cover schema, action mask, channel order, scalar order,
finite values, bounds, perspective transform id, and hidden-state exclusions.
Numeric checks cover normalized distances and scalar channels.

Record at minimum:

- `ray_count`, ray angle list, and channel names.
- Distance clipping policy and normalization denominator.
- Per-channel max error, mean error, and first mismatching ray.
- Quantized ray hash after clipping and scaling.
- Leak check result for stable player id, color, seat, and absolute position.

Default tolerance should be fixture-defined. For analytic states with no grid
rounding, start at absolute `1e-6`. For occupancy-backed rays, record a cell-size
tolerance such as half a cell after normalization, plus the exact grid and
collision semantics used to derive it.

### Local Raster Observation

Target schema: `curvyzero-observe-v1-local-raster`.

Expected shape from the current design:

```text
planes      float32[5, 48, 48]
scalars     float32[10]
action_mask bool[3]
```

Raster checks should compare semantic planes, not browser colors. For the first
nearest-cell raster, wall/out-of-bounds, own trail, opponent trail, ego head,
and opponent head planes should be exact after quantization. If interpolation,
antialiasing, crop scale, or a history stack is added later, create a new schema
hash and add numeric tolerances for those fields.

Record at minimum:

- Plane names, crop size, cell scale, ego-center cell, and heading convention.
- Out-of-arena encoding rule.
- Occupancy source: world bodies, live heads, trail bodies, or debug grid.
- Plane hashes and full tensor hash after documented quantization.
- Optional ignore masks for crop-edge cells, interpolation boundaries, or
  intentionally unsupported cells.

### Browser Pixel Checks

Browser pixel checks are render fidelity, not source gameplay authority and not
the first learned observation path.

Run them only when:

- The same scenario and frame have passing state evidence.
- The semantic observation check for the corresponding state has passed or is
  explicitly out of scope.
- Viewport, canvas size, device scale factor, frame index, palette, background,
  line width, head radius, source commit, and browser/runtime versions are
  recorded.

Compare pixels with a recorded policy: exact dimensions, per-channel max/mean
absolute error, thresholded pixel count, optional perceptual score, and masks
for browser chrome, antialiasing rims, text overlays, or intentionally dynamic
effects. Pixel thresholds must not hide a state mismatch.

## Fixture Reuse

Add observation fixtures as small manifests that reference existing verified
source scenarios and ticks. Do not fork the gameplay scenarios just to create an
observation test unless the state fixture itself is missing a required state.

Checked in now:

- `scenarios/environment/observation/obs_empty_arena_geometry_v0.json` is an
  analytic canary for `curvyzero_egocentric_rays/v0`, not promoted
  source-backed observation fidelity. `tests/test_trainer_contract.py` loads it
  and pins p0/p1 perspective symmetry in an empty centered arena, a scoped
  no-absolute-position leak check for non-wall rays plus scalars, and the
  borderless wall-channel no-hit rule.
- This canary deliberately says `source_backed_observation_fidelity=false`.
  It references nearby movement and borderless source scenarios only as the next
  promotion targets. It is not browser pixel fidelity, and the original source
  still has no learned observation.
- `scenarios/environment/observation/obs_source_movement_empty_multistep_v0.json`
  is now a small distilled source-state canary. It references
  `scenarios/environment/source_kinematics_straight_multistep.json` and tests
  read that scenario's trusted `comparison.expected.frames` for ticks `0` and
  `3` rather than duplicating source positions in the manifest. The checked
  claims are intentionally narrow: schema id/hash, source fixture reference,
  empty trail channels, p0/p1 non-wall ego-perspective symmetry, decreasing
  forward opponent-head distance as the source players move toward each other,
  and tick scalar encoding.
- The source movement canary is not browser pixel fidelity, not a source
  learned-observation oracle, and not trail/body, lifecycle, spawn, or RNG
  coverage. Keep it as a small guardrail while lifecycle/spawn/RNG remain
  higher-priority training blockers.

Recommended first fixture set:

| Observation fixture | Reused scenario | Checks |
| --- | --- | --- |
| `obs_source_movement_empty_multistep_v0` | `scenarios/environment/source_kinematics_straight_multistep.json` | Landed as a distilled source-state canary for empty-world motion, no trail-channel hits, tick scalar, action mask, and p0/p1 non-wall perspective symmetry. It is not a full observation-fidelity promotion. |
| `obs_movement_turn_perspective` | `source_kinematics_left_turn_step.json` and `source_kinematics_right_turn_step.json` | Heading-relative ray frame, scalar relative heading, left/right action order, no stable player-index leak. |
| `obs_trail_gap_hole_safe` | `source_trail_gap_hole_space_safe_step.json` | Visual hole space must not create a false collision obstacle; seeded world body remains observable only where collision state says it exists. |
| `obs_trail_gap_stored_body` | `source_trail_gap_stored_body_still_kills_step.json` | Old stored body inside a visual gap is still collision-relevant and should appear in semantic trail channels. |
| `obs_trail_gap_boundary_body` | `source_trail_gap_print_to_hole_boundary_kills_step.json` | Print-to-hole boundary point becomes an opponent-trail obstacle in post-step semantic observations; dead-player mask policy is exact. |
| `obs_borderless_wrap` | `source_borderless_wrap_step.json` | Borderless flag or schema-specific boundary rule is visible; post-wrap ego state is encoded without silently applying normal-wall semantics. |
| `obs_collision_same_frame_death` | `source_body_same_frame_point_kills_step.json` and `source_body_same_frame_point_control_safe_step.json` | Same-frame point body, killer/death debug state, terminal/dead-player masks, and live-player final observations are consistent. |
| `obs_normal_wall_terminal` | `source_normal_wall_death_step.json` | Wall danger, wall death, terminal mask, and final observation policy are explicit. |

For each manifest, record whether the observation is taken before reset, after
reset, before a listed step, after a listed step, or from a derived state
snapshot. Avoid implicit "current frame" language.

Exact source-backed manifests still needed before replacing the debug
observation in the actor bridge:

- Broaden the landed `obs_source_movement_empty_multistep_v0` canary only when
  it remains useful after higher-priority lifecycle/spawn/RNG work.
- `obs_movement_turn_perspective`
- `obs_trail_gap_hole_safe`
- `obs_trail_gap_stored_body`
- `obs_trail_gap_boundary_body`
- `obs_borderless_wrap`
- `obs_collision_same_frame_death`
- `obs_normal_wall_terminal`

## Comparison Artifacts

Later implementation can use this layout:

```text
<artifact_root>/<scenario_id>/observations/<schema_id>/
  manifest.json
  expected.json
  actual.json
  diff.json
  hashes.json
  masks/
```

`manifest.json` should include:

- Scenario id, scenario path, batch id, tick or state snapshot id, and ego ids.
- State prerequisite status, state artifact path, common trace schema, Python
  runner id, source target, source commit, ruleset id, and rules hash.
- Observation schema id/hash, observation implementation id/hash, action schema
  id/hash, perspective transform id, and player permutation id.
- Tolerances for every numeric comparison and the source of each tolerance.
- Mask names and meanings: action mask, visibility mask, leak mask, tensor
  ignore mask, and pixel ignore mask when present.
- Quantization policy for hashes, including dtype, byte order, clipping range,
  scale, rounding, and channel order.

`diff.json` should report:

- `status`: `pass`, `fail`, or `blocked`.
- First mismatch: stage, scenario id, tick, ego id, field path, expected, actual,
  tolerance, absolute error, relative error, and mask state.
- Counts by mismatch type: schema, mask, perspective, leak, numeric, hash,
  raster plane, ray channel, state prerequisite, and pixel.
- Worst numeric error per channel or scalar group.

Hashes are regression accelerators, not explanations. A hash mismatch should
also point to the first decoded field mismatch whenever possible.

## Tolerances, Masks, And Hashes

Use exact equality for:

- Schema id/hash, shape, dtype, key names, channel names, scalar names, action
  order, and player/ego ids.
- Boolean masks, alive/dead flags, terminal flags, winner/death fields, and
  hidden-state leak checks.
- Nearest-cell raster planes when the schema says cells are binary and no
  interpolation is used.

Use tolerance for:

- Positions, angles, distances, normalized ray values, normalized scalars, and
  interpolated raster or pixel values.
- Browser pixels, antialiasing boundaries, and device-scale differences.

Record masks explicitly:

- `action_mask`: exact legal-action mask used by policy/search.
- `visibility_mask`: which entities or channels the schema says are visible.
- `leak_mask`: fields that must not encode stable seat, color, player index, or
  uncanonicalized absolute state.
- `tensor_ignore_mask`: raster cells or ray samples ignored because they are
  outside the schema guarantee.
- `pixel_ignore_mask`: screenshot pixels excluded for render-only reasons.

Record hashes explicitly:

- `rules_hash`
- `action_schema_hash`
- `observation_schema_hash`
- `observation_quantized_hash`
- `state_snapshot_hash`
- `raster_plane_hashes` when raster observations are used
- `pixel_hash` only for exact pinned screenshots, not tolerant screenshot diffs

If a tolerance or mask is needed, it belongs in the fixture manifest or
comparison config, never only inside test code.

## Promotion Criteria

An observation fixture can be promoted when:

- Its source scenario has passing state evidence for the referenced tick or
  state snapshot.
- The observation schema id/hash is stable and checked into docs or config.
- The fixture records exact comparison inputs, tolerances, masks, hashes, and
  expected mismatch policy.
- It passes for every ego player named by the fixture.
- It includes at least one perspective or permutation check when the fixture is
  used as learned-observation evidence.
- Failure artifacts are small enough to inspect without opening raw traces
  first.

The first promoted observation bundle should include movement, trail gap,
borderless, and collision death. Pixel checks should not be part of that first
promotion bundle.

## Deferred

- No code, test, or scenario edits are implied by this plan.
- No learned browser-pixel observation until semantic state observations have
  earned their keep.
- No screenshot or video diff as a substitute for source state or event parity.
- No browser client visual gap behavior as gameplay truth; server/world bodies
  remain the collision source.
- No random observation noise, color randomization, crop jitter, or robustness
  wrappers until the base schema is byte-stable.
- No `reset_many`/`step_many`, vector backend, JAX/GPU environment, or
  observation micro-optimization before the single-env observation contract is
  tested.
- No bonuses, natural spawn randomness, wire replay, spectator catch-up, or full
  browser protocol observation gates until the corresponding state/event lanes
  have pinned evidence.

## Implementation Order

1. Keep the landed `obs_source_movement_empty_multistep_v0` canary small and
   source-tied; do not expand observation work ahead of lifecycle, spawn, or
   RNG blockers.
2. Add schema and purity checks for `oracle_global_debug` or its replacement.
3. Add `curvyzero_egocentric_rays/v0` checks for movement, trail gap,
   borderless, collision death, and ego permutation.
4. Add replay metadata checks: schema hashes, legal mask, ego id, wrapper
   joint-action metadata, and terminal observation policy.
5. Add `curvyzero-observe-v1-local-raster` only when CNN or MuZero work starts.
6. Add browser pixel checks only for render/demo confidence after the matching
   state and semantic observation checks pass.

The useful end state is modest: a small set of trusted observation fixtures that
prove what the agent sees from known states, without turning observation work
into a second source-gameplay oracle.
