# Observation Fidelity

Status: Draft
Date: 2026-05-08

## Purpose

Observation fidelity means the agent sees a stable, tested view of the simulator
state. It does not mean the learned observation must match the original
CurvyTron browser pixels.

Keep three ideas separate:

- Source state fidelity: does Python game state match the JS source?
- Observation fidelity: does an observation correctly encode the chosen Python
  game state?
- Render fidelity: do screenshots or videos look like the browser?

The first training path should use simulator-native observations. Browser image
checks can wait.

## Current Starting Point

`CurvyTronEnv._observations()` currently returns the same flat global vector for
both players. That is useful for deterministic tests and privileged debugging,
but it should not be the default learned observation because it exposes absolute
position and player order.

Use the observation plan from `docs/research/observation_reward_design.md`:

- v0 learned observation: ego-relative rays, scalars, and action mask.
- v1 learned observation: ego-centered local raster, scalars, and action mask.
- global state vector: debug/oracle only.

## Basic Rules

- Test observations against simulator state first, not against screenshots.
- Every observation schema needs an id and hash.
- Shape, dtype, channel order, scalar order, normalization, and action order are
  part of the schema.
- Observations must not mutate simulator state.
- Ego perspective must not leak stable seat, color, or player index unless the
  schema explicitly says so.
- Random observation noise belongs in a wrapper, with its own seed and schema or
  wrapper hash.

## Ray Observation Tests

Target schema: `curvyzero_egocentric_rays/v0`. The older research name
`curvyzero-observe-v0-rays` is a legacy alias.

Expected shape from the current design:

```text
rays        float32[24, 4]
scalars     float32[10]
action_mask bool[3]
```

Minimum tests:

- Shape, dtype, finite values, and range.
- Same state gives byte-stable rays.
- Calling observe twice does not change state.
- `observe_many` matches repeated single-player `observe`.
- Live v0 players have mask `[true, true, true]`.
- Dead or terminal padded rows have mask `[false, false, false]`.
- Known wall fixtures produce expected distances.
- Known own-trail fixtures hit the own-trail channel.
- Known opponent-trail fixtures hit the opponent-trail channel.
- Known opponent-head fixtures hit the opponent-head channel.
- Rotating the whole state with the ego should rotate/canonicalize the rays, not
  change their meaning.
- Swapping player ids, seats, or colors should not change the ego observation
  except for the intended ego/opponent perspective.

Good first fixtures:

- Ego in arena center, heading east, empty arena.
- Ego near each wall.
- One own trail segment directly ahead.
- One opponent trail segment to the left.
- Opponent head directly ahead.
- Mirrored 1v1 setup with ego changed from player 0 to player 1.

Useful later:

- Check that the shortest forward wall/trail ray agrees with one-step collision
  danger in simple cases.
- Add denser forward/side rays only after a real policy or heuristic failure
  shows the 24-ray schema is missing information.

## Local Raster Tests

Target schema: `curvyzero-observe-v1-local-raster`.

Expected shape from the current design:

```text
planes      float32[5, 48, 48]
scalars     float32[10]
action_mask bool[3]
```

Minimum tests:

- Shape, dtype, finite values, and range.
- Channel order is fixed:
  - wall or out of bounds
  - own trail
  - opponent trail
  - ego head
  - opponent head
- Ego head is at the expected center cell.
- Ego heading points in the documented raster direction.
- Out-of-arena cells are present in the wall channel.
- Own and opponent trails do not bleed into each other's channels.
- Local raster generated from the same occupancy state is byte-stable.
- Rotating the physical state around the ego gives the same canonical raster.
- Player id, seat, and color permutations do not leak into planes.
- `observe_many` matches repeated single-player raster generation.

Scale policy:

- Start with one simulator occupancy cell per raster cell.
- If crop scale, interpolation, crop size, or channel order changes, create a
  new schema hash.
- If antialiasing or interpolation is added later, test it separately from the
  first nearest-cell raster.

## Image Observation Tests

Image observations are later work. Use them only when training from pixels or
checking browser/demo rendering.

Minimum tests before trusting image observations:

- Same state and render config produce the same image.
- Render size, arena scale, line width, head radius, and background color are
  pinned.
- Ego, own trail, opponent trail, walls, and heads are visible in known fixtures.
- Pixel colors are either fixed by schema or intentionally randomized by a
  recorded wrapper.
- Browser screenshot checks run only after state traces pass for the same
  scenario.
- Screenshot diffs use tolerances for antialiasing and device scale, but state
  mismatches should be fixed in state traces, not hidden in pixel thresholds.

Do not make browser pixels the first learned observation. They are expensive,
harder to debug, and can fail for rendering reasons while the game state is
correct.

## Color, Noise, And Scale Variants

### Color

For rays and local rasters, prefer semantic channels such as ego, opponent,
wall, and trail. Do not encode stable player colors as learned identity unless a
schema explicitly needs it.

Tests:

- Swap player colors and confirm ray/raster observations are unchanged.
- If image observations use colors, record the palette and test that color
  randomization does not change non-image state observations.
- In multiplayer, permute colors and seats together and confirm the ego view is
  still canonical.

### Noise

Noise should be a training wrapper, not hidden simulator behavior.

Tests:

- Base observation without noise stays byte-stable.
- Noise seed is recorded.
- Same noise seed gives the same noisy observation.
- Different noise seeds change only the intended fields.
- Evaluation defaults to no noise unless the evaluation config says otherwise.

### Scale

Scale can mean arena size, local crop size, raster cell size, or rendered image
resolution. Treat those as behavior-affecting observation choices.

Tests:

- Ray distances stay normalized when arena size changes.
- Local raster crop keeps the ego centered across map sizes.
- Image render scale changes image resolution, not the underlying state.
- Schema hash changes when shape, crop size, cell scale, render scale, or
  normalization changes.

## Multiplayer Observation Coverage

Do not stop at 1v1.

Minimum multiplayer tests:

- 3-player fixture with one opponent ahead and one behind.
- 4-player fixture with two opponent trails in different directions.
- Seat permutation test: the same physical state gives the same ego view for
  each player after perspective transform.
- Dead-player test: dead opponents are either encoded by a documented channel or
  omitted by a documented rule.
- Action mask stays per-ego and does not depend on player index.

## Final Training Observation Checks

Before using learning curves as evidence, replay rows should store:

- observation schema id and hash
- rules hash
- reward schema id and hash
- ego player id
- perspective transform id
- legal action mask
- observation or observation reference
- wrapper action map / `joint_action`
- reward after the wrapper transition window
- `terminated` and `truncated`
- terminal observation when an autoreset wrapper is used
- opponent policy or checkpoint ids when opponents are scripted or learned

Training tests:

- Replaying seed, config, wrapper action maps, and wrapper elapsed-ms
  source-frame cadence recreates the same observations.
- Terminal observation is the real final observation, not the first observation
  of the next reset.
- Post-terminal padding masks policy, value, and reward losses.
- Replay readers reject mismatched rules, observation, action, or reward hashes.

## Practical Test Order

1. Keep the current global vector as `oracle_global_debug`.
2. Add ray observation unit fixtures.
3. Add ego/seat/color permutation tests.
4. Add local raster fixtures only when CNN or MuZero work starts.
5. Add image observation tests only if training from pixels or validating a
   browser/demo render.
6. Add color, noise, and scale variants as wrappers with explicit config and
   schema hashes.

This is enough. We do not need a large observation framework before the first
baseline agents learn.
