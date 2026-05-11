# Optimizer Visual Tensor Handoff - 2026-05-10

Status: copyable boundary note for Environment <-> Optimizer.

## Short Version

CurvyTron should have visual input and stacked frames. It should not use ALE.

ALE means Arcade Learning Environment, the Atari emulator/API used for Atari
ROM Pong. CurvyTron visual input means our own CurvyTron state/rendering becomes
a tensor that LightZero can stack and feed to a conv model.

## Ownership

Environment Reconstruction owns visual truth:

- what source state the tensor comes from;
- whether it is source-faithful or only debug/profiling data;
- required shape, dtype, value range, channel order, and stack ownership;
- player perspective: global view, ego-centered view, or per-player row;
- reset and final-observation policy;
- metadata fields and schema ids;
- source/browser comparison plan;
- the promotion gate from debug pixels to source-faithful pixels.

Optimizer owns visual smoke/profiling and LightZero adapter plumbing:

- the debug visual smoke wrapper;
- the no-train LightZero visual adapter path;
- frame stack plumbing for the visual path;
- profiler scripts and Modal runs;
- render latency and batch throughput;
- CPU versus GPU versus NumPy/JAX/PyTorch implementation;
- Modal image/function shape;
- batching, memory layout, and frame-stack cost;
- whether render, env step, policy/search, replay, and reset are inside each
  timed number.

Optimizer may implement and optimize debug tensor code. Environment must say
what that tensor means. If a tensor is debug-only, speed numbers must say so.
If Optimizer wants to promote a tensor from debug-only to source-faithful, that
promotion needs Environment source evidence.

## Current Truth

There is not yet a browser/canvas pixel-faithful CurvyTron tensor.

There is now an Environment-owned source-state-backed visual contract:

```text
curvyzero_source_state_gray64/v0
source claim: curvyzero_vector_runtime_source_state_gray64/v0
schema hash: exported as observation_schema_hash/schema_hash
renderer id: curvyzero_source_state_gray64_numpy/v0
renderer hash: exported as renderer_impl_hash
raw render shape: uint8[1,64,64], range [0,255]
normalized payload helper: float32[1,64,64], range [0,1]
surface: source_state_visual_tensor
truth level: source_state_backed_non_browser_pixel
source fidelity level: source_vector_state_geometry_raster
perspective: global_arena_source_state
frame-stack owner: Optimizer
comparison target: curvyzero_vector_runtime_state_arrays
source/browser pixel fidelity: false
ALE usage: none
```

It rasterizes vector runtime source-state arrays (`pos`, `radius`, active body
circles up to `body_write_cursor`, ownership, map size, tick/time, terminal
flags) into a fixed global grayscale frame. It is stronger than the debug
occupancy smoke because it is Environment-owned and source-state-backed; it is
still weaker than browser pixels because it does not prove canvas colors,
anti-aliasing, exact client prediction, or browser parity artifacts.

There is a tiny source-state-backed debug visual smoke target:

```text
curvyzero_debug_occupancy_gray64/v0
source claim: curvyzero_source_state_debug_occupancy_gray64/v0
schema hash: exported as observation_schema_hash/schema_hash
renderer id: curvyzero_debug_occupancy_gray64_numpy/v0
raw render shape: uint8[1,64,64]
trainer payload shape: float32[1,64,64], range [0,1]
stack target: float32[4,64,64]
base frame-stack owner: Optimizer
optional wrapper stack owner: curvyzero_wrapper_local_debug_frame_stack
truth level: debug/profiling occupancy only, from source-state coordinates
comparison target: source state coordinates only
source/browser pixel fidelity: false
```

It renders `CurvyTronSourceEnv` snapshots and
`world_bodies_snapshot()` positions into a coarse occupancy frame. It can prove
shape, dtype, value range, metadata propagation, deterministic source-state
input, and wrapper-owned FIFO frame stacking. It cannot prove browser/canvas
fidelity, exact trail geometry, exact holes, colors, anti-aliasing, or learning
quality.

Metadata required on both debug visual surfaces per the current audit punch
list:

```text
observation_schema_id
observation_schema_hash / schema_hash
renderer_impl_id
truth_level = debug_non_fidelity
source_fidelity_level = none
frame_stack_owner
shape
dtype
range/value_range
ale_usage = none
```

Current source fields named by the contract:

```text
snapshot.game.size
snapshot.avatars[].x/y/alive
world_bodies_snapshot()[].x/y
```

## What Environment Should Produce Next

Minimum truth contract for Optimizer to consume, now first implemented by
`curvyzero_source_state_gray64/v0`:

```text
observation_schema_id
renderer_impl_id
truth_level
source_fidelity_level
shape
dtype
value_range
frame_stack_owner
perspective
player_id / ego_id policy
ruleset_id and ruleset_hash
source_claim_id
reset_seed and reset_source
final_observation_policy
```

The source-state contract now carries these fields with
`source_fidelity_level=source_vector_state_geometry_raster`. The older debug
smoke also carries the fields, but its `source_fidelity_level` is still `none`
for pixels. The blocker remains explicit: no source-faithful browser/canvas
renderer or pixel parity artifact is available yet.

Minimum source-fidelity promotion checklist:

- source state fields used by the renderer are named;
- body/trail/hole geometry is covered by source-backed tests;
- wall versus torus mode is covered;
- player identity and multiplayer perspective are defined;
- reset and terminal final-frame policy are defined;
- comparison target is named: source server state, source wire events, browser
  canvas pixels, or a clearly weaker debug target.

Environment should not build a competing LightZero visual adapter while
Optimizer owns that setup. Environment should instead keep the source-truth
contract current and review any adapter output for claim accuracy.

## What Optimizer Should Produce Next

Minimum visual smoke/profiling owned by Optimizer:

- reset returns one frame `(1,64,64)` plus `action_mask` and `to_play=-1`;
- step returns next frame, reward, done, and info;
- terminal step includes final visual observation;
- info says `ale_usage=none`;
- info says whether the frame is debug-only or source-faithful.
- if using stacked observations, report whether stacking is LightZero-owned or
  wrapper-owned; the local stacked debug wrapper is FIFO and explicitly not a
  source-pixel fidelity claim.

Optimizer can time the debug tensor separately from real fidelity:

```text
surface: debug_visual_tensor
schema: curvyzero_debug_occupancy_gray64/v0
truth_level: debug/profiling only
includes_env_step: yes/no
includes_render: yes/no
includes_stack: yes/no
includes_policy/search/replay/reset: yes/no
```

Useful first measurements:

- render one frame from source snapshot;
- render batch of frames from source snapshots;
- update a 4-frame stack;
- reset/step/render wrapper smoke;
- compare cost against current `[106]` scalar/ray observation packing.

Do not compare debug tensor speed to final visual fidelity speed as if they are
the same thing.

## Copyable Handoff

Optimizer: CurvyTron visual input is intended, but not through ALE. Environment
Reconstruction owns whether visual tensors are source-faithful. Optimizer owns
the visual smoke/profiler and LightZero adapter plumbing. Current available
surface is only `curvyzero_debug_occupancy_gray64/v0`, a `uint8[1,64,64]` raw
occupancy smoke frame from source-state coordinates, normalized to
`float32[1,64,64]` for trainer payloads, with no browser/canvas pixel fidelity
claim. The optional local stacked wrapper produces `float32[4,64,64]` by FIFO
stacking normalized debug frames and labels itself wrapper-owned. Please push it
toward a usable profiling target, but label it as `debug_visual_tensor` and
report explicit booleans for render, stack, env step, policy/search, replay,
and reset. If you change tensor semantics or want to promote it beyond debug,
send the claim back to Environment for source-fidelity review.
