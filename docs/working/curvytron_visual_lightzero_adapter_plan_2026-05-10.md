# CurvyTron Visual LightZero Adapter Plan - 2026-05-10

Purpose: working-memory note for the CurvyTron visual LightZero adapter path
and Environment/Optimizer ownership split. This note does not assign visual
smoke, profiler, or LightZero adapter plumbing to Environment.

## Current Conclusion

Visual CurvyTron is the intended main training path. Treat it as
LightZero-compatible visual input, not as Atari/ALE itself:

```text
CurvyTron source-faithful state
  -> project-owned visual/raster observation
  -> grayscale frame, initially (1, 64, 64)
  -> LightZero frame stack, usually (4, 64, 64)
  -> conv MuZero-style policy/model path
```

ALE means Arcade Learning Environment, the Atari emulator/API used for official
Atari ROM setups such as Pong. CurvyTron should not depend on Atari ROMs. Use
"LightZero visual stack/conv pattern, non-ALE" for the visual training shape:
frames, preprocessing, stacking, conv model, collector/evaluator/replay
plumbing.

The scalar/ray path:

```text
[B, P, 106] rays/scalars -> replay-shaped chunks
```

is now a sidecar diagnostic and fallback/profiling surface. It is useful for
geometry sanity, source-state timing, and possible repo-native experiments, but
it is not the main CurvyTron trainer target unless the visual path is explicitly
abandoned.

## First Visual Adapter Shape

The current implemented visual surface is debug occupancy only. The eventual
source-faithful path should expose one frame:

```text
observation: float32 grayscale frame, shape (1, 64, 64), range [0, 1]
action_mask: fixed ego action mask
to_play: -1
```

Let Optimizer's LightZero adapter own frame stacking to produce:

```text
(4, 64, 64)
```

That keeps env rendering simple and makes the frame-stack behavior explicit in
the LightZero config. If a future wrapper returns pre-stacked frames, it needs a
separate schema id and matching config.

## Renderer Status

No source-faithful visual renderer exists yet.

The current debug visual smoke uses a clearly labeled helper:

```text
curvyzero_debug_occupancy_gray64/v0
```

That helper now has a tiny honest contract: it renders
`CurvyTronSourceEnv` snapshot avatar coordinates plus
`world_bodies_snapshot()` coordinates into a coarse occupancy frame. It proves
deterministic source-state input, shape, dtype, value range, metadata, and local
FIFO stack policy when the wrapper-owned stack is used. It must not claim
source visual fidelity, browser/canvas fidelity, policy quality, or Atari/ALE
compatibility. Its raw renderer frame is `uint8[1,64,64]`; any
trainer/LightZero-facing payload should be labeled as normalized
`float32[1,64,64]` in range `[0,1]`.

Immediate optimizer consequence: do not spend more main-lane effort optimizing
scalar/ray observation unless it answers a visual-adapter question. The next
measurement belongs in Optimizer's visual smoke/profiler lane, not in a
competing Environment adapter.

Boundary with Optimizer:
[docs/working/environment/optimizer_visual_tensor_handoff_2026-05-10.md](environment/optimizer_visual_tensor_handoff_2026-05-10.md).
Environment owns whether a visual tensor is source-faithful or debug-only.
Optimizer owns the visual smoke/profiler, LightZero visual adapter plumbing,
render/stack timing, batching, Modal/GPU/CPU implementation choices, and
whole-loop bottleneck reads.

Minimum metadata Environment must define for any visual tensor so Optimizer can
time it safely:

- `observation_schema_id`
- `truth_level`
- `source_fidelity_level`
- `shape`
- `dtype`
- `range`
- `perspective`
- `frame_stack_owner`
- `renderer_impl_id`
- `includes_render_cost`

For `curvyzero_debug_occupancy_gray64/v0`, `truth_level` is
`debug_non_fidelity`, `source_fidelity_level` is `none`, `perspective` is
`global_arena_debug`, `frame_stack_owner` is `optimizer`, and
`renderer_impl_id` is `curvyzero_debug_occupancy_gray64_numpy/v0`. The source
claim is only `curvyzero_source_state_debug_occupancy_gray64/v0`, with
comparison target `source_state_coordinates_only`; browser pixel fidelity is
false.

Both the single-frame debug surface and the local stacked debug surface must
carry schema id/hash, renderer id, truth level, `source_fidelity_level=none`,
frame stack owner, shape, dtype, range, and `ale_usage=none`. This is a contract
clarity gate from the environment audit punch list, not a pixel-fidelity gate.

## Adapter Boundaries

Current implemented debug visual adapter:

- local smoke:
  `src/curvyzero/training/curvyzero_debug_visual_lightzero_smoke.py`;
- registered DI-engine wrapper:
  `src/curvyzero/training/curvyzero_debug_visual_lightzero_env.py`;
- no-train runtime probe:
  `src/curvyzero/training/curvyzero_debug_visual_lightzero_runtime_probe.py`;
- installed-runtime Modal smoke:
  `src/curvyzero/infra/modal/lightzero_curvyzero_debug_visual_config_import_smoke.py`.

The installed LightZero/DI-engine no-train smoke now passes on Modal for the
debug visual tensor wrapper. It proves import/config/env-factory/direct
reset/step plumbing with a real `BaseEnvTimestep`, LightZero-facing env payload
`float32[1,64,64]`, LightZero model stack `(4,64,64)`, action space `3`, and
no ALE identity. It
does not train, does not prove learning, and does not promote the debug tensor
to source-fidelity.

The local stacked debug survival wrapper is separate and explicit:

```text
observation: float32[4,64,64]
raw frame source: curvyzero_debug_occupancy_gray64/v0
stack policy: wrapper-owned FIFO over normalized debug frames
frame_stack_owner: curvyzero_wrapper_local_debug_frame_stack
source_fidelity_claim: none
ale_usage: none
```

That wrapper proves only local stacked-frame shape and terminal
`final_observation` plumbing. It does not prove LightZero env-manager frame
stacking, source-pixel fidelity, search, replay, learner input, or training.

For the first real visual wrapper:

- LightZero chooses one ego action.
- The wrapper supplies opponent actions through a named opponent or snapshot
  policy.
- The wrapper logs the full joint action/control snapshot.
- Run summaries name the observation schema, action schema, reward schema, and
  opponent policy.

This remains a single-ego bridge until native simultaneous self-play behavior is
designed and checked.

## Non-Goals

- No ALE integration.
- No Atari ROM dependency.
- No claim that debug pixels are source visuals.
- No policy-quality claim from smoke tests.
- No native simultaneous self-play claim.

## One-Line Summary

Visual CurvyTron is primary: first expose one grayscale `(1,64,64)` frame, let
LightZero stack to `(4,64,64)`, and use the current debug occupancy pixels only
as non-fidelity smoke data. Keep `[B,P,106]` rays/scalars as sidecar
diagnostics, not the main training target.
