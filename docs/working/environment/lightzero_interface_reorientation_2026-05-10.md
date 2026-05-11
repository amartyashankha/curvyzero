# LightZero Interface Reorientation - 2026-05-10

Status: short working note after reviewing the current LightZero/Pong docs and
CurvyTron adapter code.

## Plain Verdict

CurvyTron is not an ALE environment.

ALE means Arcade Learning Environment. It is an Atari 2600 emulator/API for
Atari ROMs such as `PongNoFrameskip-v4`. It is not a generic name for "visual
RL environment."

Official LightZero Pong uses ALE because that control path is real Atari Pong.
That path is useful as a control: it proves our Modal + LightZero setup can run
the normal visual stack with grayscale frame stacking, six Atari actions, and
the stock Atari env wrapper.

CurvyTron should not copy ALE, Pong rewards, or Atari action meanings. It
should expose the LightZero env shape we need, while keeping CurvyTron rules
owned by the repo-native source-faithful runtime under hardening:
`VectorMultiplayerEnv`.

`CurvyTronSourceEnv` and the source JS oracle are proof tools, not product
environments. Strict `VectorTrainerEnv1v1NoBonus` is a narrow proof/profiling
boundary, not the destination.

Correction to earlier wording: use "LightZero visual stack/conv pattern,
non-ALE" for the intended visual training shape. Do not read it as "CurvyTron
through ALE."

## Current CurvyTron Targets

The scalar/ray sidecar remains useful for diagnostics and contract smoke:

```text
observation: float32[106]
action_mask: int8[3]
to_play: -1
action ids: 0 left, 1 straight, 2 right
step result: BaseEnvTimestep-like obs, reward, done, info
```

The source behavior is still real-time control state advanced over elapsed-ms
frames. `step`, `decision_ms`, and `joint_action` are wrapper/replay terms;
`decision_ms` is a wrapper decision window, not a native CurvyTron tick.
Restricted wrappers are temporary explicit profile configs. They do not replace
the reconstruction target: source-default CurvyTron behavior in the repo-owned
runtime.

## 2P Status

Canonical 2P status: [active_lanes.md](active_lanes.md#2p-status).

The registered wrapper currently exists at:

```text
src/curvyzero/training/curvyzero_lightzero_env.py
env type: curvyzero_v0_lightzero
```

It reuses the local smoke wrapper so it does not invent a second CurvyTron
semantic path. Installed LightZero/DI-engine scalar no-train smoke passes.
The separate debug visual wrapper also now passes installed no-train
config/import/env-factory reset/step smoke on Modal. A tiny CPU
`source_state_fixed_opponent` train smoke also passes: it calls
`train_muzero`, writes 16 action telemetry rows, keeps the opponent fixed at
action `1`, and reports source-state visual tensors over
`VectorMultiplayerEnv`. That is route readiness, not a learning-quality
claim.

The training command variant `source_state_fixed_opponent` is a fixed-opponent
source-state visual wrapper over `VectorMultiplayerEnv`. It is not ALE,
not browser pixels, and not two-seat self-play.

## LightZero Visual Stack

Visual CurvyTron is primary for optimizer profiling now. The current
implemented visual surface is debug-only occupancy smoke; eventual
source-faithful visual work should emit pixels from source-faithful state, then
a LightZero wrapper presents those pixels to the model. First visual shape is
one grayscale frame:

```text
debug raw frame: uint8[1,64,64]
LightZero-facing payload: float32[1,64,64] in [0,1]
LightZero stacked input: float32[4,64,64]
```

Default choice: let LightZero own frame stacking. If the env returns pre-stacked
frames later, that needs a separate schema id and config.

No visual adapter should claim source visual fidelity until a source-faithful
renderer exists.

Current debug visual wrapper status:

```text
env type: curvyzero_debug_visual_tensor_lightzero
env id: CurvyZeroDebugVisualTensorLightZero-v0
env raw frame: float32[1,64,64] LightZero-facing payload
model stack target: float32[4,64,64]
opponent: fixed straight policy
claim: no-train plumbing only
```

## Multiplayer Meaning

The first LightZero bridge is single-ego:

```text
LightZero chooses one ego action.
The wrapper fills opponent actions from named/versioned policies.
The wrapper logs the full wrapper action/control sidecar.
```

This is a valid learner-versus-named-opponents setup. It is not full
simultaneous self-play. Full wrapper joint-action MCTS is deferred because
branching is `3^P` and because the first priority is a clear, replayable
training row.

For multiplayer training later, the missing work is not just a different
`to_play` value. We need a real policy for ego rows, opponent policies,
replay ownership, player identity, rewards, terminal rows, and 3P/4P
observation/replay schemas.

## Next Work

1. Build a bounded whole-loop timing harness around the debug visual wrapper:
   env step, render, stack/normalize, policy/search, replay, learner/eval, and
   reset.
2. Keep strict scalar/ray reset/step/final-observation/reward metadata green.
3. Keep optimizer timings fenced to the strict wrapper and source-backed
   `[B,2,106]` surfaces.
4. Promote multiplayer separately: 3P/4P metadata, replay, and trainer
   observation schemas before any multiplayer training claim.
5. Continue visual CurvyTron profiling on the debug occupancy surface while
   source-faithful visual rendering remains separate Environment-owned work.

## Non-Claims

- Not full CurvyTron fidelity.
- Not source-faithful visual CurvyTron.
- Not ALE CurvyTron.
- Not a CurvyTron learning-quality claim.
- Not 3P/4P trainer-ready self-play.
- Not proof that Pong learning is stable.
