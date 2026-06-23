# Uint8 Payload Design Note

Date: 2026-05-20

Status: research note, not implemented.

## Plain Finding

The B512 direct-GPU profile still pickles about `67.1MB` per step because the
policy payload is:

```text
1024 policy rows * 4 * 64 * 64 * float32
```

Using `uint8` for the stacked observation would cut that array size by about
`4x`, but it is not a safe env-only flip.

## Why It Is Not A Simple Flip

Curvy's current policy/model contract is `float32` `[4,64,64]` in `[0,1]`.
Raw single frames are `uint8`, but the stack handed to LightZero is normalized
float32.

Stock LightZero collect/replay/learn code expects NumPy observations and moves
them to torch tensors. It does not automatically do `float()/255` for us.
Therefore a `uint8` env payload would need explicit cast/scale at every model
inference path:

- collector policy inference;
- replay/reanalysis target paths;
- learner batch preparation;
- padding/zero-observation helpers;
- RND latest-frame extraction if RND remains attached.

## Practical Read

`uint8` is still promising because it targets the payload/process wall, not the
render kernel. But it should be a named contract with tests, not a hidden
optimization.

Safe next design:

1. Keep real trainer/tournament default on float32 `cpu_oracle`.
2. Add a profile-only `uint8_stack_payload` canary.
3. Add one explicit cast/scale helper and prove collector/learner/RND use it.
4. Compare payload bytes, pickle time, policy inference outputs, and replay
   sample shapes against float32.

Do not make `uint8` the default until those gates pass.

