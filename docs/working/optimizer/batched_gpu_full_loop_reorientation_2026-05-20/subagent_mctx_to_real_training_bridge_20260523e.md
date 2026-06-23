# MCTX To Real Training Bridge, 2026-05-23e

Scope: sidecar architecture note. I inspected the profile-only MCTX compact
service, the compact search/slab contracts, the current direct LightZero search
hook, and the CurvyTron LightZero config builder. I did not modify production
code or touch live Coach runs.

## Current Fact

The H100 compact-slab MCTX rows are a real speed signal:

| shape | MCTX steps/sec | direct LightZero CTree baseline | speedup |
| --- | ---: | ---: | ---: |
| B512/A16/sim16 | 16,250 | 6,522 | 2.49x |
| B512/A16/sim32 | 14,306 | 4,177 | 3.43x |
| B1024/A16/sim16 | 20,557 | 6,992 | 2.94x |
| B1024/A16/sim32 | 16,255 | 5,314 | 3.06x |

Plain meaning:

```text
Compiled/device-owned search is worth pursuing.
The current MCTX row is not training-compatible yet.
```

Why it is not training-compatible:

- `MctxCompactSearchServiceV1` uses a toy JAX model, not the LightZero PyTorch
  `MuZeroModel`.
- It calls `mctx.gumbel_muzero_policy`, not LightZero's CTree search.
- It does not call `train_muzero`.
- It returns checked compact search arrays, but no stock LightZero learner owns
  those arrays yet.
- It currently has fixed-shape profile assumptions, including all roots active.

## Current Real Training Surface

The trusted CurvyTron config is still built from LightZero's Atari MuZero config:

```text
policy.model.model_type = "conv"
policy.model.observation_shape = [4,64,64]
policy.model.image_channel = 4
policy.model.frame_stack_num = 1
policy.model.action_space_size = 3
policy.num_simulations = configured run knob
policy.batch_size = configured run knob
```

LightZero's PyTorch model surface is simple:

```text
initial_inference(obs) -> value, reward=0, policy_logits, latent_state
recurrent_inference(latent_state, action) -> value, reward, policy_logits, next_latent_state
```

The current direct CTree optimizer hook keeps root latents on GPU, but every
simulation still has CPU/control work:

```text
CPU CTree traverse
-> GPU recurrent_inference
-> reward/value/policy copied back to CPU
-> CPU CTree backpropagate
```

That is the gap MCTX avoids.

## Path 1: Keep MCTX As A Speed Ceiling

Use MCTX only to answer: "how fast could this shape be if search were compiled
and device-owned?"

Cost: low.

Pros:

- Already works in the profile-only compact slab.
- Gives an honest ceiling against `direct_ctree_gpu_latent`.
- Does not risk Coach training.

Risks:

- It will not by itself produce better overnight training.
- It can mislead us if we forget the model is toy JAX.

Smallest next proof:

```text
Run one wider profile-only grid:
  H100 and L4
  B512/B1024/B2048
  sim8/sim16/sim32/sim64
  same compact slab
  same profile-only labels
```

Keep this path regardless. It is the comparator.

## Path 2: Torch Learner, JAX Shadow Model For MCTX Search

This is the cleanest near-term bridge if we want MCTX speed without rewriting
the whole trainer.

Shape:

```text
LightZero PyTorch learner owns training and checkpoints.
After learner updates, export the current PyTorch model weights.
Load/convert those weights into an equivalent JAX model.
Collector/search uses JAX model + MCTX.
MCTX returns selected actions, visit policy, root value.
Replay/learner still use the stock LightZero training path as much as possible.
```

Cost: medium to high.

Why it could work:

- It preserves the current PyTorch learner as the source of truth.
- It puts the hot search loop in JAX/MCTX, where the speed signal is strongest.
- In a synchronous loop, "stale weights" can be avoided by converting after each
  learner update, or accepted as a controlled collector snapshot if conversion is
  amortized.

Main risks:

- We must implement a JAX copy of the exact LightZero model architecture.
- Weight conversion must cover representation, dynamics, prediction heads, support
  transforms, action encoding, normalization, and any model config changes.
- MCTX search semantics still differ from LightZero CTree.
- Conversion cost may matter if done too often.
- Checkpoint/eval/tournament remain PyTorch, so the observation contract must be
  identical across PyTorch and JAX paths.

First smallest proof gate:

```text
One-batch model parity only. No MCTS yet.

Load one LightZero PyTorch MuZeroModel checkpoint.
Build the matching JAX model.
Convert weights.
Run:
  initial_inference(obs)
  recurrent_inference(latent, action)
Compare:
  policy_logits
  value logits
  reward logits
  latent shape and rough numeric tolerance
```

Pass means "JAX can represent the current policy." It does not mean training is
solved.

Second proof gate:

```text
Use the converted JAX model inside MCTX for one compact root batch.
Return CompactSearchResultV1.
Check legal actions, root ids, visit policy, root value, replay-index rows.
```

Third proof gate:

```text
Scratch stock-profile run, not live training:
collector/search uses MCTX/JAX shadow policy
replay/learner consumes the result
one learner update happens
checkpoint can be written
all labels say this is an experimental backend
```

## Path 3: Full JAX Ownership

Rewrite the real trainer around JAX/MCTX.

Shape:

```text
JAX model
JAX learner
JAX MCTX collector/search
JAX or array-native replay/sample path
PyTorch/LightZero kept only as historical reference
```

Cost: very high.

Pros:

- Cleanest compute model.
- No Torch-to-JAX sync boundary inside the hot loop.
- Best chance of preserving the 2.5x-3.4x MCTX search win in a real system.

Risks:

- Replaces the trusted LightZero training loop.
- Requires new checkpoint, eval, tournament, RND, replay, target-building, and
  config ownership.
- Harder to compare learning regressions because many things change at once.

First smallest proof gate:

```text
Tiny JAX-only MuZero learner on the CurvyTron compact observation contract.
One synthetic replay batch.
One optimizer step.
Loss decreases or parameters change.
No Coach claim.
```

Recommendation: not first. Keep it as a possible future architecture if the
Torch-shadow path becomes too ugly.

## Path 4: Custom Torch/CUDA Search Backend

Preserve PyTorch model ownership and replace only the tree/search body.

Shape:

```text
LightZero PyTorch model remains real policy.
Search tree state becomes fixed-shape arrays.
Traversal/backprop run in a compiled extension or fused Torch/CUDA kernels.
Only selected actions and replay payload leave the device.
```

Cost: high.

Pros:

- Best semantic preservation if we can keep LightZero PUCT/CTree behavior.
- No JAX model port.
- Checkpoint/eval/tournament stay PyTorch.

Risks:

- Writing a correct GPU tree search backend is nontrivial.
- Dynamic tree control flow can be awkward on GPU.
- A half-compiled Torch loop can be slower than CTree, as the current compact
  Torch service already showed.
- Debugging CUDA/C++ parity bugs can be expensive.

First smallest proof gate:

```text
Profile-only A=3 fixed-shape backend with synthetic recurrent outputs.
Inputs:
  root priors
  root values
  legal mask
  precomputed recurrent payload
Outputs:
  selected action
  visit counts/policy
  root value
Compare against direct CTree on a tiny deterministic fixture.
```

Only after that should it call the real PyTorch recurrent model.

## Path 5: Patch LightZero CTree Incrementally

Keep CTree but reduce the obvious CPU/list tax.

Cost: medium.

Pros:

- Closest to current LightZero semantics.
- Smaller than a new CUDA tree.

Risks:

- Previous flat-A3/direct CTree work did not produce a large win.
- Even with fewer Python lists, the CPU tree still needs per-simulation control
  and model-output readback.
- It is likely a 1.x improvement, not the big architecture change.

First smallest proof gate:

```text
Add one CTree ABI cleanup that removes a measured copy/listify bucket.
Run same-denominator H100 B1024 sim16/sim32.
Keep only if the full slab row improves, not just a sub-timer.
```

Recommendation: useful later, not the main next bridge.

## Recommended Order

1. Keep MCTX as the speed ceiling and continue wider profile-only grids.
2. Start the Torch-to-JAX model parity gate. This is the smallest proof that can
   turn the MCTX speed result into a possible training path.
3. In parallel, sketch the custom Torch/CUDA fixed-shape tree gate, but do not
   build the full backend until the synthetic gate proves the idea.
4. Do not promote MCTX to Coach until a scratch full-loop run proves collection,
   replay, learner update, checkpoint, RND, and observation contracts together.

## Plain Recommendation

The next real implementation path should be:

```text
PyTorch LightZero learner remains source of truth.
Build a JAX shadow of the LightZero MuZero model.
Use converted weights inside MCTX search.
Return CompactSearchResultV1.
Feed the trusted replay/learner path.
```

This is the narrowest path that tries to keep the MCTX speed win while not
throwing away the current training system.

If model parity is painful or unstable, fall back to a custom Torch/CUDA
fixed-shape search backend. If both are too expensive, treat MCTX as an
architecture ceiling only and keep optimizing the direct CTree path.
