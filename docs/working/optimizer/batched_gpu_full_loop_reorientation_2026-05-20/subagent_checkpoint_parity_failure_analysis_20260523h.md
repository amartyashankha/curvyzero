# Checkpoint Parity Failure Analysis 2026-05-23h

Scope: read-only/doc-only analysis of the LightZero PyTorch -> JAX shadow model
checkpoint parity failure. This does not touch live training, Modal volumes, or
trainer defaults.

## Current Facts

- Fresh-model parity passed on Modal L4 GPU at `atol=5e-4`, `rtol=5e-4`.
- Checkpoint parity on
  `training/lightzero-curvytron-visual-survival/cz26a-r001-out0-n0-imm0-b20w05r1/attempts/try-cz26a-r001-out0-n0-imm0-b20w05r1/train/lightzero_exp/ckpt/iteration_260000.pth.tar`
  loaded strictly, consumed all required inference keys, and ran JAX on GPU.
- Checkpoint raw outputs failed mainly on:
  - latent tensors: about `0.003` max absolute error on ramp/random;
  - value logits: about `0.011` max absolute error;
  - recurrent reward logits: about `0.0015` max absolute error.
- Policy logits were close.
- JAX conv and matmul were patched to request `jax.lax.Precision.HIGHEST`;
  the rerun was still pending when this note was written.

## Likely Causes, Ranked

1. **Accumulated GPU arithmetic drift, amplified by trained weights and BatchNorm.**

   This is the most likely explanation. Fresh random models have small,
   untrained BatchNorm statistics and mostly mild activations. A trained
   checkpoint has learned convolution weights, learned head weights, and
   nontrivial BatchNorm running means/vars. Small convolution/matmul differences
   can grow through residual blocks and then appear larger in the value head.

   The precision patch directly targets this. If the patched run drops latent
   max error from about `0.003` to near the fresh-model band, this lane is fine.

2. **The recurrent comparison currently compounds representation drift.**

   In the probe, PyTorch recurrent inference uses `torch_initial.latent_state`,
   while JAX recurrent inference uses `jax_initial["latent_state"]`.

   That means recurrent reward/value/latent differences are not a pure test of
   the dynamics network. They include any earlier representation mismatch.

   Code location:
   - `scripts/probe_lightzero_jax_shadow_model_parity.py:101-107`
   - `src/curvyzero/infra/modal/lightzero_jax_shadow_model_parity.py:296-302`

3. **PyTorch CUDA precision policy may differ from JAX precision policy.**

   Patching JAX to high precision is only half of the issue. PyTorch CUDA may be
   using different cuDNN convolution algorithms or TF32 settings. If PyTorch uses
   TF32 and JAX uses high FP32, the two paths can remain consistently different.

   Smallest check: run the same checkpoint probe with PyTorch TF32 explicitly
   disabled:

   - `torch.backends.cuda.matmul.allow_tf32 = False`
   - `torch.backends.cudnn.allow_tf32 = False`

   Also compare Torch CPU vs Torch CUDA on the same checkpoint. If Torch CPU and
   Torch CUDA differ by the same order, the failure is numeric, not a JAX model
   bug.

4. **A hidden implementation mismatch is possible but less likely.**

   The JAX implementation closely matches the visible LightZero code:

   - representation order: `src/curvyzero/training/lightzero_jax_shadow_model_parity.py:251-268`
   - dynamics order: `src/curvyzero/training/lightzero_jax_shadow_model_parity.py:270-295`
   - prediction order: `src/curvyzero/training/lightzero_jax_shadow_model_parity.py:297-311`
   - conv/matmul precision: `src/curvyzero/training/lightzero_jax_shadow_model_parity.py:325-348`
   - BatchNorm math: `src/curvyzero/training/lightzero_jax_shadow_model_parity.py:350-362`
   - average pool: `src/curvyzero/training/lightzero_jax_shadow_model_parity.py:364-373`
   - residual blocks: `src/curvyzero/training/lightzero_jax_shadow_model_parity.py:375-397`

   Matching LightZero references:

   - `lzero.model.common.DownSample.forward`: `.venv/lib/python3.11/site-packages/lzero/model/common.py:328-351`
   - `lzero.model.common.RepresentationNetwork.forward`: `.venv/lib/python3.11/site-packages/lzero/model/common.py:637-660`
   - `lzero.model.muzero_model.MuZeroModel._dynamics`: `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:309-374`
   - `lzero.model.muzero_model.DynamicsNetwork.forward`: `.venv/lib/python3.11/site-packages/lzero/model/muzero_model.py:505-538`
   - `ding.torch_utils.network.res_block.ResBlock.forward`: `.venv/lib/python3.11/site-packages/ding/torch_utils/network/res_block.py:88-105`

   Fresh-model parity passing makes a large ordering bug unlikely. Still, a
   small pool, BatchNorm epsilon, action-shape, or backend-layout mismatch can
   hide until trained weights amplify it.

5. **Raw support-logit max error may overstate MCTS impact.**

   This is not the root cause because latent tensors fail before support
   decoding. But it may be the wrong pass/fail lens for value/reward. MCTS uses
   scalar value and reward after LightZero's inverse support transform, not raw
   logit max error.

   Code reference:
   - `.venv/lib/python3.11/site-packages/lzero/policy/scaling_transform.py:97-128`

## Add Support-Transformed Scalar Checks?

Yes.

Add scalar comparisons for value and recurrent reward after the same inverse
support transform used by the policy. Report both:

- raw logits, because they reveal exact model drift;
- scalar value/reward after support transform, because that is what search uses.

This should not replace latent parity. It should tell us whether a raw
`0.011` value-logit difference is harmless for search values or actually
changes MCTS inputs.

The safest implementation is to call the existing policy handle on both tensors:

- Torch output: pass the Torch logits directly.
- JAX output: convert JAX logits to a Torch tensor on the same device, then pass
  the same `policy.inverse_scalar_transform_handle`.

That avoids reimplementing the support transform twice.

## Smallest Next Diagnostics If Precision Patch Still Fails

1. **Add scalar value/reward comparisons.**

   Keep raw comparisons, but add:

   - `initial.<kind>.value_scalar`
   - `recurrent.<kind>.value_scalar`
   - `recurrent.<kind>.reward_scalar`

   Use the policy's own inverse transform handle.

2. **Split recurrent into two modes.**

   Add a diagnostic recurrent mode where JAX recurrent uses the PyTorch latent:

   - current mode: `JAX recurrent(JAX initial latent, action)`
   - diagnostic mode: `JAX recurrent(PyTorch initial latent, action)`

   If diagnostic mode passes but current mode fails, dynamics is probably fine
   and representation drift is the source.

3. **Add module-stage checkpoints for representation.**

   Compare PyTorch vs JAX after:

   - first conv + BN + ReLU;
   - `resblocks1`;
   - `downsample_block`;
   - `resblocks2`;
   - `pooling1`;
   - `resblocks3`;
   - final representation resblocks.

   This finds the first layer where error jumps instead of guessing.

4. **Run backend precision falsifiers.**

   Run the same checkpoint on:

   - Torch CUDA default vs JAX GPU high precision;
   - Torch CUDA with TF32 disabled vs JAX GPU high precision;
   - Torch CPU vs JAX CPU if the Modal image can support it.

   If CPU/disabled-TF32 tightens the gap, treat this as numeric backend drift.

5. **Measure decision impact, not only tensor drift.**

   After scalar checks, run a tiny fixed-root search comparison using the same
   observations and legal masks:

   - action agreement;
   - visit-policy L1 distance;
   - root scalar value distance.

   This is the real promotion gate for a search replacement.

## Go / No-Go Recommendation

Go only as a profile-only lane for now: continue if the precision patch plus
scalar/recurrent-split diagnostics show small scalar/search impact; do not use
this for Coach training until checkpoint scalar parity and a real-model compact
search smoke both pass.

