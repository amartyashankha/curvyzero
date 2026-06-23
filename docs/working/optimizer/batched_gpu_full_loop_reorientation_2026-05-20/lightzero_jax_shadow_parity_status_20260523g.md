# LightZero JAX Shadow Parity Status, 2026-05-23g

## Plain Status

We now have the first real bridge gate between the fast MCTX/JAX optimizer lane
and the real CurvyTron LightZero model.

What exists:

- `src/curvyzero/training/lightzero_jax_shadow_model_parity.py`
- `scripts/probe_lightzero_jax_shadow_model_parity.py`
- `src/curvyzero/infra/modal/lightzero_jax_shadow_model_parity.py`
- `tests/test_lightzero_jax_shadow_model_parity.py`

This is still profile-only. It does not call `train_muzero`, does not call
MCTX, does not touch live runs, and does not change trainer defaults.

## What It Proves

The JAX shadow model can run the current LightZero MuZero model surface:

- model class: `lzero.model.muzero_model.MuZeroModel`
- observation: `[B,4,64,64]`
- latent: `[B,64,8,8]`
- action count: `3`
- reward/value support width: inferred from the model/checkpoint, not hard-coded
- BatchNorm eval buffers are consumed
- SSL projection/prediction-head keys are ignored for inference

Modal L4 fresh-model smoke passed:

```text
run: ap-mxbORR6cvPQwKD70Vyh7MQ
ok=true
jax backend=gpu
torch device=cuda
LightZero=0.2.0
jax=0.7.0
torch=2.8.0
coverage ok=true
required inference keys consumed=123
ignored keys=52
```

The fresh smoke rewrites the normally zero-initialized final reward/value/policy
heads so policy/value/reward comparisons are not fake zero-output passes.

## Numeric Read

On L4 GPU, Torch and JAX do not produce bit-exact latent tensors. The largest
observed latent absolute differences were about:

```text
initial ramp latent:    1.66e-4
recurrent ramp latent:  1.98e-4
initial random latent:  1.20e-4
recurrent random latent:1.53e-4
```

With explicit GPU tolerance `atol=5e-4, rtol=5e-4`, all comparisons passed.
Policy/value/reward logits were much tighter, around `1e-6` to `1e-5` max abs
in the fresh nonzero-head smoke.

The large relative errors on some latent cells are near-zero denominator effects;
the max absolute error is the useful number here.

## Checkpoint Gate

The Modal wrapper also accepts an immutable checkpoint ref:

```bash
uv run --extra modal modal run \
  -m curvyzero.infra.modal.lightzero_jax_shadow_model_parity \
  --compute l4 \
  --checkpoint-ref training/.../iteration_N.pth.tar \
  --batch-size 2
```

Mutable refs like `latest.pth.tar` and `ckpt_best.pth.tar` are rejected.

An old ref found in docs was tried:

```text
training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar
```

It is not present in the current runs volume anymore. That did not test
checkpoint parity. It only proved we need a current immutable checkpoint ref for
the next real gate.

2026-05-23h update:

```text
Current immutable checkpoint tested:
training/lightzero-curvytron-visual-survival/cz26a-r001-out0-n0-imm0-b20w05r1/attempts/try-cz26a-r001-out0-n0-imm0-b20w05r1/train/lightzero_exp/ckpt/iteration_260000.pth.tar

SHA256:
a9bcbc7212995b967d75d9ff94b189039bf0111b9d30daeb2fb29edebe402b5b
```

Plain result:

```text
The checkpoint loaded strictly.
All required inference keys were consumed.
JAX ran on GPU.
Policy logits were tight.
Raw latent tensors still failed strict tolerance.
Raw value logits were noisy, but scalar support-transformed value/reward was
much closer and is the more relevant search input.
```

This is not a pass yet. It is also not a missing-checkpoint or missing-weight
failure. The current read is trained-checkpoint numeric drift or a small
representation-layer mismatch. A follow-up diagnostic now compares recurrent
inference from the PyTorch latent so we can separate representation drift from
dynamics/prediction mismatch.

Diagnostic result:

```text
When JAX recurrent inference starts from the PyTorch latent, recurrent latent,
policy, and reward become tight under the strict gate. That means the recurrent
dynamics/prediction port is mostly aligned.

The main remaining mismatch is representation output drift from the image
observation into the root latent.
```

Practical tolerance gate now in flight:

```text
atol=5e-3, rtol=5e-3
refs:
  cz26a-r001 iteration_260000
  r18fresh champion iteration_250000
```

If those pass, this is a usable profile-only bridge gate. It does not mean exact
model parity. It means the real checkpoint can be represented by the JAX shadow
well enough for the next compact-search smoke.

## Current Recommendation

Next gate:

```text
Finish checkpoint diagnostics:
  recurrent from JAX latent
  recurrent from PyTorch latent
  scalar support-transformed value/reward

If semantic scalar/action-search impact is small, start wiring the real JAX
shadow model into the MCTX compact search service behind the existing
profile-only CompactSearchServiceV1 boundary.
```

Do not promote MCTX/JAX to Coach training before checkpoint parity passes.
