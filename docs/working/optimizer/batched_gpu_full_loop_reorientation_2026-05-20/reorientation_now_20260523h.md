# Optimizer Reorientation, 2026-05-23h

## Plain Goal

Make CurvyTron training faster without changing what Coach is trying to train.

That means:

- keep live Coach runs read-only;
- keep stock LightZero as the truth source;
- use optimizer profile lanes to test faster dataflow/search ideas;
- only promote a speed path after it proves the same observation, action,
  replay, RND, and learner-facing sample meaning.

## Current Answer

We are not in a random rabbit hole yet.

The current JAX/MCTX bridge is useful because the strict profile rows showed a
real speed signal:

| shape | direct CTree | MCTX/JAX | speedup |
| --- | ---: | ---: | ---: |
| H100 B512 sim16 | 5,864 | 11,826 | 2.02x |
| H100 B512 sim32 | 4,781 | 8,667 | 1.81x |
| H100 B1024 sim16 | 4,947 | 11,700 | 2.36x |
| H100 B1024 sim32 | 4,400 | 13,964 | 3.17x |

Update after the first real-shadow bridge:

```text
The toy MCTX rows have been superseded by real immutable-checkpoint MCTX shadow
rows. Current real-shadow profile rows:
  B64 scalar-on:       1.36x
  B512 scalar-on:      2.37x
  B512 scalar-off:     1.58x
  B1024 scalar-off:    2.20x
```

This is still profile-only. It proves a real checkpoint can drive the MCTX
profile service. It does not prove Coach training speed, because MCTX search
semantics differ from LightZero CTree and no `train_muzero` backend has changed.

## Why The Next Gate Is Search-Impact Parity

The useful next question is simple:

```text
Does real-checkpoint MCTX choose similar enough actions, visit distributions,
and root values compared with direct LightZero CTree on the exact same compact
roots?
```

Fresh-model parity passed on Modal L4 GPU. A current immutable checkpoint can
now be loaded into the JAX shadow model and used by the profile-only MCTX
service. Do not use mutable aliases like `latest.pth.tar` or `ckpt_best.pth.tar`.

## Stop Conditions

Stop this lane and rethink if any of these happen:

- current checkpoint parity fails for architectural reasons, not just tolerance;
- JAX model conversion becomes a large second trainer instead of a search shadow;
- MCTX with the real model loses most of the profile speedup;
- the measured wall moves so strongly to env/observation/replay that search is
  no longer a meaningful target;
- keeping PyTorch and JAX synchronized requires frequent host/device crossings
  that erase the win.

## Next Gates

1. Find one current immutable checkpoint ref from the runs volume or local
   manifests. Done for the first probe:
   `cz26a-r001...iteration_260000.pth.tar`.
2. Run the LightZero PyTorch -> JAX checkpoint parity probe on Modal GPU.
   Partial result: strict load and key coverage passed; raw latent tensors did
   not pass strict tolerance yet. Added scalar support-value checks and
   recurrent-from-PyTorch-latent diagnostics. The recurrent-from-PyTorch-latent
   check says dynamics/prediction are mostly aligned; root representation drift
   is the remaining mismatch.
3. Real JAX shadow wiring landed behind `MctxCompactSearchServiceV1`, still
   profile-only. Current gate: same-root MCTX-vs-direct-CTree comparison.
4. Run matched profile rows: direct CTree, real-model MCTX, toy MCTX ceiling,
   and service-tax/mock ceilings.
5. Only after that, decide whether a Coach-facing experimental backend is worth
   designing.

## Current Local Safety

The current local optimizer additions are fenced as profile-only:

- `src/curvyzero/training/lightzero_jax_shadow_model_parity.py`
- `scripts/probe_lightzero_jax_shadow_model_parity.py`
- `src/curvyzero/infra/modal/lightzero_jax_shadow_model_parity.py`
- `tests/test_lightzero_jax_shadow_model_parity.py`

They do not call `train_muzero`, do not touch live runs, do not change trainer
defaults, and do not create Coach launch advice.
