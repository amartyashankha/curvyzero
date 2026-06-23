# JAX Shadow Model Parity Harness Plan, 2026-05-23f

Scope: implementation plan only. Do not touch live Coach training runs. Do not
modify trainer defaults. MCTX remains profile-only; the first bridge proof is
PyTorch-to-JAX model parity, not a training run.

## Repository Facts To Reuse

- `src/curvyzero/training/lightzero_config_builder.py` is the trusted CurvyTron
  LightZero config builder. It patches `policy.model.observation_shape=[4,64,64]`,
  `action_space_size=3`, support sizes/ranges, `num_simulations`, and
  checkpoint hooks on copied configs.
- `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py` already has
  the best checkpoint load path for this proof:
  `load_lightzero_curvytron_visual_survival_policy(...)` builds a matching
  `MuZeroPolicy`, strict-loads a checkpoint into `policy._model`, infers support
  sizes from the checkpoint, validates policy-observation metadata, and returns
  load metadata.
- `src/curvyzero/infra/modal/mctx_dependency_smoke.py` pins the JAX/MCTX Modal
  dependency pattern, but phase 1 should not call MCTX.
- `src/curvyzero/training/mctx_compact_search_service.py` is explicitly
  `profile_only`, `calls_train_muzero=False`, `trainer_defaults_changed=False`,
  and `touches_live_runs=False`; keep the parity proof labeled the same way.
- Existing tests already cover config builders, checkpoint metadata, checkpoint
  support-size inference, and MCTX profile-only labels:
  `tests/test_lightzero_config_builder.py`,
  `tests/test_lightzero_checkpoint_opponent_provider.py`,
  `tests/test_mctx_compact_search_service.py`.

## Smallest Implementation Slice

Add a script-first harness:

```text
scripts/probe_lightzero_jax_shadow_model_parity.py
```

The script should:

1. Accept `--checkpoint-path`, optional `--state-key`, `--seed`,
   `--batch-size`, `--device={cpu,cuda}`, `--jax-platform`, and
   `--output-json`.
2. Load the PyTorch policy with
   `load_lightzero_curvytron_visual_survival_policy(...)`.
3. Extract `policy._model.state_dict()` after strict load and build the matching
   JAX shadow model from the copied LightZero model config.
4. Convert weights once, outside any search loop.
5. Run deterministic one-batch parity:
   - `initial_inference(obs)` on zero, ramp, and seeded random `[B,4,64,64]`
     observations;
   - `recurrent_inference(latent, action)` using deterministic actions
     cycling through `0,1,2`.
6. Compare raw outputs, not search results:
   - policy logits, shape `[B,3]`;
   - value logits/support output;
   - reward logits/support output, including initial reward behavior;
   - latent/hidden state shape and numeric max error;
   - recurrent next latent shape and numeric max error.
7. Emit a compact JSON report with package versions, checkpoint ref/path,
   checkpoint SHA if local, model config surface, converted/unconverted key
   counts, per-output `max_abs`, `max_rel`, `allclose`, and loud safety labels:
   `profile_only=true`, `not_train_muzero=true`, `not_mctx=true`,
   `touches_live_runs=false`, `trainer_defaults_changed=false`.

Keep the script self-contained at first. Do not add a new trainer package API
until the converter has proven useful enough to share with MCTX.

## Modal Wrapper

After the local script exists, add a thin wrapper:

```text
src/curvyzero/infra/modal/lightzero_jax_shadow_model_parity.py
```

It should only resolve an immutable checkpoint ref from the `curvyzero-runs`
volume and invoke the same script/function in a fresh Modal app. It should use a
new app name such as `curvyzero-lightzero-jax-shadow-model-parity`, install
LightZero/Torch plus `jax==0.7.0` or `jax[cuda12]==0.7.0`, and write only an
isolated JSON artifact under an optimizer/parity attempt directory. It must not
spawn Coach, poll live runs, call `train_muzero`, write checkpoints, or patch
launch manifests.

Reject mutable checkpoint refs like `latest.pth.tar` and `ckpt_best.pth.tar` in
the wrapper. Prefer immutable `iteration_N.pth.tar` refs so a parity report is
reproducible and cannot race a live training run.

## Tests

Add:

```text
tests/test_lightzero_jax_shadow_model_parity.py
```

Small unit tests should not require JAX or LightZero:

- report labels stay profile-only and `trainer_defaults_changed=false`;
- mutable checkpoint refs are rejected by the Modal/ref parser;
- tolerance comparison fails on a known perturbation and reports the largest
  offending output;
- converter coverage reports unconverted keys instead of silently ignoring them.

Add one optional integration test guarded by `pytest.importorskip("jax")`,
`pytest.importorskip("torch")`, and `pytest.importorskip("lzero")`. It can build
a tiny freshly initialized LightZero MuZero model on CPU, convert those weights,
and run initial/recurrent parity without any checkpoint or trainer. Real
checkpoint parity should remain a script/Modal smoke, not a default pytest
requirement.

## Pass Criteria

The first proof passes when a local or Modal run on one immutable checkpoint
reports:

```text
strict checkpoint load ok
converted state_dict keys cover all required LightZero model weights
initial policy/value/reward outputs allclose within tolerance
recurrent policy/value/reward outputs allclose within tolerance
latent and next latent shapes match exactly
no train_muzero call
no MCTX call
no trainer default/config mutation outside copied objects
```

Use conservative starting tolerances: `atol=1e-4, rtol=1e-4` on CPU and a
separate explicit GPU tolerance only if Modal GPU introduces expected numeric
drift. Do not promote to MCTX search until raw model parity is green.

## Explicit Non-Goals

- No Coach run, no learner update, no replay buffer, no compact slab integration.
- No `set_load_ckpt_before_run(...)` in trainer configs for this proof.
- No change to `lightzero_config_builder` defaults.
- No pyproject default dependency changes; use `uv --with ...` locally or Modal
  image pins for JAX.
- No claim that MCTX is training-compatible. This only proves that JAX can
  represent the current LightZero PyTorch policy numerically.

