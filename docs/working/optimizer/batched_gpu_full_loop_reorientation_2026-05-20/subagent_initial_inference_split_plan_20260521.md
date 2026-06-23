# MuZero initial-inference split canary plan - 2026-05-21

Scope: profile-only planning for the optimizer lane. Do not call
`lzero.entry.train_muzero`, do not attach to live runs, and do not write
checkpoints/run manifests.

## Short answer

The clean split is:

```text
uint8 [B,2,4,64,64]
-> flatten legal roots to [N,4,64,64], N <= B*2
-> torch.float32 on policy model device
-> divide by 255 when source stack is uint8
-> policy._model.initial_inference(obs_tensor)
```

This prices the real MuZero representation + prediction network only. It
excludes `MuZeroPolicy.collect_mode.forward`, action masks, Dirichlet noise,
CPU ctree/MCTS search, output action decode, and scalar timestep materialization.

## Evidence

- Current profile-only collect canary already builds a scratch CurvyTron
  `MuZeroPolicy` in
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.
  Reuse `_build_profile_lightzero_policy(...)` and `_policy_model_device(...)`;
  do not duplicate live trainer setup.
- The current collect probe normalizes the same pre-scalar stack at
  `source_state_batched_observation_boundary_profile.py:3851`: `torch.as_tensor`
  to `float32` on the model device, then `* (1.0 / 255.0)` for `uint8`.
- Cached local LightZero 0.2.0 source exists under
  `/Users/shankha/.cache/uv/archive-v0/Cu6tu6aNf1Kgvui4lBtfL`. It is not
  importable from this repo's `.venv` or global Python without installing deps.
- LightZero `MuZeroPolicy._init_collect` sets `_collect_model = self._model`;
  collect-mode calls the same model's `initial_inference(data)` before detaching
  roots/logits to CPU for MCTS. So calling `policy._model.initial_inference(...)`
  is the right narrower split.
- LightZero `MZNetworkOutput` fields are `value`, `reward`, `policy_logits`,
  and `latent_state`.

## Expected API

Use the existing scratch policy:

```python
policy_bundle = _build_profile_lightzero_policy(
    seed=seed,
    use_cuda=use_cuda,
    num_simulations=1,          # irrelevant for model-only, keep metadata honest
    policy_batch_size=N,
    max_ticks=max_ticks,
)
policy = policy_bundle["policy"]
model = policy._model
model.eval()
device = _policy_model_device(policy)
```

Run:

```python
obs_tensor = torch.as_tensor(flat_stack, dtype=torch.float32, device=device)
if source_stack.dtype == np.uint8:
    obs_tensor = obs_tensor * (1.0 / 255.0)

with torch.no_grad():
    output = model.initial_inference(obs_tensor)
```

Add CUDA synchronizes around H2D, normalization, and inference when measuring on
CUDA. Warm up once and exclude policy build from measured rows.

## Input contract

- Source stack: `[B,2,4,64,64]`.
- Flatten row/player to `[B*2,4,64,64]`.
- If action masks are present, filter all-zero-mask roots exactly like the
  collect-forward probe so denominators match; otherwise run all roots and label
  `mask_filter_applied=false`.
- Device: same as `next(policy._model.parameters()).device`.
- Dtype/range: LightZero model input should be `torch.float32`; current CurvyTron
  policy model range is `[0,1]`, derived from raw `uint8` gray64 `[0,255]`.

## Output shapes

For `N` active roots with current conv MuZero source-state config:

- `output.policy_logits`: `[N, 3]`.
- `output.value`: `[N, value_support_size]`.
- `output.reward`: initial MuZero returns a Python zero list of length `N` in
  LightZero 0.2.0, despite the dataclass annotation saying tensor.
- `output.latent_state`: usually `[N, num_channels, 8, 8]` for the stock Atari
  downsampled `64x64` conv config; verify from `tuple(output.latent_state.shape)`
  instead of hardcoding.

`value_support_size`/`reward_support_size` come from the compiled CurvyTron
target config. The fixed-opponent default caps support scale at `300`, so large
normal training configs are commonly `601`; tiny profile configs can be smaller.

## Implementation plan

1. Add `_LightZeroInitialInferenceStackProbe` next to
   `_LightZeroCollectForwardStackProbe`.
2. Reuse `_build_profile_lightzero_policy`, `_policy_model_device`, root
   flattening, zero-mask filtering, and dtype/range checks from the collect
   probe.
3. Time these buckets: tensor prepare/filter, H2D, normalize, model
   `initial_inference`, optional shape/stat readback, total.
4. Return only compact telemetry: root count, input bytes, policy/model class,
   device, dtype/range, output shapes, finite checks, logits checksum, value
   support size, latent shape, and `semantics=lightzero_model_initial_inference_only`.
5. Keep scalar timestep materialization forced off. This canary should not decode
   actions or call MCTS.

## Caveats

- `policy._model` is a private LightZero field, but this repo already uses it in
  checkpoint probes/eval helpers, and LightZero collect/eval models alias it.
- `initial_inference` ignores legal action masks and `to_play`; logits are raw
  policy-head outputs for all three actions. Do not treat this as action
  selection quality.
- Do not read back full `latent_state` for timing unless explicitly measuring
  output-copy cost. Shape/stat-only readback is enough for this split.
- Label comparisons carefully: collect-forward wall minus initial-inference wall
  is only an approximate CPU tree/search/decode remainder because the code paths
  have different readback/decode work.
