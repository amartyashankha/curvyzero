# MCTX/JAX Batched Search For CurvyTron

Date: 2026-05-12

Status: architecture research note. This is not an implementation plan for this
turn and does not edit code.

## Plain Verdict

MCTX/JAX is a credible future search primitive if CurvyTron moves toward a
repo-owned MuZero-style actor/search/replay/learner stack. It is not a
drop-in replacement for LightZero as a training system.

The right mental model is:

```text
CurvyTron env/render/stack
  -> live ego policy rows
  -> JAX representation/prediction
  -> mctx.gumbel_muzero_policy
  -> action/action_weights/root diagnostics
  -> scatter back to joint_action[B,P]
```

MCTX could replace the tree-search and search-time model-inference box. It
would not replace the environment, visual stack, simultaneous action contract,
replay semantics, learner, checkpoint publisher, evaluator, or actor fleet.

Current local evidence is promising but incomplete. Modal L4 MCTX smokes already
run `mctx.gumbel_muzero_policy` on GPU, and the synthetic benchmark reached
about `12k` decisions/sec and `193k` simulations/sec for one flat `B=64`,
`num_simulations=16`, `hidden_dim=64`, `max_depth=16` profile. Later bridge
modes consumed real debug/scalar CurvyTron-shaped arrays, but no benchmark has
yet used the current visual root shape `float32[B,P,4,64,64]` with even a tiny
CNN representation.

## What MCTX Would Replace

MCTX would replace the policy/search implementation used to choose actions from
live ego rows:

- LightZero/PyTorch/C++ MCTS for collection-time action selection, if the actor
  loop is moved to a repo-owned search policy.
- Python or ad hoc policy-head-only placeholders in the current two-seat smoke
  lane.
- Search-time repeated neural inference, but only if the model is available as
  pure JAX functions: representation, dynamics, prediction.
- The search output surface: selected action, visit/action weights, root value,
  and optional tree diagnostics per ego row.

The replacement boundary should stay row-shaped:

```text
obs_roots[R,4,64,64]
invalid_actions[R,3]
row_mask[R]
params
search_config
  -> action[R], action_weights[R,3], root_value/search diagnostics
```

where `R` is normally padded to `B * P`.

## What MCTX Would Not Replace

MCTX is a search library, not the CurvyTron training system. It would not own:

- source-faithful env stepping or elapsed-ms CurvyTron physics;
- source-state visual rendering, frame stacking, player-perspective remapping,
  reset-row refresh, or browser/canvas fidelity questions;
- live/dead row filtering, policy-row mapping, or scatter back to
  `joint_action[B,P]`;
- simultaneous-action semantics;
- replay chunk writing, final observation handling, target construction, or
  sample-age/checkpoint metadata;
- learner updates, optimizer state, gradient scaling, or checkpoint publishing;
- evaluation, checkpoint promotion, scorecards, GIFs, or Coach-owned learning
  claims;
- Modal orchestration around actors/replay/checkpoints.

It also would not let us keep a PyTorch LightZero learner unchanged without
cost. MCTX's `recurrent_fn` must be JAX-traceable. A PyTorch model cannot sit
inside that compiled recurrent function. A serious MCTX path therefore implies
either a JAX model/learner or an awkward shadow-model conversion pipeline.

## Required Data Shapes

Current visual two-seat CurvyTron policy input is:

```text
obs_env:             float32[B,P,4,64,64] in [0,1]
legal_action_mask:   bool[B,P,3]
live_mask:           bool[B,P]
reward:              float32[B,P]
done/terminated/...: bool[B]
```

For MCTX search, flatten or pad to root rows:

```text
R = B * P fixed capacity for a compiled profile

obs_roots:        float32[R,4,64,64]
invalid_actions:  bool[R,3]     # MCTX polarity: true means invalid
row_mask:         bool[R]       # active live/legal rows
env_row_id:       int32[R]      # host metadata
player_id:        int16[R]      # host metadata
```

Use `build_policy_row_mapping(..., pad_to=B*P)` style semantics. Active rows
go first, padded/dead rows carry `row_mask=false`. Do not pass all-invalid
non-terminal rows to MCTX; give padded rows a harmless valid mask and ignore
their outputs.

The JAX/MCTX model-facing shapes should be:

```text
representation(obs_roots) -> hidden[R,H]          # first benchmark
prediction(hidden)       -> prior_logits[R,3], value[R]

RootFnOutput:
  prior_logits: float32[R,3]
  value:        float32[R]
  embedding:    float32[R,H]

recurrent_fn(params, rng_key, action[R], hidden[R,H]):
  reward:       float32[R]
  discount:     float32[R]
  prior_logits: float32[R,3]
  value:        float32[R]
  next_hidden:  float32[R,H]
```

Start with a vector latent, not a spatial latent. MCTX stores tree embeddings,
so memory grows roughly with `R * (num_simulations + 1) * hidden_size`. A vector
hidden of `H=64` is boring; a spatial latent like `[64,8,8]` is thousands of
floats per node and quickly becomes a first-order memory constraint.

## GPU Batching Model

The useful GPU batch is the root batch, not one environment row at a time.

1. CPU actors/env workers produce `obs_env[B,P,4,64,64]` and
   `legal_action_mask[B,P,3]`.
2. Policy-row mapping pads to `R=B*P`.
3. Host arrays are copied with `jax.device_put`. Keep params resident on device.
4. A jitted `run_search(...)` builds the root and calls
   `mctx.gumbel_muzero_policy`.
5. The GPU runs representation/prediction and recurrent model calls across all
   roots in parallel inside the search.
6. Only small outputs need to come back before stepping the env:
   `action[R]`, plus `action_weights[R,3]` and root diagnostics for replay.
7. Host code scatters active-row actions back to `joint_action[B,P]`.

Static profile values should include batch/root capacity, observation shape,
action count, hidden shape, model architecture, `num_simulations`, and
`max_depth`. Tail batches should be padded, not compiled as new shapes. Timers
must separate compile-plus-first-run, host observation setup, host-to-device
placement, steady search, and device-to-host output transfer.

Multi-GPU is not the first MCTX question. The first question is whether one GPU
can be fed with a useful root batch. Later scaling can split roots across
devices or run one actor/search process per GPU, but that only matters after
single-GPU search is both correct and first-order in the profile.

## Smallest Next Benchmark

Run one visual-root benchmark, still without trainer/replay integration:

```text
VectorMultiplayerEnv or existing two-seat visual-stack sample
  -> SourceStateGray64Stack4.update(...)
  -> obs_env float32[B,2,4,64,64]
  -> legal_action_mask bool[B,2,3]
  -> padded roots R=B*2
  -> tiny JAX CNN representation
  -> mctx.gumbel_muzero_policy
```

Suggested first matrix:

| Env rows `B` | Roots `R` | Sims | Hidden | Goal |
| ---: | ---: | ---: | ---: | --- |
| 8 | 16 | 8 | 64 | compile and shape smoke |
| 16 | 32 | 16 | 64 | first useful L4 timing |
| 64 | 128 | 16 | 64 | compare to current LightZero search bucket |

Report at minimum:

- source tensor shape and dtype;
- root tensor shape, active rows, padded rows, mask polarity;
- host visual setup time;
- host-to-device transfer steady median;
- compile-plus-first-search time;
- steady search median, decisions/sec, simulations/sec;
- device-to-host transfer for actions and action weights;
- GPU backend/device and `nvidia-smi` memory snapshot;
- action histogram;
- finite/normalized `action_weights`;
- no all-invalid active rows.

Pass condition: the benchmark runs on Modal L4 with finite normalized search
weights, no surprise recompilation inside a fixed profile, and enough timing
detail to compare against LightZero's `policy_search_sec`. It should not claim
learning, target correctness, or CurvyTron self-play quality.

## Risks And Costs

The largest cost is framework ownership. A real MCTX path means owning a JAX
MuZero model, optimizer, replay sampler, checkpoint format, and actor policy
refresh contract, or maintaining a brittle PyTorch-to-JAX shadow model.

The algorithmic risk is simultaneous action semantics. Standard MCTX MuZero
search expands one action per root row. Independent per-seat search with
`A=3` is computationally sane, but the learned dynamics must implicitly model
the opponent distribution. A `9`-action joint controller is feasible for 2P
as a control, but it changes the problem into centralized joint-action search.

The systems risks are also real:

- JAX recompiles on shape/static-argument changes.
- Tree memory grows with root count, simulations, and hidden size.
- CPU visual rendering or host/device transfer can dominate, making faster
  search irrelevant in a render-bound profile.
- MCTX mask polarity and all-invalid rows can silently create bad action
  behavior if not asserted.
- Value perspective, reward sign, root exploration, temperature, and terminal
  discount semantics must be explicit; MCTX will not infer CurvyTron's player
  convention.
- Modal/JAX/CUDA version pins and compile caches become part of operations.
- XLA/JAX errors are less transparent than ordinary Python/PyTorch failures.

## Recommendation

Keep MCTX/JAX as the serious future search primitive, but do not migrate the
training system now. The next responsible action is the visual-root benchmark
above. If `float32[B,2,4,64,64] -> tiny CNN -> MCTX` is fast, stable, and easy
to profile, MCTX becomes a stronger candidate for a repo-owned MuZero actor
loop. If it is slow, memory-heavy, or dominated by visual setup/transfer, then
LightZero stock controls and coarse actor fanout remain the better near-term
path.

## Sources

Local sources read:

- `docs/working/optimizer/framework_reassessment_2026-05-11.md`
- `docs/working/optimizer/framework_replication_controls_2026-05-11.md`
- `docs/experiments/2026-05-09-modal-mctx-synthetic-benchmark.md`
- `docs/experiments/2026-05-09-modal-mctx-dependency-smoke.md`
- `docs/research/mctx_integration.md`
- `docs/working/optimizer/actor_loop_architecture_2026-05-09.md`
- `docs/working/optimizer/full_training_loop_worldview_2026-05-11.md`
- `docs/working/optimizer/search_hardware_scaling_sidecar_2026-05-12.md`
- `src/curvyzero/infra/modal/mctx_dependency_smoke.py`
- `src/curvyzero/infra/modal/mctx_gpu_dependency_smoke.py`
- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `src/curvyzero/training/policy_row_mapping.py`
- `src/curvyzero/training/curvytron_current_policy_selfplay_smoke.py`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`

External primary sources:

- MCTX README: https://github.com/google-deepmind/mctx
- JAX `jit`: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
- JAX `device_put`: https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html
- JAX asynchronous dispatch and benchmarking caution:
  https://docs.jax.dev/en/latest/async_dispatch.html
- JAX `pmap`/multi-device note:
  https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html
