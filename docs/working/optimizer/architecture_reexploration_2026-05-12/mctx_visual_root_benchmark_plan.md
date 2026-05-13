# MCTX Visual-Root Benchmark Plan

Date: 2026-05-12

Status: concrete benchmark plan. This is not a migration plan and makes no
learning claim.

## Question

Can one GPU run MCTX search fast enough on current CurvyTron visual roots to
justify a next architecture step?

The benchmark target is only:

```text
VectorMultiplayerEnv
  -> SourceStateGray64Stack4.update(...)
  -> obs_env float32[B,2,4,64,64]
  -> build_policy_row_mapping(..., pad_to=B*2)
  -> obs_roots float32[R,4,64,64], invalid_actions bool[R,3]
  -> tiny JAX CNN representation
  -> mctx.gumbel_muzero_policy
  -> action/action_weights/root timing
```

## What Existing MCTX Evidence Proves

`mctx_gpu_dependency_smoke.py` proves a Modal cheap GPU can import CUDA JAX,
import MCTX, and execute one tiny Gumbel MuZero search.

`mctx_synthetic_benchmark.py` proves more:

- fixed-shape jitted `mctx.gumbel_muzero_policy` runs on Modal GPU;
- timings are already split into host setup, host-to-device transfer,
  compile-plus-first-run, steady search, device-to-host action output, and
  `nvidia-smi` memory snapshot;
- MCTX mask polarity is wired as `invalid_actions=True`;
- output checks catch non-finite or non-normalized `action_weights`;
- CurvyTron-shaped scalar/debug modes can build or consume real-ish
  `[B,P,...]` arrays, filter live policy rows, pad to `B*P`, and search fixed
  root capacity;
- one reported synthetic flat profile reached about `12k` decisions/sec and
  `193k` simulations/sec at `R=64`, `num_simulations=16`, `hidden_dim=64`,
  `max_depth=16`.

It does not prove:

- current source-state visual root throughput;
- `float32[B,2,4,64,64]` transfer cost;
- tiny CNN representation cost;
- tree memory behavior with visual-root batch sizes;
- whether visual setup dominates search;
- no surprise JAX recompilation for visual profiles;
- parity with LightZero `policy_search_sec`;
- learning, target correctness, replay correctness, or self-play quality.

## Minimal Benchmark Shape

Use the current two-seat visual stack, not the old scalar trainer:

- env: `VectorMultiplayerEnv(batch_size=B, player_count=2)`;
- visual stack: `SourceStateGray64Stack4(batch_size=B, player_count=2)`;
- default render mode: `browser_lines`; optional comparison:
  `body_circles_fast`;
- source observation: `obs_env float32[B,2,4,64,64]`, normalized `[0,1]`;
- source legal mask: `legal_action_mask bool[B,2,3]`;
- live mask: `legal_action_mask.any(axis=2)` or explicit alive/live source if
  available in the sampled batch;
- mapping: `build_policy_row_mapping(obs_env, live_mask, legal_action_mask,
  pad_to=B*2)`;
- roots: `R = B * P = B * 2`, static for each benchmark profile;
- root observation: `obs_roots float32[R,4,64,64]`;
- invalid mask: `invalid_actions bool[R,3] = ~mapping.legal_action_mask`;
- padded rows: `row_mask=False`, harmless valid action mask if MCTX cannot
  tolerate all-invalid rows, outputs ignored;
- action count: `3`, independent per-seat search, not 9-action joint search;
- hidden: vector latent, `H=64` first; avoid spatial tree embeddings;
- model: tiny JAX CNN representation, then linear prediction/dynamics;
- dtype: start with `float32`; do not mix in `bfloat16` until the baseline is
  timed and numerically clean;
- static args/shapes: `B`, `P=2`, `R`, observation shape, `A=3`, hidden dim,
  CNN architecture, `num_simulations`, `max_depth`.

First matrix:

| Env rows B | Roots R | Sims | Hidden | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 8 | 16 | 8 | 64 | shape and compile smoke |
| 16 | 32 | 16 | 64 | first useful L4 timing |
| 64 | 128 | 16 | 64 | compare against LightZero search bucket |
| 64 | 128 | 32 | 64 | stress tree/search scaling |

Use `warmup_runs >= 2` and `steady_runs >= 10` for the useful profiles.

## Timing Buckets

Report JSON with these buckets, each p50 and preferably p95 when repeated:

- `visual_setup_sec`: env reset/step sample, `SourceStateGray64Stack4.update`,
  reset-row refresh if used, and policy-row mapping. Also break out
  `env_step_sec`, `visual_stack_update_sec`, and `policy_row_mapping_sec`.
- `host_to_device_transfer_sec`: `jax.device_put(obs_roots)` and
  `jax.device_put(invalid_actions)`, blocked with `.block_until_ready()`.
- `compile_plus_first_run_sec`: first jitted visual CNN + MCTX search run,
  blocked on `action_weights`.
- `steady_search_sec`: warmed fixed-profile search only, with params already
  resident on device. Report decisions/sec as `R / sec` and simulations/sec as
  `R * num_simulations / sec`.
- `device_to_host_action_output_sec`: copy `action[R]`; separately copy
  `action_weights[R,3]` and root diagnostics if collected.
- `gpu_memory`: `nvidia-smi` before first run, after compile, and after steady
  search; include device name, total/used memory, JAX/JAXLIB/MCTX versions.

Also report:

- source and root shapes/dtypes;
- active vs padded row counts;
- mask polarity and all-invalid active-row assertion;
- action histogram;
- finite `action_weights`;
- row sums close to `1.0`;
- whether any extra compile happened inside steady fixed-shape runs.

## Repo Edit Requirement

This does not need edits to existing trainer, env, LightZero, or MCTX benchmark
code.

The clean implementation should be a new isolated Modal script, probably beside
the existing MCTX scripts, that imports current repo utilities and reuses the
existing benchmark timing style. That script can be deleted or promoted later.
Do not thread MCTX into the trainer, actor bridge, replay, or LightZero path for
this benchmark.

## Pass / Fail Criteria

Pass means MCTX is worth one next step only if all of these are true on Modal L4:

- visual profile runs for `B=16,R=32,sims=16` and `B=64,R=128,sims=16`;
- outputs are finite, normalized, and legal for active rows;
- no fixed-profile recompilation during steady runs;
- GPU memory stays comfortably under L4 capacity, with headroom for a larger
  model later;
- timing buckets are complete enough to compare against trusted LightZero
  `policy_search_sec`;
- steady visual-root search is clearly first-order competitive: either faster
  than the comparable LightZero search bucket, or fast enough that
  `visual_setup_sec + H2D + steady_search + D2H` would not worsen the collector
  loop;
- visual setup/transfer is not so dominant that search acceleration becomes
  irrelevant.

Fail means pause MCTX if any of these happen:

- visual roots or tiny CNN cause unstable compile/runtime failures;
- `B=64,R=128,sims=16` does not fit or leaves poor memory headroom;
- steady search is slower than the trusted LightZero search bucket after
  warmup;
- H2D/D2H plus visual setup dominates enough that faster search cannot improve
  wall-clock collection;
- masks/padded rows require nontrivial semantic work before the benchmark is
  trustworthy.

Passing this benchmark should authorize only a narrower follow-up: a real-model
or LightZero-comparison profiling spike. It should not authorize a training
system migration.
