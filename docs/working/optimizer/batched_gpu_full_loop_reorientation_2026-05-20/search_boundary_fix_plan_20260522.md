# Search Boundary Fix Plan

Date: 2026-05-22

Status: active optimizer working memory. Profile-only unless explicitly
promoted by later full-loop gates.

## Plain Read

The old `~1.8x` direct CTree result was not the end of the story. It removed a
lot of public LightZero wrapper/output work, but it still copied MuZero latent
states between GPU and CPU every MCTS simulation.

The new tactical fix is:

```text
direct_ctree_gpu_latent
```

It keeps LightZero CTree selection/backprop on CPU, but keeps MuZero hidden
state tensors on GPU across simulations. CTree still gets only the CPU values it
needs: reward, value, policy logits, actions, and tree stats.

## Current Evidence

Profile-only H100 denominator:

```text
B512, A16, 60 measured steps, 15 warmup, uint8 stack,
no scalar materialization, no death, no live trainer changes
```

Sim8:

| impl | measured sec | scalar steps/sec | search sec | model sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stock_facade` | `26.99` | `2276.71` | public collect/search `18.56s` | `0.96s` | `2.45s` |
| `direct_ctree_arrays` | `13.45` | `4568.28` | `6.04s` | `0.86s` | `2.21s` |
| `direct_ctree_gpu_latent` pre-pool | `9.34` | `6580.32` | `1.89s` | `0.78s` | `2.68s` |
| `direct_ctree_gpu_latent` post-pool | `9.43` | `6518.13` | `2.09s` | `0.79s` | `2.36s` |

Sim16:

| impl | run | measured sec | scalar steps/sec | boundary total | search sec | model sec | observation sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock_facade` | `ap-cRxL8lanazTY29fnKhCQx7` | `35.43` | `1734.05` | `30.40s` | public collect/search `27.14s` | `1.61s` | `2.27s` |
| `direct_ctree_arrays` | `ap-zDtVC6Lw2LcPocY2AtTzRc` | `19.92` | `3083.58` | `15.10s` | `13.30s` | `1.62s` | `2.15s` |
| `direct_ctree_gpu_latent` | `ap-3v1vn4EGdk8YfJGIIVdiWD` | `12.60` | `4874.42` | `7.05s` | `4.69s` | `1.59s` | `2.50s` |

Plain ratio:

```text
At sim8, direct_ctree_gpu_latent is about 2.9x faster than stock facade
and about 1.4x faster than direct_ctree_arrays.

At sim16, direct_ctree_gpu_latent is about 2.8x faster than stock facade
and about 1.6x faster than direct_ctree_arrays.
```

Important caveat:

```text
These 2.8x/2.9x numbers are profile-only boundary roots/sec, not full-loop
train_muzero throughput. The later C64/sim16/3-learner train_muzero repeat was
stock 445.19 steps/sec versus direct 438.56 steps/sec. Use this file as
boundary history, not Coach launch advice.
```

Validation so far:

- Tiny Modal L4 smoke passed with CUDA and `gpu_latent_enabled=true`.
- Focused local boundary tests passed: `8 passed`.
- Local stock-vs-direct-vs-GPU-latent compare, 8 seeds, sim8, root noise on:
  `1.0` action agreement, `0` visit L1, `0` value diff, `0` illegal actions.
- The compare helper now records requested/actual CUDA status, predicted-value
  and policy-logit diffs, and has explicit hard-gate thresholds. The clean
  local exact gate is sim8/root-noise-on; sim2/tie-heavy rows can diverge in
  visits even when logits/values are identical, so do not use tiny tie rows as
  an exact distribution gate.

This is still not Coach training advice. It is a profile-only boundary result.

## What Changed In Code

Files:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `scripts/compare_curvytron_direct_ctree_stock.py`
- `scripts/build_curvytron_hybrid_observation_profile_grid.py`
- `tests/test_source_state_batched_observation_boundary_profile.py`
- `tests/test_curvytron_hybrid_observation_profile_grid_builder.py`

Code changes:

- Added `direct_ctree_gpu_latent` profile impl.
- Added GPU-latent timers and bytes.
- Added comparison-tool support for both direct paths against stock.
- Added GPU-latent to mixed-mask and value parity tests.
- Added a fixed-denominator manifest preset:

```text
uv run python scripts/build_curvytron_hybrid_observation_profile_grid.py \
  --next-direct-ctree-comparison-preset
```

## Amdahl Read

At this denominator, the wall moved:

```text
render is no longer the main blocker.
search/boundary work is now the main blocker.
```

The GPU-latent path made search much cheaper, but not free. Remaining visible
costs:

- CPU CTree traverse/backprop.
- Python loop per simulation.
- reward/value/policy GPU-to-CPU copies each simulation.
- Python `.tolist()` for CTree backprop.
- root preparation and final visit/value extraction.
- host stack update and observation packaging.

The preallocated latent-pool cleanup did not materially improve sim8. Keep it
only if it stays clean and does not regress future rows; the proven win is the
larger design choice of keeping latents on GPU, not the pool micro-optimization.

## Next Experiments

Run in parallel when compute is free:

1. Add L4/T4 comparison only after H100 settles; L4 is useful for cost advice,
   not the main architecture proof.
2. Reconnect only the best profile path to a real `train_muzero` full-loop
   profile after parity and profile gates pass.
3. Start the next big-swing prototype in parallel: either dense Torch GPU MCTS
   for fixed `A=3` or an array-native Cython CTree boundary. GPU-latent is a
   good bridge, not a 10x endpoint.
4. Finish one tiny CUDA parity canary with debug arrays included:
   stock/direct/GPU-latent, `roots <= 16`, sim8, same seed. This proves actual
   CUDA behavior instead of only CPU tensors running the same algorithm.

## Architecture Lanes

Tactical lane:

```text
Finish GPU-latent CTree gates.
```

Medium lane:

```text
Vendor/extend LightZero Cython CTree so root/output and then
traverse/backprop take arrays instead of Python lists.
```

Ambitious lane:

```text
Build a dense Torch GPU MCTS profile proof for A=3 and sim8/sim16.
```

Research lane:

```text
MCTX/JAX or MiniZero/KataGo-style batched leaf inference.
```

## Decision Rule

- If GPU-latent remains `>=1.3x` over direct CTree and parity stays clean,
  finish it as the next compact collector/search bridge candidate.
- If it drops below `~1.15x`, stop polishing and move to dense GPU search or
  Cython array-native CTree.
- A real Coach recommendation requires full-loop `train_muzero` profile proof,
  not just boundary roots/sec.

## Dense Torch MCTS Update

The first dense Torch MCTS prototype has landed as profile-only mode:

```text
lightzero_array_ceiling_mode=dense_torch_mcts
```

It keeps tree arrays on the model device, but it is not stock LightZero CTree.
Treat it as a speed/architecture probe.

Matched H100 row:

| impl | measured sec | scalar steps/sec | probe/search boundary | read |
| --- | ---: | ---: | ---: | --- |
| `direct_ctree_gpu_latent` repeat | `10.24` | `6001.49` | `4.68s` | current CPU CTree + GPU latents |
| `dense_torch_mcts` v0 | `9.57` | `6418.31` | `3.03s` | GPU tree prototype |
| `recurrent_toy` ceiling | `8.23` | `7466.60` | `2.58s` | fake search ceiling |

Conclusion:

```text
Dense MCTS v0 confirms the direction but not the magnitude. The next bottleneck
inside dense mode is self-inflicted Python/sync/allocation overhead, not raw
neural inference.
```

Code follow-up already applied:

- no inner-loop `.item()` CPU sync checks;
- path history tensors are preallocated;
- backprop no longer clones the bootstrap vector every depth.

Cleanup result:

| impl | measured sec | scalar steps/sec | probe/search boundary | read |
| --- | ---: | ---: | ---: | --- |
| `direct_ctree_gpu_latent` same-run repeat | `10.37` | `5927.42` | `4.70s` | CPU CTree + GPU latents |
| `dense_torch_mcts` cleanup v1 | `7.96` | `7720.30` | `2.59s` | dense GPU tree prototype |
| `recurrent_toy` same-run ceiling | `6.75` | `9097.07` | `2.09s` | fake search ceiling |

This keeps dense GPU search as the leading high-upside lane for now.

Second cleanup now in code:

- no per-recurrent CUDA sync in dense profile mode;
- all-roots-legal fast path for array-ceiling input prep;
- reverted `torch.inference_mode()` after local tests caught an in-place
  bootstrap update failure on inference tensors.

Second-cleanup H100 sim16 read:

| impl | sims | measured sec | scalar roots/sec | probe/boundary sec | search/update sec | model/recurrent sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_torch_mcts` cleanup v2 | `16` | `14.857` | `4135.37` | `5.262` | `2.514` | `1.556` | `2.394` |
| `direct_ctree_gpu_latent` | `16` | `12.263` | `5010.39` | `6.766` | `4.527` | `1.523` | `2.480` |
| `recurrent_toy` ceiling | `16` | `6.727` | `9133.74` | `2.416` | `0.143` | `1.234` | `2.089` |

Plain read:

```text
The eager dense path is not automatically better just because it is on GPU.
At sim16 it loses to LightZero CTree with GPU-resident latents. The likely
problem is many small dynamic Torch operations and Python-controlled tree
depth, not the recurrent model.
```

Third cleanup now in code:

- fixed-shape all-root masked selection/expansion/backprop in dense mode;
- fewer dynamic boolean-indexed tensor slices in the per-depth loop;
- local ruff, py_compile, and focused array-ceiling tests passed.

Next branch decision:

- If the fixed-shape dense row moves near the recurrent toy ceiling, keep pushing
  dense GPU search.
- If it remains near GPU-latent CTree, switch to an array-native Cython CTree
  patch or a bigger batched-search-service design.

Fresh branch result:

| impl | sims | measured sec | scalar roots/sec | read |
| --- | ---: | ---: | ---: | --- |
| `direct_ctree_gpu_latent` | 8 | `8.141` | `7547.12` | practical LightZero-shaped baseline |
| `dense_torch_mcts` fixed-shape after semantic fix | 8 | `7.413` | `8288.37` | fast shallow profile-only probe |
| `direct_ctree_gpu_latent` | 16 | `9.998` | `6145.25` | practical sim16 winner |
| `dense_torch_mcts` fixed-shape after semantic fix | 16 | `14.309` | `4293.88` | loses at deeper search |

Branch decision:

```text
Do not promote dense_torch_mcts to Coach training. It is not stock CTree and it
fails the sim16 scaling gate. Keep direct_ctree_gpu_latent as the best tactical
LightZero-shaped boundary. For a larger win, stop doing small eager Torch
polish and start either a compiled/fused fixed-shape dense search spike or an
array-native CTree API design.
```

See also:

- `search_boundary_next_fix_strategy_20260522.md`
