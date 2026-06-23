# Search Boundary Next Fix Strategy

Date: 2026-05-22

Status: active optimizer working memory.

## Plain Answer

Moving "everything to C" is not the right description of the next fix.

The real target is narrower:

```text
Stop crossing Python / CPU / list boundaries inside every MCTS simulation.
```

LightZero already uses C++ for the CTree traverse/backprop kernels. The slow
part is the shell around those kernels: Python loops, Python lists, CPU tree
state, GPU-to-CPU model-output copies, root setup, output extraction, and
policy/search packaging.

## What We Know

Current profile-only H100 denominator:

```text
B512, A16, sim8, 60 measured steps, 15 warmup, uint8 stack,
direct_gray64 persistent policy renderer, no scalar materialization, no death
```

| row | roots/sec | measured sec | read |
| --- | ---: | ---: | --- |
| stock public facade | `2276.71` | `26.99` | baseline LightZero public collect/search |
| direct CTree arrays | `4568.28` | `13.45` | removes public wrapper/output fanout |
| direct CTree GPU-latent | `6580.32` | `9.34` | keeps hidden states on GPU |
| dense Torch MCTS v0 | `6418.31` | `9.57` | GPU tree prototype, not LightZero CTree |
| dense Torch MCTS sync/allocation cleanup | `7720.30` | `7.96` | removed `.item()` gates and path churn |
| recurrent toy ceiling | `7466.60` | `8.23` | recurrent model pressure with fake search |

Plain read:

```text
GPU-latent CTree already removed the worst repeated latent CPU/GPU copies.
Dense GPU MCTS v0 did not blow past it, but the first cleanup did. Removing
CPU sync gates and allocation churn took dense mode to about 1.30x over the
same-run GPU-latent CTree row.
```

## Current Amdahl Bottleneck

For this denominator, render is not the wall. The wall is the collect/search
boundary.

Inside that wall, the next visible buckets are:

- Python per-simulation control.
- CPU CTree state and list-shaped APIs.
- recurrent reward/value/policy CPU copies.
- root preparation and final visit/value extraction.
- stack packaging / host observation handoff.

So the next fixes should attack these boundaries, not only raw drawing speed.

## Ranked Fix Lanes

1. **Clean up dense Torch MCTS prototype**
   - Current impact: `6418 -> 7720 roots/sec` on the H100 sim8 row after the
     first cleanup, about `1.20x` over dense v0 and `1.30x` over the same-run
     GPU-latent CTree row.
   - Why first: smallest high-upside patch already in our code.
   - Concrete work: remove `.item()` loop gates, preallocate path tensors, avoid
     per-depth clones, reduce syncs, then retest H100 B512/A16/sim8 and sim16.
   - Risk: still profile-only and not exact LightZero CTree semantics.

2. **Array-native Cython/C++ CTree boundary**
   - Expected impact: likely `1.1x-1.4x` over GPU-latent full row, maybe
     `1.2x-1.8x` on the boundary slice.
   - Why not "all C": CTree kernels already are C++; the missing part is
     array-shaped APIs around them.
   - Concrete work: add dense `prepare/traverse/backprop/output` APIs for
     fixed `A=3`, removing Python lists and `.tolist()` calls.
   - Risk: vendored LightZero extension and Cython maintenance.

3. **True batched GPU search service**
   - Expected impact: the actual `10x`-class architecture if it works.
   - Concrete shape: many roots/trees -> one batched recurrent model call per
     simulation -> compact visit/action output, with minimal host sync.
   - This can be Torch, JAX/MCTX, or a custom CUDA/C++ path.
   - Risk: bigger semantic rewrite. Needs forced-case parity and statistical
     gates before touching training.

4. **C++/CUDA environment/search integration**
   - Expected impact: only worth it after the exact host boundary is proven.
   - Concrete shape: a C++/CUDA module that owns tree arrays and maybe the
     simple game step/observation state.
   - Risk: high. Easy to spend time and still leave Python orchestration as the
     real bottleneck.

5. **Scale architecture: actor/search batching**
   - Expected impact: production-scale improvement if trainer can digest it.
   - Concrete shape: many actors feed a central batched search/model service.
   - Risk: not a small patch; changes collection architecture and replay
     cadence. Important research lane, not the next local bugfix.

## Current Local Action

Dense Torch MCTS v0 has been patched to remove the obvious inner-loop CPU sync
checks and reduce path allocation. The next proof is a same-shape H100 repeat:

```text
dense_torch_mcts after sync cleanup
vs direct_ctree_gpu_latent
vs recurrent_toy ceiling
```

The first cleanup repeat passed this gate: `7720 roots/sec` versus same-run
`5927 roots/sec` for GPU-latent CTree and `9097 roots/sec` for recurrent toy.
That keeps dense GPU search alive.

The second cleanup removed per-recurrent CUDA sync in dense mode and uses the
all-roots-legal fast path. It nudged sim8 from `7720` to `7969` roots/sec, but
sim16 dropped to `4135` roots/sec. That means eager dense Torch has a real
scaling problem as simulation count/depth grows.

Same-denominator sim16 reality check:

| impl | sims | measured sec | scalar roots/sec | probe/boundary sec | search/update sec | model/recurrent sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_torch_mcts` cleanup v2 | `16` | `14.857` | `4135.37` | `5.262` | `2.514` | `1.556` | `2.394` |
| `direct_ctree_gpu_latent` | `16` | `12.263` | `5010.39` | `6.766` | `4.527` | `1.523` | `2.480` |
| `recurrent_toy` ceiling | `16` | `6.727` | `9133.74` | `2.416` | `0.143` | `1.234` | `2.089` |

Plain read:

```text
Dense eager Torch wins at sim8 after cleanup, but loses to GPU-latent CTree at
sim16. The gap is not neural inference. It is the search/control/update shell:
many small tensor ops, dynamic indexing, Python loop structure, and kernel
launch overhead.
```

Current code has one more profile-only rewrite pending measurement: dense mode
now uses all-root fixed-shape masked updates for selection/expansion/backprop
instead of dynamic boolean-indexed slices for the common all-roots case. Local
ruff, py_compile, and focused tests passed. H100 sim8/sim16 repeats are running.

Current working rule:

```text
For sim8, dense GPU search is the best profile-only row so far.
For sim16+, dense eager Torch is not acceptable until we compile/fuse or
otherwise change the search topology.
```

The next useful dense work is not another small eager-PyTorch polish; it is
`torch.compile`/CUDA graphs/Triton-style fusion, or else switch to the
array-native CTree lane.

Fresh measurement update:

| impl | sims | measured sec | scalar roots/sec | read |
| --- | ---: | ---: | ---: | --- |
| `direct_ctree_gpu_latent` | 8 | `8.141` | `7547.12` | best practical LightZero-shaped boundary |
| `dense_torch_mcts` fixed-shape after semantic fix | 8 | `7.413` | `8288.37` | faster, but profile-only/non-CTree |
| `direct_ctree_gpu_latent` | 16 | `9.998` | `6145.25` | practical sim16 winner |
| `dense_torch_mcts` fixed-shape after semantic fix | 16 | `14.309` | `4293.88` | fails sim16 scaling gate |

Plain decision:

```text
Use direct_ctree_gpu_latent as the practical baseline. Dense Torch is useful
evidence, but not a training recommendation. The next big step must remove
Python/list/search-boundary work with either compiled/fused dense search or
array-native CTree APIs.
```

## Validation Rule

No trainer promotion from profile-only search rows.

Required gates:

- forced-action exact tests;
- mask and illegal-action tests;
- root-noise statistical tests;
- CUDA small-root compare;
- then one matched full-loop profile with the same RND/death/checkpoint knobs
  as the Coach run.

## External Pattern Notes

- MCTX is the clean reference for JAX-native, batched, JIT-compiled MCTS:
  https://github.com/google-deepmind/mctx
- MiniZero is a useful C++/Python zero-knowledge framework reference for
  AlphaZero/MuZero style training:
  https://github.com/rlglab/minizero
- KataGo is useful as a systems reference for batched neural inference around
  search:
  https://github.com/lightvector/KataGo
- PyTorch CUDA graphs / `torch.compile(mode="reduce-overhead")` are relevant
  if the dense loop becomes fixed-shape enough to replay/fuse:
  https://docs.pytorch.org/docs/stable/notes/cuda.html
