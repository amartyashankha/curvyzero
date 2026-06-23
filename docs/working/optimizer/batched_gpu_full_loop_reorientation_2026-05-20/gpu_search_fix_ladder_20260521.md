# GPU Search Fix Ladder

Date: 2026-05-21

Status: active optimizer working memory.

## Plain Problem

The current `~1.8x` speedup is bounded because `direct_ctree_arrays` still uses
LightZero's CPU CTree search loop.

The key distinction:

- CTree selection/backprop itself is C++ and relatively small.
- The search loop around it still moves latent states and model outputs through
  Python, NumPy, lists, and CPU/GPU sync points.

So the fix is not "more rendering optimization" and not just "bigger GPU".
The fix is to keep the search data in a compact batched form for longer.

## Current Tactical Lane

The repo already has a profile-only mode:

```text
direct_ctree_gpu_latent
```

File:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
```

What it does:

1. Runs LightZero `initial_inference` on GPU.
2. Keeps `latent_state_roots` as a torch tensor.
3. Still uses LightZero CPU CTree for `batch_traverse` and
   `batch_backpropagate`.
4. Gathers leaf latents on GPU with torch indexing instead of rebuilding them
   from CPU NumPy every simulation.
5. Sends only the CTree-required reward, value, and policy logits to CPU.

This is the smallest semantics-preserving experiment because it keeps the real
LightZero model and real LightZero CTree tree statistics.

Expected impact:

- It can beat `direct_ctree_arrays` only if the repeated latent CPU round trip
  is a meaningful part of `search_sec`.
- It cannot remove the CPU CTree wall, root Python list prep, final
  `get_distributions`, or CTree-required value/reward/policy CPU lists.
- If it wins, expect a tactical win, not a 10x win.

Validation already seen locally:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "gpu_latent or direct_ctree"
-> 2 passed

uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py
-> passed
```

Active profile wave:

```text
H100 B512/A16/sim8/60 measured/15 warmup:
- stock_facade
- direct_ctree_arrays
- direct_ctree_gpu_latent

L4/T4 B512/A16/sim8/60 measured/15 warmup:
- direct_ctree_gpu_latent
```

Decision rule:

- If `direct_ctree_gpu_latent` improves H100 search wall materially, finish
  semantic gates and consider it for the next compact collector prototype.
- If it regresses or barely moves, do not polish it. Move to dense tensor search
  or a C++/Cython boundary.

## Still CPU In The Tactical Lane

These pieces still go through CPU:

- CTree roots and legal-action lists.
- Dirichlet root noise lists.
- `to_play`.
- CTree `batch_traverse`.
- CTree `batch_backpropagate`.
- Reward/value/policy logits for node expansion.
- Final root visit/value extraction.
- Final action sampling and compact array assembly.

That means `direct_ctree_gpu_latent` is only a partial fix.

## More Ambitious Lane 1: Dense Torch MCTS

Build a profile-only dense tensor MCTS for CurvyTron's tiny discrete action
space.

Shape:

```text
obs [N,4,64,64] on GPU
mask [N,A] on GPU
tree arrays [N,num_simulations+1,A] on GPU
latent pool [N,num_simulations+1,...] on GPU
```

Why this is attractive:

- Action count is tiny.
- Sim count is usually small enough for fixed dense buffers.
- It removes CPU CTree, Python list policy batches, and repeated NumPy glue.
- It can be tested against LightZero CTree as a profile-only search
  replacement.

Risks:

- It is no longer using LightZero CTree.
- PUCT, value normalization, root noise, support transforms, and visit targets
  must be copied carefully.
- Exact tie behavior is not the right gate, but forced-case semantics and
  distributional checks must pass.

This is probably the best next "big swing" if GPU-latent CTree is not enough.

## More Ambitious Lane 2: Array-Native CTree Boundary

Patch/fork LightZero CTree Cython/C++ so it accepts arrays instead of Python
lists for:

- root policies/rewards/noises;
- recurrent reward/value/policy batches;
- final visit/value extraction.

This keeps LightZero CTree semantics but cuts Python list conversion.

Risks:

- Requires maintaining a local LightZero CTree fork or vendored extension.
- Still CPU tree.
- Still recurrent model is Python/PyTorch unless we also move model calls into
  C++/libtorch.

This is useful if `direct_ctree_gpu_latent` shows the CTree semantics are worth
keeping but Python list conversion remains the wall.

## More Ambitious Lane 3: MiniZero/KataGo-Style Service

External pattern:

- MiniZero self-play workers keep multiple MCTS instances and batch leaf-node
  GPU inference.
- KataGo-style engines batch many positions through a shared neural net
  evaluator.

Local translation:

```text
many vector env rows
-> many active roots/trees
-> central batched model/search service
-> compact replay rows
```

This is the production-scale architecture if we need many workers or large
simulation counts. It is bigger than a profile patch.

## More Ambitious Lane 4: MCTX/JAX

MCTX is the clean all-device version: JAX-native, JIT-able, batch-first MCTS.

Why it is attractive:

- Search tree and recurrent function are tensorized.
- Batches are first-class.
- It is much closer to the thing we wish LightZero were doing.

Why it is not the next patch:

- CurvyTron training is currently PyTorch/LightZero.
- A PyTorch model inside JAX/MCTX would recreate a host boundary.
- A real MCTX lane means JAX model/search/replay integration, not a small
  trainer patch.

Use it as a scratch falsification lane, not immediate Coach training advice.

## Research Sources

- MCTX: https://github.com/google-deepmind/mctx
- MiniZero: https://github.com/rlglab/minizero
- LightZero CTree docs:
  https://opendilab.github.io/LightZero/_modules/lzero/mcts/tree_search/mcts_ctree.html
- Treant batched evaluator pattern:
  https://mcts.dev/docs/concepts/architecture/

## Current Recommendation

1. Finish the matched `direct_ctree_gpu_latent` profile wave.
2. If it wins, add the missing telemetry/compare gates and run a same-shape
   full direct-vs-stock profile.
3. If it does not win enough, stop polishing partial CTree glue and prototype
   dense Torch MCTS in profile-only mode.
4. Keep MCTX/JAX and C++/MiniZero as architecture lanes, not next small patches.

## 2026-05-22 Update

The first H100 fixed-denominator row says this lane is worth continuing.

Same denominator for all three rows:

```text
H100, B512, A16, sim8, 60 measured steps, 15 warmup,
uint8 stack, scalar materialization off, no death, profile-only
```

| impl | measured sec | scalar steps/sec | search sec |
| --- | ---: | ---: | ---: |
| `stock_facade` | `26.99` | `2276.71` | public collect/search `18.56s` |
| `direct_ctree_arrays` | `13.45` | `4568.28` | `6.04s` |
| `direct_ctree_gpu_latent` | `9.34` | `6580.32` | `1.89s` |

Plain read:

```text
direct_ctree_gpu_latent is about 2.9x over stock facade and about 1.4x over
direct_ctree_arrays on this boundary row.
```

Validation also improved:

- mixed-mask CPU test now includes `direct_ctree_gpu_latent`;
- value parity checks compare stock against both direct paths;
- comparison script now supports both direct paths;
- 8-seed sim8/root-noise-on compare returned exact agreement for actions,
  visits, searched values, and illegal-action counts.

The inner GPU-latent loop now uses a preallocated GPU latent pool instead of a
Python list plus grouped gather. That is still the same CPU CTree algorithm; it
just removes more Python indexing work from the search loop.

Follow-up rows completed:

Same denominator, sim8 post-pool:

| impl | run | measured sec | scalar steps/sec | boundary total | search sec |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` post-pool | `ap-ug5mMvfuCWzRDjmNty1N3P` | `9.43` | `6518.13` | `4.25s` | `2.09s` |

Same denominator, sim16:

| impl | run | measured sec | scalar steps/sec | boundary total | search sec |
| --- | --- | ---: | ---: | ---: | ---: |
| `stock_facade` | `ap-cRxL8lanazTY29fnKhCQx7` | `35.43` | `1734.05` | `30.40s` | public collect/search `27.14s` |
| `direct_ctree_arrays` | `ap-zDtVC6Lw2LcPocY2AtTzRc` | `19.92` | `3083.58` | `15.10s` | `13.30s` |
| `direct_ctree_gpu_latent` | `ap-3v1vn4EGdk8YfJGIIVdiWD` | `12.60` | `4874.42` | `7.05s` | `4.69s` |

Plain read:

```text
The latent-pool cleanup itself was roughly neutral at sim8. The real win is
still the GPU-latent bridge: at sim16 it is about 2.8x over stock facade and
about 1.6x over direct CTree arrays.
```

Decision:

```text
Keep GPU-latent as the tactical profile baseline. It is not enough for the
whole project goal by itself. The next order-of-magnitude attempt should be a
dense GPU search prototype or an array-native Cython CTree boundary.
```

## 2026-05-22 Dense Torch MCTS Prototype

A profile-only dense Torch MCTS mode now exists:

```text
--hybrid-lightzero-array-ceiling-probe
--hybrid-lightzero-array-ceiling-mode dense_torch_mcts
```

It is not stock LightZero CTree. It is a fixed-action-count GPU PUCT probe for
pricing the CPU CTree/list boundary.

Same denominator:

```text
H100, B512, A16, sim8, 60 measured steps, 15 warmup,
uint8 stack, scalar materialization off, no death, profile-only
```

| impl | measured sec | scalar steps/sec | boundary/probe sec | search/update sec | model sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` repeat | `10.24` | `6001.49` | `4.68s` | `2.41s` | `0.91s` |
| `dense_torch_mcts` v0 | `9.57` | `6418.31` | `3.03s` | `0.68s` | `0.91s` |
| `recurrent_toy` ceiling | `8.23` | `7466.60` | `2.58s` | `0.11s` | `0.89s` |

Plain read:

```text
Dense GPU MCTS v0 works, but it is not a 10x answer yet. It was only about
1.07x over the same-run direct_ctree_gpu_latent row and still below the toy
recurrent ceiling.
```

Why:

- the prototype still had Python control in every simulation/depth;
- it still allocated/cloned path tensors inside the loop;
- it still measured with explicit CUDA syncs around recurrent calls;
- the H100 was not saturated.

Follow-up patch:

- removed inner-loop `.item()` CPU sync checks;
- preallocated path node/action/active history;
- removed per-depth bootstrap clone during backprop.

Next proof:

```text
Repeat H100 B512/A16/sim8 dense_torch_mcts after sync/allocation cleanup.
If it does not move materially, switch the next implementation lane to
array-native Cython CTree or a more radical batched search service.
```

Sync/allocation cleanup result:

| impl | measured sec | scalar steps/sec | probe sec | search/update sec | model sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` same-run repeat | `10.37` | `5927.42` | `4.70s` | `2.41s` | `0.92s` |
| `dense_torch_mcts` cleanup v1 | `7.96` | `7720.30` | `2.59s` | `0.71s` | `0.79s` |
| `recurrent_toy` same-run ceiling | `6.75` | `9097.07` | `2.09s` | `0.09s` | `0.79s` |

Plain read:

```text
The dense lane is alive. Removing obvious sync/allocation overhead made it
about 1.30x faster than same-run GPU-latent CTree and about 3.4x faster than
the old stock public facade. It is still not an order-of-magnitude fix.
```

Current follow-up:

- dense mode now also skips per-recurrent CUDA sync in the profile prototype;
- dense mode now avoids the host boolean filter copy when every root has at
  least one legal action;
- H100 sim8 and sim16 repeats are running.
