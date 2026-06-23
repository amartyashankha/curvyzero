# GPU / Parallel MCTS Research Synthesis, 2026-05-22

Status: active optimizer working memory. No live Coach training runs touched.

## Plain Answer

No, we should not pretend the GPU MCTS research was already complete before
this wave. We now have a better picture.

The current blocker is not simply:

```text
MCTS is not on GPU.
```

The sharper blocker is:

```text
the current LightZero-compatible search/replay boundary turns MuZero search
into Python, CPU, list, and object work even while the model runs on GPU.
```

GPU MCTS is probably one important way out, but only if it changes the boundary
shape. A GPU tree that still copies recurrent outputs back to CPU every
simulation, emits Python lists, and immediately materializes stock replay
objects will not give a 10x full-loop win.

## What The External Research Says

MCTX is the clean all-device reference. Its search is JAX-native, JIT-friendly,
and batch-first: root logits/value/embedding go in as arrays, the recurrent
function runs inside the search loop, tree state is stored as dense batched
arrays, and only compact action/action-weight summaries need to come back.

MiniZero and KataGo point to a practical production pattern:

```text
many games / roots / positions alive at once
-> batched neural inference
-> one search/evaluator service
-> compact search targets
-> replay/learner separated from acting
```

PufferLib is not a MuZero system, but it is relevant because it shows the same
systems lesson on the env side: static/contiguous buffers, pinned or async
transfer, and avoiding scalar Python APIs in the hot path.

Sources and local notes:

- `subagent_mctx_gpu_search_research_20260522.md`
- `subagent_gpu_mcts_implementation_patterns_20260522.md`
- `subagent_fast_rl_architecture_patterns_20260522.md`
- MCTX: <https://github.com/google-deepmind/mctx>
- MiniZero: <https://github.com/rlglab/minizero>
- KataGo: <https://github.com/lightvector/KataGo>
- PufferLib docs: <https://puffer.ai/docs.html>
- Batch MCTS: <https://arxiv.org/abs/2104.04278>

## What Our H100 CTree Sweep Says

The no-model H100 sweep priced LightZero CTree without env or neural-network
work:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_ctree_no_model_benchmark \
  --roots 512,1024 \
  --simulations 16,32 \
  --iterations 20 \
  --warmup 5 \
  --backends ctree-list,ctree-torch-d2h,fake-flat \
  --legal-profiles all3,mixed_2of3 \
  --compute h100
```

Representative rows:

| roots | sims | profile | CTree list nodes/sec | CUDA payload nodes/sec | fake-flat nodes/sec |
| ---: | ---: | --- | ---: | ---: | ---: |
| `512` | `16` | all3 | `0.761M` | `0.678M` | `13.86M` |
| `512` | `32` | all3 | `0.701M` | `0.651M` | `14.01M` |
| `1024` | `16` | all3 | `0.468M` | `0.570M` | `17.21M` |
| `1024` | `32` | all3 | `0.579M` | `0.512M` | `17.75M` |
| `1024` | `32` | mixed_2of3 | `0.555M` | `0.527M` | `19.37M` |

Plain read:

```text
LightZero CTree/list is much slower than a flat array update, often by
roughly 18x-37x on this no-model denominator.

But raw CTree-list alone is still hundreds of thousands of nodes/sec, far
above the current full-loop direct-hook rate. So CTree alone is not the whole
wall.
```

The CUDA payload row also matters. It preloads synthetic recurrent outputs on
GPU, then copies/listifies the per-simulation payload for CTree backprop. That
row is usually only modestly slower than `ctree-list`, sometimes noisy in the
other direction. The measured D2H payload itself was small in absolute bytes
and seconds on this no-model test; the real system is worse because it also
launches recurrent inference, gathers latents/actions, transforms scalar
outputs, and runs inside the stock collector/replay topology.

## Current Best World Model

The bottleneck is a stack of boundaries:

```text
compact CurvyTron rows
-> LightZero public env/timestep surface
-> root value/logit CPU/list prep
-> CTree CPU/list traverse/backprop
-> recurrent inference on GPU
-> reward/value/policy back to CPU/list each sim
-> action dict output
-> replay/target/RND objects
```

The reason previous patches landed around `1.3x` full-loop is that they removed
some of these costs while keeping the topology.

The reason the mock compact sidecar looks better is that it skips the real tree
and much of the public output. It shows headroom, not a training solution.

## Is GPU MCTS The Blocker?

Yes, but with a qualifier.

The thing blocking a large speedup is not the absence of a CUDA kernel in
isolation. It is the absence of a compact, batched, device-friendly
search/replay owner. A correct large move should keep this shape:

```text
obs_or_latent[N,...]
legal_mask[N,3]
active_mask[N]
search_state[N,max_nodes,...]
recurrent outputs stay device-side or array-side
-> compact action[N], visits[N,3], root_value[N]
```

Then replay/target rows should be array-native too, with stock LightZero
objects used only as validation or compatibility output.

## Next Falsifiers

1. **Precomputed recurrent-output direct search.**
   Same CTree root shapes as `direct_ctree_gpu_latent`, but replace
   `model.recurrent_inference` with resident synthetic reward/value/policy
   tensors. If speed is still close to current direct, CTree/list/control is
   the wall. If speed jumps toward the recurrent-toy/mock ceiling, recurrent
   launch/D2H/model-output handling is the wall.

2. **Compact replay writer dry run.**
   Search output becomes compact replay arrays directly, and stock target rows
   are materialized only for parity. This checks whether search wins are
   immediately swallowed by replay object materialization.

3. **MCTX visual-root toy.**
   Scratch-only: `[B,2,4,64,64] -> R=B*2 -> tiny JAX CNN -> MCTX
   gumbel_muzero_policy`. This does not train, but it tells us if a real
   device-resident search shape is fast and stable on our input scale.

4. **Array-native fixed-A=3 CTree sketch.**
   Keep LightZero semantics, but replace nested list APIs with
   `value[N]`, `policy[N,3]`, `legal_mask[N,3]`, `action[N]`,
   `visits[N,3]`. This is the conservative bridge if full GPU/JAX search is
   too large.

## Decision Rule

Do not spend weeks polishing one wrapper.

If the precomputed-output falsifier says current CTree/list/control dominates,
the next implementation should be array-native CTree or a search service.

If recurrent launch/D2H dominates, keep CTree semantics for now but make
recurrent output handling resident/batched and move replay to compact arrays.

If both are only `1.5x` class, the next big move is not another MCTS patch. It
is the MiniZero/Puffer-style architecture:

```text
native/vector CurvyTron actor buffers
-> batched search/evaluator service
-> compact replay owner
-> learner consumes compact batches
```

