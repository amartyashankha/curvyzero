# Subagent MCTX GPU Search Research, 2026-05-22

Scope: research-only note for CurvyTron optimizer work. No code, training
defaults, live runs, or unrelated files were touched.

## Short Read

MCTX is the clean reference shape for GPU-resident MuZero-style search:

```text
fixed root batch
-> JAX representation/prediction
-> JAX recurrent_fn inside search
-> JAX tree arrays and embeddings
-> small action/action_weights/root summaries out
```

It directly attacks the current LightZero boundary that is still hot after
`direct_ctree_gpu_latent`: root logits/values copied to CPU, Python/root/list
CTree APIs, per-simulation recurrent outputs copied back to CPU, dict timestep
surfaces, replay row objects, and RND extraction.

The catch is large. MCTX is not a drop-in CTree replacement for the current
PyTorch LightZero training stack. Its useful form requires pure JAX model
functions and fixed-shape arrays. A PyTorch model bridged into JAX would mostly
recreate the host boundary we are trying to remove.

## 1. How MCTX Structures Batched GPU Search

MCTX is JAX-native, JIT-compatible, and batch-first. The README says its search
algorithms operate on batches in parallel and are meant to exploit accelerators:
<https://github.com/google-deepmind/mctx>.

The MuZero API shape is:

```text
RootFnOutput:
  prior_logits [B,A]
  value        [B]
  embedding    [B,...]

recurrent_fn(params, rng_key, action[B], embedding[B,...]):
  -> RecurrentFnOutput(
       reward       [B],
       discount     [B],
       prior_logits [B,A],
       value        [B],
     ),
     next_embedding [B,...]

PolicyOutput:
  action         [B]
  action_weights [B,A]
  search_tree    batched Tree
```

Sources:

- MCTX README quickstart and API description:
  <https://github.com/google-deepmind/mctx>
- MCTX core types:
  <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/base.py>
- MCTX search implementation:
  <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/search.py>
- MCTX tree storage:
  <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/tree.py>
- Gumbel MuZero policy source:
  <https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/policies.py>

Tree state is dense and resident as JAX arrays. For `N = num_simulations + 1`,
MCTX stores at least:

```text
node_visits            [B,N]
raw_values             [B,N]
node_values            [B,N]
parents                [B,N]
action_from_parent     [B,N]
children_index         [B,N,A]
children_prior_logits  [B,N,A]
children_values        [B,N,A]
children_visits        [B,N,A]
children_rewards       [B,N,A]
children_discounts     [B,N,A]
embeddings             [B,N,...]
root_invalid_actions   [B,A]
extra_data             [B,...]  # e.g. root Gumbels
```

That means the accelerator-resident part is not just the neural net. The tree
topology, priors, values, visits, rewards, discounts, root invalid-action masks,
and every per-node embedding are compiler-visible arrays. The simulation loop
is a JAX loop, with `simulate`, `expand`, and `backward` operating over the
batch. `expand` gathers the parent embedding from the tree, calls the JAX
`recurrent_fn` over the whole root batch, writes the new node tensors, then
backs up in JAX.

For CurvyTron, the natural MCTX search batch is not scalar envs. It is padded
ego roots:

```text
obs_env        [B,2,4,64,64]
legal_mask     [B,2,3]
live_mask      [B,2]
flatten/pad -> R = B * 2

obs_roots       [R,4,64,64]
invalid_actions [R,3]       # MCTX polarity: true/one means invalid
row_mask        [R]         # active roots; padded outputs ignored

JAX tiny CNN:
  representation(obs_roots) -> hidden[R,H]
  prediction(hidden)        -> prior_logits[R,3], value[R]
  recurrent_fn(..., action[R], hidden[R,H]) -> next hidden and heads
```

This is why vector latents matter. MCTX stores `embedding` for every node, so
`[R,N,H]` with `H=64` is boring, while a spatial latent like `[R,N,64,8,8]`
can become the memory wall before search compute does.

Gumbel MuZero is the recommended first MCTX policy. The MCTX README recommends
`gumbel_muzero_policy`; the policy source implements Full Gumbel MuZero; and
the paper's pitch is improved policy improvement with few simulations, which
fits CurvyTron's low-action, many-root setting:
<https://openreview.net/forum?id=bERaNdoegnO>.

## 2. Conflicts With Current LightZero/CurvyTron Pipeline

Current CurvyTron/LightZero collect/search shape conflicts with MCTX at almost
every hot boundary:

| Current shape | MCTX-friendly shape |
| --- | --- |
| Scalar env ids, Python dict observations, `BaseEnvTimestep` objects | Compact arrays: obs, legal mask, reward, done, row ids |
| Host `float32` observation stacks copied into policy calls | Fixed root tensors, preferably resident or one coarse H2D |
| PyTorch LightZero model | Pure JAX representation, prediction, dynamics |
| Root prep converts values/logits to CPU NumPy/lists | Root logits/value/embedding remain JAX arrays |
| LightZero CTree owns CPU tree state and list APIs | Tree stats and embeddings are JAX arrays `[R,N,...]` |
| Per-simulation recurrent output crosses GPU -> CPU for backprop | `recurrent_fn` output stays in the JAX loop |
| Variable live-root/list surfaces | Static `R=B*P` with masks and padded rows |
| Action output is dict-by-env, replay as game segments | Compact action/weights/value arrays; row object materialization only at edge |
| RND extracts latest frames/hashes/reward edits on CPU | Resident latest-frame/tensor ring path would be needed later |

The biggest incompatibility is framework ownership. MCTX's `recurrent_fn` must
be JAX-traceable. Calling the existing PyTorch LightZero model from inside that
function is not the useful path; it would force host callbacks, lose JIT shape,
or reintroduce the CPU/GPU synchronization wall.

The second incompatibility is semantics. MCTX expands one action per root row.
CurvyTron physics consumes simultaneous two-player actions. The smallest sane
MCTX approximation is independent per-seat roots with `A=3`, where the learned
dynamics implicitly models the other player. A centralized `A=9` joint-action
root is a useful control, but it changes the search problem and branching
factor.

The third incompatibility is training-system scope. MCTX returns selected
actions, action weights, and a search tree. It does not provide LightZero's
collector, replay buffer, target rows, learner, checkpoint publisher,
evaluation, RND reward shaping, or tournament semantics.

## 3. Smallest Toy Experiment

The smallest useful experiment is not a trainer migration. It is a fixed-shape
visual-root search sidecar:

```text
VectorMultiplayerEnv or existing source-state sample
-> SourceStateGray64Stack4.update(...)
-> obs_env float32[B,2,4,64,64]
-> legal_action_mask bool[B,2,3]
-> flatten/pad live rows to R=B*2
-> obs_roots float32[R,4,64,64]
-> invalid_actions bool[R,3]
-> tiny JAX CNN representation, vector hidden H=64
-> mctx.gumbel_muzero_policy(num_simulations=8/16/32)
-> action[R], action_weights[R,3], root summaries
```

First matrix:

| Env rows B | Roots R | Sims | Hidden | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 8 | 16 | 8 | 64 | compile and mask smoke |
| 16 | 32 | 16 | 64 | first useful L4 timing |
| 64 | 128 | 16 | 64 | compare to LightZero policy/search bucket |
| 64 | 128 | 32 | 64 | memory/search scaling stress |

Measure:

- visual setup time, policy-row mapping time, H2D time;
- compile-plus-first-run separately from warmed steady search;
- steady search p50/p95, decisions/sec, simulations/sec;
- D2H time for `action[R]`, then for `action_weights[R,3]`;
- GPU memory before compile, after compile, and after steady search;
- active/padded rows, mask polarity, no all-invalid active rows;
- finite normalized `action_weights`, legal selected actions, action histogram;
- no surprise recompilation for fixed profile.

Pass means only: MCTX/JAX is viable as a replacement/search sidecar candidate
for the next scratch lane. It must not be called training evidence.

This can reuse the existing MCTX benchmark precedent in
`src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`, which already proves
fixed-shape jitted MCTX runs on Modal GPU for synthetic roots. The missing proof
is the real CurvyTron visual-root input shape plus tiny CNN and row mapping.

## 4. Biggest Risks And Quick Falsifiers

| Risk | Quick falsifier |
| --- | --- |
| JAX model ownership is too expensive | Any plan that bridges PyTorch LightZero model calls into `recurrent_fn` requires host callbacks or CPU copies. Stop calling that path GPU-resident. |
| Tree memory blows up | Run `R=128,sims=16,H=64`, then `sims=32`; if memory is already uncomfortable on L4, pause spatial latents and larger models. |
| Fixed shapes recompile constantly | Warm fixed-profile runs must show no recompilation. If live-root counts or masks change shapes, use padded `R` or stop. |
| Visual setup/H2D dominates | If `visual_setup + H2D + search + D2H` is not competitive with the LightZero search bucket, MCTX search speed is irrelevant for current loop speed. |
| Search is slower than current direct CTree bucket | If warmed MCTX visual-root search at `R=128,sims=16,H=64` is slower than trusted LightZero/direct CTree timing, do not pursue migration. |
| Independent per-seat search is semantically wrong | Add a tiny `A=9` joint-action control and asymmetric player fixtures. If independent roots make obviously bad/legal-impossible choices, treat MCTX as search mechanics only. |
| Mask polarity bugs | Assert MCTX `invalid_actions=True` means invalid; padded rows must have harmless valid masks and be ignored. Any all-invalid active row is a hard failure. |
| RND/replay becomes the next wall | MCTX sidecar should report only search. A later resident replay/RND design is mandatory before Coach-facing claims. |
| JAX/CUDA ops friction | Pin JAX/JAXLIB/MCTX versions in the benchmark output. If install/compile/cache instability consumes the timebox, keep MCTX as reference architecture, not near-term lane. |

## Recommendation

Run one visual-root MCTX sidecar benchmark only after the current direct CTree
boundary falsifiers are either implemented or blocked. Keep it scratch-only and
fixed-shape. The decision rule should be cold:

```text
If [B,2,4,64,64] -> tiny CNN -> MCTX is fast, stable, finite, and memory-safe,
then MCTX/JAX deserves a deeper replacement/search-sidecar design.

If it recompiles, spills memory, loses to current LightZero/direct CTree, or is
dominated by visual setup/H2D, stop and continue with array-native CTree or
fixed-shape Torch/CUDA search lanes.
```

The main value of MCTX for CurvyTron right now is not that it magically solves
MuZero training. It gives a very clean target shape for the architecture we are
missing: batched roots, device-resident tree state, recurrent model outputs
kept on accelerator, and compact action/search-policy arrays out.

