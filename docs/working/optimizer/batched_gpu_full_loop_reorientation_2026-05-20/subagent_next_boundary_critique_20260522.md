# Subagent Next Boundary Critique

Date: 2026-05-22

Scope: read-only optimizer critique of the current LightZero collect/search
boundary. No production code, live training runs, checkpoints, evals, GIFs, or
tournament state touched.

## Current Bottleneck Read

The pivot away from renderer work is correct. The current Amdahl wall is the
search boundary: Python/list/CPU tree state, recurrent-output CPU copies,
root/search plumbing, compact output extraction, and the collector shape that
forces those costs to recur per simulation.

Current H100 B512/A16 fixed-denominator profile-only rows make the read pretty
sharp:

| row | sim8 roots/sec | sim16 roots/sec | read |
| --- | ---: | ---: | --- |
| stock public facade | `2276.71` | `1734.05` | public LightZero wrapper/search baseline |
| direct CTree arrays | `4568.28` | `3083.58` | wrapper/output fanout mostly removed |
| direct CTree GPU-latent | `6580.32` | `4874.42` | best LightZero-compatible tactical baseline |
| dense Torch MCTS cleanup v2 | `7969.41` | `4135.37` | wins sim8, loses sim16 |

Amdahl consequence:

- At dense sim8, measured wall is `7.709s`, search/update is only `0.697s`,
  model is `0.764s`, and observation is `1.908s`. Making only the reported
  dense search/update bucket free is capped near `1.10x`.
- At dense sim16, search/update grows to `2.514s` inside `14.857s`, so making
  that bucket free is still only about `1.20x`.
- At GPU-latent CTree sim16, measured wall is `12.60s`, search is `4.69s`,
  boundary total is `7.05s`, and observation is `2.50s`. Perfect search alone
  is capped around `1.59x` for that row.

So the current priorities make sense only if the next step changes topology:
compile/fuse the fixed-shape dense search, push the CTree boundary deeper into
array-native Cython/C++, or build a batched search service. More eager PyTorch
micro-polish, more renderer work, or input-copy tricks alone do not have enough
remaining denominator.

## Top 5 Practical Fixes

### 1. Make sim16 the gate, then compile or fuse dense Torch MCTS

Dense Torch is the right immediate big-swing probe, but cleanup v2 already
showed the eager version does not scale. The next dense work should be
`torch.compile(mode="reduce-overhead")`, CUDA graphs, or small Triton/custom
kernels for PUCT selection/backprop over fixed `[R, S + 1, A=3]` buffers. The
main target is fewer tiny eager kernels and less Python depth-loop overhead,
not another `.item()` cleanup.

Risk: `torch.compile` may recompile or graph-break on model output objects,
dynamic indexing, in-place tree writes, Dirichlet noise, or exception-shaped
action input probing. CUDA graphs require stable shapes and no hidden
allocation. Triton selection/backprop can become a separate semantics project.

Falsifier: on the same H100 B512/A16 denominator, compiled/fused dense sim16
does not beat `direct_ctree_gpu_latent` sim16 by at least `1.2x`, or still
shows worse scaling than recurrent-toy pressure as simulations rise. A second
hard falsifier is kernel-launch count staying high despite compile/capture.

### 2. Add array-native Cython/C++ CTree APIs for the per-simulation boundary

Root/output array APIs alone are useful but too small. The higher-value CTree
patch is per-simulation array-native traverse/backprop inputs: leaf indexes,
actions, rewards, values, logits, masks, and visit/value extraction without
Python lists or `.tolist()` in the hot loop. Keep GPU-latent storage as the
baseline, but stop paying list-shaped Cython calls every simulation.

Risk: maintaining a vendored LightZero Cython extension is real work, and the
tree remains CPU-owned. If only root prep/output changes, the result may be a
tidy `5-15%` patch rather than a strategic move.

Falsifier: same-denominator sim16 `search_sec` does not fall by at least
`25%` versus `direct_ctree_gpu_latent`, or full-row throughput does not improve
by at least `1.15x` after warmup. If ctree traverse/backprop arrays get faster
but model-output D2H/list conversion remains the same wall, stop at the
tactical patch and move to a search service.

### 3. Promote direct CTree GPU-latent as the tactical LightZero baseline

Do not lose the fact that GPU-latent CTree is currently the best
LightZero-compatible sim16 row. It should become the reference baseline for
parity, CUDA canaries, and any compact arrays/full-loop bridge. It should not
receive more latent-pool micro-polish unless a timer says the pool is the wall.

Risk: it is still profile-only and still CPU CTree/list shaped. Its sim16
profile win can shrink after replay target construction, death/autoreset, RND,
checkpoint cadence, and stock trainer compatibility are reintroduced.

Falsifier: a matched profile/full-loop canary using compact arrays and the
same no-RND/no-death knobs fails to beat the stock facade by a meaningful
end-to-end margin, or CUDA parity/forced-case gates expose action, visit, value,
mask, `to_play`, or support-transform drift.

### 4. Build a batched search worker/service instead of a wider scalar collect

The production-shaped fix is MiniZero/KataGo-like: many env rows and seats keep
many trees active; a worker selects leaves across roots; one recurrent model
batch evaluates them; compact visit/action/value arrays return to replay. This
also gives collector changes a natural home: live-root compaction, active-root
refill, no-host-stack profile mode, and scalar env-id materialization only at
the compatibility edge.

Risk: this changes collection architecture, randomness, replay cadence, weight
freshness, and queueing. It can also fail under normal death if active roots
collapse and the service cannot fill batches.

Falsifier: a profile-only service with fixed weights cannot keep recurrent
batch size high enough, spends more wall in queue/scheduler overhead than it
saves in search, or loses versus `direct_ctree_gpu_latent` at sim16 under the
same root count. Under normal-death rows, mean active roots below the target
batch size is also a falsifier for one-process service scaling.

### 5. Timebox a JAX/MCTX resident-search spike

MCTX/JAX is not a small LightZero patch, but it is the cleanest falsifier for
the all-device premise: fixed-shape state, observation, model, and MuZero-style
search stay batched/JIT-resident until final action readback. Keep it scratch
only, with a CurvyTron-shaped toy, vector latent, `A=3` independent-seat roots,
and optionally an `A=9` joint-action control.

Risk: a serious MCTX path implies JAX model/search ownership or a maintained
Torch-to-JAX shadow-model lane. Bridging PyTorch into jitted JAX recreates the
boundary problem. Replay, checkpoint, and learner semantics are separate work.

Falsifier: the toy recompiles in steady state, cannot maintain `[B,2] ->
[B*2] -> [B,2]` axes without host round trips, tree memory is unacceptable at
sim16/sim32, or steady roots/sec is not materially above the current
GPU-latent/dense sim16 baselines.

## Priority Critique

The next optimization priority should not be "dense Torch MCTS until it gets
fast." It should be:

1. finish the same-denominator sim16 `direct_ctree_gpu_latent` and
   `recurrent_toy` comparison as the decision anchor;
2. try one fixed-shape compiled/fused dense pass if the shapes are stable;
3. if compiled dense does not beat GPU-latent at sim16, switch effort to
   array-native CTree or a batched search service;
4. keep GPU-latent as the tactical semantics-preserving baseline;
5. run MCTX/JAX only as a bounded architecture falsifier.

Deprioritize renderer-only work, pinned/resident input as the main prize, and
sim8-only speed claims. Under Amdahl's law, the current denominator needs
boundary/topology changes, not another local cleanup of a bucket that is now a
small fraction of the row.
