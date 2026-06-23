# Sidecar Next Phase Prioritization Critique

Date: 2026-05-22

Status: doc-only critic note. No code, live runs, checkpoints, or Modal volumes
were touched.

## 1. What Are We Fixing?

We are fixing the gap between a fast profile experiment and real Coach
training.

Plainly:

```text
The profile harness has a faster LightZero search path.
The actual Coach training loop does not use it yet.
```

The current wall is not "draw the frame faster" and it is not "add more CPUs."
The current wall is the LightZero collect/search boundary: stock
`MuZeroPolicy.collect_mode.forward` converts and moves data through
Python/list/CPU-tree-shaped APIs before producing the action, root value, and
visit targets that replay needs.

## 2. Best Thing To Optimize Next

The next best target is a profile-gated train-facing hook for:

```text
direct_ctree_gpu_latent
```

Reason: this is the strongest practical row we already have. On the fresh H100
B512/A16 denominator it is about `3x` faster than the stock public facade in
profile-only search rows, while still using real LightZero CTree semantics.

But that win currently has no Coach impact. A faster sidecar path does not
speed training unless the trusted `train_muzero` collection path actually uses
it. So the next optimizer task should be to connect this search path to the
stock training loop in the smallest reversible way, preserving stock collector,
replay, target, learner, RND, reset, and checkpoint behavior.

Do not spend the next phase chasing a new profile-only roots/sec high score
until this train-facing A/B exists.

## 3. Why We Are Blocked From Claiming A Real Speedup

We are blocked because the current fast path is profile-only.

The trusted Coach route calls stock LightZero `train_muzero`. The
`direct_ctree_gpu_latent` path lives in
`source_state_batched_observation_boundary_profile.py`, returns compact profile
arrays, and does not by itself produce the exact stock collect output consumed
by replay and target construction.

So the honest claim is:

```text
We have evidence that the search boundary can be much faster.
We do not yet have evidence that real training is much faster.
```

The CPU64 result does not change this. CPU64 was worse, which is useful as a
negative falsifier. It suggests the wall is not simple CPU capacity. It is more
likely the shape of the boundary: Python control, list/object fanout, CPU CTree
state, GPU/CPU synchronization, and output extraction.

## 4. Next Experiment After The Hook

After the hook is implemented, run a matched full-loop profile A/B:

```text
A: stock train_muzero collect/search
B: stock train_muzero with direct_ctree_gpu_latent collect/search hook
```

The run must use the same code revision, env variant, seed policy, collector
env count, batch size, `num_simulations`, RND setting, death/autoreset setting,
checkpoint/eval/GIF cadence, and hardware.

The important pass criteria are:

- `called_train_muzero=true`;
- at least one learner update happens;
- replay rows and target rows keep stock LightZero fields and shapes;
- illegal actions and illegal visit mass stay zero;
- reset/death/autoreset counters match the stock control;
- if RND is on, RND counters and reward invariants match the selected mode;
- full-loop wall-clock throughput improves, not just profile roots/sec.

If the hook improves full-loop throughput by at least about `1.2x`, then the
next serious optimization should be array-native CTree on the same train-facing
surface. If it only improves the boundary but not the full loop, stop and split
manager/replay/target/learner overhead before doing more search-only work.

## 5. What Not To Chase Right Now

Do not chase these as the main lane:

- more CPU allocation;
- CPU128 or bigger CPU grids;
- another renderer-only rewrite;
- body-circles-vs-browser-lines debates;
- dense Torch MCTS as a training path before it beats the practical baseline at
  sim16 and has a semantics story;
- custom compact replay or trainer replacement before the stock-loop hook is
  tested;
- broad multi-GPU scaling before the single-GPU stock-loop bottleneck is
  connected and measured.

The main next question is simple:

```text
Can the best profile-only search path make real train_muzero collection faster
without changing what replay and learning mean?
```

