# Search Boundary Train-Facing Recommendation

Date: 2026-05-22

Status: concise sidecar recommendation. No code, live runs, trainer defaults,
checkpoints, or Modal state changed.

Current active docs:

- [search_boundary_next_wave_20260522.md](search_boundary_next_wave_20260522.md)
- [array_native_ctree_next_design_20260522.md](array_native_ctree_next_design_20260522.md)
- [direct_ctree_promotion_gates_20260522.md](direct_ctree_promotion_gates_20260522.md)

## 2026-05-22 Superseded By Output-Fast Repeat

This note is now pre-output-fast history. Do not use it as Coach launch advice.

Current truth:

```text
C64/sim16/3-learner train_muzero profile repeat:
  stock:              433.17 steps/sec
  direct output-fast: 566.19 steps/sec

same denominator with rnd_meter_v0 after RND hash fix:
  stock:              351.02 steps/sec
  direct output-fast: 448.52 steps/sec
```

The direct hook plus all-actions-legal output fast path now has a real matched
profile-loop win. Keep it profile-only until the promotion gates pass. The next
optimizer target is the remaining collect/search boundary cost:
per-simulation model-output D2H/list conversion and the CTree Python/list API.

## Plain Recommendation

This section is historical. The current recommendation is: keep the direct
output-fast hook profile-only, use it as the baseline, and implement the next
search-boundary lane behind the promotion gates.

Rank the next optimization target as:

```text
1. direct_ctree_gpu_latent train-facing hook
2. array-native CTree
3. broader batched search service
4. compiled/fused dense search
```

The next best target is not another profile-only roots/sec record. It is a
minimal, profile-gated `direct_ctree_gpu_latent` hook in the stock
`train_muzero` collection path, preserving LightZero replay/output semantics.

Historical reason: at the time, `direct_ctree_gpu_latent` was the best
LightZero-shaped sim16 boundary row, but it had zero Coach impact until it was
connected to the train-facing denominator. That train-facing profile hook now
exists and has output-fast profile evidence, so use the current active docs
listed above instead of this old ranking.

## Critique Of Current Evidence

The current profile evidence is useful, but easy to overread.

- `direct_ctree_gpu_latent` is the practical baseline: fresh H100 B512/A16 rows
  show about `7547` roots/sec at sim8 and `6145` roots/sec at sim16, versus
  stock facade about `2430` and `2094`.
- Dense eager Torch is not a training recommendation: it wins shallow sim8
  (`8288`) but loses sim16 (`4294`) against `direct_ctree_gpu_latent`.
- CPU64 is negative/suspicious only: `direct_ctree_gpu_latent` sim16 dropped
  from `6145` to `5119`, and stock facade dropped from `2094` to `1776`. Do not
  chase CPU allocation unless a later timer identifies a real CPU-parallel
  section.
- Roots/sec sidecar rows are not Coach speed. The trusted Coach path still uses
  stock LightZero `train_muzero`; `direct_ctree_gpu_latent` now also has a
  profile-only train-facing hook, but it remains optimizer evidence until the
  promotion gates pass.
- Sim8 is no longer a sufficient gate. The search-boundary bottleneck shows up
  more honestly at sim16, where eager dense GPU search fails and the remaining
  CPU/list/search shell is visible.

The strongest Amdahl read is:

```text
Renderer/observation work has been paid down enough that the next large wall is
LightZero collect/search boundary work. But profile-only boundary wins must be
reconnected to stock collection/replay before they can affect training.
```

## Why This Ranking

### 1. direct_ctree_gpu_latent train-facing hook

This is first because it converts the strongest existing boundary win into a
testable full-loop claim. It also keeps real LightZero CTree semantics, action
masks, root values, visit distributions, replay rows, and learner interaction
closest to the trusted path.

The hook should be profile-gated first, not a production default. The goal is to
answer one question:

```text
Does the best search-boundary profile row still improve wall-clock throughput
when stock train_muzero collection, replay, targets, and learner are included?
```

If yes, then array-native CTree has a real train-facing surface to improve. If
no, deeper profile-only search work is probably optimizing the wrong
denominator.

### 2. array-native CTree

This is the next optimization after the hook proves train-facing value. It
attacks the remaining known cost inside the same semantic family: Python lists,
`.tolist()`, CPU NumPy arrays, root prep, per-simulation backprop inputs, and
visit/value extraction.

It is less risky than replacing search semantics, but it should not come before
the train-facing hook. A faster profile-only CTree boundary still has no Coach
impact if the stock loop cannot consume it.

### 3. broader batched search service

This has the largest long-term Amdahl ceiling: many actors/roots, batched
recurrent inference, compact search state, and scalar materialization only at a
compatibility edge. It is the likely shape of a real `5x-10x` system.

It is not the next local target because it changes collection architecture,
queueing, replay cadence, weight freshness, live-root refill, and failure modes
under death/autoreset. Start designing it, but do not let it preempt the smaller
experiment that tells us whether the current search win survives the full loop.

### 4. compiled/fused dense search

This is a good architecture falsifier but the weakest train-facing
recommendation today. It is non-CTree, profile-only, and already failed the
sim16 eager gate. A bounded compile/CUDA-graph/Triton spike can still answer
whether eager Torch is losing to launch/control overhead, but even a win would
then need a separate semantics and replay-compatibility story.

Treat compiled dense as a side spike, not the main optimization target, until
it beats `direct_ctree_gpu_latent` at sim16 and has a credible path to stock
collector outputs.

## Exact Next Experiment

Do not run this now; this is the next experiment once code/live-run work is
allowed.

Add one profile-only, opt-in stock-loop hook:

```text
collect_search_impl=direct_ctree_gpu_latent
```

The hook must be inside the trusted `train_muzero` collection path, not the
standalone sidecar. It must preserve the stock collect output schema consumed
by replay and target construction.

Run a matched full-loop profile A/B:

```text
mode=profile
called_train_muzero=true
profile_allow_auto_resume=false
profile_volume_commit=false
same image/revision
same env_variant=source_state_fixed_opponent
same collector_env_num, batch_size, num_simulations
same no-RND/RND choice
same death/autoreset setting
same checkpoint/eval/GIF sidecar settings
at least one learner train call
```

Rows:

```text
A: stock LightZero collect/search
B: stock LightZero collect path with direct_ctree_gpu_latent search hook
```

Primary pass criteria:

- full-loop wall-clock throughput improves, not only boundary roots/sec;
- illegal actions and illegal visit mass stay zero;
- replay/target rows keep the expected LightZero action, visit, searched-value,
  reward, and mask schema;
- reset/death/autoreset counts match the stock control;
- if RND is enabled, RND counters and target-reward invariants match the
  selected mode.

Decision after the A/B:

```text
If B improves full-loop throughput by >= 1.2x, promote array-native CTree as the
next optimization on the same train-facing surface.

If B improves only the boundary but not the full loop, stop optimizing profile
roots/sec and inspect manager/replay/target/learner overhead.

If B cannot preserve stock semantics, keep direct_ctree_gpu_latent as profile
evidence only and move the main architecture discussion to a documented batched
search service rather than silently replacing train_muzero behavior.
```
