# Search Boundary Next Wave

Date: 2026-05-22

Status: active optimizer orchestration. Profile-only. Do not touch live Coach
training runs.

## Plain Goal

Speed up CurvyTron training by attacking the current corrected bottleneck:

```text
LightZero collect/search boundary, not rendering.
```

Current profile-only search-boundary baseline, not train-facing baseline:

```text
direct_ctree_gpu_latent
```

It keeps MuZero hidden states on GPU while still using LightZero CTree. On the
fresh H100 B512/A16 denominator it is about `3x` faster than the stock public
facade.

Current CPU-scaling falsifier:

```text
We had not tested high-CPU allocations in the current search ladder.
The profile-only route `gpu-h100-cpu64` now exists. Modal rejected `cpu=128`,
so 64 cores is the legal maximum in this sidecar. Use this route only to test
whether more CPU helps the CTree/search boundary. It does not change Coach
training.
```

## Current Facts

Fresh H100 profile-only ladder:

| impl | sim8 roots/sec | sim16 roots/sec | status |
| --- | ---: | ---: | --- |
| stock facade | `2430` | `2094` | baseline public wrapper |
| direct CTree arrays | `5009` | `3448` | real CTree, compact arrays |
| direct CTree GPU-latent | `7547` | `6145` | best practical baseline |
| dense Torch after semantic fix | `8288` | `4294` | profile-only, fails sim16 |
| recurrent toy ceiling | `12834` | `9191` | fake-search ceiling |

Plain read:

```text
Dense GPU search can be fast at shallow sim8, but eager Torch does not scale to
sim16. The next win must remove Python/list/control overhead with a compiled or
array-native search boundary.
```

## Workstreams

### A. Compiled / Fused Dense Search Spike

Question:

```text
Can the fixed-shape dense Torch search be compiled or captured so sim16 beats
direct_ctree_gpu_latent?
```

Scope:

- inspect `_run_dense_torch_mcts`;
- identify graph breaks, dynamic shapes, allocations, and exception paths;
- propose the smallest profile-only patch for `torch.compile`,
  CUDA graphs, or a fixed-shape fused helper;
- do not touch trainer defaults or live runs.

Falsifier:

```text
If a bounded compiled/fused spike cannot beat direct_ctree_gpu_latent sim16 by
at least about 1.2x, stop the dense lane for now.
```

### B. Array-Native CTree Boundary

Question:

```text
Can we keep real LightZero CTree semantics but remove Python list/object fanout?
```

Scope:

- inspect LightZero CTree Python/Cython APIs in the installed environment;
- map the exact calls where lists, `.tolist()`, CPU arrays, and output
  extraction happen;
- propose a minimal array-native API for fixed `A=3`;
- include maintenance risk and expected speed ceiling.

Falsifier:

```text
If only root/output cleanup is available, expected win is probably too small.
The useful patch must reduce per-simulation list/API overhead.
```

### C. Full-Loop Integration Reality Check

Question:

```text
What is required to make the practical search baseline affect the actual Coach
training loop rather than only the profile sidecar?
```

Scope:

- inspect the current Coach launcher and LightZero config builder path;
- identify whether `direct_ctree_gpu_latent` can be wired cleanly into the
  current trusted stock/frozen route;
- list exact files/contracts that would need changes;
- keep it read-only unless a small, obvious config plumbing gap appears.

Falsifier:

```text
If integrating the profile-only boundary would mean replacing stock
train_muzero semantics, do not do it blindly. Keep it as profile evidence and
recommend only stock-safe settings.
```

### D. Parked CPU Allocation Falsifier

Question:

```text
Does giving LightZero CTree/search 64 or 128 CPUs materially speed up the
current boundary?
```

Scope:

- same H100 GPU, same B512/A16/sim16 denominator;
- compare `gpu-h100` versus `gpu-h100-cpu64`;
- first target `direct_ctree_gpu_latent`, with one `stock_facade` CPU64 row as
  a wrapper-control;
- no live runs and no trainer defaults.

Falsifier:

```text
If CPU64 does not beat CPU4 by at least ~15-20%, the wall is probably not
simple CPU capacity. It is more likely Python/list/control/sync shape.
```

Result:

```text
CPU64 failed the falsifier. direct_ctree_gpu_latent sim16 dropped from 6145 to
5119 roots/sec, and stock_facade sim16 dropped from 2094 to 1776 roots/sec.
Do not spend more time on "just add CPUs" for this boundary unless a later
profile shows a different CPU-parallel section.
```

## Current Implementation Step

The first train-facing bridge now exists:

```text
collect_search_backend=direct_ctree_gpu_latent
```

It is profile-only. It patches `MuZeroPolicy._forward_collect` inside the
existing stock `train_muzero` hook window, then restores it. The hook returns
stock collect outputs, not compact profile arrays, so the stock collector,
replay, target builder, learner, RND hooks, and checkpoint machinery still own
the rest of the loop.

Local validation passed:

```text
uv run python -m py_compile ...
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py
uv run ruff check --ignore F401 ...
```

The `--ignore F401` caveat is only because the current launcher has many
pre-existing unused imports. The new duplicate-key/lifecycle/schema checks are
clean.

Next gate:

```text
Run stock train_muzero full-loop profile A/B:
A = collect_search_backend=stock
B = collect_search_backend=direct_ctree_gpu_latent
```

Both rows must use the same H100 profile settings and must stop after at least
one learner train call. The result we care about is full-loop wall time, not
sidecar roots/sec.

2026-05-22 active wave:

```text
Launched:
- C64/sim16 one-learner stock/direct quick A/B.
- C64/sim16 three-learner stock/direct steadier A/B.
- C128/sim16 three-learner stock/direct batch-scaling A/B.
- C64/sim8 three-learner stock/direct sparse-telemetry control.
```

Why:

```text
The old sim8 row proved the hook runs but did not win wall time. The new wave
separates three questions:
1. Did telemetry stride 1 pollute the old sim8 result?
2. Does direct win when sim16 makes search a larger fraction of wall time?
3. Does the direct hook need larger root batches to show value?
```

Validity gates:

```text
called_train_muzero=true
learner_train_calls matches the row target
env_steps_collected_source=collector_envstep_delta
evaluator_eval_calls=0
direct rows: collect_search_backend_direct_ctree_gpu_latent_calls > 0
direct rows: collect_search_backend_fallback_calls == 0
```

First read:

```text
Direct improves search buckets consistently, but wall-clock is shape-sensitive:
- C64/sim16 quick: stock 387 steps/sec, direct 456.
- C64/sim16 three-learner: stock 205, direct 421, but the stock row is
  suspiciously slow and is being repeated.
- C64/sim8 three-learner: stock 495, direct 477.
- C128/sim16 three-learner: stock 478, direct 397.
```

Current conclusion:

```text
The repeat did not hold the 2x-looking C64/sim16 result:
- C64/sim16 three-learner repeat: stock 445 steps/sec, direct 439.
- Direct still reduces MCTS (`12.02s -> 10.72s`) and policy collect (`17.07s
  -> 15.95s`).
- Direct also pays `3.41s` model-output D2H/list conversion and `2.87s` stock
  output assembly.

Do not chase CPU count. Do not promote C128/direct. Do not claim a direct
full-loop speedup yet. The next target is the remaining collect/search
boundary: per-simulation model-output D2H/listifying, output assembly, and the
CTree Python/list API boundary.
```

Output fast-path update:

```text
The first local/remote low-risk patch attacked output assembly only. It keeps
the stock collect-output dict shape and adds an all-actions-legal CurvyTron
fast path. First remote row:

direct output-fast C64/sim16/3-learner:
  602.42 steps/sec
  output assembly 0.063s
  fast_path_calls 256

This is promising but not yet stable. A fresh stock/direct repeat is running.
```

Matched repeat result:

```text
stock C64/sim16/3-learner:
  433.17 steps/sec
  wall 37.82s
  policy collect 17.10s
  MCTS 12.09s

direct output-fast C64/sim16/3-learner:
  566.19 steps/sec
  wall 28.94s
  policy collect 10.31s
  MCTS 8.06s
  output assembly 0.077s
  fast_path_calls 256
  fallback_calls 0
```

Current conclusion:

```text
The output fast path produced a repeatable profile-loop win in this denominator:
about 1.31x over matched stock. Do not call it a production Coach default yet.
Do call it the current best optimizer probe.

The next Amdahl wall is no longer output assembly. It is the remaining
collect/search boundary, especially model-output D2H/list conversion and the
CTree CPU/list API, plus the stock collector/env/replay shell around search.
```

## Main-Thread Rules

- Keep `direct_ctree_gpu_latent` as the practical baseline until a stronger row
  beats it at sim16.
- Do not promote dense Torch to training.
- Do not restart or mutate live Coach runs.
- Keep docs current after each profile or code change.
- Prefer small falsifiers over broad rewrites.
