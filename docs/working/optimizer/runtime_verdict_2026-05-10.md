# Optimizer Runtime Verdict

Date: 2026-05-10

Status: compact source of truth for the current CurvyTron optimizer stance,
CPU/GPU boundary, and near-term measurement direction.

Important correction on 2026-05-11: original AlphaZero/MuZero speed comes from
searched self-play actor throughput feeding replay and a learner, not from a
single local trainer loop. See
`docs/working/optimizer/distributed_muzero_architecture_research_2026-05-11.md`.

Current launcher truth, 2026-05-11 late: Coach canonical CurvyZero launcher is
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
The older two-seat smoke wrappers below are historical optimizer evidence, not
the live launcher. Fixed/frozen-opponent stock `train_muzero` is
controls/profiling only.

## Scope Boundary

- Optimizer owns setup, measurement, profiling, speed, Amdahl reads, CPU/GPU
  split, Modal boundaries, and process architecture.
- Coach owns learning claims, checkpoint quality, eval quality, and LightZero
  replication status.
- Environment/RAM reconstruction owns source truth, fidelity, parity,
  reset/final-observation contracts, and reward semantics.

## Primary CurvyTron Target

The main CurvyTron training path should be visual LightZero-style stacked
frames and is non-ALE. The active current visual profiler/training target is
source-state gray64 `uint8[1,64,64]` / stacked training tensor. Browser/canvas
pixels are optional later debug/human evidence; source-state/event goldens are
the fidelity blocker. The old `debug_visual_tensor` /
`curvyzero_debug_occupancy_gray64/v0` occupancy surface is historical smoke
data, not the active target.

The scalar-ray `[B,2,106]` work below is a diagnostic sidecar for
timing, source-adapter pressure, and boundary probes. It is not the current
coach-facing optimizer target and should not displace the visual path without a
separate evidence gate.

Historical debug visual smoke baseline:

```text
CurvyTronSourceEnv seeded source reset
  -> advance source lifecycle to print-start (`startup_advance_ms=3000`)
  -> source step
  -> world_bodies_snapshot()
  -> debug occupancy render
  -> normalize to LightZero-facing float32 frame
  -> optional optimizer-owned frame stack
```

Bounded local profile after the one-pass debug renderer vectorization,
`B=32,T=64`, stack+copy enabled: loop `0.0927s`, env step `0.0174s`, body
snapshot `0.0131s`, render `0.0391s`, normalize `0.0056s`, stack+copy
`0.0130s`, throughput `22087` transitions/s. Matched stack no-copy was
`0.0907s` / `22583` transitions/s. Reset/startup advance was timed separately
at about `0.07s` and excluded from loop throughput. Policy/search, replay,
learner, and evaluator were not included.

Read: one obvious debug renderer Python-loop tax was removed in the historical
smoke path. Stop optimizing this surface as its own project. It is only a debug
occupancy workload, useful for plumbing and denominator discipline. The next
optimizer question is whole-loop timing around the source-state gray64 adapter
and real policy/search/replay/learner buckets, not more polishing of debug
pixels.

Installed-runtime no-train visual adapter smoke now passes on Modal for
`curvyzero_debug_visual_tensor_lightzero`: LightZero `0.2.0`, DI-engine
`0.5.3`, env factory and direct env reset/step, real `BaseEnvTimestep`, raw env
payload `(1,64,64)`, model stack `(4,64,64)`, action space `3`, and no ALE
identity. This is setup/timing evidence only; it is not a learning or
source-fidelity claim. The raw debug renderer is still `uint8[1,64,64]`; the
LightZero-facing env payload is normalized `float32[1,64,64]`.

Local direct adapter timing now exists:
`scripts/benchmark_debug_visual_lightzero_adapter.py --steps 512 --seed 5`
measured `512` CurvyTron debug visual adapter transitions in `0.0690s` of
env-step time, about `7416` transitions/s, with `4` resets and `3` done rows.
This includes direct adapter reset/step/rendered payload work but still excludes
trainer frame-stack consumption, policy/search, replay, learner, and eval.

## Two-Seat LightZero Runtime Update - 2026-05-11

Archived context: this section describes old two-seat smoke-wrapper work. The
live Coach launcher is now
`lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.

The custom two-seat CurvyTron path exists because LightZero has current-policy
self-play for normal/alternating collector shapes, but stock `train_muzero`
does not expose a simultaneous `joint_action[B,P]` collector boundary. The
right fix is a thin custom CurvyTron collector/adapter that reuses LightZero
policy/MCTS/learner pieces, not a slow private reinvention.

Current code state:

- The current two-seat self-play bridge is launched through
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode two-seat-selfplay`.
  The old `lightzero_curvytron_two_seat_train_smoke.py` Modal wrapper has been
  deleted.
- The underlying trainer batches all active seat rows into one
  `MuZeroPolicy.collect_mode.forward` call per decision step when possible. The
  old one-row call remains as fallback and records the fallback reason.
- Collection records `collect_timing_sec`, `policy_batching_counts`,
  `policy_search_call_count`, and `policy_search_row_count`.
- The canonical launcher has explicit `--compute cpu` and `--compute gpu-l4-t4`
  routes. GPU requests CUDA and the summary reports
  `lightzero_policy_model_device`; validated value was `cuda:0`.
- The path no longer explicitly commits the Modal Volume during progress or
  summary writes. Defaults are now sparse: progress every `100` iterations and
  checkpoints every `100` iterations. Final checkpoints still save for
  `allow_optimizer_step=True`.
- Compact summaries now report aggregate `collect_timing_summary`, first/last
  iteration edges, and replay counts instead of huge per-row sample-index
  payloads.

Closest old comparable baseline, recovered from Modal logs and old summary
artifacts:

| Run | Shape | Old artifacts | Old wall read |
| --- | --- | --- | ---: |
| `curvytron-two-seat-targetfix-scale-b8-s9-16x64-u1-sim2` | `B=8`, `16x64`, `updates=1`, `sims=2`, `max_ticks≈64` | `17` checkpoints, `43 MB` summary, no component timers | `~318s` inner / `~335s` including app build |

The old summary did not record `elapsed_sec` or component timing, so this is a
rough log-based baseline. It is still the best same-shape read we have before
the optimizer cleanup.

Current same-shape comparable reruns, with `--max-ticks 64` pinned to match the
old accidental episode cap:

| Compute | Elapsed | Replay rows | Rows/wall-sec | Checkpoints | Policy/search sum | Visual stack sum | Autoreset sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | `102.24s` | `16384` | `~160/s` | `1` | `29.35s` | `5.70s` | `3.71s` |
| GPU L4/T4 | `50.64s` | `16384` | `~324/s` | `1` | `18.60s` | `7.17s` | `4.63s` |

Read: yes, the cleanup improved the old heavy path. The current CPU run is
about `3.1x` faster than the rough old inner wall time, and the current GPU run
is about `6.3x` faster. The win is mostly artifact cleanup, batched policy-row
collection, and moving the policy/learner to CUDA. It is not a proven 10x
single-loop win.

Amdahl read on the comparable GPU run: measured collect buckets total
`33.15s`, with policy/search at `18.60s` (`56%`), visual stack at `7.17s`
(`22%`), and reset/autoreset at `4.63s` (`14%`). Even deleting search entirely
would not make this single synchronous loop 10x faster because learner,
checkpoint, setup, render/stack, and reset work remain. The 10x route has to
include horizontal actor parallelism and fewer synchronization/artifact costs,
while preserving MuZero search on every decision.

Collect-only profiling, `B=8`, `collect_steps=8`, `outer_iterations=10`,
`num_simulations=5`, `updates=0`, sparse/default output:

| Compute | Elapsed | Instrumented | Policy/search sum | Policy/search mean/iter | Last policy/search | Search fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | `15.28s` | `4.78s` | `4.05s` | `0.405s` | `0.350s` | `0.849` |
| GPU L4/T4 | `18.29s` | `5.40s` | `4.24s` | `0.424s` | `0.187s` | `0.785` |

Read: the first GPU search is a cold-start tax (`~2.0s` in that run), but after
warmup the same shape is faster per iteration (`~0.19s` by iteration 10). On
this tiny profile, total elapsed is still dominated by setup/framework overhead,
not just instrumented collect work. GPU alone is useful, but current evidence
does not support treating a single GPU actor loop as the full 10x path.

Full tiny training smokes, GPU L4/T4, `B=8`, `collect_steps=8`,
`outer_iterations=3`, `updates_per_iteration=1`, `num_simulations=5`,
`replay_scope=accumulated`, `learner_sample_size=128`,
`allow_optimizer_step=True`:

| Seed | Elapsed | Learner | Model changed | Final search/iter | Checkpoint |
| --- | ---: | --- | --- | ---: | --- |
| `6` | `15.59s` | `ok` | yes | `0.282s` | `iteration_3.pth.tar` |
| `7` | `12.43s` | `ok` | yes | `0.190s` | `iteration_3.pth.tar` |

Read: the loop now performs collect -> replay/sample -> learner update -> final
checkpoint on CUDA without errors. Initial checkpoint writes are off by default;
final checkpoints still save when `allow_optimizer_step=True`. Policy/search is
still the largest measured collect bucket. Env step, row mapping, replay row
build, and visual stack are small by comparison in this smoke.

Larger collect-only actor-batch probes, GPU L4/T4, `outer_iterations=5`,
`collect_steps=8`, `num_simulations=5`, `updates=0`:

| Batch | Replay rows | Elapsed | Rows/wall-sec | Policy rows/call | Last policy/search |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32` | `2560` | `14.65s` | `~175/s` | `64` | `0.654s` |
| `64` | `5120` | `24.29s` | `~211/s` | `128` | `1.883s` |
| `128` | `10240` | `57.15s` | `~179/s` | `256` | `6.852s` |

Read: bigger batches increase data per call, but they do not produce 10x by
themselves, and `B=128` regresses. The best current single-actor speed probe is
around `B=64`, but the real 10x route is horizontal: multiple warm GPU actors
collecting in parallel, sparse artifacts, and a learner/replay merge boundary
that keeps policy-version metadata explicit. Search-only optimization is also
Amdahl-limited: when policy/search is `~70-85%` of instrumented collect, even an
infinite search speedup would not make a single synchronous loop 10x faster.

Scalar-ray sidecar observation shape:

```text
24 ray directions * 4 channels + 10 scalars = 106 float32 values per ego
```

Current source-backed scalar-ray sidecar path:

```text
CurvyTronSourceEnv snapshots
  -> source_snapshot_to_vector_trainer_state(...)
  -> observe_vector_1v1_egocentric_rays_v0(...)
  -> float32[B,2,106] observations + bool[B,2,3] action masks
  -> diagnostic policy/search/replay profile
```

This is CPU/Python/NumPy source plumbing today. It is not Atari/ALE, not a
visual stacked-frame pipeline, and not a real LightZero CurvyTron environment.

## Current CPU Profile

Latest local source-backed refresh: use controlled-trail setup when timing
body/ray geometry. Default source setup can produce zero body circles and mostly
time wall/scalar work.

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=16,T=64` | `0.297s` | `0.028s` | `0.138s` | `0.124s` | n/a | `3448/s` |
| `B=32,T=64` | `0.555s` | `0.050s` | `0.279s` | `0.217s` | n/a | `3692/s` |
| `B=128,T=16` | `0.354s` | `0.052s` | `0.148s` | `0.150s` | n/a | `5786/s` |

Read: source env stepping is still small. The current source-backed local tax is
source snapshot adaptation plus observation production. A phase-timed
controlled-trail `B=32,T=64` run measured loop `0.683s`, adapter `0.322s`,
observation `0.287s`, ray cast `0.168s`, and env step `0.057s`, but phase
timers add overhead. This is a speed/setup claim only; Environment/RAM still
owns whether the observed geometry is fidelity-valid, and Coach still owns
whether a learner improves.

The repo-native toy PPO actor loop also runs, but it is not source-backed. A
`B=64,T=64` smoke measured loop `2.962s`, observation `2.485s`, env step
`0.417s`, policy `0.036s`, and rollout write `0.078s`. The optional Torch
learner smoke skipped locally because Torch is not installed.

Toy bridge refresh: serial toy object env `22063.6` env steps/s; threads
`18950.4` env steps/s (`0.813x`, bad); processes `51859.3` env steps/s
(`3.579x`, `0.895` efficiency). Caveat: toy object env, synthetic NumPy policy,
local sharding only; not source fidelity, MCTS, Modal, or production
throughput.

Native strict scalar-ray trainer profile now exists as a second, non-oracle
diagnostic speed surface:

```text
VectorTrainerEnv1v1NoBonus
  -> float32[B,2,106] observations + bool[B,2,3] masks
  -> policy-row mapping + tiny policy/search stand-in
  -> joint_action[B,2]
  -> replay-v0 recorder/chunk
```

This profile is only strict `1v1/no_bonus/P=2` scalar-ray plumbing. It is not
broad CurvyTron fidelity, bonuses, 3P/4P, visual LightZero, or a learning
claim. The source-backed `CurvyTronSourceEnv`/JS path remains the semantic
oracle.

Corrected local profile artifacts from
`scripts/benchmark_vector_trainer_actor_loop_profile.py`:

| Shape | Loop | Public env.step | Policy map+forward | Replay record | Chunk build | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.259s` | `0.252s` | `0.0041s` | `0.0010s` | `0.0054s` | `1980/s` |
| `B=16,T=64` | `0.582s` | `0.572s` | `0.0058s` | `0.0014s` | `0.0119s` | `1760/s` |
| `B=32,T=32` | `0.640s` | `0.635s` | `0.0034s` | `0.0008s` | `0.0096s` | `1600/s` |
| `B=128,T=64` | `6.846s` | `6.830s` | `0.0111s` | `0.0013s` | `0.0662s` | `1197/s` |

Read: native scalar-ray removes source snapshot/adapter overhead, but the
public step is still dominated by per-row observation/ray work. Bigger batches
do not automatically help on CPU here. Replay append and chunk build are not the
current sidecar bottleneck. The first native blocker found was reset warmup
timer callback capacity at larger `B`; the public env now scales that cap and
has a `B=128` reset regression.

First batch-array writer pass landed after that table. It switches the public
env observation path from per-row dataclass construction to one batch array
writer that validates state once while preserving the old scalar row observer
as parity oracle. It does not yet vectorize ray/body math.

Corrected post-patch profile artifacts:

| Shape | Loop | Public env.step | Throughput | Batch observation probe | Ray cast |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.257s` | `0.248s` | `1993/s` | `0.00323s` | `0.00303s` |
| `B=16,T=64` | `0.431s` | `0.424s` | `2375/s` | `0.00640s` | `0.00608s` |
| `B=32,T=64` | `0.914s` | `0.904s` | `2241/s` | `0.01293s` | `0.01237s` |
| `B=128,T=16` | `0.822s` | `0.816s` | `2493/s` | `0.04973s` | `0.04779s` |

Read: hoisting validation/dataclass work helps medium and large batches, but
the profile is still dominated by public env.step, and the corrected batch
observation probe is mostly ray casting. The next local optimization target is
preallocated/batched ray observation, not replay or policy-row mapping. The
`B=128` post-patch run is reported at `T=16` because the `T=64` straight-action
run hit a terminal row at step `18`; do not compare it as an equal steady-state
`T=64` run.

Second pass sliced body arrays by `body_write_cursor[row]` before ray/trail
work. This removes scans over unused fixed body-buffer tail while falling back
to full arrays for source-adapter states that do not expose the cursor.

| Shape | Loop | Public env.step | Throughput | Batch observation probe | Ray cast |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.169s` | `0.162s` | `3038/s` | `0.00237s` | `0.00217s` |
| `B=16,T=64` | `0.394s` | `0.383s` | `2599/s` | `0.00468s` | `0.00432s` |
| `B=32,T=64` | `0.659s` | `0.649s` | `3108/s` | `0.00936s` | `0.00862s` |
| `B=128,T=16` | `0.602s` | `0.596s` | `3399/s` | `0.03297s` | `0.03105s` |

Read: this was a real native scalar-ray speed win and confirms the fixed body
buffer tail was costing ray time. It is still strict sidecar plumbing evidence,
not source fidelity or primary visual-path evidence. The remaining public step
is still observation-heavy, so the next decision is either real model/search
calibration on these faster sidecar rows or deeper ray batching if
policy/search stays small.

Third observation pass vectorized wall-hit and hit-normalization helpers in
`vector_trainer_observation.py`. It does not change the circle-hit math, owner
filters, own-trail latency rule, player mapping, rewards, or public schema.

Strict native after the wall/normalization patch:

| Shape | Loop | Public env.step | Throughput | Ray probe |
| --- | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.120s` | `0.112s` | `4269/s` | `0.0019s` |
| `B=16,T=64` | `0.238s` | `0.229s` | `4300/s` | `0.0026s` |
| `B=32,T=64` | `0.406s` | `0.394s` | `5046/s` | `0.0048s` |
| `B=128,T=16` | `0.446s` | `0.435s` | `4589/s` | `0.0164s` |

Source-backed circle-ray after the same patch:

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.293s` | `0.013s` | `0.065s` | `0.209s` | `0.158s` | `1748/s` |
| `B=16,T=64` | `0.626s` | `0.028s` | `0.140s` | `0.450s` | `0.341s` | `1636/s` |
| `B=32,T=32` | `0.565s` | `0.028s` | `0.105s` | `0.427s` | `0.315s` | `1812/s` |

Read: yes, observation can be optimized. This patch is a measured win, but it
mostly removes cheap Python overhead around walls/normalization. Source-backed
observation is still the largest bucket, so the next useful observation work is
source-row batching and dense exact circle-ray kernels. A GPU raycaster is still
a whole-loop architecture decision, not the next local patch.

Source-backed circle-ray path is now stacked and cursor-bounded. The benchmark
stacks source vector row states into one padded batch, and the source adapter
exports `body_write_cursor` so the observer can slice each row to its live body
prefix.

Current source-backed matrix:

| Shape | Loop | Env step | Adapter | Observation | Ray cast | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=8,T=64` | `0.252s` | `0.012s` | `0.060s` | `0.174s` | `0.144s` | `2035/s` |
| `B=16,T=64` | `0.467s` | `0.023s` | `0.114s` | `0.324s` | `0.277s` | `2194/s` |
| `B=32,T=32` | `0.403s` | `0.020s` | `0.082s` | `0.298s` | `0.255s` | `2540/s` |

Read: source stacking/cursor cleanup helps, but ray casting remains the largest
source-backed bucket. The next local observation target is exact circle-ray
math itself: dense/chunked batching first, then Numba or a broad phase only if
body counts and memory say dense math is the wrong shape.

Timing denominator: table seconds are total time over the profile chunk. For
`B=16,T=64`, the chunk has `1024` env row-steps and took `0.467s`; observation
was `0.324s` of that chunk. That means observation is about `69%` of the wall
time and about `0.316ms` per env row-step, not `0.324s` per single step.

Short larger-batch probes on the same current source/cursor path reached
`6985/s` at `B=64,T=16` and `7875/s` at `B=128,T=8`, with replay chunk files
around `1MB` for `1024` env row-steps. Treat these as batch-scaling probes, not
training settings: they use shorter rollouts and lower mean body counts than
the `T=64` rows.

Dense batch circle-ray patch landed after the cursor cleanup. Current
source-backed actor chunks:

| Shape | Loop | Adapter | Observation | Ray cast | Env step | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=16,T=64` | `0.255s` | `0.120s` | `0.105s` | `0.057s` | `0.024s` | `4016/s` |
| `B=32,T=64` | `0.466s` | `0.233s` | `0.182s` | `0.097s` | `0.045s` | `4392/s` |
| `B=64,T=32` | `0.353s` | `0.164s` | `0.146s` | `0.063s` | `0.039s` | `5801/s` |

Read: ray casting is no longer the only scalar-ray local story. Source adapter
and observation/scalar packing are now comparable buckets. The next missing
measurement is the visual-path actor/replay/learner shape plus real
policy/search and learner handoff, not another sidecar env-only victory lap.

## LightZero Boundary

LightZero is not all GPU.

- CPU-side work includes subprocess envs, ALE/preprocessing, env-manager
  orchestration, replay bookkeeping, checkpoint/eval, artifact scans, and
  process/file movement.
- GPU-side work includes Torch model inference and learner updates when CUDA is
  enabled and actually used.
- MuZero/MCTS is hybrid: CPU tree/control/bookkeeping plus GPU model calls and
  host/device movement.

Optimizer should profile LightZero as a concrete slow control loop, not assume
that moving CurvyTron to GPU is the first fix.

External throughput references agree with the current measurement order:
IMPALA/SEED-style systems split acting from learning only when throughput and
staleness justify it; SEED centralizes inference to use accelerators better;
Sample Factory emphasizes keeping rollout, inference, and learning busy on one
machine; EnvPool targets environment execution when env stepping is the real
slow bucket; Brax shows that full env+learner GPU residency is powerful but
requires a new tensor-native runtime. Sources: IMPALA
`https://research.google/pubs/impala-scalable-distributed-deep-rl-with-importance-weighted-actor-learner-architectures/`,
SEED RL `https://github.com/google-research/seed_rl`, Sample Factory
`https://proceedings.mlr.press/v119/petrenko20a.html`, EnvPool
`https://huggingface.co/papers/2206.10558`, Brax
`https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/d1f491a404d6854880943e5c3cd9ca25-Abstract-round1.html`.

Plain read for this repo: do not jump straight to Modal disaggregation or a GPU
CurvyTron rewrite. First time the visual LightZero-shaped loop. If acting waits
on learning, overlap in one container. If policy/search waits on small GPU
calls, batch or centralize inference/search. If env/render dominates, scale CPU
actors or build a tensor runtime. If checkpoint/eval dominates, split those
coarse jobs.

## Modal Mctx Boundary Evidence

Synthetic trainer-flat Modal Mctx boundary timings:

| Shape | Mctx | Steady H2D | Host setup |
| --- | ---: | ---: | ---: |
| `B=8, sim=8` | `2.947ms` | `0.901ms` | `0.238ms` |
| `B=64, sim=8` | `2.410ms` | `0.490ms` | `0.447ms` |
| `B=64, sim=16, depth=16` | `6.850ms` | `0.668ms` | not primary |
| `B=512, sim=8` | `3.657ms` | `0.414ms` | `1.796ms` |

First H2D is first-use overhead and should not drive architecture decisions.
These results are useful boundary evidence, not real CurvyTron rollout,
replay, learner, or source-fidelity evidence.

Retimed Modal L4 check after the D2H patch, app
`ap-u3YpTqQcqArxzFk5PI6ZbH`, same `curvytron_trainer_flat
B=64,P=2,obs=106,sim=8,hidden=64,depth=8`: compile+first `5.0005s`, Mctx
steady median `2.904ms`, host obs setup `0.622ms`, steady H2D median
`0.545ms`, selected-action D2H median `0.0471ms`, and action-weights D2H
median `0.0055ms`. First action conversion was `19.14ms`, likely first-use/sync
overhead. Read: steady H2D is small but visible, steady selected-action D2H is
negligible, and Mctx search is about `2.9ms` at this synthetic boundary. This
still does not measure CPU ray generation or source fidelity.

Scalar-ray native-observation Modal Mctx bridge now exists:
`curvytron_vector_trainer_sample` in
`src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`. It builds a live
strict `VectorTrainerEnv1v1NoBonus` sample, maps real `[B,2,106]`
observations/masks into Mctx roots, then uses the same synthetic Mctx dynamics.
First run, app `ap-ZkCdPu0mPNrniXaQAgxDjv`, `B=64,P=2,sim=8`: host
env/obs setup `0.206s` including init/reset/two env steps/mapping; steady
Mctx `2.330ms`; steady H2D `0.536ms`; action D2H `0.0157ms`. Read: this
connects the real strict native scalar-ray surface to GPU search timing. It is
useful boundary evidence, but not visual CurvyTron path evidence.

## Verdict

There is no fundamental blocker to a full GPU env/obs/model/search stack, but
the current source environment is a CPU object graph. A full GPU path means a
new tensor runtime plus parity tests before it can replace source truth.

Environment reorientation on 2026-05-10 keeps this bounded: the strict
`VectorTrainerEnv1v1NoBonus` path is real and useful as scalar-ray sidecar
diagnostics, but only for `1v1/no_bonus/P=2`. It has final observation/reward,
terminal metadata, truncation, autoreset staging, and replay-v0 plumbing. It is
not the primary visual LightZero path, broad lifecycle, 3P/4P, bonuses, or
source-fidelity completion. `source_env` and the JS oracle remain proof tools;
source fidelity still defines the rules.

Coach context stays separate. Old official/control Atari Pong measurements are
archived shape evidence only; the optimizer lane is now CurvyTron-only unless
the user explicitly reopens Pong. Use old Pong timing only as a reminder that
synchronous LightZero can be collect/eval/search heavy, not as an active target
or CurvyTron readiness evidence.

Near-term optimizer direction:

- keep the bounded `debug_visual_tensor` smoke/profiler as the visual debug
  baseline;
- keep the non-ALE LightZero visual adapter smoke current;
- add per-stage timing for env step, render, stack/normalize,
  policy/search, replay, learner, eval/checkpoint, and reset;
- keep scalar-ray source-backed and native-vector profiles as sidecar
  diagnostics for adapter, observation, replay, and GPU-search boundary costs;
- do not optimize scalar-ray env-only work as if it were the main path without
  real policy/search, learner handoff, and a visual-path comparison in the
  report;
- calibrate real policy/search timing on the faster `[B,2,106]` sidecar rows
  only as boundary evidence;
- measure replay/learner handoff with visual trainer payloads before process
  sharding claims;
- only promote full GPU env/obs work after Amdahl shares, p95/p99 action
  latency, replay/learner handoff, and fidelity gates justify it.
