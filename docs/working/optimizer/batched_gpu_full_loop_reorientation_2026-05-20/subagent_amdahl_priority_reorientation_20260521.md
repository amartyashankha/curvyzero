# Subagent Amdahl Priority Reorientation

Date: 2026-05-21

Scope: independent prioritization critique from the reorientation docs and
optimizer front-door notes. No live runs, trainer defaults, checkpoints,
tournaments, or code were touched.

## Bottom Line

The main blocker is no longer CurvyTron rendering. Rendering was the right wall
to attack earlier, and the persistent/direct GPU lane made it much cheaper. But
the newest real-consumer rows show that the fast resident batch loses most of
its advantage when it enters actual LightZero `MuZeroPolicy.collect_mode.forward`.

The current critical numbers:

| row | H100 roots/s | L4/T4 roots/s | read |
| --- | ---: | ---: | --- |
| synthetic resident probe, scalar off | `~10980` | `~5840` | batch-resident headroom exists |
| real LightZero collect-forward, scalar off | `2669` | `2159` | public collect/search path consumes most of it |
| real LightZero collect-forward, scalar on | `2100` | `2054` | scalar edge is real but not the main collapse here |

That means:

- H100 real collect-forward keeps only about `24%` of the synthetic resident
  throughput.
- L4/T4 keeps about `37%`.
- H100 is only about `1.24x` faster than L4/T4 scalar-off and almost tied
  scalar-on, which points away from pure GPU rendering and toward CPU tree,
  Python, synchronization, or LightZero output handling.
- Rendering in these rows is tiny versus the collect-forward/probe bucket:
  about `0.2-0.3s` render versus `6-8.5s` real-consumer probe time.

So the next optimizer move should split the real collect-forward bucket. If the
model initial inference is cheap, the wall is LightZero CPU MCTS/search/output
handling. If initial inference is expensive, the wall is model/GPU batch
topology or tensor preparation. Until that split lands, deeper render work is
mostly aiming at a smaller fraction of the loop.

## Amdahl Read

There are now two useful ceilings, and they say the same thing from different
angles.

First, stock/full-loop profile rows with zero observations bound the remaining
observation upside. Recent C512 rows put real batched GPU observation within
roughly `1.17x-1.25x` of zero observation, depending on the exact sim/profile
shape. Even the more favorable C256/C512 pairs bound observation cleanup around
`1.25x-1.6x`, not `5x`.

Second, the resident batch canary shows what a large batch can do before the
real LightZero consumer: `~11k` H100 roots/s. But the actual public collect
path drops to `~2.7k` roots/s before scalar timestep materialization. That is
the current Amdahl cliff.

Plain read:

```text
renderer-only cleanup: useful bounded win
resident batching before real search: large synthetic headroom
actual LightZero collect/search: current major blocker
```

The main thread should therefore stop asking "how do we make the renderer
another 2x faster?" as the primary question. The sharper question is:

```text
Where inside collect-forward does the resident batch die: tensor/model
inference, CPU MCTS/search, decode/output construction, or synchronization?
```

## Main Blockers And Falsifiers

### 1. LightZero collect/search boundary

Hypothesis: `MuZeroPolicy.collect_mode.forward` is dominated by CPU-tree MCTS,
CPU NumPy/list conversion of roots/logits/latents, and output/decode
bookkeeping, not by CurvyTron observation render.

Evidence:

- Real-consumer H100 scalar-off is `2669 roots/s` versus synthetic resident
  `~10980 roots/s`.
- H100 only modestly beats L4/T4 in the real-consumer row.
- Scalar materialization lowers H100 real-consumer throughput from `2669` to
  `2100`, but the bigger collapse already happened before scalar-on.
- Code audits note that LightZero initial inference can run on GPU, then
  detaches values, latent roots, and logits to CPU NumPy/list before MCTS.

Falsifying experiments:

1. Run a model-initial-inference-only canary over the same pre-scalar
   `uint8[B,2,4,64,64]` stack, same scratch policy, same B512/A16 shape, H100
   and L4/T4.
2. Run collect-forward with a simulation ladder, ideally `sim1/sim2/sim4/sim8`,
   with scalar materialization off.
3. Add or report split timers:
   `tensor_prepare`, `h2d`, `normalize`, `initial_inference`, CPU-root
   conversion, MCTS/search loop, output decode, and action legality check.

Decision:

- If initial inference is a small fraction of collect-forward and time scales
  with simulations, prioritize search/CPU-tree replacement or a custom batched
  collector/search boundary.
- If initial inference dominates, optimize model batch topology, dtype/device
  prep, and Torch/JAX synchronization before touching search.
- If decode/output dominates, slim LightZero output handling before larger
  architecture work.

### 2. Stock collector/env-manager scalarization

Hypothesis: even if collect-forward is improved, the stock LightZero boundary
still destroys residency through env-id dicts, `BaseEnvTimestep` rows,
`GameSegment` lists, NumPy copies, and scalar-ready observations.

Evidence:

- The trusted stock env wrapper owns `VectorMultiplayerEnv(batch_size=1)` and
  returns one LightZero observation dict per env.
- The profile batched manager still adapts back to env-id keyed timesteps for
  stock compatibility.
- Zero-observation rows are not free: policy collect, MCTS, manager, replay,
  and learner work remain large.
- C768 zero-observation being slower than real observation in one grid is a
  topology/scheduling warning, not a renderer warning.

Falsifying experiments:

1. Build a profile-only batched-ready table canary:
   `[policy_env_id, row, player, obs, mask, reward, done]` consumed directly by
   policy/model code, with scalar env-id materialization only after action
   selection.
2. Compare three same-work rows:
   stock zero-observation manager, batched-ready zero-observation table, and
   batched-ready real observation table.
3. Report object fanout and bytes: env-id dict count, timestep count, info
   count, observation bytes copied, final-observation bytes, and GameSegment
   append count.

Decision:

- If batched-ready zero observation beats stock zero by a large margin, invest
  in a custom collector/env-manager fast path.
- If it does not beat stock zero, the current stock manager is not the main
  remaining bottleneck; look harder at search/model/replay.

### 3. Live-root collapse and topology saturation

Hypothesis: wide one-process batching does not keep enough useful roots alive
under normal death and does not scale monotonically at high width. The batch
that reaches policy/search is smaller or more irregular than the no-death
profiles suggest.

Evidence:

- Normal-death C256 passed as a semantic gate but averaged only about `108`
  roots per manager step, with low GPU utilization and `485 steps/s`.
- C768 did not scale cleanly; one zero-observation row was worse than real.
- B1024 synthetic resident did not materially beat B512 scalar-off and was
  worse scalar-on.

Falsifying experiments:

1. For normal-death rows, report live physical rows, live policy roots,
   terminal rows, omitted rows, autoresets, and roots per collect call.
2. Compare dense roots versus compacted active roots in the real-consumer
   canary, with identical live-root sets.
3. Repeat C512/C768 real-vs-zero with sim2/sim4 only if needed to decide
   saturation, not as an open-ended width hunt.

Decision:

- If compaction restores throughput, prioritize active-root compaction and
  terminal/autoreset batching.
- If C512/C768 remain flat after clean repeats, stop widening the one-process
  manager and move to actor-parallel or search-boundary work.

### 4. Host stack/update and device handoff

Hypothesis: host stack maintenance and scalar materialization still matter, but
only become the top priority if policy/search can consume the batch without
falling into the LightZero CPU tree first.

Evidence:

- Synthetic resident scalar materialization roughly halves throughput in some
  rows.
- Device-latest reduced probe H2D but regressed total throughput because the
  host stack was still maintained and a second device stack was added.
- Persistent renderer moved long-row cost toward env step and stack update.
- In the real-consumer row, scalar-on is slower but not the main collapse.

Falsifying experiments:

1. Add a no-host-stack profile mode only when the pre-scalar consumer is active:
   render/update compact `uint8` stack, feed initial inference or collect
   directly, do not maintain the host float32 chronological stack.
2. Compare four rows: host stack on/off crossed with initial-inference-only and
   full collect-forward.
3. Report stack bytes, D2H bytes, materialized timestep count, and sync count.

Decision:

- If no-host-stack helps initial inference but not collect-forward, the wall is
  search/CPU tree.
- If no-host-stack helps both, promote device/uint8 handoff as a serious
  architecture lane.
- Do not build a host-only ring buffer unless the downstream consumer avoids
  immediately materializing the same chronological float32 stack.

### 5. RND and training sidecars

Hypothesis: RND is a real overhead axis but not the current explanation for the
real-consumer collapse.

Evidence:

- C512 RND meter rows cost about `10-12%` in the recent profile shape.
- Aggressive direct-surface RND training cadence can dominate local rows, but
  that is a cadence/training knob, not a renderer result.

Falsifying experiments:

1. After the collect-forward split, run matched no-RND and `rnd_meter_v0` rows
   at the selected architecture shape.
2. Keep cadence explicit: update10/update100 should not be mixed into renderer
   conclusions.
3. Verify predictor changed, target frozen, reward unchanged, latest-frame
   source, and terminal/latest-frame handling.

Decision:

- If RND becomes dominant only after search/collector work improves, tune
  cadence/execution separately.
- Do not block the collect-forward split on positive-RND reward work.

## Rabbit Holes

- More scalar `jax_gpu` work. It is the wrong shape: one row at a time, copy
  back to NumPy, slower than CPU.
- Exact `block_704_gray64` or browser-pixel parity as the primary speed lane.
  Keep it as a reference/debug lane; the learned policy candidate is direct
  semantic observation with gates.
- Another renderer-only rewrite without a full-loop or real-consumer Amdahl
  bound. Current C512 observation headroom is too small for the stated goal.
- B1024/B2048/B4096 synthetic scaling sweeps unless tied to a specific
  consumer bottleneck. B1024 already failed to improve the useful resident
  probe.
- C768/C1024 width hunting in the one-process manager without a topology
  hypothesis. The plateau/anomaly already says width alone is not the plan.
- Host-only ring buffers that merely move the copy to policy materialization.
- A full JAX/MCTX rewrite before the initial-inference versus CPU-tree split.
  A tiny scratch spike can be useful after the split says search is the wall.
- Duplicate-seed runtime variance as a main lane. Use it only when anchor rows
  disagree in workload counts.
- Positive-RND reward normalization and launch policy as part of renderer
  optimization. Keep it separate.
- Reporting hybrid or resident profile rows as actual Coach training speed.
  The latest actual Coach speed doc is explicit: production speedup is not
  proven yet.

## What The Main Thread Should Do Next

1. Implement the real-consumer split canary:
   `pre-scalar uint8 stack -> Torch normalize -> MuZero model initial_inference`
   on the same scratch policy and shape as the collect-forward rows.

2. Run the smallest decisive matrix:

| row | compute | shape | consumer | sims | scalar edge |
| --- | --- | --- | --- | --- | --- |
| A | H100 | B512/A16 | synthetic resident anchor | 8 | off |
| B | H100 | B512/A16 | model initial inference only | n/a | off |
| C | H100 | B512/A16 | real collect-forward | 1/2/4/8 | off |
| D | L4/T4 | B512/A16 | model initial inference only | n/a | off |
| E | L4/T4 | B512/A16 | real collect-forward | 1/2/4/8 | off |
| F | H100 | B512/A16 | real collect-forward | selected sim | on |

Only add normal-death and RND rows after the split identifies the wall.

3. Add durable run/result capture before more detached Modal waves. The docs
   already note lost PTY handles and missing JSON payloads; that is cheap to
   fix compared with rerunning ambiguous rows.

4. If initial inference is cheap, choose one of two architecture lanes:

- stock-compatible custom collector boundary that preserves batched ready rows
  through policy/search and scalarizes only after actions;
- small search replacement/prototype lane, possibly MCTX/JAX scratch-only, to
  test accelerator-resident batched search without promising checkpoint
  compatibility.

5. If initial inference is expensive, stay closer to the current stack:

- optimize tensor preparation/normalization and model batch topology;
- test uint8/device handoff without host stack update;
- measure H100 versus L4 again after model prep is cleaned up.

6. Keep the current batched GPU manager profile-only. Before any Coach-facing
   recommendation, require matched no-RND, `rnd_meter_v0`, normal-death,
   long-survival, and zero-observation controls, with learner iterations/hour
   and collected env steps/sec reported separately.

## Practical Priority Order

1. Split collect-forward into initial inference versus CPU tree/search.
2. Based on that split, prototype either batched collector/search or model
   tensor-prep/device handoff.
3. Keep renderer/stack cleanup opportunistic and bounded by zero-observation
   gaps.
4. Keep normal-death and RND as promotion gates, not the main speed search.
5. Do not touch live training until the profile lane beats matched controls in
   the same denominator and preserves semantics.

