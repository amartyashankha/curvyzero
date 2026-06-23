# Reorientation: Fast Falsifiers First

Date: 2026-05-22

Status: active optimizer working memory. Do not touch live Coach training runs.

## Plain Goal

Speed up the actual CurvyTron training loop, not just a convenient benchmark.

Every optimization claim needs:

1. the measured fraction of wall time it attacks;
2. the smallest experiment that can kill or keep it;
3. a clear decision before we build more around it.

## What Went Wrong

We spent too long polishing small boundaries after their Amdahl share had
already shrunk.

The clearest example is flat-A3:

```text
matched full-loop row:
  direct LightZero CTree: 516.55 steps/sec
  flat-A3 CTree:          509.69 steps/sec
```

Flat-A3 is valid. It reduces a real CTree payload/list boundary. But the full
loop still pays root prep, recurrent inference, device-to-host conversion,
env/render/stack, replay, learner, and RND. So the full-loop row did not move.

The rule now is simple:

```text
Do not keep optimizing a sub-slice after a matched full-loop or same-denominator
falsifier says it is not moving the wall.
```

## Current Amdahl Map

The current corrected bottleneck is not rendering alone and not CTree backprop
alone.

The hot shape is:

```text
compact CurvyTron batch
-> scalar/object LightZero env boundary
-> CPU/list root prep
-> Python MCTS simulation loop
-> GPU recurrent inference
-> CPU/list reward/value/policy backprop
-> per-env action output
-> replay/target/RND/learner objects
```

The matched train-profile wins so far are real but small:

```text
no-RND:
  stock:              433.17 steps/sec
  direct output-fast: 566.19 steps/sec
  speedup:            about 1.31x

RND meter:
  stock:              351.02 steps/sec
  direct output-fast: 448.52 steps/sec
  speedup:            about 1.28x
```

That says the direct hook and output fast path helped, but did not change the
architecture enough for a 5-10x result.

## What We Have Already Falsified

- More CPUs are not the answer for the current boundary. CPU64 was slower than
  the normal H100 sidecar shape.
- Renderer-only work is no longer the primary 10x lane in the current full-loop
  denominator. It still matters for long env-only rows and fidelity, but it is
  not enough.
- Flat-A3 alone is not a launch-speed win. It is useful evidence about list ABI
  cost, not Coach advice.
- Precomputing recurrent outputs helps but is not enough by itself. The
  `direct_ctree_gpu_latent_precomputed_recurrent` falsifier improved the
  sidecar by about `1.38x`, not 10x.

## Top Hypotheses Now

### 1. Closed Compact Search/Replay Ownership

Hypothesis:

```text
The biggest remaining waste is tearing compact batches back into public
per-env LightZero objects before search/replay can use them.
```

Smallest keep/kill experiment:

```text
HybridCompactBatch
-> direct CTree compact arrays
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

The search-selected actions must drive the next env step. Stock LightZero
objects should exist only as validation adapters, not as the measured hot path.

Keep condition:

```text
same-denominator closed compact row plausibly beats current direct by a
3x-class margin, or clearly exposes the next wall.
```

Kill condition:

```text
if compact replay/index rows add more than 15-20% wall or throughput stays near
current direct, stop polishing this shape and escalate to a search-service /
native-vector architecture.
```

### 2. Search Service / Array-Native MCTS

Hypothesis:

```text
The real speedup requires a batched search owner that keeps tree/search state
as arrays and avoids Python/list backprop every simulation.
```

Smallest keep/kill experiment:

```text
mock_search_service vs direct_ctree_gpu_latent, same B512/A16/sim16 denominator
```

Keep condition:

```text
mock service is at least about 2x faster than current direct in the same shape.
```

Kill condition:

```text
if mock service is close to direct, search-service work is not the next big
move; profile replay/RND/learner/env scheduling instead.
```

### 3. RND / Replay / Learner Denominator

Hypothesis:

```text
After search is compacted, RND, replay, and learner object work becomes the
next visible denominator.
```

Smallest keep/kill experiment:

```text
same shape with:
  no RND vs rnd_meter_v0
  replay index rows vs no-op replay writer
  learner real vs fixed prebuilt tensor/no-op learner
```

Keep condition:

```text
one of these flat costs owns a meaningful fraction after compact search.
```

Kill condition:

```text
if none move wall by at least 10-15%, stop local polishing and move to topology:
many actors -> batched search service -> compact replay owner.
```

## Current Minimal Experiment Wave

Local first:

```text
profile_hybrid_batched_observation_manager.py
  zero observation
  uint8 stack
  native actor buffer
  no scalar timestep materialization
  native-vector-boundary probe

profile_hybrid_batched_observation_manager.py
  same shape
  closed-compact-consumer probe
  arrays target mode
```

Sidecar next:

```text
mock_search_service ceiling
precomputed recurrent falsifier
compact replay proof row
```

Real full-loop only after those:

```text
stock train_muzero vs direct_ctree_gpu_latent
same H100 C64/sim16/3 learner denominator
attestation required
```

## Current Priority

Measure the closed compact search/replay path first.

Do not keep polishing flat-A3, CPU count, or renderer-only lanes as the main
thread. They can remain evidence or side tests, but the next decisive question
is whether compact batches can survive through search and replay without the
public per-env object boundary.

## Result Update

The closed compact search/replay proof already exists as
`hybrid_compact_service_replay_proof`. It is not a stock trainer default. It is
a profile-only sidecar path.

Fresh H100 rows:

```text
B512/A16/sim16 direct gpu-latent:                 4651.06 roots/sec
B512/A16/sim16 precomputed recurrent:            5376.34 roots/sec
B512/A16/sim16 mock search service, no public out: 9109.54 roots/sec
B512/A16/sim16 direct + compact replay proof:    6222.32 roots/sec
```

The compact replay proof row:

```text
selected search actions drive next env step: yes
CompactReplayIndexRowsV1 target rows:       61440
public LightZero output bytes:              0
proof cost:                                 0.103s total, about 1.68 us/row
```

Updated read:

```text
Compact replay/index rows are not the wall. Recurrent inference is not the sole
wall. The next implementation target is the search-service boundary itself:
root prep, per-simulation CPU/list CTree/control, recurrent-output handling,
and the actor/env/observation scheduling around the service.
```
