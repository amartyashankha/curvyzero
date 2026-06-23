# Deferred Search Payload Design, 2026-05-22

Status: sidecar optimizer note. I inspected
`src/curvyzero/infra/modal/mctx_synthetic_benchmark.py` and the current
`batched_gpu_full_loop_reorientation_2026-05-20` working docs. No source code,
live Coach training, Modal runs, checkpoints, evals, GIFs, or tournament
artifacts were touched.

## Current Read

The fresh action-only rows isolate the important boundary:

| row | sims | roots/sec | read |
| --- | ---: | ---: | --- |
| replay-valid materialization | 16 | `47,936` | root value extract `0.258s`, visit-policy/root materialization |
| replay-valid materialization | 32 | `37,799` | root value extract `0.286s`, visit-policy/root materialization |
| action-only, replay off | 16 | `72,335` | selected actions only |
| action-only, replay off | 32 | `48,139` | selected actions only |

Replay-index rows themselves are not the apparent wall. The expensive hot-path
edge is forcing replay/training payloads to become CPU objects every step:
`action_weights`, root values, `CompactSearchResultV1` validation, and then
replay row construction. The CPU env only needs selected actions.

## Per-Iteration Data Flow

Current replay-valid compact MCTX loop:

```text
CPU HybridCompactBatch[k]
-> CPU CompactRootBatchV1 sidecars
-> JAX resident stack or H2D host stack + H2D invalid mask
-> JAX/MCTX search
-> CPU action + visit_policy + root_value
-> CPU CompactSearchResultV1 validation
-> CPU joint_action[B,P]
-> CPU VectorMultiplayerEnv.step(...)
-> CPU reward/done/legal/final sidecars + render state
-> renderer compact/delta H2D -> GPU latest frame
-> GPU resident stack[k+1]
-> CPU CompactReplayIndexRowsV1
-> CPU HybridCompactBatch[k+1]
```

Deferred-payload target loop:

```text
CPU/GPU root sidecars for batch k
-> JAX/MCTX search
-> CPU selected_action only
-> CPU joint_action[B,P]
-> CPU env/render step to produce batch k+1
-> staged search payload[k] + next-step reward/done/final sidecars
-> replay/training payload materialized in chunks, not on the action-critical edge
```

The env/search loop still alternates CPU and GPU while mechanics are CPU-owned.
The chosen serial edge should be exactly the action edge.

## GPU Vs CPU

| Data | Current owner | Hot-path need |
| --- | --- | --- |
| CurvyTron mechanics state | CPU NumPy/vector env | CPU until env rewrite |
| Compact batch sidecars: reward, done, masks, ids | CPU NumPy | CPU-visible for env/replay contracts |
| Renderer persistent layer/latest frame | JAX device | GPU-resident for next visual root |
| Visual stack `[B,P,4,64,64]` | host or JAX device | resident GPU in the preferred profile lane |
| Invalid/legal mask `[R,3]` | CPU, then JAX device | tiny H2D before search unless made persistent |
| MCTX tree/model/search tensors | JAX device | device-resident |
| Selected action `[R]` or `[active]` | JAX device, then CPU | required before CPU env step |
| Visit policy `[active,3]` | JAX device, currently CPU every step | replay/training only; delay/chunk |
| Root value `[active]` | JAX search tree, currently CPU every step | replay/training only; delay/chunk |
| Replay index rows | CPU NumPy | cheap; can be chunked or async |
| Learner target rows/RND frames | CPU/GPU depending adapter | materialize at sample/chunk edge |

## Required Per-Step Copies For Env Progression

Exactly required while the env is CPU-owned:

1. `selected_action` from search output to CPU for active roots.
   At B1024/P2 this is about `R=2048` scalar actions, roughly `8 KiB` as int32
   or less if compacted to active int16.
2. CPU scatter into `joint_action[B,P]` using already-host root sidecars
   `env_row` and `player`.
   This is a host write of about `4 KiB` at B1024/P2 as int16.
3. Pass `joint_action[B,P]` to `HybridBatchedObservationProfileManager.step`.

Not required before env progression:

- `visit_policy/action_weights`;
- root values;
- raw visit counts, logits, search tree summaries;
- `CompactSearchResultV1` object construction;
- replay-index rows;
- full observations or final observations;
- learner target rows;
- RND hashes/metrics.

Required for the next search, but not for stepping the env:

- post-step legal/invalid mask `[R,3]` to JAX unless held resident;
- renderer compact/delta/compose H2D to produce the next latest frame;
- resident stack update or host observation H2D, depending observation source.

## Delay Or Chunk

Safe to delay behind the action edge:

- `visit_policy[active,3]` and optional raw visit counts;
- `root_value[active]` and search metadata;
- `CompactSearchResultV1` validation, if replaced by sampled fast guards in the
  hot row;
- `CompactReplayIndexRowsV1`, because rows are small and not needed to choose
  the next action;
- full replay chunk assembly and learner target materialization;
- RND latest-frame extraction, hashing, and scalar metrics;
- sampled host observation mirrors.

Do not delay past semantic boundaries:

- selected actions for the CPU env step;
- reward/done/final-observation sidecars for the transition they belong to;
- terminal final observation before autoreset overwrites the row;
- record index ordering: search at record `k` consumes `observation[k]`, while
  reward/done/next/final sidecars come from step `k+1`.

## Ranked Designs

### 1. Action-Critical Loop Plus Deferred CPU Payload Chunk

Per step, read only `selected_action`, step the env, and append a lightweight
pending record id. Every `K` steps, read `visit_policy/root_value` for the chunk
and build replay rows in order.

Expected payoff: high for the current evidence, because action-only is
`1.27x-1.51x` faster than replay-valid per-step materialization on the fresh
rows.

Risk: medium. The main hazards are delayed failure localization, chunk ordering,
and terminal/final-observation alignment.

### 2. Service-Owned Search Payload Ring

Keep MCTX/search outputs in a service-owned device or pinned-host ring:
`selected_action` is exposed immediately, while `visit_policy`, root values,
raw counts, and metadata are committed asynchronously or at chunk boundaries.
CPU compact sidecars keep record ids, reward/done/final masks, and row/player
identity.

Expected payoff: high after the minimal chunk design, especially if root-value
extraction is the real sync wall rather than replay-row writing.

Risk: medium/high. It introduces lifetime and memory-pressure issues around
JAX outputs, plus stricter requirements for sampled parity and chunk flushes.

### 3. Compact Replay/Learner Chunk Owner

Promote the hot collection output from per-step objects to time-major compact
chunks. The collector stores actions immediately and stores replay/training
payloads by chunk; learner/RND adapters materialize tensors on sample, not
during action collection.

Expected payoff: strategically highest, but only after action-only/deferred
payload proves the wall moved. It prevents the next bottleneck from becoming
stock target/RND materialization.

Risk: high. This touches the training-data contract: final observations,
reward mutation, row/player ids, `to_play`, action history, visit targets, and
RND latest-frame semantics must all round-trip against the existing builders.

## Recommendation

Treat action readback as the only mandatory per-step search output. Build the
next profile canary around delayed `visit_policy/root_value` materialization
with replay rows chunked behind the env step. Keep the claim profile-only until
chunked replay materialization is parity-tested against the current compact
target-row path.
