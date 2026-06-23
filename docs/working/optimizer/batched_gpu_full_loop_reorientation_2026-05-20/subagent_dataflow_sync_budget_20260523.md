# Dataflow And Sync Budget, 2026-05-23

Status: docs-only optimizer side-agent note. I read the current optimizer docs
and the local result JSON under
`artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_*_20260523`.
I did not edit source code, touch live Coach training runs, or touch Modal
volumes.

## Measured Anchor

Shape:

```text
H100, B=512 env rows, P=2 players, R=1024 roots, A=3 actions,
actor_count=16, sim16, 80 measured iterations, 20 warmup,
host_uint8_pinned LightZero input, compact replay proof on,
profile-only, no train_muzero call.
```

Fresh durable rows:

| Row | Measured wall | Profile throughput | Probe wall | Search/model read |
| --- | ---: | ---: | ---: | --- |
| `direct_ctree_gpu_latent` | `13.733s` | `5,965 steps/sec` | `8.016s` | real initial model, real recurrent model, real CPU CTree |
| `service_tax_probe` | `6.910s` | `11,855 steps/sec` | `2.177s` | real initial/recurrent model, fake search update, no CTree |
| `mock_search_service` | `5.472s` | `14,970 steps/sec` | `0.548s` | real initial model, fake policy action, no recurrent, no CTree |

The ratios are the main Amdahl clue:

```text
service_tax / direct: 1.99x
mock / direct:        2.51x
mock / service_tax:   1.26x
```

The search boundary has real headroom, but not a standalone `10x` story. A
larger win needs ownership changes across observation, search, and replay.

## Payload Size Ledger

At `B512/P2/R1024/A3`:

| Payload | Shape | Size | Big/small read |
| --- | --- | ---: | --- |
| Joint action into env | `[512,2] int16` | `2 KiB` | Small; CPU env needs it. |
| Latest frame | `[512,2,1,64,64] uint8` | `4 MiB` | Medium; should stay GPU-resident in the hot path. |
| Policy stack | `[512,2,4,64,64] uint8` | `16 MiB` | Big current search input. |
| Compact sidecar payload | ids, masks, rewards, dones, actions | `21.7 KiB/iter` | Small; `0.13%` of rendered stack bytes. |
| Legal mask | `[1024,3] bool` | `3 KiB` | Small; sync shape matters more than bytes. |
| Selected action | `[1024] int32/int64` | `4-8 KiB` | Small; mandatory before CPU env step. |
| Visit policy | `[1024,3] float32` | `12 KiB` | Small; needed for replay target. |
| Root value | `[1024] float32` | `4 KiB` | Small; needed for replay target. |
| Current direct compact output | selected/visit/value and ids | `45 KiB` | Small. |
| Mock/service compact output | selected/visit/value and ids | `24 KiB` | Small. |
| Direct per-sim recurrent output | roughly `[R, reward/value/logits]` | O(`20 KiB/sim`) | Small bytes, bad if it forces one sync per sim. |

Plain read: the byte monster is the visual stack. The latency monster is the
repeated CPU/GPU search loop, not the few KiB of actions or visit policies.

## Current Profile Path

Current measured path:

```text
previous selected actions
-> CPU CurvyTron step
-> compact CPU sidecars
-> JAX persistent renderer and host stack update
-> host uint8 stack to Torch/LightZero
-> direct CTree / service-tax / mock search probe
-> compact search arrays
-> index-row replay proof
-> next selected actions
```

The table uses the `direct_ctree_gpu_latent` row for current timings unless
noted. Times are total over 80 measured iterations, with approximate per-env
iteration in parentheses.

| Stage | Owner | Payload/order | Required sync point | Current timing | Optimization idea |
| --- | --- | ---: | --- | ---: | --- |
| Previous action handoff | CPU | `2 KiB` joint action | Required before CPU env step | included in actor step | Keep as CPU-visible while env is CPU-owned. It is not the bottleneck. |
| CPU env step and sidecars | CPU/NumPy | Small sidecars; possible large render state if copied | Must complete before render/search input for this iteration | `actor_step_wall 2.424s` (`30.3ms`) | Native compact buffers and persistent env/render SoA; avoid scalar/public object fanout. |
| Payload merge | CPU | Small compact sidecars | Before compact batch/render request | `gather_merge 0.029s` (`0.36ms`) | Already small; keep array-shaped. |
| Render-state production to compact | CPU | trail/head/bonus arrays, O(MiB) depending body capacity | Before renderer H2D/update | `renderer_production_to_compact 0.077s` (`0.97ms`) | Persistent compact state; update deltas instead of rebuilding ownership. |
| Renderer delta pack | CPU/JAX prep | compact render deltas | Before JAX update | `renderer_persistent_delta_pack 0.223s` (`2.79ms`) | Preallocated delta buffers; fixed-shape dirty ranges. |
| Renderer H2D/update/draw | GPU/JAX with CPU submits | Latest frame target `4 MiB`; render state smaller than full stack | Current profile waits through update/draw | H2D `0.435s`, update `0.622s`, raw draw `0.025s` | Raw draw is tiny. Attack transfer/update waits and state ownership. |
| Frame readback and host stack write | GPU -> CPU, then CPU | Latest `4 MiB`; stack `16 MiB` | Required in host-stack LightZero path | D2H `0.105s`; stack shift `0.310s`; observation total `2.183s` (`27.3ms`) | Device-resident stack/search input; terminal-only host snapshots. |
| Compact batch build | CPU | `21.7 KiB` compact payload plus stack reference | Before search probe | `0.0027s` (`0.03ms`) | Fine. Keep `copy_observation=False` style. |
| Root input prepare and H2D | CPU -> GPU/Torch | `16 MiB` stack plus `~3 KiB` mask | Before model/search | input prepare `0.123s`; H2D summary `0.063s` | Make root observation a resident device view; keep mask as one batched copy. |
| Initial model inference | GPU/Torch | `[R,4,64,64] uint8/float` input | Root logits/value needed for root prep | `0.361s` (`4.52ms`) | Keep batched; no per-root Python. |
| Root prep for CTree | CPU/list | logits/value/masks, small bytes | Before CTree simulations | `0.971s` (`12.1ms`) | Fixed `[R,3]` array API; avoid scalar LightZero roots/lists. |
| Direct CTree simulation loop | CPU CTree + GPU recurrent | Per sim: small action H2D and reward/value/logits D2H | Current direct path syncs every sim | search `6.164s`; non-model `5.893s`; model total `2.123s`; CTree traverse+backprop `1.683s` | Move search body device-side or batch/service the recurrent leaves so sim16 is not 16 host/device control turns. |
| Search output decode | CPU | `45 KiB` compact output | Required before action/replay use | decode/output assembly `0.084s` (`1.05ms`) | Keep compact arrays; no public scalar output materialization. |
| Compact replay proof | CPU | Index rows; no full observation copy | Needs current search result and next-step ids/actions | direct `0.254s`; service/mock `~0.194s` | Keep index-only hot path; defer full target row materialization. |

Service comparator timings show which current direct buckets are structural:

| Bucket | Direct | Service-tax | Mock | Read |
| --- | ---: | ---: | ---: | --- |
| Full measured wall | `13.733s` | `6.910s` | `5.472s` | Current direct is about `2x` slower than real-model/no-CTree service-tax. |
| Probe wall | `8.016s` | `2.177s` | `0.548s` | Search boundary dominates the difference. |
| Initial inference | `0.361s` | `0.363s` | `0.361s` | Stable, not the problem. |
| Recurrent inference | included in `2.123s model total` | `1.380s` | `0.000s` | Recurrent model matters, but service-tax still leaves a lot of full-loop wall. |
| Real CTree traverse/backprop | `1.683s` | `0.000s` | `0.000s` | CPU CTree/list boundary is real, but not the only cost. |
| Search update | `6.164s` real MCTS | `0.254s` fake | `0.000s` | Replacing the search body changes Amdahl. |

## Next Compact-Service Path

Target path:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> learner/replay materialization at coarse boundaries
```

The key design change is not a new wrapper. The change is to make compact arrays
the owner of root identity, action, visit policy, root value, and replay index
rows, while scalar LightZero objects become validation/debug edges.

| Stage | Owner | Payload/order | Required sync point | Timing status | Optimization idea |
| --- | --- | ---: | --- | --- | --- |
| Compact env action input | CPU | `2 KiB` joint action | Required before CPU env step | Same as current until env moves | Accept this sync for CPU env. A GPU env would change it, but that is a bigger semantic rewrite. |
| Compact env/sidecar write | CPU, ideally native buffers | Small fixed arrays; no full obs copy | Before root batch for current record | Current actor wall `~2.1-2.4s/80` | Direct parent buffers; no per-env scalar dict/list payloads. |
| Device-resident observation stack | GPU/JAX/Torch boundary | Avoid `16 MiB` host stack H2D | Search needs a fresh root view; host only samples/canaries | Not proven by these LightZero rows | This is the cleanest way to stop paying full-stack ownership every iteration. |
| CompactRootBatchV1 build | CPU metadata plus device/host obs reference | `~KiB` sidecars plus obs view | Must validate ids/masks before service | Current compact build `~0.003s/80`; root prep direct `0.971s/80` | Root build must stay array-native; root prep should not explode into CTree scalar/list objects. |
| CompactSearchServiceV1 call | Service owns CPU/GPU scheduling | Root obs view, mask, ids; outputs action/visit/value | One service boundary per batch/chunk | Existing comparator ceiling: service-tax `11,855`, mock `14,970` steps/sec | Make direct, service-tax, and mock share the same boundary. Then swap in fixed-shape search behind it. |
| Search body | Device-side or batched service | Per-sim leaf payloads are KiB-scale | Target: no per-sim host-visible sync | Current direct syncs every sim; service-tax/mock avoid CTree | Fixed-action array CTree, MCTX/JAX, Torch/Triton, or local batched inference/search service. |
| Selected action readback | GPU/service -> CPU | `4-8 KiB` | Required before next CPU env step | Small; included in probe/readback | This sync is acceptable. Budget it once per iteration, not per sim. |
| Visit/root value readback | GPU/service -> CPU/replay | `~16 KiB` | Required before replay commit; can be delayed within chunk | Current direct output `45 KiB`; mock/service `24 KiB` | Defer or overlap with next env step if record attachment is proven. |
| CompactReplayIndexRowsV1 | CPU compact replay owner | Index rows, no full observation/next-observation copies | Needs search result k and transition k+1 identity | `0.19-0.25s/80` proof bucket | Keep hot path index-only; materialize learner tensors later. |
| Full target/replay materialization | Learner/replay edge | Potentially large observations and RND frames | Not in collect hot loop | Heavy materialized proof was previously too slow | Coarse learner boundary only, with parity tests against trusted target rows. |

## Sync Budget

Acceptable hot-loop syncs:

| Sync | Budget | Why acceptable |
| --- | ---: | --- |
| Selected action readback | once per iteration, `4-8 KiB` | CPU env cannot step without actions. |
| Legal/action mask H2D | once per iteration, `~3 KiB` | Fine if batched and not coupled to unrelated device waits. |
| Visit policy + root value readback | once per iteration or chunk, `~16 KiB` | Needed for replay targets; can be delayed/overlapped after action is chosen. |
| Terminal final observation snapshot | rare, correctness-critical | Pay it when terminal/autoreset rows exist. |
| Warmup/profile barriers | outside measured hot path or clearly labeled | Fine for measurement hygiene. |

Structurally bad syncs:

| Sync | Why it changes Amdahl badly |
| --- | --- |
| Per-simulation leaf action H2D plus ready wait | `sim16` means 16 ordering turns inside one env iteration. |
| Per-simulation recurrent output D2H plus listification | Bytes are tiny, but it serializes GPU model work with CPU CTree backprop. |
| Full visual stack GPU->CPU->GPU ownership | `16 MiB` per iteration at B512, and it forces search to start from host. |
| Public/scalar LightZero timestep materialization in collect | Turns array facts into per-env objects before replay/learner needs them. |
| Full replay row materialization during collection | Copies observations/next observations in the wrong part of the loop. |

## What Designs Change Amdahl

| Design | Expected effect | Why |
| --- | ---: | --- |
| Compact service boundary only | `1.0-1.1x` if only a wrapper | It improves correctness and denominator clarity, not speed by itself. |
| Compact service made real against service-tax/mock ceiling | `~2.0-2.5x` vs current direct comparator | Current rows already show this gap when CTree/search body is removed or faked. |
| Device-resident stack/search input | `1.2-1.7x` class | Removes full-stack host ownership and waits; bounded by refresh-off style rows. |
| Replay index rows and deferred materialization | `1.05-1.3x` alone | Small alone, but required to stop replay from re-expanding the hot loop. |
| Fixed `[R,3]` array-native CPU CTree | `1.1-1.6x`, maybe `2x` if list-heavy | Removes Python/list roots but still keeps CPU tree semantics. |
| Device-side or fixed-shape search body | `2-5x` for search bucket | Removes per-sim CPU/GPU ping-pong and scalar tree control. |
| Many-producer local search/inference service | `3-8x` if batch fill is real | Changes the synchronous one-batch collect/search cadence. |
| 10x architecture | compact env/render + compact replay + batched/device search | Needs multiple ownership contracts to disappear, not one faster kernel. |

## Bottom Line

The current direct profile moves one large object and many small-but-synchronizing
objects:

```text
large: host uint8 stack, 16 MiB/iteration
small but dangerous: recurrent actions/outputs, KiB/sim with one sync per sim
small and acceptable: selected action, legal mask, visit policy, root value
```

The next compact-service path should preserve only two hot ordering points:

```text
search result action -> CPU env step
search result visit/value -> replay commit
```

Everything else should either stay device/resident, stay compact/index-only, or
move to a coarse learner/debug boundary. That is the difference between a
`~2x` comparator cleanup and a real `5x+` architecture attempt.
