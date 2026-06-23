# Batched GPU Full-Loop Reorientation

Date: 2026-05-20

Status: active optimizer working memory.

## 2026-05-22 Fast Falsifier Reset

Read
[reorientation_20260522_fast_falsifiers.md](reorientation_20260522_fast_falsifiers.md)
before starting another optimizer patch.

For the current plain-language map of GPU MCTS/Search work, read
[gpu_mcts_current_flow_explainer_20260522.md](gpu_mcts_current_flow_explainer_20260522.md).

For the compact one-iteration dataflow map, read
[subagent_full_iteration_dataflow_20260522.md](subagent_full_iteration_dataflow_20260522.md).
It separates the trusted stock `train_muzero` lane from the profile-only
compact lane and names the current Amdahl wall as next-search-input handoff,
not raw frame drawing.

Plain rule:

```text
Every speed idea needs a measured wall fraction, a smallest falsifier, and a
kill/keep decision before we build around it.
```

Current priority:

```text
closed compact env/observation state handoff
```

That means:

```text
HybridCompactBatch
-> compact env step
-> renderer/observation/stack update
-> compact root batch
-> MCTX/JAX search
-> CompactSearchResultV1 / CompactReplayIndexRowsV1
```

with search-selected actions driving the next env step and stock LightZero
objects used only as validation adapters. Replay-index rows are cheap. MCTS
search is small in the current repeated compact denominator. Do not spend
primary effort on flat-A3, CPU count, or renderer-only work unless a fresh
same-denominator profile says those slices have become hot again.

Fresh result:

```text
The existing profile-only compact replay proof passed on H100 B512/A16/sim16:
  direct + CompactReplayIndexRowsV1 proof: 6222.32 roots/sec
  replay proof wall: 0.103s over 61440 rows
  public LightZero output bytes: 0
```

Plain read:

```text
Compact replay/index rows are cheap enough. The repeated compact loop moved the
wall to the env/observation side: actor render-state write, production-to-
compact conversion, renderer/stack update, and root-batch handoff. Search-
service work still matters for the larger architecture, but it is not the first
bucket to polish in the current closed-loop profile.
```

Latest patch read:

```text
The first state-ownership canary worked. Profile-only single-actor borrowed
render state removes the parent actor_render_state_write bucket and moves the
best H100 B1024/P2 resident refresh-on row from about 32.9k to 48.6k
roots/sec at sim16, and from 24.0k to 36.0k at sim32. This is still not Coach
launch advice, but it confirms the current wall is observation/search-input
handoff, not CurvyTron physics or raw GPU drawing.
```

## High-Level Goal

Speed up the actual CurvyTron stock LightZero training loop while keeping the
learning problem the same.

The optimizer should not optimize an isolated benchmark unless that benchmark
answers a question about the real loop. The real loop includes observation
construction, LightZero collection/search, replay, learner updates, RND when
enabled, checkpoint/eval sidecars when intentionally included, and tournament
compatibility.

The current high-level plan is:

1. Use profile-only probes to find the active Amdahl wall.
2. Build the smallest speed path that removes that wall without changing
   MuZero semantics.
3. Prove semantic parity against stock LightZero.
4. Reconnect the speed path to matched stock `train_muzero` profiles.

## Plain Current Truth

The trusted CurvyTron training path is still stock LightZero with
`source_state_fixed_opponent` and CPU policy observations:

```text
browser_lines + simple_symbols -> cpu_oracle -> [4,64,64]
```

`compute=gpu-*` moves LightZero model/search/learner work to GPU. It does not
move CurvyTron observation rendering to GPU.

The scalar trainer backend named `jax_gpu` is not the speed path. It renders one
env row at a time, copies back to NumPy, and was slower than `cpu_oracle` in the
stock trainer. The promising path is the profile-only batched boundary:

```text
VectorMultiplayerEnv[B,2]
-> compact state
-> batched GPU browser_lines + simple_symbols render
-> row/player stacks [B,2,4,64,64]
-> scalar LightZero timesteps only at the outer boundary
```

That path is partly trainer-shaped now, but only as profile evidence. The
stock profile manager can call `train_muzero` and materialize scalar
LightZero timesteps from a batched surface, and the hybrid boundary can feed a
resident `[B,2,4,64,64]` stack into real LightZero collect-forward. It is not a
production default and not Coach launch advice.

2026-05-21 current read:

```text
Renderer work helped.
Fresh attested stock profile rows work.
The next large wall is public LightZero collect/search/output after the model
root pass, not another renderer-only rewrite.
```

2026-05-22 current read:

CPU64 did not help; it made the current search-boundary rows slower. The first
direct_ctree_gpu_latent `train_muzero` repeat did not prove a stable full-loop
speedup: stock C64/sim16/3-learner repeated at `445.19` steps/sec and direct
repeated at `438.56` steps/sec.

The direct hook reduced MCTS/search buckets, but the win was eaten by
model-output D2H/list conversion and stock per-env output assembly. After the
output-assembly fast path, the best matched no-RND profile row became:

```text
matched H100 C64/sim16/3-learner/no-RND/no-death:
  stock:              433.17 steps/sec
  direct output-fast: 566.19 steps/sec
```

With the first `rnd_meter_v0` reward-model entrypoint, the pre-hash-fix row was:

```text
matched H100 C64/sim16/3-learner/no-death:
  stock:              342.33 steps/sec
  direct output-fast: 410.55 steps/sec
```

That proved the speedup survived RND, but it also exposed a dumb side wall:
the RND diagnostic path was hashing predictor/target state inside every RND
update. After moving those hashes outside the update loop:

```text
matched H100 C64/sim16/3-learner/no-death/rnd_meter_v0:
  stock:              351.02 steps/sec
  direct output-fast: 448.52 steps/sec
```

So the current matched stock-loop profile speedup is about `1.28x-1.31x`.
This is a real profile-loop signal, but it is still profile-only and not Coach
launch advice.

2026-05-22 radical architecture read:

```text
The small patch lane is doing what small patches do: about 1.3x in the matched
profile denominator. The credible 5-10x lane is bigger: keep CurvyTron state,
observations, action masks, search roots, and replay-shaped output in compact
batches for much longer. PufferLib is now part of the external comparison set
because its speed story is native/vector env buffers, static memory, async
transfer, CUDA graph replay, and no redundant observation copies.
```

2026-05-22 compact sidecar update:

```text
The current boundary of truth is HybridCompactBatch.
It carries row/player ids, rewards, done/final/autoreset facts, to_play, and
active_root_mask alongside [B,2,4,64,64] observations and action masks.
```

2026-05-22 closed compact consumer update:

```text
The compact sidecar is now fast enough locally that it is no longer the obvious
toy-denominator wall. After switching compact RND latest-frame extraction to
slice the latest channel before normalization, B512/A16 closed compact arrays
with native actor buffer improved from about 26.5k to about 57.9k-62.8k
timesteps/sec. B2048/A16 reached about 71.6k timesteps/sec, while the
native-vector mock ceiling was about 80.4k timesteps/sec.
```

Plain read: compact RND/target sidecars can be made cheap. The remaining
5-10x problem is still the real LightZero collect/search/replay object
boundary, not another small renderer-only or CPU-count patch.

2026-05-22 vendored CTree update:

```text
The smallest array-native CTree spike is now implemented as a profile-only
vendored module with backend ctree-flat-a3. It accepts flat float32 [B] and
[B,3] backprop payloads instead of Python nested policy lists. The first exact
parity failure was a bad gate because LightZero CTree reseeds and randomly
breaks near-ties inside traverse. With vendored deterministic tie-breaking
enabled only for the parity check, flat-A3 matches vendored list CTree exactly
on local tiny rows.
```

Local no-model read:

```text
roots=1024, sim16:
  all3       ctree-list 1.01M nodes/sec -> flat-A3 2.03M (~2.02x)
  mixed_2of3 ctree-list 1.10M nodes/sec -> flat-A3 1.92M (~1.75x)
```

Plain read:

```text
Flat-A3 is a useful bounded optimization and a cleaner proof of the CTree list
ABI cost. It is not the 10x architecture because roots, traverse, output, and
Python sim control are still stock-shaped.
```

H100 no-model read:

```text
final expand-A3 flat path, roots=1024, sim16:
  all3       ctree-list 546.7k nodes/sec -> flat-A3 922.1k (~1.69x)
  mixed_2of3 ctree-list 517.3k nodes/sec -> flat-A3 858.1k (~1.66x)
```

Next rule:

```text
Flat-A3 can be wired as a profile-only train-hook option and tested in matched
full-loop profiles. It is not Coach launch advice until that full-loop gate
passes.
```

2026-05-22 contract update:

```text
The fast lane now needs an explicit CompactSearchReplayV1 contract, not a
quiet replacement for stock collect dicts. The next trusted proof is compact
root/search/replay arrays matching current target-row semantics over live and
terminal/autoreset rows.
```

See
[compact_search_replay_contract_plan_20260522.md](compact_search_replay_contract_plan_20260522.md).

The profile-only direct CTree hook now consumes that compact sidecar directly.

2026-05-22 boundary wiring update:

```text
The profile-only direct CTree boundary now validates its real compact search
output through CompactRootBatchV1 and CompactSearchResultV1 when called via
run_compact_batch(). This proves root/result legality and identity sidecars in
the real boundary path. A second profile-only helper,
run_compact_batch_with_replay_chunk(), carries replay chunk and record_index
when available and proves CompactReplayChunkV1 target rows as well. The missing
step is now measurement, not local contract wiring.
```

2026-05-22 valid denominator update:

```text
The hybrid profile has an opt-in compact replay proof flag now. With
hybrid_compact_service_replay_proof=true, search-selected actions drive the
next env step before the two-record CompactReplayChunkV1 proof is built. This
removes the fake denominator where replay targets were checked against random
next actions. Next step: Modal H100 no-RND same-denominator speed row.
```
A corrected Modal smoke passed on L4/T4 with:

```text
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
hybrid_lightzero_mcts_arrays_boundary_impl=direct_ctree_gpu_latent
```

The smoke reported `compact_row_player_sidecar_v1`, active-root telemetry,
zero illegal decoded actions, no scalar timestep materialization, and no live
run contact. This closes a wiring proof only. It is not Coach launch advice.

2026-05-22 compact replay proof update:

```text
The first closed replay proof measured the wrong hot-path shape: it copied full
observation and next_observation tensors into target rows during collection.
That collapsed H100 throughput from direct baseline `5634/4815` steps/sec
(sim8/sim16) to `987/925`.
```

The fix is `CompactReplayIndexRowsV1`: write compact replay indices and search
arrays now, materialize observation tensors only at sampler/validation edges.
After the fix:

```text
sim8:  direct baseline 5634 steps/sec, index proof 6193, proof cost 0.181s
sim16: direct baseline 4815 steps/sec, index proof 4797, proof cost 0.193s
```

Plain read: compact replay object-copy overhead is no longer the toy wall. This
does not prove a 3x search speedup. The next 5-10x candidate is still compact
batch ownership through an array-native search service or CTree replacement.

Safety refresh after the sidecar critique:

```text
CompactReplayIndexRowsV1 now rejects stale root/search identity, terminal
next rows without final-observation markers, stale root-batch sidecars, and
mismatched policy_env_id. A validation/sampler edge,
materialize_compact_target_rows_from_index_rows_v1(), rebuilds learner-shaped
target rows from index rows and matches the old compact target-row builder on
terminal/final-observation and non-prefix active-root cases.
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_policy_row_bridge.py tests/test_compact_search_replay_contract.py
uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py -k "compact_replay_index_rows or compact_search_result_v1_rejects_identity or non_prefix_active_roots"
```

2026-05-22 compact target-row update:

```text
Compact active-root search output can now become checked PolicyRowRecordV0
records and pass through the existing source-state target-row builder.
```

New profile-only bridge:

```text
HybridCompactBatch sidecar
+ selected_action[active_root]
+ visit_policy[active_root,3]
+ root_value[active_root]
-> PolicyRowRecordV0
-> build_source_state_multiplayer_target_rows_v0(...)
```

Validation:

```text
tests/test_multiplayer_source_state_target_rows.py -> 18 passed
```

The proof covers active-root ordering, P4 rows, mixed active/done rows,
binary-mask validation, illegal selected actions, illegal visit-policy mass,
fixed-opponent `to_play=-1`, and row/player sidecar alignment. It still does
not claim native LightZero `GameSegment`, stock replay integration, learner
updates, or Coach-facing speed. It proves the next compatibility edge can be
checked without scalar LightZero timestep objects in the hot sidecar path.

The combined edge now has a local proof too:

```text
HybridCompactBatch
-> real direct CTree compact profile hook
-> debug compact action/visit/value arrays
-> compact target-row adapter
-> checked source-state target rows
```

Validation:

```text
tests/test_source_state_batched_observation_boundary_profile.py
tests/test_multiplayer_source_state_target_rows.py
-> 112 passed
```

This is still an adapter proof, not speed. It means we should stop polishing
small compatibility edges unless a new correctness hole appears.

2026-05-22 GPU/parallel MCTS research update:

```text
The current blocker is not simply "MCTS is not on GPU."
The sharper blocker is the whole LightZero-compatible search/replay boundary:
Python objects, CPU/list CTree APIs, per-simulation recurrent output handling,
and replay/RND object materialization.
```

See
[gpu_parallel_mcts_research_synthesis_20260522.md](gpu_parallel_mcts_research_synthesis_20260522.md).

H100 no-model CTree rows now show CTree/list at about `0.45M-0.76M`
nodes/sec while a fake flat array update reaches about `13.9M-19.4M`
nodes/sec. That proves the ABI is expensive, but raw CTree alone is still much
faster than the train-facing wall. The next falsifier is precomputed
recurrent-output direct search: same CTree shape, synthetic resident
reward/value/policy outputs, no recurrent model call.

Durable profile tooling:

```text
scripts/run_curvytron_hybrid_observation_profile_manifest.py
```

Use that runner for hybrid boundary manifests. Raw detached Modal commands are
not enough evidence because their JSON can be lost after session compaction.

New architecture planning doc:

```text
native_vector_buffer_architecture_plan_20260522.md
```

This is the working plan for the bigger PufferLib-shaped move: compact
CurvyTron state buffers, observation/action-mask buffers, batched search
service, and replay materialization at the edge.

Implementation status:

```text
collect_search_backend=direct_ctree_gpu_latent
```

This profile-only hook now exists in the canonical launcher. It patches
`MuZeroPolicy._forward_collect` during stock `train_muzero`, keeps
collector/replay/target/learner semantics stock, and restores after the run.
It is rejected outside `mode=profile`. The first matched full-loop repeat was
flat/slower for direct before the output-fast patch, so this hook remains a
probe, not Coach launch advice. The output-fast version now has a repeatable
profile-loop win, but it still needs semantics/parity gates before any
production recommendation.

Fresh evidence:

- `opt-semantic-identity-smoke-20260521a` passes
  `--require-attestation`, so fresh speed rows can now prove what they measured.
- H100 initial inference only: about `9466` roots/sec.
- H100 collect-forward sim8 on the same B512/A16 shape: about `2304` roots/sec.
- Cheap-pool collect-forward landed on T4 and measured about `1250` roots/sec.
- H100 pure-policy collect: about `6286` roots/sec.
- H100 MCTS collect on the same profile family: about `2572` roots/sec.
- Deep H100 MCTS split: collect-forward `35.36s`, model calls `1.81s`,
  MCTS search `10.97s`, raw ctree traverse/backpropagate only `0.98s`, and
  outside-MCTS wrapper/output residual `24.40s`.

Current lane: the profile-only direct CTree arrays boundary. It calls the real
LightZero MuZero model and real CTree MCTS, then returns compact arrays without
the public `collect_mode.forward` wrapper. The next work is fixed-seed parity
against stock LightZero, plus splitting the remaining direct-path wall. This is
still not a training algorithm change and not Coach launch advice.

2026-05-21 array-ceiling result:

| row | roots/sec |
| --- | ---: |
| H100 public MCTS collect sim8 | `2572.12` |
| H100 public pure-policy collect | `6286.61` |
| H100 array ceiling `recurrent_toy` sim8 | `8681.01` |
| H100 array ceiling `policy_arrays` | `9957.97` |

Plain read:

```text
The ceiling is high enough to justify designing a real compact arrays-in /
arrays-out MCTS boundary.
```

Important caveat:

```text
The recurrent toy is not MCTS. It proves model-call pressure is not the wall.
It does not prove a semantics-preserving MCTS replacement will reach 8.7k
roots/sec. The next implementation must preserve MCTS semantics and pass
fixed-seed comparison tests before it touches training.
```

2026-05-21 follow-up: the first model-call split answered the wrong-rabbit-hole
question plainly. In the H100 B512/A16/sim8 collect-forward row, model
`initial_inference` plus `recurrent_inference` was only about `2.7s` of about
`69.8s` spent inside collect-forward. That means the next bottleneck to split
is the LightZero search boundary itself: MCTS `search`, ctree
`batch_traverse`, ctree `batch_backpropagate`, and policy output assembly. The
hybrid profile canary now has profile-only timers for those pieces. Do not run
another broad hardware/render grid until that split returns.

2026-05-20 local bridge update: the first vector-facade profile-loop slice is
implemented and tested. It keeps one batched
`SourceStateMultiplayerTrainerSurface` alive through reset/step, then
materializes row/player LightZero-shaped payloads at the outer boundary. This
is exactly the shape we need to test whether the fast direct GPU surface
survives contact with collector/RND/payload work. It is still profile-only and
does not change trainer or tournament defaults.

Current next-canary plan:
[vector_facade_next_canary.md](vector_facade_next_canary.md).

Current Amdahl experiment plan:
[next_experiment_grid.md](next_experiment_grid.md).

Payload dtype note:
[uint8_payload_design_note.md](uint8_payload_design_note.md).

Actual Coach-run speed read:
[actual_training_speed_read_20260521.md](actual_training_speed_read_20260521.md).

Current denominator ledger:
[whole_loop_denominator_ledger_20260521.md](whole_loop_denominator_ledger_20260521.md).

Current code dataflow map:
[current_code_dataflow_map_20260521.md](current_code_dataflow_map_20260521.md).

Current architecture research read:
[architecture_research_20260521.md](architecture_research_20260521.md).

Current GPU search fix ladder:
[gpu_search_fix_ladder_20260521.md](gpu_search_fix_ladder_20260521.md).

Current array-native CTree next design:
[array_native_ctree_next_design_20260522.md](array_native_ctree_next_design_20260522.md).

Current direct CTree promotion gates:
[direct_ctree_promotion_gates_20260522.md](direct_ctree_promotion_gates_20260522.md).

Next resident-batch canary plan:
[resident_chunk_canary_plan_20260521.md](resident_chunk_canary_plan_20260521.md).

Current MCTS arrays-boundary contract:
[mcts_arrays_boundary_contract_20260521.md](mcts_arrays_boundary_contract_20260521.md).

Direct CTree promotion contract:
[direct_ctree_promotion_contract_20260521.md](direct_ctree_promotion_contract_20260521.md).

Current code status: there are now three profile-only MCTS arrays-boundary modes:

- `stock_facade`: calls public LightZero `collect_mode.forward`, then decodes
  compact arrays. This proves the boundary shape.
- `direct_ctree_arrays`: calls real LightZero model inference and real CTree
  MCTS directly, then returns compact arrays without public
  `collect_mode.forward`. This is historical but still useful as a comparison.
- `direct_ctree_gpu_latent`: keeps MuZero latent tensors on GPU across
  simulations while still using LightZero CPU CTree. This is the current
  practical direct-search baseline, not a trainer default.

Current overall speed read:

```text
Trusted production path: better stock LightZero topology/hardware/settings have
put the best documented stock profiles around the ~1k env-steps/sec band.
No direct CTree, resident input, or GPU observation backend is promoted.

Profile-only path: `direct_ctree_gpu_latent` can be about 2-3x faster than the
public stock facade in the compact boundary denominator, but matched
`train_muzero` full-loop rows are only about `1.28x-1.31x`. The array-ceiling
toys show headroom, but they are not real MCTS.
```

Plain read: we have a meaningful next optimization target, but the large
numbers are not Coach launch advice yet. The only trusted training-speed
improvement so far is better use of the stock path.

2026-05-21 fresh current-telemetry refresh:

| row | roots/sec | read |
| --- | ---: | --- |
| stock facade, `host_uint8` | `2473.11` | current public LightZero compact-facade anchor |
| direct CTree, fresh `host_uint8` | `4564.03` | about `1.85x` faster than the stock facade |
| direct CTree, fresh `host_uint8_pinned` | `4113.52` | H2D was tiny, but total wall lost to search/model/observation variance |
| direct CTree, `resident_torch_reuse` | `4884.69` | stale-input ceiling only; only about `1.07x` above fresh host input |

Plain correction:

```text
Pinned/resident input is not the big remaining win in the current direct CTree
shape. The fresh row says the validated profile-only gain is direct CTree over
the stock facade, not input-copy tricks. The next wall is the remaining real
CTree search/root-prep/model-output/observation path plus validation gates.
```

Latest H100 medium rows:

| row | run id | roots/sec | read |
| --- | --- | ---: | --- |
| stock facade | `ap-HJk70PQP2iLAvA7mxxn99u` | `2419.81` | public LightZero collect path plus compact decode |
| direct CTree, old host uint8 | `ap-XEB8GF9B2Gw5V600QVtu10` | `3859.44` | first fast direct row after output-loop cleanup |
| direct CTree, current host uint8 | `ap-DoCqvAulFMhZyoAcownQmn` | `5247.95` | matched current-image row, short warmup |
| direct CTree, current pinned uint8 | `ap-APSw7b1ZSJjSSuPtGEHO3w` | `4678.23` | H2D dropped, total wall did not beat current host uint8 |
| direct CTree, resident reuse ceiling | `ap-KCtqhJDwTuLptLKd4XSv38` | `5820.96` | not semantically fresh; upper-bound only |

The first direct row only reached `2806.64` because output assembly was still a
Python loop; that bucket is now mostly gone in the common all-legal profile
shape.

Plain read: direct CTree arrays is a real profile-only speed signal. Pinned
input cuts the H2D bucket, but it is not a proven total-wall win yet. Resident
reuse shows the input-copy ceiling, but it deliberately reuses stale input and
cannot be a training mode. The next Amdahl wall is still the remaining MCTS
search/root-prep/model/output path around real CTree plus ordinary observation
cost. Do not promote this to Coach training until parity gates are explicit.

Longer repeat rows, same H100 B512/A16/sim8 shape but `60` measured steps and
`15` warmup steps:

| row | run id | roots/sec | read |
| --- | --- | ---: | --- |
| direct CTree, host uint8 | `ap-QPLEHOs3dGrcs2tlRpbMge` | `4111.80` | fresh host stack copied to GPU each measured step |
| direct CTree, pinned uint8 | `ap-5F1tMU2HiuHXDcu4O1tGkw` | `4513.15` | H2D fell from about `1.21s` to about `0.14s`; total wall improved about `1.10x` |
| direct CTree, resident reuse ceiling | `ap-wsKyodSayU2KGsTgKKpAqc` | `5537.40` | stale-input upper bound; not a training mode |

Plain correction: pinned input is now a plausible small win in stable rows, but
it does not change the main priority. The remaining wall is still search/root
prep/model/output plus ordinary observation. Newer code also reports
`input_freshness`, transfer bytes, and model-output D2H timing; the long rows
above were launched before every new accounting field landed.

Current validation status:

- local focused tests pass for stock-facade visit decoding, illegal visit-mass
  zeroing, direct CTree accounting, and real CPU LightZero searched-value/legal
  visit sanity across sim1/sim2/sim8;
- stock-facade compact decoding now carries predicted value/logit arrays when
  available, including legal-action-length logits mapped back to full action
  ids, so the facade/debug surface is closer to direct CTree arrays;
- a biased-logit real-policy canary makes action 1 the clear winner and both
  stock facade and direct CTree choose it at sim8;
- single-legal-action and masked biased-logit canaries now pass, proving the
  direct path respects forced legal actions, masked-out winners, and zero
  illegal visit mass in deterministic cases;
- checked multiplayer target rows accept compact MCTS visit distributions as
  `policy_target` and preserve root value/source metadata, so compact search
  output fits the repo-owned target-row bridge;
- exact action/visit-distribution parity is still not proven in neutral-logit
  tie-heavy rows, and native LightZero GameSegment/full trainer integration is
  still not proven, so `direct_ctree_arrays` remains profile-only.

Parity gate correction:

```text
Exact neutral visit parity is not a good ruler by itself.
```

A local diagnostic showed fixed Python/NumPy/Torch seeds do not make stock
LightZero CTree repeat the same neutral/tie-heavy visit allocation even
stock-vs-stock. The useful production contract is exact forced-case parity
(single legal action, masked preferred action, illegal mass zeroing, schema, and
target rows), clear-preference parity, then stochastic/statistical collect-row
comparison and a matched full-loop profile before any Coach-facing advice.

Current resident Torch input probe plan:
[resident_torch_input_probe_plan_20260521.md](resident_torch_input_probe_plan_20260521.md).

2026-05-20 update: a trainer-visible `renderer_backed_profile` surface canary
now exists on `SourceStateMultiplayerTrainerSurface`. It is still not stock
LightZero/full-loop integrated. It is a profile surface for measuring batched
observation costs and semantic plumbing before touching real training.

There are now two GPU policy-observation surfaces to keep separate:

- `block_704_gray64`: dense browser-line-style render at source resolution,
  then grayscale/downsample to `[64,64]`. This is the closer fidelity probe.
- `direct_gray64`: approximate learned-observation render directly in
  `[64,64]`. This is not browser-pixel parity, but it is the obvious drastic
  speed probe because it removes the dense 704-block supersampling work.

The corrected H100 B64 steps256 surface canary passed after fixing direct
simple-symbol bonuses and draw order: `direct_gray64` surface step median was
`0.0339s`, versus `0.144s` for `block_704_gray64` and `0.237s` for CPU
dirty-cache. Device render was `0.00973s`, versus `0.123s` for
`block_704_gray64`. A separate H100 adversarial two-view canary checked
`direct_gray64 + simple_symbols` against the CPU-direct oracle with exact
parity. This is a major local observation-surface win, but it still does not
include stock LightZero search, replay, learner updates, RND training,
checkpoints, eval, GIFs, or tournaments.

The matching real full-loop RND smoke also passed separately. It used the stock
trainer with CPU-oracle observations and `rnd_meter_v0`: `16,384` env steps,
`512` MCTS searches, `12` learner calls, about `457 steps/s`, and H100 max GPU
utilization about `17%`. Plain read: the full loop is not currently limited by
GPU compute saturation. Policy/MCTS and RND meter timers are large, and
CPU-oracle observation/update-stack remains a meaningful worker-side cost.

The matched no-RND H100 C32/sim8 control also passed at about `426 steps/s`.
That one-row pair is noisy and should not be used to claim RND is faster or
free. It does say the current full-loop profile band is roughly the same with
and without meter-only RND, and the next optimization question is still how to
preserve batched observations inside a real stock-loop path.

Profile-only RND cadence rows now make the separation clearer. At the
direct-surface boundary, RND can dominate everything if trained aggressively:
CPU update100 spent about `7.43s` in predictor training, CUDA update100 spent
about `2.39s`, CUDA update10 about `0.258s`, and CUDA update1 about `0.0267s`.
This is intentionally recorded as an RND/training knob, not a render speed
claim.

## What Could Go Wrong With Batching

- Wrong player view: a row gets player 1's view when player 0 should act.
- Wrong row order: GPU returns view-major frames but the trainer consumes
  row-major env/player order.
- Wrong stack order: newest frame goes to the wrong channel, or reset leaks old
  history.
- Wrong terminal observation: `final_observation` is copied after autoreset
  instead of before.
- Hidden CPU fallback: a "GPU" timing row quietly uses the CPU oracle.
- Host overhead erases the win: pack/copy/readback/timestep/IPC costs become
  larger than saved render time.
- Scalarization erases the win: LightZero still sees many one-row envs, so the
  GPU batch never forms.
- RND axis mistake: the player axis is accidentally treated as LightZero's
  unroll axis in production replay.
- RND tiny-batch failure: a smoke with learner/RND batch size `1` can fail in
  training layers that require more than one sample.
- Tournament/checkpoint drift: a checkpoint trained with a new backend enters
  rating without matching metadata and loader support.

Tiny one-gray float32 edge differences are acceptable for learned observations.
Shape/order/perspective/reset/RND mistakes are not acceptable.

## Current Gates

1. Keep production defaults on `cpu_oracle`.
2. Keep scalar `jax_gpu` out of recommendations except as a diagnostic canary.
3. Exact pixel parity is not required for the learned observation. The real
   gate is semantic: no missing objects, no wrong row/player view, no wrong
   stack/reset/final-observation order, no hidden CPU fallback, and correct RND
   latest-frame extraction.
4. Use `float64 + exact` as a debug/reference guard only.
5. Use `float32 + tolerant` as the aggressive speed candidate gate.
6. RND meter mode passed CPU-oracle and L4/T4 smokes; compact step telemetry is
   now fixed. RND rows are still canaries, not learning proof, and
   positive-weight work remains blocked on intrinsic normalization.
7. Build a vector facade before trying to expose a trainer backend flag.
8. Only claim full-loop speed after render, env step, stack, reset,
   final-observation, timestep construction, IPC, search, replay, learner, and
   RND costs are measured.

## Next Optimization Priority

Do not spend the next phase on scalar GPU rendering, body-circle lanes, or
exact one-luma parity. Renderer work helped, but the latest real-consumer split
puts the active wall in public LightZero collect/MCTS branch
setup/conversion/result handling.

The priority order is now:

1. **Compact MCTS boundary design:** design a real arrays-in / arrays-out MCTS
   boundary that preserves stock LightZero/MuZero semantics.
2. **Parity gates before speed claims:** compare against stock LightZero on
   legal masks, `to_play`, root noise, temperature, support transforms, visit
   counts, decoded actions, replay target fields, and seeded randomness.
3. **Resident input/H2D split:** test whether the current host-stack -> Torch
   H2D copy can be removed with resident Torch input or a pinned/dtype split.
4. **Matched stock-loop reconnection:** only after the boundary/parity result
   is clear, rerun stock profile A/B/C rows: CPU oracle, batched profile
   manager, and zero observation, with no-RND and `rnd_meter_v0` separated.
5. **Renderer cleanup only when a row re-shows render dominance:** keep
   renderer docs as useful evidence, but do not restart renderer-only work as
   the main lane unless fresh Amdahl numbers justify it.

Historical queue, mostly completed or superseded:

1. **Mock collector profile:** measure vector env step, batched render seam,
   stack update, scalar LightZero-shaped row materialization, pickle/payload
   work, and optional RND latest-frame collect/train/estimate. This is the
   missing bridge between renderer-only speed and full-loop speed. First H100
   B64/S1024 payload row says scalar row materialization and pickle are tiny
   compared with device render.
2. **GPU renderer at that boundary:** replace the mock collector's CPU renderer
   with the existing batched GPU candidate and compare CPU oracle versus GPU on
   the same rows. Current B64/S1024 result is about `0.264s` per
   env-step+observation+payload batch versus about `0.743s` for the full CPU
   reference render+stack, but the GPU bucket is still render-dominated.
3. **Active trail prefix before trainer wiring:** the first H100 trail-slot
   ladder says B64/S128 is about `0.045s` and B64/S1024 is about `0.261s` at
   the mock-collector boundary. That is a `~5.8x` candidate-boundary difference,
   but the old ladder also changed env body capacity. The profiler now needs to
   report active trail prefix, then split env capacity from render width if the
   active prefix is much smaller than capacity.
4. **Full-loop A/B only after the mock wins:** rerun the H100 C512/sim4 stock
   profile shape with the safest backend available. If the mock boundary does
   not win, full trainer wiring is likely wasted motion.
5. **RND normalization remains separate:** meter mode can be profiled with this
   boundary now, but positive RND reward is still blocked on running/global
   intrinsic normalization and cap.

## Historical Renderer/Topology Speed Read

This section is useful background, but it is superseded for current priority by
the 2026-05-21 collect/MCTS split and array-ceiling result above. Keep it as
observation-boundary and topology evidence, not as the next implementation
lane.

The newest stock-path profiles say subprocess collection is the immediate
practical win. With the current CPU-oracle observation surface, subprocess C32
was about `4.2x` faster than base C32, and an earlier subprocess C64 row was
about `5.4x` faster than base C64. Within sampled env timing, stack update is
still the largest single per-step bucket; tiny pickle/copy work is not the
largest issue.

Fresh subprocess collector/sim sweep strengthens that read. On L4/T4 C64,
sim2 reached about `601 steps/s`, sim4 about `557 steps/s`, and sim8 about
`452 steps/s`. Collector scaling still helped through C64, while higher MCTS
simulation count clearly reduced throughput. Learner/replay was not the wall in
these rows.

H100 search-heavy rows show useful but bounded extra headroom. H100 C256/sim8
hit about `830 steps/s`, while H100 C256/sim16 dropped to about `554 steps/s`.
Against matched L4/T4 high-collector rows, H100 was about `1.6x-1.8x` faster for
C128 and C256/sim8, but only about `1.1x` faster for C256/sim16. So bigger GPU
can help search-heavy rows, but sim16 and very high collector counts also expose
orchestration/search interaction; it is not a pure GPU problem.

Latest H100 topology rows put the current best tested throughput at about
`885 steps/s` for C256/sim2, with C256/sim4 essentially tied at about
`877 steps/s`. Since sim4 is almost as fast as sim2 and likely a better search
quality floor, C256/sim4 looked like the first aggressive setting. Follow-up
stress rows moved the top point to H100 C512/sim4 at about `1061 steps/s`.
C512/sim2 was slower at about `952 steps/s`, C768/sim4 nearly tied but did not
improve at about `1054 steps/s`, and C1024/sim4 regressed to about
`853 steps/s`. Current read: H100 C512/sim4 is the best tested
throughput/search compromise; more collectors past C512 do not help this
profile, and sim8+ costs sharply more.

Inside the batched GPU observation boundary, the newest Amdahl read is simpler:
device render dominates. At B64/S1024, mock-collector total is about `0.261s`
and device render is about `0.244s`, so render is roughly `93%` of that
candidate boundary. A fixed trail-slot ladder shows S128/S256/S512/S1024 at
about `0.045s`/`0.075s`/`0.137s`/`0.261s`, which makes active-prefix/trail-slot
reduction the next renderer optimization to test. This does not yet prove a
full-loop win because the boundary still has to beat the optimized CPU
dirty-cache path and then the H100 C512/sim4 full-loop row.

The first active-prefix row sharpened this: B64/S1024 used a median of `15`
active trail slots and p95 of `22`, meaning the renderer was paying for mostly
inactive tail slots in that profile. The boundary profiler now supports
profile-only decoupling of env `body_capacity` from render `trail_slots` so the
next rows can test smaller render widths without lowering env capacity.

Historical observation-boundary result, now superseded as the main optimizer
priority by the 2026-05-21 direct CTree / collect-search findings:

That decoupled test is now the clearest renderer result. With env
`body_capacity=1024`, render S16 failed parity by step 5, S32 passed all 8
measured checks at about `0.028s` mock-collector median, S64 also passed at
about `0.035s`, and S1024 was about `0.261s`. So the current boundary headroom
is real: S32 is about `9x` faster than S1024 for this short trajectory. The
safe profile-candidate shape should be dynamic render width, not fixed S32 forever,
because longer games will eventually need more active trail slots.

The first dynamic row passed the same checks: env `body_capacity=1024`, max
render `1024`, min render `32`, selected render width `32`, no truncation, and
mock-collector median about `0.0256s`. That is about `10x` faster than the
S1024 candidate boundary. This is a boundary result, not yet a trainer result;
the next Amdahl question is how much of the real H100 C512/sim4 stock loop can
actually be replaced by this boundary.

Longer rows keep the dynamic story intact. At 64 steps, fixed S1024 was about
`0.260s` while dynamic was about `0.046s`, a `~5.7x` boundary win. At 128
steps, dynamic selected median S256 and p95 S512, with no truncation, and landed
at about `0.075s`. So active-prefix rendering degrades gracefully as trail count
grows; it is not just a short-game trick.

Historical next-step note from the observation phase: trainer-visible batching
was the next question then. Current active work has moved one level inward to
the direct CTree / LightZero collect-search boundary. Stock LightZero currently creates many scalar envs, and the old scalar
`jax_gpu` backend renders one env at a time, so it cannot preserve the GPU
batch. The profiler now has the first profile-only vector facade shape:

```text
VectorMultiplayerEnv[B,2]
-> dynamic batched GPU render
-> [B,2,4,64,64] stacks
-> LightZero-shaped scalar timesteps
-> optional RND meter
```

The trainer-visible surface canary passes locally and on Modal. At B64/steps64
it measured surface step median `0.056s`, renderer median `0.040s`, and device
render median `0.032s`. At B64/steps256 it measured surface step median
`0.144s`, renderer median `0.130s`, and device render median `0.123s`, with
dynamic render width S512 and no truncation. Plain read: the surface wrapper is
not the new wall; for long observation rows, device render is still the wall
inside this canary. This is still not a full training-loop result because it
does not include stock LightZero search, replay, learner, checkpointing, eval,
or tournaments.

The matched CPU dirty-cache surface control at B64/steps256 measured `0.237s`.
So the trainer-visible GPU observation canary is about `1.65x` faster than the
current optimized CPU visual stack on this long no-death surface row. That is a
real win, but it is not a 10x full-training claim. Amdahl now says the next
local observation target is the GPU device render itself; host transfer,
stacking, scalarization, and pickle are small in the matched row.

Only after this passes reset/final-observation/RND gates in the full stock loop
shape should we try a real stock-loop A/B against the current H100 C512/sim4
CPU-oracle control.

Seeded duplicate profile rows matched workload counts exactly for RND-off
profiles. RND rows are currently reward-model canaries, not throughput proof,
but compact output now has a profile-only MCTS-root fallback so these rows no
longer print a fake zero-step rate. Check `env_steps_collected_source` before
comparing rates.

## Linked Notes

- [Task Board](task_board.md)
- [Orchestration](orchestration.md)
- [World Model](world_model.md)
- [Experiment Log](experiment_log.md)
- [Host Overhead Map](host_overhead_map.md)
- [Search Boundary Escape Plan](search_boundary_escape_plan_20260521.md)
- [Search Boundary Fix Plan](search_boundary_fix_plan_20260522.md)
- [Radical Search Architecture Reorientation](radical_search_architecture_reorientation_20260522.md)

## 2026-05-22 Reorientation

Plain answer:

```text
We are fixing the stock LightZero collect/search boundary plus one RND
diagnostic wall. We are not chasing CPU count.
```

Fresh profile-loop facts:

- `direct_ctree_gpu_latent` plus the all-actions-legal output fast path is the
  best current profile probe, but it is still profile-only.
- The packed model-output CPU transfer was safe and validated, but it was a
  small cleanup, not a strategy change: the direct no-RND row measured about
  `480` steps/sec, close to the prior listify row at `472` and below the best
  repeat at `566`.
- The RND hash fix was a real independent win. Moving predictor/target hashes
  outside the 100-update RND loop dropped `rnd_train_with_data` from about
  `3.5s` to `0.6s` and `rnd_state_hash` from about `3.0s` to `0.14s`.
- With RND meter enabled after that fix, matched H100 C64/sim16/3-learner rows
  measured stock `351.02` steps/sec and direct `448.52` steps/sec.

Current Amdahl read:

```text
The highest-value remaining wall is still the per-simulation CTree/search
boundary: GPU model outputs are converted into CPU arrays/lists for CTree, then
LightZero builds object-shaped per-env outputs. Output assembly and RND hashing
are no longer the obvious walls. More CPU cores were a negative falsifier, not
a lane.
```

Why we are blocked from a 10x full-loop claim:

```text
The current train_muzero profile denominator still includes stock collection,
environment stepping, replay, learner, reward-model hooks, and Python/object
control around MCTS. Deleting only one small bucket cannot yield 10x. A 10x
attempt needs a larger topology change such as an array-native CTree API or a
compiled/fused batched search path, and it must preserve MuZero semantics.
```

## 2026-05-22 Latest Optimizer Read

The fixed-shape dense compile spike was tested on H100 and failed the sim16
gate:

```text
dense_torch_mcts_compile_spike sim16: 4872.70 roots/sec
direct_ctree_gpu_latent sim16:        6153.95 roots/sec
```

So the current line is:

```text
The 1.3x train-profile win is real, but it is the size of a boundary cleanup.
The next 5-10x attempt must change who owns the batched search boundary.
```

Near-term files:

- [Radical Search Architecture Reorientation](radical_search_architecture_reorientation_20260522.md)
- [World Model](world_model.md)
- [Task Board](task_board.md)
- [Dense Torch Compile Spike Feasibility](dense_torch_compile_spike_feasibility_20260522.md)
- [External Big-Move Architecture Critique](subagent_external_big_moves_architecture_20260522.md)
- [Current Hot Path Bottleneck Map](current_hot_path_bottleneck_map_20260522.md)

Current active next question:

```text
Can the compact sidecar produce a real matched full-loop speedup once search,
RND input, and target-row compatibility are all checked?
```

Current next gates:

1. Keep direct CTree compact parity/statistical gates green.
2. Build the closed compact batch consumer falsifier:
   compact batch -> real compact search output -> RND latest-frame input ->
   compact target rows, with scalar objects only at validation edges.
3. Compare that against the current direct full-loop/profile denominator. If
   it cannot plausibly beat current direct by about `3x`, stop pretending this
   small-patch lane is the 5-10x answer and move to a search-service/native
   vector-buffer prototype.
4. Only after matched full-loop A/B should we make Coach advice.

RND sidecar status:

```text
Compact [B,P,4,64,64] observations can now feed the existing RND latest-frame
adapter with scalar timestep materialization off.
```

This is only an input contract proof. RND cadence, positive-bonus
normalization, and full training throughput remain separate decisions.

## 2026-05-22 GPU MCTS Research Follow-Up

The new parallel research wave agrees with the current profile evidence:

```text
The blocker is not merely "MCTS is not on GPU."
The blocker is the LightZero-compatible boundary around search and replay:
CPU/list CTree payloads, Python simulation control, GPU recurrent outputs
copied/listified each simulation, scalar env objects, replay rows, and RND
materialization.
```

Important sidecar reports:

- [GPU MCTS parallel architecture follow-up](subagent_gpu_mcts_parallel_architecture_followup_20260522.md)
- [Current boundary code audit](subagent_current_boundary_code_audit_20260522.md)
- [Search/replay architecture plan](subagent_search_replay_architecture_plan_20260522.md)
- [Precomputed recurrent implementation critique](subagent_precomputed_recurrent_impl_critique_20260522.md)
- [Compact search/replay service contract](compact_search_replay_service_contract_20260522.md)

The recommended architecture target is now explicit:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> batched search service
-> CompactSearchResultV1
-> CompactReplayChunkV1
```

Use the current `direct_ctree_gpu_latent` hook as a control and validation
adapter, not as the final architecture. Use MCTX/JAX as a separate scratch
reference for accelerator-native search, not as an immediate trainer rewrite.

The first precomputed recurrent-output falsifier is implemented as profile-only
mode:

```text
direct_ctree_gpu_latent_precomputed_recurrent
```

It now reports separate logical and actual model-eval counts, plus synthetic
recurrent eval count and recurrent-output listification time. This matters
because the mode skips real recurrent inference by design; it is a wall-split
falsifier, not a valid training mode.

Small H100 smoke:

```text
B64/A8/sim8, 16 measured, 4 warmup:
  direct_ctree_gpu_latent:                         2357.77 roots/sec
  direct_ctree_gpu_latent_precomputed_recurrent:  3745.00 roots/sec
```

Large H100 row:

```text
B512/A16/sim16, 60 measured, 15 warmup:
  direct_ctree_gpu_latent:                         4920.30 roots/sec
  direct_ctree_gpu_latent_precomputed_recurrent:  6771.37 roots/sec
```

Plain read:

```text
Recurrent inference/output handling is a real bucket, but removing it only
gave about 1.59x on the small profile row and about 1.38x on the larger stable
row. That is enough to keep the falsifier, but not enough to make recurrent
inference the main 10x lane. The next serious implementation must be compact
batched search/replay ownership, not another wrapper shave.
```

## 2026-05-22 Flat-A3 CTree Spike

The smallest conservative CTree-boundary spike is now wired into the profile
launcher:

```text
collect_search_backend=direct_ctree_gpu_latent
collect_search_ctree_backend=flat_a3
```

It is profile-only and not Coach advice. It preserves the stock
`train_muzero` shell, but swaps the CTree backprop payload from nested Python
lists into fixed `A=3` arrays inside a vendored LightZero CTree copy.

Current evidence:

```text
Local no-model parity: deterministic vendored-list vs flat-A3 exact match.
H100 no-model speed: about 1.66x-1.69x over vendored list CTree.
Train-facing smoke: passed after fixing Modal package-path preference.
Matched full-loop C64/sim16/3-learner A/B:
  direct LightZero CTree 516.55 steps/sec
  flat-A3 CTree          509.69 steps/sec
```

Plain read:

```text
Flat-A3 is a useful validated falsifier, not a launch-speed win. It removes
one list-shaped CTree backprop boundary, but the full loop is still dominated
by env/render/stack, recurrent inference, model-output D2H, root prepare,
collector fanout, and learner/replay overhead.
```

Important guardrails:

```text
- The profile grid builder now emits both search backend flags.
- Compact profile output includes observed runtime proof for the chosen CTree.
- The summarizer rejects flat-A3 rows that only echo the command.
- The Cython build is isolated to flat-A3 CPU40 optimizer images, so normal
  stock/live launches still use the ordinary image.
```

## 2026-05-22 Real Compact MCTX Gate

The latest useful result is no longer renderer-only and no longer synthetic
visual roots:

```text
HybridCompactBatch real [B,2,4,64,64] visual observations
-> CompactRootBatchV1
-> MCTX/JAX search
-> CompactSearchResultV1
-> selected actions step compact env once
-> CompactReplayIndexRowsV1
```

H100 B512/P2, GPU persistent renderer, 8 steady runs:

| row | fresh-boundary roots/sec | resident roots/sec | replay-index rows |
| --- | ---: | ---: | ---: |
| sim16/h64/v8 | `124,090` | `167,516` | `1024` |
| sim32/h64/v8 | `51,454` | `65,228` | `1024` |

Plain read:

```text
This validates the next optimizer direction: device-resident compact search has
10x-class search-boundary headroom on real compact CurvyTron visual roots.

It is still profile-only. It does not call train_muzero, does not prove learning,
and uses a toy JAX model rather than the current LightZero PyTorch model.
```

Next proof gate:

```text
Current-model realism plus compact learner/RND/replay integration. If the
PyTorch/JAX or model-port boundary erases the margin, this becomes a research
lane rather than Coach-facing launch advice. If the margin survives, this is the
real 5-10x architecture path.
```

2026-05-23i update:

```text
The toy compact MCTX row above has been superseded by real immutable-checkpoint
MCTX shadow profiling.

Current headline row:
  H100 B1024/A16/sim8 scalar-off
  real-checkpoint MCTX: 19.3k steps/sec
  direct CTree:          8.8k steps/sec
  speedup:               2.20x

Still profile-only:
  MCTX uses Gumbel MuZero semantics, not LightZero CTree.
  Checkpoint parity is close-not-exact.
  No stock `train_muzero` backend has changed.
```

Important Amdahl update:

```text
The MCTX numbers above are search-boundary numbers. The same rows also measured
the next compact env step plus replay-index edge, and that edge is hundreds of
milliseconds at B512/B1024. So the next optimizer target is not another
search-only row. It is a repeated closed compact loop with env/observation,
search, replay-index, RND/target input, and learner-facing samples in one
denominator.
```

Latest pressure wave:

| row | fresh-boundary roots/sec | rough search+replay roots/sec |
| --- | ---: | ---: |
| B512 sim16 h64/v8 | `124,090` | `4,227` |
| B512 sim32 h64/v8 | `51,454` | `3,653` |
| B512 sim16 h128/v16 | `105,399` | `2,910` |
| B512 sim32 h128/v16 | `61,221` | `2,977` |
| B1024 sim16 h64/v8 | `168,630` | `3,813` |
| B1024 sim32 h64/v8 | `96,336` | `4,759` |

The rough search+replay number is not production throughput. It is a warning:
once search gets fast, env/observation/replay synchronization becomes the next
Amdahl wall.
