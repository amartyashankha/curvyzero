# Experiment Log

Date: 2026-05-20

## 2026-05-23d MCTX Compact Slab Smoke

Purpose:

```text
Prove MCTX/JAX can run behind CompactSearchServiceV1 inside the current compact
slab loop without touching train_muzero or live Coach runs.
```

Tiny H100 smoke:

```text
batch=4
actors=2
simulations=2
hidden_dim=16
visual_channels=4
steps=2
warmup=1
materialize_scalar_timestep=false
```

Result:

```text
ok=true
jax backend=gpu
compact_rollout_slab_calls=2
compact_rollout_slab_total_roots=16
compact_rollout_slab_committed_index_row_count=16
compact_rollout_slab_search_impl=mctx_compact_search_service_profile_only_v0
calls_train_muzero=false
touches_live_runs=false
trainer_defaults_changed=false
```

Read:

```text
Wiring works. This is not a throughput claim because the shape is tiny and most
wall time is warmup/render setup. Next row must be the H100 B512/B1024 sim16/32
comparator against direct_ctree_gpu_latent.
```

## 2026-05-23d MCTX Compact Slab H100 Comparator

Status: profile-only. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, or Modal volume was touched.

Artifact:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-mctx-compact-slab-h100-20260523d/
```

Rows:

| row | shape | MCTX steps/sec | direct CTree baseline | speedup |
| --- | --- | ---: | ---: | ---: |
| `001` | B512/A16/sim16 | `16,250` | `6,522` | `2.49x` |
| `002` | B512/A16/sim32 | `14,306` | `4,177` | `3.43x` |
| `003` | B1024/A16/sim16 | `20,557` | `6,992` | `2.94x` |
| `004` | B1024/A16/sim32 | `16,255` | `5,314` | `3.06x` |

Telemetry sanity:

```text
compact_rollout_slab_search_impl=mctx_compact_search_service_profile_only_v0
compact_rollout_slab_committed_index_rows matches total roots
python_rows_materialized=0
rnd_materialized_rows=0
calls_train_muzero=false
touches_live_runs=false
promotion_eligible=false
```

Plain read:

```text
Keep this lane. It is the first same-denominator signal that replacing the
LightZero CTree/list/control search body with a compiled/device search body can
produce a multi-x speed gain in the optimizer profile shell.

Do not call it Coach-ready. The model is a toy JAX model and the search is
MCTX/Gumbel MuZero, not the current LightZero PyTorch + CTree training stack.
```

Validation added after this result:

```text
ruff clean
focused tests: 13 passed

Covered:
  stronger MCTX profile-only labels
  no-legal-action rejection before JAX/MCTX import
  inactive-root rejection in fixed-shape mode before JAX/MCTX import
  raw_visit_counts cannot assign mass to illegal actions
  MCTX profile telemetry promotes into compact slab summary fields
```

Sidecar critique:

```text
The speed signal is real enough to continue.
The exact multiplier needs a matched direct-CTree rerun with the same row
length/warmup/code state.
The bridge to training starts with PyTorch-to-JAX model parity, not a Coach run.
```

## 2026-05-23g LightZero JAX Shadow Model Parity

Status: profile-only. No live Coach training run, checkpoint write, eval, GIF,
tournament artifact, or trainer default was touched.

Fresh-model Modal smoke:

```text
app run: ap-mxbORR6cvPQwKD70Vyh7MQ
compute: L4
batch_size: 2
seed: 17
LightZero: 0.2.0
torch: 2.8.0
jax: 0.7.0
jax backend: gpu
ok: true
```

Model surface:

```text
lzero.model.muzero_model.MuZeroModel
observation: [B,4,64,64]
latent: [B,64,8,8]
action_space_size: 3
reward_support_size: 3
value_support_size: 3
state_dict keys: 175
required inference keys consumed: 123
ignored keys: 52
```

Why the smoke rewrites fresh heads:

```text
LightZero zero-initializes the final reward/value/policy linear heads. A fresh
model with untouched heads can make policy/value/reward parity look perfect
even if hidden tensors drift. The Modal smoke rewrites only those fresh local
heads so the output comparison is meaningful.
```

Numeric result:

```text
All comparisons passed with GPU tolerance atol=5e-4, rtol=5e-4.
Largest latent max_abs was about 2e-4.
Policy/value/reward max_abs values were about 1e-6 to 1e-5.
```

Failed-but-useful probe:

```text
An old immutable checkpoint ref from docs was tried, but it no longer exists in
the current runs volume. This means no checkpoint parity claim exists yet.
The wrapper now reports missing checkpoint refs as JSON instead of a raw stack.
```

Read:

```text
The raw model bridge is real enough to keep moving. Next gate is a current
immutable checkpoint parity run. Only after that should the real JAX shadow
model be plugged into the profile-only MCTX compact search service.
```

## 2026-05-23i Real-Checkpoint MCTX Shadow Bridge

Status: profile-only. These rows read one immutable checkpoint from the runs
volume. They did not write checkpoints, evals, GIFs, tournaments, or live Coach
runs.

Checkpoint:

```text
training/lightzero-curvytron-visual-survival/.../iteration_260000.pth.tar
sha256: a9bcbc7212995b967d75d9ff94b189039bf0111b9d30daeb2fb29edebe402b5b
```

Local validation:

```text
ruff clean
focused tests: 10 passed
```

Modal validation:

```text
tiny L4 smoke:
  ok=true
  jax backend=gpu
  model_backend=lightzero_jax_shadow_model
  shadow coverage ok=true
  required inference keys consumed=123
  calls_train_muzero=false
```

Matched H100 profile rows:

| row | shape | scalar rows | steps/sec | direct CTree baseline | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| real MCTX shadow | B64/A4/sim8, 80/20 | on | `3,817` | `2,813` | `1.36x` |
| real MCTX shadow | B512/A8/sim8, 40/10 | on | `10,037` | `4,233` | `2.37x` |
| real MCTX shadow | B512/A8/sim8, 40/10 | off | `14,257` | `8,999` | `1.58x` |
| real MCTX shadow | B1024/A16/sim8, 30/10 | off | `19,334` | `8,792` | `2.20x` |

Plain read:

```text
This is a real profile-only speed signal using the real checkpoint weights.
It is not Coach-ready.

The B1024 scalar-off row is the cleanest current optimizer row:
  19.3k profile roots/sec for MCTX
  8.8k profile roots/sec for direct CTree

The scalar-row rows show a second wall:
  avoiding scalar LightZero timestep materialization can be as important as
  replacing the search backend.
```

Caveat:

```text
Per-bucket times can move when GPU work synchronizes late. Trust measured
wall-clock speed first. Use bucket timings to pick the next experiment, not as
perfect exclusive accounting.
```

## 2026-05-23j Same-Root MCTX vs Direct CTree Comparator Smoke

Status: profile-only. This reads one immutable checkpoint and runs two search
services on the same compact roots. It does not call `train_muzero`, write
checkpoints, run eval, create GIFs, touch tournaments, or mutate live Coach
runs.

Shape:

```text
H100
B2, actor1
steps2, warmup1
sim2
materialize_scalar_timestep=false
primary search: real-checkpoint MCTX/JAX shadow model
reference search: same-checkpoint direct_ctree_gpu_latent
```

Result:

```text
ok=true
compact_search_comparator_identity_match=true
materialized_timestep_count=0
shadow coverage ok=true
required inference keys consumed=123
```

Comparison:

| metric | value |
| --- | ---: |
| selected action matches | `2 / 4` |
| selected action match fraction | `0.50` |
| primary action checksum | `23` |
| reference action checksum | `19` |
| visit L1 mean | `1.4623` |
| visit L1 max | `1.9629` |
| root value abs diff mean | `57.42` |
| root value abs diff max | `65.06` |

Plain read:

```text
The same-root comparator works and catches a real semantic warning.

Identity matched, so this is not a root-ordering bug.
The value/visit/action deltas are too large to ignore.
The next row is a larger sim8 comparator before any MCTX promotion claim.
```

Larger sim8 follow-up rows:

| reference | app | selected-action match | visit L1 mean | root value abs diff mean | steps/sec |
| --- | --- | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` | `ap-2OQLjJ7OVJyaCtDVSXEeRH` | `44/128 = 0.3438` | `1.2799` | `15.97` | `2033` |
| `direct_ctree_arrays` | `ap-OG7nd86Yiv118SCHHTWrBv` | `48/128 = 0.3750` | `1.2877` | `15.97` | `2261` |

Read:

```text
The sim8 mismatch is real and repeats against both direct CTree references.
This points away from a direct-adapter bug. Next diagnostic is pre-search model
parity in the same comparator row: predicted root value and policy logits.
```

Pre-search model parity diagnostic:

```text
app: ap-9v6Qj9C1sptGIKx7chEG19
shape: B2/A1/steps2/warmup1/sim8
identity_match=true
predicted policy logits abs diff mean=0.0000084, max=0.0000203
predicted value abs diff mean=0.0090, max=0.0219
searched root value abs diff mean=17.67
visit L1 mean=1.34
selected action match=0.50
```

Read:

```text
The MCTX shadow model and LightZero direct CTree model agree before search.
The remaining mismatch is search-side: Gumbel MuZero semantics, backup rules,
or root-value summary definition. Keep MCTX as a fast profile-only architecture
lane; do not call it a LightZero CTree drop-in.
```

## 2026-05-23 Compact Torch Search-Service H100 Smoke

Status: profile-only. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, or Modal volume was touched.

Shape:

```text
H100, B512/A16, sim16, 60 measured steps, 15 warmup steps,
compact replay proof on, host_uint8, scalar timestep materialization off.
```

Rows:

| row | steps/sec | probe sec | model sec | search/tree sec | read |
| --- | ---: | ---: | ---: | ---: | --- |
| `direct_ctree_gpu_latent` | `4,965.96` | `5.702` | `1.283` | `3.569` | direct LightZero CTree comparator |
| `service_tax_probe` | `5,853.08` | `2.857` | `1.448` | `0.213` | ceiling/falsifier, not MCTS |
| `compact_torch_search_service` | `5,139.55` | `5.867` | `0.000` | `5.867` | first remote service row before timing split |
| `compact_torch_search_service` timing split | `5,575.16` | `5.098` | `0.271` | `4.250` | same lane with phase-honest telemetry |
| `direct_ctree_gpu_latent` no-noise | `3,955.36` | `6.944` | `1.528` | `4.404` | root noise forced to `0.0` |
| `compact_torch_search_service` no-noise | `5,704.22` | `5.077` | `0.271` | `4.235` | root noise forced to `0.0` |

Read:

```text
The compact Torch service boundary is real: it runs remotely, calls one compact
service pass, and passes compact replay proof. It is not the big win yet. The
timing split says the wall is the eager Python/Torch tree plus recurrent loop,
not input transfer or initial inference.

Do not recommend this to Coach. Keep it as an optimizer lane. The next speed
attempt should be a genuinely compiled/fused fixed-shape search body or an
array-native MCTX/JAX-style comparator behind the same CompactSearchResultV1
contract.
```

Artifacts:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-direct-20260523
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-tax-20260523
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-torch-20260523
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-torch-service-timing-split-20260523
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-no-noise-direct-20260523
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-no-noise-torch-20260523
```

## 2026-05-23 Fresh H100 Batch Scaling Check

Status: profile-only. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, or Modal volume was touched.

Shape:

```text
H100, P2, body4096, hidden64, max_depth16, loop96,
native actor buffer, resident GPU stack, replay-index on,
explicit resident-stack sync off, root observation copy off,
refresh-on compact visual path.
```

Rows:

| batch | sims | run | roots/sec | total sec | env frac | search frac |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 512 | 16 | `ap-OUzGwd38pR4ajLQTefYMG9` | `29,068` | `3.382` | `71.8%` | `18.7%` |
| 512 | 32 | `ap-J26GapdNEbmEuGn8g8f2du` | `24,548` | `4.005` | `57.2%` | `35.8%` |
| 1024 | 16 | `ap-R52JRpKZc9fkOxBBzQ6u6l` | `53,899` | `3.648` | `75.2%` | `16.9%` |
| 1024 | 32 | `ap-FbNK1dC03hCQwymagb4vzY` | `36,415` | `5.399` | `65.0%` | `28.6%` |
| 2048 | 16 | `ap-mOO2KI2p1HlPuNPZfTgbvo` | `57,399` | `6.851` | `83.6%` | `10.5%` |
| 2048 | 32 | `ap-MjcaCJ1yNITrFLapA6JdqH` | `40,346` | `9.746` | `76.6%` | `18.0%` |

Important read:

```text
Batch size helps, but not magically. B2048 beats B1024 only modestly on this
shape, while production-to-compact, delta pack, H2D/update, public packaging,
and game mechanics all grow. B1024 looks like a good current profiling
denominator; B2048 is useful as a stress row, not an automatic Coach default.
```

Next patch implied by this row set:

```text
Attack redundant compact visual state conversion before another scaling wave.
The measured production-to-compact leaf is not the whole wall, but it is a
clean removable tax and it gets worse at B2048.
```

## 2026-05-23 Persistent Compact Render-State Buffer A/B

Status: profile-only. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, or Modal volume was touched.

Patch:

```text
Added an opt-in `persistent_compact_render_state_buffer` path. The hybrid
manager can write compact persistent-renderer arrays directly and the renderer
detects that marked compact state, validates it, and reports
`production_to_compact_sec=0.0`.
```

Local validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_compact_search_replay_contract.py
=> All checks passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py
=> 165 passed
```

Fresh no-borrow A/B, H100 B1024/P2/body4096/loop96/resident stack:

| row | sims | run | roots/sec | total sec | prod->compact | actor render write | renderer H2D | delta pack | persistent update |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| copied production state | 16 | `ap-XXFFhGnrZ8Q9oXVSJmKNnX` | `26,281` | `7.481` | `0.664s` | `1.250s` | `0.824s` | `0.791s` | `0.464s` |
| compact state buffer | 16 | `ap-eV3MlDjXu9WuHxUmT3twTP` | `35,843` | `5.485` | `0.000s` | `2.359s` | `0.353s` | `0.473s` | `0.294s` |
| copied production state | 32 | `ap-S5pxMQT3Po3PFGPKoO5ZvN` | `29,541` | `6.655` | `0.517s` | `1.078s` | `0.420s` | `0.591s` | `0.638s` |
| compact state buffer | 32 | `ap-XoQlp60McJPvJOIn3JA2mI` | `25,281` | `7.777` | `0.000s` | `2.811s` | `0.447s` | `0.541s` | `0.648s` |

Same current-code borrowed single-actor comparator:

| row | sims | run | roots/sec | total sec | prod->compact | actor render write |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| borrowed production state | 16 | `ap-hkymWoODNXWgeUwWS9vDs9` | `48,569` | `4.048` | `0.489s` | `0.000s` |
| borrowed production state | 32 | `ap-s6XSWUxfZJ987qaRwWcwLK` | `37,915` | `5.186` | `0.336s` | `0.000s` |

Plain read:

```text
The patch proves the conversion leaf can be removed. It is not a promotion
candidate yet. The removed renderer conversion is partly or fully replaced by
parent-side compact trail writes. At sim16 the net is positive versus copied
production state; at sim32 it is negative. The borrowed single-actor
production-state path is still faster on both sim counts because it avoids the
parent trail copy entirely.
```

Current decision:

```text
Keep this path as a diagnostic and a stepping stone, not as Coach advice. The
real next move is not "parent copies production into compact buffers"; it is
one owner for compact render state: either env/runtime maintains compact x/y
state directly, the persistent renderer consumes production visual_trail_pos
without splitting on the host, or the env emits a small per-step trail delta.
```

## 2026-05-23 Compact Replay Learner-Batch Edge

Status: local validation. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, Modal run, or Modal volume was touched.

What changed:

```text
tests/test_compact_search_replay_contract.py now composes:

CompactReplayIndexRowsV1
-> materialize_compact_target_rows_from_index_rows_v1
-> build_source_state_multiplayer_sample_batch_v0

and compares the sampled learner-facing batch against the trusted immediate
target-row path using the same seed.
```

Validation:

```text
uv run ruff check tests/test_compact_search_replay_contract.py
=> All checks passed

uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  -k 'materialized_sample_batch or materialize_same_as_immediate_rows'
=> 2 passed
```

Plain read:

```text
The compact index-row path now reaches the repo learner-facing sample batch
without changing learner-visible fields. This still does not prove stock
LightZero GameBuffer parity, but it closes the immediate local sampler edge.
```

## 2026-05-23 Compact Replay RND/Terminal Validation

Status: local validation. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, Modal run, or Modal volume was touched.

What changed:

```text
tests/test_compact_search_replay_contract.py now has a combined canary that
builds compact service replay index rows, materializes them, checks a terminal
row uses final_observation instead of the latest live observation, and feeds
the resulting replay observations through the actual CurvyRNDRewardModel
collect/train/estimate path.
```

What it proves:

```text
The compact service replay contract is not only shape-compatible with RND. It
can move real RND predictor weights, keep the frozen RND target network fixed,
produce bounded intrinsic reward changes, and keep terminal final observations
attached to the same compact records.
```

Validation:

```text
uv run ruff check src/curvyzero/training/compact_search_service.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_exploration_bonus.py
=> All checks passed

uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_exploration_bonus.py
=> 180 passed
```

Remaining gap:

```text
This still does not make the compact service path trainer-facing. The next
proof should cover the learner sampler/sample batch edge and compare a faster
compact service path against the trusted immediate replay path over a
multi-record closed loop.
```

Additional identity gate:

```text
`test_compact_search_result_v1_rejects_identity_and_legality_errors` now also
rejects swapped-player compact search results and duplicate/missing root ids
before compact replay rows are written. This is aimed at future async or
batched services where result ordering can be wrong even if array shapes look
valid.

Validation:
  uv run ruff check tests/test_compact_search_replay_contract.py docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20
  => All checks passed

  uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py
  => 12 passed
```

## 2026-05-23 Fresh Current-Code H100 Refresh Matrix

Status: profile-only. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, or Modal volume was touched.

Shape:

```text
H100, B1024/P2, body4096, hidden64, max_depth16, loop96,
native_actor_buffer=True, resident GPU stack, replay-index on,
explicit resident-stack sync off, root observation copy off.
Refresh-on rows use borrowed single-actor render state.
Refresh-off rows disable borrowed state because that flag requires refresh-on.
```

Rows:

| row | run | sims | refresh | roots/sec | total sec | env frac | search frac | slowest |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| refresh-on | `ap-R52JRpKZc9fkOxBBzQ6u6l` | 16 | on | `53,899` | `3.648` | `75.2%` | `16.9%` | env |
| refresh-on | `ap-FbNK1dC03hCQwymagb4vzY` | 32 | on | `36,415` | `5.399` | `65.0%` | `28.6%` | env |
| refresh-off ceiling | `ap-gaE4f8WuWQJjqKA37xAvVL` | 16 | off | `91,763` | `2.143` | `56.2%` | `29.6%` | env |
| refresh-off ceiling | `ap-ToDKndEWNjnMEp1FWtRUJn` | 32 | off | `57,891` | `3.396` | `39.5%` | `49.2%` | search |

Refresh-off ceiling ratios:

```text
sim16 refresh-off / refresh-on: 1.70x
sim32 refresh-off / refresh-on: 1.59x
```

Important leaf timers:

| row | game mechanics | public packaging | observation handoff | gpu draw | prod->compact | delta pack | renderer H2D | persistent update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| on sim16 | `0.549s` | `0.442s` | `1.702s` | `0.0206s` | `0.381s` | `0.475s` | `0.386s` | `0.332s` |
| on sim32 | `0.666s` | `0.503s` | `2.278s` | `0.0236s` | `0.473s` | `0.504s` | `0.459s` | `0.689s` |
| off sim16 | `0.547s` | `0.596s` | `0.0015s` | `0.000s` | `0.000s` | `0.000s` | `0.000s` | `0.000s` |
| off sim32 | `0.652s` | `0.634s` | `0.0009s` | `0.000s` | `0.000s` | `0.000s` | `0.000s` | `0.000s` |

Plain read:

```text
The refresh path is still worth attacking, but the ceiling is only about
1.6-1.7x on this denominator. Raw GPU draw is not the target: it is only about
20-24ms over 96 closed-loop steps. The hot refresh-on leaves are
production-to-compact, delta pack, renderer H2D, persistent update, public
packaging, and search.

At sim32, search is already large: 28.6% of refresh-on wall and 49.2% of the
refresh-off ceiling. So the next serious speed work must combine compact
state/search-input ownership with search-service/backend work. A renderer-only
patch cannot produce 5-10x.
```

## 2026-05-23 Durable Compact Service Common-Telemetry Rerun

Status: profile-only. No live Coach training run, checkpoint, eval, GIF,
tournament artifact, or Modal volume was touched.

Why this rerun exists:

```text
The first same-shape 2026-05-23 mock/service-tax Modal jobs finished after the
local process table changed, and their JSON return was not recoverable from app
logs. I reran the rows through scripts/run_curvytron_hybrid_observation_profile_manifest.py
so every result is saved locally under artifacts/local.
```

Shape:

```text
H100, B512/A16, sim16, 80 measured steps, 20 warmup steps,
direct_gray64, persistent GPU policy framebuffer profile,
uint8 stack storage, scalar timestep materialization off,
compact service replay proof on, root noise 0.0,
host_uint8_pinned LightZero input.
```

Durable result dirs:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_direct_20260523
artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_mock_20260523
artifacts/local/curvytron_hybrid_observation_profile_results/rerun_current_compact_service_tax_20260523
```

Rows:

| row | search semantics | steps/sec | measured sec | probe total sec | probe roots/sec | model sec | search sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mock_search_service` | fake search, compact arrays only | `14,969.95` | `5.472s` | `0.548s` | `149,470` | `0.361s` | `0.000s` |
| `service_tax_probe` | real initial + recurrent model calls, fake tree | `11,854.61` | `6.910s` | `2.177s` | `37,626` | `1.743s` | `0.254s` |
| `direct_ctree_gpu_latent` | real LightZero CTree MCTS | `5,965.11` | `13.733s` | `8.016s` | `10,220` | `2.123s` | `6.164s` |

Ratios by end-to-end profile steps/sec:

```text
mock / direct:        2.51x
service_tax / direct: 1.99x
mock / service_tax:   1.26x
```

Plain read:

```text
This confirms the current Amdahl read under common compact telemetry. The
largest profile-only opportunity is not raw rendering. It is the search/dataflow
boundary around direct LightZero CTree: CPU CTree control, recurrent model calls,
per-simulation output movement/listification, and replay-safe payload assembly.

The service-tax row is the important conservative signal. It still pays real
model calls, but avoids the CTree/list/control path, and it is about 2x faster
than the direct CTree row on this denominator.

This is not a trainer-facing recommendation yet. It says the next optimizer
lane should build a real compact/fixed-shape search service behind the validated
CompactSearchServiceV1 boundary, then extend the proof through RND and player
perspective before touching Coach training.
```

## 2026-05-22 Fresh Compact Service Sidecar Comparator

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Shape:

```text
H100, B512/A16, 60 measured steps, 15 warmup steps,
direct_gray64, persistent GPU policy framebuffer profile,
uint8 stack storage, scalar timestep materialization off,
compact service replay proof on, root noise 0.0.
```

Rows:

| row | run | search semantics | steps/sec | measured sec | main read |
| --- | --- | --- | ---: | ---: | --- |
| `mock_search_service` | `ap-XChnzStDiIYMLtprr7qQ0H` | fake search, real initial inference | `17,711.89` | `3.469s` | ceiling if search/control were cheap |
| `service_tax_probe` | `ap-Yx4retayDwFuKDzFWSQgHM` | real initial + recurrent inference, fake tree | `12,461.65` | `4.930s` | model/recurrent tax without CTree/list control |
| `direct_ctree_gpu_latent` | `ap-AxV03FKEyemvd1okLy0Zxc` | real LightZero CTree MCTS | `7,155.66` | `8.586s` | current real-search comparator |

Ratios:

```text
mock / direct:        2.48x
service_tax / direct: 1.74x
mock / service_tax:   1.42x
```

Direct CTree timing buckets:

```text
batched_stack_probe_wall_sec:             5.423s
lightzero_mcts_arrays_boundary_total_sec: 5.048s
lightzero_mcts_arrays_boundary_search_sec: 3.941s
lightzero_consumer_model_total_sec:       1.290s
lightzero_consumer_direct_boundary_non_model_sec: 3.758s
ctree traverse + backprop:                1.037s
root_prepare_sec:                         0.494s
observation_sec / renderer_stack_update:  1.263s
actor_step_wall_sec:                      1.550s
compact_service_replay_proof_sec:         0.174s
```

Plain read:

```text
The current real CTree boundary is still a large wall. The fake-search ceiling
shows a real `~2.5x` service/topology opportunity on this profile shape.

The service-tax row is the more conservative signal: even after paying real
initial and recurrent model calls, avoiding the CTree/list/control path is
`~1.7x` faster than direct CTree.

This still does not prove a trainer-facing 10x. It says the next serious
optimizer lane should be a compact search-service / fixed-shape search boundary
with replay/RND/player/terminal parity gates, not more isolated renderer work.
```

Rerun note:

```text
The earlier direct app `ap-XllGkDYeIlyL2hbjlRcpax` emitted only startup logs and
no JSON result, so it is not used as evidence. The clean direct comparator is
`ap-AxV03FKEyemvd1okLy0Zxc`.
```

## 2026-05-22 Direct Root-Node Extraction Fix

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
_extract_mctx_root_values now tries the direct root node value from the MCTX
search tree before falling back to search_tree.summary().

The earlier code path effectively paid for the expensive summary/materialized
payload path just to get root values.
```

Local validation:

```text
uv run ruff check src/curvyzero/infra/modal/mctx_synthetic_benchmark.py tests/test_mctx_synthetic_benchmark_legality.py tests/test_compact_search_replay_contract.py
uv run python -m py_compile src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
uv run pytest -q -p no:cacheprovider tests/test_mctx_synthetic_benchmark_legality.py tests/test_compact_search_replay_contract.py
```

Result: ruff passed, py_compile passed, focused tests `12 passed`.

Shape:

```text
H100, B1024/P2, loop24, no-death compact loop,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack, no root observation copy,
explicit resident-stack sync off,
vectorized delta pack off.
```

Rows:

| row | run | sims | replay index | roots/sec | total sec | root value extract | env step | search |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| full materialization | `ap-h8eXVxgdCZN2LIopHF7B2u` | 16 | off | `71,244` | `0.690` | `0.014s` | `0.480s` | `0.152s` |
| deferred payload flush | `ap-6NR7HJCrkjYufLOIKfCs2t` | 16 | off | `63,163` | `0.778` | flush `0.021s` | action loop `0.757s` | n/a |
| full materialization | `ap-Oq9wRkcR6m9g8TpDVCoKzM` | 32 | off | `43,457` | `1.131` | `0.021s` | `0.677s` | `0.374s` |
| deferred payload flush | `ap-6UK2GEsLAxna5qU7ZmSrVy` | 32 | off | `49,159` | `1.000` | flush `0.017s` | action loop `0.983s` | n/a |
| replay-valid full | `ap-Y6LC6pbtD7ZbiIvXPb7d3D` | 16 | on | `54,977` | `0.894` | `0.019s` | `0.656s` | `0.157s` |
| replay-valid full | `ap-QDsbC1GNnAiyGnsrIQRXU6` | 32 | on | `38,122` | `1.289` | `0.024s` | `0.771s` | `0.418s` |

Longer stability rows:

| row | run | sims | loop steps | replay index | roots/sec | total sec | replay index sec | root value extract | env step | search |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| replay-valid full | `ap-yqMpUJbZwFMhYourDP65u5` | 16 | 48 | on | `44,011` | `2.234` | `0.027s` | `0.049s` | `1.678s` | `0.347s` |
| replay off full | `ap-n6voAA6ytAJTwv2x5H6d4f` | 16 | 48 | off | `70,408` | `1.396` | `0.000s` | `0.028s` | `0.977s` | `0.305s` |
| replay-valid full | `ap-x6Fy3jNx0jfJWkucx7utG0` | 32 | 48 | on | `46,946` | `2.094` | `0.018s` | `0.037s` | `1.194s` | `0.754s` |
| replay off full | `ap-jyGhPww8F3PgQfhnIf6jkQ` | 32 | 48 | off | `45,494` | `2.161` | `0.000s` | `0.035s` | `1.240s` | `0.785s` |
| replay-valid full | `ap-X4qFnFS9XbkuHKIO6vmwci` | 16 | 96 | on | `50,617` | `3.884` | `0.038s` | `0.069s` | `2.963s` | `0.621s` |
| replay off full | `ap-snG5Z3yGKzGWo8hdTIKqfp` | 16 | 96 | off | `53,579` | `3.669` | `0.000s` | `0.061s` | `2.806s` | `0.613s` |

Plain read:

```text
This is a real optimizer bug fix. Root-value extraction fell from about
0.26-0.31s to about 0.014-0.024s on the same shape.

The old action-only ceiling was useful because it exposed that the full result
path was too expensive. After this fix, full materialization is close enough
that action-only is only a diagnostic ceiling again.

Replay-index construction is not the main wall. In the replay-valid rows it is
about 0.010-0.012s over 24 closed-loop steps, about 0.018-0.027s over 48
steps, and about 0.038s over 96 steps. The loop48 sim16 replay-on/off mismatch
was not explained by replay_index_sec; env/render time changed sharply. The
loop96 sim16 pair is the cleaner read and shows replay-valid rows cost only a
small fraction of total wall time.
```

Current decision:

```text
Do not promote serial deferred payload flushing. It is no longer the next big
move. Do not promote Python-thread overlap either; the overlap canary inflated
env/render time through contention.

Next target: env/observation/search-input handoff at sim16, and both that
handoff plus MCTS search at sim32+.
```

## 2026-05-22 Async Renderer H2D And Hidden Timer Follow-Up

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
The existing persistent-renderer async device-only profile flag now also avoids
blocking inside `_copy_state_to_device`. Default behavior still blocks after
device_put; only the opt-in profile flag defers those waits.

Added `compact_batch_build_sec` and `batched_stack_probe_wall_sec` to expose
whether `_make_compact_batch`/capture-probe work was hidden inside env_step_sec.
```

Local validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py src/curvyzero/infra/modal/mctx_synthetic_benchmark.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py -k 'timing or persistent or compact_service_replay_proof or async or device'
```

Result: ruff passed; focused tests `23 passed`.

Rows:

| row | run | sims | loop steps | async device-only | roots/sec | total sec | read |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| baseline | `ap-72TeEccePu9pjaGgaD35lj` | 16 | 96 | off | `50,358` | `3.904` | paired baseline |
| async H2D | `ap-sDRS3TY7mm4EWhXBeWN9Qv` | 16 | 96 | on | `53,192` | `3.696` | small `1.06x` win |
| baseline | `ap-o3ZC9X3cX4AdC6WN2wYNbV` | 32 | 96 | off | `40,010` | `4.914` | paired baseline |
| async H2D | `ap-WPk47lTw1F6FCIaOCCSORz` | 32 | 96 | on | `41,902` | `4.692` | small `1.05x` win |
| timer check | `ap-XQkmXe2e1L18k2ehqphZIY` | 16 | 48 | off | `54,358` | `1.808` | compact batch build `0.0016s`, probe wall `0.0006s` |

Plain read:

```text
Deferring renderer H2D waits helps a little, but not enough to change the
architecture priority. Keep it as an opt-in profiling flag.

The hidden compact-batch timer is tiny on the repeated closed loop. The next
wall is not `_make_compact_batch`; it is the repeated renderer/search-input
work around production-to-compact, delta pack, H2D/update, resident stack,
public packaging, and actual CPU mechanics.
```

## 2026-05-22 Search Payload Readback / Deferred Flush

Superseded by the direct root-node extraction section above. This earlier
section records the first payload-split read before the root-value extractor
bug was fixed.

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
Added closed_loop_deferred_payload_profile to
src/curvyzero/infra/modal/mctx_synthetic_benchmark.py.

It reads only selected actions during the closed env/search loop, then flushes
visit-policy/root-value payloads after the loop and reports that flush as its
own bucket. It is not replay-valid yet; replay rows remain off.
```

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/mctx_synthetic_benchmark.py tests/test_mctx_synthetic_benchmark_legality.py
uv run python -m py_compile src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
uv run pytest -q -p no:cacheprovider tests/test_mctx_synthetic_benchmark_legality.py
```

Result: ruff passed, py_compile passed, legality tests `5 passed`.

Shape:

```text
H100, B1024/P2, loop24, no-death compact loop,
native_actor_buffer=True, actor_count=1,
borrow_single_actor_render_state=True,
resident GPU stack, no root observation copy,
explicit resident-stack sync off,
vectorized delta pack off,
closed_loop_replay_index=False.
```

Rows:

| row | run | sims | roots/sec | total sec | key read |
| --- | --- | ---: | ---: | ---: | --- |
| full materialization | `ap-vVzFVLrXWDKjyXOm3ud17S` | 16 | `44,793` | `1.097` | root value extract `0.310s` |
| action-only ceiling | `ap-06pBG47BIR417IiTT1IdBD` | 16 | `68,848` | `0.714` | no visit-policy/root-value copy |
| deferred payload flush | `ap-AMnIjrImmEdmfhO0l3h7N9` | 16 | `42,872` | `1.146` | action loop `0.834s`, flush `0.313s` |
| full materialization | `ap-QkdwWsg9MiBGzSeQ6xFhLe` | 32 | `39,780` | `1.236` | root value extract `0.266s` |
| action-only ceiling | `ap-v7cXGsiiF76GVsaetIJ9Fj` | 32 | `53,275` | `0.923` | no visit-policy/root-value copy |
| deferred payload flush | `ap-c7uPohbOne9qVdBXz3lCql` | 32 | `39,330` | `1.250` | action loop `1.010s`, flush `0.240s` |

Plain read:

```text
Action-only is a useful ceiling, not a training/replay speed number. It shows
the action-critical path can be much smaller when selected actions are the only
per-step search output read by the CPU env.

The deferred flush rows are the corrective result: if we simply move the
visit-policy/root-value copy to the end and still pay it serially, total
throughput is basically baseline. The cost was moved, not removed.
```

Next architecture implication:

```text
The next serious canary should not be "defer then flush serially." It should be
one of:
  1. overlap search payload copy with CPU env/render work;
  2. keep policy/value payloads in a device or pinned-host ring;
  3. build replay chunks only when payloads are ready, without exposing partial
     rows to the learner/sampler.

Any claim must remain profile-only until replay rows match the current compact
builder on actions, policy target, root value, reward, done, final reward, row
ids, and player ids.
```

## 2026-05-22 Fresh Resident Sync / Async Device-Only Retest

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch after this retest:

```text
_persistent_delta_state now has a guarded vectorized fast path for the common
case where each row's new trail span is contiguous, active, and has valid
owners. Sparse or invalid spans fall back to the old exact row/slot loop and
preserve the compact delta width contract.
```

Local validation:

```text
uv run ruff check src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_boundary_profile.py
uv run python -m py_compile src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py -k persistent_delta_state
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py
```

Result after the A/B flag patch: ruff and py_compile passed; focused delta
tests `5 passed`; full boundary profile test file `105 passed`.

Corrected current-code A/B after adding an explicit profile flag:

```text
shape: B1024/P2 loop24 borrowed/resident/sync-off, exact same code build

sim16:
  old exact delta pack, ap-bs1m5ljCv5wVdk2XZQHK9C: 51,879 roots/sec
  vectorized delta pack, ap-i38zOVqa0ep73d9YJkLRHg: 53,066 roots/sec
  gain: about 1.02x

sim32:
  old exact delta pack, ap-ovaqKpNGTn9YfePrFZ22wp: 45,094 roots/sec
  vectorized delta pack, ap-i01GoPWwcBkuHQlbtCcKAq: 37,869 roots/sec
  result: regression/noisy enough that it is not promotable
```

Important sim32 exact-path buckets:

```text
total_sec: 1.090s
env_step_sec: 0.373s, 34.2%
search_sec: 0.368s, 33.8%
unlabeled_residual_sec: 0.307s, 28.2%
renderer_persistent_delta_pack_sec: 0.009s
renderer_host_to_device_sec: 0.085s
renderer_production_to_compact_sec: 0.056s
public_packaging_leaf_sec: 0.111s
game_mechanics_leaf_sec: 0.065s
```

Plain read:

```text
The vectorized fast path is validated but not promoted. It gives only a tiny
sim16 win and regresses the sim32 A/B row. It now stays opt-in under
persistent_vectorized_delta_pack_profile=False by default. The recommended
current profile path is the old exact delta pack plus borrowed render state,
resident GPU stack, no root observation copy, and resident-stack sync off.
```

Search-payload materialization split:

```text
New timers split root_sidecar, root_value_extract, search_result_validate, and
joint_action_build out of the old unlabeled/control residual.

Action-only profile is an invalid training/replay ceiling:
  it reads selected actions only, drives the next env step, skips root
  values/visit policies/search-result validation, and requires replay-index off.
```

Rows:

| shape | run | sims | roots/sec | read |
| --- | --- | ---: | ---: | --- |
| replay-valid materialization, replay on | `ap-EAsYtOV1CmLCd7jQNl8Tog` | 16 | `47,936` | root value extract `0.258s`, residual now explained |
| replay-valid materialization, replay on | `ap-iWshipr4X6YjZi8gHVPR4J` | 32 | `37,799` | root value extract `0.286s`, residual now explained |
| action-only ceiling, replay off | `ap-h5jLY46rV2IVci4hGb6HhV` | 16 | `72,335` | skips root value / visit-policy materialization |
| action-only ceiling, replay off | `ap-6IQioNVpBpj2UqT367eTxl` | 32 | `48,139` | skips root value / visit-policy materialization |

Plain read:

```text
The next serious architecture lever is search payload ownership. The env only
needs selected actions each step. Replay/training needs visit policies and root
values, but those should be chunked, delayed, or kept resident instead of
forcing per-step CPU materialization in the action loop. This is profile-only
evidence; action-only is not a training-valid lane by itself.
```

Shape:

```text
H100, B1024/P2, loop24, native actor buffer, actor_count=1,
body_capacity=4096, hidden_dim=64, max_depth=16, rollout_steps=4,
borrow_single_actor_render_state=True, resident GPU stack,
no root observation copy, closed-loop replay-index on.
```

Rows:

| row | run id | sims | roots/sec | total sec | env frac | search frac | observation | GPU draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| explicit resident sync on | `ap-zSmLwNjivhnezvD2d53YBO` | 16 | `45,362` | `1.0835s` | `51.7%` | `14.3%` | `0.316s` | `0.0054s` |
| explicit resident sync on | `ap-gnfhF65yYsJDqQQfMjFn50` | 32 | `35,674` | `1.3778s` | `43.8%` | `27.0%` | `0.333s` | `0.0060s` |
| explicit resident sync off | `ap-o96ymaHMxE66bsMbdSKMY7` | 16 | `50,357` | `0.9761s` | `53.7%` | `15.6%` | `0.2996s` | `0.0045s` |
| explicit resident sync off | `ap-Cp19V4fhXnl159itWx4A9z` | 32 | `43,340` | `1.1341s` | `40.9%` | `32.9%` | `0.2652s` | `0.0040s` |
| sync off + async internal renderer | `ap-k3bz0aOUHkgn3vEo99WYeD` | 16 | `50,239` | `0.9784s` | `48.8%` | `15.2%` | `0.2682s` | `0.0031s` |
| sync off + async internal renderer | `ap-HOxmTsNXKn9ugoGCHUZHvZ` | 32 | `40,909` | `1.2015s` | `42.5%` | `31.9%` | `0.2858s` | `0.0025s` |

Plain read:

```text
The best current profile-only row is borrowed render state + resident GPU stack
+ explicit resident-stack sync off. The extra async internal renderer flag
does not improve total wall time and should stay an attribution canary.
```

What this means:

```text
The raw GPU draw wait is not the next large wall. The next wall is the broader
next-search-input boundary: production-to-compact, delta pack, H2D/update,
stack/root ownership, public packaging, search, and Python/control residual.
The next useful architecture experiment is a more resident compact state owner
or actor-emitted compact deltas, not another renderer wait toggle.
```

## 2026-05-22 Borrowed Single-Actor Render State Canary

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

What this tested:

```text
actor owns env.state
-> skip parent visual_trail render-state copy
-> persistent GPU renderer borrows actor env.state directly
-> resident GPU stack
-> CompactRootBatchV1 -> MCTX -> CompactSearchResultV1 / replay-index rows
```

This is only valid for the current profile shape:

```text
native_actor_buffer=True
actor_count=1
refresh_observation_stack=True
resident_gpu stack
no terminal/autoreset rows
```

The code fails closed if terminal rows appear, because borrowed post-reset state
would not be the same as a captured terminal frame.

Validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/infra/modal/mctx_synthetic_benchmark.py tests/test_source_state_hybrid_observation_profile.py
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py -k borrow_single_actor_render_state
```

Result: ruff passed; `3 passed, 32 deselected`.

Rows:

| row | run id | sims | roots/sec | total sec | env frac | search frac | actor render-state write | observation | GPU draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| borrowed resident | `ap-zSmLwNjivhnezvD2d53YBO` | 16 | `45,362` | `1.084s` | `51.7%` | `14.3%` | `0.000s` | `0.316s` | `0.005s` |
| borrowed resident | `ap-gnfhF65yYsJDqQQfMjFn50` | 32 | `35,674` | `1.378s` | `43.8%` | `27.0%` | `0.000s` | `0.333s` | `0.006s` |

Matched comparison against the latest pre-borrow resident rows:

| shape | pre-borrow roots/sec | borrowed roots/sec | speedup |
| --- | ---: | ---: | ---: |
| sim16 resident, loop24 | `30,297` | `45,362` | `1.50x` |
| sim32 resident, loop24 | `26,805` | `35,674` | `1.33x` |

Plain read:

```text
Borrowing actor env.state directly removes the full visual_trail_* parent-buffer
copy and moves total wall time, not just a timer leaf. The remaining wall is now
renderer delta-pack/H2D plus observation/stack ownership, public packaging, MCTS
search, and a large unlabeled residual. This confirms state ownership is the
right architecture lane. It does not make the full loop 10x faster by itself.
```

Next:

```text
Do not polish raw GPU drawing first; it is still only about 5-6ms in these rows.
Split the remaining observation_sec/renderer_render_sec into exclusive
delta-pack, H2D, update, stack, and synchronization costs, then prototype a
more resident compact render/search state owner that avoids repacking the same
state each step.
```

## 2026-05-22 Vector Ray-Cast Capacity Bug

Status: code patch plus local profile. No live Coach training run, checkpoint,
eval, GIF, or tournament artifact was touched.

Finding:

```text
VectorTrainerEnv1v1NoBonus batched ray observations were scanning the full
body_capacity=4096 trail allocation even when body_write_cursor.max() was only
about 2 in the short sample.
```

Patch:

```text
_cast_rays_batch_1v1 now slices body_active/body_pos/body_radius/body_owner
and body_num to max(body_write_cursor) before building ray masks.
```

Validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_vector_trainer_observation.py
uv run ruff check src/curvyzero/env/vector_trainer_observation.py tests/test_vector_trainer_observation.py
```

Result: `18 passed`; ruff passed.

Local profile, B512/rollout8/straight/no-event/no-replay:

| row | ego decisions/sec | env transitions/sec | env step sec | reset sec | ray-cast probe sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| before trim | `456.70` | `228.35` | `2.1129` | `3.413` | `1.891` |
| after trim | `20007.78` | `10003.89` | `0.0389` | `0.165` | `0.0037` |

Caveat:

```text
This local B512 script still stops the rollout after the first terminal row, so
do not quote it as full training throughput. It is a valid diagnosis of the
ray observer waste: inactive trail capacity was the wall in this vector sample.
```

Plain read:

```text
This is the kind of easy Amdahl bug we should take immediately. It does not
replace the larger compact MCTX/search-boundary investigation, and it is not
the current stock visual Coach path by itself.
```

Modal rerun after trim, H100/B512/rollout8/sim16:

| field | value |
| --- | ---: |
| `ok` | `true` |
| host observation setup | `0.457s` |
| env reset | `0.140s` |
| 8 rollout steps total | `0.260s` |
| active decisions/sec, search resident | `162,690` |
| active decisions/sec, fresh H2D/search/D2H | `146,784` |

Before the trim, the same vector sample setup was about `43.65s`. That is a
roughly `95x` setup collapse for the short-cursor vector sample. The search
rate itself did not need to change; the wasted observation construction did.

## 2026-05-22 Real Compact Visual Roots Into MCTX

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

What changed:

```text
Added curvytron_hybrid_compact_visual_sample to mctx_synthetic_benchmark.
It builds real HybridCompactBatch observations, validates CompactRootBatchV1,
runs MCTX/JAX Gumbel MuZero search, then validates CompactSearchResultV1.
```

The mode can use either:

- `cpu_oracle`: slow reference renderer.
- `jax_gpu_persistent_policy_framebuffer_profile`: persistent GPU policy-space
  renderer; this is now the default for this profile mode.

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/mctx_synthetic_benchmark.py tests/test_mctx_synthetic_benchmark_legality.py
uv run pytest -q -p no:cacheprovider tests/test_mctx_synthetic_benchmark_legality.py tests/test_vector_trainer_observation.py
uv run python -m py_compile src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
```

Result: `20 passed`; ruff and py_compile passed.

Rows:

| renderer | shape | sims | run | ok | host setup | last render | resident roots/s | fresh-boundary roots/s |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| CPU oracle | B64/P2 | 8 | `ap-efXuZtm4JrvCbZFmlCskXE` | true | `2.954s` | `0.665s` | `41,191` | `29,307` |
| CPU oracle | B256/P2 | 16 | `ap-qg5YZjJRDl0PbZW4RWhVUS` | true | `16.890s` | `3.909s` | `91,190` | `33,753` |
| persistent GPU | B64/P2 | 8 | `ap-0YRVN6LxR9gV1G9LFhS8O5` | true | `2.577s` | `0.007s` | `41,725` | `31,945` |
| persistent GPU | B256/P2 | 16 | `ap-tVRQXSbfQ245qydl6zc8Es` | true | `2.860s` | `0.014s` | `89,747` | `70,330` |

Plain read:

```text
The compact visual MCTX gate now uses real CurvyTron visual roots and passes
the compact root/search contracts. CPU rendering is the wrong denominator for
speed. Persistent GPU rendering removes the big host setup wall for this gate,
especially at B256. The remaining speed question is no longer "can we draw the
frame"; it is whether this compact MCTX/search result can drive replay/learner
semantics without falling back into stock LightZero object boundaries.
```

## 2026-05-22 Compact Closed-Loop Mechanics vs Observation Split

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Shape:

```text
H100, B1024/P2, sim16, loop16, native actor buffer,
body_capacity=4096, fast compact visual adapter, no root observation copy.
```

Rows:

| stack | run id | roots/sec | env frac | search frac | mechanics | render-state write | visual-trail write | observation | raw GPU draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resident | `ap-tPtlc8abDmtUsjLuqYuNOi` | `37,883` | `64.2%` | `11.8%` | `0.052s` | `0.245s` | `0.244s` | `0.193s` | `0.003s` |
| host | `ap-SKFkPMiLLBRws9KNyTDPyJ` | `25,712` | `67.6%` | `7.9%` | `0.098s` | `0.323s` | `0.321s` | `0.374s` | n/a |

Plain read:

```text
env_step_sec is not game mechanics. In this profile shape, actual mechanics are
about 10% of env_step_sec. Observation/search-input handoff is the wall. The
largest named leaf is copying visual_trail_* into parent render-state buffers.
The GPU draw itself is tiny.
```

Next target:

```text
Prototype a profile-only live-prefix visual-trail write or compact render-state
owner. The falsifier is matched total roots/sec, not timer movement alone.
```

## 2026-05-22 Borrowed Single-Actor Render-State Falsifier

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

What this tested:

```text
actor env.state -> renderer borrowed mapping
instead of:
actor env.state -> parent native_render_state buffers -> renderer mapping
```

The mode is guarded to `actor_count=1`, `native_actor_buffer=True`, persistent
renderer, refresh-on rows, and no terminal/autoreset rows. It fails closed
instead of silently rendering post-reset state.

Local validation:

```text
uv run ruff check src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/infra/modal/mctx_synthetic_benchmark.py tests/test_source_state_hybrid_observation_profile.py
uv run python -m py_compile src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py -k "borrow_single_actor or native_actor_buffer_filters or persistent_device_only"
```

Result: ruff and py_compile passed; `6 passed`.

Shape:

```text
H100, B1024/P2, loop24, native actor buffer, actor_count=1,
body_capacity=4096, no root observation copy, replay-index on.
```

Rows:

| stack | sims | mode | run id | roots/sec | env frac | search frac | actor render-state write | observation |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| host | 16 | copied | `ap-MhVC3PNhGgrzmmIh70R3rA` | `21,310` | `62.3%` | `8.0%` | `0.314s` | `0.782s` |
| host | 16 | borrowed | `ap-EhCBouA9LkFZdEO8ECsiPA` | `35,687` | `56.3%` | `11.0%` | `0.000s` | `0.515s` |
| resident | 16 | copied | `ap-evx57wUrZf78h3bdWzb4Ui` | `32,271` | `65.7%` | `10.2%` | `0.336s` | `0.334s` |
| resident | 16 | borrowed | `ap-7sAJ8IvVTKxUzLNxVhjsxv` | `44,787` | `53.2%` | `14.3%` | `0.000s` | `0.336s` |
| resident | 32 | copied | `ap-cq5OPumBe5kaj8Lku26p1t` | `34,646` | `49.4%` | `25.8%` | `0.217s` | `0.278s` |
| resident | 32 | borrowed | `ap-A08HI7rRr6LYyUK8hwhp84` | `36,124` | `44.6%` | `26.2%` | `0.000s` | `0.330s` |
| resident | 16 | refresh off | `ap-QM5ngR21iGuPrdAeOoV1dI` | `57,086` | `33.5%` | `18.7%` | `0.000s` | `0.000s` |
| resident | 32 | refresh off | `ap-DAG2RiZiBHJj37fS2huhqW` | `37,728` | `24.4%` | `33.0%` | `0.000s` | `0.000s` |

Speed read:

```text
host sim16 copied -> borrowed:     1.67x
resident sim16 copied -> borrowed: 1.39x
resident sim32 copied -> borrowed: 1.04x
```

Plain read:

```text
Borrowed single-actor render state is worth keeping in the profile lane. It
deletes the visual_trail_* parent-buffer copy and moves total throughput at
sim16. At sim32, search and residual host/control work become visible enough
that the same render-state fix is almost exhausted. This is the Amdahl pivot:
after borrowing, the next big work is not the parent visual-trail copy; it is
renderer/observation handoff plus search/control topology.
```

Important guardrail:

```text
The failed refresh-off borrowed rows were expected. Borrowed render state only
applies when observations are refreshed; no-refresh rows are a ceiling test, not
a legal borrowed-state mode.
```

## 2026-05-21 MCTS Arrays-Boundary Facade Smoke

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

What this tested:

```text
pre-scalar [B,2,4,64,64] stack
-> stock LightZero MuZero collect_mode.forward
-> compact arrays decoded from the public LightZero result
```

Run:

```text
ap-Amg22e2oRyJHZMNqPy9god
```

Result:

| field | value |
| --- | --- |
| `ok` | `true` |
| `batched_stack_probe_backend_name` | `lightzero_mcts_arrays_boundary_consumer` |
| `batched_stack_probe_semantics` | `stock_lightzero_mcts_arrays_facade` |
| `calls_train_muzero` | `false` |
| `touches_live_runs` | `false` |
| `trainer_defaults_changed` | `false` |
| `lightzero_policy_device` | `cuda:0` |
| `action_shape` | `[8]` |
| `visit_shape` | `[8,3]` |
| `searched_value_shape` | `[8]` |
| `compact_output_bytes` | `192` |
| `public_output_bytes` | `1086` |
| `illegal_action_count` | `0` |

Plain read:

```text
The new facade is wired. It gives us the compact arrays boundary shape while
still letting stock LightZero own MCTS semantics internally.
```

Do not read this as a speed result. The row is tiny and still calls the public
LightZero collect path. The next useful speed rows are medium H100/L4
arrays-boundary rows plus a resident-input/H2D split.

### Medium Rows

Shape:

```text
B512 physical rows
2 player roots per row = 1024 roots per collect-forward call
25 measured steps, 5 warmup steps
sim8
uint8 [B,2,4,64,64] stack
no scalar timestep materialization
```

| compute | run id | scalar roots/sec | physical rows/sec | measured sec | collect-forward sec | decode sec | H2D sec | model sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L4/T4 | `ap-5COErgYQQR2Gb9IsShxOnY` | `1421.28` | `710.64` | `18.01` | `14.23` | `0.69` | `0.81` | `1.16` |
| H100 | `ap-4KPxuHpOOw4AfgD7rrlqIu` | `2319.65` | `1159.82` | `11.04` | `8.24` | `0.41` | `0.53` | `0.44` |

Both rows:

```text
ok=true
trainer_defaults_changed=false
touches_live_runs=false
calls_train_muzero=false
illegal_action_count=0
compact_output_bytes=24576
public_output_bytes=110825
```

Plain read:

```text
The compact facade adds a clean arrays boundary, but it does not remove the
current Amdahl wall yet. H100 is faster than L4/T4 here, but most time is still
inside public LightZero collect/MCTS wrapper work, not model inference and not
rendering.
```

## 2026-05-21 Direct CTree Arrays Probe

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

What this tested:

```text
pre-scalar [B,2,4,64,64] stack
-> real LightZero MuZero initial_inference
-> real policy._mcts_collect.search / CTree MCTS
-> compact action/value/visit arrays
```

Same medium shape as the stock facade rows:

```text
B512 physical rows
1024 player-view roots per call
25 measured steps, 5 warmup steps
sim8
uint8 stack
no scalar timestep materialization
```

| row | run id | scalar roots/sec | probe total | search | output assembly | H2D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100 stock facade | `ap-HJk70PQP2iLAvA7mxxn99u` | `2419.81` | `8.971s` | `3.101s` | `0.383s` decode | `0.587s` |
| H100 direct CTree arrays | `ap-XEoAIwCpbbQTuFLmSnjvwY` | `2806.64` | `7.777s` | `2.288s` | `4.709s` | `0.378s` |
| H100 direct CTree arrays fast path | `ap-XEB8GF9B2Gw5V600QVtu10` | `3859.44` | `4.927s` | `3.800s` | `0.027s` | `0.660s` |
| L4/T4 direct CTree arrays | `ap-5OB4ye6HKiGfPQ3UjP221v` | `1460.41` | `15.373s` | `4.020s` | `9.450s` | `0.647s` |

All rows:

```text
calls_train_muzero=false
touches_live_runs=false
trainer_defaults_changed=false
illegal_action_count=0
```

Plain read:

```text
Direct CTree arrays is real and faster than the stock facade, but the first
direct version mostly proved that Python output assembly can dominate if we are
careless. The all-actions-legal fast path removed that bucket and raised H100
throughput to 3859 roots/sec. The remaining profile wall is now the real MCTS
search/root-prep/input path around CTree, plus ordinary observation/H2D cost.
This is still profile-only, not Coach launch advice.
```

### Direct CTree Input-Mode Repeat

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

Shape:

```text
H100
B512 physical rows
1024 player-view roots per call
sim8
uint8 [B,2,4,64,64] stack
no scalar timestep materialization
60 measured steps
15 warmup steps
```

| input mode | run id | scalar roots/sec | measured sec | boundary total | H2D | search | root prep | model total | observation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `host_uint8` | `ap-QPLEHOs3dGrcs2tlRpbMge` | `4111.80` | `14.94s` | `10.24s` | `1.21s` | `7.86s` | `0.66s` | `1.04s` | `2.08s` |
| `host_uint8_pinned` | `ap-5F1tMU2HiuHXDcu4O1tGkw` | `4513.15` | `13.61s` | `8.87s` | `0.14s` | `7.54s` | `0.66s` | `1.06s` | `2.05s` |
| `resident_torch_reuse` | `ap-wsKyodSayU2KGsTgKKpAqc` | `5537.40` | `11.10s` | `6.95s` | `0.00s` | `5.83s` | `0.63s` | `0.84s` | `1.85s` |

Plain read:

```text
The short host-vs-pinned row was noisy. In the longer repeat, pinned input is a
real total-wall win of about 10% over host uint8, mainly because H2D drops from
about 1.21s to about 0.14s. Resident reuse is faster but stale-input-only, so
it is an upper bound and not a valid training mode.
```

Accounting note:

```text
These Modal rows were launched before every new accounting field landed. Future
rows now report input_freshness, input_transfer_bytes, and model-output D2H
sec/bytes. In current code, root_prepare excludes model-output D2H, and pinned
input reports pin/input-prep separately from H2D transfer. Use this table for
throughput and broad Amdahl shape, not as the final complete transfer ledger.
```

## 2026-05-21 Array-Ceiling H2D Input Split

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

What this tested:

```text
pre-scalar uint8 [B,2,4,64,64] stack
-> array-ceiling recurrent_toy model pressure
-> different ways to feed Torch/CUDA input
```

Common shape:

```text
H100
B512 physical rows
2 roots per row = 1024 roots/call
30 measured steps, 6 warmup steps
sim8
uint8 stack storage
scalar timestep materialization off
```

| input mode | run id | scalar roots/sec | physical rows/sec | probe total | H2D | host prep | device/model bucket |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `host_uint8` | `ap-8cD9SIXhWFkQyNYxTf4BbK` | `10086.23` | `5043.12` | `1.184s` | `0.597s` | `0.000s` | `0.455s` |
| `host_uint8_pinned` | `ap-ILMAfyK6FArmAMUa4BSrff` | `12295.15` | `6147.58` | `0.625s` | `0.071s` | `0.051s` pin | `0.429s` |
| `host_float32` | `ap-Ol4ORUHyRgLFwLiKeoGNSj` | `9641.80` | `4820.90` | `1.341s` | `0.171s` | `0.676s` | `0.378s` |
| `resident_torch_reuse` | `ap-PPENgpZQpWCTP4KLKilPSJ` | `14414.56` | `7207.28` | `0.497s` | `0.000s` | first fill excluded | `0.384s` |

Correction:

```text
The first host_float32 row was misleading because host_prenormalize_sec was
reported but not included in the probe total. Run ap-Ol4ORUHyRgLFwLiKeoGNSj is
the corrected row and includes host preprocessing.
```

Plain read:

```text
Pinned uint8 copy is a real low-effort transfer win. Keeping the model input
resident is the bigger ceiling. Host float32 preprocessing is not useful here.
This is still not a trainer-speed claim; it is a transfer-boundary split for
the next compact MCTS-boundary design.
```

## 2026-05-22 Train-Facing Direct CTree Hook Read

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

What was implemented:

```text
stock train_muzero
-> MuZeroPolicy._forward_collect hook during profile window only
-> direct_ctree_gpu_latent search backend
-> stock collector/replay/target/learner continue as before
```

Local validation:

```text
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_phase_profiler.py
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py
uv run ruff check --ignore F401 src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_phase_profiler.py
```

Result: `8 passed`; ruff passed. The F401 ignore is for pre-existing launcher
unused imports, not a new hook issue.

### Direct-Row Attestation Gate

Status: local tooling gate. No live Coach training run, checkpoint, eval, GIF,
or tournament artifact was changed.

What changed:

```text
optimizer profile summary rows now require collect_search_backend identity
and direct-backend self-audit counters before a direct_ctree_gpu_latent speed
claim can pass --require-attestation.
```

Focused validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_summarize_curvytron_optimizer_profile_results.py
uv run ruff check scripts/summarize_curvytron_optimizer_profile_results.py tests/test_summarize_curvytron_optimizer_profile_results.py
uv run python -m py_compile scripts/summarize_curvytron_optimizer_profile_results.py tests/test_summarize_curvytron_optimizer_profile_results.py
```

Result: `3 passed`; ruff and py_compile passed.

Plain read:

```text
The summary tool can no longer accidentally treat a direct-backend row as real
if it silently fell back to stock, omitted backend counters, or reported a
semantic identity that disagrees with the command.
```

### Compile-Spike Integration And Focused Validation

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

What changed:

```text
source_state_batched_observation_boundary_profile.py now has a
dense_torch_mcts_compile_spike mode. It keeps LightZero recurrent inference
eager, but tries to compile the fixed-shape dense selection and backup helpers.
```

Focused validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_lightzero_phase_profiler.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "collect_search_hook or profile_attestation or array_ceiling"

uv run ruff check --ignore F401 \
  scripts/compare_curvytron_direct_ctree_stock.py \
  scripts/summarize_curvytron_optimizer_profile_results.py \
  tests/test_lightzero_phase_profiler.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py

uv run python -m py_compile \
  scripts/compare_curvytron_direct_ctree_stock.py \
  scripts/summarize_curvytron_optimizer_profile_results.py \
  tests/test_lightzero_phase_profiler.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py
```

Result: `20 passed, 88 deselected`; ruff and py_compile passed.

Plain read:

```text
The compile-spike mode is wired and locally guarded, but it has not yet proved
a speed win. The next evidence must be H100 profile rows where compile telemetry
says compilation actually ran.
```

### H100 Compile-Spike Falsifier

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

Shape:

```text
H100
B512 physical rows
A16 actor count
1024 player-view roots per call
60 measured steps
15 warmup steps
root-noise0
all actions legal
host_uint8_pinned input
scalar timestep materialization off
```

Rows:

| row | app id | sim8 roots/sec | sim16 roots/sec | read |
| --- | --- | ---: | ---: | --- |
| `dense_torch_mcts_compile_spike` | `ap-dMjPlGmbGGtFrf1JJMysOW` / `ap-cOoZ5pLWOJhN38qans8r8o` | `10298.01` | `4872.70` | wins sim8, fails sim16 |
| `direct_ctree_gpu_latent` | `ap-hNe9labJXf6Z17LN5GNoJF` / `ap-OVE29tvXDfUEAGgVEdi37t` | `7567.35` | `6153.95` | practical baseline still wins sim16 |
| `recurrent_toy` ceiling | `ap-d5dqTBaaVlh6p7KZQT5lvG` / `ap-BkIEqGLr28TuETnXOb19Tf` | `9524.57` | `8969.89` | not MCTS; model-call ceiling sanity row |

Important telemetry:

- sim8 compile-spike reported `lightzero_array_ceiling_compile_enabled=1.0`
  and `compile_status=compiled_cached`.
- Both compile-spike rows emitted Torch warnings about skipped CUDA graphs due
  to mutated inputs in `select_leaf` and `expand_and_backup`.
- sim16 emitted repeated Torch Dynamo `recompile_limit` warnings keyed to
  `simulation_index` and the triangular `range(simulation_index + 1)` loop.

Plain read:

```text
This exact compile helper is not the 5-10x lane. It can look good at sim8, but
it fails the sim16 practical gate against direct_ctree_gpu_latent. The failure
is not rendering. The failure is the dynamic, Python-shaped search/update shell:
simulation-depth recompiles, in-place tree mutation, and many small pieces
around recurrent inference.
```

Decision:

```text
Stop polishing this exact compile-spike helper. Keep direct_ctree_gpu_latent as
the practical near-term baseline. The next radical test should either make the
search body fixed-sim/fixed-buffer enough to compile cleanly, or move to an
array-native/search-service design that owns batching more explicitly.
```

First full-loop H100 A/B, C64/sim8/no-RND/no-death/telemetry-stride-1:

| backend | run id | steps/sec | train wall | collector | policy collect | MCTS search | learner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | `opt-search-hook-ab-stock-20260522b` | `535.73` | `30.58s` | `21.61s` | `12.00s` | `6.55s` | `0.78s` |
| direct_ctree_gpu_latent | `opt-search-hook-ab-direct-20260522b` | `505.30` | `32.42s` | `21.57s` | `10.93s` | `5.47s` | `1.50s` |

Plain read:

```text
The hook is structurally real: it ran inside stock train_muzero and collected
the same 16,384 env steps with one learner call. It improved the search-ish
buckets, but not whole-loop wall time in this noisy sim8 profile. Do not claim
a training speedup from this row.
```

Current next falsifier:

```text
Run H100 C64/sim16/no-RND/no-death with sparse env telemetry, stock versus
direct_ctree_gpu_latent. If direct still does not win there, the current hook
is not removing the actual wall and the next optimization should target the
remaining collect/search Python/object/control boundary rather than CPU count
or render.
```

Instrumentation update:

```text
Compact output now reports direct-hook call count, fallback count, output rows,
CTree traverse/backprop counts, recurrent calls, recurrent batch mean, and
model-output D2H bytes/timers.
```

### Sim16/Sparse Follow-Up Wave

Status: profile-only. All rows used stock `train_muzero`, H100, no RND,
no-death, eval/GIF/checkpoint commit off, sparse env telemetry
(`env_telemetry_stride=256`), source-state fixed opponent, CPU-oracle policy
observation, browser-lines/simple-symbols observation surface.

| row | backend | C | sims | learner calls | env steps | steps/sec | wall | collect | policy collect | MCTS | learner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quick | stock | 64 | 16 | 1 | `16,384` | `387.23` | `42.31s` | `31.75s` | `21.60s` | `15.80s` | `1.32s` |
| quick | direct | 64 | 16 | 1 | `16,384` | `456.15` | `35.92s` | `25.72s` | `15.62s` | `10.37s` | `1.26s` |
| stable | stock | 64 | 16 | 3 | `16,384` | `205.55` | `79.71s` | `68.77s` | `49.45s` | `41.07s` | `1.19s` |
| stable | direct | 64 | 16 | 3 | `16,384` | `420.67` | `38.95s` | `22.74s` | `13.17s` | `8.38s` | `8.01s` |
| batch | stock | 128 | 16 | 3 | `32,768` | `477.68` | `68.60s` | `52.41s` | `31.87s` | `22.81s` | `1.78s` |
| batch | direct | 128 | 16 | 3 | `32,768` | `396.84` | `82.57s` | `61.28s` | `21.75s` | `11.54s` | `7.18s` |
| sparse-control | stock | 64 | 8 | 3 | `16,384` | `495.05` | `33.10s` | `21.76s` | `13.20s` | `7.04s` | `3.32s` |
| sparse-control | direct | 64 | 8 | 3 | `16,384` | `477.19` | `34.33s` | `24.81s` | `10.77s` | `5.07s` | `1.18s` |

Plain read:

```text
The direct hook consistently reduces the search-ish buckets. It is not a
general full-loop speedup yet.

It won the quick C64/sim16 row, but the first C64/sim16 three-learner stock row
was suspiciously slow. The repeat below is now the cleaner read.
```

Amdahl read:

```text
The current direct hook is a partial fix. It keeps latent states on GPU, but
still does per-simulation CPU/list work for CTree reward/value/policy logits
and still assembles one stock LightZero output dict per env id. In the direct
C64/sim16 stable row, direct sub-buckets include recurrent inference
`4.47s`, model-output D2H `2.58s`, and output assembly `2.69s`. At C128,
output assembly grows to `7.41s`.
```

Current best next target:

```text
Do not chase CPU count. Attack the remaining direct hook overhead:
model-output D2H/listifying, output assembly, and the CTree Python/list
boundary. C128 is not a recommendation from this wave.
```

### C64/Sim16 Repeat Result

Status: profile-only. Same denominator as the suspicious C64/sim16
three-learner row above: stock `train_muzero`, H100, no RND, no-death,
eval/GIF/checkpoint commit off, sparse env telemetry, source-state fixed
opponent, CPU-oracle policy observation, browser-lines/simple-symbols
observation surface.

| backend | run id | steps/sec | wall | collect | policy collect | MCTS | learner | direct D2H | direct output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | `opt-search-hook-ab-stock-c64-sim16-3learn-repeat-20260522b` | `445.19` | `36.80s` | `26.16s` | `17.07s` | `12.02s` | `3.10s` | n/a | n/a |
| direct | `opt-search-hook-ab-direct-c64-sim16-3learn-repeat-20260522b` | `438.56` | `37.36s` | `26.43s` | `15.95s` | `10.72s` | `1.61s` | `3.41s` | `2.87s` |

Validity:

```text
called_train_muzero=true
env_steps_collected=16384
learner_train_calls=3
direct_calls=256
direct_fallback_calls=0
direct_output_rows=16384
```

Corrected read:

```text
The earlier stock C64/sim16 three-learner row was anomalously slow. The repeat
does not support a 2x full-loop speedup. The direct hook still reduces the
MCTS bucket (`12.02s -> 10.72s`) and policy collect bucket (`17.07s ->
15.95s`), but that win is eaten by the remaining direct boundary costs:
model-output device-to-host/list conversion (`3.41s`) and stock output
assembly (`2.87s`).
```

Amdahl read:

```text
The thing to fix is not CPU count. The thing to fix is the LightZero
collect/search boundary shape: per-simulation CPU/list payloads into CTree and
per-env Python dict output fanout back to the collector. The current hook is a
useful probe, not a Coach-facing training recommendation.
```

### Output Assembly Fast Path

Status: profile-only. The patch preserves the stock per-env collect-output dict
shape, but uses an all-actions-legal fast path for CurvyTron's `[1,1,1]` masks.
It skips per-row `np.where(mask == 1)` work and vectorizes MCTS action sampling
from `roots.get_distributions()`.

Local validation:

```text
uv run python -m py_compile ...
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py
uv run ruff check --ignore F401 ...
```

Result: focused tests passed (`8 passed`); ruff passed. The F401 ignore is for
pre-existing launcher imports.

First remote row, same H100 C64/sim16/3-learner denominator:

| backend | run id | steps/sec | wall | collect | policy collect | MCTS | direct D2H | direct output | fast path calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct fast-output | `opt-search-hook-direct-outputfast-c64-sim16-3learn-20260522a` | `602.42` | `27.20s` | `18.31s` | `9.92s` | `7.80s` | `2.38s` | `0.063s` | `256` |

Plain read:

```text
This is the first direct-hook row that cleanly beats the latest stock repeat
(`445.19` steps/sec). The measured output-assembly tax dropped from `2.87s` to
`0.063s`. Because the result is good enough to be suspect, a fresh matched
stock/direct repeat is running before promotion.
```

Matched repeat:

| backend | run id | steps/sec | wall | collect | policy collect | MCTS | learner | direct D2H | direct output | fast path calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | `opt-search-hook-stock-c64-sim16-3learn-outputfast-denom-20260522b` | `433.17` | `37.82s` | `26.02s` | `17.10s` | `12.09s` | `4.17s` | n/a | n/a | `0` |
| direct fast-output | `opt-search-hook-direct-outputfast-c64-sim16-3learn-repeat-20260522b` | `566.19` | `28.94s` | `19.41s` | `10.31s` | `8.06s` | `1.03s` | `2.47s` | `0.077s` | `256` |

Corrected stable read:

```text
The output fast path converts the direct hook from flat/slower to a real
profile full-loop win for H100 C64/sim16/3-learner/no-RND/no-death:
about 1.31x over the matched stock row (`566.19 / 433.17`).

This is still profile-only and not a Coach default. It changes stochastic action
sampling implementation details inside the profile-only hook, so promotion
would need a clearer semantics/parity gate. But as an optimizer probe, it
confirms that per-env stock output assembly was one real wall.
```

New Amdahl read:

```text
Output assembly is no longer the wall. The remaining direct buckets are:
- MCTS/search: `8.06s`
- recurrent inference: `4.28s`
- model-output D2H/list conversion: `2.47s`
- stock collector/env/replay shell outside direct search still consumes the
  rest of the `28.94s` wall.

The next local target is model-output D2H/list conversion or a larger
array-native CTree boundary. CPU count remains parked.
```

### RND Meter Check

Status: profile-only. Same H100 C64/sim16/3-learner/no-death denominator, but
with `exploration_bonus_mode=rnd_meter_v0`, `weight=0.0`, and
`require_rnd_metrics=true`. This calls `train_muzero_with_reward_model`.

| backend | run id | steps/sec | wall | policy collect | MCTS | learner | RND train | RND hash | direct D2H | direct output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock | `opt-search-hook-stock-c64-sim16-3learn-rndmeter-20260522a` | `342.33` | `47.86s` | `22.14s` | `16.12s` | `1.37s` | `3.43s` | `2.92s` | n/a | n/a |
| direct fast-output | `opt-search-hook-direct-outputfast-c64-sim16-3learn-rndmeter-20260522a` | `410.55` | `39.91s` | `13.11s` | `10.36s` | `1.79s` | `3.55s` | `2.98s` | `3.25s` | `0.084s` |

RND validity:

```text
constructed=true
required=true
collect_data_calls=1
train_with_data_calls=1
train_cnt_rnd=100
estimate_calls=3
target_changed=false
predictor_changed=true
```

Plain read:

```text
The output-fast direct hook still wins with the current RND meter path, but the
gain shrinks to about 1.20x (`410.55 / 342.33`). This is expected: RND adds
work that the search hook does not touch, especially RND training and state
hashing. The search-boundary win is real but not a complete training-system
fix.
```

Caveat:

```text
These RND rows report env steps through the profile fallback
`mcts_search_root_sum_profile_fallback`; the reward-model entrypoint did not
populate the collector env-step counter the same way as the no-RND row. Both
stock and direct used the same fallback denominator.
```

### Fresh Current-Telemetry Direct CTree Refresh

Status: profile-only. These rows did not call `train_muzero`, did not touch
live Coach runs, and did not write checkpoint/eval/GIF/tournament artifacts.

Why this rerun exists:

```text
The earlier 60/15 rows predated the newer transfer/freshness/model-output D2H
accounting. This rerun used the same H100 B512/A16/sim8 shape with 60 measured
steps and 15 warmup steps, then compared stock facade, direct CTree host input,
direct CTree pinned input, and the stale resident ceiling.
```

Modal app ids from the batch, not variant-resolved from stdout:

```text
ap-tO4Vzm1aWiXdxHLKZuaB2W
ap-bGGcJAUkKZ7z3VUnkOjHiL
ap-097VHQnzD219MW346jePLm
ap-BKT1g9k1EoyHdYq4PouTg9
```

The stdout JSON is the source of truth for the row labels below.

| row | input freshness | scalar roots/sec | measured sec | boundary total | H2D | input prep/pin | model-output D2H | search | root prep | model total | observation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock facade, `host_uint8` | fresh | `2473.11` | `24.84s` | `20.22s` | `1.04s` | `0.00s` | n/a | stock wrapper | n/a | `0.96s` | `2.08s` |
| direct CTree, `host_uint8` | fresh | `4564.03` | `13.46s` | `8.48s` | `1.14s` | `0.00s` | `0.10s` | `6.26s` | `0.49s` | `0.83s` | `2.23s` |
| direct CTree, `host_uint8_pinned` | fresh | `4113.52` | `14.94s` | `9.38s` | `0.05s` | `0.07s` | `0.12s` | `8.11s` | `0.53s` | `1.09s` | `2.46s` |
| direct CTree, `resident_torch_reuse` | stale ceiling | `4884.69` | `12.58s` | `7.56s` | `0.00s` | `0.00s` | `0.13s` | `6.50s` | `0.46s` | `0.90s` | `2.21s` |

Plain read:

```text
The current direct CTree profile lane is about 1.85x faster than the stock
facade in the fresh host-input row.

Pinned input is not a reliable total-wall win in this refresh. It cuts H2D
hard, but the run spent more time in search/model/observation variance and
ended slower than plain host_uint8.

Resident reuse is only about 1.07x above fresh direct host in this refresh.
That means input transfer is not the main remaining wall.
```

Amdahl read:

```text
The current speed probe has removed public collect-output/fanout overhead, but
the remaining wall is still real CTree search/root-prep/model-output handling
plus ordinary observation/stack work. The direct path is promising enough to
keep validating, but not enough to advise Coach until the promotion contract
passes and a matched full-loop gate exists.
```

### 2026-05-22 Same-Denominator Sim16 Dense Search Check

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

Common shape:

```text
H100
B512 physical rows
1024 player-view roots per call
sim16
60 measured steps
15 warmup steps
uint8 stack storage
scalar timestep materialization off
root noise weight 0.0
```

| impl | measured sec | scalar roots/sec | probe/boundary sec | search/update sec | model/recurrent sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_torch_mcts` cleanup v2 | `14.857` | `4135.37` | `5.262` | `2.514` | `1.556` | `2.394` |
| `direct_ctree_gpu_latent` | `12.263` | `5010.39` | `6.766` | `4.527` | `1.523` | `2.480` |
| `recurrent_toy` ceiling | `6.727` | `9133.74` | `2.416` | `0.143` | `1.234` | `2.089` |

Plain read:

```text
The second-cleanup dense GPU tree was promising at sim8 but bad at sim16.
The direct CTree GPU-latent path is still the tactical baseline until dense
search proves it scales with simulation count.
```

Follow-up already applied:

```text
The dense profile code now uses fixed-shape all-root masked
selection/expansion/backprop for the common all-roots case. Local ruff,
py_compile, and focused tests passed. H100 sim8/sim16 repeats are running.
```

## 2026-05-21 Fresh L4 Current-Batch Anchor Grid

Status: profile-only. No live Coach training run was changed. Rows used
`mode=profile`, no eval/GIF/checkpoint I/O in the speed denominator, no-death,
`gpu-l4-t4-cpu40`, C256/N256, batch64, sim8, and `source_max_steps=512`.

Artifact:

```text
artifacts/local/curvytron_optimizer_profile_results/opt-current-l4-c256-ab-rnd-20260521a/
```

Rows:

| row | manager | RND | steps/sec | wall sec | key read |
| --- | --- | --- | ---: | ---: | --- |
| 001 | `subprocess` CPU oracle | none | `641.10` | `204.45` | current L4 CPU-oracle profile anchor |
| 002 | `subprocess` CPU oracle | `rnd_meter_v0` | `651.80` | `201.09` | RND meter not slower in this one row; treat as noisy |
| 003 | `curvyzero_batched_profile` | none | `942.63` | `139.05` | batched GPU observation beat CPU oracle by about `1.47x` |
| 004 | `curvyzero_batched_profile` | `rnd_meter_v0` | `618.82` | `211.81` | RND + batched GPU contended hard; GPU max hit `97%` |
| 005 | `curvyzero_batched_zero_obs_profile` | none | `769.33` | `170.37` | zero-observation was slower than real-render; not a clean ceiling |
| 006 | `curvyzero_batched_zero_obs_profile` | `rnd_meter_v0` | `748.00` | `175.23` | similar zero-observation RND row |

Important timers:

- Row 003 batched GPU no-RND: manager step `38.88s`, renderer `26.63s`,
  device render `16.02s`, surface stack update `29.30s`, MCTS `38.17s`,
  policy collect `79.94s`.
- Row 004 batched GPU with RND: manager step `72.11s`, renderer `53.07s`,
  device render `38.47s`, surface stack update `58.25s`, RND train `7.42s`,
  RND state hash `7.18s`, GPU max `97%`.
- Row 005 zero-observation no-RND: MCTS `62.11s`, policy collect `122.60s`,
  which explains why it did not act like a clean upper bound.

Plain read:

```text
The batched GPU observation profile path can beat the current L4 CPU-oracle
profile anchor by ~1.5x in a matched no-RND row.

But this grid also says the next wall is not only observation. MCTS/policy
timing and GPU scheduling variance are large enough that zero-observation did
not form a stable ceiling on this pass.
```

Do not translate this directly into actual Coach training speed. The latest
actual Coach batch still used CPU oracle and measured about `18.4k` mean /
`19.7k` median checkpoint iterations/hour. This grid is the right next profile
anchor, not a production-speed proof.

### Repeat Grid

Follow-up artifact:

```text
artifacts/local/curvytron_optimizer_profile_results/opt-current-l4-c256-ab-repeat-20260521a/
```

Same shape, no-RND only, seeds `305` and `306`.

| seed | subprocess CPU oracle | batched GPU obs | zero observation |
| ---: | ---: | ---: | ---: |
| 305 | `508.66` | `792.94` | `1038.72` |
| 306 | `608.67` | `945.07` | `1085.13` |

Combining the first no-RND row plus the two repeats:

| manager | n | mean steps/sec | min | max |
| --- | ---: | ---: | ---: | ---: |
| `subprocess` CPU oracle | 3 | `586.14` | `508.66` | `641.10` |
| `curvyzero_batched_profile` | 3 | `893.55` | `792.94` | `945.07` |
| `curvyzero_batched_zero_obs_profile` | 3 | `964.39` | `769.33` | `1085.13` |

Updated read:

```text
batched GPU obs vs CPU oracle: ~1.52x mean speedup in this current L4/C256
profile shape.

zero obs vs batched GPU obs: only ~1.08x mean headroom when including the noisy
first zero row; repeat-only headroom is closer to ~1.22x.
```

This makes the practical bottleneck clearer. The current batched observation
profile is already close to the zero-observation ceiling in this one-process
profile shape. The next large gains are unlikely to come from shaving only the
renderer. They need better policy/search/manager topology, or a real training
integration that preserves the batched boundary without extra scalar overhead.

## 2026-05-21 LightZero Collect-Forward Real-Consumer Smoke

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

Command shape:

```text
source_state_batched_observation_boundary_profile
--hybrid-observation-canary
--hybrid-lightzero-collect-forward-probe
--compute gpu-l4-t4
--batch-size 4
--actor-count 2
--steps 2
--warmup-steps 1
--render-surface direct_gray64
--observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile
--hybrid-stack-storage-dtype uint8
--no-hybrid-materialize-scalar-timestep
--hybrid-lightzero-consumer-num-simulations 1
```

Important setup note:

- Letting `torch>=2.2` float installed a CUDA13 Torch wheel and broke against
  the JAX CUDA12 image.
- Pinning Torch too old (`2.5.1`) brought cuDNN `9.1`, below JAX `0.7.0`'s
  cuDNN `9.8+` expectation.
- The profile image now pins `torch==2.8.0`, which keeps CUDA12 and cuDNN
  `9.10`.

Result:

| field | value |
| --- | ---: |
| `ok` | `true` |
| `calls_train_muzero` | `false` |
| `materialized_timestep_count` | `0` |
| `batched_stack_probe_backend_name` | `lightzero_collect_forward_consumer` |
| `batched_stack_probe_semantics` | `lightzero_collect_forward_search_cpu_tree` |
| `lightzero_policy_device` | `cuda:0` |
| `lightzero_root_count` | `8` per call |
| `lightzero_policy_forward_calls` | `1` per measured step |
| `lightzero_illegal_action_count` | `0` |
| `lightzero_consumer_total_sec` | about `12.8ms` for the last tiny call |

Plain read:

```text
The real-consumer canary is wired. It can feed the pre-scalar resident batch
into actual LightZero MuZero collect-mode policy/search without building scalar
LightZero timesteps first.
```

Caveat:

```text
This is only a wiring smoke. It is too small and too cold-start-heavy to use as
a throughput conclusion. The next useful rows are B512/A16/sim8, H100 and L4,
scalar edge off/on.
```

## 2026-05-21 LightZero Collect-Forward Medium Rows

Status: profile-only. These rows do not call `train_muzero`, do not touch live
training, and do not write checkpoint/eval/GIF artifacts. They measure the
pre-scalar GPU observation stack consumed by real `MuZeroPolicy.collect_mode.forward`.

Shape:

```text
B512 physical rows
2 player roots per row = 1024 roots per collect-forward call
20 measured steps, 5 warmup steps
sim8
uint8 [B,2,4,64,64] stack
jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
```

Run registry:

| row | run id | compute | scalar LightZero timestep materialization | status |
| --- | --- | --- | --- | --- |
| H100 off | `ap-W4oKFCjUlsxf4iKThO1dsl` | H100 | off | recovered from pre-compaction output |
| H100 on | `ap-YeL8fNtJLdaXK0PWp2kCa4` | H100 | on | rerun captured |
| L4 off | `ap-krlU7DlOiIjJ3QaBReoYJj` | L4/T4 | off | rerun captured |
| L4 on | `ap-8Ky9lfBKWbvWkP6NPAKwSy` | L4/T4 | on | rerun captured |

Main table:

| compute | scalar edge | scalar roots/sec | physical rows/sec | measured sec | materialized timesteps | last collect-forward call |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100 | off | `2669.32` | `1334.66` | `7.67` | `0` | `0.284s` |
| H100 | on | `2100.31` | `1050.16` | `9.75` | `20480` | `0.377s` |
| L4/T4 | off | `2159.35` | `1079.67` | `9.48` | `0` | `0.419s` |
| L4/T4 | on | `2053.57` | `1026.78` | `9.97` | `20480` | `0.410s` |

Key timing buckets:

| compute | scalar edge | probe sec | collect/search device sec | H2D sec | decode+readback sec | observation sec | render sec | scalar materialization sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | off | `6.13` | `5.41` | `0.40` | `0.31` approx | `0.64` | `0.26` | `0.00` |
| H100 | on | `7.77` | `7.00` | `0.34` | `0.36` approx | `0.47` | `0.21` | `0.79` |
| L4/T4 | off | `8.50` | `7.85` | `0.28` | `0.40` approx | `0.36` | `0.17` | `0.00` |
| L4/T4 | on | `8.36` | `7.70` | `0.28` | `0.38` approx | `0.37` | `0.19` | `0.58` |

Comparison to the synthetic resident probe:

| compute | scalar edge | synthetic resident roots/sec | real collect-forward roots/sec | real / synthetic |
| --- | --- | ---: | ---: | ---: |
| H100 | off | `10980.47` | `2669.32` | `24%` |
| H100 | on | `7620.76` | `2100.31` | `28%` |
| L4/T4 | off | `5839.67` | `2159.35` | `37%` |
| L4/T4 | on | `4133.28` | `2053.57` | `50%` |

Plain read:

```text
The real LightZero collect-forward path can consume the pre-scalar batch, but
it collapses most of the synthetic resident speedup. The hot bucket is now the
real collect/search call, not rendering.
```

Amdahl read:

```text
In these rows, making render free would barely move the total. H100 scalar-off
spent about 0.26s rendering inside a 7.67s measured window. L4 scalar-off spent
about 0.17s rendering inside 9.48s. The bigger wall is the public LightZero
collect-forward/search path, which still crosses into CPU tree/search work.
```

Important caveat:

```text
This is still not a training-speed proof. It is the right boundary probe: real
LightZero policy/search pressure, no scalar timestep materialization in the
primary row, but no replay target build, learner update, RND, normal death, or
full stock train_muzero loop.
```

Implementation critique follow-up:

- The canary now fails closed for this intended row shape:
  `uint8` stack, `direct_gray64`, and
  `jax_gpu_persistent_policy_framebuffer_profile`.
- It now uses the fixed-opponent scalar convention for `to_play`: `-1` for each
  root. Player ids are still recorded as metadata, but not passed as
  board-game player turns.
- It filters all-zero action-mask roots before calling LightZero, so terminal
  or inactive rows do not hit collect-forward with no legal action.
- Telemetry now records that `collect_forward` includes CPU tree/search work.
  Do not read `batched_stack_probe_device_sec` as pure GPU time for this probe.

## 2026-05-21 Search/HW Sweep

Status: profile-only. No live Coach training run was changed. Rows used
`mode=profile`, no eval/GIF/checkpoint I/O in the speed denominator, no-death,
C256/N256, batch64, no-RND, and `source_max_steps=512`.

Artifacts:

```text
artifacts/local/curvytron_optimizer_profile_results/opt-current-search-hw-sweep-20260521a/
artifacts/local/curvytron_optimizer_profile_results/opt-current-search-h100-sweep-20260521b/
```

The first grid used the bad shorthand `gpu-h100`; those H100 rows failed before
launch. `scripts/build_curvytron_profile_grid.py` now canonicalizes `gpu-h100`
to `gpu-h100-cpu40` so future grids do not waste those rows.

| compute | manager | sims | steps/sec | wall sec | policy collect | MCTS | manager step | render |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L4/T4+40CPU | real batched obs | 4 | `996.14` | `131.58` | `65.84` | `23.21` | `42.08` | `29.18` |
| L4/T4+40CPU | real batched obs | 8 | `627.49` | `208.88` | `119.66` | `59.46` | `56.66` | `36.17` |
| L4/T4+40CPU | real batched obs | 16 | `618.76` | `211.83` | `142.69` | `98.05` | `44.33` | `29.13` |
| L4/T4+40CPU | zero obs | 4 | `1353.10` | `96.87` | `64.57` | `21.52` | `12.79` | `0.07` |
| L4/T4+40CPU | zero obs | 8 | `1067.44` | `122.79` | `88.44` | `45.45` | `15.40` | `0.06` |
| L4/T4+40CPU | zero obs | 16 | `777.75` | `168.53` | `133.76` | `89.45` | `15.64` | `0.06` |
| H100+40CPU | real batched obs | 4 | `1341.50` | `97.71` | `40.35` | `14.20` | `35.69` | `24.66` |
| H100+40CPU | real batched obs | 8 | `659.06` | `198.88` | `116.55` | `63.83` | `54.44` | `37.92` |
| H100+40CPU | real batched obs | 16 | `710.59` | `184.46` | `113.98` | `81.60` | `43.82` | `28.88` |
| H100+40CPU | zero obs | 4 | `1676.20` | `78.20` | `47.38` | `18.23` | `10.65` | `0.06` |
| H100+40CPU | zero obs | 8 | `1413.10` | `92.76` | `62.25` | `33.77` | `12.03` | `0.06` |
| H100+40CPU | zero obs | 16 | `866.49` | `151.27` | `115.81` | `83.35` | `15.31` | `0.06` |

Plain read:

```text
H100 helps at sim4, but it does not make sim8/sim16 scale cleanly.
Zero-observation is faster, but it still spends most wall time in collect/search.
Therefore the next large win is not another isolated render tweak. It is
batch/search/manager topology.
```

Caveat: `policy_forward_collect` includes nested MCTS/model work. Do not add
policy and MCTS timers as independent wall buckets.

### Split-Timer Instrumentation Smoke

Tiny Modal smoke:

```text
opt-instrument-smoke-20260521a / profile-c2-sim1
```

Read: the run reached compact output and reported the new split timers, then
failed at learner time because `batch_size=1` is too small for the LightZero
training layer (`Expected more than 1 value per channel...`). Treat this as an
instrumentation-output pass and a deliberately invalid learner-shape smoke, not
a training failure.

New compact timer families now visible:

- `batched_profile_bridge_*`: action sort, joint fill, loop step, ready obs,
  timestep split, stock timestep conversion.
- `batched_profile_surface_package_*`: mask copies, live mask, policy rows,
  policy observation, action mask, final observation, info dict, output copy.

### Split-Timer HW/C Width Grid

Artifact:

```text
artifacts/local/curvytron_optimizer_profile_results/opt-split-timer-hw-c256-c512-20260521a/
```

Rows used `mode=profile`, no eval/GIF/checkpoint I/O in the speed denominator,
no-death, no-RND, batch64, C256/C512, sim4/sim8, L4/H100, and real-vs-zero
batched managers.

| compute | manager | C | sims | steps/sec | wall sec | policy collect | MCTS | manager step | render | stack |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L4/T4+40CPU | real batched obs | 256 | 4 | `1016.41` | `128.96` | `63.99` | `21.70` | `42.27` | `28.72` | `31.91` |
| L4/T4+40CPU | real batched obs | 256 | 8 | `859.26` | `152.54` | `84.36` | `41.58` | `44.66` | `29.12` | `32.91` |
| L4/T4+40CPU | real batched obs | 512 | 4 | `1220.44` | `214.79` | `112.76` | `31.94` | `64.24` | `39.99` | `45.64` |
| L4/T4+40CPU | real batched obs | 512 | 8 | `1077.35` | `243.32` | `142.86` | `58.82` | `63.77` | `39.30` | `44.77` |
| L4/T4+40CPU | zero obs | 256 | 4 | `1367.11` | `95.88` | `65.15` | `22.36` | `11.65` | `0.06` | `2.53` |
| L4/T4+40CPU | zero obs | 256 | 8 | `1082.23` | `121.11` | `86.30` | `43.55` | `15.80` | `0.08` | `3.81` |
| L4/T4+40CPU | zero obs | 512 | 4 | `1502.57` | `174.46` | `116.64` | `31.99` | `22.95` | `0.11` | `4.94` |
| L4/T4+40CPU | zero obs | 512 | 8 | `1263.33` | `207.50` | `151.07` | `67.69` | `23.69` | `0.10` | `4.33` |
| H100+40CPU | real batched obs | 256 | 4 | `1136.17` | `115.36` | `49.13` | `18.46` | `41.40` | `28.55` | `31.86` |
| H100+40CPU | real batched obs | 256 | 8 | `881.76` | `148.65` | `77.99` | `44.37` | `45.24` | `30.03` | `34.12` |
| H100+40CPU | real batched obs | 512 | 4 | `1228.92` | `213.31` | `92.14` | `34.25` | `70.50` | `41.85` | `49.37` |
| H100+40CPU | real batched obs | 512 | 8 | `1378.16` | `190.21` | `95.96` | `44.10` | `60.13` | `36.76` | `42.99` |
| H100+40CPU | zero obs | 256 | 4 | `1133.02` | `115.68` | `78.28` | `26.27` | `14.05` | `0.06` | `2.45` |
| H100+40CPU | zero obs | 256 | 8 | `1110.08` | `118.07` | `81.85` | `47.59` | `15.91` | `0.06` | `3.98` |
| H100+40CPU | zero obs | 512 | 4 | `1805.32` | `145.21` | `84.55` | `27.38` | `23.20` | `0.11` | `6.16` |
| H100+40CPU | zero obs | 512 | 8 | `1608.35` | `162.99` | `105.20` | `48.72` | `24.59` | `0.12` | `5.83` |

Key ratios:

- L4 C512 sim8 zero/real: `1263 / 1077 = 1.17x`.
- H100 C512 sim8 zero/real: `1608 / 1378 = 1.17x`.
- H100 C512 sim8 real/L4 C512 sim8 real: `1378 / 1077 = 1.28x`.

Split-timer read:

- Bridge object churn is visible but not the main wall. At C512 sim8 it is
  single-digit seconds for ready-obs/timestep split/conversion, while policy
  collect is about `96-143s` and manager/render/stack is about `60-64s` in
  real-render rows.
- Surface package sub-buckets are tiny in this shape. The large surface bucket
  is stack/render, not `_info` or policy-row selection.
- Zero-observation rows still spend most of their wall in policy collect and
  MCTS/search. That is the decisive Amdahl signal.

Updated conclusion:

```text
Pure render/observation work still matters, but the C512 sim8 render-only
headroom is about 1.17x. The next multi-x target must preserve larger batches
through policy/search/collector or prototype a central actor/search batching
shape.
```

## 2026-05-21 Persistent GPU Framebuffer Pass

Status: profile-only. No trainer, tournament, checkpoint, or live-run default
was changed.

Backend label:

```text
jax_gpu_persistent_policy_framebuffer_profile
```

This is a direct 64x64 policy-space framebuffer. It is not browser-pixel
parity and it is not the production observation default. The point of this pass
is to test the best-known GPU-rendering direction without touching Coach runs.

### Surface Facade H100 Rows

Matched B512/100 comparison, no death, no CPU reference verification:

| backend | total step median | render median | device render median | stack update median | env step median | read |
|---|---:|---:|---:|---:|---:|---|
| dynamic direct64 | `81.24ms` | `38.70ms` | `7.86ms` | `69.64ms` | `10.81ms` | redraws trail history |
| persistent direct64 | `44.61ms` | `9.99ms` | `0.17ms` | `33.34ms` | `10.82ms` | updates only new trail deltas |

Speed read:

- surface step: about `1.82x` faster;
- renderer bucket: about `3.87x` faster;
- device render kernel: effectively no longer the wall for this shape.

Longer B512/500 persistent row:

| bucket | median |
|---|---:|
| total step | `78.94ms` |
| render | `15.64ms` |
| device render | `0.17ms` |
| stack update | `38.90ms` |
| env step | `38.95ms` |

Read: the persistent renderer paid out. At longer trajectories, Amdahl moves
to two buckets: host-side float32 stack update and env body/collision work.
Further render-kernel work cannot give a large full-loop win by itself in this
profile shape.

### Current Follow-Up Row

After patching collision broad-phase to scan only live body cursor prefixes,
rerunning the same B512/500 H100 persistent surface row:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --compute gpu-h100 --surface-facade-canary \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --render-surface direct_gray64 --batch-size 512 --steps 500 \
  --warmup-steps 20 --verify-steps 0 --cpu-reference-interval 0 \
  --trail-slots 2048 --body-capacity 2048 --max-ticks 2000
```

Result:

- `ok=true`
- total surface step median `65.70ms`
- env step median `13.75ms`
- stack update median `44.92ms`
- render median `16.02ms`
- device render median `0.21ms`
- active trail count median `534`, p95 `948`
- persistent delta slots median `889.5`, p95 `1024`

Comparison with the pre-patch B512/500 persistent row:

| bucket | before | after | read |
|---|---:|---:|---|
| total step | `78.94ms` | `65.70ms` | about `1.20x` faster |
| env step | `38.95ms` | `13.75ms` | about `2.83x` faster |
| render | `15.64ms` | `16.02ms` | unchanged/noisy |
| stack update | `38.90ms` | `44.92ms` | now the main wall |

Read: the cursor-bound collision scan is a real env win. It does not create a
large full-loop win by itself because host stack update/materialization now
dominates the surface row.

### Failed Long Full-Manager/RND Attempt

A profile-env-manager + RND row was started synchronously, but Modal stopped it
when the local client heartbeat disconnected before returning a result. Treat
that as an orchestration/tooling failure, not an env or RND failure. Longer
manager/RND rows should use detached Modal jobs or a result artifact path.

## 2026-05-21 Hybrid Device-Stack Probe

Purpose: test the next Amdahl wall without changing trainer defaults. This is a
profile-only hybrid canary, not stock `train_muzero`.

Common shape:

```text
H100
B512 / actor_count=16
steps=100, warmup_steps=20
renderer=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
hybrid_batched_stack_probe_simulations=4
hybrid_batched_stack_probe_channels=16
no death
```

Rows:

| stack dtype | scalar materialization | steps/sec | physical rows/sec | observation sec | probe H2D sec | scalarization sec | read |
|---|---:|---:|---:|---:|---:|---:|---|
| `uint8` | off | `16309.79` | `8154.90` | `2.748s` | `0.178s` | `0.000s` | best profile-only device-stack shape |
| `uint8` | on | `9801.46` | `4900.73` | `2.947s` | `0.196s` | `3.590s` | scalarization is a real wall |
| `float32` | off | `9208.00` | `4604.00` | `5.978s` | `0.876s` | `0.000s` | float32 stack bytes are expensive |

Byte read:

- `uint8` stack per step: `16.8MB`.
- `float32` stack per step: `67.1MB`.
- The batched probe H2D bucket was about `0.178-0.196s` total for `uint8`
  across 100 measured steps, versus `0.876s` for `float32`.

Plain read:

```text
The next real speed lane is not more isolated rendering.
It is preserving a compact uint8 batched stack until a GPU consumer can
normalize/use it, and avoiding scalar LightZero materialization until the edge.
```

Caveats:

- This does not call `train_muzero`.
- The stack probe is synthetic; it is not LightZero MCTS.
- Actors are still in-process partitions, not subprocess IPC.
- Persistent renderer is policy-space direct64, not browser-pixel parity.

### Device-Latest Probe Variant

Added an explicit profile-only flag:

```text
--hybrid-batched-stack-probe-device-latest
```

It requires the persistent renderer. The probe reads the renderer's latest JAX
device frame and maintains its own device stack, so the probe no longer uploads
the full host stack. This is still no-death/profile-only; terminal reset
semantics are not claimed.

Matched row:

| stack dtype | scalar materialization | device-latest | steps/sec | observation sec | probe H2D sec | device stack update in last step |
|---|---:|---:|---:|---:|---:|---:|
| `uint8` | off | no | `16309.79` | `2.748s` | `0.178s` | `0.000s` |
| `uint8` | off | yes | `11595.65` | `4.039s` | `0.042s` | last step `0.0010s` |

Read:

- The device-latest variant correctly reduced the synthetic probe's H2D bucket.
- It did not improve total throughput, because the manager still maintained the
  host stack and the probe added a second device stack.
- Therefore the next real implementation must replace the host stack in the
  no-scalar profile lane. Adding a device stack beside the host stack is not
  the optimization.

### Sim8 Resident-Chunk Repeat

Status: profile-only. These rows do not touch live runs and do not call stock
`train_muzero`. They use the existing hybrid observation canary with the
persistent direct64 GPU renderer, `uint8` stack storage, B512/A16, 100 measured
steps, 20 warmup steps, and a synthetic batched stack consumer with 8
simulations and 16 channels.

Commands used the boundary app's native compute names, not profile-grid
aliases:

```text
--compute gpu-h100
--compute gpu-l4-t4
```

The older doc snippet that used `gpu-h100-cpu40` is wrong for this Modal
entrypoint; that name belongs to the train-profile grid builder.

| compute | scalar edge | device-latest | steps/sec | physical rows/sec | measured sec | observation sec | renderer sec | probe sec | scalarization sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | off | no | `13773.66` | `6886.83` | `7.43` | `3.24` | `1.65` | `0.70` | `0.00` |
| H100 | on | no | `6487.59` | `3243.80` | `15.78` | `4.57` | `2.35` | `0.95` | `5.05` |
| L4/T4 | off | no | `8970.81` | `4485.40` | `11.41` | `3.68` | `2.14` | `3.48` | `0.00` |
| L4/T4 | on | no | `4221.22` | `2110.61` | `24.26` | `6.29` | `3.09` | `3.66` | `6.24` |
| H100 | off | yes | `9762.40` | `4881.20` | `10.49` | `4.65` | `2.37` | `0.57` | `0.00` |

Read:

```text
The resident chunk canary gives a much faster profile-only ceiling than the
stock-shaped loop. The important win is not just GPU rendering; it is keeping
the B512 x 2 x 4 x 64 x 64 uint8 stack as a batch and avoiding scalar
LightZero-shaped rows until the edge.
```

Concrete ratios:

- H100 scalar-off versus L4 scalar-off: about `1.54x`.
- H100 scalar-on versus H100 scalar-off: about `0.47x` throughput, so scalar
  materialization is expensive.
- L4 scalar-on versus L4 scalar-off: about `0.47x` throughput, same story.
- Device-latest reduced the probe H2D bucket, but total throughput still
  dropped from about `13.8k` to about `9.8k` roots/sec because the host stack
  is still maintained and the device stack is extra work.

This does not prove a trainer speedup. It proves that a batched resident
boundary is worth pursuing. The next falsification gate is a stock-boundary or
custom-collector canary that keeps this batched shape alive while adding real
policy/search/replay pressure.

## Completed This Pass

### Boundary H100 Float32 Speed-Only Row

Config:

```text
batch_size=64
compute=gpu-h100
steps=8
warmup_steps=4
trail_slots=1024
verify_steps=0
geometry_dtype=float32
max_ticks=2000
```

Result:

- `ok=true`
- candidate observation median about `0.255s`
- candidate step+observation median about `0.258s`
- CPU reference render+stack median about `0.744s`
- device render median about `0.244s`

Read: about `2.9x` faster than the CPU reference render+stack boundary for this
profile row. This is not a full-loop speedup.

### Boundary H100 Float64 Exact Row

Config:

```text
batch_size=64
compute=gpu-h100
steps=6
warmup_steps=3
trail_slots=1024
verify_steps=2
geometry_dtype=float64
max_ticks=2000
```

Result:

- `ok=true`
- exact reset/step parity
- candidate observation median about `0.376s`
- CPU reference render+stack median about `0.684s`

Read: exact debug mode is slower than float32, but useful for diagnosis.

### RND CPU-Oracle Smoke With Batch Size 1

Config: profile mode, L4/T4, CPU oracle, `rnd_meter_v0`, one collector env, one
episode, learner `batch_size=1`.

Result:

- failed inside `train_muzero_with_reward_model`
- error: `Expected more than 1 value per channel when training`

Read: this does not condemn RND or CPU oracle. It means RND smokes need a small
real batch, not batch size 1.

Local gates after adding explicit parity modes:

```text
uv run pytest tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_profile_cpu.py \
  tests/test_exploration_bonus.py -q
```

Result: `46 passed, 1 skipped`.

### Boundary H100 Float32 Tolerant Verified Row

Config:

```text
batch_size=64
compute=gpu-h100
steps=8
warmup_steps=4
trail_slots=1024
verify_steps=4
geometry_dtype=float32
parity_mode=tolerant
parity_max_abs_diff=2
parity_max_mismatch_fraction=0.0001
max_ticks=2000
```

Result:

- `ok=true`
- checked reset plus 4 steady steps
- raw frames/stacks not exact, but tolerated
- total mismatch count `5`
- max raw-frame diff `1`
- candidate observation median about `0.255s`
- candidate step+observation median about `0.259s`
- CPU reference render+stack median about `0.690s`

Read: the aggressive float32 boundary can run real step verification without
turning parity off. The observed drift is the known one-luma edge pixel
propagating through stack history, not a shape/order/view failure.

### RND CPU-Oracle Smoke With Batch Size 2

Config: profile mode, L4/T4, CPU oracle, `rnd_meter_v0`, two collector envs,
two episodes, learner `batch_size=2`, one learner train call.

Result:

- `ok=true`
- entrypoint `lzero.entry.train_muzero_with_reward_model`
- `learner_train_calls=1`
- `replay_sample_calls=1`
- RND metrics exist
- predictor changed, target stayed frozen
- `last_target_reward_changed=false`
- `last_target_reward_delta_abs_max=0.0`
- train wall about `11.73s`

Read: RND meter mode is compatible with the current CPU-oracle stock profile
path when the smoke uses a real batch. This proves plumbing, not learning.

### Coarse Host-Overhead Ladder

Run id family: `opt-host-overhead-ladder-20260520b`.

Config: L4/T4+40CPU, stock LightZero profile path, CPU oracle policy
observation, no death, sim8, batch32, no RND/eval/GIF, four learner calls.

| manager | collectors | steps | wall sec | steps/sec |
|---|---:|---:|---:|---:|
| base | 1 | 512 | 24.04 | 21.30 |
| base | 8 | 4096 | 63.34 | 64.67 |
| base | 32 | 16384 | 184.16 | 88.97 |
| base | 64 | 32768 | 337.94 | 96.96 |
| subprocess | 1 | 512 | 25.02 | 20.46 |
| subprocess | 8 | 4096 | 29.71 | 137.86 |
| subprocess | 32 | 16384 | 50.00 | 327.65 |
| subprocess | 64 | 32768 | 62.13 | 527.41 |

Read:

- C1 subprocess is slightly slower than base, so process overhead is real when
  there is no parallel work to hide it.
- C8/C32/C64 subprocess is much faster than base. At C64 it is about `5.4x`
  faster than base for the same 32768 no-death env steps.
- Full-loop Amdahl for the current stock path points at collection/process
  architecture first, not renderer-only work.
- These rows were spawned before the new BaseEnvTimestep payload timing patch,
  so they are throughput evidence but not a payload-byte diagnosis.

## Pending Runs

### Stock-Boundary Batched GPU Env-Manager Canary

Run family: `opt-batched-stock-canary-20260520a`.

First successful stock-boundary row:

```text
attempt=envmgr-b16-sim2e
compute=gpu-h100-cpu40
env_manager_type=curvyzero_batched_profile
collector_env_num=16
n_episode=16
source_max_steps=1024
num_simulations=2
batch_size=32
RND=off
eval/GIF=off
death disabled for profile
```

Result:

- `ok=true`
- `called_train_muzero=true`
- `env_steps_collected=16384`
- `mcts_search_calls=1024`
- `mcts_search_root_sum=16384`
- `replay_sample_calls=1`
- `learner_train_calls=1`
- H100 max memory about `61979 MiB`
- H100 max util about `77%`
- `steps_per_sec=150.19`
- `collector_collect=102.70s`
- `policy_forward_collect=17.17s`
- `mcts_search=6.77s`
- `learner_train=1.18s`

Read:

- This is the first proof that the profile-only batched direct GPU observation
  bridge can cross into stock LightZero collection/search/replay/learner.
- This is not a speed win yet. At C16/sim2/no-death, collection dominates wall
  time and throughput is far below the best known stock subprocess CPU-oracle
  rows. The next comparison must use matched controls and larger batch/root
  topology before recommending this path.
- The failed earlier attempts were useful plumbing gates:
  `envmgr-b16-sim2b` died before `train_muzero` on compile-config manager type;
  `envmgr-b16-sim2c` entered `train_muzero` but died on manager registry lookup;
  `envmgr-b16-sim2d` reached model/search then died because step timesteps were
  missing `to_play`.

Matched controls and topology follow-up:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | policy forward sec | MCTS sec | learner sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpuoracle-base-c16-sim2-control` | base CPU oracle | 16 | 16384 | 2 | `98.01` | `167.16` | `161.36` | `16.48` | `5.67` | `3.25` |
| `envmgr-b16-sim2e` | profile-only batched GPU manager | 16 | 16384 | 2 | `150.19` | `109.09` | `102.70` | `17.17` | `6.77` | `1.18` |
| `envmgr-b64-sim2` | profile-only batched GPU manager | 64 | 65536 | 2 | `416.89` | `157.20` | `150.12` | `29.08` | `9.72` | `1.07` |
| `cpuoracle-subproc-c64-sim2-control` | subprocess CPU oracle | 64 | 65536 | 2 | `883.03` | `74.22` | `66.05` | `28.04` | `8.39` | `0.95` |

Read:

- The profile-only batched GPU manager crossed the stock LightZero boundary and
  beats scalar base CPU-oracle at the same C16 shape by about `1.53x`.
- Increasing the batched GPU row to C64 improves throughput to about `416.89`
  steps/s, so larger root/env batches help that path.
- The production-like subprocess CPU-oracle C64 control is still about `2.12x`
  faster than the batched GPU manager C64 row.
- Plain reason: subprocess CPU-oracle hides observation/render work across many
  worker processes. The current batched GPU manager keeps one batched surface in
  one process, uses a lot of H100 memory, and does not yet expose where the
  manager step time is going.
- Current recommendation: keep real training on stock subprocess CPU-oracle
  while treating batched GPU observation as a research/profiling lane. The next
  optimizer step is to time the batched manager `step`/`reset` directly and see
  whether the loss is manager/env boundary overhead, device synchronization, or
  collection topology.

Timed C64 rerun after adding direct manager timing:

Run: `opt-batched-stock-timed-c64-20260520a/envmgr-c64-sim2-timed`.

| bucket | sec | read |
|---|---:|---|
| train wall | `151.50` | full profile row wall |
| collector collect | `144.28` | still the main wall |
| batched manager step | `109.44` | direct proof that most collect time is the manager/env boundary |
| surface stack update | `94.07` | includes renderer-backed observation stack work |
| renderer render | `92.02` | direct GPU renderer aggregate inside the stack path |
| renderer device render | `82.28` | actual device render bucket |
| host to device | `4.49` | visible but not the main wall |
| owner ordered pack | `3.84` | visible CPU packing cost |
| device to host | `0.41` | not the wall |
| vector/env step | `9.77` | physics plus surface step is much smaller than observation |
| policy forward collect | `27.67` | visible, but below manager step |
| MCTS search | `8.23` | small at sim2 |
| learner train | `1.05` | not relevant for this profile |

Zero-observation stock manager diagnostic, same stock LightZero path:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | manager step sec | policy forward sec | MCTS sec | learner sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zeroobs-c64-steps1024` | profile-only batched manager, zero `[4,64,64]` obs | 64 | 65536 | 2 | `1259.60` | `52.03` | `49.23` | `15.43` | `26.96` | `8.23` | `1.10` |
| `zeroobs-c128-steps1024` | profile-only batched manager, zero `[4,64,64]` obs | 128 | 131072 | 2 | `1557.42` | `84.16` | `81.37` | `24.92` | `45.83` | `12.06` | `1.11` |

Read:

- This is the clearest Amdahl split so far. The one-process batched manager is
  not inherently doomed: with real env stepping, scalar LightZero timesteps,
  policy/search, replay, and learner still active, but render pixels replaced
  by zeros, it beats the known subprocess CPU-oracle C64/C128 controls.
- C64 zero-observation is about `2.95x` faster than the older C64 real
  batched-GPU-render row (`1259.60` versus `416.89` steps/s), and about
  `1.43x` faster than the subprocess CPU-oracle C64 control (`883.03`
  steps/s).
- C128 zero-observation is about `1.77x` faster than the newer C128 real
  batched-GPU-render row (`1557.42` versus `879` steps/s), and about `1.66x`
  faster than the subprocess CPU-oracle C128 control (`940` steps/s).
- Plain conclusion: the current full-loop gap is still mostly observation
  render/stack cost in the batched manager path, not MCTS at sim2 and not
  learner/replay. The manager/boundary has enough headroom if the render path
  becomes cheap.
- Caveat: zero-observation is not a production observation. It is a diagnostic
  row that removes renderer pixel work while preserving most of the stock
  LightZero collection/search/replay/learner path.

Fresh matched rows launched after this diagnostic:

- `opt-stock-cpuoracle-sync-c128-steps1024-20260520b/cpuoracle-c128-steps1024`
  completed at about `857.48 steps/s`.
- `opt-batched-gpu-sync-c128-steps1024-20260520b/batchedgpu-c128-steps1024`
  completed at about `978.96 steps/s`.
- `opt-batched-gpu-sync-c256-steps1024-20260520b/batchedgpu-c256-steps1024`
  completed at about `1193.48 steps/s`.
- `opt-batched-zeroobs-sync-c256-steps1024-20260520b/zeroobs-c256-steps1024`
  completed at about `1748.18 steps/s`.

Fresh C128 read:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | manager step sec | renderer render sec | device render sec | policy forward sec | MCTS sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpuoracle-c128-steps1024` | subprocess CPU oracle | 128 | 131072 | 2 | `857.48` | `152.86` | `127.67` | n/a | n/a | n/a | `50.27` | `15.27` |
| `batchedgpu-c128-steps1024` | one-process batched GPU render | 128 | 131072 | 2 | `978.96` | `133.89` | `128.26` | `75.21` | `48.22` | `35.10` | `41.13` | `12.50` |
| `zeroobs-c128-steps1024` | one-process batched manager, zero obs | 128 | 131072 | 2 | `1557.42` | `84.16` | `81.37` | `24.92` | `0.047` | `0.0` | `45.83` | `12.06` |

Read:

- The C128 real batched-GPU observation path now beats the fresh C128
  subprocess CPU-oracle control by about `14%` (`978.96` versus `857.48`
  steps/s). This is the first clean full-loop win for the batched GPU manager
  at this topology.
- It is still only about `63%` of the zero-observation ceiling (`978.96` versus
  `1557.42` steps/s). The missing speed is not mysterious: manager step drops
  from `75.21s` to `24.92s` when real render is removed, while policy/MCTS stay
  in the same rough band.
- Amdahl read: at C128/sim2, render/stack remains the main optimizable tax
  inside the batched manager; MCTS/search is visible but not the dominant gap.

Fresh C256 read:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | manager step sec | renderer render sec | device render sec | policy forward sec | MCTS sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `batchedgpu-c256-steps1024` | one-process batched GPU render | 256 | 262144 | 2 | `1193.48` | `219.65` | `213.17` | `114.98` | `62.80` | `38.78` | `76.51` | `17.80` |
| `zeroobs-c256-steps1024` | one-process batched manager, zero obs | 256 | 262144 | 2 | `1748.18` | `149.95` | `144.47` | `51.65` | `0.107` | `0.0` | `70.15` | `16.35` |

Read:

- C256 real GPU render improves over C128 real GPU render (`1193.48` versus
  `978.96` steps/s), so larger root/env batching helps.
- C256 real GPU render is still only about `68%` of the C256 zero-observation
  ceiling. Removing real render drops manager step from `114.98s` to `51.65s`.
- Device render alone is not the whole gap: render pack (`15.15s`), host/device
  transfer (`~5.05s`), stack update (`68.23s` total), and surface env step
  (`31.09s`) are all visible.
- This makes the next optimization more concrete: reduce real observation
  render/stack tax inside the batched manager, then retest at C256/C512.

New rows launched after the C256 read:

- `opt-stock-cpuoracle-sync-c256-steps1024-20260520b/cpuoracle-c256-steps1024`
  to get the matched subprocess CPU-oracle C256 control.
- `opt-batched-gpu-sync-c512-steps1024-20260520b/batchedgpu-c512-steps1024`
  to see whether real batched GPU render keeps scaling.
- `opt-batched-zeroobs-sync-c512-steps1024-20260520b/zeroobs-c512-steps1024`
  completed at about `1805.22 steps/s`.

Fresh C512 zero-observation read:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | manager step sec | surface env step sec | stack update sec | policy forward sec | MCTS sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zeroobs-c512-steps1024` | one-process batched manager, zero obs | 512 | 524288 | 2 | `1805.22` | `290.43` | `284.88` | `108.17` | `63.33` | `11.63` | `123.73` | `23.62` |

Read:

- The zero-observation ceiling flattens from C256 (`1748.18`) to C512
  (`1805.22`) rather than doubling. Once render is gone, policy forward,
  env-step/manager work, and MCTS become the visible next walls.
- This does not weaken the render conclusion. It bounds the payoff: at the
  current sim2/no-RND shape, closing the real-render gap can plausibly move
  C256 from about `1193` toward `1748`, but after that Amdahl moves to
  policy/search/manager/env-step.

Fresh C512 real-render and C256 CPU-oracle read:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | manager step sec | renderer render sec | device render sec | policy forward sec | MCTS sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpuoracle-c256-steps1024` | subprocess CPU oracle | 256 | 262144 | 2 | `722.43` | `362.86` | `319.44` | n/a | n/a | n/a | `102.55` | `26.76` |
| `batchedgpu-c512-steps1024` | one-process batched GPU render | 512 | 524288 | 2 | `1352.47` | `387.65` | `377.63` | `197.36` | `84.46` | `41.80` | `123.60` | `23.82` |
| `zeroobs-c512-steps1024` | one-process batched manager, zero obs | 512 | 524288 | 2 | `1805.22` | `290.43` | `284.88` | `108.17` | `0.222` | `0.0` | `123.73` | `23.62` |

Read:

- The fresh C256 CPU-oracle subprocess row is slower than C128 and much slower
  than C256/C512 batched GPU. At these widths, the current one-process batched
  GPU manager is the faster profile path.
- Real C512 batched GPU improves over real C256 (`1352.47` versus `1193.48`)
  but does not approach the C512 zero-observation ceiling (`1805.22`).
- The C512 real-vs-zero delta is about `1.33x`, so render/stack/pack still
  matter, but the maximum remaining win from removing them is smaller than at
  C128/C256.
- Fresh Amdahl: after this point, even perfect observation rendering leaves
  policy forward, MCTS, and manager/env-step as visible walls. The next renderer
  work should be targeted, not a deep rewrite for its own sake.

Post-patch canaries launched after local tests:

- `opt-batched-gpu-postpatch-c256-steps1024-20260520b`
- `opt-batched-zeroobs-postpatch-c256-steps1024-20260520b`
- `opt-batched-gpu-postpatch-c512-steps1024-20260520b`

Patch under test:

- cache full-row render index/player arrays in
  `_RendererBackedSourceStateGray64Stack4`;
- avoid a temporary float array for the full-row latest-frame stack write;
- reuse an already materialized full timestep when the requested scalar env-id
  order is unchanged.

Local checks after the patch:

```text
py_compile: passed
focused pytest: 18 passed, 122 deselected
```

Post-patch C256 results:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | manager step sec | renderer render sec | device render sec | policy forward sec | MCTS sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `batchedgpu-postpatch-c256-steps1024` | one-process batched GPU render | 256 | 262144 | 2 | `1096.18` | `239.14` | `230.29` | `114.65` | `65.41` | `40.36` | `78.26` | `19.22` |
| `zeroobs-postpatch-c256-steps1024` | one-process batched manager, zero obs | 256 | 262144 | 2 | `1735.15` | `151.08` | `147.38` | `43.93` | `0.11` | `0.0` | `72.81` | `17.48` |
| `batchedgpu-postpatch-c512-steps1024` | one-process batched GPU render | 512 | 524288 | 2 | `1439.84` | `364.13` | `352.61` | `178.18` | `83.52` | `41.72` | `119.54` | `22.52` |

Read:

- The cleanup patch did **not** prove a speedup at C256. The real-render row
  dropped from the prior C256 anchor (`1193.48` steps/s) to `1096.18`
  steps/s, while the zero-observation row stayed roughly flat (`1748.18` to
  `1735.15` steps/s).
- At C512 the same patch did move the full loop: real GPU render improved from
  the prior C512 anchor (`1352.47` steps/s) to `1439.84` steps/s, about a
  `6.5%` speedup.
- Treat the patch as a small C512 win and a neutral/noisy C256 result, not a
  general breakthrough.
- Amdahl is still plain: in the real C256 row, manager step is `114.65s`;
  removing real observations drops it to `43.93s`. Perfect observation removal
  would move this specific row by about `1.58x`, not `10x`.
- At C512, comparing real post-patch (`1439.84`) to the existing zero-observation
  ceiling (`1805.22`) says perfect observation removal would be only about a
  `1.25x` further win. After that, policy forward, MCTS/search, manager/env
  step, and LightZero collection shape are the wall.
- Important naming caveat: the command JSON still reports
  `policy_observation_backend=cpu_oracle` because that field belongs to the
  scalar wrapper config. In these rows the actual profile lane is identified by
  `env_manager_type=curvyzero_batched_profile` or
  `curvyzero_batched_zero_obs_profile`.

### Batched GPU Gate Grid, First Launch Attempt

Experiment ids:

- `opt-batched-gpu-gates-c512-rnd10-20260521a`
- `opt-batched-gpu-gates-c512-rnd100-20260521a`
- `opt-batched-gpu-gates-normaldeath-c256-20260521a`
- `opt-batched-gpu-gates-c768-realzero-20260521a`

Result: all failed immediately with:

```text
ValueError: curvyzero_batched_profile requires even collector/evaluator env counts
```

Read: this was a profile-grid tooling bug, not a performance result. The grid
builder set even collector counts but did not pass `--evaluator-env-num`, so
the launcher defaulted to one evaluator env. I patched
`scripts/build_curvytron_profile_grid.py` to emit `--evaluator-env-num` and to
record it per row.

Corrected `20260521b` rows launched:

- `opt-batched-gpu-gates-c512-rnd10-20260521b`: C512 no-RND repeat plus C512
  `rnd_meter_v0` with `rnd_update_per_collect=10`.
- `opt-batched-gpu-gates-c512-rnd100-20260521b`: C512 `rnd_meter_v0` with
  `rnd_update_per_collect=100`.
- `opt-batched-gpu-gates-normaldeath-c256-20260521b`: C256 real batched GPU
  with normal death/autoreset enabled.
- `opt-batched-gpu-gates-c768-realzero-20260521b`: C768 real batched GPU and
  C768 zero-observation saturation probe.

Normal-death `20260521b` result:

- `called_train_muzero=true`
- failed after `44` batched manager steps and `44` MCTS searches
- problem:

```text
dynamic JAX renderer currently requires full row-major rows; got [80, 80]
```

Read: no-death speed rows are useful, but batched GPU was not promotable for
normal-death/autoreset because LightZero eventually steps a subset of env ids.
The dynamic renderer had been written for the full row-major hot path only.
I patched `_DynamicJaxBatchedObservationRenderer.render(...)` to render the
full batch internally and gather requested row/player frames for partial
requests. Focused local tests now pass:

```text
71 passed, 2 skipped
```

Corrected normal-death gate launched:

- `opt-batched-gpu-gates-normaldeath-c256-20260521c`

`20260521c` result:

- passed the previous partial-render row/player request blocker;
- failed after `45` manager steps with:

```text
action_by_env_id must contain exactly the current ready env ids; missing=[160, 161]
```

Read: this is another stock env-manager contract issue. LightZero can stop
requesting actions for a whole scalar-env pair once it has enough episodes.
The bridge still required every ready id. I patched the scalar-action bridge to
allow omission of complete physical CurvyTron rows while still rejecting
half-row omissions. That keeps the simultaneous-action rule intact: either both
players in a row act, or the row is not represented in the returned timestep
mapping for that LightZero step.

Focused local checks after this second normal-death fix:

```text
71 passed, 2 skipped
py_compile: passed
```

Corrected normal-death gate after the complete-row omission fix:

- `opt-batched-gpu-gates-normaldeath-c256-20260521d`
- `called_train_muzero=true`
- `ok=true`
- `env_steps_collected=36014`
- `steps_per_sec=485.01`
- `learner_train_calls=1`
- `replay_sample_calls=1`
- `mcts_search_calls=333`
- `mcts_search_root_sum=36014`

Timing read:

| bucket | sec |
|---|---:|
| train wall | `74.25` |
| collector collect | `66.79` |
| batched manager step | `45.42` |
| renderer render | `16.93` |
| renderer device render | `8.40` |
| surface stack update | `11.34` |
| surface env step | `3.26` |
| policy forward collect | `17.27` |
| MCTS search | `5.00` |
| learner train | `1.25` |

Read:

- This is a semantic gate, not a speed recommendation. It proves the batched
  GPU profile manager can survive normal death/autoreset through stock
  LightZero `train_muzero`.
- The lower throughput is expected: rows die, LightZero requests partial sets
  of scalar env ids, and the average MCTS root batch falls to about `108`
  instead of the full `256`.
- The two fixes matter because they are real stock-boundary compatibility
  issues: partial render requests and complete physical-row omission. They are
  not trainer-default promotions.

RND gate rows:

| row | RND mode | update/collect | steps source | steps/s | wall sec | manager step sec | render sec | policy sec | MCTS sec | RND collect/train/estimate sec | gate read |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| `c512-rnd10/row_001` | off | n/a | `collector_envstep_delta` | `1379.36` | `380.09` | `188.05` | `87.05` | `126.66` | `24.46` | n/a | no-RND repeat |
| `c512-rnd10/row_002` | `rnd_meter_v0` | `10` | `mcts_search_root_sum_profile_fallback` | `1218.94` | `430.12` | `203.29` | `91.50` | `138.31` | `28.12` | `9.07 / 0.95 / 0.19` | predictor changed, target frozen, reward unchanged |
| `c512-rnd100/row_001` | `rnd_meter_v0` | `100` | `mcts_search_root_sum_profile_fallback` | `1234.55` | `424.68` | `201.95` | `91.71` | `138.17` | `27.76` | `8.66 / 3.72 / 0.13` | predictor changed, target frozen, reward unchanged |

Read:

- RND meter mode is wired through the batched GPU profile lane and the key
  safety claims passed: predictor changes, target stays frozen, and the
  reward target is unchanged in meter mode.
- The RND rows are overhead rows. Their step denominator uses the MCTS-root
  fallback because the reward-model path still reports raw compact env steps
  as zero.
- In this shape, RND meter costs roughly `10-12%` versus the no-RND repeat.
  That is meaningful but not the main Amdahl wall.

C768 saturation probe:

| row | observation path | collectors | env steps | sim | steps/s | wall sec | manager step sec | render sec | device render sec | surface env step sec | stack update sec | policy sec | MCTS sec | max GPU util |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c768-real/row_001` | batched GPU real render | 768 | 786432 | 2 | `1420.45` | `553.65` | `273.34` | `111.55` | `43.27` | `96.93` | `136.23` | `185.79` | `30.23` | `55%` |
| `c768-zero/row_002` | zero observation | 768 | 786432 | 2 | `1191.71` | `659.92` | `243.67` | `0.26` | `0.0` | `150.48` | `43.00` | `262.48` | `53.25` | `10%` |

Read:

- C768 did not produce a clean scaling win. The real-render row is about the
  same as the C512 post-patch real row (`1420.45` versus `1439.84` steps/s).
- The zero-observation row being slower than real render is a warning that
  C768 is in a different/noisier regime, not evidence that rendering helps.
  Policy forward, MCTS, and env-step buckets all got worse in that row.
- Practical recommendation for the next profile wave: do not chase C768 until
  we understand the policy/search/manager saturation. Use C256/C512 as the
  clean Amdahl comparison points.

Current high-level conclusion:

- Batched GPU observation is a real profile-lane win at wider collector counts,
  but it is not a 10x win by itself.
- At C512, perfect renderer removal is only worth about `1.25x` against the
  stable zero-observation ceiling (`1805.22 / 1439.84`).
- To get a much larger gain, the next architecture needs to reduce the rest of
  the loop too: manager/env-step scalar Python work, policy/search batching
  behavior, and possibly actor-parallel collection combined with batched GPU
  rendering.

### Saturation Grid Launched

Launched 2026-05-21, detached profile rows only; no live training defaults,
eval, GIF, or tournament work touched.

Manifests:

- `artifacts/local/curvytron_optimizer_profile_manifests/opt-batched-gpu-saturation-c512-20260521a.json`
- `artifacts/local/curvytron_optimizer_profile_manifests/opt-batched-gpu-saturation-c768-20260521a.json`

Local launch records:

- `artifacts/local/curvytron_optimizer_profile_results/opt-batched-gpu-saturation-c512-20260521a/launches.jsonl`
- `artifacts/local/curvytron_optimizer_profile_results/opt-batched-gpu-saturation-c768-20260521a/launches.jsonl`

Results:

| row | C | observation path | sim | steps/s | wall sec | manager step sec | render sec | device render sec | surface env sec | stack sec | policy sec | MCTS sec | max GPU util |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C512 row 001 | 512 | real GPU render | 2 | `1246.21` | `420.71` | `202.81` | `93.36` | `43.95` | `70.37` | `108.25` | `143.16` | `28.67` | `53%` |
| C512 row 002 | 512 | real GPU render | 4 | `1292.23` | `405.72` | `186.88` | `87.32` | `42.72` | `63.75` | `100.48` | `149.95` | `48.17` | `53%` |
| C512 row 003 | 512 | zero observation | 2 | `1825.66` | `287.18` | `91.43` | `0.21` | `0.0` | `62.62` | `10.11` | `138.73` | `27.47` | `33%` |
| C512 row 004 | 512 | zero observation | 4 | `1730.46` | `302.98` | `93.24` | `0.23` | `0.0` | `62.32` | `10.45` | `147.24` | `47.51` | `12%` |
| C768 row 001 | 768 | real GPU render | 2 | `1264.67` | `621.85` | `303.00` | `120.37` | `45.98` | `109.90` | `150.49` | `205.61` | `37.61` | `56%` |
| C768 row 002 | 768 | real GPU render | 4 | `1251.53` | `628.37` | `293.33` | `117.13` | `45.43` | `108.19` | `140.32` | `228.10` | `67.95` | `56%` |
| C768 row 003 | 768 | zero observation | 2 | `1656.87` | `474.65` | `177.14` | `0.34` | `0.0` | `103.76` | `25.93` | `196.67` | `36.35` | `14%` |
| C768 row 004 | 768 | zero observation | 4 | `1529.88` | `514.05` | `178.14` | `0.35` | `0.0` | `105.77` | `20.69` | `233.36` | `69.59` | `15%` |

Read:

- C768 did not scale. It is slower than the best C512 zero rows and roughly
  flat/slower than C512 real rows. More rows in one process are not the next
  magic knob.
- Sim4 is not catastrophic in this profile shape, but it shifts visible wall
  time into MCTS/policy. That reinforces the need to measure search and
  manager topology together.
- C512 real versus zero now gives an observation/removal upper bound of about
  `1.47x` for sim2 and `1.34x` for sim4. Renderer cleanup remains useful, but
  it cannot produce a 5-10x same-loop win.
- These rows launched before the newest partial/live-root instrumentation, so
  the ready/partial-render fields are blank. Use future rows for those fields.

### Hybrid Actor + Central Zero-Observation Local Prototype

Profile-only scaffold:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `scripts/profile_hybrid_batched_observation_manager.py`
- `tests/test_source_state_hybrid_observation_profile.py`

This does **not** call `train_muzero`; it steps compact CurvyTron rows in
in-process actor objects, merges compact metadata, fills zero stacks centrally,
and materializes LightZero-shaped scalar rows only at the outside edge.

Local results with no rendered pixels and no payload pickling:

| B | actor count | measured steps | scalar timesteps/s | physical rows/s | total sec | actor step sec | zero stack sec | scalar materialize sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 100 | `15425.32` | `7712.66` | `0.830` | `0.511` | `0.136` | `0.0043` |
| 256 | 8 | 100 | `21605.00` | `10802.50` | `2.370` | `1.270` | `0.602` | `0.0122` |
| 512 | 16 | 80 | `24878.13` | `12439.07` | `3.293` | `1.588` | `0.887` | `0.0254` |

Read:

- This is not a trainer speed claim. It intentionally excludes policy/search,
  replay, learner, RND, GPU render, IPC, and real subprocess scheduling.
- It is a useful topology proof: compact actor stepping plus central stack
  ownership has enough zero-observation headroom to justify the next prototype.
- The next useful version is not "promote this"; it is "replace zero stack with
  the direct batched GPU observation renderer and measure compact payload plus
  render service overhead before touching live training."

- `artifacts/local/curvytron_optimizer_profile_results/opt-batched-gpu-saturation-c512-20260521a/launches.jsonl`
- `artifacts/local/curvytron_optimizer_profile_results/opt-batched-gpu-saturation-c768-20260521a/launches.jsonl`

Rows:

| experiment | rows | shape |
|---|---|---|
| `opt-batched-gpu-saturation-c512-20260521a` | 4 | C512, batch1024, real-vs-zero, sim2/sim4, no-death, no-RND |
| `opt-batched-gpu-saturation-c768-20260521a` | 4 | C768, batch1536, real-vs-zero, sim2/sim4, no-death, no-RND |

Question:

- Does C512 remain the clean plateau around `1.4k-1.8k steps/s`?
- Was the earlier C768 zero-observation slowdown noise, or a real topology wall?
- Does sim4 widen the policy/search wall enough that renderer work becomes
  secondary?

Decision rule:

- If C768 remains flat or worse, stop chasing width past C512 in this
  one-process batched manager.
- If sim4 hurts both real and zero similarly, policy/search/topology is the
  wall.
- If real rows move close to zero rows, stop primary renderer work and move to
  actor-parallel plus batched-observation architecture.

### Host/Scalar Floor Instrumentation Patch

Patch scope: profile-only LightZero batched manager hook in
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.

Added samples:

- ready obs before/after step;
- ready physical row count before/after step;
- action row count;
- complete-row omission count;
- returned timestep count and timestep row count;
- renderer partial-request flag;
- renderer output count.

Compact outputs now surface selected means under `derived`, so future manifest
summaries can see whether a row is full-width hot path, partial/autoreset
path, or live-root collapsed path without digging through the full
`phase_profile`.

Validation:

```text
py_compile: passed
focused pytest: 74 passed, 2 skipped
```

Read: this is measurement only. It does not change trainer defaults, live
training runs, tournament behavior, or the observation contract.

Follow-up normal-death gate launched:

- `opt-batched-gpu-gates-normaldeath-c256-20260521d`

### Batched GPU Gate Grid, RND C512 Results

Experiment ids:

- `opt-batched-gpu-gates-c512-rnd10-20260521b`
- `opt-batched-gpu-gates-c512-rnd100-20260521b`

Rows:

| row | manager / observation path | RND mode | RND updates | steps denominator | steps/s | wall sec | manager step sec | renderer sec | policy sec | MCTS sec | RND collect/train/estimate sec |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `c512-rnd10 row001` | batched GPU real obs | none | 0 | collector envstep delta | `1379.36` | `380.09` | `188.05` | `87.05` | `126.66` | `24.46` | n/a |
| `c512-rnd10 row002` | batched GPU real obs | `rnd_meter_v0` | 10 | MCTS-root fallback | `1218.94` | `430.12` | `203.29` | `91.50` | `138.31` | `28.12` | `9.07 / 0.95 / 0.19` |
| `c512-rnd100 row001` | batched GPU real obs | `rnd_meter_v0` | 100 | MCTS-root fallback | `1234.55` | `424.68` | `201.95` | `91.71` | `138.17` | `27.76` | `8.66 / 3.72 / 0.13` |

RND proof fields:

- `feature_source=policy_gray64_latest/v0`;
- `predictor_changed=true`;
- `target_changed=false`;
- `last_target_reward_changed=false`;
- `last_target_reward_delta_abs_max=0.0`;
- `train_cnt_per_estimate=10.0` in update10 and `100.0` in update100;
- `train_with_data_skipped_small_buffer_count=0`.

Read:

- The RND meter rows are meaningful overhead/cadence rows, not learning proof
  and not positive-reward RND.
- C512 no-RND repeated at `1379.36 steps/s`, which is lower than the best
  post-patch C512 row (`1439.84`) but the same regime.
- RND meter costs about `10-12%` wall throughput in this shape, mostly through
  reward-model collection plus extra policy/manager wall. Update100 did not
  look much slower than update10 in whole-loop throughput, but it did increase
  `rnd_train_with_data` from `0.95s` to `3.72s`.
- The compact reward-model path still reports raw env steps as zero and falls
  back to MCTS-root count. That is acceptable for this profile comparison only
  because MCTS root count equals the intended `524288` env-step denominator and
  eval is disabled.

Read: for the current one-process batched GPU manager, Amdahl points back at
observation stack/render work. The GPU renderer is not "done" in the full loop:
it is faster than the old dense render, but the current batched path still burns
about `92s` of a `151s` row in renderer/stack work. The subprocess CPU-oracle
path wins because it spreads similar per-env work across many worker processes.
That means the next useful work is either a much faster batched renderer/stack
kernel, a multi-worker batched manager shape that does not trip CUDA/process
issues, or staying with subprocess CPU-oracle while optimizing smaller CPU
payload/telemetry costs.

Detached rows that landed after the timed C64 run:

| row | manager / observation path | collectors | env steps | sim | steps/s | wall sec | collector sec | manager step sec | renderer sec | policy sec | MCTS sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `envmgr-c80-sim2-timed` | profile-only batched GPU manager | 80 | 81920 | 2 | `512.95` | `159.70` | `153.20` | `115.36` | `95.12` | `29.73` | `8.41` |
| `cpuoracle-subproc-c128-sim2-control` | subprocess CPU oracle | 128 | 131072 | 2 | `940.42` | `139.38` | `116.64` | n/a | n/a | `45.15` | `13.31` |

Read:

- Wider one-process batched GPU rows do improve: C80 is faster than C64.
- The production-like subprocess CPU-oracle control is still faster in the same
  currency: about `940` steps/s at C128 versus about `513` steps/s for C80
  batched GPU.
- The C80 timer shape is the same as C64: manager step and renderer/stack still
  dominate. MCTS is visible, but not the Amdahl wall in these sim2 profile rows.

Dynamic-width stock-hook follow-up:

- I changed the stock batched GPU profile hook to request dynamic render trail
  slots, but the first C64/C96 detached rows failed before writing
  `summary.json`.
- A tiny synchronous C2 smoke passed, but prewarming every dynamic width made
  reset take about `19s`. I then disabled multi-width prewarm in the stock
  full-loop hook because compiling several width-specialized JAX render kernels
  inside one LightZero row is a memory/time risk.
- The no-prewarm C2 smoke passed and reset dropped to about `3.47s`. JAX still
  reported about `62GB` H100 memory because it preallocates a large GPU chunk;
  that number is not by itself proof that the row is out of memory.
- C16 and C32 no-prewarm dynamic rows at 64 source steps passed:

| row | collectors | source steps | env steps | steps/s | wall sec | collector sec | manager step sec | renderer sec | policy sec | MCTS sec | learner sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `envmgr-c16-steps64` | 16 | 64 | 1024 | `82.48` | `12.42` | `5.61` | `3.79` | `2.54` | `1.67` | `0.37` | `2.69` |
| `envmgr-c32-steps64` | 32 | 64 | 2048 | `167.42` | `12.23` | `6.66` | `4.27` | `2.79` | `2.15` | `0.42` | `1.25` |
| `envmgr-c64-steps256` | 64 | 256 | 16384 | `541.08` | `30.28` | `23.69` | `13.66` | `10.14` | `8.16` | `2.30` | `1.05` |
| `envmgr-c64-steps512` | 64 | 512 | 32768 | `362.43` | `90.41` | `62.29` | `32.34` | `23.62` | `25.53` | `8.31` | `21.32` |
| `envmgr-c64-steps1024` | 64 | 1024 | 65536 | `674.99` | `97.09` | `91.51` | `59.50` | `43.70` | `25.44` | `7.01` | `0.87` |
| `envmgr-c96-steps1024` | 96 | 1024 | 98304 | `758.58` | `129.59` | `122.67` | `73.73` | `48.97` | `38.22` | `12.06` | `1.32` |
| `envmgr-c128-steps1024` | 128 | 1024 | 131072 | `879.16` | `149.09` | `142.15` | `81.93` | `52.12` | `46.61` | `14.73` | `1.15` |

Current read:

- Dynamic render width is not ready as a Coach recommendation.
- C64 at 1024 source steps passed synchronously. The earlier C64/C96
  profile-spawn failures look like spawn/readback/artifact fragility, not a hard
  runtime failure.
- Dynamic no-prewarm C64/1024 is a real improvement over the old full-width
  batched C64 row: about `675` steps/s versus about `417-433` steps/s.
- C64 at 512 source steps passed but slowed down and had a suspicious
  `learner_train_sec=21.32` bucket. That needs explanation before treating the
  dynamic lane as stable.
- Dynamic batched GPU scales with width, but it still does not clearly beat the
  production-like subprocess CPU-oracle anchors: C128 dynamic reached about
  `879` steps/s versus C128 subprocess CPU-oracle about `940` steps/s.
- This no longer says "GPU observation is useless"; it says the current
  one-process batched GPU manager is only near parity with subprocess CPU-oracle
  after dynamic width and large C128 batching.
- The plain architectural reason is that GPU rendering gives one benefit
  while taking one away: it batches render work on the GPU, but it collapses the
  subprocess CPU-oracle lane's process-level parallelism into one synchronous
  manager with one collection barrier.
- The next aggressive fix is not more blind scaling. It is one of:
  a multi-worker batched GPU manager that keeps subprocess-style parallelism, a
  zero-observation manager row to isolate LightZero/scalar manager overhead, or
  a simpler fixed/capped render shape that reduces JAX shape/kernel churn.

### Duplicate-Seed H100 C512/sim4 Background Diagnostic

Manifest:

```text
artifacts/local/curvytron_optimizer_profile_manifests/opt-stock-h100-c512-sim4-dupseed-20260520a.json
```

Result:

| row | seed | steps | wall sec | steps/sec |
|---|---:|---:|---:|---:|
| 001 | 304 | 262144 | 264.72 | 990.26 |
| 002 | 304 | 262144 | 277.76 | 943.79 |
| 003 | 305 | 262144 | 242.39 | 1081.48 |
| 004 | 305 | 262144 | 297.02 | 882.57 |
| 005 | 306 | 262144 | 288.57 | 908.44 |
| 006 | 306 | 262144 | 242.74 | 1079.95 |

Read: identical workload counts still vary by roughly `~20%`. This is useful
for anchoring A/B comparisons, but it is not the main optimization lane. The
main lane remains the vector/full-loop bridge for preserving batched direct GPU
observations through the LightZero-shaped boundary.

### Local Scalar-Action Bridge Canary

Code:

```text
src/curvyzero/training/source_state_batched_observation_mock_collector.py
```

First result:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_mock_collector.py
11 passed, 2 skipped
```

Read: the new `BatchedLightZeroScalarActionBridge` is profile-only and does not
call stock LightZero yet. It does prove the next boundary shape locally:
LightZero-style scalar env-id actions can be validated, converted into one
batched joint CurvyTron step, and returned as scalar timesteps keyed by env id
without giving up the batched surface internally.

Follow-up result:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_multiplayer_source_state_trainer_surface.py \
  tests/test_source_state_batched_observation_boundary_profile.py
70 passed, 2 skipped
```

Read: the bridge now has a local env-manager-shaped wrapper with `env_num`,
`ready_obs`, `reset`, `step`, `seed`, `close`, and `last_reset_info`. The
terminal-row bug is fixed too: the bridge returns timesteps for the scalar env
ids that were actually stepped, even when those rows terminate and are no longer
policy-ready. A local `max_ticks=1` test proves terminal timesteps keep
`done=true` and `final_observation` before autoreset.

### H100 Profile Env-Manager Facade Canary

Config:

```text
compute=gpu-h100
batch_size=64
logical env ids=128
steps=64
warmup_steps=8
render_surface=direct_gray64
surface_stack_backend=renderer_backed_profile
dynamic_render_trail_slots=true
include_lightzero_payload_profile=true
pickle_lightzero_payload=true
RND off
```

Result:

- `ok=true`
- profile-only; `calls_train_muzero=false`
- `profile_env_manager_canary=true`
- `env_num=128`
- `vector_surface_batch_size=64`
- `scalar_env_instances_created=0`
- renderer backend `jax_gpu_batched_profile`
- ready obs count median `128`
- timestep count median `128`
- render slots median/p95 `128/256`
- no render truncation
- manager step median/p95 `0.0236s` / `0.0545s`
- renderer total median/p95 `0.0125s` / `0.0181s`
- device render median/p95 `0.00621s` / `0.0113s`
- env step median `0.00256s`
- payload pickle median `0.00376s`
- payload bytes median `8.41MB`

Read: this is the first Modal proof that the manager-shaped bridge can keep one
batched CurvyTron surface and expose scalar LightZero-like env ids without
creating scalar env wrappers. It is still not stock `train_muzero`, so it is a
bridge proof, not a full-loop speed claim.

Follow-up B256 no-RND row:

- `ok=true`
- `env_num=512`
- `scalar_env_instances_created=0`
- renderer backend `jax_gpu_batched_profile`
- ready obs/timestep count median `512`
- render slots median/p95 `128/256`
- no render truncation
- manager step median/p95 `0.0806s` / `0.133s`
- renderer total median/p95 `0.0354s` / `0.0518s`
- device render median/p95 `0.0149s` / `0.0296s`
- env step median `0.0105s`
- payload pickle median `0.0344s`
- payload bytes median `33.6MB`

Read: the manager bridge scales, but payload size and pickle time become
visible at B256. This supports the earlier `uint8`/payload-slimming lane if the
bridge later enters a real full loop.

Follow-up B128 CUDA RND update10 row:

- `ok=true`
- `env_num=256`
- RND device `cuda`
- `rnd_update_per_collect=10`
- RND collect/train/estimate calls `32/32/32`
- `train_cnt_per_estimate=10`
- target reward unchanged
- manager step median `0.0461s`
- payload pickle median `0.0152s`
- RND train median `0.253s`

Prewarmed follow-up rows:

| row | batch | env_num | RND | manager step median | manager step p95 | render total median | device render median | stack median | payload pickle median | payload bytes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| H100 B128 prewarm | 128 | 256 | off | `0.0438s` | `0.0557s` | `0.0213s` | `0.00877s` | `0.0263s` | `0.0106s` | `16.8MB` |
| H100 B256 prewarm | 256 | 512 | off | `0.0755s` | `0.0983s` | `0.0345s` | `0.0149s` | `0.0440s` | `0.0344s` | `33.6MB` |
| H100 B128 prewarm RND update10 | 128 | 256 | CUDA update10 | `0.0427s` | `0.0493s` | `0.0177s` | `0.00558s` | `0.0242s` | `0.0162s` | `16.8MB` |

RND update10 timing from the same prewarmed row:

- collect data median `0.00338s`
- estimate median `0.0163s`
- train median `0.253s`
- train count per estimate `10`
- target network stayed frozen and target reward was unchanged

Read: prewarming dynamic render widths fixed the earlier p95 spike problem. The
direct manager surface is now stable enough to use as the next stock-boundary
canary input. At B128/B256 no-RND, stack update and payload/pickle are already
visible next walls. With CUDA RND update10, RND predictor training dominates
the manager surface by a wide margin. This is still profile-only and
`calls_train_muzero=false`.
- RND estimate median `0.0151s`

Read: RND cadence is much larger than the direct manager surface when trained
10 times per collect. The row also had a large renderer p95 spike from dynamic
JAX render-width compilation after warmup. That is warmup/tooling noise, not a
steady-state render claim. The next profile tooling fix is to prewarm the
possible dynamic render widths before measured steps.

### Direct `64x64` GPU Surface Canary

Status: passed, then rerun after the direct-symbol fix.

Purpose: compare the existing dense `block_704_gray64` renderer-backed surface
row against the approximate `direct_gray64` learned-observation path. This is
not browser-pixel parity and not full training. It answers whether dense
source-resolution supersampling is still the local GPU render wall.

Config shape:

```text
compute=gpu-h100
batch_size=64
steps=256
warmup_steps=16
body_capacity=1024
trail_slots=1024
dynamic_render_trail_slots=true
min_render_trail_slots=32
surface_facade_canary=true
surface_stack_backend=renderer_backed_profile
render_surface=direct_gray64
verify_steps=0
cpu_reference_interval=0
include_lightzero_payload_profile=true
```

The seeded comparison rows below completed. The first row was useful speed
evidence but had an important semantic caveat: direct simple-symbol bonuses
were too coarse. The corrected row below is the one to cite.

Result:

- `ok=true`
- profile-only, `calls_train_muzero=false`
- render surface: `direct_gray64`
- median active trail count max: `292`
- p95 active trail count max: `498`
- render width median/p95: `512/512`
- no render truncation
- surface step median/p95: `0.0316s` / `0.0435s`
- renderer total median/p95: `0.0177s` / `0.0187s`
- device render median/p95: `0.00975s` / `0.0111s`
- host-to-device median: `0.00319s`
- device-to-host median: `0.00030s`

Corrected direct-symbol result:

- local tests: `16 passed, 6 skipped` for the renderer CPU/direct tests;
  broader renderer/boundary/surface target: `66 passed, 6 skipped`
- H100 adversarial two-view direct canary: exact CPU-direct parity,
  `mismatch_count=0`, output shape `[8,1,64,64]`
- corrected H100 B64 steps256 surface canary: `ok=true`
- profile-only, `calls_train_muzero=false`
- surface step median/p95: `0.0339s` / `0.0512s`
- renderer total median/p95: `0.0190s` / `0.0249s`
- device render median/p95: `0.00973s` / `0.0118s`
- host-to-device median: `0.00410s`
- device-to-host median: `0.00038s`
- no render truncation

Read: compared with the latest matched `block_704_gray64` surface row
(`0.144s` surface step, `0.123s` device render), corrected `direct_gray64` is
about `4.2x` faster for the surface canary and about `12.6x` faster inside
device render. Compared with CPU dirty-cache (`0.237s`), the local surface
canary is about `7.0x` faster. This is still not a stock LightZero full-loop/RND
claim.

### Surface-Facade Payload Trust Fix

Status: patched after critique.

Faraday found that the surface-facade profiler was timing mock LightZero
payloads by flattening `step.observation` into all `B * P` seats. That was the
wrong thing for the next A/B because the trainer surface already defines the
live policy rows through `policy_observation`, `policy_env_row`, and
`policy_player`.

Fix:

- surface-facade payload timing now calls
  `materialize_trainer_surface_policy_timestep`;
- terminal final observations are attached to mock timestep info;
- RND metrics are returned from the surface-facade profiler;
- terminal rows are reset after measuring a terminal step;
- focused tests passed:
  `tests/test_source_state_batched_observation_mock_collector.py` and
  `tests/test_source_state_batched_observation_boundary_profile.py`:
  `44 passed, 2 skipped`.

Read: this does not produce a new speed number. It makes the next speed number
harder to fool.

Follow-up promotion-gate fix:

- `SourceStateMultiplayerTrainerSurface` can now require an exact renderer
  backend name when using `renderer_backed_profile`.
- The Modal surface-facade canary now requires the JAX batched profile backend,
  so a wrong renderer cannot silently enter a GPU/direct profile row.
- `materialize_trainer_surface_policy_timestep` now raises on missing or
  malformed terminal final observations/reward maps instead of silently
  replacing terminal rows with zeros.
- Focused verification:
  `tests/test_source_state_batched_observation_mock_collector.py`,
  `tests/test_multiplayer_source_state_trainer_surface.py`, and
  `tests/test_source_state_batched_observation_boundary_profile.py`:
  `62 passed, 2 skipped`.

### Direct Surface Batch-Width Scale Rows

Status: passed, profile-only.

Rows:

| batch | policy rows | surface step median | renderer median | device render median | pack median | payload pickle median |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 128 | `0.0339s` | `0.0190s` | `0.00973s` | not recorded in old row | not recorded in old row |
| 128 | 256 | `0.0568s` | `0.0241s` | `0.0112s` | `0.00756s` | `0.00314s` |
| 256 | 512 | `0.154s` | `0.0330s` | `0.0124s` | `0.0143s` | `0.0158s` |
| 512 | 1024 | `0.225s` | `0.0484s` | `0.0141s` | `0.0287s` | `0.0214s` |

All rows were H100, `direct_gray64`, `surface_facade_canary=true`,
`calls_train_muzero=false`, `include_lightzero_payload_profile=true`, no RND,
no render truncation, no terminal rows.

Read:

- The direct GPU device render scales well in this range. It is not the whole
  surface wall anymore.
- At B512, renderer total is only about `21%` of surface step
  (`0.0484 / 0.225`). Optimizing just device render from here has limited
  Amdahl headroom for this surface canary.
- The remaining surface time is now the next local target: env step/state
  mutation, compact packing, stack update/copy, and host-side array movement.
- Payload pickle is visible but still outside `surface_step_total_sec`; it
  matters for subprocess/full-loop cost, not for renderer-only claims.

Follow-up patch: the profiler now records derived `surface_nonrenderer_sec`
and `lightzero_payload_total_sec` so future rows expose this split directly.

Instrumented B512 rerun:

| bucket | median |
|---|---:|
| surface step total | `0.254s` |
| renderer total | `0.0523s` |
| device render | `0.0142s` |
| non-render surface remainder | `0.202s` |
| env physics/state step | `0.0211s` |
| stack update | `0.104s` |
| surface package/materialize | `0.126s` |
| LightZero payload total outside surface step | `0.0265s` |
| payload pickle bytes | `67.1MB` |

Read: the current direct-GPU surface bottleneck is host-side observation
materialization. Renderer work is now small enough that further kernel work
alone cannot move the surface canary much. The low-hanging targets are stack
copy/update, policy-row packaging, final-observation allocation/copy, and
payload layout/dtype.

### Host-Side Package Copy Fix

Status: passed and measured.

Patch:

- renderer-backed profile steps reuse the full row-major observation view for
  policy rows when all rows/seats are live;
- renderer-backed profile steps avoid a second full observation copy at return;
- nonterminal steps use broadcast zero final-observation/final-reward arrays
  and do not put giant zero final-observation arrays into `info`;
- terminal rows still copy and expose final observations.

B512 before/after:

| bucket | before | after |
|---|---:|---:|
| surface step total | `0.254s` | `0.143s` |
| surface package/materialize | `0.126s` | `0.00038s` |
| non-render surface remainder | `0.202s` | `0.0879s` |
| renderer total | `0.0523s` | `0.0552s` |
| stack update | `0.104s` | `0.120s` |
| env step | `0.0211s` | `0.0220s` |
| payload pickle outside surface | `0.0265s` | `0.0324s` |
| payload size | `67.1MB` | `67.1MB` |

Read: this is a real local surface win, about `1.78x` on the B512 direct
surface step. It did not shrink the payload itself. The next wall is still
large CPU float32 payload movement plus stack update/render/pack.

Follow-up no-copy stack return:

| bucket | after package fix | after stack `copy=False` |
|---|---:|---:|
| surface step total | `0.143s` | `0.123s` |
| renderer total | `0.0552s` | `0.0764s` |
| non-render surface remainder | `0.0879s` | `0.0485s` |
| stack update, including renderer | `0.120s` | `0.101s` |
| surface package/materialize | `0.00038s` | `0.00048s` |
| payload pickle outside surface | `0.0324s` | `0.0198s` |

Read: the profile-only surface can now avoid the public stack copy too. The
B512 surface step is about `2.1x` faster than the pre-fix instrumented row
(`0.254s -> 0.123s`) and about `1.8x` faster than the first B512 scale row
(`0.225s -> 0.123s`). The current surface wall is renderer/pack plus remaining
host-side stack update. `stack_update_sec` includes the renderer call, so do not
add those two buckets together.

### Surface-Facade RND Cadence Microprofiles

Status: passed, profile-only.

Purpose: separate RND cost from renderer cost. These rows run the direct
surface-facade canary with RND meter hooks; they do not call stock
`train_muzero`, do not prove positive-weight RND learning, and should not be
mixed into renderer-only claims.

Config shape:

```text
compute=gpu-h100
batch_size=128
steps=32
warmup_steps=8
render_surface=direct_gray64
surface_stack_backend=renderer_backed_profile
include_rnd_meter=true
rnd_batch_size=64
```

Rows:

| RND device | update_per_collect | surface step median | RND train median | RND estimate median | payload total median |
|---|---:|---:|---:|---:|---:|
| CPU | 100 | `0.0312s` | `7.4266s` | `0.167s` | `7.611s` |
| CUDA, cuDNN disabled | 100 | `0.0282s` | `2.3889s` | `0.0145s` | `2.414s` |
| CUDA, cuDNN disabled | 10 | `0.0288s` | `0.258s` | `0.0164s` | `0.285s` |
| CUDA, cuDNN disabled | 1 | `0.0291s` | `0.0267s` | `0.0163s` | `0.0560s` |

Read:

- RND cost can dwarf the observation surface if we train the predictor many
  times per collect. That is a separate training/observability knob, not a
  renderer regression.
- CUDA RND is much faster than CPU RND in this smoke, but the cadence setting
  dominates the wall clock.
- `rnd_update_per_collect=100` may be appropriate for a serious positive-RND
  canary, but it is not free. `10` and `1` are useful ablations/profiles.
- The CUDA row needed cuDNN disabled in this Modal image because torch/cuDNN
  produced a sublibrary version mismatch. The profile config now records that
  flag so this workaround is explicit.

### Stock Full-Loop RND Meter Profile

Status: passed.

Purpose: measure the real stock trainer path with RND meter plumbing, while
keeping tournament/eval/GIF/checkpoint side work out of the way. This is the
answer to "is the GPU observation surface canary the full training loop?" The
answer is no; this row is the separate full-loop smoke.

Config shape:

```text
entrypoint=lightzero_curvyzero_stacked_debug_visual_survival_train::main
mode=profile
compute=gpu-h100-cpu40
env_variant=source_state_fixed_opponent
policy_observation_backend=cpu_oracle
env_manager_type=subprocess
collector_env_num=32
n_episode=32
batch_size=64
num_simulations=8
exploration_bonus_mode=rnd_meter_v0
exploration_bonus_weight=0
rnd_update_per_collect=100
stop_after_learner_train_calls=12
eval/GIF/tournament/checkpoint cadence disabled for profiling
```

Result:

- `ok=true`
- `called_train_muzero=true`
- trainer entrypoint: `lzero.entry.train_muzero_with_reward_model`
- env steps collected: `16,384`
- MCTS search calls: `512`
- replay samples: `12`
- learner train calls: `12`
- RND collect calls: `1`
- RND train calls: `1`, with `100` predictor updates
- RND estimate calls: `12`
- target reward changed: `false`
- throughput: about `457 steps/s`
- GPU max utilization: `17%`
- GPU max memory: about `1.7 GiB`
- `train_muzero_wall`: `35.84s`

Key timers:

- `policy_forward_collect`: `17.17s`
- `mcts_search`: `10.96s`
- `model_initial_inference`: `1.82s`
- `model_recurrent_inference`: `4.82s`
- `rnd_train_with_data`: `3.13s`
- `rnd_state_hash`: `2.80s`
- `learner_train`: `1.31s`
- `replay_sample`: `0.37s`

Sampled env timing, worker-side:

- `observation_sec` mean: `6.50ms`
- `update_stack_sec` mean: `6.47ms`
- `physical_loop_sec` mean: `4.55ms`
- `base_env_timestep_pickle_sec` mean: `1.82ms`

Read: this is the real stock full-loop/RND profile, and it is not the same as
the GPU observation surface canary. On this row, H100 GPU utilization is low and
the large named timers are policy/MCTS plus RND meter overhead, while CPU-oracle
observation/update-stack remains a meaningful worker-side cost. The direct GPU
surface win is promising, but it still needs a stock-loop integration plan that
preserves batching; scalar `jax_gpu` is not that plan.

### Matched Stock Full-Loop No-RND Control

Status: passed.

Purpose: compare against the RND meter row above using the same H100 C32/sim8
profile shape, but with `exploration_bonus_mode=none`. This separates RND meter
overhead from ordinary stock LightZero collection/search/learner cost.

Result:

- `ok=true`
- trainer entrypoint: `lzero.entry.train_muzero`
- env steps collected: `16,384`
- MCTS search calls: `512`
- replay samples: `12`
- learner train calls: `12`
- throughput: about `426 steps/s`
- `train_muzero_wall`: `38.45s`
- GPU max utilization: `1%`
- GPU max memory: about `1.6 GiB`

Key timers:

- `collector_collect`: `32.22s`
- `policy_forward_collect`: `20.74s`
- `mcts_search`: `13.84s`
- `model_initial_inference`: `2.29s`
- `model_recurrent_inference`: `6.12s`
- `learner_train`: `2.02s`
- `replay_sample`: `0.33s`

Sampled env timing, worker-side:

- `observation_sec` mean: `8.88ms`
- `update_stack_sec` mean: `8.85ms`
- `physical_loop_sec` mean: `2.38ms`
- `base_env_timestep_pickle_sec` mean: `0.43ms`

Read: one matched row does not prove RND overhead because the RND-meter row was
slightly faster (`457 steps/s`) despite having RND-specific timers. Treat this
as profile variance or timer-boundary difference until repeated. The stable
read is that both stock full-loop rows are in the same throughput band, GPU
utilization is low, and CPU-oracle observation/update-stack is still a
meaningful worker-side bucket.

### Stock H100 C512/sim4 Rebaseline

Status: passed, but noisy.

Manifest:
`artifacts/local/curvytron_optimizer_profile_manifests/opt-stock-h100-c512-sim4-rebaseline-20260520a.json`

Purpose: repeat the current best stock full-loop topology before comparing
direct GPU observation or RND variants. This is the clean Amdahl anchor.

Rows:

- seeds: `304`, `305`, `306`
- compute: `gpu-h100-cpu40`
- env manager: `subprocess`
- collectors/episodes: `512`
- batch size: `64`
- MCTS simulations: `4`
- source max steps: `512`
- stop after learner train calls: `12`
- RND: off
- eval/GIF/tournament/checkpoint clutter: disabled/effectively off for profile
- death: disabled for long-trajectory profile

Trust condition: compare only rows with matched env steps, MCTS roots, sim
budget, replay samples, and learner calls. Treat small deltas as noise.

Results:

| row | seed | steps | MCTS roots | sim budget | replay | learner | wall sec | steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 304 | 262144 | 262144 | 2048 | 12 | 12 | `472.79` | `554.46` |
| 002 | 305 | 262144 | 262144 | 2048 | 12 | 12 | `236.08` | `1110.42` |
| 003 | 306 | 262144 | 262144 | 2048 | 12 | 12 | `472.71` | `554.55` |

Read:

- Workload counts match exactly, so the split is not caused by fewer env steps,
  fewer roots, fewer replay samples, or fewer learner calls.
- Row 002 is about `2x` faster across collector/policy/MCTS/learner timers.
  This looks like infrastructure/runtime variance or another hidden topology
  difference, not a learning/algorithm difference.
- Do not use the mean as a recommendation. The conservative current stock
  anchor is about `554 steps/s`; the best observed repeat is about
  `1110 steps/s`.
- Next repeat should duplicate seeds or capture hardware/process details before
  making a hard H100 C512/sim4 throughput claim.

### Forced-Terminal Direct Surface Canary

Status: passed.

Purpose: check Dalton's terminal/final-observation promotion gate in the
profile-only direct surface path.

Config shape:

```text
compute=gpu-h100
batch_size=64
steps=32
warmup_steps=8
body_capacity=1024
trail_slots=1024
dynamic_render_trail_slots=true
min_render_trail_slots=32
max_ticks=5
render_surface=direct_gray64
surface_facade_canary=true
surface_stack_backend=renderer_backed_profile
include_lightzero_payload_profile=true
verify_steps=0
cpu_reference_interval=0
```

Result:

- `ok=true`
- `profile_only=true`
- `calls_train_muzero=false`
- `touches_live_runs=false`
- JAX backend: GPU, one device
- terminal rows occurred: `terminal_row_count` p95 `64`
- final observations occurred: `final_observation_row_count` p95 `64`
- no render truncation
- policy rows median `128`
- render width median/p95 `32/32`
- surface step median/p95: `0.0164s` / `0.0274s`
- renderer median/p95: `0.00940s` / `0.00991s`
- device render median/p95: `0.00165s` / `0.00183s`
- LightZero payload total median/p95: `0.00479s` / `0.00553s`

Read: this does not prove natural-death trainer semantics, but it does prove
the profile-only direct surface can carry terminal rows and final observations
through the surface-facade payload path without touching live runs.

Final local verification for this pass:

```text
uv run pytest \
  tests/test_curvytron_profile_grid_builder.py \
  tests/test_curvytron_optimizer_profile_manifest_runner.py \
  tests/test_exploration_bonus.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_multiplayer_source_state_trainer_surface.py -q
```

Result: `82 passed, 6 skipped`.

`git diff --check` also passed.

### Duplicate-Seed H100 C512/sim4 Rebaseline

Status: launched/collecting.

Manifest:
`artifacts/local/curvytron_optimizer_profile_manifests/opt-stock-h100-c512-sim4-dupseed-20260520a.json`

Purpose: explain the previous `~554` versus `~1110 steps/s` split. This repeats
the same stock full-loop H100 C512/sim4 no-RND CPU-oracle profile with duplicate
seeds:

```text
seeds = 304, 304, 305, 305, 306, 306
```

Trust condition: if duplicate seeds still split, treat the variance as
infrastructure/runtime. If seed 305 is consistently fast and 304/306 are
consistently slow despite equal workload counts, inspect seed-specific env or
trajectory behavior.

### Vector-Facade Local Gate Additions

Status: passed locally.

Added local tests for:

- missing scalar action rows reject;
- extra scalar action rows reject;
- partial reset preserves neighboring row observations and row/player order.

Focused result:
`tests/test_source_state_batched_observation_mock_collector.py` and
`tests/test_multiplayer_source_state_trainer_surface.py`: `24 passed, 2 skipped`.

### Direct Surface RND Latest-Frame Canary

Status: passed.

Purpose: verify that the direct surface-facade path feeds RND from live surface
policy rows/latest frames, without claiming positive-RND learning.

Config shape:

```text
compute=gpu-h100
batch_size=64
steps=16
warmup_steps=4
render_surface=direct_gray64
surface_facade_canary=true
surface_stack_backend=renderer_backed_profile
include_lightzero_payload_profile=true
include_rnd_meter=true
rnd_device=cuda
rnd_update_per_collect=1
rnd_batch_size=64
```

Result:

- `ok=true`
- `profile_only=true`
- `calls_train_muzero=false`
- `uses_surface_policy_rows=true`
- RND device: `cuda`, `disable_cudnn=true`
- RND collect/train/estimate calls: `16 / 16 / 16`
- `train_cnt_per_estimate=1.0`
- `last_target_reward_changed=false`
- `buffer_count=2048`
- policy rows median: `128`
- no render truncation
- surface step median: `0.0187s`
- LightZero payload total median: `0.0482s`
- RND train median: `0.0324s`
- RND estimate median: `0.0101s`

Read: RND latest-frame extraction through the surface-facade wrapper works.
With CUDA update1, RND is already larger than the direct surface step median,
so keep RND timing separate from render timing.

### Instrumented Host-Overhead Ladder

Run id family: `opt-host-overhead-instrumented-20260520a`.

Config: L4/T4+40CPU, stock LightZero profile path, CPU oracle policy
observation, no death, sim8, batch32, no RND/eval/GIF, four learner calls,
env timing stride 64.

| manager | collectors | steps | wall sec | steps/sec | sub/base |
|---|---:|---:|---:|---:|---:|
| base | 8 | 4096 | 63.46 | 64.55 | - |
| subprocess | 8 | 4096 | 33.15 | 123.56 | 1.91x |
| base | 32 | 16384 | 191.81 | 85.42 | - |
| subprocess | 32 | 16384 | 45.75 | 358.15 | 4.19x |

Sampled per-env timing:

| manager | collectors | update stack | physical loop | base info | timestep pickle |
|---|---:|---:|---:|---:|---:|
| base | 8 | 6.699 ms | 1.960 ms | 0.244 ms | 0.127 ms |
| base | 32 | 6.965 ms | 2.147 ms | 0.271 ms | 0.142 ms |
| subprocess | 8 | 7.378 ms | 2.028 ms | 0.258 ms | 0.659 ms |
| subprocess | 32 | 12.052 ms | 4.429 ms | 1.442 ms | 0.187 ms |

Read:

- `update_stack_sec` is still the largest measured per-env component.
- `BaseEnvTimestep` payload is about `72.8 KB` of NumPy arrays and about
  `86.0 KB` pickled; pickle itself is not the obvious largest cost.
- Subprocess collection hides enough per-env work that C32 is about `4.2x`
  faster than base in this profile. The earlier coarse C64 row was about
  `5.4x` faster.
- This says the current low-hanging full-loop win is collector/process
  parallelism and eventually a vector/batched observation boundary, not tiny
  pickle/copy shaving.

Measurement caveat: env timing is sampled and subprocess timings are
worker-side. Treat component means as directional, not as a perfect wall-time
decomposition.

### RND Overhead Pair, First Attempt

Run id family: `opt-rnd-overhead-20260520a`.

Result:

- `rnd_meter_v0` row completed and proved the reward-model entrypoint still
  runs.
- The `none` row did not do useful work: it failed a metadata readiness check
  because the profile builder passed RND-only batch/update flags even when
  `exploration_bonus_mode=none`.
- The builder has been patched so disabled-RND commands omit RND-only flags.

Do not use this pair as an RND overhead comparison.

### Seeded Base/Subprocess Comparison

Runs:

- `opt-seeded-base-compare-s304a-20260520`
- `opt-seeded-base-compare-s304b-20260520`
- `opt-seeded-base-compare-s305-20260520`
- `opt-seeded-base-compare-rnd-s304-20260520`

Config: CPU profile, collectors 2, batch16, sim4, source max steps 256,
four learner train calls, no eval/GIF/checkpoint I/O.

| run | mode | manager | steps | wall sec | steps/sec | learner | replay | MCTS roots | sim budget |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| s304a | none | base | 317 | 12.76 | 24.84 | 4 | 4 | 317 | 664 |
| s304a | none | subprocess | 317 | 8.84 | 35.86 | 4 | 4 | 317 | 664 |
| s304b | none | base | 317 | 9.20 | 34.45 | 4 | 4 | 317 | 664 |
| s304b | none | subprocess | 317 | 8.83 | 35.91 | 4 | 4 | 317 | 664 |
| s305 | none | base | 314 | 13.15 | 23.88 | 4 | 4 | 314 | 664 |
| s305 | none | subprocess | 314 | 11.61 | 27.05 | 4 | 4 | 314 | 664 |
| s304 RND | rnd_meter_v0 | base | 0* | 12.30 | 0.00 | 4 | 4 | 317 | 664 |
| s304 RND | rnd_meter_v0 | subprocess | 0* | 12.60 | 0.00 | 4 | 4 | 317 | 664 |

`*` The RND rows reported zero `env_steps_collected` in the compact telemetry
even though MCTS root/simulation counts show collection work occurred. Treat
those rows as reward-model entrypoint canaries, not throughput rows, until the
RND compact step counter is fixed.

Read:

- Same-seed `none` rows matched workload counts exactly across duplicate
  manifests: env steps, learner calls, replay samples, MCTS calls, root count,
  and sim budget.
- Different seed changed trajectory length slightly (`317` to `314`) while
  preserving the static workload shape.
- Base and subprocess matched workload counts for these tiny rows.
- Timing is noisy at this size, but subprocess was not behaviorally divergent.

### Seed And RND Guardrails Added

- Profile runner now requires `--manifest`, validates the current Modal
  entrypoint, fixes numeric `--rows 1` selection to `001`, and writes missing
  call-id collect records.
- Profile builder labels RND rows with RND batch/update/weight and omits
  RND-only CLI flags for `mode=none`.
- Trainer process now seeds Python, NumPy, Torch CPU/CUDA, and deterministic
  Torch/cuDNN flags before LightZero/RND construction.
- RND config now receives a seed; `CurvyRNDRewardModel` uses seeded Torch
  initialization and a private seeded sampler instead of global `random.sample`.
- Focused local tests passed:
  `21 passed, 2 skipped` for RND/config/entrypoint seed checks.

### RND Cadence Concern

Open question: current RND training cadence may be far too low for positive
intrinsic reward. In stock LightZero's reward-model loop, the reward model
collects rollout data, trains at most once per collection wave when enough
buffer exists, then `estimate()` is called inside every learner update. With
large collector batches and a few collection waves, this can produce a tiny
number of predictor updates and many target estimates.

Second concern: current Curvy RND normalizes prediction error by min/max inside
each estimate batch. That makes the bonus a relative ranking within the sampled
batch, not a clearly globally decaying novelty signal.

This is a correctness/research lane. Do not mix it into speed conclusions until
we know the intended RND normalization contract.

2026-05-20 follow-up: the obvious cadence problem is patched. Current code
defaults to `rnd_update_per_collect=100`, seeds the RND target/predictor/sampler,
and logs raw MSE plus train/estimate ratio metrics. Positive-RND work still
needs a normalization decision because LightZero-style batch min/max makes the
bonus batch-relative.

2026-05-20 compact telemetry fix: profile compact output now keeps
`env_steps_collected_raw` and `env_steps_collected_source`. If the collector
envstep counter is zero, `lightzero_eval_freq=0`, profile stock eval skipping
is enabled, evaluator calls/skips are both zero, and MCTS root count is
positive, compact output uses `mcts_search_root_sum_profile_fallback` for the
displayed step count and `steps_per_sec`. This is only a profile fallback; full
summaries still retain the raw phase counts.

### RND Cadence Smoke Relaunch

Manifest:
`artifacts/local/curvytron_optimizer_profile_manifests/opt-rnd-cadence-smoke-20260520b.json`

Purpose: compare `none` versus `rnd_meter_v0` after the cadence/seed/telemetry
patches, without treating it as a learning proof.

Rows:

- compute: `gpu-l4-t4-cpu40`
- env manager: `base`, `subprocess`
- collectors/episodes: `8`
- batch size: `64`
- MCTS simulations: `4`
- source max steps: `512`
- stop after learner train calls: `4`
- eval/GIF/checkpoint frequency: disabled or effectively off for profile
- RND rows: `weight=0.0`, `rnd_batch_size=64`,
  `rnd_update_per_collect=100`, `require_rnd_metrics=true`

Expected read: use the row as a smoke for RND construction, predictor cadence,
raw MSE telemetry, and overhead. Do not compare it to live training quality.

Results:

| row | manager | RND | steps | step source | wall sec | steps/s | RND train/estimate |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 001 | base | none | 4096 | collector delta | 55.263 | 74.12 | |
| 002 | base | meter | 4096 | MCTS-root fallback | 57.810 | 70.85 | 100 / 4 |
| 003 | subprocess | none | 4096 | collector delta | 25.229 | 162.35 | |
| 004 | subprocess | meter | 4096 | MCTS-root fallback | 25.805 | 158.73 | 100 / 4 |

Read:

- Subprocess is still the large speed lever for this row shape: about `2.19x`
  faster than base for no-RND and `2.24x` faster for RND-meter.
- RND meter with `rnd_update_per_collect=100` adds only a small observed
  overhead in this smoke: about `4.4%` on base and `2.2%` on subprocess.
- RND metrics prove the predictor trained (`train_cnt_rnd=100`), was estimated
  four times, target rewards stayed unchanged at `weight=0.0`, and raw MSE
  stats were written.
- The RND rows use the explicit MCTS-root fallback because LightZero's
  reward-model path did not advance the collector envstep counter in compact
  phase counts. Compare `env_steps_collected_source` before comparing rates.

Quick external anchors:

- OpenAI's archived RND README describes scaling collection via MPI and many
  parallel envs; it is not a one-predictor-update-per-huge-run pattern.
- LightZero `train_muzero_with_reward_model` does:
  collect -> `reward_model.collect_data(new_data)` -> maybe
  `reward_model.train_with_data()` -> replay sample loop -> `reward_model.estimate(train_data)`.
  That confirms why train calls can be much lower than estimate calls.

### Subprocess Collector / Simulation Sweep

Manifest:
`artifacts/local/curvytron_optimizer_profile_manifests/opt-subprocess-collector-sims-sweep-20260520a.json`

Purpose: now that subprocess is clearly the practical full-loop speed lever,
measure whether the next limit is collector count, MCTS search, or learner/replay
when the trusted stock path runs no-death profile rows.

Rows:

- compute: `gpu-l4-t4-cpu40`
- env manager: `subprocess`
- collectors/episodes: `8`, `16`, `32`, `64`
- MCTS simulations: `2`, `4`, `8`
- batch size: `64`
- source max steps: `512`
- stop after learner train calls: `4`
- eval/GIF/checkpoint frequency: disabled or effectively off for profile
- RND: off

Expected read:

- If C64 is much faster than C32 at all sim counts, keep scaling collectors.
- If sim8 collapses throughput and MCTS seconds dominate, search/GPU batching is
  the next lane.
- If C64 flattens or regresses while MCTS stays small, focus on collector/env
  orchestration and payload/stack overhead.
- If learner/replay seconds grow with collector count, batch/learner cadence may
  be the next practical knob.

Results:

| collectors | sim2 steps/s | sim4 steps/s | sim8 steps/s |
| ---: | ---: | ---: | ---: |
| 8 | 194.90 | 177.42 | 129.68 |
| 16 | 268.52 | 253.85 | 206.34 |
| 32 | 461.91 | 380.95 | 326.97 |
| 64 | 601.24 | 557.26 | 452.41 |

Timer read:

- Best tested L4/T4 row is C64/sim2: `601.24 steps/s`.
- C64/sim4 is still strong: `557.26 steps/s`.
- C64/sim8 drops to `452.41 steps/s`; MCTS grows to `20.51s` while learner is
  only `2.28s`.
- Collector scaling still helps through C64, so do not stop at C32.
- Simulation count matters. At C64, sim8 is about `25%` slower than sim2; at
  C8, sim8 is about `33%` slower than sim2.

Current read: the next wall is not learner/replay. It is collector/search
interaction: more collectors still help, while higher MCTS simulation count
costs enough that H100/search-heavy rows are worth checking.

### H100 Search-Heavy Sweep

Manifest:
`artifacts/local/curvytron_optimizer_profile_manifests/opt-h100-search-heavy-sweep-20260520a.json`

Purpose: run a parallel high-load check while the L4/T4 sweep collects. This
tests whether H100 materially improves the search-heavy regime once collectors
and simulations are high.

Rows:

- compute: `gpu-h100-cpu40`
- env manager: `subprocess`
- collectors/episodes: `64`, `128`, `256`
- MCTS simulations: `8`, `16`
- batch size: `64`
- source max steps: `512`
- stop after learner train calls: `4`
- eval/GIF/checkpoint frequency: disabled or effectively off for profile
- RND: off

Expected read:

- If H100 only marginally improves steps/s relative to L4/T4 at the same high
  sim rows, the bottleneck is not raw GPU compute.
- If H100 materially improves sim16 rows while C scaling remains healthy, search
  compute is a valid next scaling knob.
- If C128/C256 regress even on H100, CPU/process orchestration is probably the
  next wall.

Results:

| collectors | sim8 steps/s | sim16 steps/s |
| ---: | ---: | ---: |
| 64 | 597.24 | 457.10 |
| 128 | 681.88 | 605.53 |
| 256 | 829.76 | 553.54 |

Read:

- Best tested row so far is H100 C256/sim8: `829.76 steps/s`.
- H100 C64/sim8 is `597.24 steps/s`, about `1.32x` the L4/T4 C64/sim8 row
  (`452.41 steps/s`).
- H100 C256/sim16 regressed to `553.54 steps/s`, so sim16 is not a free win.
- H100 C128/sim16 beat C64/sim16, but C256/sim16 got worse; this points toward
  a mixed search/orchestration ceiling rather than pure GPU compute.

Follow-up launched:
`artifacts/local/curvytron_optimizer_profile_manifests/opt-l4-high-collector-compare-20260520a.json`

Purpose: compare L4/T4 at C128/C256 sim8/sim16 so the H100 result is not
confused with "more collectors" alone.

Follow-up results:

| collectors | sims | L4/T4 steps/s | H100 steps/s | H100 / L4 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 433.86 | 681.88 | 1.57x |
| 128 | 16 | 338.68 | 605.53 | 1.79x |
| 256 | 8 | 528.81 | 829.76 | 1.57x |
| 256 | 16 | 494.07 | 553.54 | 1.12x |

Read:

- H100 materially helps high-load sim8 and C128/sim16 rows.
- H100 does not rescue C256/sim16; that row still regresses versus C256/sim8,
  which points to a mixed search/orchestration ceiling.
- Current best tested speed is H100 C256/sim8 at `829.76 steps/s`.
- Best cheap setting tested is L4/T4 C64/sim2 at `601.24 steps/s`, or L4/T4
  C64/sim4 at `557.26 steps/s` if sim4 is the minimum acceptable proof search.

### H100 Topology Follow-Ups

Launched:

- `artifacts/local/curvytron_optimizer_profile_manifests/opt-h100-throughput-topology-20260520a.json`
  - H100 C128/C256 at sim2/sim4.
- `artifacts/local/curvytron_optimizer_profile_manifests/opt-h100-c512-probe-20260520a.json`
  - H100 C512 at sim4/sim8.

Purpose: find the current top throughput point without sim16 overreach and test
whether C512 gives more useful parallelism or simply exposes CPU/process
orchestration limits.

H100 C128/C256 sim2/sim4 results:

| collectors | sim2 steps/s | sim4 steps/s |
| ---: | ---: | ---: |
| 128 | 734.72 | 863.49 |
| 256 | 885.35 | 876.81 |

Read:

- New best tested row is H100 C256/sim2: `885.35 steps/s`.
- H100 C256/sim4 is effectively tied at `876.81 steps/s`, so sim4 is a very
  practical default if it gives better search quality than sim2.
- H100 C128/sim4 beat C128/sim2, likely from run noise and/or better GPU/search
  batching at this shape. Do not overread a single row.
- Compared with H100 C256/sim8 (`829.76 steps/s`), sim4 only gains about `6%`,
  while sim16 loses heavily. The danger zone starts around sim16, not sim4.

H100 C512 probe results:

| collectors | sim4 steps/s | sim8 steps/s |
| ---: | ---: | ---: |
| 512 | 1061.28 | 825.71 |

Read:

- New best tested row is H100 C512/sim4: `1061.28 steps/s`.
- C512/sim8 falls back to roughly the C256/sim8 range, so sim4 is the current
  best throughput/search compromise in these profile rows.

Follow-up stress launched:

- `opt-h100-collector-stress-20260520a`: C512/sim2.
- `opt-h100-collector-stress-c768-c1024-20260520a`: C768/C1024 at sim4.

Purpose: check whether C512/sim4 is the current top point or whether more
collector parallelism still helps.

C512/sim2 result:

| collectors | sims | steps/s |
| ---: | ---: | ---: |
| 512 | 2 | 952.11 |

Read: C512/sim2 is slower than C512/sim4 (`1061.28 steps/s`), so lower search
count alone is not the throughput answer. C512/sim4 remains the best tested row
while C768/C1024 sim4 collect.

C768/C1024 sim4 results:

| collectors | sims | steps/s |
| ---: | ---: | ---: |
| 768 | 4 | 1054.44 |
| 1024 | 4 | 852.73 |

Read:

- C768/sim4 is essentially tied with C512/sim4 but slightly slower.
- C1024/sim4 regresses hard.
- Current top tested point remains H100 C512/sim4 at `1061.28 steps/s`.
- The next speed wall is now collector/process/observation-stack orchestration,
  not learner/replay. More collectors past C512 do not improve this profile.

## New Instrumentation For Next Runs

- RND methods are now timed at the existing phase-profiler boundary:
  `collect_data`, `train_with_data`, `estimate`, metrics snapshot/write, and
  state hash.
- Registered LightZero env profile rows now record BaseEnvTimestep construction
  time, pickle time, pickle bytes, and contained NumPy array bytes.
- Local env profile timing now splits observation into stack update, stack copy,
  action-mask copy, base-info build, and terminal final-observation copy.

## Mock Collector Scaffold

Added local profile-only tool:

```text
scripts/profile_batched_observation_mock_collector.py
```

It measures the next missing boundary:

```text
Vector env step
-> batched observation facade
-> row/player [B,P,4,64,64] stacks
-> scalar LightZero-shaped rows
-> pickle/payload proxy
-> optional RND latest-frame collect/train/estimate
```

Local smokes:

| config | measured rows | rows/s | read |
| --- | ---: | ---: | --- |
| B2, steps2, RND off | 8 | 262.75 | scaffold runs and emits payload timing |
| B4, steps2, RND meter on, batch2/update1 | 16 | 160.46 | RND collect/train/estimate runs on latest policy frames |
| B8, steps8, RND off | 128 | 314.47 | slightly steadier local scaffold row; facade step dominates |

These are local CPU scaffold numbers, not Modal/H100 recommendations. Their
value is contract coverage: exact pixel parity is not the gate; row/player
order, stack order, final observation, no hidden fallback, and RND latest-frame
wiring are the gate.

Validation:

```text
uv run pytest tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_source_state_batched_observation_profile_cpu.py \
  tests/test_source_state_batched_observation_boundary_profile.py -q
```

Result: `30 passed, 2 skipped`. The skipped tests are RND/Torch tests in the
local `uv` environment when Torch is absent; the same RND script path ran in the
plain local Python environment with Torch available.

## H100 Boundary With Mock Collector Payload

Command shape:

```text
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --compute gpu-h100 \
  --batch-size 64 \
  --steps 8 \
  --warmup-steps 2 \
  --trail-slots 1024 \
  --geometry-dtype float32 \
  --parity-mode tolerant \
  --verify-steps 2 \
  --include-lightzero-payload-profile
```

Result:

| bucket | median sec | p95 sec | read |
| --- | ---: | ---: | --- |
| candidate total observation | 0.2572 | 0.2593 | GPU boundary render+readback+stack |
| env step + observation | 0.2612 | 0.2647 | physics is small in this no-death row |
| mock collector total | 0.2638 | 0.2683 | payload adds only about 2ms |
| device render | 0.2458 | 0.2459 | still the dominant bucket |
| host to device | 0.00438 | 0.00559 | visible but not the wall |
| stack update | 0.00241 | 0.00503 | visible but not the wall |
| scalarize LightZero rows | 0.00021 | 0.00029 | tiny |
| pickle LightZero payload | 0.00176 | 0.00441 | tiny relative to render |
| CPU reference render+stack | 0.7433 | 0.7939 | GPU boundary is about 2.9x faster than full CPU reference render+stack |

Payload size was about `8.39MB` per B64/P2 step. Parity was exact in the
checked tolerant row. Plain read: the missing LightZero-shaped row materializer
is not the Amdahl wall; the batched GPU boundary is still dominated by the
device render bucket itself.

## H100 Boundary Trail-Slot Ladder

Command shape: H100, B64, steps8, warmup2, float32 tolerant, verify1,
mock-collector payload included.

| trail slots | mock collector total median sec | observation median sec | env step + obs median sec | device render median sec | CPU ref render+stack median sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0453 | 0.0413 | 0.0434 | 0.0315 | 0.7490 |
| 256 | 0.0750 | 0.0710 | 0.0732 | 0.0621 | 0.7504 |
| 512 | 0.1372 | 0.1324 | 0.1353 | 0.1224 | 0.7077 |
| 1024 | 0.2607 | 0.2553 | 0.2588 | 0.2444 | 0.7589 |

Read:

- Device render scales roughly with configured trail slots.
- S128 is about `5.8x` faster than S1024 for the candidate mock-collector
  boundary.
- Scalar row materialization, pickle, stack update, and host/device transfer
  are not the main wall in this row shape.
- Important caveat: in this profiler revision, `trail_slots` is both env body
  capacity and render width. These rows prove that trail width is a major
  render-cost lever, but they do not yet prove that we can run a large-capacity
  env while rendering only the active prefix.

Next profiler patch: report active trail count/prefix after owner-ordered
packing. That tells us whether real profiles pay for mostly inactive tail
slots. Only then should we split env `body_capacity` from render
`trail_slots`.

### Active-Prefix Instrumentation Row

Same shape as the B64/S1024 payload row, after adding active trail stats.

Result:

- mock collector total median `0.2613s`
- observation median `0.2557s`
- device render median `0.2446s`
- CPU reference render+stack median `0.7398s`
- median active trail count `15`
- p95 active trail count `22`
- median active fraction `0.0146`
- p95 active fraction `0.0215`

Read: this profile paid S1024 render cost while using only about `1.5%` to
`2.1%` of the trail slots. That makes active-prefix/render-width reduction the
highest-signal renderer optimization to test. It also explains why payload,
pickle, and stack shaving were not the right next target.

Follow-up patch: boundary profiles can now keep env `body_capacity` high while
using a smaller render `trail_slots`. This is still profile-only. The point is
to prove speed and parity for the same large-capacity env before any trainer
wiring.

### Decoupled Body Capacity / Render Width Rows

Command shape: H100, B64, steps8, warmup2, env `body_capacity=1024`, float32
tolerant, verify all 8 measured steps, mock-collector payload included.

| render slots | result | mock collector median sec | observation median sec | device render median sec | active p95 | truncation p95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | failed step 5 parity | 0.0209 before failure | 0.0143 before failure | 0.0048 before failure | 22 | active rows dropped |
| 32 | passed tolerant | 0.0281 | 0.0197 | 0.0086 | 22 | 0 |
| 64 | passed tolerant | 0.0347 | 0.0272 | 0.0161 | 22 | 0 |
| 1024 | passed tolerant | 0.2613 | 0.2557 | 0.2446 | 22 | 0 |

Read:

- The clean version keeps full env capacity, packs all active trails, then caps
  render width. This avoids the earlier fake win from lowering env capacity.
- S16 is not safe for this short trajectory because it starts dropping active
  trails by step 5.
- S32 is the best tested fixed render width for this row: exact/tolerant
  semantic checks pass, no active rows are dropped, and the candidate boundary
  is about `9.3x` faster than S1024 (`0.0281s` versus `0.2613s`).
- This is still a boundary win, not a full-loop win. Amdahl says the full-loop
  gain depends on how much of the H100 C512/sim4 stock loop is observation
  boundary work.
- For longer trajectories, a fixed S32 will eventually fail. The likely
  production shape is a dynamic power-of-two render width such as
  `max(32, ceil_power_of_two(active_trail_count_p95_or_max))`, with a
  fail-closed no-truncation check.

### Dynamic Render-Width Row

Command shape: H100, B64, steps8, warmup2, env `body_capacity=1024`, max render
`trail_slots=1024`, `dynamic_render_trail_slots=true`, `min_render_trail_slots=32`,
float32 tolerant, verify all 8 measured steps, mock-collector payload included.

Result:

- selected render width median/p95: `32`
- mock collector median `0.0256s`
- observation median `0.0188s`
- device render median `0.0085s`
- env step + observation median `0.0223s`
- CPU reference render+stack median `0.6490s`
- active p95 `22`
- truncation p95 `0`
- parity passed all 8 measured checks with only the known one-luma edge drift

Read:

- This is the strongest renderer-side profile result so far.
- Compared with fixed S1024 at `0.2613s`, dynamic S32 is about `10.2x` faster
  at the candidate mock-collector boundary for this short trajectory.
- Compared with fixed S32, dynamic is slightly faster/noisier but effectively
  the same shape. The important win is not magic dynamic logic; it is that
  dynamic logic can keep this speed on short games and automatically grow for
  longer games.
- This still must be tested in a real full-loop A/B before claiming trainer
  speedup.

Follow-up guard: render truncation is now fail-closed by default in this
profile-only boundary. A lossy diagnostic row must explicitly set
`allow_render_truncation=true`. This keeps fixed S16-style mistakes from being
confused with a valid speed candidate.

Guarded rerun:

- dynamic min32/max1024
- `allow_render_truncation=false`
- passed all 8 measured checks
- selected render width median/p95 `32`
- truncation p95 `0`
- mock collector median `0.0271s`

Read: the fail-closed guard did not change the conclusion. The dynamic boundary
still gives about a `10x` candidate-boundary win over fixed S1024 in this short
profile row.

Longer trajectory first attempt:

- B64, steps64, warmup4, dynamic min32/max1024, fail-closed truncation
- strict tolerant threshold: max abs diff `2`, mismatch fraction `0.0001`
- failed at measured step 7 with only a few pixels differing, but max luma drift
  reached `4`

Read: this was not a truncation failure and not a row/order/stack failure. It
is the known learned-observation float32/GPU luma drift getting a little larger
over longer trajectories. A follow-up row uses max abs diff `8` and mismatch
fraction `0.001` to capture long-trajectory timing while still checking for
gross semantic mistakes.

Longer trajectory learned-observation row:

- B64, steps64, warmup4, dynamic min32/max1024, fail-closed truncation
- max abs diff `8`, mismatch fraction `0.001`
- passed 16 measured checks
- selected render width median `128`, p95 `256`
- active trail count median `75`, p95 `134`
- truncation p95 `0`
- mock collector median `0.0456s`, p95 `0.0797s`
- device render median `0.0311s`, p95 `0.0615s`
- CPU reference render+stack median `1.0769s`

Read: dynamic width grows as the trajectory grows, which is the intended
behavior. Even in this longer row, candidate boundary cost stayed low relative
to the full CPU reference path. A matched fixed-S1024 long row is running so the
next comparison is same-length dynamic versus same-length fixed render width.

Matched long rows:

| row | steps | render width | active p95 | truncation p95 | mock collector median sec | device render median sec |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| fixed | 64 | 1024 | 134 | 0 | 0.2599 | 0.2443 |
| dynamic | 64 | median 128, p95 256 | 134 | 0 | 0.0456 | 0.0311 |
| dynamic | 128 | median 256, p95 512 | 263 | 0 | 0.0751 | 0.0616 |

Read:

- Same-length dynamic versus fixed S1024 at 64 steps is about `5.7x` faster at
  the candidate boundary.
- Dynamic 128-step cost grows as active trails grow, but still stays far below
  fixed S1024. It selected median S256 and p95 S512 with no truncation.
- The active-prefix idea is not just a short-trajectory artifact. It degrades
  gracefully as games get longer.
- The current long profiler spends a lot of wall time on CPU reference renders.
  That is fine for proof rows, but future long candidate timing should reduce
  reference cadence once semantics are already checked.

Integration critique:

- Quick stock `train_muzero` wiring is not a clean flag flip. The trusted stock
  path uses many scalar LightZero envs, and the old scalar `jax_gpu` backend
  renders one env row at a time.
- The next A/B should be a profile-only vector facade: one vector env batch,
  dynamic batched render, stack update, LightZero-shaped scalar row
  materialization, payload timing, and optional RND meter.
- Use current H100 C512/sim4 CPU-oracle stock throughput (`~1061 steps/s`) as
  the real full-loop control once the vector facade canary exists.

## H100 Boundary With RND Meter Smoke

First attempt failed because the lean JAX renderer image did not include Torch.
The boundary module now uses a separate torch-capable Modal image only when
`--include-rnd-meter` is passed, so normal renderer probes stay lean.

Command shape:

```text
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --compute gpu-h100 \
  --batch-size 16 \
  --steps 4 \
  --warmup-steps 1 \
  --trail-slots 1024 \
  --geometry-dtype float32 \
  --parity-mode tolerant \
  --verify-steps 1 \
  --include-lightzero-payload-profile \
  --include-rnd-meter \
  --rnd-batch-size 16 \
  --rnd-update-per-collect 1
```

Result:

| bucket | median sec | p95 sec | read |
| --- | ---: | ---: | --- |
| candidate total observation | 0.0827 | 0.0828 | B16 GPU boundary |
| mock collector total | 0.1138 | 0.1140 | RND update1 adds visible overhead |
| device render | 0.0780 | 0.0782 | still the largest bucket |
| RND collect | 0.00037 | 0.00044 | small |
| RND estimate | 0.00325 | 0.00366 | moderate |
| RND train update1 | 0.0252 | 0.0257 | largest RND component |

RND metrics showed collect/train/estimate all ran and target reward was not
mutated because this was meter mode with zero intrinsic weight. Plain read:
RND meter can now be profiled at the boundary, but real positive RND is still a
separate learning/normalization decision.

## Candidate-Only Reference Cadence Tooling

Patch result:

- `source_state_batched_observation_boundary_profile.py` now has
  `cpu_reference_interval`.
- The safe default is still `1`, which renders the CPU oracle every measured
  step and keeps proof rows aligned.
- Candidate-only timing rows may set `verify_steps=0` and
  `cpu_reference_interval=0`. This keeps the reset smoke/parity check, then
  skips the expensive CPU reference path during the measured loop.
- The validator rejects `cpu_reference_interval != 1` when `verify_steps > 0`
  so we cannot accidentally run partial parity with a stale reference stack.

Validation:

```text
python -m py_compile src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
uv run pytest tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_source_state_batched_observation_profile_cpu.py -q
```

Result: `46 passed, 2 skipped`.

Active row:

```text
uv run --extra modal modal run \
  -m curvyzero.infra.modal.source_state_batched_observation_boundary_profile \
  --compute gpu-h100 \
  --batch-size 64 \
  --steps 256 \
  --warmup-steps 16 \
  --trail-slots 1024 \
  --body-capacity 1024 \
  --dynamic-render-trail-slots \
  --min-render-trail-slots 32 \
  --geometry-dtype float32 \
  --parity-mode tolerant \
  --parity-max-abs-diff 8 \
  --parity-max-mismatch-fraction 0.001 \
  --verify-steps 0 \
  --cpu-reference-interval 0 \
  --include-lightzero-payload-profile
```

Question this row answers: once the semantic proof rows are already done, how
fast is the candidate dynamic GPU boundary by itself on a longer no-death
trajectory?

Result:

- ok: `true`
- JAX backend: `gpu`, one H100
- parity: reset checked, measured step parity intentionally skipped
  (`verify_steps=0`)
- selected render width median/p95: `512` / `512`
- active trail count median max/p95 max: `292` / `506`
- truncation median/p95: `0`
- candidate total mock collector median/p95: `0.1348s` / `0.1379s`
- candidate total observation median/p95: `0.1309s` / `0.1333s`
- device render median/p95: `0.1227s` / `0.1232s`
- host-to-device median: `0.0028s`
- device-to-host median: `0.00026s`
- env step median: `0.0030s`
- stack median: `0.00135s`
- LightZero scalarize median: `0.00013s`
- LightZero payload pickle median: `0.00098s`

Read:

- At 256 no-death steps, dynamic active-prefix rendering grows to S512. That is
  expected: longer survival creates more active trail segments.
- The candidate boundary is still much cheaper than fixed S1024 would be, but
  the long-trajectory Amdahl wall inside this boundary is again the device
  render itself. The small buckets are not the next target.
- Host transfer is not the practical bottleneck in this row. It is milliseconds
  or less compared with roughly `0.123s` device render.
- This row is a timing row, not a new semantic proof row. It relies on the
  earlier proof rows for row/player/stack semantics and uses reset parity only.

## Trainer-Visible Renderer-Backed Surface Hook

Patch result:

- `SourceStateMultiplayerTrainerSurface` now has an explicit
  `observation_stack_backend` seam.
- Default remains `cpu_dirty_cache`, which is the current trusted CPU
  `SourceStateGray64Stack4` path.
- The profile canary backend is `renderer_backed_profile`.
- `renderer_backed_profile` requires an explicit `observation_renderer`; there
  is no hidden CPU fallback.
- The renderer-backed stack calls the batched renderer once per surface update
  for row-major `(row, player)` views, then writes `[B,P,4,64,64]` stacks into
  the same trainer surface contract.
- This still does not touch stock `train_muzero`, live runs, tournaments,
  checkpoints, or default trainer settings.

Validation:

```text
python -m py_compile src/curvyzero/training/multiplayer_source_state_trainer_surface.py
uv run pytest tests/test_multiplayer_source_state_trainer_surface.py \
  tests/test_source_state_batched_observation_profile_cpu.py -q
```

Result: `29 passed`.

Read:

- This is the smallest clean trainer-visible canary seam. It proves the
  multiplayer trainer surface can consume a batched renderer without changing
  reward, policy row mapping, final observation, or live-mask behavior.
- The real dynamic GPU renderer is still only in the Modal boundary profiler.
  The next implementation step is to plug that renderer into this seam in a
  profile-only Modal canary.
- A direct stock `train_muzero` flag remains the wrong next step because stock
  scalar env managers would destroy the render batch before the GPU renderer can
  help.

Follow-up patch:

- The Modal boundary profile now has `--surface-facade-canary`.
- That mode runs `SourceStateMultiplayerTrainerSurface` with
  `observation_stack_backend=renderer_backed_profile` and the dynamic JAX GPU
  batched renderer.
- It still does not call `train_muzero`; this is a trainer-surface canary, not a
  live training run.

First H100 surface canary row:

- B64, steps64, warmup8, body1024, dynamic min32/max1024
- policy rows per step: `128`
- render width median/p95: `128` / `256`
- active trail max median/p95: `83` / `140`
- truncation median/p95: `0`
- surface step total median/p95: `0.0561s` / `0.1015s`
- renderer total median/p95: `0.0399s` / `0.0715s`
- device render median/p95: `0.0316s` / `0.0625s`
- host-to-device median: `0.00418s`
- device-to-host median: `0.00039s`
- LightZero scalarize median: `0.00020s`
- payload pickle median: `0.00177s`

Read:

- This is the first successful trainer-visible dynamic GPU observation canary.
- It lines up with the earlier boundary row: the dynamic render remains the
  largest local bucket, but the surface/policy-row wrapper adds real overhead on
  top of the pure boundary.
- The row is still profile-only. The next matched row is steps256 so we can
  compare long-boundary and long-surface costs.

Matched long H100 surface canary row:

- B64, steps256, warmup16, body1024, dynamic min32/max1024
- policy rows per step: `128`
- render width median/p95: `512` / `512`
- active trail max median/p95: `292` / `498`
- truncation median/p95: `0`
- surface step total median/p95: `0.1442s` / `0.1582s`
- renderer total median/p95: `0.1303s` / `0.1319s`
- device render median/p95: `0.1225s` / `0.1230s`
- host-to-device median: `0.00342s`
- device-to-host median: `0.00038s`
- LightZero scalarize median: `0.00019s`
- payload pickle median: `0.00162s`

Read:

- This matches the candidate-only boundary row at the same trajectory length:
  the surface wrapper adds overhead, but not a new dominant bucket.
- At longer survival, the dynamic renderer chooses S512 and the device render
  is again the Amdahl wall inside the observation/surface canary.
- Host transfer, scalarization, and pickle are not the next target in this row.
- This still is not a full training-loop result. It does not include LightZero
  search, replay, learner, checkpointing, eval, or tournament jobs.

Small surface RND meter smoke:

- B16, steps4, warmup1, dynamic min32/max1024
- RND enabled with update-per-collect `1`, CPU RND device
- ok: `true`
- policy rows per step: `32`
- render width median/p95: `32` / `32`
- surface step total median/p95: `0.0164s` / `0.0190s`
- renderer total median: `0.00885s`
- RND collect median: `0.00053s`
- RND estimate median: `0.00431s`
- RND train median: `0.0288s`

Read:

- RND hooks run on the trainer-visible surface canary path.
- This is a plumbing smoke, not a throughput recommendation. At this tiny B16
  row, one RND train update is larger than the renderer step.
- A real RND throughput row must use the coach's actual update cadence and
  should be compared separately from renderer-only rows.

Matched CPU Dirty-Cache Surface Control:

- B64, steps256, warmup16, same no-death surface loop
- `surface_stack_backend=cpu_dirty_cache`
- policy rows per step: `128`
- surface step total median/p95: `0.2375s` / `0.2606s`
- LightZero scalarize median: `0.00018s`
- payload pickle median: `0.00130s`

Comparison against the matched GPU renderer-backed surface row:

| row | surface step median sec | read |
| --- | ---: | --- |
| CPU dirty-cache surface | `0.2375` | current optimized CPU visual stack |
| GPU renderer-backed surface | `0.1442` | dynamic JAX render, S512, no truncation |

Read:

- The trainer-visible GPU observation canary is about `1.65x` faster than the
  optimized CPU dirty-cache surface on this long no-death row.
- This is the fairer comparison than GPU versus the old full CPU oracle.
- It is still only an observation/surface loop, not full LightZero training.
- Inside the GPU surface row, device render is the remaining local wall:
  `0.1225s` of `0.1442s`. If we made GPU device render free, the absolute
  ceiling for this surface canary would be roughly `6.8x`, but full training
  will have lower headroom once search/learner/replay are included.

## 2026-05-21 Hybrid Renderer-Backed Profile Hook

Patch status:

- `source_state_hybrid_observation_profile.py` now supports an injected
  renderer-backed observation mode in addition to zero stacks.
- `scripts/profile_hybrid_batched_observation_manager.py` exposes
  `--observation-mode zero|cpu-oracle`.
- The CPU oracle path is explicit and named. There is no hidden fallback to CPU
  for a future GPU row.
- This still does not call `train_muzero`, change trainer defaults, change
  tournament defaults, or touch live runs.

Local tests:

- `uv run python -m py_compile scripts/profile_hybrid_batched_observation_manager.py src/curvyzero/training/source_state_hybrid_observation_profile.py`
- `uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py`
- `uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_curvytron_optimizer_profile_manifest_runner.py tests/test_curvytron_profile_grid_builder.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_mock_collector.py`

Result:

- `6 passed` for the hybrid profile tests.
- `64 passed, 2 skipped` for the focused optimizer/profile set.

Local smoke rows:

| mode | batch | actors | steps/warmup | scalar timesteps/s | physical rows/s | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cpu-oracle` | 4 | 2 | 2/1 | `230.06` | `115.03` | interface smoke only; CPU rendering is slow |
| `zero` | 64 | 4 | 40/10 | `24297.56` | `12148.78` | topology-only ceiling, no render/policy/search |

Correctness gate added:

- A sentinel renderer fills each `(row, player)` frame with a unique value.
- The test asserts row-major scalar order:
  `[(0,0),(0,1),(1,0),(1,1),...]`.
- This protects against the easiest silent bug: observations shaped correctly
  but assigned to the wrong player or row.

Read:

- This is a profile seam, not a speed recommendation.
- The next useful GPU step is to construct the real dynamic JAX renderer outside
  the training module and inject it through this seam in a Modal/profile-only
  harness.
- Do not wire this into Coach training until the GPU-injected hybrid profile
  beats the current stock/full-loop anchor and passes terminal/final-observation
  gates.

## 2026-05-21 Modal Hybrid GPU Renderer Smoke

Code/tooling fix:

- The hybrid profiler previously divided measured steps by wall time that
  included warmup/JIT. That made GPU rows look artificially slow.
- `run_hybrid_observation_profile` now reports `total_sec`, `warmup_sec`, and
  `measured_sec`, and `steps_per_sec` uses only `measured_sec`.
- Modal hybrid output is compact by default so B256/B512 rows do not print every
  scalar row.

Validation:

- `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_hybrid_observation_profile.py`
- Result: `48 passed`.

H100 profile-only rows:

| row | shape | measured sec | scalar steps/s | physical rows/s | observation sec | renderer render sec | device render sec | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | B16/A4, steps4/warmup3 | `0.0487` | `2630.55` | `1315.28` | `0.0295` | `0.0188` | `0.00386` | wrapper alive, too small for launch read |
| hybrid real | B256/A8, steps20/warmup5 | `2.2775` | `4496.19` | `2248.09` | `1.9180` | `1.3728` | `1.0821` | first meaningful real-render hybrid row |
| hybrid real | B512/A16, steps20/warmup5 | `3.7600` | `5446.80` | `2723.40` | `3.0400` | `1.7175` | `1.1762` | wider row improves, but observation remains dominant |
| hybrid real + compact pickle | B512/A16, steps20/warmup5 | `3.3030` | `6200.51` | `3100.25` | `2.6831` | `1.5515` | `1.0334` | payload pickle tiny; speed delta is likely runtime noise |
| hybrid real | B1024/A16, steps20/warmup5 | `6.1486` | `6661.70` | `3330.85` | `4.9929` | `2.3884` | `1.2888` | wider still improves, not saturated; scalarization grows |

Interpretation:

- The Modal injection path works: `renderer_backend_name=jax_gpu_batched_profile`,
  `jax.default_backend=gpu`, `calls_train_muzero=false`, and
  `touches_live_runs=false`.
- B512/A16 clears the rough hybrid pass bar from the next-wave plan
  (`>2166 scalar steps/s`) in this no-training harness.
- This is **not** a training speed number. It excludes policy/search/replay,
  learner, RND, subprocess IPC, checkpoints, eval, and tournament.
- The "actor" count is not true parallelism yet. These are in-process actor
  partitions stepped sequentially by the profile manager. They validate row
  partitioning and central render shape, not subprocess fan-in performance.
- The current hybrid real-render rows are still observation-heavy:
  B512 observation was `3.04s` of `3.76s` measured wall; render itself was
  `1.72s`, device render `1.18s`.
- B1024 improves aggregate throughput again, but observation is still the
  largest bucket (`4.99s` of `6.15s`) and scalar materialization grows to
  `0.088s`. This points at batch-size scaling headroom plus a future
  scalarization/policy boundary wall.
- Compact payload pickle is not the visible wall in the paired row:
  about `0.0019s` total for 20 steps, `~21 bytes` per scalar timestep, and
  only `0.032%` of the rendered-stack bytes. The faster pickle-on wall time
  should be treated as runtime noise, not a serialization win.
- GPU utilization from `nvidia_smi` is still low (`~14%` max snapshot on B512),
  so the H100 is not saturated in this profile. Bigger batches, fewer host
  copies, or keeping more downstream work batched may matter more than a larger
  GPU.

Next reads:

- Run a paired B512/A16 pickle-on row to measure compact payload cost.
- Run a small Modal GPU terminal/autoreset row now that local
  `final_observation` semantics exist.
- Add a clearly labeled synthetic policy/search pressure probe. The current
  real-render rows only prove env+central-render+scalarization shape.

Terminal gate update:

- The hybrid actor now snapshots post-step source state before autoreset.
- Terminal rows attach `final_observation` from the rendered terminal stack.
- After materializing the returned timestep, autoreset rows reset the internal
  stack for the next step.
- Local CPU-oracle terminal smoke with `max_ticks=1` passed:
  `terminal_row_count=2`, `autoreset_row_count=2`,
  `renderer_backend_name=cpu_oracle`.
- Focused tests after this patch: `67 passed, 2 skipped`.

Policy/search probe and guardrail update:

- The hybrid profile now accepts an injected policy/search pressure probe.
- The current Modal probe is explicitly synthetic:
  `policy_search_probe_semantics=synthetic_jax_conv_policy_pressure`. It is
  repeated JAX conv/pool/logit-like work over batched `[N,4,64,64]` roots, not
  LightZero MCTS and not learning evidence.
- Compact Modal output now includes probe backend, semantics, root count, input
  shape/dtype/bytes, and last timing telemetry.
- The scalar timestep materializer now preserves supplied action masks instead
  of always emitting all-true masks. The hybrid profile passes the merged actor
  mask through.
- Validation after the patch:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_mock_collector.py tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py`
  -> `66 passed, 2 skipped`.

Tiny H100 synthetic-probe smoke:

- Shape: B16/A4, steps4/warmup3, `direct_gray64`, no pickle payload, probe
  simulations `2`, channels `16`.
- Result: `ok=true`, `profile_only=true`, `calls_train_muzero=false`,
  `jax.default_backend=gpu`.
- Measured wall: `0.0556s`.
- Throughput: `2302.37` scalar steps/s, `1151.18` physical rows/s.
- Observation: `0.0309s`; renderer render `0.0188s`; device render `0.00431s`.
- Probe: `0.00517s` total over four measured steps, with last-call telemetry
  H2D `0.00079s`, device `0.00028s`, readback `0.00018s`.
- Read: the probe wiring works, but this row is too small for architecture
  conclusions. Run B512/B1024 rows before changing the Amdahl read.

Modal terminal/autoreset GPU row:

- Shape: B16/A4, steps4/warmup3, `max_ticks=1`, `direct_gray64`, no probe.
- Result: `terminal_row_count=64`, `autoreset_row_count=64`,
  `done_rows=64`, compact row/player heads and tails present.
- The stale gap string is gone. Modal now reports:
  `terminal final_observation is modeled for the profile scaffold, but
  natural-death trainer semantics are not claimed`.
- Measured wall: `0.2661s`; throughput: `481.07` scalar steps/s,
  `240.54` physical rows/s.
- Read: terminal/autoreset rows work on the Modal GPU path, but reset-heavy
  throughput is much slower because every step forces terminal snapshot plus
  reset-stack work. This is a correctness gate, not a no-death throughput row.

B512/B1024 synthetic policy/search pressure rows, sim2:

| row | shape | measured sec | scalar steps/s | physical rows/s | observation sec | probe sec | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| hybrid real + probe sim2 | B512/A16, steps20/warmup5 | `3.6880` | `5553.10` | `2776.55` | `2.8229` | `0.2024` | probe is visible but not the wall |
| hybrid real + probe sim2 | B1024/A16, steps20/warmup5 | `5.6444` | `7256.80` | `3628.40` | `4.4420` | `0.3510` | wider still helps; observation dominates |

Read:

- The synthetic probe is not heavy enough at sim2 to dominate. At B1024 it is
  about `6.2%` of measured wall (`0.351s / 5.644s`), while observation/stack is
  about `78.7%`.
- B1024 with sim2 is faster than the previous B1024 no-probe row. Treat that as
  runtime variance and/or JIT/cache effects, not proof that adding probe work
  makes the system faster.
- H2D dominates the synthetic probe timing at these shapes. That supports the
  broader warning: if model/search consumes host materialized float32 stacks,
  host/device transfer remains a real boundary even when device kernels are
  cheap.
- Next run: sim16 rows, because sim2 does not approximate a serious repeated
  search workload.

B512/B1024 synthetic policy/search pressure rows, sim16:

| row | shape | measured sec | scalar steps/s | physical rows/s | observation sec | probe sec | probe share | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hybrid real + probe sim16 | B512/A16, steps20/warmup5 | `4.5498` | `4501.25` | `2250.63` | `3.4377` | `0.4031` | `8.9%` | heavier probe visible, still not wall |
| hybrid real + probe sim16 | B1024/A16, steps20/warmup5 | `5.6924` | `7195.59` | `3597.79` | `4.2150` | `0.6352` | `11.2%` | observation still dominates |

Read:

- Even at sim16, the synthetic policy/search pressure probe is not the Amdahl
  wall in this profile-only harness. The B1024 row spends about `74%` of
  measured wall in observation/stack (`4.215s / 5.692s`) and about `11%` in the
  synthetic probe.
- The probe's device work is real and grows with the simulation knob, but the
  bigger bucket is still the observation boundary: renderer service, device
  render, device-to-host/readback, and CPU stack update.
- This still does **not** prove real LightZero MCTS is cheap. It only narrows
  the current profile-only next target: before wiring a trainer bridge, reduce
  the observation boundary or build an experiment that uses real policy/search
  semantics instead of the synthetic conv loop.

No-copy stack snapshot patch:

- Before this patch, the hybrid manager returned `self._zero_stack.copy()` on
  every non-terminal step. At B1024 that copies about `134MB` per profile step
  before scalar materialization.
- The manager now returns the internal stack view for non-terminal rows and
  only takes a snapshot when terminal rows need pre-autoreset semantics.
- Terminal semantics are preserved: if any row is done, the code snapshots the
  returned observation before reset and attaches terminal `final_observation`.
- Local validation:
  `uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_mock_collector.py tests/test_source_state_batched_observation_boundary_profile.py`
  -> `66 passed, 2 skipped`.
- Fresh H100 rows quantified the win:

| row | before scalar steps/s | after scalar steps/s | before obs sec | after obs sec | read |
| --- | ---: | ---: | ---: | ---: | --- |
| B512/A16 no probe | `5446.80` | `6822.14` | `3.0400` | `2.3426` | about `1.25x` faster |
| B1024/A16 no probe | `6661.70` | `9495.28` | `4.9929` | `3.4609` | about `1.43x` faster |
| B1024/A16 sim16 | `7195.59` | `7523.64` | `4.2150` | `3.7407` | smaller win because probe/actor costs also matter |

- Amdahl read after the patch: the avoidable full-stack copy was real, but
  observation remains the dominant bucket. At B1024 no-probe, observation is
  still about `80%` of measured wall (`3.4609s / 4.3137s`).
- The remaining observation wall is renderer service/device render/output
  transfer plus in-place stack shift/update, not the old per-step return copy.
- B2048 rows are running to test whether wider batches keep amortizing the
  renderer boundary.

B2048 scaling rows before fine-grain stack timing:

| row | measured sec | scalar steps/s | physical rows/s | observation sec | probe sec | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B2048/A32 no probe | `6.8889` | `11891.53` | `5945.76` | `5.3023` | `0.0` | wider still helps |
| B2048/A32 sim16 | `7.7752` | `10536.00` | `5268.00` | `5.0363` | `1.2362` | probe becomes visible but observation remains larger |

Read:

- B2048 continues to improve aggregate no-probe throughput over B1024
  (`11892` vs `9495` scalar steps/s), but observation is still about `77%` of
  measured wall.
- Sim16 at B2048 makes policy/search pressure more visible (`1.236s`, about
  `16%`), but the observation bucket remains roughly `65%`.
- Next patch adds `stack_shift_sec` and `stack_latest_update_sec` so the
  observation bucket can be split between render/transfer and CPU stack work.

Fine-grain stack timing rows:

| row | measured sec | scalar steps/s | observation sec | renderer render sec | device render sec | stack shift sec | latest update sec | scalar materialization sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B2048/A32 float32 stack | `8.2558` | `9922.70` | `6.4255` | `3.0757` | `1.5021` | `2.1567` | `0.2499` | `0.0989` |
| B4096/A64 float32 stack | `14.4515` | `11337.24` | `11.3381` | `5.8747` | `1.9913` | `3.5648` | `0.4771` | `0.1793` |

Read:

- B4096 does not clearly beat B2048; physical rows/s is slightly lower
  (`5669` vs the earlier B2048 `5946`). The useful current width is probably
  B2048-ish for this in-process profile.
- The observation wall is now split enough to act on. For B2048, stack shift
  alone is about `26%` of measured wall, renderer render about `37%`, and
  scalar materialization about `1%`.
- Next low-risk experiment: store the profile stack as uint8, normalize back to
  float32 only at the LightZero-shaped scalar boundary. This should reduce the
  stack-shift/latest-update memory traffic, but may move cost into scalar
  materialization. That is exactly what the next rows test.

Uint8 stack storage rows:

| row | measured sec | scalar steps/s | observation sec | renderer render sec | device render sec | stack shift sec | latest update sec | scalar materialization sec | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1024/A16 uint8 stack | `6.0887` | `6727.22` | `3.1398` | `2.4089` | `1.6626` | `0.1717` | `0.0325` | `1.9858` | stack work drops; CPU scalarization grows |
| B2048/A32 uint8 stack | `10.8253` | `7567.49` | `4.8686` | `3.2181` | `1.5618` | `0.6028` | `0.0778` | `3.9795` | not a current win |
| B2048/A32 uint8 stack + sim16 | `9.6990` | `8446.27` | `4.0483` | `2.8769` | `1.3640` | `0.4026` | `0.0473` | `2.8980` | policy probe visible; conversion still hurts |

Read:

- The uint8 stack implementation works mechanically: the manager stores byte
  observations, and the scalar LightZero-shaped rows still materialize as
  normalized float32.
- It proves the stack memory traffic theory. B2048 stack shift drops from
  about `2.16s` in the float32 timing row to about `0.60s` in the uint8 row.
- It is not a current throughput win because the cost moves to CPU
  scalarization. At B2048, scalar materialization grows from about `0.10s` to
  about `3.98s`.
- Amdahl read: uint8 storage only becomes useful if the next boundary consumes
  compact/uint8 stacks directly and normalizes on GPU, or otherwise avoids
  scalarizing every row to float32 on the host.
- Practical next step: stop optimizing stack dtype in isolation. Build a
  profile-only device-resident handoff probe or a real policy/search gate that
  consumes the batched stack before CPU scalarization.

Pre-scalarization batched-stack probe and dynamic-slot correction:

The first B2048 comparison accidentally used fixed `trail_slots=1024`. That
made the renderer draw the entire trail buffer even though the row only had
about 20 active profile steps. Those rows are useful as a warning, not as the
current-speed baseline:

| row | dynamic slots | stack dtype | scalar materialization | measured sec | scalar steps/s | observation sec | renderer render sec | scalar materialization sec | batched-stack probe sec |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2048 sim16 | false | float32 | on | `186.79` | `438.56` | `182.48` | `179.34` | `0.074` | `1.326` |
| B2048 sim16 | false | float32 | off | `188.28` | `435.09` | `183.24` | `179.49` | `0.000` | `1.384` |
| B2048 sim16 | false | uint8 | on | `189.67` | `431.91` | `181.60` | `179.78` | `3.721` | `0.931` |
| B2048 sim16 | false | uint8 | off | `186.01` | `440.41` | `181.15` | `179.34` | `0.000` | `0.959` |

Read:

- Fixed full-slot rendering is about `10x` slower than dynamic-slot rendering
  for this short-trajectory profile. It should not be the optimizer default.
- The current Coach training launcher already sets `dynamic_render_trail_slots`
  to true; the bad default was in this profile wrapper. The profile wrapper now
  defaults to dynamic slots too.

Corrected dynamic-slot rows:

| row | stack dtype | probe | scalar materialization | measured sec | scalar steps/s | observation sec | renderer render sec | stack shift sec | scalar materialization sec | probe sec |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B512 no probe | float32 | none | on | `5.151` | `3975.90` | `4.141` | `3.499` | `0.304` | `0.015` | `0.000` |
| B2048 no probe | float32 | none | on | `17.417` | `4703.49` | `14.360` | `11.436` | `1.744` | `0.099` | `0.000` |
| B2048 no probe | uint8 | none | on | `18.735` | `4372.58` | `13.131` | `11.707` | `0.328` | `2.779` | `0.000` |
| B2048 sim16 | float32 | pre-scalar | on | `19.432` | `4215.77` | `15.269` | `12.472` | `1.679` | `0.108` | `1.243` |
| B2048 sim16 | float32 | pre-scalar | off | `19.087` | `4291.98` | `15.072` | `12.418` | `1.601` | `0.000` | `1.231` |
| B2048 sim16 | uint8 | pre-scalar | off | `18.453` | `4439.44` | `13.599` | `11.871` | `0.395` | `0.000` | `0.956` |

Read:

- The pre-scalarization probe works and consumes `[B,2,4,64,64]` plus
  `[B,2,3]` action masks before scalar LightZero row materialization.
- Skipping scalar materialization is not a large win in float32 at this shape:
  about `0.35s` on a `19s` row. That falsifies the idea that the current big
  wall is just CPU scalar row creation.
- Uint8 becomes plausible only when scalarization is skipped: it avoids the
  `~2.8s` CPU float32 conversion seen in the uint8/no-probe row. But even then
  the corrected dynamic row is still render/stack-service dominated.
- Current Amdahl target for long no-death profiles is dynamic renderer cost and
  stack/service overhead. For short profiles, B2048 is only about `18%` better
  than B512 in scalar steps/sec, so widening alone is not the solution.
- Next experiment family should vary active trajectory length, because dynamic
  slots keep short rows cheap but will approach the full-slot cost as trails
  grow.

Dynamic-slot trajectory-length ladder, B512/A16, no probe:

| measured steps | warmup steps | scalar steps/s | physical rows/s | measured sec | observation sec | renderer render sec | device render sec | read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | 5 | `3975.90` | `1987.95` | `5.151` | `4.141` | `3.499` | `2.827` | short-policy regime |
| 100 | 20 | `1779.81` | `889.90` | `57.534` | `51.815` | `48.085` | `44.420` | dynamic slots already much heavier |
| 200 | 20 | `1149.00` | `574.50` | `178.242` | `168.225` | `161.786` | `154.874` | render dominates wall |
| 500 | 20 | `598.01` | `299.00` | `856.176` | `821.539` | `802.428` | `780.973` | long-trail render wall |

Read:

- This is the cleanest Amdahl signal from the current pass. If policies survive
  longer, render cost rises very quickly even with dynamic slots.
- Dynamic slots prevent the short-row fixed-capacity disaster, but they do not
  remove the long-trajectory wall. Around 200 no-death steps at B512, about
  `94%` of measured wall is observation and about `91%` is renderer render.
- At 500 no-death steps, observation is about `96%` of measured wall and
  renderer render alone is about `94%`. Actor step wall is only about `32.7s`,
  stack shift is about `9.0s`, and scalar materialization is about `0.67s`.
- The next optimization phase should treat long-trajectory renderer cost as a
  first-class wall again. Pre-scalarization handoff and uint8 help specific
  buckets, but they do not address the dominant long-row renderer work.

Existing dirty/incremental renderer discovery:

- The repo already has `SourceStateCanvasGray64DirtyRenderCache` in
  `src/curvyzero/env/vector_visual_observation.py` and a focused prototype at
  `scripts/prototype_incremental_trail_layer_bench.py`.
- Local prototype command:
  `python3 scripts/prototype_incremental_trail_layer_bench.py --trail-lengths 100,200,500,1000 --append-step 4 --iterations 3 --warmup-iterations 1 --format json`
- Parity: all built-in parity cases passed; benchmark rows had no mismatches.
- Timing:

| trail length | full us/frame | cached us/frame | speedup |
| ---: | ---: | ---: | ---: |
| 100 | `4062.8` | `12986.7` | `0.31x` |
| 200 | `5784.0` | `13414.9` | `0.43x` |
| 500 | `11425.4` | `14615.7` | `0.78x` |
| 1000 | `20623.8` | `16463.2` | `1.25x` |

Read: the incremental idea is semantically promising and exact in these local
cases, but the standalone Python prototype is not enough by itself. It only
beats full redraw at very long trails and only modestly. The useful lane is
either the already-integrated dirty-block cache in the current environment, or
a batched/device-side version of the same idea; a naive Python incremental
renderer is not a 10x answer.

Current local CPU dirty-cache trajectory profile:

- Command:
  `python3 scripts/profile_curvytron_render_trajectory_lengths.py --lengths 100 500 1000 --render-modes browser_lines --repeats 1 --warmup-steps 20 --seed 20260521 --opponent-runtime-mode blank_canvas_noop --policy wall_avoidant --no-natural-bonus-spawn --output docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/local_dirty_cache_trajectory_profile_20260521.json --markdown`
- Tooling fix: the profiler now tolerates the removed
  `render_source_state_gray64_fast_player_perspectives` hook instead of failing
  before measurement.

| steps | wall sec | steps/sec | render sec | render % | observation sec | vector step sec | dirty-cache status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100 | `0.232` | `430.5` | `0.179` | `76.9%` | `0.182` | `0.032` | cold start only, `100` hits |
| 500 | `1.154` | `433.1` | `0.887` | `76.8%` | `0.903` | `0.158` | cold start only, `500` hits |
| 1000 | `2.350` | `425.5` | `1.800` | `76.6%` | `1.835` | `0.324` | cold start only, `1000` hits |

Read:

- This is local fixed-opponent env-only CPU dirty-cache profiling, not the
  Modal batched GPU manager profile. Do not compare the absolute steps/sec
  directly with the B512/A16 H100 ladder.
- The shape is still useful: dirty cache keeps throughput roughly flat as
  trajectory length grows. That is exactly the cost model we want.
- Render is still about `77%` of local wall, so even the dirty path has
  headroom. But it avoids the catastrophic long-trail slowdown seen in the
  GPU full-redraw ladder.
- Focused tests passed after the profiler fix:
  `uv run pytest -q -p no:cacheprovider tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_scalar_dirty_render_cache_matches_full_renderer_after_steps tests/test_curvytron_two_seat_render_mode.py::test_two_seat_dirty_render_cache_matches_full_render_with_sprites tests/test_source_state_hybrid_observation_profile.py`
  returned `15 passed`.

Synthetic persistent policy-space framebuffer benchmark:

- Added isolated Modal benchmark:
  `src/curvyzero/infra/modal/source_state_persistent_framebuffer_benchmark.py`.
- This is not source-state physics, browser parity, LightZero training, or a
  tournament/checkpoint observation claim. It tests one cost-model question:
  persistent append-only policy-space trails versus stateless redraw of all
  previous synthetic segments.
- H100 smoke command:
  `uv run --extra modal modal run -m curvyzero.infra.modal.source_state_persistent_framebuffer_benchmark --compute gpu-h100 --batch-size 128 --steps 64 --warmup-steps 8 --target-size 64 --parity-interval 16`

| compute | B | steps | exact parity | stateless total sec | persistent total sec | total speedup | device speedup |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| H100 | 128 | 64 | yes | `0.001481` | `0.000404` | `3.67x` | `7.86x` |
| H100 | 512 | 512 | yes | `0.010102` | `0.001192` | `8.48x` | `40.26x` |
| H100 | 2048 | 256 | yes | `0.014077` | `0.002782` | `5.06x` | `48.13x` |
| L4 | 512 | 512 | yes | `0.014456` | `0.001331` | `10.86x` | `53.23x` |
| H100 no readback | 512 | 512 | yes | `0.008685` | `0.000225` | `38.57x` | `38.80x` |
| H100 target128 | 512 | 256 | yes | `0.013335` | `0.002900` | `4.60x` | `40.21x` |

Read: the cost-model hypothesis is alive. Even with full frame readback
enabled, persistent policy-space updates are several times faster than
stateless synthetic redraw. The exact multiplier depends on batch/readback
shape, but the broader result is clear: if we can implement this cost model
against real source-state render events, it is the first plausible 10x-class
renderer lane from this pass.

The no-readback row shows the device-side win is about `39x`; the readback row
falls to `8.5x`, so host/frame ownership still matters. The `128x128` row is
still `4.6x` faster end-to-end, but readback is much heavier; larger policy
frames are feasible from render math, but they need model/search/readback
profiling before becoming a training recommendation.

Independent local toybench:

- Subagent artifact:
  `docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_incremental_render_toybench_20260521.md`.
- Scope: local CPU/NumPy direct-64x64 append-only toy, with precomputed line
  pixels. It is not production timing, but it isolates old-trail replay volume.
- Final-frame parity was exact in every cell.

| batch | steps | full sec | incremental sec | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 20 | `0.000103` | `0.0000102` | `10.1x` |
| 64 | 100 | `0.001104` | `0.0000478` | `23.1x` |
| 64 | 200 | `0.003424` | `0.0000974` | `35.1x` |
| 64 | 500 | `0.019670` | `0.0002448` | `80.4x` |
| 512 | 20 | `0.000863` | `0.0000429` | `20.1x` |
| 512 | 100 | `0.010087` | `0.0002129` | `47.4x` |
| 512 | 200 | `0.039180` | `0.0004555` | `86.0x` |
| 512 | 500 | `0.229214` | `0.0010887` | `210.5x` |

Read: two independent synthetic lanes now agree. Replay-all-old-trails is the
wrong cost model for long survival; persistent append/update is the right next
profile-only implementation target.

Validation after this phase:

- `python3 -m py_compile src/curvyzero/infra/modal/source_state_persistent_framebuffer_benchmark.py src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py src/curvyzero/training/source_state_hybrid_observation_profile.py scripts/profile_curvytron_render_trajectory_lengths.py`
  passed.
- `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_hybrid_observation_profile.py tests/test_curvyzero_source_state_visual_survival_lightzero_env.py::test_source_state_scalar_dirty_render_cache_matches_full_renderer_after_steps tests/test_curvytron_two_seat_render_mode.py::test_two_seat_dirty_render_cache_matches_full_render_with_sprites`
  returned `57 passed`.

## 2026-05-21 Step-Back: Are We Optimizing The Real Observation?

User critique: speed rows are not enough. The real question is whether the
candidate fast observation stays close enough to the current policy observation
over long trajectories. Existing tests covered exact fixture parity and some
short dynamic GPU parity, but there was no long rollout gate for the persistent
GPU `direct_gray64` profile path.

Code added:

- `source_state_batched_observation_boundary_profile.py` now has
  `--surface-facade-divergence-canary`.
- It is profile-only and requires `--surface-facade-canary`.
- It drives the same `SourceStateMultiplayerTrainerSurface` env, compares the
  candidate surface stack to `CpuOracleBatchedObservationRenderer`, and reports
  raw/latest-frame and full `[B,2,4,64,64]` stack divergence.
- It reports total compared values, overall mismatch fraction, max per-check
  mismatch fraction, and a bounding box for the first mismatch plane.
- It also exercises terminal `final_observation` and autoreset comparison when
  terminal rows occur.
- It is off by default so timing rows do not pay CPU oracle overhead unless the
  divergence gate is explicitly requested.

Local validation:

- `python3 -m py_compile src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_boundary_profile.py`
- `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py`
- Result: `50 passed`.

Modal control smoke, CPU candidate versus CPU oracle:

```text
compute=gpu-l4-t4
surface_stack_backend=cpu_dirty_cache
surface_facade_divergence_canary=true
B=2, steps=4, warmup=1, verify_steps=4
geometry_dtype=float64, parity=exact
```

Result:

- `ok=true`
- `checked_step_count=4`
- `all_exact=true`
- `total_mismatch_count=0`
- candidate median surface step `0.0199s`
- CPU reference median render+stack `0.0409s`

This proves the new divergence canary plumbing itself is aligned.

First persistent GPU policy-space divergence smoke:

```text
compute=gpu-l4-t4
renderer=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
B=8, steps=32, warmup=4, verify_steps=32
trail_slots=512, body_capacity=512
parity=tolerant, max_abs_diff=96, mismatch_fraction=0.25
```

Result:

- `ok=true`
- candidate backend `jax_gpu_persistent_policy_framebuffer_profile`
- reference backend `cpu_oracle`
- `all_exact=false`
- `max_abs_diff_observed=61`
- `total_mismatch_count=53414`
- approximate total compared values: `10813440`
- approximate overall mismatch fraction: `0.0049`
- candidate median surface step `0.00955s`
- candidate median render `0.00578s`
- CPU reference median render+stack `0.183s`

Read:

- This is not exact parity. That is expected: persistent `direct_gray64` is a
  policy-space approximation, while the CPU oracle renders full 704x704
  browser-lines/simple-symbols and downsamples.
- The smoke did not reveal catastrophic divergence. Roughly half a percent of
  compared values differed under this short rollout, with max uint8 difference
  `61`.
- This is still not enough for promotion. The next gate is longer no-death
  rollout plus a timeout/autoreset row, both with the same divergence canary.

Follow-up persistent GPU divergence rows:

| row | compute | B | steps | max ticks | max abs diff | mismatch fraction | truncation rows | median candidate step | median CPU reference render+stack | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| timeout/autoreset | L4/T4 | 4 | 64 | 6 | `64` | `0.00174` | `0` | `0.00794s` | `0.0593s` | exercised terminal rows; p95 final rows `4` |
| long no-death | L4/T4 | 8 | 256 | 2000 | `67` | `0.02713` | `0` | `0.00956s` | `0.4905s` | active trail median `291`, p95 `479` |

Read:

- The persistent GPU policy-space approximation still diverges from the CPU
  oracle, but it does not show uncontrolled drift over 256 no-death steps.
- The mismatch fraction grows with trail length, which is expected because
  `direct_gray64` is not the same as 704-to-64 area downsampling.
- The timeout/autoreset row is especially important: final rows appeared and
  the persistent renderer still stayed within the semantic gate.
- This is now a meaningful approximation lane. It is still not a production
  recommendation until we add at least one visual/sample artifact or component
  diff summary, plus a matched full-loop A/B row.

Validation after the divergence gate:

- `python3 -m py_compile src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_boundary_profile.py`
- `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_gpu_render_benchmark_cpu.py`
- Result: `66 passed, 6 skipped`.

## 2026-05-21 Resident Chunk Probe Smoke And Medium Rows

Purpose:

```text
Test whether the profile-only resident chunk can keep CurvyTron observation
batches on GPU through replay/search-shaped synthetic work before any scalar
LightZero materialization.
```

Tiny H100 smoke:

- `ok=true`
- `calls_train_muzero=false`
- `touches_live_runs=false`
- JAX backend `gpu`, device `cuda:0`
- B64/A4/sim2, 8 measured steps, 2 warmup steps
- resident replay capacity `1024` roots
- resident model eval count `256`
- scalar roots/sec `5091.82`
- physical rows/sec `2545.91`
- observation `0.0937s`
- resident probe total `0.0245s`

Medium B512/A16/sim8 rows, 60 measured steps, 20 warmup steps, scalar edge off:

| compute | scalar edge | scalar roots/sec | physical rows/sec | measured sec | observation sec | renderer sec | resident probe sec | scalarization sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | off | `10980.47` | `5490.24` | `5.60` | `2.33` | `1.24` | `0.58` | `0.00` |
| H100 | on | `7620.76` | `3810.38` | `8.06` | `2.45` | `1.42` | `0.67` | `1.96` |
| L4/T4 | off | `5839.67` | `2919.83` | `10.52` | `3.44` | `1.73` | `2.87` | `0.00` |
| L4/T4 | on | `4133.28` | `2066.64` | `14.86` | `3.70` | `1.80` | `2.86` | `3.68` |

Plain read:

- The resident chunk implementation works as a profile-only canary.
- The harsher medium row is lower than the earlier stack-only H100 `~13.8k`
  root/sec anchor, which is expected because it adds replay/search-shaped
  synthetic pressure.
- H100 has a real advantage once the row includes enough GPU-shaped work:
  about `1.88x` over L4/T4 on this medium resident row with scalar edge off,
  and about `1.84x` with scalar edge on.
- Scalar materialization is a serious but not total tax in this profile-only
  shape: H100 keeps about `69%` of scalar-off throughput; L4/T4 keeps about
  `71%`.
- The scalar edge is still the wrong place to put the hot loop. It creates
  `61,440` materialized timesteps in this row and converts the resident batch
  into LightZero-shaped host objects.
- This still does not prove full training speed. It does not call stock
  LightZero, real MCTS, replay target construction, learner updates, RND,
  checkpointing, eval, GIF, or tournament code.
- The next useful row is a real-consumer canary: actual LightZero
  policy/search-shaped work, or the closest profile-only hook we can build
  without touching the trusted Coach lane.

B1024 H100 scale check, 40 measured steps, 20 warmup steps:

| compute | B | scalar edge | scalar roots/sec | physical rows/sec | measured sec | observation sec | resident probe sec | scalarization sec |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | 1024 | off | `11065.92` | `5532.96` | `7.40` | `3.00` | `0.72` | `0.00` |
| H100 | 1024 | on | `6753.37` | `3376.69` | `12.13` | `3.76` | `0.72` | `3.87` |

Read:

- B1024 does not materially improve scalar-off throughput versus B512
  (`~11.07k` versus `~10.98k` roots/sec).
- B1024 scalar-on is worse than B512 scalar-on (`~6.75k` versus `~7.62k`).
- This points to host/stack/materialization and scheduling pressure; simply
  widening the current profile-only resident batch is not the next obvious win.
- Use B512 as the default resident-probe shape until a real consumer changes
  the scaling curve.

Validation after resident probe and instrumentation updates:

- `uv run pytest -q -p no:cacheprovider tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_source_state_batched_observation_boundary_profile.py`
- Result: `52 passed`
- `uv run ruff check scripts/build_curvytron_hybrid_observation_profile_grid.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py`
- Result: passed
- `uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_mock_collector.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_profile_cpu.py`
- Result: `82 passed, 2 skipped`

## 2026-05-21 Corrected LightZero Split Rows

Status: profile-only. These rows do not call `train_muzero`, do not touch live
training, and do not write checkpoint/eval/GIF/tournament artifacts.

Shared shape:

```text
B512 physical rows
2 player roots per row = 1024 roots per measured call
20 measured steps, 5 warmup steps
uint8 [B,2,4,64,64] stack
jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
death_mode=profile_no_death
hybrid_observation_canary=true
materialized scalar timesteps off
```

Run registry:

| row | run id | compute | consumer | simulations | status |
| --- | --- | --- | --- | ---: | --- |
| H100 initial inference | `ap-Kptsuxj88SWsBcNEIMWKIM` | H100 | `policy._model.initial_inference` | `0` | passed |
| L4 initial inference | `ap-aoxesrinITXEmG20DCXv5r` | L4/T4 | `policy._model.initial_inference` | `0` | passed |
| H100 collect-forward | `ap-P2vQfJQdPOPhCzazdfc0L9` | H100 | `MuZeroPolicy.collect_mode.forward` | `8` | passed |
| L4 collect-forward | `ap-TJsCxsf76q3j4nbcgqv9hn` | L4/T4 | `MuZeroPolicy.collect_mode.forward` | `8` | passed |
| H100 collect-forward | `ap-sFirUwyIvk47PHh6r3Y5K1` | H100 | `MuZeroPolicy.collect_mode.forward` | `1` | passed |
| L4 collect-forward | `ap-JFXHc2QWx84AdR1B7PZND2` | L4/T4 | `MuZeroPolicy.collect_mode.forward` | `1` | passed |

Main table:

| compute | consumer | simulations | roots/sec | physical rows/sec | measured sec | probe sec | model/search sec | H2D sec | render sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | initial inference only | `0` | `9238.85` | `4619.42` | `2.22` | `0.94` | `0.09` | `0.80` | `0.22` |
| L4/T4 | initial inference only | `0` | `6790.63` | `3395.32` | `3.02` | `1.28` | `0.55` | `0.63` | `0.28` |
| H100 | collect-forward | `8` | `2693.10` | `1346.55` | `7.60` | `6.53` | `5.82` | `0.38` | `0.19` |
| L4/T4 | collect-forward | `8` | `1381.35` | `690.67` | `14.83` | `13.07` | `11.82` | `0.57` | `0.29` |
| H100 | collect-forward | `1` | `3296.02` | `1648.01` | `6.21` | `5.17` | `4.19` | `0.37` | `0.18` |
| L4/T4 | collect-forward | `1` | `1687.81` | `843.91` | `12.13` | `10.36` | `8.82` | `0.53` | `0.28` |

Plain read:

```text
The neural network root inference is not the main wall in this profile.
The public LightZero collect/search path is the wall.
```

Why:

- H100 initial inference reaches about `9239` roots/sec, while H100
  collect-forward sim8 reaches about `2693` roots/sec.
- L4 initial inference reaches about `6791` roots/sec, while L4 collect-forward
  sim8 reaches about `1381` roots/sec.
- Dropping collect-forward from sim8 to sim1 helps only modestly: H100
  `2693 -> 3296` roots/sec and L4 `1381 -> 1688` roots/sec.
- Render is small in the corrected collect rows: H100 sim8 render is about
  `0.19s` inside a `7.60s` measured window; L4 sim8 render is about `0.29s`
  inside `14.83s`.

Amdahl read:

```text
Making render free would barely move these rows. The highest-leverage next
split is inside or around LightZero collect/search: CPU tree work, Python
output fanout, public collect wrapper overhead, and search topology.
```

Important caveat:

```text
These are still profile-only boundary rows. They prove the batch can reach real
LightZero model/search APIs, not that Coach training should switch defaults.
The next validation layer is a boundary test matrix plus stock-loop gates for
normal death/autoreset and RND.
```

## 2026-05-21 Deeper LightZero Search Split Rows

Status: passed/settled.

Purpose: split the `collect_mode.forward` non-model residual. The previous H100
B512/A16/sim8 internal timer row spent about `69.8s` inside collect-forward but
only about `2.7s` in model initial/recurrent calls. These rows add profile-only
timers for:

- `policy._mcts_collect.search`;
- ctree `batch_traverse`, if the binding can be monkey-patched;
- ctree `batch_backpropagate`, if the binding can be monkey-patched;
- outside-MCTS residual;
- MCTS non-model residual.

Run registry:

| row | run id | compute | consumer | simulations | status |
| --- | --- | --- | --- | ---: | --- |
| H100 deeper search split | `ap-48okBeP8PPdnGdQp2glocA` | H100 | `MuZeroPolicy.collect_mode.forward` | `8` | passed; last-call only |
| cheap-pool deeper search split | `ap-QfBSf0jtlr4rEElba1kVd5` | L4/T4 pool | `MuZeroPolicy.collect_mode.forward` | `8` | passed; landed on L4 |
| H100 deeper search split with aggregate fields | `ap-G0t2cwcHd10P2nLfMmyNVI` | H100 | `MuZeroPolicy.collect_mode.forward` | `8` | passed |

Validation before launch:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py \
  tests/test_curvytron_optimizer_profile_manifest_runner.py
-> 87 passed

uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_hybrid_observation_profile.py
-> passed
```

Decision after rows return:

```text
If MCTS search ~= collect-forward residual, search/tree is the wall.
If ctree calls are small but MCTS search is large, conversion/list/root glue
around ctree is the wall.
If outside-MCTS dominates, root setup/action-mask/output assembly is the wall.
```

Preliminary first-row read, based on last-call telemetry only:

| compute | roots/sec | collect-forward last call | model calls | MCTS search | ctree traverse + backprop | outside MCTS | decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | `2360.52` | `0.304s` | `0.0155s` | `0.105s` | `0.0099s` | `0.200s` | `0.0145s` |
| L4 | `1291.41` | `0.599s` | `0.0501s` | `0.173s` | `0.0162s` | `0.426s` | `0.0258s` |

Plain preliminary read:

```text
MCTS search is important, but it is not the whole residual.
ctree traverse/backpropagate is tiny.
Large time remains outside MCTS search, likely root setup, data conversion,
legal-action preparation, action/output assembly, or other policy glue.
```

The first two rows were launched before the aggregate timing-field patch, so
they expose these values in `batched_stack_probe_last_telemetry` only. The
follow-up H100 row should produce aggregate timing totals.

Aggregate H100 rerun:

| bucket | aggregate sec | fraction of collect-forward |
| --- | ---: | ---: |
| collect-forward total | `35.36` | `100%` |
| model calls total | `1.81` | `5.1%` |
| MCTS search total | `10.97` | `31.0%` |
| MCTS search non-model | `9.70` | `27.4%` |
| ctree traverse + backpropagate | `0.98` | `2.8%` |
| outside MCTS | `24.40` | `69.0%` |
| decode after forward | `2.30` | outside collect-forward |

Aggregate-row plain read:

```text
The big wall is not neural-net inference.
The big wall is also not the raw ctree traverse/backprop calls.
MCTS search is meaningful, but the largest single bucket is outside the MCTS
search call: root setup, action-mask/to_play plumbing, data conversion,
action selection, visit/output assembly, and public collect wrapper overhead.
```

Next experiment from this result:

```text
Inspect LightZero _forward_collect and MCTS search implementation enough to
split root setup/output conversion from search. Then either patch a narrower
timing hook or build a tiny replacement-ceiling toy that avoids per-root
Python dict/list output fanout.
```

## 2026-05-21 Pure-Policy Collect Wrapper Split

Status: passed/settled.

Purpose: keep the same public LightZero `collect_mode.forward` wrapper, but set
`collect_with_pure_policy=true` so it skips MCTS search. This isolates the
wrapper-side cost: initial inference, CPU NumPy/list conversion, legal-action
handling, per-root policy softmax/action sampling, and output dict assembly.

Run registry:

| row | run id | compute | consumer | status |
| --- | --- | --- | --- | --- |
| H100 pure-policy collect wrapper | `ap-vxtn2gLBpiywWo8cwHZ5GR` | H100 | `MuZeroPolicy.collect_mode.forward`, pure-policy branch | failed before remote function; Modal app stopped during long image-build heartbeat |
| H100 pure-policy collect wrapper retry | `ap-wUFLcUYLGhyyqdyVptqgvW` | H100 | `MuZeroPolicy.collect_mode.forward`, pure-policy branch | passed |

Plain expectation:

```text
If pure-policy collect is still slow, the wrapper/output fanout is the next
target.
If pure-policy collect is fast, the expensive work is specific to MCTS root
setup/search/result conversion.
```

Result:

| row | roots/sec | measured sec | collect-forward total | model total | MCTS search | output decode | render |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 MCTS collect, sim8 | `2572.12` | `47.77` | `35.36s` | `1.81s` | `10.97s` | `2.30s` | `1.77s` |
| H100 pure-policy collect | `6286.61` | `19.55` | `4.88s` | `0.54s` | `0.00s` | `1.35s` | `2.33s` |

Plain read:

```text
Pure-policy collect is about 2.4x faster than MCTS collect on this profile.
So the expensive work is specific to the MCTS branch.
But raw ctree traverse/backprop was only about 0.98s in the MCTS row.
That points at MCTS-branch setup/conversion/result handling, not just C++
tree math.
```

Small cleanup after the row: the pure-policy telemetry now labels
`model_eval_count` as one model evaluation per root and sets
`lightzero_consumer_cpu_tree_included=0` for pure-policy rows. Validation after
that cleanup: focused tests `88 passed`, focused ruff passed.

## 2026-05-21 LightZero Array Ceiling Rows

Status: passed/settled as a profile-only ceiling.

Purpose: profile the removable overhead ceiling for the public MCTS branch
without changing the trainer. These rows start from the same pre-scalar
`uint8/direct_gray64/persistent` `[B,2,4,64,64]` stack and use a scratch MuZero
model, but they do **not** call `collect_mode.forward` or
`_mcts_collect.search`.

Modes:

- `policy_arrays`: real initial inference, masked compact policy/value/action
  arrays out.
- `recurrent_toy`: real initial inference plus batched recurrent calls and a
  synthetic compact visit/value update. This is recurrent pressure, not MCTS.

Run registry:

| row | run id | compute | mode | status |
| --- | --- | --- | --- | --- |
| H100 policy arrays | `ap-IubMKmFoUag2Fq2alyd2ZU` | H100 | `policy_arrays` | passed |
| L4/T4 policy arrays | `ap-sIbE42nQBd7vTSi1zWvXnf` | L4/T4 pool; landed on L4 | `policy_arrays` | passed |
| H100 recurrent toy | `ap-0vgrMYtrZCzD21rXCmDbDX` | H100 | `recurrent_toy` | passed |
| L4/T4 recurrent toy | `ap-WdE4qyUHLyPoaU1KyYPtO9` | L4/T4 pool; landed on L4 | `recurrent_toy` | passed |

Validation before launch:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> 91 passed

uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/build_curvytron_hybrid_observation_profile_grid.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> passed
```

Decision rule:

```text
If the toy is much faster than public MCTS collect, design a real arrays-in /
arrays-out search-boundary replacement. If it is not, stop chasing this lane.
Do not call either toy real MCTS or training speed.
```

Result table:

| compute | mode | roots/sec | measured sec | array-ceiling total | H2D | initial inference | recurrent inference | synthetic update | output/readback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | `policy_arrays` | `9957.97` | `12.34` | `3.26s` | `2.41s` | `0.55s` | `0.00s` | `0.00s` | `0.05s` |
| H100 | `recurrent_toy` | `8681.01` | `14.16` | `4.66s` | `2.40s` | `0.55s` | `1.17s` | `0.21s` | `0.05s` |
| L4 | `policy_arrays` | `5589.96` | `21.98` | `7.85s` | `3.76s` | `3.31s` | `0.00s` | `0.00s` | `0.08s` |
| L4 | `recurrent_toy` | `5030.25` | `24.43` | `10.16s` | `3.50s` | `3.30s` | `2.25s` | `0.37s` | `0.09s` |

Comparison to the public H100 collect rows:

| row | roots/sec | ratio vs public MCTS collect |
| --- | ---: | ---: |
| public MCTS collect sim8 | `2572.12` | `1.00x` |
| public pure-policy collect | `6286.61` | `2.44x` |
| array ceiling `recurrent_toy` sim8 | `8681.01` | `3.37x` |
| array ceiling `policy_arrays` | `9957.97` | `3.87x` |

Plain read:

```text
The ceiling toy is much faster than public MCTS collect.
The recurrent toy still does 8 real batched recurrent model calls and is still
about 3.4x faster than public MCTS collect.
So the MCTS branch boundary is worth designing around.
```

Caveats:

```text
This is not real MCTS and not trainer speed.
It skips tree policy, backup semantics, temperature/noise semantics, and
LightZero's public per-root dict/list output fanout.
It is useful because it prices the removable boundary shape around real model
calls.
```

Next experiment from this result:

```text
Design a real arrays-in / arrays-out MCTS boundary, but do not promote it to
training until it passes fixed-seed parity against LightZero on action masks,
to_play, root noise, temperature, support/value transforms, visit counts, and
decoded actions. In parallel, test whether the H2D copy in the toy can be
removed by keeping the stack/model input resident.
```

## 2026-05-21 Late P2 Direct CTree Boundary Refresh

Question:

```text
Does the direct CTree arrays signal survive current code/current image on the
same H100 B512/A16/sim8 shape?
```

Shape:

- profile-only hybrid observation canary;
- H100;
- `batch_size=512`, `actor_count=16`;
- `60` measured steps, `15` warmup steps;
- `sim8`;
- `uint8 [512,2,4,64,64]` pre-scalar stack;
- scalar LightZero timestep materialization off;
- persistent profile GPU renderer and direct gray64 surface;
- no live training touched, `calls_train_muzero=false`.

Rows:

| row | run id | roots/sec | measured sec | boundary total | search | model total | root prep | model-output D2H | H2D/input | observation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock facade, host uint8 | `ap-ZxundYpxBqsTLzG3fKY525` | `2670.68` | `23.01s` | `18.61s` | public wrapper | `0.82s` | n/a | n/a | `0.91s` | `2.09s` |
| direct CTree, host uint8 | `ap-BmYXN0UtX73yZQhSUmoiLz` | `4764.06` | `12.90s` | `8.20s` | `6.06s` | `0.85s` | `0.48s` | `0.14s` | `1.05s` | `2.08s` |
| direct CTree, pinned uint8 | `ap-A2Ng4ZEURnSnYZdU2impSJ` | `3689.15` | `16.65s` | `11.36s` | `10.15s` | `0.99s` | `0.50s` | `0.10s` | `0.03s` | `2.30s` |
| direct CTree, resident stale ceiling | `ap-9KQ7NdYM9cVodS0FY9hMO6` | `3069.08` | `20.02s` | `14.41s` | `12.35s` | `1.38s` | `0.72s` | `0.87s` | `0.00s` | `2.58s` |

Plain read:

```text
The direct boundary signal survived. Fresh direct host uint8 is about 1.78x
faster than the stock facade on the same shape.
```

Input-copy tricks are not the current prize:

- pinned input cut H2D dramatically, but total wall got worse because search
  and other direct-boundary buckets grew;
- resident stale reuse removed H2D, but was slower than fresh host input in
  this row;
- this agrees with the previous refresh: direct CTree over public facade is the
  robust signal, not pinned/resident input.

Caveat:

```text
These rows were launched in parallel for speed. Treat exact ordering among
host/pinned/resident as noisy. The decision-level signal is that direct CTree
is still clearly faster than the stock public facade, and input transfer is not
the main remaining wall.
```

Next:

```text
Do not chase pinned/resident input right now.
Finish direct CTree P0/P1 validation: forced-case parity, support/noise/eval
checks, and a statistical stock-vs-direct comparison over many roots/seeds.
Only after that run a matched full-loop profile.
```

## 2026-05-21 Direct CTree Statistical Compare Tool

Added:

```text
scripts/compare_curvytron_direct_ctree_stock.py
```

Purpose:

```text
Run small CPU LightZero batches through stock facade and direct CTree arrays,
then report action agreement, visit L1 distance, searched-value delta, and
illegal-action counts.
```

Smoke:

```text
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 2 \
  --batch-rows 2 \
  --num-simulations 2 \
  --root-noise-weight 0.0 \
  --min-action-agreement 0.0
```

Result:

| metric | value |
| --- | ---: |
| mean action agreement | `0.75` |
| min action agreement | `0.50` |
| mean visit L1 | `0.375` |
| max visit L1 | `1.0` |
| max searched-value abs diff | `0.0` |
| illegal actions | `0` |

Plain read:

```text
The tool works and catches the known reality: low-simulation/tie-ish MCTS can
choose different actions even when searched values match and illegal-action
counts stay zero. Use this as a distributional validator, not as an exact
neutral-tie parity gate.
```

Validation:

```text
uv run ruff check scripts/compare_curvytron_direct_ctree_stock.py
-> passed
```

Follow-up validation:

```text
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 4 \
  --batch-rows 4 \
  --num-simulations 4 \
  --root-noise-weight 0.25 \
  --epsilon 0.0
```

Result:

| metric | value |
| --- | ---: |
| mean action agreement | `1.0` |
| min action agreement | `1.0` |
| mean visit L1 | `0.0` |
| max visit L1 | `0.0` |
| max searched-value abs diff | `0.0` |
| illegal actions | `0` |

Focused direct-boundary tests:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "direct_ctree or mcts_arrays_boundary_real_policy_cpu"
-> 8 passed, 75 deselected
```

Plain read:

```text
The direct CTree validation story got stronger. We still have not promoted it
to training, but the latest local statistical/root-noise check did not expose a
semantic mismatch.
```

Stronger sim8/root-noise-on follow-up:

```text
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 8 \
  --batch-rows 4 \
  --num-simulations 8 \
  --root-noise-weight 0.25 \
  --epsilon 0.0
```

Result:

| metric | value |
| --- | ---: |
| mean action agreement | `1.0` |
| min action agreement | `1.0` |
| mean visit L1 | `0.0` |
| max visit L1 | `0.0` |
| max searched-value abs diff | `0.0` |
| illegal actions | `0` |

Plain read:

```text
This is not a full production proof, but it is a strong local signal that the
direct CTree compact arrays path can match stock facade semantics on ordinary
small CPU collect batches with root noise enabled.
```

Focused full boundary-suite validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py
-> 83 passed

uv run ruff check \
  scripts/compare_curvytron_direct_ctree_stock.py \
  tests/test_source_state_batched_observation_boundary_profile.py
-> passed

uv run python -m py_compile scripts/compare_curvytron_direct_ctree_stock.py
-> passed
```

## 2026-05-21 - Beyond 1.8x Search Boundary Reorientation

Prompt:

```text
Why is direct_ctree_arrays only about 1.8x faster, and how do we actually fix
the remaining bottleneck?
```

Local/source read:

- LightZero's CTree kernels are already C++.
- The installed wrapper is
  `.venv/lib/python3.11/site-packages/lzero/mcts/tree_search/mcts_ctree.py`.
- The compiled module is
  `.venv/lib/python3.11/site-packages/lzero/mcts/ctree/ctree_muzero/mz_tree...so`.
- Upstream source is available at
  `/private/tmp/LightZero/lzero/mcts/ctree/ctree_muzero/`.
- `mz_tree.pyx` exposes `Roots.prepare`, `batch_traverse`,
  `batch_backpropagate`, `get_distributions`, and `get_values` through
  list/vector shaped APIs.

Concrete finding:

```text
The 1.8x ceiling is not because CTree is off. CTree is on. The wall is the
Python/list/Cython boundary around every simulation.
```

Docs updated:

- `search_boundary_escape_plan_20260521.md`
- `world_model.md`
- `task_board.md`
- `orchestration.md`
- `README.md`

Parallel sidecars launched:

- C++/Cython boundary feasibility:
  `/private/tmp/curvy_ctree_sidecar_handoff.md`.
- Accelerator-native MCTS feasibility:
  `/private/tmp/curvy_accelerator_mcts_handoff.md`.
- Batched actor architecture:
  `/private/tmp/curvy_batched_actor_arch_handoff.md`.

## 2026-05-22 - GPU-Latent Direct CTree Boundary

Implemented a profile-only `direct_ctree_gpu_latent` boundary. It keeps
LightZero CTree selection/backprop on CPU, but keeps MuZero latent tensors on
GPU across the MCTS simulation loop.

Tiny Modal smoke:

| run | compute | B | sims | status |
| --- | --- | ---: | ---: | --- |
| `ap-9ukAtmJG8cWN8IKVGrRHjA` | L4/T4 | 4 | 2 | passed |

Fixed H100 denominator:

```text
B512, A16, sim8, 60 measured, 15 warmup,
uint8 stack, scalar materialization off, no death, profile-only
```

| impl | run | measured sec | scalar steps/sec | boundary total | search sec | model sec | obs sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock_facade` | `ap-11O8aTHqzyJs26WEzfWNT1` | `26.99` | `2276.71` | `21.57s` | public collect/search `18.56s` | `0.96s` | `2.45s` |
| `direct_ctree_arrays` | `ap-FtdW9kJbiD0FTbgRvTtKSh` | `13.45` | `4568.28` | `8.31s` | `6.04s` | `0.86s` | `2.21s` |
| `direct_ctree_gpu_latent` | `ap-S5xLYBaiqsB3tHNWO1eiwU` | `9.34` | `6580.32` | `4.23s` | `1.89s` | `0.78s` | `2.68s` |

Plain read:

```text
The GPU-latent bridge is a real tactical win in this profile-only boundary:
about 2.9x over stock facade and 1.4x over direct CTree.
```

Validation:

```text
uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  scripts/compare_curvytron_direct_ctree_stock.py \
  tests/test_source_state_batched_observation_boundary_profile.py
-> passed

uv run python -m py_compile \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  scripts/compare_curvytron_direct_ctree_stock.py \
  tests/test_source_state_batched_observation_boundary_profile.py
-> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "mcts_arrays_boundary_real_policy_cpu or direct_ctree"
-> 8 passed

uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 8 --batch-rows 4 --num-simulations 8 \
  --root-noise-weight 0.25 --epsilon 0.0 \
  --impls direct_ctree_arrays,direct_ctree_gpu_latent \
  --max-mean-visit-l1 0.0 --min-action-agreement 1.0
-> exact agreement: actions 1.0, visit L1 0, value diff 0, illegal actions 0
```

After the validation audit, the compare helper was hardened:

- `--use-cuda` and `--require-cuda` report/fail on actual CUDA use;
- predicted-value and policy-logit diffs are included;
- `--strict-exact` sets exact gates for action/visit/value/logit equality.

The known clean exact gate still passes:

```text
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 8 --batch-rows 4 --num-simulations 8 \
  --root-noise-weight 0.25 --epsilon 0.0 \
  --impls direct_ctree_arrays,direct_ctree_gpu_latent \
  --max-mean-visit-l1 0.0 \
  --max-max-visit-l1 0.0 \
  --min-action-agreement 1.0 \
  --max-max-value-abs-diff 0.0 \
  --max-max-predicted-value-abs-diff 0.0 \
  --max-max-policy-logit-abs-diff 0.0
-> exact agreement on CPU tensors for actions, visits, values, predicted values,
   policy logits, and illegal-action count.
```

A tiny sim2 strict row failed on visit/action exactness while logits and values
matched exactly. Treat that as the existing tie/small-search caveat: sim2 is a
bad exact distribution gate.

Follow-up rows launched:

- post-pool `direct_ctree_gpu_latent` sim8 repeat;
- H100 sim16 `stock_facade`, `direct_ctree_arrays`, and
  `direct_ctree_gpu_latent`.

Follow-up rows completed:

Same denominator, sim8 post-pool:

| impl | run | measured sec | scalar steps/sec | boundary total | search sec | model sec | obs sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` post-pool | `ap-ug5mMvfuCWzRDjmNty1N3P` | `9.43` | `6518.13` | `4.25s` | `2.09s` | `0.79s` | `2.36s` |

Same denominator, sim16:

| impl | run | measured sec | scalar steps/sec | boundary total | search sec | model sec | obs sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock_facade` | `ap-cRxL8lanazTY29fnKhCQx7` | `35.43` | `1734.05` | `30.40s` | public collect/search `27.14s` | `1.61s` | `2.27s` |
| `direct_ctree_arrays` | `ap-zDtVC6Lw2LcPocY2AtTzRc` | `19.92` | `3083.58` | `15.10s` | `13.30s` | `1.62s` | `2.15s` |
| `direct_ctree_gpu_latent` | `ap-3v1vn4EGdk8YfJGIIVdiWD` | `12.60` | `4874.42` | `7.05s` | `4.69s` | `1.59s` | `2.50s` |

Plain read:

```text
GPU-latent is the best current profile-only search boundary:
sim8: about 2.9x over stock, about 1.4x over direct arrays.
sim16: about 2.8x over stock, about 1.6x over direct arrays.

The preallocated latent pool did not improve sim8. The speedup comes from
keeping latents on GPU, not from that micro-cleanup.
```

Amdahl read after GPU-latent:

```text
At sim8, reported search is only about 2 seconds inside about 9.4 seconds.
Even making that bucket free would not produce a large full-row speedup.

The broader remaining wall is still the collect/search boundary plus
observation/stack/packaging overhead. For a bigger jump, the next lane must
remove more CPU/list/search boundary work, not polish rendering again.
```

## 2026-05-22 - Dense Torch MCTS v0

Implemented a profile-only `dense_torch_mcts` mode in
`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`.

Purpose:

```text
Price a GPU-resident tree/search shape for CurvyTron's fixed A=3 action space.
This is not stock LightZero CTree and is not trainer-ready.
```

Local validation:

```text
uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  scripts/build_curvytron_hybrid_observation_profile_grid.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> passed

uv run python -m py_compile \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  scripts/build_curvytron_hybrid_observation_profile_grid.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "array_ceiling_dense_torch_mcts or array_ceiling_policy_arrays or array_ceiling_host_float32"
-> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> passed
```

Tiny Modal smoke:

| row | compute | B | A | sim | measured sec | scalar steps/sec | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense smoke | H100 | 64 | 4 | 4 | `0.4268` | `1199.67` | runtime/shape smoke only |

Matched H100 denominator:

```text
B512, A16, sim8, 60 measured, 15 warmup,
uint8 stack, scalar materialization off, no death, profile-only
```

| impl | measured sec | scalar steps/sec | physical rows/sec | probe/boundary sec | recurrent/model sec | search/update sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` repeat | `10.237` | `6001.49` | `3000.74` | `4.679` | `0.915` | `2.408` | `2.492` |
| `dense_torch_mcts` v0 | `9.573` | `6418.31` | `3209.16` | `3.032` | `0.911` | `0.678` | `2.611` |
| `recurrent_toy` ceiling | `8.229` | `7466.60` | `3733.30` | `2.582` | `0.885` | `0.112` | `2.589` |

Plain read:

```text
Dense v0 works, but it is not the 10x fix. It is only about 1.07x faster than
the same-run GPU-latent CTree row, and about 0.86x of the recurrent-toy ceiling.
```

Likely cause:

- Python control still ran inside every simulation/depth.
- The prototype used CPU `.item()` gates in the traversal loop.
- It allocated/cloned path tensors repeatedly.
- H100 utilization was low, so the GPU was not saturated.

Follow-up code cleanup:

- removed inner-loop `.item()` gates;
- preallocated path node/action/active history;
- removed per-depth bootstrap clone during backprop.

Next row:

```text
Repeat dense_torch_mcts on the same H100 B512/A16/sim8 denominator after the
sync/allocation cleanup.
```

Decision rule:

```text
If cleaned dense moves toward the recurrent-toy ceiling, continue dense GPU
search. If not, stop polishing this prototype and move to array-native CTree
or a larger compiled/batched search service.
```

Cleanup v1 result:

Same denominator, root noise forced to `0.0`:

| impl | measured sec | scalar steps/sec | physical rows/sec | probe/boundary sec | search/update sec | recurrent/model sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` same-run repeat | `10.365` | `5927.42` | `2963.71` | `4.702` | `2.411` | `0.920` | `2.572` |
| `dense_torch_mcts` cleanup v1 | `7.958` | `7720.30` | `3860.15` | `2.594` | `0.709` | `0.792` | `1.948` |
| `recurrent_toy` same-run ceiling | `6.754` | `9097.07` | `4548.53` | `2.088` | `0.093` | `0.794` | `2.164` |

Plain read:

```text
The dense lane survived the hard falsifier. It is now about 1.30x faster than
same-run GPU-latent CTree and about 0.85x of the recurrent-toy ceiling on this
profile-only row.
```

Important caveats:

- still profile-only;
- still not stock LightZero CTree;
- still not a full `train_muzero` wall-clock result;
- still has dynamic indexing and many tiny eager Torch ops.

Second cleanup applied after this row:

- dense recurrent calls no longer synchronize CUDA every simulation;
- the array-ceiling probe skips host boolean filtering when all roots are legal;
- local lint, py_compile, and focused tests passed after reverting an
  over-aggressive `torch.inference_mode()` change that broke in-place bootstrap
  updates.

Second-cleanup H100 sim8 and sim16 rows are running.

Second-cleanup results:

| impl | sims | measured sec | scalar steps/sec | physical rows/sec | probe sec | search/update sec | recurrent/model sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_torch_mcts` cleanup v2 | 8 | `7.709` | `7969.41` | `3984.70` | `2.433` | `0.697` | `0.764` | `1.908` |
| `dense_torch_mcts` cleanup v2 | 16 | `14.857` | `4135.37` | `2067.68` | `5.262` | `2.514` | `1.556` | `2.394` |

Plain read:

```text
The second cleanup only nudged sim8. Sim16 is the danger sign: this dense eager
Torch tree scales poorly as simulation count grows because the depth loop and
dynamic indexing cost grow with the tree.
```

Follow-up rows launched:

- same-denominator sim16 `direct_ctree_gpu_latent`;
- same-denominator sim16 `recurrent_toy`.

Do not choose the next lane until those rows return.

## 2026-05-22 Fresh H100 Search-Boundary Ladder

Status: profile-only. No live Coach run, checkpoint, eval, GIF, or tournament
artifact was changed.

Common denominator unless noted:

```text
H100, B512 physical rows, 2 player roots per row, A16 actors,
60 measured steps, 15 warmup steps, uint8 [B,2,4,64,64] stack,
no scalar timestep materialization, profile no-death, root noise 0.0
```

Fresh rows:

| impl | sims | measured sec | scalar roots/sec | boundary/probe sec | search/update sec | model/recurrent sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock_facade` | 8 | `25.282` | `2430.21` | `20.871` | `6.110` | `1.202` | `2.017` |
| `direct_ctree_arrays` | 8 | `12.266` | `5008.97` | `7.822` | `5.834` | `0.982` | `1.975` |
| `direct_ctree_gpu_latent` | 8 | `8.141` | `7547.12` | `3.819` | `1.962` | `0.783` | `1.921` |
| `dense_torch_mcts` fixed-shape before semantic fix | 8 | `8.711` | `7053.10` | `2.777` | `0.657` | `0.567` | `2.401` |
| `dense_torch_mcts` fixed-shape after semantic fix | 8 | `7.413` | `8288.37` | `2.363` | `0.525` | `0.446` | `2.106` |
| `recurrent_toy` ceiling | 8 | `4.787` | `12834.00` | `1.539` | `0.075` | `0.485` | `1.581` |
| `stock_facade` | 16 | `29.337` | `2094.27` | `24.795` | `11.854` | `1.431` | `2.191` |
| `direct_ctree_arrays` | 16 | `17.818` | `3448.25` | `13.485` | `11.777` | `1.512` | `1.939` |
| `direct_ctree_gpu_latent` | 16 | `9.998` | `6145.25` | `5.638` | `3.713` | `1.279` | `1.920` |
| `dense_torch_mcts` fixed-shape before semantic fix | 16 | `11.225` | `5473.26` | `4.174` | `1.598` | `1.049` | `2.091` |
| `dense_torch_mcts` fixed-shape after semantic fix | 16 | `14.309` | `4293.88` | `5.313` | `2.103` | `1.285` | `2.660` |
| `recurrent_toy` ceiling | 16 | `6.685` | `9191.12` | `2.433` | `0.145` | `0.950` | `1.994` |

Notes:

- The sim8 `recurrent_toy` row ran on an H100 NVL 95GB image, while most other
  rows used H100 80GB. Treat it as a ceiling, not an exact matched denominator.
- `dense_torch_mcts` is a profile-only GPU tensor search. It is not stock
  LightZero CTree and is not wired into Coach training.
- The semantic fix changed dense backup from "child bootstrap only" to
  `reward + discount * child_value`, fixed root-noise masking, and kept
  fallback visit totals separate from clamped totals.
- The direct rows use real LightZero model calls and CTree search. The correct
  direct flag family is
  `--hybrid-lightzero-mcts-arrays-boundary-probe` with
  `--hybrid-lightzero-mcts-arrays-boundary-impl ...`; array-ceiling modes are
  `policy_arrays`, `recurrent_toy`, and `dense_torch_mcts`.

Plain read:

```text
direct_ctree_gpu_latent is still the best practical LightZero-shaped boundary.
It is about 3.1x faster than the stock facade at sim8 and about 2.9x faster at
sim16 on this fresh ladder.

dense_torch_mcts is promising at sim8, but it fails the sim16 scaling gate after
the semantic fix. Do not promote eager dense Torch to training. If we want this
lane, the next step is compiled/fused fixed-shape search or a different batched
search service, not another small eager Torch polish.
```

Amdahl read:

```text
Rendering is not the corrected wall in this denominator. The wall is still the
search boundary: Python/control work, CPU tree state, per-simulation model/tree
handoff, output extraction, and observation/stack handoff. GPU-latent CTree
removes a real chunk of that wall; a 10x-class jump needs a larger batched or
compiled search architecture.
```

## 2026-05-22 Sim32 Search-Depth Falsifier

Status: profile-only. No live Coach run, checkpoint, eval, GIF, or tournament
artifact was changed.

Same H100 B512/A16/60 measured/15 warmup denominator as the fresh ladder, but
with `num_simulations=32`.

| impl | measured sec | scalar roots/sec | boundary/probe sec | search/update sec | model/recurrent sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` | `14.887` | `4127.10` | `9.719` | `7.434` | `2.273` | `2.286` |
| `dense_torch_mcts` after semantic fix | `30.614` | `2006.94` | `13.367` | `6.139` | `2.559` | `2.844` |
| `recurrent_toy` ceiling | `9.971` | `6161.83` | `4.187` | `0.415` | `2.269` | `2.605` |

Plain read:

```text
Deeper search makes eager dense Torch worse. It is not the next practical
training lane unless we compile/fuse or otherwise change the topology.
direct_ctree_gpu_latent remains the practical LightZero-shaped baseline.
```

CPU-scaling follow-up launched:

```text
Same sim16 direct_ctree_gpu_latent row, but with Modal CPU allocation changed
from H100+4 CPU to H100+64 CPU. A stock-facade H100+64 CPU row is also running.
Modal rejected cpu=128 because the function CPU request limit is 64 cores. This
tests whether "more CPUs" helps the current LightZero CTree/search boundary, or
whether the wall is mostly single-loop Python/list/sync shape.
```

CPU-scaling result:

| impl | CPU cores | measured sec | scalar roots/sec | boundary/probe sec | search/collect sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_ctree_gpu_latent` | 4 | `9.998` | `6145.25` | `5.638` | `3.713` | `1.920` |
| `direct_ctree_gpu_latent` | 64 | `12.001` | `5119.60` | `6.594` | `4.214` | `2.406` |
| `stock_facade` | 4 | `29.337` | `2094.27` | `24.795` | `11.854` | `2.191` |
| `stock_facade` | 64 | `34.599` | `1775.77` | `28.800` | collect-forward `25.460` | `2.724` |

Plain read:

```text
More CPU cores did not help this boundary. It made both direct GPU-latent CTree
and the stock facade slower on the same profile shape. The bottleneck is not a
simple CPU-capacity shortage; it is the search/boundary topology: Python/list
control, synchronization, CPU/GPU handoff, and object-shaped LightZero APIs.
```

## 2026-05-22 Train-Facing Hook Implementation

Status: local code/test only. No live Coach run, checkpoint, eval, GIF, or
tournament artifact was changed.

Implemented the first profile-only full-loop bridge:

```text
collect_search_backend=direct_ctree_gpu_latent
```

Files touched:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `tests/test_lightzero_phase_profiler.py`

What it does:

- rejects non-stock collect-search backends outside `mode=profile`;
- rejects the direct backend on CPU compute;
- patches `MuZeroPolicy._forward_collect` during the stock `train_muzero`
  profile window;
- keeps stock LightZero collector/replay/target/learner ownership;
- returns the stock per-env collect dict with raw legal-action visit counts;
- restores the patched method after the run.

Local validation:

```text
uv run python -m py_compile src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_phase_profiler.py
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py
uv run ruff check --ignore F401 src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_phase_profiler.py
```

Result:

```text
pytest: 8 passed
ruff: passed with F401 ignored because this launcher already has pre-existing
unused imports unrelated to the hook
```

Next experiment:

```text
Matched stock train_muzero profile A/B:
A: collect_search_backend=stock
B: collect_search_backend=direct_ctree_gpu_latent
```

The row must include at least one learner train call. The pass/fail metric is
full-loop wall-clock throughput, not profile-only roots/sec.

## 2026-05-22 Output-Fast, Packed Transfer, and RND Hash Fix

Status: profile-only. No live Coach run, checkpoint, eval, GIF, or tournament
artifact was changed.

Matched H100 C64/sim16/3-learner no-RND rows after the output fast path:

| row | steps/sec | wall | policy collect | MCTS/search | recurrent | D2H | listify | output assembly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock repeat | `433.17` | `37.82s` | `17.10s` | `12.09s` | n/a | n/a | n/a | n/a |
| direct output-fast repeat | `566.19` | `28.94s` | `10.31s` | `8.06s` | `4.28s` | `2.47s` | n/a | `0.077s` |
| direct listify split | `472.13` | `34.70s` | `13.16s` | `10.55s` | `5.75s` | `3.30s` | `0.079s` | `0.076s` |
| direct packed-transfer | `480.24` | `34.12s` | `12.84s` | `10.25s` | `5.57s` | `3.21s` | `0.077s` | `0.071s` |

Plain read:

```text
The output fast path is the real small patch. Packed transfer is safe hygiene
but not a major speedup; the D2H-labelled bucket is mostly GPU wait/transform
and boundary shape, not Python .tolist().
```

RND meter rows after moving predictor/target hashing outside the 100-update
loop:

| row | steps/sec | wall | policy collect | MCTS/search | RND train | RND hash | RND estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock `rnd_meter_v0` hash-fixed | `351.02` | `46.68s` | `23.62s` | `17.30s` | `0.590s` | `0.131s` | `0.086s` |
| direct `rnd_meter_v0` hash-fixed | `448.52` | `36.53s` | `13.32s` | `10.66s` | `0.603s` | `0.140s` | `0.093s` |

Historical comparison before the hash fix:

| row | steps/sec | RND train | RND hash |
| --- | ---: | ---: | ---: |
| stock `rnd_meter_v0` before hash fix | `342.33` | `3.43s` | `2.92s` |
| direct `rnd_meter_v0` before hash fix | `410.55` | `3.55s` | `2.98s` |

Plain read:

```text
RND hashing was a real independent wall and is now mostly gone. The direct
search hook still wins the matched RND profile after that fix: 351 -> 449
steps/sec, about 1.28x. RND rows use the MCTS-root profile fallback for step
count, so compare only matched RND stock/direct rows unless a cleaner
collector-envstep counter is restored.
```

Current blocker:

```text
The active wall is back to collect/search topology. CTree still takes
per-simulation CPU/list reward, value, and policy-logit payloads. More CPU
cores already failed as a useful lane. The next meaningful implementation work
is array-native CTree or compiled/fused batched search, with parity gates.
```

### Local Direct CTree Compare Rerun

Status: local validation harness only. No live Coach run, checkpoint, eval, GIF,
or tournament artifact was changed.

Strict sim8 compare:

```text
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 8 --batch-rows 4 --num-simulations 8 \
  --impls direct_ctree_arrays,direct_ctree_gpu_latent \
  --max-mean-visit-l1 0.0 --max-max-visit-l1 0.0 \
  --min-action-agreement 1.0 \
  --max-max-value-abs-diff 0.0 \
  --max-max-predicted-value-abs-diff 0.0 \
  --max-max-policy-logit-abs-diff 0.0
```

Result:

```text
passed
direct_ctree_arrays: mean_action_agreement=1.0, max_visit_l1=0.0
direct_ctree_gpu_latent: mean_action_agreement=1.0, max_visit_l1=0.0
illegal actions: 0
gpu-latent enabled rows: 8/8
```

Strict sim16 neutral/tie-heavy compare:

```text
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 4 --batch-rows 4 --num-simulations 16 \
  --impls direct_ctree_gpu_latent \
  --root-noise-weight 0.0 \
  --max-mean-visit-l1 0.0 --max-max-visit-l1 0.0 \
  --min-action-agreement 1.0 \
  --max-max-value-abs-diff 0.0 \
  --max-max-predicted-value-abs-diff 0.0 \
  --max-max-policy-logit-abs-diff 0.0
```

Result:

```text
failed strict action/visit equality
mean_action_agreement=0.65625
mean_visit_l1=0.04296875
max_visit_l1=0.125
max value/logit diffs=0
illegal actions=0
gpu-latent enabled rows=4/4
```

Plain read:

```text
This is not a reason to chase CPUs or abandon the direct path. It is a reason
to keep the promotion contract honest: exact gates for forced cases, masks,
schema, values/logits, and replay fields; statistical gates for ordinary
tie-heavy collect rows.
```

### Harness And Architecture Reorientation

Status: local validation/docs/research only. No live Coach run, checkpoint,
eval, GIF, or tournament artifact was changed.

Added harness knobs to `scripts/compare_curvytron_direct_ctree_stock.py`:

- `--action-mask-scenario`: `random`, `all_legal`, `single_legal_cycle`,
  `mixed_legal_cycle`;
- `--require-gpu-latent-enabled`: fails if a `direct_ctree_gpu_latent` row does
  not actually report GPU-latent activation.

Added a train-facing schema smoke in `tests/test_lightzero_phase_profiler.py`.
It patches the actual `_forward_collect` hook installation path, uses a fake
stock output, and checks direct output keys/values, output-fast counters, and
fallback count.

Validation:

```text
uv run python -m py_compile scripts/compare_curvytron_direct_ctree_stock.py
uv run python scripts/compare_curvytron_direct_ctree_stock.py \
  --seeds 2 --batch-rows 2 --num-simulations 4 \
  --impls direct_ctree_arrays,direct_ctree_gpu_latent \
  --action-mask-scenario single_legal_cycle \
  --root-noise-weight 0.0 --strict-exact --require-gpu-latent-enabled
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py \
  -k "collect_search_hook"
uv run ruff check --ignore F401 \
  scripts/compare_curvytron_direct_ctree_stock.py \
  tests/test_lightzero_phase_profiler.py
```

Result:

```text
single-legal exact gate passed for both direct paths;
GPU-latent enabled rows 2/2;
illegal actions 0;
hook schema smoke 2 passed, 7 deselected;
ruff passed.
```

Neutral all-legal/mixed strict action/visit equality remains the wrong gate:
it can fail while values/logits and illegal-action checks are clean. Keep those
rows statistical.

Architecture reorientation:

```text
The direct hook is a useful 1.28-1.31x tactical profile win.
The 5-10x hypothesis requires a larger boundary change: compact batched search
ownership, not another scalar wrapper patch.
```

External source refresh recorded in:

```text
radical_search_architecture_reorientation_20260522.md
```

Next immediate ladder:

1. Add/finish a stronger train-facing hook-vs-stock parity test over forced
   masks and clear preferences.
2. Run one fixed-shape `dense_torch_mcts_compile_spike` falsifier.
3. If sim16 stays below `direct_ctree_gpu_latent`, pivot to array-native CTree
   or a MiniZero-style batched search service design.

### Train-Hook Forced-Mask Validation

Status: local validation only. No live Coach run, checkpoint, eval, GIF, or
tournament artifact was changed.

Added P0 train-facing hook tests in `tests/test_lightzero_phase_profiler.py`:

- mixed-mask raw visit contract;
- single-legal exact gate;
- fail-closed fractional, zero, and ready-env-id mismatch masks.
- all-actions-legal stochastic fast-path distributional parity against the
  stock selector.

Plain read:

```text
The direct train hook now proves the dangerous action-id target bug locally:
actions are full action ids, visit distributions remain raw legal-action visit
lists, and mixed masks do not accidentally use the all-actions-legal fast path.
```

Validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py \
  -k "collect_search_hook"
-> 8 passed, 7 deselected

uv run ruff check --ignore F401 tests/test_lightzero_phase_profiler.py
-> passed

uv run python -m py_compile tests/test_lightzero_phase_profiler.py
-> passed
```

### Direct-Row Attestation Gate

Status: local tooling/test only. No live Coach run, checkpoint, eval, GIF, or
tournament artifact was changed.

Tightened `scripts/summarize_curvytron_optimizer_profile_results.py` so direct
backend rows cannot pass `--require-attestation` unless they self-identify and
self-audit the direct hook:

```text
command.collect_search_backend == direct_ctree_gpu_latent
semantic_identity.collect_search_backend == command.collect_search_backend
collect_search_backend_direct_ctree_gpu_latent_calls > 0
collect_search_backend_output_rows > 0
collect_search_backend_fallback_calls == 0
```

Validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_summarize_curvytron_optimizer_profile_results.py
-> 3 passed

uv run ruff check \
  scripts/summarize_curvytron_optimizer_profile_results.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py
-> passed

uv run python -m py_compile \
  scripts/summarize_curvytron_optimizer_profile_results.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py
-> passed
```

### Mock Search-Service Ceiling Mode

Status: profile-only sidecar code and first H100 ceiling wave. No live Coach
run, checkpoint, eval, GIF, tournament, or trainer default was changed.

What changed:

```text
source_state_batched_observation_boundary_profile.py now accepts
hybrid_lightzero_array_ceiling_mode=mock_search_service.
```

What the mode does:

```text
real batched CurvyTron observation + legal mask
-> real scratch MuZero initial_inference
-> fake compact search-service action / visit / value arrays
```

What it deliberately does not do:

```text
no train_muzero
no policy.collect_mode.forward
no CTree roots/search/traverse/backprop
no recurrent rollout
no live-run side effects
```

Focused validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "mock_search_service or array_ceiling"
-> 11 passed, 81 deselected

uv run pytest -q -p no:cacheprovider \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py \
  -k "mock_search_service or array_ceiling"
-> 2 passed, 12 deselected

uv run ruff check ...
-> passed

uv run python -m py_compile ...
-> passed
```

First H100 wave launched on the same B512/A16/60 measured/15 warmup shape:

| row | purpose |
| --- | --- |
| `mock_search_service` sim8 | compact fake-search ceiling sanity |
| `mock_search_service` sim16 | main radical-architecture ceiling |
| `direct_ctree_gpu_latent` sim16 | current practical real-search comparator |
| `recurrent_toy` sim16 | no-CTree recurrent-model ceiling |

Decision rule:

```text
If mock sim16 is far above direct_ctree_gpu_latent sim16, search-service
architecture stays alive.
If mock sim16 is only close to direct_ctree_gpu_latent, search alone is not the
10x lane and the next wall is collector/replay/RND/topology.
```

### Durable Hybrid Profile Runner

Status: tooling/test plus a fresh profile-only H100 wave. No live Coach run,
checkpoint, eval, GIF, tournament, or trainer default was changed.

Why this was added:

```text
Detached raw Modal sessions are not a reliable experiment record. PTY handles
can disappear after compaction, and app logs can drop or hide the compact JSON.
```

New runner:

```text
scripts/run_curvytron_hybrid_observation_profile_manifest.py
```

Contract:

```text
input:  manifest from scripts/build_curvytron_hybrid_observation_profile_grid.py
action: run blocking profile-only modal commands, optionally in local parallel
output: artifacts/local/curvytron_hybrid_observation_profile_results/<experiment>/
        row_<id>_stdout.log
        row_<id>_result.json
        rows.jsonl
        collected_results.json
```

It rejects detached rows on purpose. This runner is for durable result capture,
not background job management.

Validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_curvytron_hybrid_observation_profile_manifest_runner.py
-> 3 passed

uv run ruff check \
  scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  tests/test_curvytron_hybrid_observation_profile_manifest_runner.py
-> passed

uv run python -m py_compile \
  scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  tests/test_curvytron_hybrid_observation_profile_manifest_runner.py
-> passed
```

Dry-run smoke:

```text
opt-hybrid-runner-smoke-20260522
```

Fresh durable H100 falsifier wave launched with B512/A16, 60 measured steps,
15 warmup steps, `direct_gray64`, persistent GPU renderer, `uint8` stack,
scalar materialization off, root noise `0.0`, and `host_uint8_pinned` input:

| experiment | rows |
| --- | --- |
| `opt-hybrid-durable-mock-h100-20260522a` | `mock_search_service` sim8, sim16 |
| `opt-hybrid-durable-direct-h100-20260522a` | `direct_ctree_gpu_latent` sim8, sim16 |
| `opt-hybrid-durable-recurrent-h100-20260522a` | `recurrent_toy` sim8, sim16 |

Decision read after results:

```text
mock much faster than direct:
  compact search-service ownership remains a serious candidate.
mock close to direct:
  search-service alone cannot buy 5-10x; move the target to collector/replay/RND
  topology or a native contiguous-env contract.
recurrent_toy much faster than direct:
  real MCTS/search bookkeeping is the wall.
recurrent_toy close to direct:
  model/recurrent pressure is not the only escape route; look at wrapper and
  output topology.
```

Results:

| row | sim | roots/sec | measured sec | last probe total | last search/update | last model/recurrent | read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mock_search_service` | 8 | `11978.11` | `5.13` | `0.0066s` | n/a | n/a | compact search-service ceiling |
| `mock_search_service` | 16 | `11648.29` | `5.27` | `0.0065s` | n/a | n/a | compact search-service ceiling |
| `direct_ctree_gpu_latent` | 8 | `7233.81` | `8.49` | `0.0429s` | `0.0262s` | `0.0131s` | current real-CTree comparator |
| `direct_ctree_gpu_latent` | 16 | `5303.97` | `11.58` | `0.0880s` | `0.0681s` | `0.0256s` | current real-CTree comparator |
| `recurrent_toy` | 8 | `9623.42` | `6.38` | `0.0189s` | `0.0018s` | `0.0103s` | no-CTree recurrent-model ceiling |
| `recurrent_toy` | 16 | `8512.57` | `7.22` | `0.0313s` | `0.0037s` | `0.0208s` | no-CTree recurrent-model ceiling |

Plain read:

```text
mock_search_service sim16 is about 2.20x faster than direct_ctree_gpu_latent.
recurrent_toy sim16 is about 1.60x faster than direct_ctree_gpu_latent.
mock_search_service sim16 is about 1.37x faster than recurrent_toy.
```

Decision:

```text
The search-service architecture is still alive, but this row does not prove a
standalone 10x search rewrite. It says the compact arrays/search-service
boundary has real headroom, while a larger 5-10x path probably also needs the
PufferLib-style part: contiguous native actor/env buffers, fewer scalar Python
objects, overlapped collection/model work, and compact replay materialization.
```

Do not promote this to Coach training:

```text
mock_search_service is deliberately not MCTS.
recurrent_toy is deliberately not MCTS.
direct_ctree_gpu_latent is still profile-only unless its promotion gates and
matched full-loop rows pass.
```

## 2026-05-22 Local Native-Vector Boundary Probe

Context:

```text
Local script: scripts/profile_hybrid_batched_observation_manager.py
Shape: B512/A16/steps100
Observation: zero
Stack dtype: uint8
Payload: no-pickle
Live runs: no
Semantics: profile-only Puffer-style compact array consumer, not MCTS
```

Rows:

| row | timesteps/sec | measured sec over 102400 timesteps | read |
| --- | ---: | ---: | --- |
| no scalar + native probe | `23515` | `4.35s` | fastest compact-boundary row |
| scalar-only | `18604` | `5.50s` | scalar LightZero materialization is a real cost |
| scalar + native probe | `17380` | `5.89s` | paying scalar materialization plus native probe is worse |

Component prices from the fresh report:

```text
scalar materialization: about 2.07s over 102400 timesteps, about 20.2 us/timestep
native compact probe:   about 0.62s over 102400 timesteps, about  6.1 us/timestep
actor_step_wall:        about 3.42s in the no-scalar row, about 33.4 us/timestep
```

Plain read:

```text
The object edge matters.
The compact native probe itself is cheap.
Actor/env scheduling is now visible and can become the next Amdahl wall.
```

Do not promote this to Coach training. It is a local topology probe. Use it to
shape the next falsifier: real contiguous env buffers plus compact
search/replay-shaped outputs, with actor scheduling, env step, observation,
consumer, scalarization, and replay materialization timed separately.

### Local CPU-Oracle Comparator

Follow-up rows:

```text
B128/A8/steps40/warmup10
Observation: cpu-oracle renderer-backed stack
Stack dtype: uint8
Payload: no-pickle
Live runs: no
```

| row | timesteps/sec | measured sec | renderer/observation sec | edge/probe sec | read |
| --- | ---: | ---: | ---: | ---: | --- |
| no scalar + native probe | `192.91` | `53.08s` | `52.23s` | native probe `0.065s` | renderer dominates; not useful for judging scalar edge |
| scalar-only | `192.17` | `53.29s` | `52.35s` | scalar materialization `0.133s` | same wall; scalar edge is hidden by CPU renderer |

Plain read:

```text
The CPU-oracle local row is a renderer benchmark wearing the hybrid-manager
shape. It does not falsify the Puffer-style buffer thesis because the old
renderer consumes almost all wall time. Use zero-observation or GPU-observation
rows for topology claims, and use CPU-oracle rows only as a reminder that a
slow observation surface can still drown any boundary work.
```

One extra topology row:

```text
B256/A8/steps40/warmup10/zero-observation/uint8/no-scalar/native-probe
timesteps/sec: 32764
measured sec: 0.625s over 20480 timesteps
native probe: 0.095s
actor wall: 0.472s
observation/stack: 0.045s
```

Plain read:

```text
With observation removed, the compact consumer is cheap and actor scheduling is
the visible cost. That matches the PufferLib lesson: the next prototype must
own the actor/env buffer layout, not just delete scalar materialization.
```

## 2026-05-22 Native Actor Buffer Falsifier

Status: profile-only code path. No trainer defaults, live runs, checkpoints,
evals, GIFs, or tournament paths touched.

What changed:

```text
HybridObservationProfileConfig(native_actor_buffer=True)
scripts/profile_hybrid_batched_observation_manager.py --native-actor-buffer
```

This lets the in-process actors write reward/done/episode/alive/action-mask
fields directly into parent-owned compact arrays instead of returning
`HybridActorStepPayload` objects and then merging them. It is zero-observation
only for now. Renderer-backed rows still use the older payload path because
render-state dictionaries need their own compact-buffer contract.

Validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py
-> 17 passed

uv run ruff check \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/profile_hybrid_batched_observation_manager.py \
  tests/test_source_state_hybrid_observation_profile.py
-> passed

uv run python -m py_compile \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/profile_hybrid_batched_observation_manager.py \
  tests/test_source_state_hybrid_observation_profile.py
-> passed
```

Matched local zero-observation topology rows:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
```

| row | timesteps/sec | measured sec | actor wall | gather merge | native probe | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| old actor payload + merge | `40477` | `2.53s` | `1.879s` | `0.0138s` | `0.460s` | current local topology baseline |
| native actor buffer | `67890` | `1.51s` | `0.907s` | `0.000018s` | `0.411s` | about `1.68x` faster on this denominator |

Tiny smoke row:

```text
B64/A4/steps10/warmup2/uint8/no-pickle/no-scalar/native-vector-probe
old payload path:      17612 timesteps/sec
native actor buffer:   31609 timesteps/sec
```

Plain read:

```text
The Puffer-style lesson is real at this boundary. Removing actor payload
objects plus parent merge roughly halves actor-wall time in the zero-observation
topology row. This is not a full training speedup yet, because real search,
replay/RND, and real observation can still dominate. But it proves the buffer
layout direction is worth carrying into the next compact search/replay-shaped
falsifier.
```

## 2026-05-22 PufferLib Repo Inspection

Status: read-only external architecture inspection. No CurvyTron production
code or live runs touched.

Source inspected:

```text
https://github.com/PufferAI/PufferLib
local clone: /private/tmp/pufferlib
```

Useful files:

- `src/vecenv.h`
- `src/bindings.cu`
- `src/kernels.cu`
- `src/pufferlib.cu`
- `examples/vectorization.py`

Concrete pattern found:

```text
StaticVec owns flat observation/action/reward/terminal/action-mask buffers.
Each env struct receives pointers into its assigned slice.
GPU mode uses pinned host buffers plus device buffers.
Threaded rollout chunks work by buffer and CUDA stream.
The allocator registers tensor shapes, then creates one contiguous allocation.
```

CurvyTron read:

```text
PufferLib is not the algorithm answer.
It is a buffer-boundary answer.
The next useful prototype should make CurvyTron look like a compact batch
provider, not another source of scalar LightZero timestep objects.
```

## 2026-05-22 Wide Compact Sidecar Probe

Code change:

```text
HybridCompactBatch
HybridBatchedStackProbe.run_compact_batch(batch)
```

The old probe contract only saw:

```text
observation[B,P,4,64,64]
action_mask[B,P,3]
```

That was too thin. Replay, search, RND, target rows, terminal handling, and
autoreset all need row/player/reward/done sidecars. The profile harness now
prefers `run_compact_batch(batch)` when present and falls back to the legacy
two-argument `run(observation, action_mask)` for older probes.

Validation:

```text
uv run ruff format src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/profile_hybrid_batched_observation_manager.py \
  tests/test_source_state_hybrid_observation_profile.py

uv run python -m py_compile \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/profile_hybrid_batched_observation_manager.py \
  tests/test_source_state_hybrid_observation_profile.py

uv run ruff check \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/profile_hybrid_batched_observation_manager.py \
  tests/test_source_state_hybrid_observation_profile.py

uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py
-> 19 passed, 2 warnings
```

New tests cover:

- Legacy `run(observation, action_mask)` compatibility.
- Wide compact sidecars for row/player ids, reward, target reward, done roots,
  terminal rows, autoreset rows, and final observations.
- Native actor buffer versus payload merge parity on terminal/autoreset rows.

Matched local zero-observation rows after the wide sidecar:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
```

| row | timesteps/sec | measured sec | actor wall | gather merge | native probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| actor payload + merge | `22060` | `4.64s` | `3.586s` | `0.0261s` | `0.715s` |
| native actor buffer | `30499` | `3.36s` | `2.180s` | `0.000043s` | `0.869s` |

Read:

```text
The wide sidecar makes the probe more honest and still preserves a native
actor-buffer win, about 1.38x on this local denominator. This row is slower
than the earlier narrow-probe row because the current local code/env and probe
work changed; do not compare it as a clean regression without a frozen commit.
The useful conclusion is architectural: the compact boundary now carries enough
metadata to test replay/search/RND-shaped consumers without scalar timesteps.
```

Terminal/autoreset smoke:

```text
B4/A2/steps2/max_ticks1/uint8/no-pickle/no-scalar/native-vector-probe/native-actor-buffer
```

Last telemetry reported:

```text
native_vector_boundary_batch_contract = compact_row_player_sidecar_v1
native_vector_boundary_done_count = 4
native_vector_boundary_done_root_count = 8
native_vector_boundary_terminal_count = 4
native_vector_boundary_autoreset_count = 4
native_vector_boundary_final_observation_present = true
native_vector_boundary_final_observation_rows = 4
```

Plain read:

```text
The profile-only compact sidecar can now see the terminal/final/autoreset facts
that matter for real loop correctness. The next serious falsifier should plug a
real search/replay/RND-shaped consumer into this sidecar instead of measuring
only observation and action masks.
```

## 2026-05-22 Compact Sidecar Into Direct CTree Hook

Code change:

```text
_LightZeroCollectForwardStackProbe.run_compact_batch(batch)
```

This connects the compact row/player sidecar to the existing profile-only
direct CTree arrays path. It does not change trainer defaults and it does not
touch live runs.

New compact fields used by the hook:

```text
to_play[M]
active_root_mask[M]
target_reward[M,1]
policy_env_id[M]
policy_env_row[M]
policy_player[M]
done_root[M]
terminal/autoreset/final-observation sidecars
```

The hook checks the sidecar before search:

- row/player ids must be row-major;
- fixed-opponent `to_play` must be `-1` for active roots;
- `target_reward` must match row-major reward;
- `active_root_mask` must match `legal_actions_any && !done_root`;
- inactive roots have action masks zeroed before the direct CTree call.

Validation:

```text
uv run ruff format \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  scripts/profile_hybrid_batched_observation_manager.py \
  tests/test_source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py

uv run python -m py_compile ...

uv run ruff check ...

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "hybrid_profile or compact_batch or direct_ctree_returns_compact_arrays"
-> 11 passed, 100 deselected, 2 warnings

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py
-> 92 passed, 2 warnings
```

Local compact profile refresh:

```text
B512/A16/steps100/warmup20/uint8/no-pickle/no-scalar/native-vector-probe
payload+merge:       44651.32 timesteps/sec
native actor buffer: 79650.24 timesteps/sec
```

This row is local/profile-only and uses zero observations. It is useful for
pricing the object boundary, not for Coach launch advice.

Remote smoke:

```text
First attempt failed before work:
ValueError: hybrid_lightzero_mcts_arrays_boundary_probe requires
observation_renderer_backend='jax_gpu_persistent_policy_framebuffer_profile'

Second attempt also failed before work:
ValueError: persistent policy framebuffer requires render_surface='direct_gray64'

Third attempt also failed before work:
ValueError: min_render_trail_slots must be <= trail_slots
```

Read:

```text
Good failures. The direct CTree boundary now refuses stale renderer/surface
combinations and invalid trail-slot configs instead of silently measuring the
wrong thing.
```

Corrected remote smoke:

```text
Modal app: ap-RztU5jMKmKBpXuaDY3vZB0
compute: L4/T4
batch_size=2, actor_count=1, steps=1, warmup_steps=1
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
hybrid_stack_storage_dtype=uint8
hybrid_materialize_scalar_timestep=false
hybrid_lightzero_mcts_arrays_boundary_impl=direct_ctree_arrays
num_simulations=1
```

Result:

```text
ok=true
profile_only=true
touches_live_runs=false
materialized_timestep_count=0
batched_stack_probe_backend_name=lightzero_mcts_arrays_direct_ctree_consumer
compact_batch_contract=compact_row_player_sidecar_v1
lightzero_compact_batch_active_root_count=4
lightzero_compact_batch_to_play_checksum=-4
lightzero_illegal_action_count=0
lightzero_policy_device=cpu
renderer_backend_name=jax_gpu_persistent_policy_framebuffer_profile
```

Plain read:

```text
The compact sidecar reaches the real direct CTree arrays profile hook remotely.
This closes the wiring smoke only. It is still profile-only and not Coach
training advice.
```

Follow-up validation from Planck critique:

```text
Compact action masks are now validated before bool coercion.
The compact hook now also checks done_root, terminal/autoreset masks, final
observation mask, and sidecar shapes before search.
Focused malformed-sidecar tests passed.
```

## 2026-05-22 Compact RND Input Proof

Code change:

```text
normalize_policy_gray64_stack_for_rnd(obs_batch)
extract_policy_gray64_latest_for_rnd_from_compact_observation(obs_batch, target_reward)
```

Why:

```text
RND currently expects normalized policy gray64 latest frames.
The compact sidecar stores [B,P,4,64,64] stacks, often uint8.
This adapter proves the compact sidecar can feed RND input without first
materializing scalar LightZero timesteps.
```

Validation:

```text
tests/test_exploration_bonus.py -k "compact_policy_gray64_adapter or latest_gray64_adapter"
-> 9 passed

tests/test_source_state_hybrid_observation_profile.py -k "compact_batch_can_feed_rnd or native_actor_buffer_rejects_duplicate"
-> 2 passed

Full focused sweep:
tests/test_source_state_hybrid_observation_profile.py
tests/test_source_state_batched_observation_boundary_profile.py
tests/test_exploration_bonus.py
-> 137 passed, 2 warnings
```

Plain read:

```text
RND latest-frame extraction can now consume the compact sidecar with scalar
timestep materialization off. This is an input/contract proof, not a claim that
RND training cadence or positive-bonus normalization is solved.
```

## 2026-05-22 Post-Guard Native Buffer Refresh

After adding compact-sidecar validation, RND compact input extraction, and
native actor-buffer row ownership checks, the local zero-observation topology
signal still holds.

Command shape:

```text
scripts/profile_hybrid_batched_observation_manager.py
--batch-size 512
--actor-count 16
--steps 100
--warmup-steps 20
--max-ticks 2000
--stack-storage-dtype uint8
--no-pickle-payload
--no-materialize-scalar-timestep
--native-vector-boundary-probe
```

Rows:

| row | timesteps/sec | measured sec | actor wall | gather merge | native probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| payload + merge | `40471.44` | `2.530s` | `1.888s` | `0.0134s` | `0.4485s` |
| native actor buffer | `66136.26` | `1.548s` | `0.939s` | `0.000016s` | `0.4089s` |

Plain read:

```text
Native actor buffer remains about 1.63x faster on this local/profile-only
zero-observation denominator. This is still boundary evidence, not Coach
training speed.
```

Terminal/autoreset smoke:

```text
B4/A2/steps2/max_ticks1/native-actor-buffer/native-vector-probe
terminal_row_count=8
autoreset_row_count=8
native_vector_boundary_done_root_count=8
native_vector_boundary_active_root_count_from_sidecar=0
native_vector_boundary_final_observation_present=true
native_vector_boundary_final_observation_rows=4
```

Plain read:

```text
Terminal rows are visible to the compact consumer and are not active search
roots. The sidecar is still only a one-step profile batch; replay/target chunks
remain the next proof.
```

## 2026-05-22 Compact Target-Row Adapter Proof

Implemented a profile-only bridge:

```text
HybridCompactBatch
+ selected_action[active_root]
+ visit_policy[active_root,3]
+ root_value[active_root]
-> PolicyRowRecordV0
-> build_source_state_multiplayer_target_rows_v0(...)
```

Changed files:

- `src/curvyzero/training/compact_policy_row_bridge.py`
- `tests/test_multiplayer_source_state_target_rows.py`

Validation:

```text
uv run ruff check src/curvyzero/training/compact_policy_row_bridge.py tests/test_multiplayer_source_state_target_rows.py
-> all checks passed

uv run pytest -q -p no:cacheprovider tests/test_multiplayer_source_state_target_rows.py
-> 18 passed, 2 warnings
```

What it checks:

- compact row-major `policy_env_id`, `policy_env_row`, and `policy_player`;
- fixed-opponent `to_play=-1`;
- `active_root_mask == ~done_root & action_mask.any(axis=1)`;
- binary action masks before bool coercion;
- active-root ordered search outputs, including P4 and mixed active/done rows;
- legal selected actions;
- finite, nonnegative visit policies that sum to one and put zero mass on
  illegal actions;
- target reward equals compact reward reshaped to `[B*P,1]`;
- existing target-row builder still owns the temporal check that action at
  decision record `k` matches replay `joint_action[k+1]`.

Plain read:

```text
The compact sidecar can cross the replay/target compatibility edge in a checked
way without scalar LightZero timestep objects in this adapter proof.
```

Non-claims:

```text
not native LightZero GameSegment
not stock replay integration
not learner update
not policy improvement
not Coach-facing speed
not live-run contact
```

## 2026-05-22 Combined Direct-CTree Output To Target-Row Proof

Added a local combined edge test:

```text
HybridCompactBatch
-> _LightZeroCollectForwardStackProbe.run_compact_batch(...)
-> direct CTree compact action/visit/value arrays
-> build_policy_row_records_from_compact_search_v0(...)
-> build_source_state_multiplayer_target_rows_v0(...)
```

Validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py -k "direct_ctree_compact_output_can_feed_checked_target_rows"
-> 1 passed

uv run ruff check src/curvyzero/training/compact_policy_row_bridge.py tests/test_multiplayer_source_state_target_rows.py tests/test_source_state_batched_observation_boundary_profile.py
-> all checks passed

uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py tests/test_multiplayer_source_state_target_rows.py
-> 112 passed, 2 warnings

uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_exploration_bonus.py tests/test_multiplayer_source_state_target_rows.py
-> 156 passed, 2 warnings
```

Plain read:

```text
The compact search-output and target-row adapters line up in one local
profile-shaped chain. This is still not speed and not trainer integration.
```

Aggressive next move:

```text
Build a closed compact batch consumer falsifier:
compact batch -> real compact search output -> RND latest-frame input ->
compact target rows, scalar objects only at validation edges.
```

Kill rule:

```text
If the closed compact consumer cannot plausibly beat current direct profile
rows by about 3x, stop treating direct-CTree wrapper work as the 5-10x lane and
escalate to a MiniZero/KataGo-style batched search service or native/vector
buffer prototype.
```

## 2026-05-22 Closed Compact Consumer Local Falsifier

Status: local profile-only. No live Coach training run, checkpoint, eval, GIF,
or tournament artifact was touched.

What this tested:

```text
HybridCompactBatch
-> compact RND latest-frame input
-> mock compact legal action/visit/value arrays
-> compact target validation
```

Important code change:

```text
extract_policy_gray64_latest_for_rnd_from_compact_observation(...)
```

now slices the latest channel first and normalizes only `[B*P,1,64,64]`.
Before this, it normalized the full `[B,P,4,64,64]` stack and then threw away
three channels. That was wasted memory movement in the compact path.

Focused validation:

```text
uv run ruff check src/curvyzero/training/exploration_bonus.py tests/test_exploration_bonus.py scripts/profile_hybrid_batched_observation_manager.py
-> all checks passed

uv run pytest -q -p no:cacheprovider tests/test_exploration_bonus.py -k "compact_policy_gray64"
-> 4 passed, 21 deselected

uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_exploration_bonus.py tests/test_multiplayer_source_state_target_rows.py
-> 158 passed, 2 warnings
```

After tightening compact target shape to exactly `[B*P,1]` and adding a flat
`[N,4,64,64]` compact RND test:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_exploration_bonus.py tests/test_multiplayer_source_state_target_rows.py
-> 159 passed, 2 warnings
```

Local profile rows:

| shape | probe | target mode | timesteps/sec | measured sec | probe sec | note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| B512/A16/steps100 | closed compact | arrays | `57888.20` | `1.77s` | `0.67s` | after latest-only RND fix |
| B512/A16/steps100 | closed compact | arrays | `62827.92` | `1.63s` | `0.59s` | after explicit `[B*P,1]` target guard |
| B512/A16/steps100 | native-vector mock | n/a | `69091.92` | `1.48s` | `0.38s` | local ceiling control |
| B512/A16/steps100 | closed compact | records | `47362.90` | `2.16s` | `0.91s` | per-root record objects still cost |
| B1024/A16/steps120 | closed compact | arrays | `54262.42` | `4.53s` | `1.89s` | bigger batch, still stable |
| B1024/A16/steps120 | native-vector mock | n/a | `60008.81` | `4.10s` | `1.34s` | matched ceiling |
| B512/A32/steps120 | closed compact | arrays | `29187.46` | `4.21s` | `1.08s` | too many actor partitions regressed |
| B512/A32/steps120 | native-vector mock | n/a | `32311.10` | `3.80s` | `0.73s` | matched ceiling |
| B1024/A8/steps120 | closed compact | arrays | `44271.30` | `5.55s` | `2.54s` | fewer, fatter chunks also regressed |
| B1024/A8/steps120 | native-vector mock | n/a | `52083.36` | `4.72s` | `1.64s` | matched ceiling |
| B2048/A16/steps80 | closed compact | arrays | `71605.25` | `4.58s` | `2.24s` | best closed compact local row |
| B2048/A16/steps80 | native-vector mock | n/a | `80443.30` | `4.07s` | `1.70s` | best local ceiling row |

Plain read:

```text
The compact sidecar itself can be made fast in this local denominator. The RND
latest-frame bug-shaped copy was real, and per-root PolicyRowRecord objects are
still slower than compact arrays. But this is not real MCTS and not
train_muzero. The remaining 5-10x question is the real collect/search/replay
boundary: keep the batch compact through real search and replay, or build a
MiniZero/KataGo-style batched search service.
```

Batch-shape read:

```text
On this local harness, B2048/A16 was better than B512/A16 and B1024/A16, while
B512/A32 and B1024/A8 regressed. For the next local compact prototypes, prefer
fewer, fatter batches around A16 before trying more actor partitions.
```

## 2026-05-22 Production Hook All-Legal Root-Prep Cleanup

Status: profile-only code path only. No live Coach training run, checkpoint,
eval, GIF, or tournament artifact was touched.

Side-agent critique found that the train-facing `direct_ctree_gpu_latent` hook
still did avoidable Python work before real CTree search in the common CurvyTron
case where every root has the full action mask `[1,1,1]`.

Patch:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Changes:

- parse rectangular action masks as one matrix;
- remove unused `mask_arrays`;
- use a shared legal-action pattern for the all-actions-legal case;
- generate Dirichlet root noise as one matrix instead of one Python call per
  root;
- keep the public `dict[env_id] -> dict` LightZero collect output unchanged.

Validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_lightzero_phase_profiler.py -k "direct_ctree_collect_search_hook"
-> 8 passed, 7 deselected, 2 warnings

uv run pytest -q -p no:cacheprovider tests/test_exploration_bonus.py -k "curvy_rnd_reward_model_trains_predictor or compact_policy_gray64"
-> 5 passed, 20 deselected

uv run ruff check --ignore F401 src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py tests/test_lightzero_phase_profiler.py src/curvyzero/training/exploration_bonus.py tests/test_exploration_bonus.py
-> all checks passed
```

Plain read:

```text
This is a narrow cleanup, not the 5-10x architecture. It removes dumb root-prep
Python work from the real train-facing hook while preserving stock public
output shape. The bigger wall remains the CTree Python/list simulation loop and
the final scalar collect/replay contract.
```

## 2026-05-22 RND Meter Zero-Weight Estimate Cleanup

Status: profile-only-safe RND cleanup. No RND cadence, normalization, or
positive-reward semantics changed.

Patch:

```text
CurvyRNDRewardModel.estimate(...)
```

When `intrinsic_reward_weight == 0.0`, meter mode now avoids deep-copying
`target_reward` and avoids target-delta array work. It still runs the RND
forward estimate and metrics, then records:

```text
last_target_reward_changed = false
last_target_reward_delta_abs_mean = 0.0
last_target_reward_delta_abs_max = 0.0
```

The output target object is the original target reward in meter mode.

Plain read:

```text
This keeps RND meter mode reward-neutral and removes avoidable bookkeeping from
the weight-zero path. It does not change RND training cadence, min/max novelty
normalization, positive RND rewards, or stock LightZero reward-model cadence.
Those remain separate research/profile axes.
```

Combined validation after the direct-hook and RND cleanup patches:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_exploration_bonus.py tests/test_multiplayer_source_state_target_rows.py tests/test_lightzero_phase_profiler.py
-> 174 passed, 2 warnings
```

## 2026-05-22 Mock Search-Service Public Output Edge

Status: profile-only tooling. No trainer defaults, Coach runs, checkpoints,
evals, GIFs, or tournaments were touched.

Patch:

```text
--hybrid-lightzero-mock-service-materialize-public-output
```

The mock search-service ceiling can now optionally build LightZero-shaped
public collect dictionaries from compact arrays. This prices the object/dict
edge without pretending to run real MCTS.

Focused validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_source_state_batched_observation_boundary_profile.py -k "mock_search_service or hybrid_profile_grid_can_emit_mock or validate_boundary_config_accepts_mock or rejects_public_output"
-> 7 passed, 104 deselected, 2 warnings

uv run ruff check src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_boundary_profile.py scripts/build_curvytron_hybrid_observation_profile_grid.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> all checks passed

uv run python -m py_compile src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py scripts/build_curvytron_hybrid_observation_profile_grid.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> passed
```

Local compact scale checks:

| shape | profile-only mode | timesteps/sec | read |
| --- | --- | ---: | --- |
| B4096/A16/steps80 | closed compact arrays | `66772.29` | stable wider batch |
| B4096/A8/steps80 | closed compact arrays | `68396.37` | slightly better than A16 |
| B2048/A32/steps80 | closed compact arrays | `44946.41` | too many actor partitions hurt |

Plain read:

```text
The compact sidecar stays fast at B4096. More actor partitions are not free.
The next real question is whether compact search output can stay compact
through replay/target writing, or whether converting back to public LightZero
dicts burns the headroom.
```

Remote H100 public-output edge rows:

| experiment | public output | steps/sec | measured sec | probe sec | last public-output sec | read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `opt-mock-public-edge-h100-off-20260522a` | off | `8543.81` | `7.19s` | `1.65s` | `0.0000s` | compact ceiling |
| `opt-mock-public-edge-h100-on-20260522a` | on | `8285.40` | `7.42s` | `1.97s` | `0.0023s` | about `3%` wall hit |

Plain read:

```text
Turning compact mock search output into LightZero-shaped public dicts is not
free, but it is not the main wall on this row. The bigger gap remains real
search/CTree behavior and the larger replay/trainer topology, not just final
collect dict materialization.
```

Fresh same-shape comparators:

| experiment | mode | steps/sec | measured sec | probe sec | read |
| --- | --- | ---: | ---: | ---: | --- |
| `opt-current-direct-vs-mock-h100-direct-20260522a` | `direct_ctree_gpu_latent` | `5382.86` | `11.41s` | `0.0889s` | current real-CTree comparator |
| `opt-current-direct-vs-mock-h100-recurrent-20260522a` | `recurrent_toy` | `9068.59` | `6.78s` | `0.0396s` | no-CTree recurrent pressure reference |

Current matched read:

```text
mock_search_service compact:      8543.81 steps/sec
mock_search_service public edge:  8285.40 steps/sec
direct_ctree_gpu_latent:          5382.86 steps/sec
recurrent_toy:                    9068.59 steps/sec

The mock compact shape is about 1.59x over current direct CTree and the
recurrent toy is about 1.69x. That is real headroom, but not a standalone 10x
search-only proof.
```

## 2026-05-22 Compact Target Rows Without PolicyRowRecord Objects

Status: local target/replay parity proof. No Modal, trainer defaults, live
runs, checkpoints, evals, GIFs, or tournaments were touched.

Patch:

```text
build_compact_target_rows_from_search_arrays_v0(...)
```

The new builder consumes `HybridCompactBatch` plus compact search arrays and
builds `SourceStateMultiplayerTargetRowsV0` directly, without first allocating
one `PolicyRowRecordV0` object per active root.

Focused validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_multiplayer_source_state_target_rows.py -k "compact_search_arrays_build_target_rows_without_policy_record_objects or compact_search_arrays_use_terminal_final_observation_without_records or compact_search_arrays_validate_without_policy_record_objects"
-> 3 passed, 18 deselected, 2 warnings

uv run ruff check src/curvyzero/training/compact_policy_row_bridge.py tests/test_multiplayer_source_state_target_rows.py
-> all checks passed

uv run python -m py_compile src/curvyzero/training/compact_policy_row_bridge.py tests/test_multiplayer_source_state_target_rows.py
-> passed
```

Broader focused validation after all 2026-05-22 optimizer edits:

```text
uv run pytest -q -p no:cacheprovider tests/test_source_state_hybrid_observation_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_exploration_bonus.py tests/test_multiplayer_source_state_target_rows.py tests/test_lightzero_phase_profiler.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> 193 passed, 2 warnings

uv run ruff check src/curvyzero/training/compact_policy_row_bridge.py src/curvyzero/training/exploration_bonus.py src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_multiplayer_source_state_target_rows.py tests/test_exploration_bonus.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py scripts/build_curvytron_hybrid_observation_profile_grid.py
-> all checks passed
```

Plain read:

```text
This is not a trainer speed claim yet. It is a semantic gate: compact search
arrays can now produce the same target rows as the existing object bridge for
live rows and terminal/final-observation rows.
```

## 2026-05-22 Compact Replay Contract Hardening

Status: local semantic fix plus tests. No Modal, live Coach runs, eval, GIF, or
tournament state was touched.

Darwin found the important bug:

```text
compact_root_row != replay policy_row once active roots are not full row-major.
```

The bridge now treats them separately:

- `compact_root_row`: flat `[B*P]` row from the compact search/root batch;
- `policy_row`: compacted replay policy-row index from `chunk.policy_rows`;
- `source_record_ref.policy_env_id`: preserved from the compact root row.

It also validates that the compact batch observation/reward/done matches the
replay chunk record before writing target rows. This catches stale frames and
player-perspective swaps instead of silently producing plausible rows from the
wrong input.

Focused validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py tests/test_multiplayer_source_state_target_rows.py -k "compact"
-> 14 passed, 13 deselected, 2 warnings

uv run ruff check src/curvyzero/training/compact_policy_row_bridge.py tests/test_multiplayer_source_state_target_rows.py tests/test_compact_search_replay_contract.py scripts/benchmark_lightzero_ctree_no_model.py
-> all checks passed

uv run python -m py_compile scripts/benchmark_lightzero_ctree_no_model.py src/curvyzero/training/compact_policy_row_bridge.py tests/test_compact_search_replay_contract.py tests/test_multiplayer_source_state_target_rows.py
-> passed
```

Covered now:

- two-record live plus terminal/final-observation rows;
- three-record `record_index=1` rows;
- non-prefix active roots, including active compact roots `[1, 3]`;
- RND latest-frame row order;
- player-perspective swap rejection;
- non-identity `policy_env_id` provenance.

## 2026-05-22 No-Model LightZero CTree Microbench

Status: local falsifier script added:

```text
scripts/benchmark_lightzero_ctree_no_model.py
```

This script does not train, step CurvyTron, call Modal, or call a neural
network. It prices LightZero's MuZero CTree list ABI directly:

```text
Roots.prepare -> batch_traverse -> list payload -> batch_backpropagate
  -> get_distributions/get_values
```

First rows:

| roots | sims | legal profile | CTree nodes/sec | fake-flat nodes/sec / CTree | CTree boundary | CTree core |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `256` | `8` | all3 | `1.57M` | `12.9x` | `15.9%` | `62.9%` |
| `512` | `16` | all3 | `1.63M` | `17.6x` | `12.8%` | `69.4%` |
| `1024` | `16` | all3 | `0.99M` | `34.6x` | `32.4%` | `51.9%` |
| `2048` | `16` | mixed_2of3 | `0.98M` | `42.1x` | `38.9%` | `48.1%` |
| `2048` | `32` | all3 | `1.06M` | `35.9x` | `29.7%` | `54.3%` |

Artifacts:

```text
artifacts/local/ctree_no_model_microbench_20260522a.jsonl
artifacts/local/ctree_no_model_microbench_20260522b.jsonl
```

Plain read:

```text
The LightZero CTree/list boundary is much slower than a flat vectorized update
in this no-model falsifier. However, raw CTree-list alone still reaches about
1M-1.6M nodes/sec locally, so the current ~5k roots/sec full-loop sidecar wall
cannot be explained by CTree alone. The next 5-10x candidate remains the whole
compact search/replay service shape: keep compact roots, recurrent outputs,
search arrays, target rows, RND, and replay writes resident/array-native, with
stock LightZero objects only as validation or compatibility outputs.
```

H100 Modal rerun with visible JSON output:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_ctree_no_model_benchmark \
  --roots 512,1024 \
  --simulations 16,32 \
  --iterations 20 \
  --warmup 5 \
  --backends ctree-list,ctree-torch-d2h,fake-flat \
  --legal-profiles all3,mixed_2of3 \
  --compute h100

run: ap-RRyZjNk6EmVj2XEWLxZyre
```

Representative H100 rows:

| roots | sims | legal profile | CTree list nodes/sec | CUDA payload nodes/sec | fake-flat nodes/sec |
| ---: | ---: | --- | ---: | ---: | ---: |
| `512` | `16` | all3 | `0.761M` | `0.678M` | `13.86M` |
| `512` | `32` | all3 | `0.701M` | `0.651M` | `14.01M` |
| `1024` | `16` | all3 | `0.468M` | `0.570M` | `17.21M` |
| `1024` | `32` | all3 | `0.579M` | `0.512M` | `17.75M` |
| `1024` | `32` | mixed_2of3 | `0.555M` | `0.527M` | `19.37M` |

Plain read:

```text
The CTree/list API is ugly: fake-flat is roughly 18x-37x faster on these
representative H100 rows.

But the absolute CTree-only rate is still hundreds of thousands of nodes/sec.
The current train-facing/direct sidecar wall is therefore not just one CTree
function. It is the whole model/search/replay boundary: recurrent launch and
output handling, CPU/list backprop payloads, Python control, public output,
and replay/RND object topology.
```

Research synthesis:

```text
gpu_parallel_mcts_research_synthesis_20260522.md
subagent_mctx_gpu_search_research_20260522.md
subagent_gpu_mcts_implementation_patterns_20260522.md
subagent_fast_rl_architecture_patterns_20260522.md
```

## 2026-05-22 GPU/Parallel MCTS Research Follow-Up

Status: docs plus profile-only falsifier. No live Coach runs, checkpoint,
eval, GIF, or tournament state touched.

New sidecar reports:

```text
subagent_gpu_mcts_parallel_architecture_followup_20260522.md
subagent_current_boundary_code_audit_20260522.md
subagent_search_replay_architecture_plan_20260522.md
subagent_precomputed_recurrent_impl_critique_20260522.md
```

Plain synthesis:

```text
Earlier GPU-MCTS research was not enough to choose an architecture. The new
read is consistent: the blocker is not merely "MCTS is not on GPU." The blocker
is the Python/CPU/list/replay boundary around search. MCTX is the clean
accelerator-native reference, MiniZero/KataGo show the practical batched
search/evaluator service pattern, and PufferLib shows the contiguous-buffer
pattern for env/replay ownership.
```

Implemented profile-only falsifier:

```text
direct_ctree_gpu_latent_precomputed_recurrent
```

This keeps the same direct CTree root/search shape, but replaces real recurrent
model calls with resident synthetic reward/value/policy tensors. It is explicit
opt-in and excluded from the default direct CTree comparison preset.

Telemetry cleanup after critique:

- `model_eval_count` remains the logical MuZero search shape.
- `logical_model_eval_count`, `actual_model_eval_count`, and
  `synthetic_recurrent_eval_count` now disambiguate the falsifier.
- recurrent-output `.tolist()` time is now reported separately from D2H.
- mixed-mask local coverage was added so the precomputed path does not only
  pass the all-actions-legal fast path.

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> all checks passed

uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py -k "precomputed_recurrent"
-> 2 passed, 96 deselected
```

Small H100 smoke:

```text
experiment: precomputed-recurrent-h100-falsifier-20260522b
manifest: artifacts/local/curvytron_hybrid_observation_profile_manifests/precomputed-recurrent-h100-falsifier-20260522b/manifest.json
results:  artifacts/local/curvytron_hybrid_observation_profile_results/precomputed-recurrent-h100-falsifier-20260522b/

B64/A8/sim8, 16 measured, 4 warmup:
  direct_ctree_gpu_latent:                         2357.77 roots/sec
  direct_ctree_gpu_latent_precomputed_recurrent:  3745.00 roots/sec
```

Key telemetry from the small row:

| impl | search sec | model total sec | recurrent calls | actual model evals | synthetic recurrent evals | D2H sec | listify sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | `0.02191` | `0.01356` | `8` | `1152` | `0` | `0.000708` | `0.000388` |
| precomputed | `0.003925` | `0.002231` | `0` | `128` | `1024` | `0.000420` | `0.000352` |

Plain read:

```text
Recurrent inference/output handling is a real bucket in this small denominator:
precomputing it improved throughput by about 1.59x. But it did not make the
loop 10x faster, and it does not include replay/RND/learner ownership. The
larger H100 B512/A16/sim16 pair is running to check the stable regime.
```

Big pair queued:

```text
experiment: precomputed-recurrent-h100-falsifier-big-20260522b
manifest: artifacts/local/curvytron_hybrid_observation_profile_manifests/precomputed-recurrent-h100-falsifier-big-20260522b/manifest.json
shape: B512/A16/sim16, 60 measured, 15 warmup
```

Big H100 pair result:

```text
results: artifacts/local/curvytron_hybrid_observation_profile_results/precomputed-recurrent-h100-falsifier-big-20260522b/

B512/A16/sim16, 60 measured, 15 warmup:
  direct_ctree_gpu_latent:                         4920.30 roots/sec
  direct_ctree_gpu_latent_precomputed_recurrent:  6771.37 roots/sec
```

Key telemetry:

| impl | search sec | model total sec | recurrent calls | actual model evals | synthetic recurrent evals | D2H sec | listify sec | CTree traverse | CTree backprop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | `0.07147` | `0.02791` | `16` | `17408` | `0` | `0.001489` | `0.004405` | `0.01084` | `0.01268` |
| precomputed | `0.02533` | `0.004545` | `0` | `1024` | `16384` | `0.000806` | `0.002968` | `0.007983` | `0.008142` |

Plain read:

```text
Precomputing recurrent outputs improved the large row by about 1.38x. This
confirms recurrent inference/output handling is not fake, but it also kills the
idea that this is the whole 10x wall. Even after deleting recurrent calls, the
profile still pays CTree/list traversal/backprop, Python control, root/output
handling, observation setup, and no replay/RND/learner ownership. Next serious
lane: explicit compact search/replay service contract.
```

## 2026-05-22 Compact Search/Replay Service Contract Slice

Status: local/profile-only contract code plus tests. No live Coach runs,
Modal training runs, checkpoint, eval, GIF, or tournament state touched.

Implemented in:

```text
src/curvyzero/training/compact_policy_row_bridge.py
```

New explicit contract objects:

```text
CompactRootBatchV1
CompactSearchResultV1
CompactReplayChunkV1
```

New helpers:

```text
build_compact_root_batch_v1(...)
validate_compact_search_result_v1(...)
build_compact_replay_chunk_v1_from_search_result(...)
```

Plain meaning:

```text
This is the first concrete slice of the MiniZero/KataGo/Puffer-shaped lane:
HybridCompactBatch -> CompactRootBatchV1 -> CompactSearchResultV1 ->
CompactReplayChunkV1 -> target-row parity edge.

It is not a trainer change. It is a fail-closed service contract scaffold that
keeps root/search/replay identity explicit before any deeper search-service
implementation.
```

Focused tests added in:

```text
tests/test_compact_search_replay_contract.py
```

New coverage:

- compact root batch validates shape, row/player identity, active roots, final
  observation requirements, fixed-opponent `to_play=-1`, and binary masks;
- compact search result rejects illegal selected actions and illegal visit mass;
- compact replay chunk rejects search results whose identity sidecars drift
  from the root batch;
- compact service chunk can materialize target rows that exactly match the
  existing object bridge on the two-record final-observation fixture.

Validation:

```text
uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py
-> 4 passed, 2 warnings

uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py tests/test_multiplayer_source_state_target_rows.py -k "compact"
-> 16 passed, 13 deselected, 2 warnings

uv run ruff check src/curvyzero/training/compact_policy_row_bridge.py tests/test_compact_search_replay_contract.py tests/test_multiplayer_source_state_target_rows.py
-> all checks passed
```

Plain read:

```text
The compact service lane now has a typed local contract and identity/legality
tests. The next speed task is not to call this a trainer; it is to wire the
profile-only direct CTree compact output into this service contract and measure
whether skipping public per-env collect output and per-root policy-record
objects can plausibly give a 3x-class boundary win.
```

## 2026-05-22 Compact Search/Replay Boundary Wiring

Status: local/profile-only boundary proof. No live Coach runs, Modal training
runs, checkpoint, eval, GIF, or tournament state touched.

Implemented in:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
tests/test_source_state_batched_observation_boundary_profile.py
```

What changed:

```text
HybridCompactBatch
-> direct CTree compact arrays search
-> CompactRootBatchV1 validation
-> CompactSearchResultV1 validation
```

`run_compact_batch()` now validates the direct search output against the compact
service root/result contract when the probe is using a direct CTree arrays
implementation. The direct search path itself is unchanged. This is a contract
proof and telemetry hook, not a trainer default.

New telemetry:

```text
compact_service_contract_v1_enabled
compact_service_contract_v1_validation_sec
compact_service_contract_v1_contract_id
compact_service_root_batch_schema_id
compact_service_search_result_schema_id
compact_service_root_count
compact_service_active_root_count
compact_service_selected_action_checksum
compact_service_visit_policy_checksum
compact_service_identity_checksum
```

Validation:

```text
uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  src/curvyzero/training/compact_policy_row_bridge.py \
  tests/test_compact_search_replay_contract.py
-> all checks passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "direct_ctree_returns_compact_arrays or precomputed_recurrent" \
  tests/test_compact_search_replay_contract.py
-> 3 passed, 99 deselected
```

Plain read:

```text
The root/result half of the compact service contract is now wired through the
real profile boundary. Full `CompactReplayChunkV1` parity is still separate,
because this profile hook has the current compact batch but not the replay
chunk and record index needed to prove the next-transition target edge.
```

## 2026-05-22 Compact Replay Edge Wiring

Status: local/profile-only proof. No live Coach runs, Modal training runs,
checkpoint, eval, GIF, or tournament state touched.

Implemented in:

```text
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
tests/test_source_state_batched_observation_boundary_profile.py
```

What changed:

```text
HybridCompactBatch + replay chunk + record_index
-> direct CTree compact arrays search
-> CompactRootBatchV1
-> CompactSearchResultV1
-> CompactReplayChunkV1
-> checked target rows
```

New profile-only helper:

```text
_LightZeroCollectForwardStackProbe.run_compact_batch_with_replay_chunk(...)
```

This helper is intentionally not a trainer default. It exists so optimizer
profiles can carry the exact replay chunk and record index needed to validate
the next-transition target edge.

New telemetry:

```text
compact_service_replay_chunk_v1_enabled
compact_service_replay_chunk_v1_validation_sec
compact_service_replay_chunk_schema_id
compact_service_replay_chunk_record_index
compact_service_replay_chunk_next_record_index
compact_service_replay_chunk_target_row_count
compact_service_replay_chunk_action_checksum
compact_service_replay_chunk_reward_checksum
```

Validation:

```text
uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py
-> all checks passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "direct_ctree_compact_output_can_feed_checked_target_rows or direct_ctree_returns_compact_arrays"
-> 2 passed, 96 deselected
```

Plain read:

```text
The compact service contract can now be proven through root, search result, and
replay target rows in one profile-only boundary call. The next useful optimizer
step is a same-denominator profile row that measures whether this closed compact
proof actually removes enough object/replay fanout to justify the bigger
architecture.
```

## 2026-05-22 Valid Replay-Denominator Profile Mode

Status: local/profile-only loop wiring plus tests. No live Coach runs, Modal
training runs, checkpoint, eval, GIF, or tournament state touched.

Implemented in:

```text
src/curvyzero/training/source_state_hybrid_observation_profile.py
src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
scripts/build_curvytron_hybrid_observation_profile_grid.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_curvytron_hybrid_observation_profile_grid_builder.py
```

What changed:

```text
--hybrid-compact-service-replay-proof
```

This flag fixes the most important denominator problem for compact replay
profiling:

```text
record k observation
-> direct compact search chooses actions
-> record k+1 environment step uses those selected actions
-> two-record replay chunk
-> CompactReplayChunkV1 target rows
```

That means the target proof no longer compares search output against an
unrelated random next action. The default profile loop is unchanged.

Current limitation:

```text
This is still a proof/measurement mode, not a speed win by itself. It validates
and times the compact replay edge after the direct search path has already run.
The real 3x-class falsifier still needs a Modal row comparing current best
direct CTree against this closed compact proof on the same denominator, with
validation overhead reported separately.
```

Validation:

```text
uv run ruff check \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  src/curvyzero/training/compact_policy_row_bridge.py \
  scripts/build_curvytron_hybrid_observation_profile_grid.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_compact_search_replay_contract.py \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py
-> all checks passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "compact_service_replay_proof or direct_ctree_compact_output_can_feed_checked_target_rows or direct_ctree_returns_compact_arrays or compact_batch_can_feed_rnd or precomputed_recurrent" \
  tests/test_compact_search_replay_contract.py
-> 6 passed, 118 deselected
```

## 2026-05-22 Compact Replay Proof Falsifier

Status: profile-only. No live Coach training, checkpoint, eval, GIF, or
tournament state was touched.

What we measured:

```text
direct_ctree_gpu_latent search
-> search-selected actions drive next env step
-> compact replay proof
```

First attempt used the old materialized target-row validation edge. That was
the wrong hot-path shape:

| row | sim | steps/sec | compact proof sec | roots |
| --- | ---: | ---: | ---: | ---: |
| baseline direct | 8 | `5634.16` | `0.000` | `61440` |
| materialized proof | 8 | `987.31` | `52.075` | `61440` |
| baseline direct | 16 | `4814.66` | `0.000` | `61440` |
| materialized proof | 16 | `925.02` | `53.596` | `61440` |

Plain read:

```text
The collapse was not compact search. It was copying/materializing full
observation and next_observation target tensors inside the proof on every
collect step.
```

Fix:

```text
CompactReplayIndexRowsV1
```

This records index rows and search arrays only:

```text
record_index, next_record_index, compact_root_row, policy_env_id, policy_row,
env_row, player, action, action_mask, visit policy, root_value, reward,
final_reward, done/terminated/truncated, next_final_observation_row, to_play
```

It deliberately sets:

```text
observation_materialized=false
next_observation_materialized=false
```

and leaves full learner tensors for the sampler/validation edge.

Retest:

| row | sim | steps/sec | compact proof sec | roots |
| --- | ---: | ---: | ---: | ---: |
| baseline direct | 8 | `5634.16` | `0.000` | `61440` |
| index-row proof | 8 | `6193.25` | `0.181` | `61440` |
| baseline direct | 16 | `4814.66` | `0.000` | `61440` |
| index-row proof | 16 | `4797.46` | `0.193` | `61440` |

Tiny L4 smoke also passed and reported:

```text
compact_service_replay_proof_mode=index_rows_v1
compact_service_replay_chunk_schema_id=curvyzero_compact_replay_index_rows/v1
```

Validation:

```text
uv run ruff check \
  src/curvyzero/training/compact_policy_row_bridge.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  scripts/build_curvytron_hybrid_observation_profile_grid.py \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_hybrid_observation_profile.py
-> all checks passed

uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_hybrid_observation_profile.py \
  -k "compact_replay_index_rows or compact_service_replay_proof"
-> 3 passed, 25 deselected

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "direct_ctree_compact_output_can_feed_checked_target_rows or direct_ctree_returns_compact_arrays"
-> 2 passed, 96 deselected
```

Guardrails added:

```text
compact_service_replay_proof requires warmup_steps >= 1
compact_service_replay_proof rejects resident_torch_reuse stale-input mode
```

Decision:

```text
Index-only compact replay fixes the replay-proof object-copy wall. It does not
by itself create a 3x search speedup. The next big lane is still compact batch
ownership through a real search service or array-native CTree, not more proof
materialization.
```

## 2026-05-22 Vendored Flat-A3 CTree Spike

Status: profile-only. No live Coach training, checkpoint, eval, GIF, or
tournament state was touched.

Question:

```text
Can we remove the Python nested-list backprop payload from LightZero CTree
without changing tree semantics?
```

Implemented:

```text
src/curvyzero/vendor/lightzero_ctree_a3/
scripts/build_lightzero_ctree_a3.py
scripts/benchmark_lightzero_ctree_no_model.py --backends ctree-flat-a3
```

First bug:

```text
Exact visit parity against an independent stock CTree run failed even after the
flat path called the original C++ expand(...). The reason was not a flat-policy
math bug. CTree calls get_time_and_set_rand_seed() inside batch_traverse() and
randomly breaks near-ties, so two separate searches can assign the same visit
mass to different actions.
```

Fix:

```text
Add set_deterministic_tie_breaking(True) to the vendored module and use it only
inside the benchmark parity check. Default benchmark/runtime behavior remains
stock-like. The conservative vector-per-row flat path passed first; the final
local path uses the fixed-A=3 expand_a3(...) overload and still passes parity.
```

Local parity:

```text
uv run python scripts/benchmark_lightzero_ctree_no_model.py \
  --roots 64 --simulations 1,2,4,8 --iterations 2 --warmup 0 \
  --backends ctree-flat-a3 --legal-profiles all3,mixed_2of3 \
  --root-noise zero --root-noise-weight 0.0 --flat-a3-parity-check

Result: exact deterministic vendored-list vs flat-A3 visit/value parity in all
rows.
```

Local no-model speed gate:

```text
uv run python scripts/benchmark_lightzero_ctree_no_model.py \
  --roots 1024 --simulations 16 --iterations 100 --warmup 10 \
  --backends ctree-list,ctree-flat-a3 --legal-profiles all3,mixed_2of3 \
  --root-noise zero --root-noise-weight 0.0

all3:
  ctree-list    1.01M nodes/sec
  ctree-flat-a3 2.03M nodes/sec

mixed_2of3:
  ctree-list    1.10M nodes/sec
  ctree-flat-a3 1.92M nodes/sec
```

Plain read:

```text
This is a real small boundary win. It is still not the big architecture shift:
root prepare, traverse, output extraction, and Python simulation control are
still stock/list/object-shaped.
```

Validation:

```text
uv run --with Cython --with numpy python \
  scripts/build_lightzero_ctree_a3.py build_ext --inplace

uv run ruff check \
  scripts/benchmark_lightzero_ctree_no_model.py \
  scripts/build_lightzero_ctree_a3.py \
  src/curvyzero/infra/modal/lightzero_ctree_no_model_benchmark.py
-> all checks passed
```

H100 gate running:

```text
First H100 attempt ap-qkuots2GcuITpTroeFzl3x failed before measurement because
Modal imported packaged /root/curvyzero instead of the built /repo/src vendored
extension.

Fixed by prepending /repo/src/curvyzero to curvyzero.__path__ inside:
  src/curvyzero/infra/modal/lightzero_ctree_no_model_benchmark.py

Conservative vector-per-row H100 gate ap-ZaRkAcT7smnhIr410LweJ4 passed:
  all3:       600.6k -> 953.3k nodes/sec (~1.59x)
  mixed_2of3: 576.1k -> 888.3k nodes/sec (~1.54x)

Final expand_a3 H100 gate running:
  modal run ap-rQtLiZTWYGQi16v2rrf4Wm

Final expand_a3 H100 gate passed:
  all3:       546.7k -> 922.1k nodes/sec (~1.69x)
  mixed_2of3: 517.3k -> 858.1k nodes/sec (~1.66x)

Both flat rows passed deterministic vendored-list parity:
  exact_visit_match=true
  max_visit_abs_diff=0
  max_value_abs_diff=0

roots=1024, simulations=16, iterations=100, warmup=10,
backends=ctree-list,ctree-flat-a3, legal_profiles=all3,mixed_2of3,
flat_a3_parity_check=true.
```

Decision:

```text
Promote flat-A3 only to the next train-facing profile candidate. It is not yet
Coach launch advice. The next proof is a matched full-loop profile with the
same direct_ctree_gpu_latent hook but flat-A3 backprop enabled.
```

2026-05-22 train-facing wiring update:

```text
Implemented profile-only flag:
  --collect-search-ctree-backend flat_a3

It is valid only with:
  --collect-search-backend direct_ctree_gpu_latent

The normal value remains:
  collect_search_ctree_backend=lightzero
```

Safety/observability changes:

```text
- Stock/live default still routes to LightZero CTree.
- Flat-A3 uses isolated CPU40 optimizer Modal images so normal stock launches
  do not depend on the vendored Cython build.
- Compact output now has search_backend_proof:
    observed_collect_search_backends
    observed_collect_search_ctree_backends
    flat_payload_timer_present
- The summarizer rejects flat-A3 rows that only echo the command without
  proving the profiler actually observed flat-A3.
- The profile grid builder now emits both search backend flags, so generated
  manifests cannot silently stay stock when the intent is flat-A3.
```

Validation so far:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_curvytron_profile_grid_builder.py \
  tests/test_summarize_curvytron_optimizer_profile_results.py \
  tests/test_lightzero_phase_profiler.py
-> 27 passed

uv run ruff check --select E9,F821,F822,F823 ...
-> all checks passed

Tiny train-facing Modal smoke is running:
  opt-flat-a3-smoke-20260522a / smoke-flat-a3-sim2-c64-steps128
```

Smoke result:

```text
opt-flat-a3-smoke-20260522a failed before search:
  ImportError loaded curvyzero.vendor... from /root instead of the built /repo/src
  extension path.

Fix:
  _import_collect_search_tree_muzero(flat_a3) now prepends /repo/src and the
  nested repo package paths before importing the vendored extension.

opt-flat-a3-smoke-20260522b passed:
  called_train_muzero=true
  learner_train_calls=1
  collect_search_backend_direct_ctree_gpu_latent_calls=256
  collect_search_backend_fallback_calls=0
  collect_search_backend_output_rows=16384
  search_backend_proof.observed_collect_search_ctree_backends=["flat_a3"]
  steps_per_sec=541.20

This proves the train-facing hook runs. It does not prove a speedup.
Matched C64/sim16/3-learner A/B is running under opt-flat-a3-ab-20260522a.
```

Matched A/B result:

```text
opt-flat-a3-ab-20260522a
H100, curvyzero_batched_profile, C64, batch64, sim16, 3 learner calls,
no RND, no eval/GIF, no checkpoint commit, profile no-death.

direct LightZero CTree:
  attempt: direct-lzctree-c64-sim16-l3
  steps/sec: 516.55
  train_muzero_wall: 63.436s
  mcts_search: 22.638s
  ctree_traverse: 0.422s
  ctree_backpropagate: 0.567s
  model_output_listify: 0.145s

flat-A3 CTree:
  attempt: flat-a3-c64-sim16-l3
  steps/sec: 509.69
  train_muzero_wall: 64.289s
  mcts_search: 22.604s
  ctree_traverse: 0.460s
  ctree_backpropagate: 0.463s
  flat_payload: 0.114s
```

Read:

```text
Flat-A3 is valid, but it did not win the matched full-loop denominator. The
small backprop/list win is swallowed by root prepare, recurrent inference,
model-output D2H, env/render/stack, and learner overhead. This confirms the
current Amdahl read: a CTree backprop payload shave is not the 10x lane.
```

## 2026-05-22 Fast Falsifier Reset

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was changed.

Why this exists:

```text
The previous wave proved flat-A3 was a valid micro-optimization but not a
full-loop speedup. The next wave is designed to falsify bigger assumptions
quickly before building more machinery.
```

Local controls:

```text
uv run python scripts/profile_hybrid_batched_observation_manager.py \
  --batch-size 512 --actor-count 16 --steps 100 --warmup-steps 20 \
  --observation-mode zero --stack-storage-dtype uint8 \
  --no-materialize-scalar-timestep \
  --native-vector-boundary-probe --native-actor-buffer --metrics-only
```

Result:

```text
native vector zero-observation control:
  steps_per_sec: 65424.55
  measured_sec: 1.565s
  actor_step_wall_sec: 0.918s
  batched_stack_probe_sec: 0.465s
```

Second local control:

```text
uv run python scripts/profile_hybrid_batched_observation_manager.py \
  --batch-size 512 --actor-count 16 --steps 100 --warmup-steps 20 \
  --observation-mode zero --stack-storage-dtype uint8 \
  --no-materialize-scalar-timestep \
  --closed-compact-consumer-probe --closed-compact-target-mode arrays \
  --native-actor-buffer --metrics-only
```

Result:

```text
closed compact mock consumer:
  steps_per_sec: 57581.96
  measured_sec: 1.778s
  actor_step_wall_sec: 0.903s
  batched_stack_probe_sec: 0.673s
  closed_compact_consumer_total_sec_per_call: about 0.0069s for 1024 roots
```

Plain read:

```text
The compact sidecar itself is not the obvious wall. It is cheap enough to use
as the next real-search/replay falsifier. The next question is whether real
direct CTree search plus CompactReplayIndexRowsV1 stays fast when selected
search actions drive the next env step.
```

Focused validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "compact_replay_index_rows or compact_service_replay or closed_compact"

-> 3 passed, 123 deselected
```

Active sidecar rows:

```text
opt-compact-replay-proof-20260522b:
  H100, B128/A16, sim16, compact_service_replay_proof=true

opt-mock-search-ceiling-20260522b:
  H100, B512/A16, sim16, mock_search_service + public-output edge

opt-precomputed-recurrent-20260522b:
  H100, B512/A16, sim16, stock_facade vs direct_ctree_gpu_latent vs
  direct_ctree_gpu_latent_precomputed_recurrent
```

Decision rules:

```text
- If compact replay proof cannot beat current direct by a 2-3x class margin
  after scalar output is truly zero, stop calling replay ownership the next big
  win and focus on search-service internals.
- If mock search service is not at least about 2x over current direct in the
  same shape, stop chasing search-service rewrite as the next priority.
- If precomputed recurrent only gives a small win, recurrent inference is not
  the sole wall; the broader CPU/list/control topology remains the target.
```

Sidecar results:

```text
opt-precomputed-recurrent-20260522b, H100 B512/A16/sim16:
  stock facade:        1897.44 roots/sec
  direct gpu-latent:   4651.06 roots/sec
  precomputed recurrent: 5376.34 roots/sec

opt-mock-search-ceiling-20260522b, H100 B512/A16/sim16 with public-output edge:
  mock search service: 8839.68 roots/sec

opt-mock-search-ceiling-nopublic-20260522a, same but compact output only:
  mock search service: 9109.54 roots/sec

opt-compact-replay-proof-b512-20260522a, H100 B512/A16/sim16:
  direct gpu-latent + CompactReplayIndexRowsV1 proof: 6222.32 roots/sec
  compact replay proof wall: 0.103s over 61440 target rows
  compact replay proof cost: about 1.68 us per target row
  public output bytes: 0
```

Plain read:

```text
Compact replay index rows are not the wall. The B512 proof made selected
search actions drive the next env step, wrote index rows, and kept public
LightZero output at zero bytes. The replay proof itself was about 1% of measured
wall.

Precomputed recurrent improved direct from 4651 to 5376 roots/sec, only about
1.16x. Recurrent inference matters, but deleting it is not the whole answer.

Mock search service reached about 8.8k-9.1k roots/sec, roughly 1.95x over the
fresh direct row and about 1.46x over the compact-replay direct row. That is
real headroom, but still not a 10x wall by itself.
```

Updated decision:

```text
Do not build more replay/index-row machinery right now; it is already cheap.
The next real target is the search-service boundary itself: root prep,
per-simulation CPU/list CTree/control, recurrent-output handling, and the
actor/env/observation scheduling around that service.
```

### Sim-Scaling Follow-Up

Status: profile-only. H100 B512/A16, 40 measured steps, 10 warmup steps.
These rows were intentionally launched in parallel to move quickly, so treat
single-row surprises as noisy until repeated.

| row | sim8 | sim16 | sim32 |
| --- | ---: | ---: | ---: |
| direct gpu-latent | `6369.06` | `7884.87` | `3445.25` |
| precomputed recurrent | `11096.16` | `5898.60` | `4153.05` |
| mock search service | `10459.45` | `9537.17` | `9459.32` |
| direct + compact replay proof | `6720.26` | `4765.38` | `3711.15` |

Plain read:

```text
Mock search service stays roughly flat because it is not doing real MCTS.
Direct rows fall hard at sim32. Compact replay proof remains cheap:
about 0.09-0.11s total over 40960 index rows in these rows.

The sim16 direct row in the first simscale run is suspiciously high versus
both the earlier repeat and the compact-replay row. A sequential sim16
direct-vs-precomputed repeat is running before we use that row for decisions.
```

Sequential sim16 repeat:

```text
opt-precomputed-recurrent-repeat-20260522c, H100 B512/A16/sim16:
  direct gpu-latent:       4689.02 roots/sec
  precomputed recurrent:   5880.45 roots/sec
  speedup:                 about 1.25x
```

Corrected read:

```text
The anomalous parallel sim16 direct row was noise. Recurrent output/model work
is a real slice, but even deleting it only gives about 1.2-1.3x on this
denominator. The larger wall remains the search-service topology: CPU/list
CTree/control, root prep, output handling, and surrounding actor/observation
scheduling.
```

### Fresh Search-Boundary Wave, 2026-05-22d

Status: running. Profile-only, no live Coach training touched.

Local gate before launch:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k 'dense_torch_mcts or array_ceiling'

11 passed, 87 deselected
```

Important correction:

```text
The current dense_torch_mcts code is newer than the old audit note. It already
skips unused pre-dense masked policy decoding, rejects fractional masks in the
array-ceiling probe, normalizes root noise over legal actions, and backs up
reward + discount * bootstrap into edge_value_sum.
```

Fresh H100 denominator:

```text
B512, A16, 80 measured steps, 20 warmup steps, sim16 and sim32,
direct_gray64 persistent policy renderer, no scalar materialization,
profile-only hybrid observation canary.
```

Launched manifests:

```text
opt-search-boundary-fresh-direct-20260522d:
  direct_ctree_gpu_latent
  direct_ctree_gpu_latent_precomputed_recurrent

opt-search-boundary-fresh-dense-20260522d:
  dense_torch_mcts

opt-search-boundary-fresh-compile-20260522d:
  dense_torch_mcts_compile_spike

opt-search-boundary-fresh-mock-20260522d:
  mock_search_service
```

Decision rule:

```text
If dense_torch_mcts or compile_spike cannot clearly beat direct_ctree_gpu_latent
at sim16 and sim32 after warmup, stop polishing eager dense Torch. The next
implementation must change the search-service topology: fewer Python/list
CTree boundaries, fewer per-simulation host transfers, or a compiled/fused
batched search body.
```

Fresh results from this wave:

```text
H100 B512/A16, 80 measured, 20 warmup:

direct_ctree_gpu_latent:
  sim16 5007 roots/sec
  sim32 3447 roots/sec

direct_ctree_gpu_latent_precomputed_recurrent:
  sim16 5759 roots/sec
  sim32 4559 roots/sec

dense_torch_mcts:
  sim16 5700 roots/sec
  sim32 2931 roots/sec

dense_torch_mcts_compile_spike:
  sim16 4574 roots/sec
  sim32 2287 roots/sec

mock_search_service:
  sim16 8935 roots/sec
  sim32 11363 roots/sec
```

Plain read:

```text
Dense Torch is not the next main lane. It can edge direct at sim16, but it
loses badly at sim32, and compile_spike is worse on both fresh rows.
Precomputed recurrent is useful signal, but it is still only a 1.15-1.32x
slice. The only clear remaining headroom is the compact/search-service
topology shown by mock_search_service.
```

### Closed Compact Service Falsifier, 2026-05-22

Status: implemented and rerunning on current code.

What changed:

```text
service_tax_probe is now an array-ceiling mode. It pays real input packing,
real initial inference, real recurrent inference calls, fake compact search
updates, final compact array readback, and optional compact replay proof.

The compact replay proof now accepts either direct CTree arrays or explicit
array-ceiling compact search arrays. The arrays carry search_impl, source,
requested_simulations, and actual_search_simulations so mock rows cannot be
mistaken for real MCTS rows.

Warmup-seeded compact proofs are now reported separately. A measured proof row
must come from a measured search result; the first measured env step can still
be driven by a warmup search action, but that proof is not counted as a
measured replay proof.
```

Focused local validation:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k 'service_tax or mock_search_service or array_ceiling'
=> 14 passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  -k 'compact_service_replay_proof'
=> 4 passed

uv run pytest -q -p no:cacheprovider \
  tests/test_curvytron_hybrid_observation_profile_grid_builder.py \
  -k 'compact_service_replay_proof or compact_replay or array_ceiling or service_tax or mock_search_service'
=> 7 passed
```

Current rerun manifests:

```text
opt-closed-service-direct-20260522b:
  direct_ctree_gpu_latent + compact_service_replay_proof

opt-closed-service-mock-20260522b:
  mock_search_service + compact_service_replay_proof

opt-closed-service-tax-20260522b:
  service_tax_probe + compact_service_replay_proof

All are H100 B512/A16, sim16/sim32, 80 measured, 20 warmup, direct_gray64,
scalar materialization off, profile-only, calls_train_muzero=false.
```

Decision rule:

```text
Use aggregate compact.timings, not the runner's printed last-step probe fields,
for Amdahl math. Continue this lane only if the closed compact/service shape
beats direct by a clear 2x+ class margin after paying replay proof and real
model/recurrent tax. If service_tax lands near direct, the next leap needs a
real fixed-shape search service or MCTX/JAX scratch path, not more wrapper
polish.
```

Rerun results from aggregate `compact.timings`:

```text
H100 B512/A16, 80 measured, 20 warmup, compact replay proof on:

sim16:
  direct_ctree_gpu_latent  5215 roots/sec
  mock_search_service      9767 roots/sec  (1.87x direct)
  service_tax_probe       10926 roots/sec  (2.10x direct)

sim32:
  direct_ctree_gpu_latent  4360 roots/sec
  mock_search_service      7950 roots/sec  (1.82x direct)
  service_tax_probe        5637 roots/sec  (1.29x direct)
```

Amdahl split from aggregate timing:

```text
sim16 direct:
  probe/search boundary is 53.4% of measured wall.
  compact replay proof is 1.5% of measured wall.

sim32 direct:
  probe/search boundary is 64.9% of measured wall.
  compact replay proof is 0.8% of measured wall.

sim16 service_tax:
  service-tax probe is 35.8% of measured wall.
  model/recurrent is 61.3% of that probe.

sim32 service_tax:
  service-tax probe is 40.5% of measured wall.
  model/recurrent is 60.4% of that probe.
```

Plain read:

```text
The closed compact topology has real headroom, but the same-denominator result
is not yet a 10x lane. Mock says deleting real search/topology is worth about
1.8x in the closed profile. Service-tax says that once real recurrent/model tax
is paid, sim32 falls back to about 1.3x. The next aggressive move must be a real
fixed-shape/device-resident search body or MCTX/JAX scratch path, not another
small wrapper cleanup.
```

Tooling fix after the rerun:

```text
scripts/run_curvytron_hybrid_observation_profile_manifest.py now prints
aggregate timing fields in its compact summary instead of last-step telemetry.
Future rows can use runner stdout for probe_total_sec/probe_roots_per_sec.
```

### MCTX Visual-Root H100 Scratch Gate, 2026-05-22

Status: running. Scratch/profile-only. No live Coach training touched.

Why this exists:

```text
The closed compact service-tax probe says wrapper cleanup is capped around
1.3x-2.1x on the current denominator. The next bigger question is whether a
device-resident JAX/MCTX search shape can beat the current LightZero CTree
boundary by a large enough margin to justify a real architecture change.
```

Benchmark shape:

```text
curvytron_visual_root:
  synthetic [B,2,4,64,64] uint8 policy-observation stack
  -> [B*2,4,64,64] roots
  -> tiny JAX visual encoder
  -> mctx.gumbel_muzero_policy
  -> action and action_weights readback
```

Important caveat:

```text
This is not a trainer, not real CurvyTron rollout, not replay, and not a
learning-quality claim. It is a search-architecture denominator test.
```

Code guard added before the rerun:

```text
src/curvyzero/infra/modal/mctx_synthetic_benchmark.py now reports whether
selected actions respect the legal mask and includes a simple
search + steady H2D + policy-output D2H timing currency.
```

Live H100 rows:

```text
B64/P2/sim16/max_depth16/mixed_2of3:
  https://modal.com/apps/modal-labs/shankha-dev/ap-FtUsnTbmtoMsc9iIgQf2P5

B512/P2/sim16/max_depth16/mixed_2of3:
  https://modal.com/apps/modal-labs/shankha-dev/ap-yRGfU7HmVX7wA0TB58KSuT

B512/P2/sim32/max_depth32/mixed_2of3:
  https://modal.com/apps/modal-labs/shankha-dev/ap-MIYMal25nsqL1vQnrSvi4h

B1024/P2/sim16/max_depth16/mixed_2of3:
  https://modal.com/apps/modal-labs/shankha-dev/ap-u7tUi8FTnt8CKujgD7Kz8Z
```

First results:

```text
All rows: H100, mixed_2of3 legal masks, actions_legal=true,
action_weights normalized, no reported problems.

Reference current direct_ctree_gpu_latent denominator:
  sim16 closed direct loop:   5,215 roots/sec
  sim16 direct probe/search:  9,776 roots/sec
  sim32 closed direct loop:   4,360 roots/sec
  sim32 direct probe/search:  6,721 roots/sec

MCTX visual-root, hidden64/visual8:

  B64 / roots128 / sim16:
    search only:                    20,345 roots/sec
    search + steady H2D + policy D2H:18,701 roots/sec
    read: small batches do not fully amortize the setup.

  B512 / roots1024 / sim16:
    search only:                    151,059 roots/sec
    search + steady H2D + policy D2H:122,219 roots/sec
    speedup versus direct loop:      23.4x
    speedup versus direct probe:     12.5x

  B512 / roots1024 / sim32:
    search only:                    57,240 roots/sec
    search + steady H2D + policy D2H:50,388 roots/sec
    speedup versus direct loop:      11.6x
    speedup versus direct probe:     7.5x

  B1024 / roots2048 / sim16:
    search only:                    287,488 roots/sec
    search + steady H2D + policy D2H:192,932 roots/sec
    speedup versus direct loop:      37.0x
    speedup versus direct probe:     19.7x
```

Plain read:

```text
This does not prove a trainer migration, but it kills the idea that GPU search
cannot produce a large speed multiplier. Once enough roots are batched, the
MCTX/JAX device-resident shape is far above the current LightZero direct CTree
profile denominator.

The immediate risk is that this benchmark has a tiny synthetic visual encoder
and no real CurvyTron rollout/replay/learner. The next stress rows increase
hidden_dim and visual_channels to see whether the margin survives a heavier
toy network.
```

Second live stress wave:

```text
B512/P2/sim16/hidden128/visual16:
  https://modal.com/apps/modal-labs/shankha-dev/ap-jZefg1p2c0kCjHG3e2mytz

B512/P2/sim32/hidden128/visual16:
  https://modal.com/apps/modal-labs/shankha-dev/ap-yhTEhJumtZoL5FC6qfslta

B512/P2/sim16/hidden256/visual32:
  https://modal.com/apps/modal-labs/shankha-dev/ap-p7AMjcz8F2L2HXZdW1zIc8

B1024/P2/sim32/hidden64/visual8:
  https://modal.com/apps/modal-labs/shankha-dev/ap-1qeylI9cLY7HTC3vuFwpz6
```

Second-wave results:

```text
All rows: actions_legal=true, mixed_2of3 masks, no reported problems.

B512/P2/sim16/hidden128/visual16:
  search only:                     160,452 roots/sec
  search + steady H2D + policy D2H:114,455 roots/sec

B512/P2/sim32/hidden128/visual16:
  search only:                     65,367 roots/sec
  search + steady H2D + policy D2H:52,374 roots/sec

B512/P2/sim16/hidden256/visual32:
  search only:                     169,307 roots/sec
  search + steady H2D + policy D2H:104,047 roots/sec

B1024/P2/sim32/hidden64/visual8:
  search only:                     104,837 roots/sec
  search + steady H2D + policy D2H:89,190 roots/sec
```

Plain read:

```text
The MCTX/JAX scratch margin survives heavier toy hidden/visual sizes. This is
still not training proof, but it is enough to make the next architecture spike
larger than wrapper cleanup: bring MCTX or an equivalent device-resident search
service closer to the real compact CurvyTron boundary.
```

Guard patch after implementation-scout critique:

```text
The benchmark now reports:
  - active_root_mask accounting;
  - selected-action legality;
  - illegal action-weight mass;
  - a true fresh end-to-end boundary loop:
    device_put(obs/mask) -> run_search -> read action/action_weights;
  - compact-search-result-shaped samples.

Local validation:
  ruff on benchmark + legality test: passed
  tests/test_mctx_synthetic_benchmark_legality.py: 2 passed
  py_compile benchmark: passed
```

Guarded rerun rows:

```text
B64/P2/sim16/hidden64/visual8:
  https://modal.com/apps/modal-labs/shankha-dev/ap-Sd6SpjJzOkLEMn9Suv2f4Y

B512/P2/sim16/hidden64/visual8:
  https://modal.com/apps/modal-labs/shankha-dev/ap-TMG6lqkefHkWxHPQ8puLCR

B512/P2/sim32/hidden64/visual8:
  https://modal.com/apps/modal-labs/shankha-dev/ap-YGHCOcqPx57ZtnBpxhg7uT

B512/P2/sim32/hidden128/visual16:
  https://modal.com/apps/modal-labs/shankha-dev/ap-tgTrAiWyJnnP3M7JyFSaZ0
```

Decision rule:

```text
Keep the MCTX/JAX lane only if the end-to-end-ish
search + H2D + policy-output D2H roots/sec beats direct_ctree_gpu_latent by a
clear class margin:

sim16: target >= 2x direct loop roots/sec, preferably >= 3x search-only.
sim32: target >= 1.5x direct loop roots/sec, preferably >= 2x search-only.

If it lands in the same 1.3x-1.8x band, stop treating MCTX scratch as the next
quick win and move the bigger architecture question to native/vector actor
buffers plus compact replay/search-service ownership.
```

Guarded rerun results:

```text
All guarded rows:
  ok=true
  actions_legal=true
  illegal_selected_action_count=0
  illegal_action_weight_mass_max=0
  action_weights_normalized=true
  compact_search_result_sample includes selected_action, visit_policy,
  root_value, and root_value_source=summary.value.
```

Primary currency:

```text
end_to_end_active_decisions_per_sec_median

This includes a fresh boundary loop:
device_put(obs/mask) -> run_search -> read action/action_weights.
It is stricter than the older stitched search + steady H2D + policy-D2H number.
```

| row | resident search roots/sec | fresh-boundary roots/sec | direct-loop comparison | direct-probe comparison |
| --- | ---: | ---: | ---: | ---: |
| B64/P2/sim16/h64/v8 | `19,379` | `15,781` | `3.0x` vs `5,215` | `1.6x` vs `9,776` |
| B512/P2/sim16/h64/v8 | `150,768` | `88,452` | `17.0x` vs `5,215` | `9.1x` vs `9,776` |
| B512/P2/sim32/h64/v8 | `62,762` | `45,514` | `10.4x` vs `4,360` | `6.8x` vs `6,721` |
| B512/P2/sim32/h128/v16 | `61,035` | `47,506` | `10.9x` vs `4,360` | `7.1x` vs `6,721` |

Plain read:

```text
The guarded rerun passes the decision rule. The small B64 row is only a modest
win because it does not batch enough roots. The B512 rows are the useful signal:
even after fresh H2D and policy-output readback, MCTX/JAX is a different class
from the current LightZero CTree boundary.

This does not prove a trainer migration and does not prove learning quality.
It does prove that the next aggressive optimizer lane should be a real compact
CurvyTron boundary / device-resident search spike, not more wrapper cleanup.
```

### Real Compact Visual Roots Into MCTX, B512 H100 Gate

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

What changed from the previous scratch rows:

```text
This no longer uses synthetic visual roots.

HybridBatchedObservationProfileManager
-> real renderer-backed HybridCompactBatch [B,2,4,64,64]
-> CompactRootBatchV1
-> MCTX/JAX Gumbel MuZero search
-> CompactSearchResultV1
-> selected actions step the compact env once
-> CompactReplayIndexRowsV1
```

Focused validation before the H100 rows:

```text
uv run ruff check \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  src/curvyzero/env/vector_trainer_observation.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py \
  tests/test_vector_trainer_observation.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py \
  tests/test_vector_trainer_observation.py
=> 25 passed, 2 warnings
```

Rows:

| row | run | ok | resident roots/sec | fresh-boundary roots/sec | replay-index timing | next render | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| B512/P2/sim16/h64/v8 | `ap-5r4WIcSSF6Lre1pgcBsvYA` | true | `167,516` | `124,090` | `0.234s` | `0.039s` | legal actions, normalized visits, `1024` compact replay index rows |
| B512/P2/sim32/h64/v8 | `ap-p3ayFUnB91xgPpeaQEWUxA` | true | `65,228` | `51,454` | `0.260s` | `0.042s` | legal actions, normalized visits, `1024` compact replay index rows |

Important timing detail:

```text
The primary metric is end_to_end_active_decisions_per_sec_median.
That includes device_put(obs/mask), search, and action/action_weights readback.
It does not include the one-time 3-4s compact observation setup, and it is not
a full trainer wall-clock claim.
```

Comparison against the current direct CTree denominator:

```text
direct_ctree_gpu_latent closed-loop reference:
  sim16: about 5,215 roots/sec
  sim32: about 4,360 roots/sec

real compact visual MCTX:
  sim16: 124,090 fresh-boundary roots/sec, about 23.8x the direct loop
  sim32:  51,454 fresh-boundary roots/sec, about 11.8x the direct loop
```

Plain read:

```text
This is the first current optimizer gate that uses real compact visual
CurvyTron roots and still shows 10x-class search-boundary headroom.

It is still not trainer advice. The search model is a toy JAX visual/recurrent
model, not the actual current PyTorch LightZero model. The next missing proof is
current-model realism plus learner/RND/replay integration without falling back
to scalar LightZero timestep objects.
```

Amdahl correction:

```text
The fresh-boundary roots/sec metric is a search-boundary currency. It includes
fresh device_put(obs/mask), MCTX search, and action/action_weights readback. It
does not include host observation construction, renderer/stack update, or the
next compact env step plus replay-index edge.

The replay-index proof deliberately steps the compact env once after search.
That edge is hundreds of milliseconds at B512/B1024, so once search is this
fast the next wall moves back to env/observation/replay synchronization.
```

Code cleanup after this critique:

```text
mctx_synthetic_benchmark now reports:
  host_setup_plus_fresh_boundary_sec
  host_setup_plus_fresh_boundary_active_decisions_per_sec
  compact_replay_index_timing_sec
  closed_one_step_search_replay_edge_sec
  closed_one_step_search_replay_edge_active_decisions_per_sec

The top-level measurement_claim now says host observation setup and replay edge
are separate fields, so future rows cannot honestly call fresh-boundary roots/sec
a full-loop number.
```

## 2026-05-22 Real Compact MCTX Scale/Pressure Wave

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

All rows:

```text
observation_mode=curvytron_hybrid_compact_visual_sample
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
compute=H100
body_capacity=4096
rollout_steps=4
warmup_runs=3
steady_runs=8
CompactSearchResultV1 and CompactReplayIndexRowsV1 validated
```

Search-boundary currency:

| row | run | ok | fresh-boundary roots/sec | resident roots/sec | replay-index edge | rough search+replay roots/sec |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| B512/P2/sim16/h64/v8 | `ap-5r4WIcSSF6Lre1pgcBsvYA` | true | `124,090` | `167,516` | `0.234s` | `4,227` |
| B512/P2/sim32/h64/v8 | `ap-p3ayFUnB91xgPpeaQEWUxA` | true | `51,454` | `65,228` | `0.260s` | `3,653` |
| B512/P2/sim16/h128/v16 | `ap-sBPbLW6KkV6JJo6DwSUOdy` | true | `105,399` | `170,847` | `0.342s` | `2,910` |
| B512/P2/sim32/h128/v16 | `ap-GPQb3F9TG6U4ThHGWGsdxn` | true | `61,221` | `78,166` | `0.327s` | `2,977` |
| B1024/P2/sim16/h64/v8 | `ap-I8OOgAY5sEqUKPPHEV2rMC` | true | `168,630` | `318,857` | `0.525s` | `3,813` |
| B1024/P2/sim32/h64/v8 | `ap-eMNztlt4EMsOTR4YiErs5T` | true | `96,336` | `124,923` | `0.409s` | `4,759` |

Plain read:

```text
The MCTX/JAX search boundary still has huge headroom on real compact visual
roots, even at B1024 and with a heavier toy visual encoder. But if we include
the next env step and compact replay-index edge, the rough one-step denominator
falls back to a few thousand roots/sec.

That means the correct next target is the closed compact loop:
search -> selected joint actions -> env/observation update -> compact replay/RND
edge -> repeat. The bottleneck to attack after MCTX is now likely actor/env
step, renderer/stack update, replay/RND materialization, or their synchronization
around the compact batch.
```

## 2026-05-22 Closed Compact MCTX Loop Smoke

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Purpose:

```text
Stop measuring only the search boundary. Put repeated env/observation/replay
edges into the denominator:

CompactRootBatchV1
-> MCTX search
-> selected joint action
-> compact env/observation step
-> CompactReplayIndexRowsV1
-> repeat
```

Local validation before launch:

```text
uv run ruff check src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py
=> 7 passed, 2 warnings
```

Rows completed so far:

| row | run | ok | fresh-boundary roots/sec | closed one-step roots/sec | repeated closed-loop roots/sec |
| --- | --- | --- | ---: | ---: | ---: |
| B256/P2/sim16/h64/v8/loop4 | `ap-Il0Ntwl3bnTWBpb9y4mVw0` | true | `64,441` | `2,048` | `3,245` |
| B512/P2/sim16/h64/v8/loop8 | `ap-HCqLIjFwbBYnUxV1LAISA0` | true | `90,032` | `2,841` | `5,056` |
| B512/P2/sim32/h64/v8/loop8 | `ap-1TXVXNzYMabDDnZZJLBQ4C` | true | `57,771` | `3,104` | `4,870` |
| B1024/P2/sim16/h64/v8/loop8 | `ap-4AicewTOsZAa2AYvXUC74F` | true | `176,179` | `5,205` | `6,410` |
| B1024/P2/sim32/h64/v8/loop8 | `ap-GRbdyyEHWdenbxuYYTjT60` | true | `95,697` | `4,076` | `5,033` |

Plain read:

```text
The search boundary remains fast. Once the compact env/observation/replay edge
is repeated, throughput falls back to roughly 3k-5k active roots/sec in these
first H100 rows, and about 5k-6.4k at B1024.

This is exactly why fresh-boundary roots/sec cannot be sold as full-loop speed.
The next optimizer target is not another CTree wrapper shave. It is the closed
compact edge: compact env step, stack/update, replay-index materialization, RND
latest-frame path, and any synchronization around them.

Bigger batches help, but they do not rescue the 50k-200k roots/sec search
number. The next useful implementation question is which part of the closed
compact edge scales linearly with batch size and which part is fixed overhead.
```

### Closed Compact MCTX Bucketed + Native Actor-Buffer Rows

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Purpose:

```text
Measure the repeated closed compact loop with aggregate buckets, then remove
one obvious host-copy layer by letting in-process actors write render-state rows
directly into parent-sized native buffers.
```

Validation before native actor-buffer Modal rows:

```text
uv run ruff check \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py
=> 34 passed, 2 warnings
```

Rows:

| row | run | native actor buffer | closed-loop roots/sec | env step total | env step fraction | search fraction | slowest bucket |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| B512/P2/sim16/h64/v8/loop8 | `ap-x2AEPTSxDMVae8sqnPZD7m` | false | `5,792` | `1.212s` | `85.7%` | `3.6%` | `env_step_sec` |
| B512/P2/sim16/h64/v8/loop8 | `ap-noPmiqnP1fHXK5hKUL8LdN` | true | `6,823` | `0.893s` | `74.4%` | `4.9%` | `env_step_sec` |
| B1024/P2/sim16/h64/v8/loop8 | `ap-Di4YwwJhAubEgZv1HPLpez` | false | `6,253` | `2.406s` | `91.8%` | `2.1%` | `env_step_sec` |
| B1024/P2/sim16/h64/v8/loop8 | `ap-DkeTLrEKmYCerCoDA0H9iL` | true | `8,918` | `1.493s` | `81.3%` | `2.9%` | `env_step_sec` |

Native actor-buffer improvement:

| row | closed-loop speedup | env-step reduction |
| --- | ---: | ---: |
| B512/P2/sim16 | `1.18x` | `1.36x` |
| B1024/P2/sim16 | `1.43x` | `1.61x` |

Native actor-buffer sub-buckets:

| row | actor step wall | observation/stack update | renderer render | stack shift |
| --- | ---: | ---: | ---: | ---: |
| B512/P2/sim16/native | `0.316s` | `0.575s` | `0.516s` | `0.035s` |
| B1024/P2/sim16/native | `0.631s` | `0.860s` | `0.705s` | `0.118s` |

Plain read:

```text
The native actor-buffer patch is a real cleanup and should stay, especially at
B1024. It is not the big architecture win. The repeated closed loop is still
dominated by env_step_sec, and inside that bucket the current wall is
observation/renderer/stack work plus actor stepping.

This changes the next move. Do not polish MCTX search or replay-index rows
until env_step_sec is split and attacked. The obvious next falsifier is a
device-resident or lower-copy observation path: render/update the compact stack
where search consumes it, or bypass the current render-state-to-host-stack loop
in a profile-only path with parity/fidelity checks.
```

### Persistent Compact Live-Prefix Trim + Actor-Count Grid

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
_persistent_compact_state_from_production now derives persistent trail_slots
from the live visual/body cursor prefix instead of defaulting to the full
body_capacity=4096 allocation on every step.
```

Why this mattered:

```text
The previous native actor-buffer rows showed a large hidden bucket inside the
renderer path: renderer_production_to_compact_sec. That was not GPU drawing.
It was mostly host-side production-state -> compact-render-state conversion
over inactive capacity.
```

Validation:

```text
uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
=> passed

uv run python -m py_compile \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_boundary_profile.py \
    -k "persistent_compact_state_trims or persistent_delta_state or persistent_renderer_full_request" \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py
=> 5 passed, 103 deselected, 2 warnings
```

Fresh H100 rows after the trim:

| row | run | closed-loop roots/sec | env fraction | search fraction | production->compact | actor step wall | observation/stack | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B512/P2/sim16/loop16/native | `ap-sn47V3H3xOqDx9huiuYnXE` | `8,196` | `71.9%` | `5.5%` | `0.356s` | `0.762s` | `0.670s` | small win versus pre-trim B512 |
| B1024/P2/sim16/loop16/native | `ap-ujxlzZ7Ou0KD2pSEQZxpPA` | `15,261` | `79.7%` | not copied | `0.370s` | `0.984s` | `0.723s` | strong win versus pre-trim B1024 |
| B1024/P2/sim16/loop32/native | `ap-f3ElgHQ3TKBXoXgZmxM9Gn` | `12,379` | `75.7%` | not copied | `0.763s` | `2.085s` | `1.907s` | longer loop; more env/observation pressure |
| B1024/P2/sim32/loop32/native | `ap-IpM6JT4fQdHIuPQj3uI0JB` | `11,480` | `71.3%` | `10.1%` | `0.767s` | `2.132s` | `1.921s` | deeper search still not the main wall |
| B2048/P2/sim16/loop16/native | `ap-1MWgTXQ20JWncgmhJPzGO2` | `13,547` | `78.6%` | `2.8%` | `0.790s` | `1.913s` | `1.878s` | larger batch does not double throughput |

Actor-count grid, B1024/P2/sim16/loop16/native:

| actor_count | run | closed-loop roots/sec | env fraction | actor step wall | observation/stack |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `ap-A7vve2EqhvBDNVRAvQLdiG` | `16,423` | `78.0%` | `0.805s` | `0.746s` |
| 4 | `ap-wSXVgBIB76bT0UB9XYEgev` | `13,153` | `82.4%` | `1.233s` | `0.815s` |
| 16 | `ap-21sruglBvpzlGJIqH048Oy` | `11,923` | `82.5%` | `1.427s` | `0.834s` |

Plain read:

```text
The live-prefix trim is worth keeping. It removed a real inactive-capacity copy,
especially at B1024.

But the broader Amdahl read did not change: the closed compact loop is still
mostly env_step_sec. Search is small in the matched repeated denominator.
Making more in-process actor shards did not help; actor_count=1 was fastest in
this harness. That means the next aggressive move should not be "more shards
inside the same Python manager." It should be a cleaner state-residency change:
keep compact env/render/stack state in the layout the search consumes, or split
VectorMultiplayerEnv.step enough to remove the next host packaging/copy wall.
```

### Env-Step Timing Split + Render-State Filter

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
VectorMultiplayerEnv.step now accepts optional profile timers.
Hybrid compact actors now propagate actor/env timing leaves through the closed
compact loop.

Persistent GPU profile render-state now copies only the fields consumed by the
GPU renderer instead of copying the whole env state into parent render buffers:
visual trail arrays, head/player arrays, avatar colors, and bonus arrays.
Generic renderers and CPU oracle paths keep the full state.
```

Local validation:

```text
uv run ruff check \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  tests/test_source_state_hybrid_observation_profile.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  -k "native_actor_buffer or timing"
=> 7 passed, 21 deselected, 2 warnings

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_compact_search_replay_contract.py \
  -k "native_actor_buffer or timing or closed_loop or compact"
=> 18 passed, 17 deselected, 2 warnings
```

First timing-split Modal smoke:

| row | run | closed-loop roots/sec | env fraction | search fraction | actor render-state write | observation/stack | renderer render | actual env runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke defaults, B1024/sim16/loop16/native | `ap-DFQo12tUrD86I9NivkFtvL` | `11,795` | `80.4%` | `4.0%` | `1.049s` | `0.979s` | `0.723s` | `0.085s` |

Caveat:

```text
That smoke accidentally used smaller CLI defaults for some model/search knobs
(`body_capacity=4`, `hidden_dim=32`, `max_depth=8`, `rollout_steps=2` in the
top-level config). Treat it as a wiring and timing-shape smoke only, not a
matched speed row.
```

Plain read:

```text
The split corrected the mental model. The env_step_sec bucket is not mostly
physics. In this smoke, actual env runtime was tiny. The hot path was state
handoff for rendering plus observation/stack update. That makes another MCTX
wrapper cleanup the wrong next move.

The immediate falsifier is a matched H100 row after the render-state filter.
If actor_render_state_write_sec collapses, keep this patch and then attack
host stack/readback. If it does not, the next larger move is resident compact
observation ownership: keep the latest policy frame/stack in the GPU layout the
search consumes and materialize host stacks only for sampled validation.
```

Matched H100 row after the render-state filter:

| row | run | closed-loop roots/sec | env fraction | search fraction | actor render-state write | observation/stack | renderer render | production->compact | actual env runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1024/P2/sim16/h64/v16/loop16/native | `ap-FVVPGkA3oKem8yPvnetWdL` | `15,906` | `73.4%` | `5.3%` | `0.368s` | `0.988s` | `0.728s` | `0.517s` | `0.089s` |

Comparison:

```text
This is not a big speed win over the previous best B1024 actor_count=1 row
(`16,423` roots/sec), and only a small/noisy improvement over the earlier
B1024 native row (`15,261` roots/sec). The key filter is still useful as a
measurement cleanup, but it did not change the Amdahl wall.

The corrected target is now more specific:

1. Stop rebuilding compact render state from production state every hot step
   if the persistent renderer can own that state incrementally.
2. Stop mandatory device->host frame readback plus host stack update for the
   profile hot loop.
3. Feed MCTX from a resident device stack and keep host observation only as a
   sampled validation artifact.
```

### Resident Stack and Replay-Index Grid

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
mctx_synthetic_benchmark now has compact JSON summaries for Modal profile
grids and a replay-index on/off toggle for the repeated closed compact loop.
Resident compact visual rows now assert uint8 latest-frame shape and row-major
[env row, player] root ordering before reshaping the resident stack for MCTX.
```

Local validation:

```text
uv run ruff check src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_source_state_hybrid_observation_profile.py
=> passed

uv run python -m py_compile src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  -k "native_actor_buffer or resident or device_only or timing"
=> 10 passed, 21 deselected, 2 warnings
```

Matched H100 rows:

| row | run | roots/sec | env frac | search frac | replay-index frac | root-build frac | h2d frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host stack + replay rows | `ap-CiaaKQLDzcQy3O8E1YwMWl` | `20,686` | `71.2%` | `6.8%` | `0.30%` | `3.9%` | `4.5%` |
| host stack, no replay rows | `ap-kBdK7CDwEXq14rLFz7uVnP` | `16,963` | `62.5%` | `6.4%` | `0.00%` | `2.8%` | `9.4%` |
| resident GPU stack + replay rows | `ap-yTg5CYKgQePjg9tJJ1cegp` | `16,249` | `68.8%` | `5.2%` | `0.35%` | `12.0%` | `0.5%` |
| resident GPU stack, no replay rows | `ap-lBzgD9Hbc6GvbjrlKoMR3K` | `17,904` | `63.8%` | `6.5%` | `0.00%` | `9.1%` | `0.8%` |

Plain read:

```text
Replay-index construction is dead as a primary optimization target. It is tiny.

Resident GPU stack is valid profile plumbing, but it is not a speed win in this
matched row. It removes most observation H2D, but the wall moves to root-build
and production-to-compact/render-state/observation ownership.

The current bottleneck is still env_step_sec. In plain terms that bucket is not
mostly game physics; it is feeding the next search call.
```

Active ceiling test:

```text
Run the same compact loop with refresh_observation_stack=false. This keeps the
compact env/search/replay shape but skips render-state write and observation
refresh. It is intentionally not a training lane; it prices the maximum win
from deleting the current env/observation handoff wall.
```

Ceiling rows:

| row | run | roots/sec | env frac | search frac | root-build frac | read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sim16, refresh on | `ap-CiaaKQLDzcQy3O8E1YwMWl` | `20,686` | `71.2%` | `6.8%` | `3.9%` | matched current row |
| sim16, refresh off | `ap-sE7EJOnI5VOzz5mwdEySlY` | `48,608` | `21.9%` | `15.2%` | `24.4%` | `2.35x` ceiling |
| sim32, refresh on | `ap-PNXUk0s1GukZGGcEIn4vAU` | `17,855` | `66.7%` | `14.2%` | `3.2%` | matched current row |
| sim32, refresh off | `ap-kzQpwi92EPoTeNEMKIcIb3` | `32,133` | `18.5%` | `25.7%` | `23.7%` | `1.80x` ceiling |

Plain read:

```text
The observation/render-state wall is real and large enough to justify a lower-
copy compact-state architecture. It is not a 10x wall by itself. Once removed,
root-batch construction, H2D, and search become visible. The next small patch is
to avoid copying the root observation tensor in profile-only root batches; the
next big architecture is to keep compact state in the layout consumed by search
and replay instead of rebuilding it every decision.
```

### Root Observation View + Fast Visual Compact State

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patches:

```text
CompactRootBatchV1 now has an explicit copy_observation flag. The default still
copies observations; profile hot loops can request a view. Focused tests prove
the default copy and opt-in view behavior.

The persistent GPU renderer now has a visual-trail fast adapter. When the env
already exposes reconstructed visual_trail_* arrays, the renderer no longer
allocates and rebuilds full compact trail arrays every step. The generic
production-state path remains for non-visual states.
```

Validation:

```text
uv run ruff check \
  src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py \
  src/curvyzero/training/compact_policy_row_bridge.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  src/curvyzero/training/source_state_hybrid_observation_profile.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_compact_search_replay_contract.py \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  -k "root_batch_can_keep or resident or device_only or native_actor_buffer or timing or persistent_compact_state_trims or persistent_renderer_full_request"
=> 15 passed, 127 deselected, 2 warnings
```

Fresh H100 sim16 rows:

| row | run | roots/sec | production->compact | observation | root-build | read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| before this patch, host+copy | `ap-CiaaKQLDzcQy3O8E1YwMWl` | `20,686` | `0.374s` | `0.734s` | `0.062s` | old best current row |
| fast visual, copy | `ap-qyxPDJnYnemi47Pk3Hdtol` | `19,800` | `0.054s` | `0.427s` | `0.251s` | production conversion fixed, root copy exposed |
| fast visual, no-copy | `ap-x0NddqBL2HX3sY04cihsWu` | `26,610` | `0.057s` | `0.412s` | `0.009s` | best refresh-on row so far |
| no refresh, no-copy | `ap-RUYZt1HsjD4bq4AhHOunz3` | `63,560` | skipped | near zero | `0.007s` | ceiling row |

Plain read:

```text
The fast visual adapter solved the intended sub-bucket: production-to-compact is
now small. The no-copy root option also solved its sub-bucket. Together they
move the real refresh-on sim16 compact loop from about 20.7k to 26.6k active
roots/sec, about 1.29x.

The remaining wall is now actor render-state write plus observation/stack work,
not root-batch copying and not replay-index rows. The next live question is
whether resident GPU stack now wins after root-copy is gone.
```

### Fresh H100 Mechanics vs Observation Grid

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
mctx_synthetic_benchmark compact stdout now includes a plain_breakdown that
separates top-level wall buckets from diagnostic leaves inside env_step_sec.
This is instrumentation only.
```

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  tests/test_mctx_synthetic_benchmark_legality.py
=> passed

uv run pytest -q -p no:cacheprovider tests/test_mctx_synthetic_benchmark_legality.py
=> 5 passed
```

Fresh H100 rows, B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay
rows:

| row | run | roots/sec | env frac | search frac | game mechanics / env | observation handoff / env | gpu draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host stack, sim16 | `ap-iHy06LxrTZNDIPX0soLPwJ` | `23,109` | `68.1%` | `7.5%` | `9.6%` | `80.0%` | `0.0066s` |
| resident stack, sim16 | `ap-HGSnFYPldyzmgSjQedA9CK` | `30,297` | `68.3%` | `9.6%` | `8.4%` | `79.2%` | `0.0054s` |
| host stack, sim32 | `ap-fD9oGLwDmrnakTogszpz9W` | `19,485` | `63.0%` | `15.2%` | `8.8%` | `79.8%` | `0.0071s` |
| resident stack, sim32 | `ap-CTD65oO51yoHPexUwvA5JQ` | `26,805` | `59.8%` | `19.7%` | `10.8%` | `75.8%` | `0.0057s` |
| refresh off ceiling, sim16 | `ap-aBw0riUkgxyj97vyuhtUVA` | `57,895` | `26.1%` | `18.5%` | `43.1%` | `0.0%` | skipped |

Plain read:

```text
The game mechanics are not the current wall. In refresh-on rows, actual
mechanics are about 8-11% of env_step_sec. The expensive part is feeding the
next search call: actor render-state write plus renderer/observation/stack
handoff.

Resident stack now clearly helps after root-copy was removed:
  sim16: 23.1k -> 30.3k roots/sec, about 1.31x.
  sim32: 19.5k -> 26.8k roots/sec, about 1.38x.

The GPU draw itself is already tiny, about 5-7ms over the measured loop. The
remaining renderer-related cost is not drawing pixels; it is copying/packing
state into the renderer and owning the stack/search input.
```

Next target:

```text
Stop copying full actor render-state rows into parent render buffers every
step. The next aggressive profile-only patch should give the manager/renderer a
compact state ownership path or a delta path so the persistent renderer consumes
the state layout it already needs.
```

Resident retest after root no-copy:

| row | run | roots/sec | env frac | search frac | runtime leaf | render-state write | observation | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| resident, sim16, no-copy | `ap-vKRD0yvuSrOshqPdzwdadv` | `31,611` | `63.7%` | `10.6%` | `0.0757s` | `0.2777s` | `0.2282s` | resident now wins sim16 |
| host, sim32, no-copy | `ap-71Bpg0rqmawTSCtudDHYMQ` | `21,167` | `60.9%` | `16.5%` | `0.0890s` | `0.3511s` | `0.4353s` | host stack pays stack/D2H |
| resident, sim32, no-copy | `ap-glx59a295Henfmy7bLWdeK` | `28,855` | `59.0%` | `22.3%` | `0.0548s` | `0.3351s` | `0.2091s` | resident now wins sim32 |

Plain read:

```text
After root-copy was removed, resident GPU stack became a real profile win:
sim16 improves about 1.19x over the host no-copy row, and sim32 improves about
1.36x over the host no-copy row.

The Amdahl wall is still env_step_sec, but the leaf timers say that is not
mostly CurvyTron physics. Actual actor env runtime is small. The larger leaves
are render-state write, observation/update, delta packing, H2D, and stack
ownership. Search becomes more visible at sim32, but it is still not the first
sim16 bottleneck.
```

Audit doc:

```text
docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/subagent_mechanics_vs_observation_audit_20260522.md
```

### Borrowed Single-Actor Render State

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
HybridObservationProfileConfig now has borrow_single_actor_render_state.
In native_actor_buffer + actor_count=1 + renderer-backed refresh-on profiles,
the manager can pass the actor env.state directly to the persistent renderer
instead of copying visual_trail/player/bonus arrays into parent render buffers.
Terminal rows fail closed because final-observation-before-autoreset needs a
separate snapshot protocol.
```

Validation:

```text
uv run ruff check \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  tests/test_source_state_hybrid_observation_profile.py
=> passed

uv run python -m py_compile \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  -k "borrowed or native_actor_buffer or persistent_device_only"
=> 10 passed, 23 deselected
```

Fresh H100 rows, B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay
rows:

| row | run | roots/sec | env frac | search frac | actor render-state write | observation | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| host stack, sim16, copied | `ap-CkRbOJPyo3qwIUvHv8bVBZ` | `26,603` | `64.1%` | `8.5%` | `0.299s` | `0.599s` | copied parent buffers |
| host stack, sim16, borrowed | `ap-AiZAJBfb8xmiA9BMGrSQbk` | `32,830` | `57.6%` | `10.7%` | `0.000s` | `0.564s` | `1.23x` over host copied |
| resident stack, sim16, copied | `ap-PFzVZsVxVLzBx3nh2u6ja3` | `32,873` | `62.5%` | `10.5%` | `0.305s` | `0.347s` | copied resident baseline |
| resident stack, sim16, borrowed | `ap-Mgvb8AFe3q1HzbiMAepQjL` | `48,579` | `53.8%` | `15.2%` | `0.000s` | `0.303s` | `1.48x` over resident copied |
| resident stack, sim32, copied | `ap-l2qk2v2MoIZm6dhqKmPyIO` | `24,020` | `51.5%` | `20.9%` | `0.238s` | `0.458s` | copied resident sim32 |
| resident stack, sim32, borrowed | `ap-IkSystwnJLV5kDHSFobmZe` | `36,041` | `43.3%` | `28.3%` | `0.000s` | `0.331s` | `1.50x` over resident copied |

Plain read:

```text
The state-ownership hypothesis passed. Removing the parent render-state copy is
a real total-wall win, not just a timer shuffle. The best refresh-on row moved
from the previous resident no-copy 31.6k-32.9k band to 48.6k roots/sec.

The current wall is now narrower. In the best sim16 resident borrowed row,
actor_render_state_write is gone, GPU draw is still tiny, and the remaining
hot leaves are observation/renderer delta-pack/H2D/update plus ordinary actor
public packaging and search. This is still profile-only and not Coach launch
advice.
```

### Borrowed State Retest After Key-Filter Tightening

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch refinement:

```text
Borrowed render-state mode now passes only renderer-required state keys by
reference. It no longer hands the persistent renderer the whole actor env.state
mapping. This keeps the no-copy win while preserving the earlier key-filter
contract.
```

Validation:

```text
uv run ruff check \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  tests/test_source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py \
  tests/test_mctx_synthetic_benchmark_legality.py
=> passed

uv run python -m py_compile \
  src/curvyzero/training/source_state_hybrid_observation_profile.py \
  src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
=> passed

uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_hybrid_observation_profile.py \
  -k "borrow_single_actor or resident or device_only or native_actor_buffer"
=> 12 passed, 23 deselected
```

Fresh H100 rows, B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay:

| row | run | roots/sec | env frac | search frac | key read |
| --- | --- | ---: | ---: | ---: | --- |
| resident sim16 copied | `ap-EK7pQiHw6eDMit9ddPxEit` | `34,092` | `64.6%` | `10.2%` | copied render-state write `0.349s` |
| resident sim16 borrowed | `ap-lDpp0GPxRLxEUzyRd7rpYi` | `51,791` | `52.5%` | `16.0%` | borrowed/copy-free, `1.52x` over copied |
| resident sim32 borrowed | `ap-0TyfdWOEotTTsXQsCaEvfY` | `38,451` | `42.6%` | `30.6%` | search now visibly hot |
| host sim16 borrowed | `ap-KwQDNEFJjgJVgvS1qDTeyh` | `34,543` | `56.6%` | `11.2%` | host stack shift/latest still visible |

Fresh no-observation-refresh ceilings:

| row | run | roots/sec | env frac | search frac | read |
| --- | --- | ---: | ---: | ---: | --- |
| resident sim16 refresh off | `ap-zKXn9DRQpRigvGGUYxvrlz` | `61,873` | `32.1%` | `19.3%` | borrowed sim16 is within `1.19x` of this ceiling |
| resident sim32 refresh off | `ap-6JrJh4ib7zR9KI0ytSzfEe` | `37,165` | `25.7%` | `32.4%` | sim32 is now search-dominated in this ceiling |

Scale check:

| row | run | roots/sec | env frac | search frac | read |
| --- | --- | ---: | ---: | ---: | --- |
| B2048 resident sim16 borrowed | `ap-6Lt81IxP3yMdwA9NwtI9xT` | `52,325` | `69.8%` | `9.1%` | no meaningful per-root throughput win over B1024 |

Plain read:

```text
Borrowed render-state is a keep. It moved resident sim16 from 34.1k to 51.8k
active roots/sec in the matched key-filtered row.

The old "render is the wall" story is now mostly false for the best compact
profile row. Raw GPU draw is still only about 4-6ms over loop24. The remaining
refresh-on gap to the no-refresh ceiling is only about 1.2x at sim16.

At sim32, search is already about 30% of wall in borrowed/ceiling rows. The next
big target should not be another one-frame renderer kernel. It should be either
the remaining state/observation handoff ownership or the real search/service
boundary, with a normal-death/RND edge check before Coach-facing advice.
```

### Lazy Resident Stack Sync Falsifier

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Patch:

```text
mctx_synthetic_benchmark.py can now set compact_visual_resident_sync=false.
That removes the explicit block_until_ready() after the resident GPU stack FIFO
update. Search still consumes the stack, so this only tests whether the explicit
wait was useful timing attribution or real wall-clock cost.
```

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
=> passed

uv run python -m py_compile src/curvyzero/infra/modal/mctx_synthetic_benchmark.py
=> passed

uv run pytest -q -p no:cacheprovider tests/test_mctx_synthetic_benchmark_legality.py
=> 5 passed
```

Fresh H100 rows, B1024/P2/body4096/h64/depth16/loop24/native/no-copy/replay,
resident GPU stack, borrowed single-actor render state:

| row | run | sims | resident sync | roots/sec | env frac | search frac | observation | resident stack update | read |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | `ap-Mgvb8AFe3q1HzbiMAepQjL` | 16 | on | `48,579` | `53.8%` | `15.2%` | `0.303s` | `0.020s` | earlier matched row |
| lazy sync | `ap-iA7EpkLNZLvCPjMDEWodxi` | 16 | off | `54,717` | `52.4%` | `16.7%` | `0.264s` | `0.015s` | `1.13x` over baseline |
| baseline | `ap-IkSystwnJLV5kDHSFobmZe` | 32 | on | `36,041` | `43.3%` | `28.3%` | `0.331s` | `0.023s` | earlier matched row |
| lazy sync | `ap-QKJXlY0dZ0zIQpJU4g4oA6` | 32 | off | `27,944` | `44.4%` | `24.4%` | `0.446s` | `0.026s` | regression |

Plain read:

```text
Lazy resident-stack sync is not a clear speed mode. Sim16 improved, but sim32
regressed enough that this should stay an attribution switch until repeated.

The current next target is larger than the resident stack wait itself. In the
lazy sim16 row, raw GPU draw was only 0.004s over loop24, while
production-to-compact was 0.078s, delta pack was 0.104s, and renderer H2D was
0.067s. The next clean prototype is a compact render-state owner/direct compact
delta path, not another block_until_ready tweak.
```

### Current-Code Refresh Ceiling Retest

Status: profile-only. No live Coach training run, checkpoint, eval, GIF, or
tournament artifact was touched.

Fresh H100 rows, B1024/P2/body4096/h64/depth16/loop48/native/no-copy/replay,
resident GPU stack, explicit resident sync off:

| row | run | sims | roots/sec | env frac | search frac | observation handoff leaf | public packaging leaf | read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| refresh-on, borrowed | `ap-XmQM6DmkFvDls76ZpU0xzE` | 16 | `62,652` | `71.2%` | `19.8%` | `0.664s` | `0.238s` | current best refresh-on profile shape |
| refresh-on, borrowed | `ap-ABmbtE2z87CaEfvcERadBS` | 32 | `49,097` | `54.1%` | `38.9%` | `0.645s` | `0.233s` | sim32 makes search large |
| refresh-off ceiling | `ap-Re7uaeDqBjNOI8LxQdRWeI` | 16 | `98,547` | `54.3%` | `29.8%` | `0.00045s` | `0.261s` | borrow flag is incompatible with refresh-off and was disabled |
| refresh-off ceiling | `ap-DDZi5dTudchj4ZnaoDvWGw` | 32 | `74,871` | `31.9%` | `58.6%` | `0.00035s` | `0.218s` | search is now slowest bucket |

Plain read:

```text
The refresh-off ceiling is only about 1.57x over refresh-on at sim16 and about
1.52x at sim32. That prices the maximum current gain from deleting the remaining
observation refresh on this denominator.

This is enough to stop treating the direct visual-delta canary as the main 5-10x
path. It may still be worth a small profile-only test, but the main speed thesis
must move to compact-buffer/search-service ownership.
```

Subagent docs added:

```text
subagent_full_dataflow_map_20260522.md
subagent_gpu_sync_model_20260522.md
subagent_architecture_design_critiques_20260522.md
subagent_direct_visual_delta_canary_critique_20260522.md
```

## 2026-05-23 Compact Service Validation And Dense Torch Wiring

Status: local/profile-only wiring. No live Coach runs, training runs,
checkpoints, eval, GIF, tournament artifact, or Modal volume state touched.

Patch:

```text
tests/test_compact_search_replay_contract.py
  Add opt-in stock LightZero GameSegment/MuZeroGameBuffer target-hook parity.
  CompactReplayIndexRowsV1 materializes to target rows, builds real GameSegments
  when lzero is installed, pushes them into MuZeroGameBuffer, and compares
  stock reward/value/policy target hooks against the materialized rows.

src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py
  dense_torch_mcts now stores compact search arrays just like mock/service-tax.
  run_compact_batch() can validate those arrays into CompactSearchResultV1.

scripts/build_curvytron_hybrid_observation_profile_grid.py
  dense_torch_mcts modes are allowed with compact_service_replay_proof.
```

Validation:

```text
uv run ruff check src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py scripts/build_curvytron_hybrid_observation_profile_grid.py tests/test_source_state_batched_observation_boundary_profile.py tests/test_curvytron_hybrid_observation_profile_grid_builder.py tests/test_compact_search_replay_contract.py
=> passed

uv run pytest -q -p no:cacheprovider tests/test_source_state_batched_observation_boundary_profile.py -k 'dense_torch_mcts_is_profile_only or dense_torch_compact_replay_contract or service_tax_probe_stores_compact_search_arrays'
=> 3 passed

uv run pytest -q -p no:cacheprovider tests/test_curvytron_hybrid_observation_profile_grid_builder.py -k 'dense_torch'
=> 2 passed

uv run pytest -q -p no:cacheprovider tests/test_compact_search_replay_contract.py -k 'stock_lightzero_target_hooks or materialized_sample_batch'
=> 2 passed
```

Plain read:

```text
The compact path has a stronger native LightZero target proof now, but still
does not claim public MuZeroGameBuffer.sample parity.

Dense Torch MCTS is now a same-boundary profile candidate. The next run should
compare direct_ctree_gpu_latent, dense_torch_mcts, service_tax_probe, and
mock_search_service on the same H100 B512/A16 compact replay denominator.
```

### Same-Denominator Compact Search Service Grid

Status: profile-only. No Coach training run, checkpoint, eval, GIF, tournament
artifact, or live Modal volume state was touched.

Runner:

```text
scripts/run_curvytron_hybrid_observation_profile_manifest.py
```

Results:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-direct-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-dense-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-service-tax-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-mock-20260523b
```

Shape:

```text
H100, B512, actor_count=16, steps=80, warmup=20, direct_gray64,
jax_gpu_persistent_policy_framebuffer_profile, uint8 stack,
host_uint8_pinned input, no scalar timestep materialization,
compact_service_replay_proof=true, root_noise_weight=0.
```

| sims | mode | measured sec | steps/sec | probe sec | probe roots/sec | model sec | search sec | compact proof sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `direct_ctree_gpu_latent` | `18.82` | `4,353` | `7.734` | `10,592` | `2.047` | `5.924` | `5.577` |
| 16 | `dense_torch_mcts` | `15.23` | `5,380` | `3.687` | `22,219` | `1.576` | `1.942` | `4.086` |
| 16 | `service_tax_probe` | `10.94` | `7,487` | `2.041` | `40,135` | `1.642` | `0.228` | `4.082` |
| 16 | `mock_search_service` | `11.78` | `6,955` | `0.615` | `133,107` | `0.362` | `0.000` | `5.240` |
| 32 | `direct_ctree_gpu_latent` | `30.15` | `2,717` | `14.096` | `5,812` | `3.836` | `11.523` | `8.738` |
| 32 | `dense_torch_mcts` | `31.75` | `2,580` | `9.230` | `8,875` | `2.980` | `6.058` | `6.535` |
| 32 | `service_tax_probe` | `11.97` | `6,847` | `3.478` | `23,553` | `2.911` | `0.422` | `4.087` |
| 32 | `mock_search_service` | `14.38` | `5,696` | `0.519` | `157,708` | `0.360` | `0.000` | `9.011` |

Ratios versus direct CTree:

```text
sim16:
  dense_torch_mcts: 1.24x measured steps/sec, 2.10x raw probe
  service_tax_probe: 1.72x measured steps/sec, 3.79x raw probe
  mock_search_service: 1.60x measured steps/sec, 12.57x raw probe

sim32:
  dense_torch_mcts: 0.95x measured steps/sec, 1.53x raw probe
  service_tax_probe: 2.52x measured steps/sec, 4.05x raw probe
  mock_search_service: 2.10x measured steps/sec, 27.14x raw probe
```

Plain read:

```text
Direct LightZero CTree with GPU latent inference is expensive on this boundary.
At sim32 it spends 14.1s in the probe, with 11.5s in search and 3.8s in model.

Dense Torch is not a promotion candidate yet. It cuts raw probe time at sim16,
but at sim32 the full measured row is slightly slower than direct CTree, and it
is not LightZero CTree semantics. Keep it as a research probe only.

The service-tax rows say there is real upside in a compact search service:
roughly 1.7x measured at sim16 and 2.5x measured at sim32 before changing the
game loop. The mock rows show a larger theoretical ceiling if the real search
control loop can be moved out of the current Python/CTree boundary.

`compact_service_replay_proof_sec` is validation cost in this profile, not a
claimed training hot path. Use `probe sec` for search/service cost and
`measured sec` as a guard against hidden overhead.
```

Next step:

```text
Do not promote dense_torch_mcts. The next practical implementation slice is a
real compact service adapter around the current direct CTree path, then a
closed-loop proof where selected actions drive the next env step and compact
replay/RND rows attach correctly. In parallel, keep the MCTX/fixed-shape search
route as the bigger architecture bet.
```

Subagent cross-checks:

```text
Socrates: the same-denominator grid points at the direct CTree/list/CPU-GPU
boundary. The telemetry to trust is aggregate timings, especially
lightzero_mcts_arrays_boundary_total_sec/search_sec/non_model_sec and the
matching service-tax fields. Dense Torch should not be polished unless it wins
the same sim16/sim32 gate.

Darwin: external high-throughput RL systems point at the same design:
array-native batched search, contiguous buffers, and fewer per-simulation
Python/host synchronization points. MCTX is the clean sidecar comparator; the
trainer-facing path still needs compact replay/RND/sampler gates.
```

### MCTX/JAX Compact Visual Sidecar Refresh

Status: profile-only architecture evidence. This is not LightZero-equivalent and
not Coach launch advice.

Commands:

```text
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark
  --observation-mode curvytron_hybrid_compact_visual_sample
  --compute h100
  --batch-size 512
  --player-count 2
  --body-capacity 1024
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile
  --num-simulations 16|32
  --hidden-dim 64
  --max-depth 16
  --warmup-runs 2
  --steady-runs 5
  --closed-loop-steps 24
  --native-actor-buffer
  --no-compact-root-copy-observation
  --no-compact-visual-resident-sync
  --no-emit-full-json
```

Modal apps:

```text
sim16: ap-1dPOsP9dPDNO5a9XsZXnt2
sim32: ap-UMr7DGD1Zt6cV55qWhozvk
```

Results:

| sims | active roots/sec | total sec | search sec | env step sec | h2d sec | d2h sec | replay index sec | one-step active roots/sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `27,635` | `0.889` | `0.149` | `0.647` | `0.0437` | `0.0093` | `0.0056` | `118,968` |
| 32 | `22,219` | `1.106` | `0.362` | `0.650` | `0.0431` | `0.0104` | `0.0055` | `59,655` |

Plain read:

```text
The MCTX/JAX sidecar still shows the architecture upside. Even in a real compact
visual closed loop with replay-index rows, it is much faster than the direct
LightZero CTree compact-service rows. But it uses a toy JAX model/search, not
the current LightZero model, so the result is a direction signal only.

At this speed, env/observation becomes the wall again: 72.8% of sim16 wall and
58.7% of sim32 wall. The next architecture must address both sides:
device/array-native search plus compact env/observation ownership.
```

### Dataflow Wave 2 Same-Denominator Profile, 2026-05-23

Status: running. Profile-only; no live Coach training runs touched.

Purpose:

```text
Refresh the current direct-vs-service headroom after the latest code/doc
reorientation, with one denominator and no root noise.
```

Common shape:

```text
compute: H100 and L4
batch_size: 512
actor_count: 16
measured steps: 80
warmup steps: 20
simulations: 16 and 32
scalar timestep materialization: off
compact service replay proof: on
root_noise_weight: 0.0
profile_only: true
calls_train_muzero: false
touches_live_runs: false
```

Manifests:

- `artifacts/local/curvytron_hybrid_observation_profile_manifests/dataflow_wave2_direct_20260523a/manifest.json`
- `artifacts/local/curvytron_hybrid_observation_profile_manifests/dataflow_wave2_compacttorch_20260523a/manifest.json`
- `artifacts/local/curvytron_hybrid_observation_profile_manifests/dataflow_wave2_servicetax_20260523a/manifest.json`
- `artifacts/local/curvytron_hybrid_observation_profile_manifests/dataflow_wave2_mock_20260523a/manifest.json`

Rows:

- direct CTree GPU latent: real LightZero CTree boundary baseline;
- compact Torch search service: current profile-only real Torch candidate;
- service-tax probe: model/service/control ceiling without real MCTS update;
- mock search service: no-search output ceiling and materialization guard.

Read rules:

- compare measured `steps_per_sec` for same-denominator guardrails;
- compare `probe_total_sec`, `search_sec`, and `model_sec` to locate the wall;
- treat compact replay proof time as validation cost in this profile, not Coach
  training speed;
- kill any row with missing identity/replay telemetry, fallback calls, or a
  mixed currency summary.

Results:

| compute | sims | direct CTree steps/sec | compact Torch steps/sec | service-tax steps/sec | mock steps/sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 | 16 | `5,467` | `4,047` | `7,812` | `7,462` |
| H100 | 32 | `3,137` | `2,674` | `5,192` | `9,171` |
| L4/T4 | 16 | `2,671` | `2,724` | `3,408` | `7,611` |
| L4/T4 | 32 | `2,727` | `1,859` | `4,395` | `4,016` |

Ratios versus direct CTree:

```text
H100 sim16:
  compact Torch: 0.74x
  service-tax:   1.43x
  mock ceiling:  1.36x

H100 sim32:
  compact Torch: 0.85x
  service-tax:   1.65x
  mock ceiling:  2.92x

L4/T4 sim16:
  compact Torch: 1.02x
  service-tax:   1.28x
  mock ceiling:  2.85x

L4/T4 sim32:
  compact Torch: 0.68x
  service-tax:   1.61x
  mock ceiling:  1.47x
```

Timing read:

```text
H100 direct CTree sim16:
  measured 14.98s, probe 6.17s, model 1.66s, search 4.82s

H100 direct CTree sim32:
  measured 26.11s, probe 13.90s, model 3.77s, search 11.66s

H100 compact Torch sim16:
  measured 20.24s, probe 9.55s, initial model 0.36s, search 8.18s

H100 compact Torch sim32:
  measured 30.64s, probe 22.26s, initial model 0.36s, search 21.10s
```

Plain read:

```text
The current eager compact Torch service is not the next optimization lane.
It is worse than direct CTree on H100 for both sim16 and sim32, despite using
the compact service boundary.

The service-tax and mock rows still show meaningful headroom. That means the
right next move is not polishing this eager Torch tree. It is changing the
search/data ownership: fixed-shape compiled search, array-native CTree, MCTX/JAX
comparator, or a Puffer-style slab/search service with delayed replay payloads.
```

L4 read:

```text
Do not overread the L4 rows. They are useful sanity checks, but validation cost
and variance are large enough that H100 remains the cleaner architecture
denominator. L4 is probably acceptable for cheap canaries; H100 is the right
machine for final architecture profiling.
```

Tooling note:

```text
The first summaries showed null root-noise values for some rows even though the
manifest forced root_noise_weight=0.0. The runner now falls back to the manifest
row for root noise and to generic input-byte telemetry for array-ceiling
obs_h2d_bytes when the explicit ledger is absent.
```

## 2026-05-23d Compact Slab H100 Denominator

Purpose:

```text
Test the profile-only CompactRolloutSlab path with real direct CTree search,
action feedback into the next env step, and compact replay-index commits.
```

Important correction:

```text
The first summary divided all slab roots by the last search-service call time.
That was wrong. The correct denominator is aggregate compact_rollout_slab_sec.
The runner now keeps last-call timings under compact_rollout_slab_last_*.
```

Corrected warm smoke:

| experiment | impl | steps/sec | slab roots/sec | slab sec | last service sec |
| --- | --- | ---: | ---: | ---: | ---: |
| `opt-compact-slab-direct-warm-telemetry-20260523` | `direct_ctree_arrays` | `1560` | `2270` | `1.128` | `0.0505` |
| `opt-compact-slab-direct-warm-telemetry-20260523` | `direct_ctree_gpu_latent` | `2204` | `3726` | `0.687` | `0.0305` |

Active follow-up wave:

```text
opt-compact-slab-h100-main-20260523d
  direct_ctree_gpu_latent, B256/B512/B1024, sim8/sim16/sim32, A16

opt-compact-slab-h100-direct-arrays-controls-20260523d
  direct_ctree_arrays, B512/B1024, sim8/sim16/sim32, A16

opt-compact-slab-h100-service-tax-20260523d
  service_tax_probe, B512/B1024, sim16/sim32, A16

opt-compact-slab-h100-mock-ceiling-20260523d
  mock_search_service, B512/B1024, sim16/sim32, A16

opt-compact-slab-h100-actor-tax-a8-20260523d
opt-compact-slab-h100-actor-tax-a32-20260523d
  direct_ctree_gpu_latent, B512/sim16, actor-count tax rows
```

All rows are profile-only and must not be used as Coach training claims.

Result summary:

```text
Best real-search throughput:
  direct_ctree_gpu_latent B1024/A16/sim8:  8291 steps/sec
  direct_ctree_gpu_latent B1024/A16/sim16: 6992 steps/sec
  direct_ctree_gpu_latent B1024/A16/sim32: 5314 steps/sec

Best balanced default for stronger search:
  H100 B1024/A16/sim16, direct_ctree_gpu_latent

Direct arrays controls:
  B1024/A16/sim16: 2613 steps/sec
  B1024/A16/sim32:  947 steps/sec

Ceilings:
  service_tax_probe B1024/A16/sim16: 12072 steps/sec
  mock_search_service B1024/A16/sim16: 15590 steps/sec
```

Detailed table:

```text
docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/compact_slab_h100_profile_summary_20260523d.md
```

Current read:

```text
The slab path works and direct_ctree_gpu_latent is the right real-search
baseline. The remaining headroom is real but bounded in this shape: about
1.7x-2.8x from service-tax/mock ceilings depending on sim count. To go beyond
that, the next move must attack the CTree/search control loop or the broader
dataflow architecture behind CompactSearchServiceV1.
```

Precomputed recurrent falsifier, current code:

```text
opt-compact-slab-h100-precomputed-recurrent-20260523d

B512/A16/sim16:  7086 steps/sec
B512/A16/sim32:  5967 steps/sec
B1024/A16/sim16: 8875 steps/sec
B1024/A16/sim32: 8055 steps/sec
```

Plain read:

```text
This is explicitly synthetic and not a valid training backend. It shows that
deleting recurrent model calls helps, especially at sim32, but it is only
1.09x-1.52x over real direct_ctree_gpu_latent on the tested shapes. The
remaining wall is still the CTree/search/control/list path plus H2D and
env/observation overhead.
```

## 2026-05-23d Compact Slab Sample Gate

Local-only validation, no live Coach runs touched.

What changed:

```text
The hybrid profile can now enable:
  --hybrid-compact-rollout-slab-sample-gate
  --no-hybrid-materialize-scalar-timestep

This prices the edge from compact replay-index rows into target rows and a
sample batch. It rejects scalar timestep materialization so the row cannot
accidentally hide old BaseEnvTimestep work.
```

Validation:

```text
ruff check: passed
tests/test_source_state_hybrid_observation_profile.py
tests/test_source_state_batched_observation_boundary_profile.py
tests/test_curvytron_hybrid_observation_profile_grid_builder.py
tests/test_curvytron_hybrid_observation_profile_manifest_runner.py
  -k "compact_rollout_slab or sample_gate or compact_hybrid_observation_profile_result"
  -> 9 passed, 188 deselected
```

Tiny Modal smoke:

```text
app: ap-g8C6fnvHtHym386bAq7odu
compute: gpu-l4-t4
shape: B2/A1/steps2/warmup1/mock_search_service
scalar timestep materialization: false
compact_rollout_slab_sample_gate_enabled: true
sample gate rows:
  index=8
  target=8
  sample=8
mock BaseEnvTimestep rows: 0
ok: true
```

Next profile row to run:

```text
MCTX or direct CTree compact slab, scalar off, sample gate on.
Goal: price sample-edge overhead on the same H100/L4 profile rows as the search
throughput grid.
```

Matched H100 overhead rows, direct CTree GPU-latent, B64/A4/sim8/steps20/warmup5:

```text
baseline slab, no sample gate:
  steps/sec: 3660
  measured_sec: 0.699
  compact_rollout_slab_sec: 0.475

sample gate every collected step, sample all rows:
  steps/sec: 1064
  sample_gate_sec: 1.581
  rows: index=2560, target=2560, sample=2560

sample gate every collected step, sample batch size 64:
  steps/sec: 1532
  sample_gate_sec: 1.042
  rows: index=2560, target=2560, sample=1280

sample gate once per 20 opportunities, sample batch size 64:
  steps/sec: 3428
  sample_gate_sec: 0.063
  rows: index=128, target=128, sample=64
  opportunities=20, skipped=19
```

Read:

```text
The compact sample edge works remotely and does not require scalar
BaseEnvTimestep rows. But running it on every env step is a stress test, not a
realistic learner cadence. At a chunk-like cadence, the sample edge is a small
tax in this shape; the main profile wall remains search/model/control plus the
observation/dataflow path, not replay sampling.
```
