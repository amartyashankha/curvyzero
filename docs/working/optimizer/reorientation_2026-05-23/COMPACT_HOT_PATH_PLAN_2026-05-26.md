# Compact Hot Path Plan, 2026-05-26

## Goal

Build one honest optimizer profile row for the compact-owned hot path.

The row must include:

- search feedback actions;
- bounded compact replay-ring sampling;
- learner-gate consumption;
- RND-style latest-frame input;
- scalar timestep materialization off;
- explicit profile-only metadata.

It must not touch live Coach runs or claim stock `train_muzero` speed.

## Current Artifact

`src/curvyzero/training/source_state_hybrid_observation_profile.py`

Current object:

```text
_CompactReplayRingV1 plus compact_rollout_slab_sample_gate and learner_gate
```

## Local Gates

- [x] Bounded replay-ring pair capacity and eviction counters.
- [x] Empty replay ring returns an explicit no-op result.
- [x] Multi-record sampling preserves RNG row order.
- [x] Fast sample path uses terminal final observations.
- [x] Fast sample path uses index-row final rewards.
- [x] Sample metadata uses explicit sample-batch contract keys and non-claims.
- [x] Manifest command includes replay-ring pair capacity.
- [x] Broad local profile/tooling tests.
- [x] One warm profile-only row.

## Warm Row Result

Label: promotion gate for the optimizer profile-only compact denominator.

Artifacts:

- manifest:
  `artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-compact-hotpath-t1-20260526/manifest.json`
- result:
  `artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-t1-20260526/row_001_result.json`

Shape:

- H100;
- B1024/A16;
- 100 warmup steps, 120 measured steps;
- sim8 compact Torch;
- `search_feedback`;
- replay-ring sample gate batch `512`, interval `8`, pair capacity `4096`;
- CUDA learner gate with RND-style latest-frame input.

Result:

- measured `22.262s`;
- `11039.6` roots/sec;
- scalar materialization `0.0s`;
- Python rows materialized `0.0`;
- committed replay rows `245760`;
- replay payload bytes `6881280`;
- sampled rows `7680`;
- learner/RND updates `15`;
- sample gate `1.702s`;
- learner gate `0.532s`;
- search/probe `6.934s`;
- observation `6.674s`;
- actor/env wall `6.221s`.

Read:

The compact accounting path is now honest enough to profile. It is still
profile-only and does not call `train_muzero`. The next wall is no longer replay
materialization alone. The largest visible buckets are search/probe,
observation movement, and actor/env stepping.

Caveat:

Compact Torch reported a compile fallback because root noise is on. Keep that
explicit in speed claims.

## Controlled Comparison

Artifacts:

- direct CTree:
  `artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-c1-direct-20260526/row_001_result.json`
- compact Torch:
  `artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-crows-20260526/row_001_result.json`
- fixed-shape floor:
  `artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-c3-floor-20260526/row_001_result.json`

Shape:

- H100;
- B1024/A16;
- 80 warmup steps, 200 measured steps;
- sim8;
- `scripted_random`;
- scalar timestep materialization off;
- no replay sampling and no learner gate.

All rows match:

```text
env_action_checksum_total      420223838
env_trajectory_checksum_total  1967449471838
```

Result:

| row | measured sec | roots/sec | probe sec | actor sec | observation sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct CTree GPU latent | `47.864` | `8557.7` | `29.111` | `9.220` | `9.193` |
| compact Torch | `32.553` | `12582.4` | `11.982` | `10.632` | `9.536` |
| fixed-shape floor | `22.766` | `17992.1` | `0.889` | `11.767` | `9.644` |

Read:

Compact Torch is `1.47x` faster than direct CTree on this controlled profile
denominator. The no-real-search floor is still `1.43x` faster than compact
Torch. Search is still a worthwhile lever, but the actor/env and observation
buckets are already large enough that a pure search win cannot deliver a `10x`
loop speedup by itself.

## Compile And Staging Follow-Up

The compact Torch service used to report compile eligibility without actually
calling `torch.compile`. That was misleading. The service now has explicit
runtime telemetry:

- `compact_torch_search_compile_attempted`;
- `compact_torch_search_compile_used`;
- `compact_torch_search_compile_cache_hit`;
- `compact_torch_search_compile_runtime_status`.

Local gates passed:

```text
uv run pytest tests/test_compact_torch_search_service.py tests/test_source_state_batched_observation_boundary_profile.py -k compact_torch_search_service
uv run pytest tests/test_curvytron_hybrid_observation_profile_manifest_runner.py -k 'compile_runtime or compact_rollout_slab_rows or root_noise'
uv run ruff check src/curvyzero/training/compact_torch_search_service.py tests/test_compact_torch_search_service.py tests/test_source_state_batched_observation_boundary_profile.py scripts/run_curvytron_hybrid_observation_profile_manifest.py tests/test_curvytron_hybrid_observation_profile_manifest_runner.py
```

Support row:

```text
artifact: artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-c2-root0-compiled-20260526/row_001_result.json
shape: H100 B1024/A16, 80 warmup, 200 measured, sim8, scripted_random, root_noise_weight=0.0
measured_sec: 24.107
roots/sec: 16990.9
probe_sec: 7.788
search_sec: 4.828
model_sec: 1.704
h2d_sec: 0.775
compile telemetry at last measured step: used=true, cache_hit=true, runtime_status=cache_hit
```

Read:

Root-noise-zero alone did not help; before the actual compile patch the same
shape was `33.148s`. Actual compiled helpers moved the controlled compact Torch
row from `32.553s` to `24.107s`, about `1.35x`. This is support evidence, not a
training recommendation, because it changes root-noise behavior unless
pre-sampled root noise is added to the compiled path.

Follow-up correction:

Root noise does not need to block compiled helper use. The helpers consume
already-built priors; root noise is applied before them. The default compile
contract now allows positive root noise and reports actual runtime compile use.

Default-root-noise controlled row:

```text
artifact: artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-c2-defaultnoise-compiled-20260526/row_001_result.json
measured_sec: 29.530
roots/sec: 13870.8
probe_sec: 8.606
search_sec: 5.441
compile telemetry: status=eligible, used=true, cache_hit=true
```

Read:

With default root noise still on, compiled helpers improve the controlled
compact Torch row from `32.553s` to `29.530s`, about `1.10x`. The bigger
root0 support win was partly because root-noise work disappeared.

Native actor-buffer row:

```text
artifact: artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-t1-nativeactor-20260526/row_001_result.json
shape: same as warm trainer-like row, plus --hybrid-native-actor-buffer
measured_sec: 18.459
roots/sec: 13314.1
probe_sec: 6.021
observation_sec: 5.163
actor_wall_sec: 5.141
sample_gate_sec: 1.577
learner_gate_sec: 0.455
```

Read:

Native actor buffering improves the trainer-like profile row from `22.262s` to
`18.459s`, about `1.21x`. This is the cleanest practical win from the current
wave.

Native actor plus default-root-noise compiled helpers:

```text
artifact: artifacts/local/curvytron_hybrid_observation_profile_results/optimizer-compact-hotpath-t1-nativeactor-defaultnoise-compiled-20260526/row_001_result.json
measured_sec: 17.127
roots/sec: 14349.2
probe_sec: 5.103
observation_sec: 4.871
actor_wall_sec: 5.132
sample_gate_sec: 1.439
learner_gate_sec: 0.481
compile telemetry: status=eligible, used=true, cache_hit=true
```

Read:

This is the best current trainer-like profile-only row: `1.30x` faster than the
warm baseline while keeping default root noise on.

Observation staging ceilings on the fixed-shape floor:

| row | measured sec | roots/sec | observation sec | actor sec |
| --- | ---: | ---: | ---: | ---: |
| baseline floor | `22.766` | `17992.1` | `9.644` | `11.767` |
| device-only stack ceiling | `14.433` | `28379.2` | `5.623` | `8.010` |
| no-refresh ceiling | `11.344` | `36107.1` | `2.945` | `7.490` |

Read:

Observation staging has real headroom, but these are ceilings. The next useful
implementation is not "turn refresh off" as a training setting. It is a
resident observation/search contract where the consumer actually reads the
fresh device-owned stack without round-tripping through the host.

## Experiment Rule

Run local tests first. Then run one warm profile-only row. Label the result as:

- promotion gate;
- falsifier;
- support evidence;
- rerun because denominator was wrong.

Do not open a new optimization lane before that label exists.
