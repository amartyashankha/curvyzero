# Resident Observation/Search Contract, 2026-05-26

## Goal

Turn the observation-staging ceiling into a real candidate path.

The fast rows are not valid training semantics by themselves:

- `--hybrid-device-only-stack` is only valid if the search consumer reads the
  fresh device stack directly.
- `--no-hybrid-refresh-observation-stack` is a ceiling only. It must not become
  a training setting.

The implementation target is a fresh device-owned observation handle that flows
from renderer to compact root batch to compact Torch search without a host
round-trip.

## Required Contract

Create a small explicit object, not a hidden fallback:

```text
ResidentObservationBatchV1
```

It should carry:

- `device_observation`: device tensor/array with shape `[B, P, C, H, W]`;
- `root_device_observation`: optional view with shape `[B*P, C, H, W]`;
- `generation_id`: integer incremented after every successful render/stack
  update;
- `batch_size`, `player_count`, `stack_shape`, `dtype`, `device`;
- `row_major_order=true`;
- `fresh_for_step_index`;
- `source_backend`;
- `host_fallback_allowed=false`.

The compact root batch may still carry a host observation for old lanes, but the
resident path must say which source is authoritative:

```text
observation_source = "resident_device_v1"
resident_generation_id = ...
host_observation_authoritative = false
```

## Fail-Closed Rules

The resident path must raise instead of silently falling back when:

- the renderer did not produce a resident device observation;
- the generation id is stale;
- the root order is not row-major `[env row, player]`;
- the consumer asks for `resident_device_v1` but receives only host `np.ndarray`;
- `device_only_stack=true` is set while the compact search consumer still reads
  `root_batch.observation`;
- a terminal/final observation row needs host-only final-observation semantics
  that the resident path does not yet implement.

## Minimal Implementation Sequence

1. Add `ResidentObservationBatchV1` in a small shared module near
   `compact_policy_row_bridge.py`.
2. Teach `HybridBatchedObservationProfileManager._update_observation` to return
   both the existing host stack and an optional resident handle when the
   persistent GPU renderer is active.
3. Extend `HybridCompactBatch` with optional resident observation metadata and
   handle. Keep existing host observation fields for non-resident lanes.
4. Extend `build_compact_root_batch_v1` with an explicit
   `observation_source` argument:
   - `host_array_v1`;
   - `resident_device_v1`.
5. Teach `CompactTorchSearchServiceV1` to consume resident-device observations
   only when `observation_source == "resident_device_v1"`.
6. Add Modal/grid flags that say exactly what is being measured:
   - `--hybrid-resident-observation-search`;
   - keep `--hybrid-device-only-stack` as a ceiling unless this flag is also on.
7. Emit counters in every row:
   - `resident_observation_used`;
   - `resident_observation_generation_id`;
   - `resident_observation_stale_count`;
   - `resident_observation_host_fallback_count`;
   - `resident_observation_h2d_bytes`;
   - `resident_observation_d2h_bytes`;
   - `resident_observation_source`.

## Tests

Unit tests:

- resident handle validates shape, dtype, row-major order, generation id, and
  device text;
- compact root batch rejects `resident_device_v1` without a resident handle;
- compact Torch search rejects resident mode if it would read host observation;
- stale generation id raises;
- host fallback count must stay zero in resident mode;
- terminal/final rows either have a supported resident final-observation path or
  fail closed.

Profile tests:

- resident search row reports `resident_observation_used=true`;
- `obs_h2d_bytes=0` for the search input;
- `resident_observation_host_fallback_count=0`;
- selected-action and trajectory checksums match the equivalent host row for
  controlled `scripted_random` actions.

## Current Priority

Do this after the current native actor plus compiled-helper profile is stable.
The expected headroom is the difference between the fixed-shape floor baseline
and observation ceilings:

```text
floor baseline:       22.766s
device-only ceiling:  14.433s
no-refresh ceiling:   11.344s
```

Treat the likely practical target as closer to the device-only ceiling than the
no-refresh ceiling, because valid training still needs fresh observations.

## Implementation Status

2026-05-26 local handoff patch:

- `ResidentObservationBatchV1` lives in
  `src/curvyzero/training/compact_observation_contract.py`.
- The persistent GPU renderer returns explicit `device_frames`.
- `HybridBatchedObservationProfileManager` builds a resident Torch stack from
  renderer device frames through Torch/DLPack, newest frame last.
- `HybridCompactBatch` carries `observation_source` plus the resident handle.
- `CompactRolloutSlab` passes the resident handle into `CompactRootBatchV1`.
- `_LightZeroArrayCeilingStackProbe` can instantiate
  `CompactTorchSearchServiceV1(require_resident_observation=True)`.
- `CompactTorchSearchServiceV1` rejects host fallback, including zero-active
  root batches, and reports search observation H2D bytes as `0.0` when resident
  observations are used.
- 2026-05-28 update: resident terminal/final-observation rows no longer fail
  closed just because they are terminal. They must carry resident-owned
  `final_device_observation`, `root_final_device_observation`, and
  `final_observation_row_mask`; host final observations are rejected as
  authoritative input on the resident path.

Still not done:

- run the larger B1024/A16 controlled-action profile before any speed
  recommendation.

2026-05-26 local hardening update:

- Added tests proving resident search ignores poisoned host frames.
- Added tests proving the slab consumes the current resident generation on each
  step.
- Added a direct `_make_compact_batch` misuse test proving host-owned resident
  terminal final-observation rows fail at the root-batch boundary.
- Fixed `compact_torch_search_service_resident_obs_reused` so it reports `1.0`
  when compact Torch actually consumes resident observations.

2026-05-28 terminal local update:

- Resident compact batches now own device final observations and final-row
  masks.
- Resident replay samples terminal final observations instead of autoreset or
  current observations.
- Terminal N-step resident samples propagate action/reward/policy/value
  validity masks into compact MuZero and zero post-terminal targets.

Modal proof rows in flight:

```text
candidate: artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-resident-proof-20260526/manifest.json
host comparator: artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-host-compare-proof-20260526/manifest.json
shape: H100, B256/A8, sim8, 20 warmup, 60 measured, scripted_random
```

Modal proof result:

- resident row completed with `resident_observation_used=true`;
- host fallback count was `0.0`;
- search observation H2D bytes were `0.0`;
- raw compact Torch service telemetry reported resident reuse as `1.0`;
- the small proof row was `2.601s` resident versus `2.783s` host, about
  `1.07x`.

Larger matched profile rows in flight:

```text
resident: artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-resident-b1024-proof-20260526/manifest.json
host:     artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-host-b1024-proof-20260526/manifest.json
shape: H100, B1024/A16, sim8, 80 warmup, 200 measured, scripted_random
```

Larger matched profile result:

- resident row: `21.245s`, `19279.8` steps/sec, observation `3.291s`,
  `obs_h2d_bytes=0`, resident used true;
- host row: `27.733s`, `14769.2` steps/sec, observation `8.734s`,
  `obs_h2d_bytes=6710886400`, resident used false;
- both rows matched action and trajectory checksums;
- speed read: `1.31x` resident over host on this controlled profile-only
  denominator.

Trainer-like resident rows in flight:

```text
resident: artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-resident-trainerlike-20260526/manifest.json
host:     artifacts/local/curvytron_hybrid_observation_profile_manifests/optimizer-host-trainerlike-20260526/manifest.json
shape: H100, B1024/A16, sim8, 100 warmup, 120 measured, search_feedback,
       sample gate 512 every 8 commits, learner gate cuda, RND-style input
```

Trainer-like result:

- resident row: `15.010s`, `16373.5` steps/sec, observation `2.288s`,
  probe `4.775s`, sample gate `1.855s`, learner gate `0.637s`,
  `obs_h2d_bytes=0`, resident used true;
- host row: `19.980s`, `12300.0` steps/sec, observation `5.865s`,
  probe `5.660s`, sample gate `1.933s`, learner gate `0.561s`,
  `obs_h2d_bytes=4026531840`, resident used false;
- both rows matched action checksum, trajectory checksum, replay rows, sampled
  rows, and learner/RND update count;
- speed read: `1.33x` resident over host on the compact trainer-like
  profile-only denominator.

Operating guardrail:

- `OPERATING_CONTRACT_2026-05-26.md` now makes this resident handoff the current
  critical path.
- Do not claim a resident speed win until a profile row proves the resident
  path was actually used and host fallback count is zero.

Passed local gates:

```text
uv run pytest tests/test_compact_torch_search_service.py -k 'resident or compile'
uv run pytest tests/test_compact_search_replay_contract.py -k resident_observation_contract
uv run pytest tests/test_source_state_hybrid_observation_profile.py -k 'resident_observation or persistent_device_only or compact_rollout_slab'
uv run pytest tests/test_source_state_hybrid_observation_profile.py tests/test_compact_torch_search_service.py tests/test_compact_search_replay_contract.py tests/test_source_state_batched_observation_boundary_profile.py -k 'resident or compact_torch_search_service or persistent_device_only'
uv run ruff check src/curvyzero/training/compact_observation_contract.py src/curvyzero/training/source_state_batched_observation_profile.py src/curvyzero/training/source_state_hybrid_observation_profile.py src/curvyzero/training/compact_policy_row_bridge.py src/curvyzero/training/compact_rollout_slab.py src/curvyzero/training/compact_torch_search_service.py src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py tests/test_source_state_hybrid_observation_profile.py tests/test_compact_torch_search_service.py
```
