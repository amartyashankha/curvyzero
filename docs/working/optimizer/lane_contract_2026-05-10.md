# Optimizer Lane Contract

Date: 2026-05-10

Status: active boundary note. Keep this short and boring.

## Ownership

Optimizer owns setup and measurement:

- actor-loop timing, profiling, Amdahl reads, CPU/GPU split, Modal/process
  boundaries, replay/write timing, policy/search cost timing, and when a speed
  rewrite is justified;
- profile report shape, denominator discipline, latency, throughput, policy
  staleness, artifact sizes, and caveats;
- optimizer-only tiny probes when they answer setup or timing questions.

Coach owns learning claims:

- eval protocol, checkpoint quality, scorecards, target correctness, LightZero
  reproduction status, and whether a learner/framework has useful signal;
- return curves, win rates, checkpoint promotion, and "does this policy improve?"
  calls.

Environment/RAM reconstruction owns source truth:

- JS oracle/probe parity, source behavior, unsupported semantics labels, vector
  parity, reset/autoreset/final-observation contracts, reward semantics, and
  source-fidelity claims;
- whether a fast path is allowed to stand for CurvyTron behavior.

## Handoff Rules

- Optimizer may consume source-backed contracts from Environment/RAM, but must
  not redefine source behavior for speed.
- Optimizer may consume Coach quality labels, but must not infer learning from
  throughput.
- Coach may consume Optimizer profile reports, but speed evidence is not eval
  evidence.
- Environment/RAM may consume Optimizer bottleneck reports to prioritize fast
  paths, but timing does not relax fidelity requirements.

## Current CurvyTron Profiling Contract

Primary visual profiling target:

```text
debug_visual_tensor
  -> curvyzero_debug_occupancy_gray64/v0
  -> raw uint8[1,64,64] CHW occupancy smoke frame
  -> optional LightZero-facing float32[1,64,64] CHW normalization
  -> optional frame stack/adapter timing if explicitly included
```

This is the primary optimizer path for CurvyTron visual smoke/profiling because
CurvyTron is visual and non-ALE. It is not source-faithful visual truth, not a
browser/canvas claim, and not learning-quality evidence.

Current strict wrapper speed bench:

```text
VectorTrainerEnv1v1NoBonus
  -> strict trainer-wrapper reset/step for 1v1/no_bonus/P=2
  -> wrapper action map converted to held source control state
  -> elapsed-ms server-frame advance under wrapper `decision_ms`
  -> float32[B,2,106] + bool[B,2,3]
  -> optional policy/search timing if explicitly included
  -> replay-v0 plumbing if explicitly included
  -> profile artifact with reset/autoreset timing if explicitly included
```

This is strict `VectorTrainerEnv1v1NoBonus` `[B,2,106]` trainer-wrapper
plumbing plus replay-v0. It is not Atari, ALE, a Gym ROM, or a real LightZero
env. Its `step` cadence and `decision_ms` are wrapper/profile choices, not
native CurvyTron source rules; source CurvyTron holds player control state and
advances it over elapsed-ms server frames.
The 1v1/no-bonus restriction is an explicit non-fidelity profiling config. It
must not be used as the reconstruction path or as a reason to avoid
source-default CurvyTron behavior.
LightZero-compatible field names only mean the payload has `observation`,
`action_mask`, and `to_play` shapes that can feed a later wrapper.

Current source-backed oracle-adjacent bench:

```text
CurvyTronSourceEnv snapshots
  -> source_snapshot_to_vector_trainer_state(...)
  -> source_world_bodies_circle_rays_v0 when body rays are timed
  -> float32[B,2,106] + bool[B,2,3]
  -> optional policy/search timing if explicitly included
  -> replay-v0 chunk/write timing if explicitly included
```

This is useful for adapter and observation timing because it starts from source
snapshots. It is still not full CurvyTron fidelity, bonuses, 3P/4P, visual
LightZero, or a learning claim.

Current multiplayer setup/profiling surface:

```text
VectorMultiplayerEnv
  -> metadata-only public 2P/3P/4P env
  -> curvyzero_debug_metadata_only/v0 observation
  -> reset provenance fields carried in info
MetadataOnlyMultiplayerEgoWrapper
  -> one ego action per configured live ego row
  -> opponent policy fills non-ego live slots
  -> full [B,P] wrapper action map plus sidecars
MultiplayerMetadataReplayRecorder
  -> metadata-only replay rows, not trainer replay
```

This is not a separate optimized game implementation. It is the shared
Environment/RAM multiplayer surface that Optimizer may time for setup,
wrapper, policy/search, and recorder costs. Reports must label it
`metadata_only=true` unless Environment promotes a learned observation or
source-visual contract. Do not compare it directly to strict
`VectorTrainerEnv1v1NoBonus` timings without naming the difference in player
count, observation, replay, and wrapper work.

Every profile report for these CurvyTron timing surfaces must carry the identity
fields below:

- `env_impl_id` or `visual_surface`: exactly `debug_visual_tensor` for the
  visual smoke/profiler, `VectorTrainerEnv1v1NoBonus` for the native scalar-ray
  lane, or `CurvyTronSourceEnv` for the source-backed scalar-ray lane.
- `ruleset_id` and `ruleset_hash`: the compatibility identity for the measured
  rules. Do not compare timings across different hashes.
- `source_claim_id` or source claim reference: the source-backed claim the
  timing is allowed to measure.
- `feature_flags`: at minimum `1v1=true`, `no_bonus=true`, `P=2`; any broader
  flag must be false or absent for this lane.
- For multiplayer setup reports, use the actual player count and mark
  `metadata_only=true`, `learned_observation_claim=false`, and
  `trainer_replay_claim=false`.
- `event_mode`: the event policy used by the run, such as no-event, debug-event,
  or replay-visible event refs.
- `decision_ms`: the fixed wrapper/profile decision window, not a native source
  tick rate.
- `body_capacity`, `event_capacity`, and `timer_capacity`: fixed capacities for
  the run.
- `reset_seed` and `reset_source`: the reset identity used for row initialization.
- Reset provenance fields when present: `random_tape_source`,
  `random_tape_length`, `rng_impl_id`, and `source_fixture_ref`.
- `final_observation_policy`: how terminal final observations are staged relative
  to autoreset and next observations.
- `includes_env_step`, `includes_render`, `includes_stack_normalize`,
  `includes_policy`, `includes_search`, `includes_replay`, and
  `includes_reset`: booleans saying whether each stage is inside the timed
  denominator.

If a legacy report lacks one of these fields, mark the gap in the readout and
do not treat the number as comparison-valid until the report is fixed.

Do not generalize scalar/ray timings to bonuses, broad lifecycle, 3P/4P, visual
LightZero, or full CurvyTron. Source-snapshot conversion probes can still be
used as setup scouts, but scalar/ray is diagnostic, not the primary training
target.

## Copyable Short Handoff

Optimizer boundary: use `debug_visual_tensor` /
`curvyzero_debug_occupancy_gray64/v0` as the current visual smoke/profiling
target, with raw `uint8[1,64,64]` CHW frames and optional normalized
`float32[1,64,64]` CHW LightZero payloads. Use strict
`VectorTrainerEnv1v1NoBonus` `[B,2,106]` and source-backed
`CurvyTronSourceEnv -> source_snapshot_to_vector_trainer_state(...)` reports as
scalar/ray diagnostics. Do not treat any of these surfaces as learning evidence
or full CurvyTron visual fidelity. Coach owns checkpoint/eval quality.
Environment/RAM owns source truth and whether fast/vector/visual paths are
faithful.
Every report must identify env impl, ruleset/hash, source claim, feature flags,
event mode, decision cadence, capacities, reset seed/source, final-observation
policy, and which of env step/render/stack-normalize/policy/search/replay/reset
are in the timed denominator.
