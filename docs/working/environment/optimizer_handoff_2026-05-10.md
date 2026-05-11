# Environment Handoff To Optimizer - 2026-05-10

Status: working handoff. Keep this short enough to paste, with links for the
longer proof.

## Copyable Handoff

Optimizer may profile the CurvyTron surfaces below, but every number must name
which surface it timed.

One runtime is under hardening: `VectorMultiplayerEnv`. Canonical 2P
status lives in [active_lanes.md](active_lanes.md#2p-status); this optimizer
handoff should only name the timed surface.

Strict 1v1/no-bonus proof and profiling boundary:

```text
1v1 / no bonuses / P=2
fixed trainer decision cadence
wrapper action map converted to held source control state
elapsed-ms server-frame advance under wrapper `decision_ms`
float32[B,2,106] ray/scalar observations
bool[B,2,3] masks, or int8 masks at LightZero boundaries
replay-v0 only when explicitly included
```

This is an explicit restricted proof/profile config, not the reconstruction
path. Do not use the restriction to avoid source-default CurvyTron behavior.

Oracle/proof adapter timing surface, not product path:

```text
CurvyTronSourceEnv snapshots
source_snapshot_to_vector_trainer_state(...)
source_world_bodies_circle_rays_v0 when body rays are timed
float32[B,2,106] ray/scalar observations
bool[B,2,3] masks
replay-v0 only when explicitly included
```

Debug visual tensor surface:

```text
curvyzero_debug_occupancy_gray64/v0
CurvyTron source/project state -> uint8[1,64,64]
optional stack helper -> uint8[4,64,64]
truth level: debug/profiling only, not source visual fidelity
```

Optimizer owns pushing this debug visual tensor into a usable smoke/profiler and
LightZero adapter plumbing target. Use it only when the report calls it a debug
visual tensor and says whether rendering, stacking, env step, reset, replay,
policy, and search are included. Do not compare it as if it were a final
source-faithful CurvyTron renderer.

Current fast shared multiplayer path under hardening; currently metadata-only,
not trainer-ready:

```text
VectorMultiplayerEnv
MetadataOnlyMultiplayerEgoWrapper
MultiplayerMetadataReplayRecorder
metadata-only 2P/3P/4P public env
debug metadata observation, not learned observation
reset provenance: seed/source/cursor/draw count plus random_tape_source,
  random_tape_length, rng_impl_id, optional source_fixture_ref
ego wrapper: one ego action in, opponent policy fills non-ego actions,
  full [B,P] wrapper action map plus sidecars out
```

This is the single current multiplayer path. Optimizer should not create a
second multiplayer CurvyTron env for speed. If Optimizer profiles multiplayer,
it should wrap or call these shared modules and report the surface as
metadata-only unless Environment promotes a learned observation or visual
contract. Do not compare these numbers against strict `VectorTrainerEnv1v1NoBonus`
numbers without saying that the surfaces are different.

Please do not generalize either surface to full CurvyTron, bonuses, 3P/4P,
source-faithful visual LightZero, or learning quality. Every profile should
name:

- `env_impl_id`
- `ruleset_id` and `ruleset_hash`
- `source_claim_id` or a concrete source claim reference
- feature flags: player count, bonus setting, public/strict surface, and
  whether the run is metadata-only, learned-observation, visual-debug, or
  source-visual
- event mode
- `decision_ms` as the wrapper decision window, not a native source tick
- body/event/timer capacities
- reset seed and reset source
- final-observation policy
- reset provenance fields when present: `random_tape_source`,
  `random_tape_length`, `rng_impl_id`, and `source_fixture_ref`
- whether policy, search, replay, and reset are inside the timed number

If a report cannot emit one of these fields yet, mark it as missing in the
report/readout and do not use that number as comparison-valid evidence.

Environment/RAM owns whether a fast path is faithful to source CurvyTron.
Optimizer owns timing and bottlenecks. Coach owns checkpoint and learning
quality.

## Fidelity Status

We are not done with full environment fidelity.

What is strong enough for current optimizer work:

- strict public `VectorTrainerEnv1v1NoBonus` reset-to-terminal parity for the
  named long 1v1/no-bonus fixture;
- trainer observations `curvyzero_egocentric_rays/v0` as `float32[106]`;
- terminal metadata, final observations, sparse final rewards, truncation
  flags, and replay-v0 plumbing for the strict 1v1/no-bonus path;
- source-backed proof tools for movement, borders, trail/body collisions,
  PrintManager behavior, selected lifecycle fixtures, and narrow bonus facts.
- metadata-only multiplayer plumbing for setup/profiling scouts:
  `VectorMultiplayerEnv`, `MetadataOnlyMultiplayerEgoWrapper`, and
  widened multiplayer replay metadata. This is useful for measuring wrapper and
  recorder cost, not for claiming trainer-ready multiplayer.

What is not done:

- full CurvyTron fidelity;
- production 3P/4P replay writer/reader;
- learned 3P/4P trainer observation schema;
- natural public multiplayer reset/warmup/autoreset parity;
- broad match lifecycle through the public multiplayer env;
- bonuses in the fast/vector runtime;
- source-faithful visual renderer;
- installed-runtime CurvyTron LightZero config/import smoke and real CurvyTron
  LightZero training. A thin local registered wrapper exists, but it has not
  been proven inside the installed LightZero/DI-engine runtime.

Update: the installed no-train CurvyZero LightZero config/import smoke now
passes on Modal for the scalar/ray wrapper. That proves config/import/reset/step
plumbing for `curvyzero_v0_lightzero`; it does not prove training quality or a
source-faithful visual tensor.

Update: the installed no-train debug visual LightZero config/import smoke also
passes on Modal for `curvyzero_debug_visual_tensor_lightzero`. That proves
config/import/env-factory reset/step plumbing with a real `BaseEnvTimestep`,
LightZero-facing env payload `float32[1,64,64]`, model stack `(4,64,64)`,
action space `3`, and no ALE identity. It still proves only debug visual
plumbing, not source-fidelity pixels or learning.

Plain read: the strict 1v1/no-bonus path is valid for proof-wrapper profiling,
and the source-backed path is valid for oracle-adjacent adapter/observation
timing. The metadata-only multiplayer path is valid for wrapper/replay/setup
profiling. None of these is a signal that the whole environment is finished.

This is a parallel handoff, not a phase change. Optimizer can measure the
strict surfaces above while Environment/RAM keeps moving source behavior into
the fast runtime. Do not read this as "fidelity is done"; read it as "these two
surfaces are stable enough to time without redefining game truth."

## Separation Of Work

Environment/RAM should do:

- source-read CurvyTron behavior;
- add JS/source oracle fixtures and Python parity;
- decide when fast/vector behavior may claim source fidelity;
- decide when visual tensors may claim source fidelity;
- define observation, reward, action, reset, terminal, replay, and sidecar
  contracts;
- promote 3P/4P and bonus mechanics only after source-backed proof.

Optimizer should do:

- own the debug visual smoke/profiler and LightZero visual adapter plumbing;
- time the actor loop and whole training-loop pieces;
- separate env step, observation packing, policy/search, reset/autoreset, and
  replay costs;
- decide whether speed rewrites are justified by real bottlenecks;
- keep Modal/process/GPU boundaries explicit;
- report latency and throughput with clear denominators.

Coach should do:

- LightZero/Pong reproduction and eval quality;
- checkpoint promotion;
- learning signal, survival metrics, scorecards, and policy quality.

## Critique

The current optimizer separation is mostly right. The main risk is language:
speed numbers can sound more general than they are. A report that says
`VectorTrainerEnv1v1NoBonus` is fast, or that a source-backed adapter path is
fast, must not become "CurvyTron is fast" unless the source claim, feature
flags, timed surface, and denominator are explicit.

The second risk is duplicated adapter work. Environment should own the
single-ego LightZero adapter contract and source semantics. Optimizer can use
that contract for timing, but should not create a separate semantic wrapper that
drifts from the source-backed contract. For multiplayer, the same rule applies:
use `VectorMultiplayerEnv` plus `MetadataOnlyMultiplayerEgoWrapper`
instead of inventing a second CurvyTron multiplayer env.

The visual tensor boundary is now explicit:
[optimizer_visual_tensor_handoff_2026-05-10.md](optimizer_visual_tensor_handoff_2026-05-10.md).
Environment owns tensor truth. Optimizer owns the visual smoke/profiler,
LightZero adapter plumbing, tensor timing, and implementation speed choices.

The third risk is over-optimizing the wrong bucket. Current optimizer docs say
ray/observation work is the visible cost. That is useful, but the next rewrite
should still be checked against the whole loop because MuZero search and
collector/evaluator time can dominate once real policy/search is included.

## Source Links

- Optimizer boundary:
  [docs/working/optimizer/lane_contract_2026-05-10.md](../optimizer/lane_contract_2026-05-10.md)
- Optimizer runtime read:
  [docs/working/optimizer/runtime_verdict_2026-05-10.md](../optimizer/runtime_verdict_2026-05-10.md)
- Full-iteration status:
  [docs/working/optimizer/full_iteration_status_2026-05-10.md](../optimizer/full_iteration_status_2026-05-10.md)
- Environment claim tracker:
  [docs/working/environment/coverage_tracker.md](coverage_tracker.md)
- Multiplayer gap list:
  [docs/working/environment/multiplayer_env_gap_targets_2026-05-10.md](multiplayer_env_gap_targets_2026-05-10.md)
- LightZero env requirements:
  [docs/working/environment/lightzero_env_requirements_2026-05-10.md](lightzero_env_requirements_2026-05-10.md)
