# Current State Audit, 2026-05-23

Status: optimizer working memory. This file records the current map so older
docs do not drag the optimizer back into stale lanes.

## Trusted Training Lane

The trusted Coach training lane is stock LightZero `train_muzero`, not the
optimizer compact path:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
env_variant=source_state_fixed_opponent
policy observation: browser_lines + simple_symbols -> cpu_oracle -> [4,64,64]
learner seat: random_per_episode
```

Current broad launch defaults live in:

```text
docs/working/training/r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md
src/curvyzero/contracts/curvytron.py
```

The optimizer must not treat profile-only compact rows as Coach launch advice.

## Profile-Only Optimizer Lane

The active optimizer lane is compact/MCTX/search-dataflow:

```text
HybridCompactBatch
-> compact env/observation state
-> CompactRootBatchV1
-> compact/fixed-shape search or MCTX
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

The compact replay/RND/terminal proof is now strong locally:

```text
identity sidecars align
RND latest-frame extraction aligns
CurvyRNDRewardModel collect/train/estimate runs
RND predictor moves
RND target hash stays fixed
positive intrinsic reward changes the target reward
terminal rows use final_observation instead of latest live observation
focused compact/hybrid/RND pytest set: 180 passed
repo learner-facing sample batches match immediate rows
stock LightZero GameSegment/MuZeroGameBuffer target hooks match materialized rows
stock LightZero public MuZeroGameBuffer.sample matches materialized rows in the
opt-in local canary when lzero is installed
real direct CTree compact service can drive a tiny hybrid env step and produce
compact deferred rows/sample batches matching the trusted immediate path
```

Still not trainer-facing:

```text
matched stock-vs-candidate full-loop smoke
initial immutable-checkpoint JAX shadow realism exists for profile-only MCTX
still missing search-impact parity, terminal/inactive stress, and a Coach/full-loop backend
```

2026-05-23 fixed-shape Torch helper status:

```text
src/curvyzero/training/compact_torch_search_service.py
tests/test_compact_torch_search_service.py
```

This helper is profile-only. It gives the optimizer a small place to define
fixed-shape compile preconditions, profile-only telemetry labels, and tiny
Torch select/backup helper tests without importing the large Modal profile
module. `CompactTorchSearchServiceV1` now also exists as the first local
profile-only candidate service: it selects active roots, runs one model/search
pass, and validates into `CompactSearchResultV1`. Focused local validation:
ruff clean, compact Torch tests, closed-loop Torch service smoke, and profile
wiring tests. The Modal profile surface now has explicit
`compact_torch_search_service` mode that calls the service directly; it is still
not Coach training advice.

## Demoted Lanes

Ignore these as primary optimizer lanes unless a newer same-denominator profile
re-promotes them:

```text
old custom two-seat-selfplay trainer
body_circles_fast / fast_gray64_direct / browser_sprites as policy surfaces
scalar policy_observation_backend=jax_gpu
flat-A3 as main path
dense Torch MCTS polishing
renderer-only/raw draw work
CPU-count scaling
```

Plain reason:

```text
They were useful falsifiers or historical controls. Current evidence points at
state/search-input/search-service dataflow and, at higher simulation counts,
search itself.
```

## Top Open Optimizer Tasks

1. Prove the learner sampler edge: compact replay/index/RND/terminal rows must
   feed learner-facing samples matching the trusted immediate replay path over a
   multi-record closed loop. Repo-local sample batch, stock LightZero target
   hooks, and opt-in public `MuZeroGameBuffer.sample(...)` parity are now green.
   The first real direct-CTree compact-service closed-loop smoke is now green.
   The next trainer-facing proof is the same shape with the new candidate
   compact backend once it exists.
2. Keep one real backend behind `CompactSearchServiceV1`, with one service call
   owning the probe and returning compact arrays without double-running search.
3. Split and attack the repeated refresh-on path: game runtime, public
   packaging, production-to-compact, delta pack, renderer H2D/update, resident
   stack ownership, search, and residual glue.
4. Prototype a resident/lower-copy compact observation loop if the refresh-on
   ceiling stays hot in fresh rows.
5. Run matched stock-vs-candidate full-loop smokes only after the above parity
   gates are green.

## Latest Compact Search Grid

Fresh same-denominator H100 profile-only rows:

```text
H100, B512/A16, steps=80, warmup=20, compact replay proof on,
host_uint8_pinned input, no scalar timestep materialization.
```

Results:

```text
sim16 direct_ctree_gpu_latent: 4,353 steps/sec, probe 7.734s
sim16 dense_torch_mcts:       5,380 steps/sec, probe 3.687s
sim16 service_tax_probe:      7,487 steps/sec, probe 2.041s
sim16 mock_search_service:    6,955 steps/sec, probe 0.615s

sim32 direct_ctree_gpu_latent: 2,717 steps/sec, probe 14.096s
sim32 dense_torch_mcts:       2,580 steps/sec, probe 9.230s
sim32 service_tax_probe:      6,847 steps/sec, probe 3.478s
sim32 mock_search_service:    5,696 steps/sec, probe 0.519s
```

Decision:

```text
Dense Torch is not promoted. It helps at sim16 but loses the measured sim32 row
and is not LightZero CTree semantics.

The real opportunity is a compact search-service/dataflow rewrite. Service-tax
is 1.72x measured over direct at sim16 and 2.52x at sim32 while still paying
real model calls.
```

MCTX/JAX sidecar refresh:

```text
Profile-only H100 compact visual closed loop:
  sim16: 27.6k active roots/sec
  sim32: 22.2k active roots/sec

This older row is superseded by `real_mctx_shadow_bridge_20260523i.md` for the
current real-checkpoint MCTX result. It was useful architecture evidence, but
the current row to quote is the immutable-checkpoint shadow bridge, not this toy
JAX model/search row.

The MCTX benchmark now emits a comparable `compact_search_service_profile` row
with explicit profile-only labels. It is useful for optimizer tables and still
must not be treated as a Coach training backend.
```

Artifacts:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-direct-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-dense-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-service-tax-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-mock-20260523b
```

2026-05-23 compact Torch service smoke:

```text
H100, B512/A16, sim16, 60 measured, 15 warmup, compact replay proof on,
host_uint8, no scalar timestep materialization.

direct_ctree_gpu_latent:      4,966 steps/sec, probe 5.702s
service_tax_probe:           5,853 steps/sec, probe 2.857s
compact_torch_search_service 5,140 steps/sec, probe 5.867s
compact_torch timing split:  5,575 steps/sec, probe 5.098s
  initial model: 0.271s
  tree/recurrent loop: 4.250s

no-noise pair:
direct_ctree_gpu_latent:      3,955 steps/sec, probe 6.944s
compact_torch_search_service 5,704 steps/sec, probe 5.077s
  initial model: 0.271s
  tree/recurrent loop: 4.235s
```

Read:

```text
The compact Torch service is real enough to profile and replay-proof remotely.
It does not yet deliver the big win. The time is now visibly inside the eager
Torch tree/recurrent loop, not input transfer or initial inference. Direct CTree
varies enough that the honest claim is "roughly direct speed to about 1.4x
faster on this row," not a 5x/10x breakthrough.

The compact Torch service currently accepts only `host_uint8` profile input mode
because it consumes `CompactRootBatchV1` host uint8 observations directly. Other
input modes would be mislabeled and now fail closed.
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

## Latest Render-State Read

The opt-in `persistent_compact_render_state_buffer` is a diagnostic, not the
next promoted speed setting.

Plain reason:

```text
It makes renderer production-to-compact cost report zero, but it writes the
full visual trail state from actor/env state into parent compact buffers. That
moves work into actor_render_state_write_sec instead of deleting it.
```

Fresh same-denominator read:

```text
sim16 copied production state: 26.3k roots/sec
sim16 parent compact buffer:   35.8k roots/sec
sim16 borrowed production:     48.6k roots/sec

sim32 copied production state: 29.5k roots/sec
sim32 parent compact buffer:   25.3k roots/sec
sim32 borrowed production:     37.9k roots/sec
```

What this means:

```text
Do not keep optimizing a parent-side full-copy compact buffer as the main lane.
The next real observation-state architecture must avoid both conversion and
parent full copying.
```

Best next render-state canaries, in order:

1. Env-emitted render delta/event log from append/reset sites.
2. Actor-owned compact x/y render state that the renderer can borrow directly.
3. Renderer direct consumption of production `visual_trail_pos` as a small
   falsifier only.

This is still second to the search/dataflow parity lane unless a fresh profile
re-promotes observation handoff as the largest wall.

## Architecture Priority

Carver's 2026-05-23 sidecar critique agrees with the main read:

1. Make `CompactSearchServiceV1` trainer-facing enough to be trusted at the
   replay/sampler edge. Expected profile upside if the service-tax/mock ceiling
   transfers: about `1.5-2.5x`, not standalone 10x.
2. Build a Puffer-style compact owner, but not as a parent full-copy buffer:
   preallocated row/player buffers for actions, rewards, dones, masks,
   resident stacks, replay indices, plus either env-emitted render deltas or
   actor-owned compact render state. Expected upside with the compact service:
   about `1.5-3x` if it removes the current handoff wall.
3. Design a many-roots batched search/inference service after the compact
   contract is proven. Plausible larger upside is `3-8x`, with 10x only if
   compact env/replay ownership also lands and batch fill stays high.

Do not start with a giant async service before the compact replay/sampler proof
is green. That would make fast numbers easier to get and easier to distrust.

## Validation Priority

Gibbs's 2026-05-23 audit says the compact path stays profile-only until these
proofs land:

1. One hybrid closed-loop oracle test that materializes compact replay rows and
   compares target rows against the trusted object/direct oracle, including a
   terminal/autoreset row.
2. Adversarial identity tests for swapped player, duplicate/missing root ids,
   and stale `policy_env_id`, failing before replay writing.
3. Deterministic RND attachment parity: prove any RND reward change attaches to
   the intended `env_row/player/root_index`, not just to whatever tensor row
   happened to be in that slot.

RND cadence is still not fully proven at the actual LightZero trainer cadence.
Current tests prove the model mechanics and compact row attachment, not that
large training runs call RND updates at the ideal frequency.

Meitner's compiled-backend critique adds the next set of "fast but wrong"
guards before a Torch backend can be trusted:

```text
active-root order must match compacted root ids
legal masks must reject illegal visit mass and single-action drift
PUCT/reward/discount backup values must match direct CTree toy cases
noise-off rows must be exact; noise-on rows must be seeded and legal
recurrent action shape and latent slots need a fake-model test
same-shape repeated calls must prove fresh observations, not stale tensor reuse
compile cache signatures must include shape, sims, dtype, device, and model
timing must include compile, warm steady state, model calls, sync, and readback
```

## Stale Doc Warnings

```text
leaderboard_to_training_2026-05-13 has archival files with "current" headings.
Use explicit newer current docs instead.

optimizer_speed_axis.md has older L4/C256 advice. It remains useful only as
training-default evidence, not as the active optimizer map.

Older test counts are superseded by the current compact/hybrid/RND set:
Current local validation:
  compact Torch service + MCTX legality + compact replay + boundary profile: 144 passed
  hybrid observation profile + grid builder: 59 passed
```
