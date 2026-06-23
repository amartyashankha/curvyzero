# Next Phase Optimizer Synthesis

Date: 2026-05-23

Status: documentation synthesis only. No code, Modal run, live Coach run,
checkpoint, eval, GIF, tournament artifact, or Modal volume was touched.

## Current Known Facts

The trusted Coach lane remains stock LightZero `train_muzero`:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
env_variant=source_state_fixed_opponent
policy observation: browser_lines + simple_symbols -> cpu_oracle -> [4,64,64]
```

The active optimizer lane is profile-only:

```text
HybridCompactBatch
-> CompactSearchServiceV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
```

Current source and proof surfaces:

```text
src/curvyzero/training/compact_search_service.py
tests/test_compact_search_replay_contract.py
tests/test_source_state_hybrid_observation_profile.py
tests/test_source_state_batched_observation_boundary_profile.py
docs/working/optimizer/batched_gpu_full_loop_reorientation_2026-05-20/current_state_audit_20260523.md
```

Fresh H100 compact service grid, B512/A16, sim16/sim32:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-direct-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-dense-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-service-tax-20260523b
artifacts/local/curvytron_hybrid_observation_profile_results/opt-compact-service-compare-mock-20260523b

sim16 direct_ctree_gpu_latent: 4,353 steps/sec, probe 7.734s
sim16 dense_torch_mcts:       5,380 steps/sec, probe 3.687s
sim16 service_tax_probe:      7,487 steps/sec, probe 2.041s
sim16 mock_search_service:    6,955 steps/sec, probe 0.615s

sim32 direct_ctree_gpu_latent: 2,717 steps/sec, probe 14.096s
sim32 dense_torch_mcts:       2,580 steps/sec, probe 9.230s
sim32 service_tax_probe:      6,847 steps/sec, probe 3.478s
sim32 mock_search_service:    5,696 steps/sec, probe 0.519s
```

The latest render/refresh work helped but is bounded. On the current H100
B1024/P2/loop96 denominator, refresh-off is only about `1.6-1.7x` over
refresh-on. Raw GPU drawing is tiny; the expensive leaves are compact state
handoff, delta packing, renderer H2D/update, public packaging, and search.

The opt-in parent `persistent_compact_render_state_buffer` is diagnostic only.
It removes renderer production-to-compact conversion, but pays for full
parent-side trail writes. Borrowed single-actor render state is still faster on
the measured rows.

## What Is Validated

- Compact search results can be validated into `CompactSearchResultV1` and
  materialized as `CompactReplayIndexRowsV1`.
- Compact replay rows keep identity sidecars aligned: env row, player,
  `policy_env_id`, compact root row, selected action, visit policy, and root
  value.
- RND latest-frame extraction is attached to the same compact root records.
- The combined RND/terminal canary runs the real `CurvyRNDRewardModel`
  collect/train/estimate path, moves predictor weights, keeps the target hash
  fixed, and uses `final_observation` for terminal rows.
- Compact index rows compose into the repo learner-facing sample batch and
  match the trusted immediate replay path under the same sample seed.
- Opt-in stock LightZero `GameSegment`/`MuZeroGameBuffer` target-hook parity is
  green when `lzero` is available locally.
- Opt-in public stock LightZero `MuZeroGameBuffer.sample(...)` parity is green
  when `lzero` is available locally. The canary maps sampled transition ids
  back to compact row ids and compares sampled observations, target policies,
  reward targets, and zero-model value targets.
- The first real-backend local closed-loop smoke is green. It uses the direct
  CTree compact service, applies the service-selected actions to the next
  hybrid env step, and compares compact deferred rows plus learner sample
  batches against the trusted immediate path.

Recent local validation references:

```text
compact/hybrid/RND focused pytest set: 180 passed
tests/test_compact_search_replay_contract.py: 12 passed after adversarial identity gates
tests/test_compact_search_replay_contract.py: 15 passed after public sample parity
latest compact Torch service + MCTX legality + compact replay + boundary profile: 144 passed
latest hybrid observation profile + grid builder: 59 passed
real direct CTree compact-service closed-loop smoke: 1 passed focused
compact Torch service closed-loop smoke: 1 passed focused
```

## What Is Still Profile-Only

These are not Coach launch advice:

- `direct_ctree_gpu_latent`
- `service_tax_probe`
- `mock_search_service`
- dense Torch compact replay/search rows
- borrowed single-actor render state rows
- parent compact render-state buffer rows
- closed-loop deferred/overlap payload canaries

The compact lane still lacks a matched stock-vs-candidate full-loop smoke with
the compact service backend. It also does not yet prove RND cadence at full
stock trainer cadence.

The first fixed-shape Torch service now exists, but it is deliberately narrow:

```text
src/curvyzero/training/compact_torch_search_service.py
tests/test_compact_torch_search_service.py
```

It records compile eligibility/profile-only telemetry, tests tiny select/backup
helpers, and includes `CompactTorchSearchServiceV1`, which runs one compact
Torch model/search pass and validates into `CompactSearchResultV1`. Local
validation passed: ruff clean, compact Torch tests, one closed-loop replay smoke,
and profile wiring tests. The profile-only array-ceiling surface has a
`compact_torch_search_service` mode that calls this service directly. It does
not change the trusted Coach lane and is not a trainer-facing backend.

Remote H100 status:

```text
B512/A16/sim16/steps60/warmup15, compact replay proof on.

direct_ctree_gpu_latent:       4,966 steps/sec, probe 5.702s
service_tax_probe:            5,853 steps/sec, probe 2.857s
compact_torch_search_service: 5,140 steps/sec, probe 5.867s
compact_torch timing split:   5,575 steps/sec, probe 5.098s
  initial model: 0.271s
  tree/recurrent loop: 4.250s

no-noise pair:
direct_ctree_gpu_latent:       3,955 steps/sec, probe 6.944s
compact_torch_search_service: 5,704 steps/sec, probe 5.077s
```

Plain read: the compact Torch service is wired and replay-valid in the Modal
profile. It is not the 5x/10x answer yet. The next speed wall inside that lane is
the eager Python/Torch tree plus recurrent loop.

## Current Amdahl Bottleneck Model

The latest model is:

```text
1. Compact replay/index rows are cheap enough.
2. Raw GPU frame drawing is not the wall.
3. Observation refresh/handoff is real but bounded at roughly 1.6-1.7x.
4. At sim32, search becomes a first-class wall.
5. The largest current profile-only opportunity is repeated search/control/list
   movement around real model calls, plus compact state/search-input ownership.
```

Common-telemetry split:

```text
sim16 service_tax / direct: 1.72x measured, 3.79x raw probe
sim32 service_tax / direct: 2.52x measured, 4.05x raw probe

sim16 dense_torch / direct: 1.24x measured, 2.10x raw probe
sim32 dense_torch / direct: 0.95x measured, 1.53x raw probe
```

Plain read: tiny action payloads are fine once per env step. Tiny payloads are
bad when forced through CPU/GPU sync and Python/list object paths inside every
MCTS simulation. Dense Torch is not a promotion candidate yet because it loses
the sim32 measured row and is not LightZero CTree semantics. The new compact
Torch service proves the boundary can run remotely, but eager Python-per-sim
Torch is still too slow. The useful signal remains service-tax/MCTX-style: a
real compiled or array-native search service can plausibly move 2x on
search-heavy shapes before larger architecture work.

## External Architecture Read

Darwin's read-only research pass agrees with the local profile:

```text
MCTX-style systems win by keeping search as JIT-compiled batched array work.
PufferLib/EnvPool/Sample Factory-style systems win by owning contiguous buffers
and keeping env, search, replay, and learner components independently busy.
Batch-MCTS work points at batching neural inference instead of doing one tiny
GPU call per tree expansion.
```

Practical translation for CurvyTron:

```text
Do not keep polishing wrapper-only dense Torch or fake-search rows.
Do build the compact search boundary and closed-loop replay proof first.
Then try one real fixed-shape backend behind that boundary:
  1. Torch device-tree if it can beat direct CTree on sim16 and not regress sim32.
  2. MCTX/JAX as a sidecar comparator with real compact roots, not trainer advice.
  3. Larger multi-producer search service only after identity/RND/sampler gates pass.
```

The external examples are guidance, not promotion proof. Any backend still must
emit `selected_action`, `visit_policy`, and `root_value` through
`CompactSearchResultV1` and pass compact replay/RND/player identity gates.

## Next Three Experiments

1. Run the first matched stock-vs-candidate compact-service smoke once the new
   candidate backend exists.
   The public sampler edge and the real direct-CTree closed-loop compact-service
   smoke are now covered locally. The next proof must show the candidate compact
   backend consumes the same roots and preserves selected actions, visit
   policies, root values, terminal/final observations, RND attachments, and
   player identity over a small closed loop.

2. Replace or compile the eager compact Torch tree body, or demote it.
   The first service backend now runs, but the timing split says the hot work is
   the Python/Torch tree/recurrent loop. The next backend attempt should be a
   real compiled/fixed-shape search body or an MCTX/JAX comparator, compared
   against the same B512/A16 common telemetry: `direct_ctree_gpu_latent`,
   `service_tax_probe`, and `mock_search_service`.

3. Test compact state ownership without parent full-copy buffers.
   Try one small canary in this order: env-emitted render delta/event log from
   append/reset sites, actor-owned compact x/y state borrowed by the renderer,
   or renderer direct consumption of production `visual_trail_pos` as a small
   falsifier. Do not promote the parent compact render-state buffer unless a new
   same-denominator profile overturns the current mixed result.

## Risks And Validation Gates

- Identity risk: search one root and train another. Gate with swapped-player,
  duplicate/missing root id, stale `policy_env_id`, and non-prefix root tests.
- RND risk: intrinsic reward attaches to the wrong record or cadence. Gate with
  deterministic RND latest-frame/reward attachment and a later full trainer
  cadence check.
- Terminal risk: terminal replay uses latest live observation instead of
  `final_observation`. Keep terminal/autoreset rows in every promotion proof.
- Search-service risk: faster rows silently change MuZero semantics. Gate every
  backend through `CompactSearchResultV1` and stock LightZero target/sample
  comparisons before trainer-facing claims.
- Amdahl risk: optimizing a stale leaf. Every speed claim must state its
  denominator: Coach training, stock full-loop profile, or profile-only compact
  boundary.
- Compiled-backend risk: a Torch backend can be fast and wrong. Gate active-root
  order, legal-mask polarity, PUCT/backup semantics, seeded root noise,
  recurrent action/latent slots, stale tensor reuse, compile-cache signatures,
  and timing that includes CUDA sync/readback.

## Coach-Facing Warning

Do not tell Coach to launch compact search, borrowed render state, dense Torch
search, service-tax, mock-search, or parent compact-buffer modes. They are
optimizer probes, not training defaults.

Coach should keep using stock LightZero `train_muzero` with
`source_state_fixed_opponent` until the compact service path passes sampler
parity, RND/player-perspective gates, and a matched stock-vs-candidate
full-loop smoke. Fast profile-only roots/sec is evidence for the optimizer, not
a promise about learning speed or run quality.
