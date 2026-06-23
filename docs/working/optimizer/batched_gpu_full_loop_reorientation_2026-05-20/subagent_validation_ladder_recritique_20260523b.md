# Subagent Validation Ladder Recritique - 2026-05-23b

Scope: validation/test critique only. I reviewed the compact search service,
compact rollout slab, hybrid observation profile, batched observation boundary
profile, RND exploration-bonus hooks, and stock `train_muzero` entrypoint tests.
I did not touch live training runs and did not run Modal jobs.

## Bottom Line

The local validation story is no longer empty. The repo now has strong
contract-level tests for compact root/search/replay identity, legal masks,
selected-action feedback, terminal final observations, RND latest-frame
attachment, and even local real-CTree/compact-Torch closed-loop profile proofs.

But this is still not a Coach-training proof. The trusted Coach lane remains
stock LightZero `train_muzero` on `source_state_fixed_opponent`, and the current
trainer code explicitly blocks profile env managers and non-stock collect-search
backends outside `mode="profile"` (`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5474`,
`src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5522`).
That is good safety, but it means profile speedups cannot be promoted as Coach
speedups until they pass a capped original-loop smoke through the actual
trainer entrypoint.

The ladder below is the line between useful optimizer research and optimizing
bullshit.

## 1. Promotion Invariants

### L0. Original-Loop Boundary

Before any big architecture move is promoted toward Coach training, it must say
which loop it is proving:

- `mode=train`: stock training lane, calling `lzero.entry.train_muzero` or, for
  RND, `lzero.entry.train_muzero_with_reward_model`.
- `mode=profile`: profiler lane, allowed to install hooks, stop early, replace
  env managers, or use non-stock search backends, but not allowed to claim
  Coach training speed.
- `profile_only=true` rows must be promotion-ineligible by default.

Hard invariant: a candidate speed row must include `called_train_muzero`,
`trainer_entrypoint`, backend kind, fallback counts, profile/training label,
root-noise metadata, action-feedback proof, replay/sample proof, and RND mode.
Missing fields mean "not promotion eligible", not "unknown but probably fine".

### L1. Root Identity And Perspective

The same `root_index`, `env_row`, `player`, `policy_env_id`, `active_root_mask`,
`done_root`, `to_play=-1`, controlled-player perspective, and legal mask must
survive:

```text
HybridCompactBatch
-> CompactRootBatchV1
-> CompactSearchResultV1
-> CompactReplayIndexRowsV1
-> materialized target rows
-> learner-visible samples
```

Compaction must be by stable sidecars, not by incidental array position.
Non-prefix active roots and non-identity `policy_env_id` values are required
test cases.

### L2. Legal Search Semantics

Selected actions must be legal. Visit policy and raw visit counts must be
finite, nonnegative, sum correctly on active rows, and assign zero mass to
illegal actions. Empty masks must mean inactive/rejected roots, never "choose
action 0 because the tensor had a zero there".

For direct CTree and Torch-like candidates, no-noise rows must be exact against
a reference. Noisy rows need explicit seed/noise/tie metadata and a statistical
gate. No candidate may hide a fallback to stock or a public Python row path in
a promoted speed result.

### L3. Action Feedback

The selected search action at record `k` must be exactly the `joint_action`
that steps the environment into record `k+1`. This must be checked after the
real service/probe returns, not only in a synthetic replay builder.

Fast fail: if `selected_action` differs from next-step `joint_action` for any
`env_row/player`, the path is invalid for training.

### L4. Replay, Samples, And Payload Visibility

Replay rows must be hidden until all replay-critical payload is attached and
identity-checked. A learner must never sample a partial compact row.

Index rows must materialize to the same target rows as the immediate trusted
path, and learner-facing sample batches must match on observation, action,
mask, policy target, root value, reward, terminal flags, next observation,
`env_row`, `player`, `policy_row`, and `to_play`.

### L5. Observation And Terminal Semantics

The optimizer must not choose a new observation contract by accident. The
training-facing policy view remains the pinned policy observation surface:
row-major player views, correct controlled-player perspective, uint8 direct64
or normalized float32 as declared, and avatar-color semantics unchanged.

Terminal rows must use `final_observation` from the terminal transition, never
the autoreset frame. Terminal roots must not be searched as live.

### L6. RND Exploration Bonus

RND is not just a speed sidecar. If enabled, the trainer entrypoint must switch
to the reward-model entrypoint, and metrics must prove:

- `collect_data`, `train_with_data`, and `estimate` ran;
- `train_cnt_rnd > 0`;
- predictor weights changed;
- target weights stayed fixed;
- `rnd_meter_v0` left target rewards unchanged;
- positive `rnd_replay_target_v0` changed target rewards and is labelled as an
  objective change;
- the latest gray64 frame comes from the same row/player/root as the replay
  target.

### L7. Capped Full-Loop Gate

Before Coach-facing promotion, run a tiny matched original-loop smoke:

- same seed/config for stock control and candidate;
- no background eval/GIF sidecars or checkpoint churn unless explicitly part of
  the measurement;
- cap after one learner train call or an equally tiny bound;
- require `called_train_muzero=true`;
- require no fallback;
- require replay/sample/RND digests;
- require the speed summary to distinguish stock, profile-only, and
  train-facing paths.

This is the first gate that can justify Coach advice. Everything before it is
only contract work.

## 2. Existing Tests

### Compact Search Service And Replay

Code:

- `src/curvyzero/training/compact_search_service.py:23` defines
  `CompactSearchServiceV1`.
- `src/curvyzero/training/compact_search_service.py:63` validates search arrays
  into `CompactSearchResultV1`.
- `src/curvyzero/training/compact_search_service.py:163` checks two-phase
  action/replay payload identity.
- `src/curvyzero/training/compact_search_service.py:180` hides replay payloads
  behind `CompactSearchPayloadGateV1`.
- `src/curvyzero/training/compact_policy_row_bridge.py:124`,
  `src/curvyzero/training/compact_policy_row_bridge.py:254`,
  `src/curvyzero/training/compact_policy_row_bridge.py:424`, and
  `src/curvyzero/training/compact_policy_row_bridge.py:589` define the root,
  search, index-row, and materialization contracts.

Tests:

- `tests/test_compact_search_replay_contract.py:166` round-trips compact
  service roots/search/replay chunks with identity sidecars.
- `tests/test_compact_search_replay_contract.py:240`,
  `tests/test_compact_search_replay_contract.py:288`,
  `tests/test_compact_search_replay_contract.py:339`, and
  `tests/test_compact_search_replay_contract.py:387` cover two-phase payloads
  and visibility gating.
- `tests/test_compact_search_replay_contract.py:570` proves index rows skip
  observation materialization but materialize back to immediate rows.
- `tests/test_compact_search_replay_contract.py:707` rejects illegal actions,
  illegal visit mass, stale sidecars, duplicate roots, and stale masks.
- `tests/test_compact_search_replay_contract.py:885`,
  `tests/test_compact_search_replay_contract.py:991`, and
  `tests/test_compact_search_replay_contract.py:1073` cover non-prefix active
  roots and deferred payload parity.
- `tests/test_compact_search_replay_contract.py:1185` proves materialized
  sample batches match immediate rows.
- Optional local LightZero tests at
  `tests/test_compact_search_replay_contract.py:1284` and
  `tests/test_compact_search_replay_contract.py:1449` check stock
  `MuZeroGameBuffer` target/sample behavior when `lzero` is installed.

### Compact Rollout Slab

Code:

- `src/curvyzero/training/compact_rollout_slab.py:48` is explicitly
  profile-only and does not call `train_muzero`.
- `src/curvyzero/training/compact_rollout_slab.py:117` commits the previous
  search result only if the next batch applied the staged actions.
- `src/curvyzero/training/compact_rollout_slab.py:155` maps active-root actions
  back to dense joint actions and rejects illegal selected actions.

Tests:

- `tests/test_compact_search_replay_contract.py:442` stages actions and commits
  previous index rows.
- `tests/test_compact_search_replay_contract.py:493` rejects a next batch that
  ignored staged selected actions.
- `tests/test_compact_search_replay_contract.py:535` rejects illegal actions in
  dense joint-action mapping.
- `tests/test_source_state_hybrid_observation_profile.py:79` plugs the slab
  into the hybrid profile manager and proves the committed index rows line up.

### Hybrid Observation Profile

Code:

- `src/curvyzero/training/source_state_hybrid_observation_profile.py:581`
  defines the profile manager.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py:584`
  marks it `profile_only=true`, `calls_train_muzero=false`, and not stock
  integrated.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py:1358`
  runs the profile harness.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py:1377`
  requires a compact batch consumer for compact replay proof.
- `src/curvyzero/training/source_state_hybrid_observation_profile.py:2084`
  builds the compact replay proof telemetry.

Tests:

- `tests/test_source_state_hybrid_observation_profile.py:28` through
  `tests/test_source_state_hybrid_observation_profile.py:153` cover deterministic
  shape/counts, scalar row ids, metadata, and timing fields.
- `tests/test_source_state_hybrid_observation_profile.py:182` through
  `tests/test_source_state_hybrid_observation_profile.py:243` cover native actor
  buffers and terminal/autoreset counts.
- `tests/test_source_state_hybrid_observation_profile.py:264` through
  `tests/test_source_state_hybrid_observation_profile.py:808` cover
  renderer-backed row-major order, uint8 stack storage, final observation before
  autoreset, device-only/latest behavior, and persistent compact render state.
- `tests/test_source_state_hybrid_observation_profile.py:831` through
  `tests/test_source_state_hybrid_observation_profile.py:920` prove probes run
  before scalar materialization and receive compact sidecars.
- `tests/test_source_state_hybrid_observation_profile.py:1020` proves compact
  RND latest-frame extraction without scalar timesteps.
- `tests/test_source_state_hybrid_observation_profile.py:1050`,
  `tests/test_source_state_hybrid_observation_profile.py:1105`, and
  `tests/test_source_state_hybrid_observation_profile.py:1149` cover compact
  service replay proof calls, array-ceiling arrays, and warmup accounting.

### Batched Boundary, Direct CTree, And Compact Torch

Code:

- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:131`
  is a profile-only Modal boundary.
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1691`,
  `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:1970`,
  and `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2502`
  label surface, manager, and hybrid canaries profile-only.
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2254`
  requires compact replay proof to be attached to an actual direct/array
  compact-search source.
- `src/curvyzero/training/compact_torch_search_service.py:21` and
  `src/curvyzero/training/compact_torch_search_service.py:126` label compact
  Torch as profile-only, not LightZero CTree, and not trainer-ready.

Tests:

- `tests/test_source_state_batched_observation_boundary_profile.py:83` through
  `tests/test_source_state_batched_observation_boundary_profile.py:220` cover
  row-major rendering and persistent compact state basics.
- `tests/test_source_state_batched_observation_boundary_profile.py:281` through
  `tests/test_source_state_batched_observation_boundary_profile.py:708` cover
  boundary config modes, direct64, collect-forward, initial-inference,
  array-ceiling, service-tax, compact Torch service, and MCTS arrays-boundary
  validation.
- `tests/test_source_state_batched_observation_boundary_profile.py:793` through
  `tests/test_source_state_batched_observation_boundary_profile.py:1180` cover
  LightZero collect-forward flatten/decode/timing/filtering.
- `tests/test_source_state_batched_observation_boundary_profile.py:1190` and
  `tests/test_source_state_batched_observation_boundary_profile.py:1299` cover
  facade/direct CTree compact arrays.
- `tests/test_source_state_batched_observation_boundary_profile.py:1598` and
  `tests/test_source_state_batched_observation_boundary_profile.py:1684` prove
  compact search service adapters preserve root identity.
- `tests/test_source_state_batched_observation_boundary_profile.py:1766` proves
  real direct CTree compact service drives the next hybrid env step and matches
  materialized rows/sample batches.
- `tests/test_source_state_batched_observation_boundary_profile.py:1921` gives
  the same local closed-loop shape for compact Torch.
- `tests/test_source_state_batched_observation_boundary_profile.py:2073` proves
  array-ceiling compact Torch mode owns a compact service run.
- `tests/test_compact_torch_search_service.py:54`, `tests/test_compact_torch_search_service.py:81`,
  `tests/test_compact_torch_search_service.py:159`,
  `tests/test_compact_torch_search_service.py:173`,
  `tests/test_compact_torch_search_service.py:221`,
  `tests/test_compact_torch_search_service.py:327`, and
  `tests/test_compact_torch_search_service.py:404` cover profile-only labels,
  compile signatures, binary masks, inactive roots, active-root order, fresh
  observations, and deterministic preconditions.
- `tests/test_mctx_synthetic_benchmark_legality.py:15` through
  `tests/test_mctx_synthetic_benchmark_legality.py:173` cover MCTX legality,
  row-major guards, profile-only labelling, and root-value payload extraction.

### RND Exploration Bonus

Code:

- `src/curvyzero/training/exploration_bonus.py:250` normalizes fail-closed RND
  specs.
- `src/curvyzero/training/exploration_bonus.py:451` patches LightZero configs
  for the reward-model entrypoint.
- `src/curvyzero/training/exploration_bonus.py:481` and
  `src/curvyzero/training/exploration_bonus.py:582` extract latest gray64 frames
  for LightZero and compact shapes.
- `src/curvyzero/training/exploration_bonus.py:629` implements
  `CurvyRNDRewardModel`.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5254`
  validates required RND metrics.

Tests:

- `tests/test_exploration_bonus.py:9` through `tests/test_exploration_bonus.py:176`
  cover config normalization, stale metadata rejection, entrypoint selection,
  and LightZero reward-model patches.
- `tests/test_exploration_bonus.py:184` through `tests/test_exploration_bonus.py:336`
  cover latest-frame extraction, compact uint8 normalization, stale channel
  ignore, and bad target/range rejection.
- `tests/test_exploration_bonus.py:338`, `tests/test_exploration_bonus.py:437`,
  `tests/test_exploration_bonus.py:479`, and `tests/test_exploration_bonus.py:543`
  prove predictor update, target freeze, seed control, update cadence metrics,
  and CUDNN flag reporting.
- `tests/test_lightzero_config_builder.py:447` and
  `tests/test_lightzero_config_builder.py:477` prove public visual-survival
  configs can add RND meter and positive RND bundles.

### Stock `train_muzero` Entry Points

Code:

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5980`
  selects the LightZero entrypoint from the exploration-bonus spec.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:5984`
  fails if the selected entrypoint is missing.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6183`
  only calls training/profile when validation passed.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6367`
  sets `called_train_muzero=true` immediately before invoking the entrypoint.

Tests and smokes:

- `tests/test_lightzero_phase_profiler.py:200` validates stock collect-search is
  a no-op hook path.
- `tests/test_lightzero_phase_profiler.py:268` and follow-on tests compare
  direct CTree hook output schema/statistics against stock collect output.
- `tests/test_lightzero_phase_profiler.py:824` verifies compact profile output
  carries `called_train_muzero` and `trainer_entrypoint`.
- `src/curvyzero/infra/modal/lightzero_connect4_tiny_train_smoke.py:595` and
  `src/curvyzero/infra/modal/lightzero_connect4_tiny_train_smoke.py:605` call
  stock `lzero.entry.train_muzero` in a tiny Connect4 smoke.
- `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py:687` and
  `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py:697` call stock
  `lzero.entry.train_muzero` in a tiny Pong smoke.
- `tests/test_curvytron_survivaldiag_manifest.py:135` pins survival diagnostic
  executable rows as stock `train_muzero` rows.

## 3. Important Missing Tests

1. Promotion eligibility gate.

There is no single fail-closed result validator that says "this profile result
cannot be summarized as Coach training" unless all promotion fields are present
and consistent. This should cover `profile_only`, `not_lightzero_ctree`,
fallback counts, action-feedback proof, replay/sample proof, RND metrics, and
root-noise metadata.

2. Capped original-loop stock-vs-candidate smoke.

Current local tests prove many pieces, but not that a candidate architecture
survives the original stock `train_muzero` loop with collector, replay, learner,
target building, and the current Coach command defaults. The code currently
prevents non-stock search backends in `mode=train`, so any candidate still needs
a profile-mode trainer smoke before it can become train-mode work.

3. Out-of-order service attachment.

The two-phase payload gate handles handles and delayed payloads, but the next
useful test should force a service to return results in a different order than
root compaction and prove replay/sample attachment is by stable ids, not by the
current tensor row.

4. Incomplete-row sampler visibility.

The gate hides payloads, and index rows materialize correctly, but there is not
yet a promotion-level sampler test that proves incomplete compact replay rows
cannot become learner-visible under a real replay-store shape.

5. Mixed terminal/live real-service final observation and RND.

Synthetic compact tests prove terminal final observations and RND latest-frame
attachment. The real direct CTree and compact Torch closed-loop tests are
strong, but they should include a mixed terminal/live row where one row
autoresets and one row stays live, then prove terminal next observation and RND
latest frame are immutable.

6. Full-loop RND metrics smoke.

RND model mechanics are well tested locally. What is missing is a tiny
trainer-entrypoint smoke proving the actual LightZero reward-model entrypoint
calls RND at the intended cadence under current CurvyTron trainer settings.

7. Noise/tie parity ladder.

No-noise direct/Torch tests exist. Promotion still needs exact no-noise parity
against a reference and a separate seeded noisy statistical gate. A path with
unlabelled root noise, tie behavior, or epsilon behavior is not valid.

8. Remote artifact digest.

Profile artifacts should carry a compact digest over root identity, selected
actions, applied joint actions, replay index rows, sample rows, RND latest
frames, and fallback counts. Without that digest, a remote speed chart can look
good while training on the wrong data.

## 4. Fastest Small Proof To Run Next

Run one capped original-loop bridge, not another pure profile speed row.

Shape:

```text
fresh non-live run_id
mode=profile
env_variant=source_state_fixed_opponent
stock trusted observation surface
background eval/GIF disabled
stop_after_learner_train_calls=1
output_detail=compact
stock control row plus one candidate-hook row, same seed/config
```

Required pass conditions:

- `called_train_muzero=true`;
- `trainer_entrypoint` is `lzero.entry.train_muzero`, or
  `lzero.entry.train_muzero_with_reward_model` for RND;
- `learner_train_calls >= 1`;
- collector/replay hooks observed at least one batch;
- candidate fallback count is zero;
- selected-action digest equals applied-joint-action digest;
- root identity digest survives root -> search -> replay -> sample;
- terminal rows use final observations;
- RND metrics are present when RND is enabled;
- the summary still says `promotion_eligible=false` unless every invariant in
  this report is present.

If that is too much for the next step, do the smallest local precursor:

```text
tests/test_source_state_batched_observation_boundary_profile.py
  test_real_compact_service_mixed_terminal_live_final_obs_and_rnd_latest

tests/test_compact_search_replay_contract.py
  test_compact_replay_payload_attach_by_stable_ids_after_out_of_order_flush
  test_compact_replay_sampler_hides_incomplete_index_rows

tests/test_lightzero_phase_profiler.py or a new promotion validator file
  test_candidate_summary_requires_action_replay_rnd_digests_and_zero_fallback
```

That local precursor is not enough for Coach promotion, but it will kill the
most likely "fast but wrong" failure modes before spending remote time.

## 5. Fast Invalidators

Stop promotion immediately if any of these happens:

- the path does not enter stock `train_muzero` or the RND reward-model
  entrypoint when it claims trainer-facing proof;
- a profile-only path is summarized as Coach training;
- non-stock collect search is used in `mode=train` without a new explicit
  trainer contract and matching tests;
- `collect_search_backend_fallback_calls > 0`;
- selected action is not exactly the next env `joint_action`;
- `root_index`, `env_row`, `player`, or `policy_env_id` changes across root,
  search, replay, materialization, or sample;
- selected action is illegal, or visit/raw count mass lands on illegal actions;
- terminal next observation is an autoreset frame;
- a terminal row remains active for search;
- the optimizer chooses player perspective instead of preserving the training
  observation contract;
- RND meter mode changes target rewards;
- RND-enabled trainer rows lack reward-model metrics;
- a learner can sample a partial compact row;
- root noise, tie handling, epsilon, or seed state is missing from a parity
  claim;
- MCTX, service-tax, mock-search, or compact-Torch helper rows are described as
  LightZero CTree or Coach-equivalent without a separate trainer proof.

## Recommendation

Keep the compact/MCTX/Torch work in the optimizer lane until a capped
`train_muzero` bridge exists. The code and tests now prove many local contracts,
which is good. They do not yet prove a Coach training speedup. The next useful
move is not a bigger speed table. It is a tiny original-loop proof with a hard
promotion eligibility gate.
