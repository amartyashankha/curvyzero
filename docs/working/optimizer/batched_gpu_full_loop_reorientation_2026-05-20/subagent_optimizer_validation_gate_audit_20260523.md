# Optimizer Validation Gate Audit - 2026-05-23

Scope: validation/test critique only. I inspected the current optimizer docs,
compact replay/search tests, batched observation tests, RND tests, phase-profiler
tests, and policy-observation contract docs. I did not touch live Coach runs and
did not change code.

## Short Read

Do not promote a fast optimizer path toward training just because it is fast.
Promote only after it proves that the same policy rows keep the same identity,
legal actions, chosen actions, replay targets, observations, player perspective,
RND inputs, and seed/tie semantics.

Current state:

- Compact root/search/replay contracts are locally strong.
- Direct CTree and mock/service-tax profile probes have legality and identity
  checks at the probe boundary.
- Observation stack and controlled-player perspective have useful tests.
- RND has good unit coverage for config, latest-frame extraction, predictor
  training, target freeze, and reward-preservation in meter mode.
- The biggest missing proof is closed-loop: search-selected actions must drive
  the next environment step, then the same selected actions and same identities
  must become replay rows, RND rows, and trainer-facing samples.
- Full-loop promotion still needs a tiny matched smoke that calls the right
  LightZero entrypoint with fallback counts, RND metrics when enabled, and
  replay/observation digests.

## Minimal Promotion Gates

These are the smallest gates I would require before any optimizer path is
described as train-facing.

| Gate | What Must Be True | Current Coverage | Missing Before Promotion |
| --- | --- | --- | --- |
| 1. Root identity | `root_index`, `env_row`, `player`, `policy_env_id`, `active_root_mask`, `done_root`, and `to_play` survive compaction/search/replay. Non-prefix roots and non-identity ids must work. | `tests/test_compact_search_replay_contract.py` covers non-prefix active roots, non-identity `policy_env_id`, stale identity rejection, active-root masks, `to_play=-1`, and view/copy observation mode. `tests/test_source_state_batched_observation_boundary_profile.py` covers direct and array-ceiling `CompactSearchServiceV1` adapters preserving root identity. | One real profile-path closed-loop proof that the exact root ids emitted by the live probe are the ids used by replay/RND/trainer samples, not just synthetic fixtures. |
| 2. Action legality | Legal masks are binary; empty masks are inactive or rejected; selected action is legal; visit policy and raw counts have zero illegal mass and legal rows sum to one. | Compact result validation rejects illegal selected actions and illegal visit mass. Boundary tests reject fractional masks and decoded illegal actions. MCTS array tests cover all-legal, mixed-mask, single-legal, and biased-logit cases. | Promotion result metadata must always report `illegal_action_count`, filtered zero-mask root count, fallback count, and legal-mask schema. This is partly present in profile telemetry but not yet a single mandatory gate. |
| 3. Selected action drives next env state | The action chosen by search at record `k` is exactly the `joint_action` used to step the env into record `k+1`. | Compact replay index builder rejects mismatches between `selected_action` and `next_joint_action`. `test_deferred_search_payload_rows_match_immediate_rows_for_non_prefix_roots` and `test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows` stage selected actions into `joint_action` in synthetic chunks. | Still missing the real env loop verifier. Need a profile/local closed-loop test where the probe's actual search result writes `joint_action`, `surface.step()` consumes it, and the emitted next record proves `applied_joint_action_checksum == selected_action_checksum`. Without this, a fast loop could search one action while stepping another. |
| 4. Replay row targets | Search output, reward, done, final observation, target reward, policy target, root value, row/player ids, and next observation all match existing target-row builders. | Compact rows match object/direct target rows. Index rows avoid materializing observations and then materialize back to the same rows. Terminal final observation before autoreset is covered. | Need the same parity over a real probe/search output, not just synthetic chunks. Also need an inactive-root poison sentinel check so inactive roots cannot leak values into replay. |
| 5. Terminal/final observation/autoreset | Terminal next observation uses `final_observation`, not post-reset observation. Terminal roots are not searched; live roots in the same batch still are. | Compact target tests cover final observation before autoreset. Batched observation tests cover terminal final stack capture. Boundary config has terminal/autoreset sidecar checks. | Need a mixed terminal/live closed-loop profile gate for the current fast path. Borrowed/persistent render state must fail closed or explicitly copy terminal frames before reset. |
| 6. Observation perspective | The controlled player view is Coach/training-owned, not optimizer-owned. Both player views must stay row-major and attached to the same action/reward/mask. | `policy_observation_perspective_contract_2026-05-15.md` defines the rule. `tests/test_source_state_batched_observation_profile_cpu.py` covers controlled rows, both-player row-major order, per-row controlled actions, palette tracking avatar color ids, and RND materialization per player view. `tests/test_lightzero_config_builder.py` pins the public observation contract. | Need the compact/search/replay closed-loop to carry perspective metadata all the way through root batch, search result, replay rows, and any tournament/eval consumer. Current compact gates mostly prove row/player ids, not full trainer/tournament perspective handoff. |
| 7. RND/exploration bonus | If RND is enabled, the reward-model entrypoint is used; RND reads the latest policy gray64 frame for the same row/player; meter mode leaves target rewards unchanged; predictor changes; target stays frozen; cadence metrics are visible. | `tests/test_exploration_bonus.py` covers config fail-closed behavior, LightZero patching, latest-frame extraction, compact uint8 normalization, stale channel ignore, predictor training, target freeze, seed control, update cadence, and metrics. Source/batched observation tests cover latest-frame extraction from compact/player-view stacks. Trainer code has `require_rnd_metrics` validation hooks. | Need a full-loop smoke with the current trainer settings and RND enabled when claiming compatibility. It must prove `train_muzero_with_reward_model`, `collect_data`, `train_with_data`, `estimate`, `train_cnt_rnd > 0`, predictor changed, target unchanged, and target reward unchanged for `rnd_meter_v0`. Also need a compact terminal/autoreset RND latest-frame test so reset frames do not leak into terminal RND rows. |
| 8. Deterministic seeds and tie handling | Every exact comparison must state seed policy, root noise, epsilon, PRNG key or CTree seed behavior, and tie policy. Single-legal/clear-preference cases can be exact; neutral tie-heavy cases should be statistical. | Many tests set seeds, set `root_noise_weight=0.0`, and use single-legal or biased-logit fixtures. RND seed tests cover predictor init/sampling. | Need a formal result metadata gate for `seed`, `actor_seed`, `root_noise_weight`, `epsilon`, `temperature`, and tie classification. Need deterministic tie-breaking only for fixtures that are supposed to be exact. |
| 9. Full-loop smoke | A candidate that touches training must run a tiny full-loop smoke through the right LightZero entrypoint with side costs disabled or matched. | Phase-profiler tests prove `called_train_muzero`/profile compact output summaries, evaluator skip shape, collect-search backend schema/fallback counters, and phase timing fields. Trainer code records entrypoint, command, RND metrics, checkpoints, and phase profile. | Need one matched stock-vs-candidate smoke before promotion: same seed/config, no live sidecars, sparse/no checkpoints, no hidden fallback, replay/target/observation digests, and RND metrics if RND is enabled. Profile-only boundary rows are not enough. |

## Current Useful Coverage By File

### `tests/test_compact_search_replay_contract.py`

Useful:

- final observation before autoreset;
- compact target rows equal object target rows;
- compact service root/search/replay chunk round trip;
- compact replay index rows skip observation materialization and materialize back;
- selected-action mismatch rejection against `next_joint_action`;
- terminal next-done rows require final-observation masks;
- illegal selected action and illegal visit mass rejection;
- stale env row, stale `policy_env_id`, and stale legal-mask rejection;
- non-prefix active roots;
- deferred payload rows matching immediate rows;
- non-identity `policy_env_id` success path;
- fake `CompactSearchServiceV1` protocol into index rows;
- observation view option for profile hot path;
- compact RND latest-frame extraction.

Missing:

- real search result drives real env step;
- inactive-root poison sentinel;
- normal-death/no-death versions of the same closed-loop compact service proof;
- RND terminal/autoreset latest-frame proof in compact replay rows.

### `tests/test_source_state_batched_observation_boundary_profile.py`

Useful:

- row-major view conversion and render order;
- boundary config rejects unsafe hidden fallback shapes;
- direct CTree arrays return compact arrays;
- direct and array-ceiling compact search service adapters preserve root identity;
- direct CTree compact output can feed checked target rows;
- malformed sidecars rejected before search;
- fractional action masks rejected;
- all-legal fast path, mixed-mask, single-legal, and biased-logit MCTS cases;
- precomputed recurrent mode respects masks and reports synthetic eval counts;
- decoded illegal action rejected;
- array ceiling/mock/service-tax probes store compact search arrays;
- persistent delta state and stack FIFO tests;
- RND without payload profile rejected;
- exact/tolerant parity helpers.

Missing:

- one mandatory closed-loop action-feedback checksum over the real profile env;
- a promotion-level result object that refuses to summarize speed if fallback or
  action-feedback proof is absent;
- seed/tie metadata assertion attached to every exact direct-vs-stock comparison;
- long no-death sampled observation check wired to promotion metadata.

### `tests/test_source_state_batched_observation_profile_cpu.py`

Useful:

- controlled-player row shape;
- both-player row-major order;
- CPU oracle matches direct per-player renders in that facade;
- FIFO stack shift;
- per-row controlled-player action routing;
- terminal final stack capture;
- profile GPU candidate requires explicit renderer and forbids hidden CPU fallback;
- RND latest-frame extraction from controlled and both-player stacks;
- palette tracks avatar color ids, not player indices.

Missing:

- direct connection to compact search/replay root ids;
- tournament/eval consumer proof that checkpoint policies receive the same
  observation perspective that training recorded.

### `tests/test_exploration_bonus.py`

Useful:

- RND modes fail closed;
- `rnd_meter_v0` selects `train_muzero_with_reward_model` metadata and has
  target-reward unchanged semantics;
- positive `rnd_replay_target_v0` is clearly objective-changing;
- latest gray64 extraction supports LightZero batch shapes and compact row/player
  stacks;
- bad shapes/ranges rejected;
- RND predictor trains and target freezes;
- zero-weight meter leaves target rewards unchanged;
- positive weight changes target rewards;
- seed controls init/sampling;
- update cadence and small-buffer metrics are visible.

Missing:

- a current full-loop RND smoke with `require_rnd_metrics=true`;
- proof that compact terminal/autoreset rows feed RND from terminal latest frame,
  not reset latest frame;
- a training recommendation rule for RND cadence, separate from validation.

### `tests/test_lightzero_phase_profiler.py`

Useful:

- phase profiler can stop after learner train calls;
- profile hooks record MCTS/model device and batch sizes;
- evaluator skip keeps LightZero return shape;
- collect-search backend validation and stock no-op;
- direct CTree collect hook fallback count;
- direct CTree collect output schema matches stock;
- masked raw visit contract;
- compact output summary records backend, fallback counts, raw-vs-derived env
  step counts, and selected timings.

Missing:

- full-loop candidate smoke comparing replay/target/observation digests against
  trusted stock under one fixed config;
- hard fail if a candidate path claims promotion while `collect_search_backend_fallback_calls > 0`;
- hard fail if RND is enabled but reward-model entrypoint/metrics are absent.

### Perspective Docs

Useful source of truth:

- `docs/working/training/leaderboard_to_training_2026-05-13/policy_observation_perspective_contract_2026-05-15.md`

Important rule:

- Optimizer does not choose the perspective. Training chooses the controlled
  physical player; optimizer must emit that controlled-player view exactly.

Missing:

- compact/search/replay result metadata must cite this contract id and carry the
  controlled player id through every edge.

## Minimal Gate Order

1. Add a real closed-loop action-feedback verifier.
   Search actions must become `joint_action` for the next env step. This is the
   highest-risk missing gate.

2. Add promotion metadata fail-closed checks.
   A speed row cannot be promotion-eligible unless it has root identity, legal
   action, fallback count, seed/tie, observation contract, RND mode, and
   action-feedback fields.

3. Add mixed terminal/live final-observation gate for the current fast path.
   Normal death and no death need separate proof. Borrowed state must fail
   closed on terminal rows or explicitly copy final frames.

4. Add RND full-loop smoke when claiming RND compatibility.
   Use `rnd_meter_v0` with weight zero and `require_rnd_metrics=true`. Positive
   reward-changing RND is not a shortcut for plumbing validation.

5. Add controlled-player perspective digest through compact root -> search ->
   replay -> trainer sample.
   This can start as checksums/metadata, but both players must be exercised.

6. Run a tiny matched full-loop stock-vs-candidate smoke.
   Same seed, same config, sidecars disabled or matched, no hidden fallback,
   replay/target/observation digests recorded.

Only after these pass should the optimizer path become a training candidate.

## Kill Conditions

Stop promotion and fix validation first if any of these happens:

- selected search action is not proven to be the next env `joint_action`;
- `policy_env_id`, `env_row`, or `player` changes across root/search/replay;
- illegal actions or illegal visit mass are tolerated;
- terminal rows use autoreset observations as next observations;
- RND meter changes target rewards;
- RND-enabled run does not use the reward-model entrypoint;
- exact parity is claimed for an intentionally approximate observation surface;
- fallback count is nonzero in a promoted speed claim;
- speed tables mix production training, stock full-loop profile, and profile-only
  boundary roots/sec as one number.

## Recommendation

The current optimizer path should stay profile-only until the closed-loop
action-feedback gate and tiny matched full-loop smoke exist. The local compact
contracts are good enough to continue aggressive optimizer experiments, but not
enough to tell Coach that a fast path is training-safe.

