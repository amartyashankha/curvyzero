# Subagent Validation Gate Review - 2026-05-21

Scope: validation/regression audit for the current optimizer lane. I did not
touch live training runs, launch Modal jobs, change trainer defaults, or edit
production code. I only read tests/docs/code, ran focused local tests, and wrote
this review.

Focused local check run:

```text
uv run pytest -q -p no:cacheprovider \
  tests/test_source_state_batched_observation_mock_collector.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_multiplayer_source_state_trainer_surface.py
```

Result:

```text
101 passed, 2 skipped in 1.31s
```

## Plain Read

The local profile-only validation is better than the older risk notes suggest.
The current tree already fixes three earlier collect-forward concerns:

- `LightZeroCollectForwardStackProbe` now passes fixed-opponent `to_play=-1`,
  not player ids, and tests assert `lightzero_to_play_mode=fixed_opponent_minus_one`.
- Zero-action-mask roots are filtered before `policy.collect_mode.forward`, and
  telemetry reports total roots versus consumed roots.
- The LightZero collect-forward config now fails closed to the intended
  contract: hybrid profile canary, persistent direct GPU profile backend,
  `direct_gray64`, and `uint8` stack storage.

The remaining trust gap is not basic row/player reshaping. It is promotion
context. We have strong local profile-manager and materializer tests, plus
profile-only speed rows, but the smallest missing gates before trusting new
speed numbers are still:

1. A stock `train_muzero` batched-profile normal-death/autoreset gate with mixed
   terminal and live rows.
2. A stock `train_muzero_with_reward_model` batched-profile `rnd_meter_v0` gate
   proving latest-frame source, meter neutrality, and RND train/estimate
   counters.
3. A compact speed-row attestation check that refuses to summarize rows missing
   semantic identity fields.

Until those pass, the new speed rows are useful optimizer evidence, not trusted
training-default evidence.

## Existing Executable Coverage

| Contract | Current executable coverage | Residual gap |
| --- | --- | --- |
| Row/player order | `_row_major_render_rows`, `_row_major_render_players`, `_view_major_to_row_major_frames`; renderer-backed trainer surface tests; hybrid manager tests; scalar env id bridge tests. Expected order is `[row0/player0, row0/player1, row1/player0, row1/player1, ...]`. | No stock-loop mixed terminal row proves this mapping survives through `train_muzero` when some rows disappear from policy-ready roots but still need terminal timesteps. |
| Player perspective | Trainer surface asserts policy rows equal `observation[row, player]`; renderer-backed GPU candidate tests use sentinels that differ by row and player. Surface metadata records policy perspective schema and player ids. | Real persistent `direct_gray64` profile rows should carry a small row/player checksum or sampled frame proof, not only aggregate timings. |
| Stack order | `_push_row_major_frames_into_stack` tests newest-frame-last FIFO, uint8 normalization, selected-row reset, and latest-channel extraction. Renderer-backed surface tests prove reset frame then step frame occupy the expected stack channels. | Need the same stack proof attached to current medium/large speed artifacts, especially when scalar timestep materialization is disabled. |
| Reset and final observation | Source-state LightZero env tests prove terminal snapshots survive reset. Trainer surface tests prove terminal final observations are visual stacks, not metadata. Mock collector/materializer tests require terminal final observations and reject malformed shapes. Profile manager tests prove terminal timestep is kept before autoreset. | Existing profile-manager terminal test uses all rows terminal. The next stock gate should force mixed rows: at least one terminal row and at least one live row in the same batch. |
| Action masks | Trainer surface copies `legal_action_mask` to `lightzero_action_mask`; materializer preserves `[B,P,3] -> [B*P,3]`; policy row mapping compacts only live rows with legal actions; collect-forward probe filters zero masks. | Need a stock-loop artifact proving terminal zero-mask rows are not passed as collect roots while their terminal timesteps are still delivered. |
| `to_play` | Scalar trusted env emits `to_play=-1`; scalar materializers emit `-1`; collect-forward probe now emits `[-1] * active_root_count` and tests assert that mode. | Good for the current fixed-opponent lane. A future two-seat current-policy lane needs a separate `to_play=player_id` gate, not reuse this proof. |
| RND separation | Exploration bonus config tests fail closed, select reward-model entrypoint for `rnd_meter_v0`, require latest-frame feature shape `[1,64,64]`, and enforce meter weight zero. Mock collector tests exercise collect/train/estimate and target reward unchanged. | Missing stock batched-manager RND proof: actual reward-model entrypoint, predictor hash changes, target hash unchanged, latest-frame matches policy stack after manager/autoreset, and reward targets unchanged. |
| Normal death/autoreset | Vector autoreset tests stage copied terminal data before reset. Trainer surface has normal-death terminal final-observation tests and profile-no-death labeling tests. Hybrid profile counts terminal/autoreset rows. | No current stock `train_muzero` batched-manager speed row should be trusted as normal-death-safe until a mixed normal-death canary passes. |
| Observation surface fidelity | Renderer-backed CPU oracle matches dirty cache; renderer-backed profile requires explicit renderer and can require exact backend; direct/persistent config fails closed; docs record direct CPU parity/adversarial two-view canaries. | Speed rows should include or link a same-commit direct-surface semantic checksum/frame proof for asymmetric two-player states and bonus symbols. |

## Smallest Missing Gates

### P0. Stock Batched-Manager Mixed Normal-Death Gate

Run a bounded profile-only stock entrypoint canary. It should call stock
`train_muzero`, use `env_manager_type=curvyzero_batched_profile`, disable
eval/GIF/checkpoint clutter, and stop after the first learner call or a tiny
fixed collection horizon.

Pass criteria:

- `called_train_muzero=true`.
- `env_manager_type=curvyzero_batched_profile`.
- `death_mode=normal`, not `profile_no_death`.
- One batch contains both terminal and live physical rows.
- Terminal rows emit `done=true`, zero policy-ready roots, and
  `final_observation_present=true` before autoreset.
- Live rows remain ready after the terminal rows autoreset.
- `policy_env_id`, `policy_env_row`, and `policy_player` are row-major for live
  roots.
- `action_mask` shape is `[N,3]` for consumed roots and terminal zero-mask roots
  are counted as filtered/skipped, not forwarded to LightZero.
- `materialized_timestep_count`, `terminal_row_count`, `autoreset_row_count`,
  and `final_observation_nbytes` are nonzero and internally consistent.

This is the smallest gate that directly addresses row/player order, reset,
final observation, masks, and normal death in the stock-loop context.

### P0. Stock Batched-Manager RND Meter Gate

Run the same stock batched profile shape with `rnd_meter_v0` and a matched no-RND
anchor. This must use `train_muzero_with_reward_model`, not only the mock
collector.

Pass criteria:

- `exploration_bonus.mode=rnd_meter_v0`.
- `training_effect=reward_target_unchanged`.
- `feature_source=policy_gray64_latest/v0`.
- `rnd_metrics.input_shape=[1,64,64]`.
- `rnd_metrics.source_observation_shape=[4,64,64]`.
- `collect_data_calls > 0`, `train_with_data_calls > 0`, `estimate_calls > 0`.
- Predictor hash changes after training; target hash stays unchanged.
- `last_target_reward_changed=false` and reward deltas are exactly zero.
- A semantic marker or sampled checksum proves RND latest frames equal the latest
  channel of the manager policy stack for the same env ids/players.
- For terminal/autoreset rows, RND does not read post-reset frames as terminal
  final frames.

This keeps RND overhead and RND correctness separate from renderer speed.

### P0. Speed-Row Semantic Attestation Gate

Before interpreting any new speed table, reject rows missing the following
fields or equivalent compact evidence:

- profile identity: `profile_only`, `called_train_muzero`,
  `stock_lightzero_integrated`, `touches_live_runs`;
- backend identity: manager type, renderer backend, render surface, stack dtype,
  no-hidden-fallback flag;
- denominator identity: env steps, MCTS roots, simulation count, learner calls,
  replay sample calls, warmup policy;
- mapping identity: `policy_env_id`, `policy_env_row`, `policy_player`,
  row/player head and tail samples;
- stack identity: observation shape, latest-frame/stack dtype, materialized
  scalar timestep count;
- terminal identity: death mode, terminal row count, autoreset row count,
  final-observation presence/count/bytes;
- LightZero consumer identity when present: `to_play` mode, total roots,
  consumed roots, filtered zero-mask roots, illegal action count, CPU-tree label;
- RND identity when present: mode, feature source, train/estimate counters,
  predictor/target hashes, target reward unchanged.

This can be a small summary/parser test rather than a new large run. It prevents
the most dangerous failure mode: a fast row that silently changed surface,
death mode, denominator, or consumer path.

### P1. Collect-Forward Decode And Action-Feedback Gate

Current fake-policy tests cover batched mapping output and zero-mask filtering.
The smallest extra local coverage should add:

- dict keyed by string ready ids;
- list output;
- dict-of-arrays output with visit distributions and values;
- illegal decoded action raises and marks the row invalid;
- optional profile-only action feedback: decoded actions from the real
  collect-forward consumer are stepped through the scalar action bridge for one
  measured step, with row/player mapping and masks rechecked.

This is not required to read no-feedback collect-forward throughput, but it is
required before using collect-forward output as a behavioral loop proof.

### P1. Direct-Surface Fidelity Attachment Gate

Docs say the direct surface passed adversarial CPU-direct parity. Keep that, but
attach a tiny same-commit semantic proof to speed artifacts:

- asymmetric two-player state;
- both controlled-player views;
- diagonal trail and close parallel trail;
- at least one terminal final-observation frame;
- every simple-symbol bonus group or a documented sampled subset;
- checksum/frame sample for `direct_gray64` and CPU-direct oracle.

This should be cheap and deterministic. It keeps observation surface fidelity
from becoming a prose-only assumption in future speed summaries.

## Trust Decision

Safe to trust now:

- Local profile-only shape tests for row/player flattening, stack FIFO,
  final-observation materialization, action-mask preservation, fixed-opponent
  `to_play=-1`, zero-mask collect-forward filtering, and RND meter config basics.
- Profile-only no-death speed rows as optimizer/Amdahl evidence, if the row
  carries enough semantic identity fields.

Not safe to trust yet:

- Any claim that the batched GPU manager is a training default.
- Any normal-death speed row without the mixed stock-loop gate.
- Any RND speed row without the stock batched-manager RND meter gate.
- Any surface-speed claim that lacks backend/surface/stack identity and a
  current semantic checksum or parity attachment.

Bottom line: the next validation work should be two small stock-entrypoint gates
plus one row-summary attestation check, not another renderer microbenchmark.
