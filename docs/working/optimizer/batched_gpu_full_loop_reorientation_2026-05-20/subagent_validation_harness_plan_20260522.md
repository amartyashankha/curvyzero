# Subagent Validation Harness Plan - 2026-05-22

Scope: validation plan only. Do not implement from this document without a
separate task. Do not touch live Coach training runs, Modal volumes,
checkpoints, evals, GIFs, or tournaments.

The target lane is the profile-only compact closed loop:

```text
HybridCompactBatch
-> compact env step
-> persistent GPU direct_gray64 observation stack
-> CompactRootBatchV1
-> MCTX/JAX Gumbel MuZero search
-> CompactSearchResultV1 / CompactReplayIndexRowsV1
-> search-selected actions drive the next env step
```

Trusted observation semantics remain:

```text
browser_lines + simple_symbols -> cpu_oracle -> [4,64,64]
```

The current fast profile lane uses:

```text
jax_gpu_persistent_policy_framebuffer_profile + direct_gray64
browser-line-like trail semantics + simple_symbols
```

Treat this as a profile surface until it earns the gates below.

## Speed Claim Separation

Keep these claims separate in every harness result and summary:

| claim class | entrypoint | metric | current interpretation |
| --- | --- | --- | --- |
| Production Coach training | `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`, `mode=train` | training steps/sec, eval/tournament outcomes | Trusted path is stock `train_muzero` with CPU policy observation. Do not change or use as optimizer scratch. |
| Stock full-loop profile | same file, `mode=profile`, profile-only hooks allowed | profile steps/sec | May call `train_muzero`; useful for matched stock/direct/RND rows, but still not Coach launch advice. |
| Profile-only compact boundary | `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py` and `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py` | active roots/sec or boundary timings | Architecture falsifier only. Current best row is borrowed actor render state + resident stack + no explicit resident sync, about `50.4k` roots/sec at sim16 and `43.3k` at sim32. Async renderer sync did not help. |

The next harness should prove semantic boundaries before more aggressive
ownership/search rewrites. A fast row that lacks these fields is experiment-log
data, not a speed recommendation:

```text
profile_only / called_train_muzero / observation contract / renderer surface
death mode / RND mode / seed policy / fallback counts / action-feedback proof
sidecars disabled or intentionally matched
```

## P0 Harness Matrix

| boundary | smallest local proof | smallest Modal/profile proof | pass/fail rule |
| --- | --- | --- | --- |
| Deterministic seeds and tie policy | Add seed/tie metadata assertions around compact MCTX and compact replay proof. | One tiny H100 or L4 compact MCTX closed-loop row with fixed `seed`, `actor_seed`, `PRNGKey` base, `root_noise_weight=0.0`, and recorded tie policy. | Exact only for clear preferences and forced masks. Neutral/tie-heavy action equality is not a kill gate; classify it statistical. |
| CPU oracle vs GPU profile observation | Long no-death local fixture comparing trusted CPU oracle to profile GPU/direct surface with explicit divergence labels, plus exact persistent-vs-stateless same-surface parity. | Long no-death `source_state_batched_observation_boundary_profile` row with sampled CPU oracle checkpoints and direct_gray64 divergence telemetry. | Same-surface persistent/stateless must be exact. CPU-oracle vs direct_gray64 may diverge, but must be bounded, sampled, and never labeled exact parity. |
| Simple symbols and bonus overlays | One combined fixture with all bonus symbols, non-identity avatar colors, trail under bonus, head over bonus, both player views. | Small adversarial GPU render smoke for `direct_gray64 + simple_symbols`. | Bonus identities distinct; bonus overlays trail; heads overlay bonus; row/player view luma correct. |
| Borrowed render-state terminal/autoreset | Mixed terminal/live local test: borrowed state raises before renderer/probe, copied baseline preserves final frame then reset frame. | Optional tiny expected-failure profile row after exposing `max_ticks` in MCTX profile config; otherwise local is enough. | Borrowed state must fail closed on any terminal row. No silent fallback and no post-reset final observation. |
| MCTX actions drive env | Existing compact replay proof plus an MCTX-specific action-feedback verifier. | Compact closed-loop MCTX row must record and verify selected actions equal the next step's `joint_action`. | If next env step is not driven by previous search result, every closed-loop roots/sec number is invalid. |
| RND when trainer uses it | Existing RND latest-frame and target-reward tests plus one compact terminal/autoreset RND latest-frame check. | Matched profile-only stock full-loop no-RND and `rnd_meter_v0` rows if the current trainer config enables RND. | RND meter must use reward-model entrypoint, predictor changes, target stays frozen, target rewards unchanged, latest-frame source proven. |

## Exact Local Tests To Add Or Promote

### 1. Seed And Tie Harness

Likely files:

- `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`
- `tests/test_mctx_synthetic_benchmark_legality.py`
- `tests/test_source_state_hybrid_observation_profile.py`

Add:

```text
tests/test_mctx_synthetic_benchmark_legality.py::
  test_compact_mctx_closed_loop_reports_seed_and_tie_policy

tests/test_source_state_hybrid_observation_profile.py::
  test_compact_service_replay_proof_replays_fixed_seed_actions
```

Minimum checks:

- Result metadata records `seed`, `actor_seed`, MCTX PRNG key base or per-loop
  key formula, `num_simulations`, `root_noise_weight`, and tie classification.
- Clear-preference or single-legal fixtures are exact.
- Neutral/tie-heavy fixtures are recorded as statistical, not exact.
- If vendored CTree is used in a comparison, deterministic tie-breaking is
  opt-in and named, for example `set_deterministic_tie_breaking(True)`.
- Stock LightZero CTree reseeding/near-ties are not used as exact-fail gates.

Small source touch if needed: expose a tiny helper or result block in
`mctx_synthetic_benchmark.py`; do not alter production trainer defaults.

### 2. Long No-Death Observation Surface Check

Likely files:

- `src/curvyzero/infra/modal/source_state_gpu_render_benchmark.py`
- `src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py`
- `src/curvyzero/env/vector_visual_observation.py`
- `tests/test_source_state_gpu_render_benchmark_cpu.py`
- `tests/test_vector_visual_observation.py`
- `tests/test_source_state_batched_observation_boundary_profile.py`

Add:

```text
tests/test_source_state_gpu_render_benchmark_cpu.py::
  test_long_no_death_direct_gray64_vs_cpu_oracle_is_labeled_divergence

tests/test_source_state_batched_observation_boundary_profile.py::
  test_persistent_direct_gray64_matches_stateless_direct_gray64_on_sampled_long_rollout
```

Minimum fixture:

- B=2 or B=4, P=2, fixed row seeds, deterministic no-death actions, at least
  128 local source steps. Modal can use 512 or 1000 source steps.
- Sample checkpoints at reset, early append, mid-trail, late trail.
- Compare:
  - CPU oracle `browser_lines + simple_symbols` latest frame and stack.
  - Stateless `direct_gray64`.
  - Persistent `direct_gray64`.

Pass criteria:

- Persistent direct_gray64 equals stateless direct_gray64 exactly for uint8
  latest frames and normalized stack within `atol=1e-7`.
- CPU-oracle comparison reports `max_abs_diff`, mismatch fraction, sample
  pixels, and connected/bbox samples if available.
- Any summary field must say `cpu_oracle_divergence`, not `parity_exact`, for
  CPU oracle vs direct_gray64.
- Active trail count, cursor, reset rows, and renderer fallback/full-rebuild
  counts are recorded.

### 3. Simple Symbols / Bonus Overlay Combined Fixture

Existing useful tests:

- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_simple_symbols_keep_all_twelve_bonus_identities`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_simple_symbol_bonus_overwrites_underlying_trail`
- `tests/test_source_state_gpu_render_benchmark_cpu.py::test_direct_gray64_heads_overwrite_bonus_symbols`
- `tests/test_vector_visual_observation.py::test_cpu_oracle_simple_bonus_symbols_overlay_browser_line_trails`

Add one combined test:

```text
tests/test_source_state_gpu_render_benchmark_cpu.py::
  test_direct_gray64_combined_simple_symbols_perspective_and_overlay_fixture
```

Minimum checks:

- Non-identity `avatar_color`, including duplicate or nonzero color ids.
- Both controlled-player views.
- One bonus symbol on top of a trail.
- One live head on top of a bonus.
- One inactive stale trail slot past cursor.
- At least three bonus types, plus a separate all-12 identity check remains.

Pass criteria:

- CPU-direct and JAX direct outputs match exactly when JAX is available.
- Self/other luma flips correctly by controlled player.
- Bonus symbol pixels are distinct and overwrite trails.
- Head pixels overwrite bonus where they overlap.

This does not prove production CPU-oracle parity. It proves the fast profile
surface is internally stable for the policy-space symbols it claims.

### 4. Borrowed State Terminal/Autoreset Fail-Closed

Existing useful tests:

- `tests/test_source_state_hybrid_observation_profile.py::test_borrowed_single_actor_render_state_rejects_terminal_rows`
- `tests/test_source_state_hybrid_observation_profile.py::test_borrow_single_actor_render_state_fails_closed_on_terminal_rows`
- `tests/test_source_state_hybrid_observation_profile.py::test_native_actor_buffer_renderer_backed_autoreset_matches_terminal_then_reset_frames`

Add:

```text
tests/test_source_state_hybrid_observation_profile.py::
  test_borrow_single_actor_render_state_fails_closed_on_mixed_terminal_live_rows

tests/test_source_state_hybrid_observation_profile.py::
  test_copied_single_actor_render_state_preserves_mixed_terminal_final_and_live_rows
```

Minimum checks:

- One physical row terminal, one physical row live in the same batch.
- Borrowed mode raises before using renderer output as a final observation.
- `render_state_borrowed_steps` does not increment on failed terminal step.
- Copied baseline emits terminal final observation before autoreset.
- Live row remains policy-ready and keeps row/player ids.
- Terminal roots are inactive after zero-mask filtering.

Likely source touch if current helpers cannot force one row terminal:

- `tests/test_source_state_hybrid_observation_profile.py` can use a small fake
  actor/payload path or monkeypatch row-specific done state.
- Avoid changing `VectorMultiplayerEnv` or production reset semantics.

### 5. MCTX Action Feedback Into Next Env State

Existing useful test:

- `tests/test_source_state_hybrid_observation_profile.py::test_compact_service_replay_proof_steps_with_search_actions_and_builds_targets`

Add:

```text
tests/test_mctx_synthetic_benchmark_legality.py::
  test_closed_loop_action_feedback_record_rejects_unapplied_search_actions
```

Small source touch:

- In `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`, factor the
  closed-loop action feedback check into a helper or add step-record fields:
  `selected_action_checksum`, `next_joint_action_checksum`,
  `applied_joint_action_checksum`, `action_feedback_verified`.

Pass criteria:

- `loop_search_result.selected_action` is written into `loop_joint_action` at
  exactly `(env_row, player)`.
- `loop_next_step.payload["joint_action"]` equals `loop_joint_action`.
- Illegal selected actions fail through `validate_compact_search_result_v1`.
- The replay-index builder still checks `selected_action` against
  `next_joint_action`.

This is the harness that prevents the most embarrassing fast loop: MCTX runs,
but the env keeps stepping a constant action.

### 6. RND End-To-End Smoke If Current Trainer Uses It

Existing useful tests:

- `tests/test_lightzero_config_builder.py::test_public_visual_survival_builder_can_add_rnd_meter_bundle`
- `tests/test_exploration_bonus.py::test_compact_policy_gray64_adapter_normalizes_uint8_row_player_stacks`
- `tests/test_exploration_bonus.py::test_compact_policy_gray64_latest_adapter_ignores_stale_stack_channels`
- `tests/test_source_state_hybrid_observation_profile.py::test_compact_batch_can_feed_rnd_latest_frame_without_scalar_timestep`
- `tests/test_compact_search_replay_contract.py::test_two_record_compact_rows_use_final_observation_before_autoreset_and_rnd_latest`
- `tests/test_source_state_batched_observation_mock_collector.py::test_mock_collector_profile_runs_rnd_latest_frame_meter`

Add only if missing after the current config audit:

```text
tests/test_source_state_hybrid_observation_profile.py::
  test_compact_rnd_latest_frame_labels_terminal_autoreset_rows
```

Pass criteria:

- The current trainer config says whether exploration bonus mode is `none`,
  `rnd_meter_v0`, or `rnd_replay_target_v0`.
- If mode is not `none`, the smoke path uses
  `train_muzero_with_reward_model`, not plain `train_muzero`.
- RND reads latest policy gray64 channel for the same `(env_row, player)` as
  the replay row.
- `rnd_meter_v0` keeps target rewards unchanged:
  `last_target_reward_changed=false`,
  `last_target_reward_delta_abs_mean=0.0`,
  `last_target_reward_delta_abs_max=0.0`.
- Predictor hash changes and target hash does not.
- Terminal/autoreset rows are explicitly labeled so post-reset latest frames do
  not masquerade as terminal final frames.

## Modal/Profile Rows

Do not launch these from this document. They are the smallest rows to run after
the local tests pass.

### M1. Long No-Death Observation Divergence Row

Module:

```text
curvyzero.infra.modal.source_state_batched_observation_boundary_profile
```

Shape:

```text
compute=h100
batch_size=16
steps=512
warmup_steps=8
max_ticks=2000
surface_stack_backend=renderer_backed_profile
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
render_surface=direct_gray64
surface_facade_divergence_canary=true
verify_steps=16
cpu_reference_interval=64
include_rnd_meter=false
```

Required readout:

- `profile_only=true`
- `render_surface=direct_gray64`
- `input_surface=browser_lines_plus_simple_symbols_information`
- CPU-oracle divergence fields present
- persistent/stateless same-surface exact check present, if implemented in this
  row
- no live-run/checkpoint/eval/GIF/tournament side effects

### M2. Compact Closed-Loop MCTX Action Feedback Anchor

Module:

```text
curvyzero.infra.modal.mctx_synthetic_benchmark
```

Shape:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --compute h100 \
  --observation-mode curvytron_hybrid_compact_visual_sample \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --compact-visual-observation-source resident_gpu \
  --batch-size 1024 \
  --actor-count 1 \
  --player-count 2 \
  --body-capacity 4096 \
  --hidden-dim 64 \
  --rollout-steps 4 \
  --num-simulations 16 \
  --max-depth 16 \
  --closed-loop-steps 32 \
  --warmup-runs 8 \
  --steady-runs 12 \
  --native-actor-buffer \
  --hybrid-refresh-observation-stack \
  --hybrid-borrow-single-actor-render-state \
  --closed-loop-replay-index \
  --no-compact-root-copy-observation \
  --no-compact-visual-resident-sync \
  --no-emit-full-json
```

Repeat once with `--num-simulations 32`.

Required readout:

- `closed_loop.completed_steps == requested_steps`
- `closed_loop.total_active_roots > 0`
- `closed_loop.active_roots_per_sec` near the current same-denominator anchor:
  about `50.4k` sim16 and `43.3k` sim32, allowing normal Modal noise
- `render_state_handoff_mode=borrow_single_actor_env_state`
- `render_state_borrowed_steps > 0`, `render_state_copy_steps == 0`
- `compact_visual_observation_source=resident_gpu`
- `compact_visual_resident_sync=false`
- action-feedback verifier present and true for every loop step
- `illegal_selected_action_count == 0`
- no terminal/autoreset rows in the borrowed row

If the action-feedback verifier is absent, this row is a speed sample only, not
a validation row.

### M3. Borrowed Terminal Expected-Failure Smoke

Prefer local-only for this gate. Modal is optional because the behavior is a
fail-closed logic check, not a throughput question.

If a Modal smoke is required, first add a tiny profile-only `max_ticks` or
terminal-forcing flag to `mctx_synthetic_benchmark.py`; then run a small L4 row:

```text
compute=l4
batch_size=4
actor_count=1
player_count=2
max_ticks=1
closed_loop_steps=1
native_actor_buffer=true
hybrid_borrow_single_actor_render_state=true
observation_renderer_backend=jax_gpu_persistent_policy_framebuffer_profile
compact_visual_observation_source=host
```

Pass condition:

```text
expected failure containing "borrow_single_actor_render_state encountered terminal rows"
```

Do not convert this into silent fallback while still quoting borrowed-state
throughput.

### M4. Stock Full-Loop RND Meter Smoke

Only run this if the current trainer config uses RND or the speed table wants
to claim RND compatibility.

Module:

```text
curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train
```

Matched rows:

```text
row A: mode=profile, exploration_bonus_mode=none
row B: mode=profile, exploration_bonus_mode=rnd_meter_v0, exploration_bonus_weight=0.0
```

Use small profile caps and disable sidecars:

```text
mode=profile
collector_env_num=64
num_simulations=16
batch_size=64
max_train_iter small
max_env_step small
skip_lightzero_eval_in_profile=true
profile_allow_auto_resume=false
profile_volume_commit=false
background_eval_enabled=false
background_gif_enabled=false
commit_on_checkpoint=false
save_ckpt_after_iter very large
require_rnd_metrics=true for row B
```

Required readout for row B:

- `called_train_muzero_with_reward_model=true` or equivalent entrypoint proof
- `rnd_reward_model_metrics.enabled=true`
- `collect_data_calls > 0`
- `train_with_data_calls > 0`
- `estimate_calls > 0`
- `train_cnt_rnd > 0`
- `predictor_changed=true`
- `target_changed=false`
- `last_target_reward_changed=false`
- target reward deltas exactly zero
- latest-frame source is `policy_gray64_latest/v0`

Do not use positive `rnd_replay_target_v0` as a validation shortcut. Meter mode
proves plumbing and overhead only.

## Minimal Test Command

After implementing the local tests, run only the focused validation slice first:

```bash
uv run pytest -q -p no:cacheprovider \
  tests/test_mctx_synthetic_benchmark_legality.py \
  tests/test_source_state_hybrid_observation_profile.py \
  tests/test_source_state_gpu_render_benchmark_cpu.py \
  tests/test_vector_visual_observation.py \
  tests/test_source_state_batched_observation_boundary_profile.py \
  tests/test_compact_search_replay_contract.py \
  tests/test_exploration_bonus.py \
  tests/test_lightzero_config_builder.py
```

Then run ruff only on touched files.

## Kill Criteria

Stop architecture work and fix validation first if any of these happen:

- Borrowed render state can render a terminal row instead of failing closed.
- MCTX selected actions are not proven to be the next env `joint_action`.
- Long no-death direct_gray64 rows are summarized as CPU-oracle exact parity.
- Simple-symbol overlays drift or bonus identities collapse.
- RND-enabled trainer rows do not exercise the reward-model entrypoint.
- RND meter mutates target rewards.
- Any speed table mixes production training speed, stock full-loop profile
  speed, and profile-only boundary roots/sec in one claim.

## Recommended Order

1. Add the seed/tie metadata and action-feedback verifier tests.
2. Add mixed terminal/live borrowed fail-closed coverage.
3. Add the long no-death observation divergence and same-surface exact check.
4. Add the combined simple-symbol overlay fixture.
5. Run the focused local test slice.
6. Run M2 sim16/sim32 profile-only rows.
7. Run M4 only if the current trainer/RND claim requires it.

Only after those pass should more aggressive state-owner, search-service, or
array-native changes be treated as measuring the right thing.
