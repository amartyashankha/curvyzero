# Compact Service Validation Gates - 2026-05-22

Scope: docs-only validation design for promoting compact-buffer and search-service
optimizations toward trainer-facing use. I reviewed the current compact replay,
hybrid observation, boundary/search, RND, MCTX legality, and GPU render tests. I
did not touch live Coach training, Modal runs, checkpoints, evals, GIFs, or
tournament state.

## Short Read

The current tests prove many local contracts. They do not yet prove that a fast
compact/search-service loop can replace the trainer-facing path.

The promotion rule should be simple:

```text
profile speed is not enough
the same roots must produce the same actions, visit policies, root values,
replay rows, RND inputs/rewards, terminal frames, masks, and player views
```

Until that proof exists, the compact-buffer/search-service work should stay
profile-only.

## Promotion Ladder

### Gate 0: Profile-Only Experiment

Allowed claim:

```text
this shape is faster in a controlled profile
```

Required:

- explicit `profile_only` metadata;
- backend name and implementation id;
- seed, model identity, renderer/observation schema, death mode, RND mode;
- active root count, terminal root count, filtered root count;
- no hidden fallback count, or fallback count labeled clearly;
- timing buckets for env step, observation handoff, search, root extraction,
  replay/index materialization, RND, and sync/readback.

Not allowed:

- Coach-facing training recommendation;
- "replay-valid" claim if replay rows were not built;
- comparing action-only/deferred/overlap rows as if they were trainer rows.

### Gate 1: Root And Search Contract

The optimized service must consume the same compact roots and return the same
kind of compact search result.

Hard checks:

- `policy_env_id`, `env_row`, `player`, `compact_root_row`, and `active_root_mask`
  round-trip exactly.
- Legal masks are binary. Fractional masks fail closed.
- Empty-mask roots are either inactive before search or rejected before model
  inference.
- Selected actions are legal.
- Visit policy has zero mass on illegal actions and sums to one over legal
  actions.
- Root values are attached to active roots only. Inactive-root poison sentinels
  must not leak into replay.
- Deterministic forced cases pass:
  - single legal action;
  - clear preferred action under full masks;
  - mixed masks where the preferred action is illegal.

### Gate 2: Replay Parity

For the same record `k`, the compact path must match the existing target-row
builder:

```text
root observation[k]
search output[k]
selected action stored into transition k -> k+1
reward/done/final observation from record k+1
target rows equal current object/direct compact target rows
```

Hard checks:

- selected action used to step the env equals the action stored in replay;
- visit policy and root value attach to the same root identity as the selected
  action;
- `policy_row` and `compact_root_row` remain distinct and correct;
- non-prefix active roots work, such as active roots `[1, 3]`;
- non-identity `policy_env_id` values survive every edge;
- materialized compact index rows equal immediate compact target rows and object
  path target rows.

### Gate 3: Terminal, Final Observation, And Autoreset

Terminal rows are where silent training bugs hide.

Hard checks:

- live and terminal rows appear in the same batch;
- terminal roots are not searched;
- live roots in the same batch are still searched;
- terminal next observation uses `final_observation`, not the autoreset frame;
- autoreset sidecars point to the reset rows but do not overwrite the terminal
  replay target;
- normal-death mode and no-death mode both run the same contract:
  - normal death must prove final-observation/autoreset behavior;
  - no-death must prove long-run stack/order behavior without terminal churn.

### Gate 4: RND Latest-Frame And Reward-Shaping Parity

RND is a separate contract, not just metadata.

For `rnd_meter_v0`:

- RND reads the latest policy frame for the same root ids and players;
- target reward is unchanged;
- predictor updates and target-network freeze are separately attested by the RND
  tests;
- skipped/small-buffer update counts are visible.

For reward-shaping modes such as `rnd_replay_target_v0`:

- the same latest-frame tensor must feed RND in baseline and optimized paths;
- the intrinsic reward tensor must attach to the same replay rows;
- reward deltas must match a deterministic fake-RND oracle before any real
  model noise is allowed;
- normalization mode must be explicit, because batch-local min/max is not the
  same claim as a globally decaying novelty signal.

### Gate 5: Player Perspective

The compact path must not silently swap player views.

Hard checks:

- player 0 and player 1 observations use different sentinel pixels or checksums;
- row-major order is `[row0/player0, row0/player1, row1/player0, row1/player1]`;
- `controlled_player`/perspective metadata is present at observation creation,
  root batch, search result, and replay row edges;
- player-specific legal masks remain attached to the same player after compaction;
- future multiplayer paths must add a player-count/schema guard instead of
  assuming two players everywhere.

### Gate 6: Sampled Observation And Frame Parity

For any renderer/stack optimization, compare sampled frames against the trusted
path at the observation boundary and replay boundary.

Hard checks:

- latest frame is newest channel;
- stack FIFO order is stable after several steps;
- terminal final observation stack is the terminal stack, not the reset stack;
- sampled host-stack and resident/device-stack frames match under the chosen
  parity mode;
- sampled replay rows contain the same `observation` and `next_observation`
  frames after materialization;
- normal-death and no-death sampled trajectories are both represented.

Exact pixel parity is ideal for a replacement renderer. If the chosen training
surface is intentionally approximate, the gate must say that plainly and use a
fixed tolerance plus information checks for trails, heads, bonus symbols, and
player perspective.

### Gate 7: Trainer-Facing Canary

Only after Gates 1-6 pass locally:

- run a tiny trainer-facing canary with the same seed and config as the trusted
  path;
- disable or match checkpoint/eval/GIF side costs;
- emit one replay sample digest from baseline and candidate;
- compare actions, masks, visits, root values, target rewards, RND fields,
  done/final-observation fields, and observation checksums;
- require `fallback_count == 0` for any optimized-speed claim.

## Current Test Coverage

### `tests/test_compact_search_replay_contract.py`

Strong coverage:

- two-record mixed live/terminal rows;
- final observation before autoreset;
- compact target rows match object target rows;
- compact root/search/replay chunk round-trip;
- compact replay index rows avoid observation materialization, then materialize
  back to target rows;
- selected action mismatch rejection;
- terminal next-done rows require final-observation masks;
- illegal selected actions and illegal visit mass reject;
- stale env row, policy env id, and legal mask reject;
- non-prefix active roots;
- deferred payload rows matching immediate rows for a synthetic fixture;
- compact root observation can be a view for the profile hot path;
- RND latest-frame extraction from compact observation.

Still missing:

- one closed-loop parity test that spans multiple records and proves the search
  action used for stepping is the same payload later materialized into replay;
- out-of-order deferred/overlap payload commit by explicit record/root identity;
- non-identity `policy_env_id` on every success path, not only rejection paths;
- inactive-root poison values proving only active root values reach replay;
- normal-death and no-death versions of the same compact service contract.

### `tests/test_source_state_hybrid_observation_profile.py`

Strong coverage:

- deterministic shapes and row/player scalar ids;
- profile-only metadata does not touch trainer defaults;
- native actor buffer matches payload merge for core arrays;
- autoreset terminal rows are counted;
- renderer-backed row-major player order;
- renderer-backed terminal final observation uses the terminal frame;
- native actor buffer supports renderer-backed rows;
- persistent device-only latest frame matches host stack order for same actions;
- borrowed render state rejects terminal rows;
- uint8 stack storage and FIFO order;
- compact batch sidecars passed to probes;
- compact RND latest frame without scalar timestep;
- compact service replay proof steps with search actions and builds targets;
- compact service proof accepts array-ceiling compact search arrays;
- warmup seed is separated and required;
- action-mask order is preserved for probes and scalar timesteps.

Still missing:

- compact service replay proof that compares the produced rows against the
  object/direct compact target-row path in the same test;
- RND reward-shaping parity, not just latest-frame extraction;
- a combined terminal/live + non-prefix-active + non-identity-id proof;
- sampled observation/frame parity between the optimized resident path and the
  trusted host/materialized path at the replay-row edge.

### `tests/test_source_state_batched_observation_boundary_profile.py`

Strong coverage:

- row-major frame/player helpers;
- dynamic renderer partial row and requested player order;
- persistent renderer full row-major request validation;
- compact trail live-cursor trimming;
- config gates for direct surfaces, persistent GPU profile, async device-only,
  LightZero collect/initial inference, array ceiling, mock search service, and
  compact service replay proof;
- LightZero collect-forward stack probe decoding and illegal-action rejection;
- direct CTree compact arrays with legal masks, rows, players, values, and visits;
- direct CTree compact output can feed checked target rows;
- real-policy CPU tests for stock facade, direct CTree, and GPU-latent variants;
- single-legal action exactness and mixed-mask respect;
- precomputed recurrent tests showing recurrent calls are skipped while masks
  remain respected;
- dense/array-ceiling profile-only guardrails;
- persistent delta-state tests;
- stack push/latest/reset tests;
- tolerant and exact parity helpers.

Still missing:

- one trainer-style replay parity canary that uses the boundary probe output,
  env step, compact replay index rows, materialization, and object-row oracle;
- explicit normal-death versus no-death paired parity for the compact service;
- sampled frame parity carried all the way into replay rows, not only boundary
  render/stack helpers;
- RND reward-shaping parity through the boundary path.

### Related Tests

`tests/test_exploration_bonus.py` covers RND config normalization, fail-closed
specs, latest-gray64 adapters, compact latest-frame adapters, predictor/target
behavior, update cadence, seeding, and meter metrics. It does not prove that a
compact search-service replay row receives the same RND-shaped reward as the
trusted trainer path.

`tests/test_mctx_synthetic_benchmark_legality.py` covers legality summaries,
resident stack shape/order guards, closed-loop timing labels, and direct MCTX
root-value extraction. It does not prove that action-only/deferred/overlap
profile rows are replay-valid.

`tests/test_source_state_gpu_render_benchmark_cpu.py` covers render palette,
owner ordering, adversarial fixtures, two-view parity when JAX is available,
direct gray64 simple symbols, symbol overwrite order, and heads over symbols.
It is useful for observation parity, but it is not a replay/trainer contract.

## Required Attestation Fields

Every promotion candidate should emit these fields in profiles and proof rows:

- `profile_only`
- `trainer_facing_candidate`
- `compact_service_contract_id`
- `observation_schema`
- `renderer_surface`
- `player_perspective_schema`
- `death_mode`
- `autoreset_enabled`
- `rnd_mode`
- `rnd_reward_effect`
- `search_impl`
- `num_simulations`
- `model_identity`
- `root_count`
- `active_root_count`
- `terminal_root_count`
- `filtered_zero_mask_root_count`
- `selected_action_checksum`
- `visit_policy_checksum`
- `root_value_checksum`
- `illegal_action_count`
- `illegal_visit_mass_max`
- `fallback_count`
- `replay_materialized`
- `final_observation_row_count`
- `sampled_observation_checksum`
- `sampled_next_observation_checksum`

## Smallest Next Test Additions

These are enough to give the main optimizer confidence without turning this into
a giant test maze:

1. Add
   `test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows` to
   `tests/test_compact_search_replay_contract.py`.

   Use a three-record synthetic chunk with:
   - one terminal row,
   - one live row,
   - non-prefix active roots,
   - non-identity `policy_env_id`,
   - sentinel selected actions, visit policies, and root values.

   Assert compact index rows materialize to the same target rows as
   `build_compact_target_rows_from_search_arrays_v0` and the object-row builder.

2. Add
   `test_compact_service_replay_proof_matches_target_row_oracle_with_terminal_and_ids`
   to `tests/test_source_state_hybrid_observation_profile.py`.

   Extend the existing compact-service proof so the probe's output is compared
   against a direct target-row oracle, not only counted. Include terminal
   final-observation-before-autoreset and non-identity ids.

3. Add
   `test_compact_rnd_reward_shaping_matches_fake_oracle_latest_frames` near the
   RND tests.

   Use a deterministic fake RND reward over latest frames. Prove:
   - compact latest-frame input is the same as the trusted path;
   - `rnd_meter_v0` leaves target rewards unchanged;
   - reward-shaping mode attaches the same reward delta to the same rows.

4. Add
   `test_search_result_ignores_inactive_root_poison_values_and_masks` to
   `tests/test_compact_search_replay_contract.py` or the boundary test.

   Put impossible root values and visit mass on inactive roots. Assert only
   active roots are validated and materialized.

5. Add
   `test_sampled_host_and_resident_observation_rows_match_replay_frames` to the
   boundary or hybrid observation tests.

   With fixed actions, sample a few roots after several steps. Compare latest
   channel, FIFO order, player perspective, final observation, and materialized
   replay observation/next-observation checksums between the trusted host path
   and optimized resident path.

If only one test can be written first, write item 1. It is the smallest test
that catches the worst promotion bug: fast search payloads attached to the wrong
replay facts.
