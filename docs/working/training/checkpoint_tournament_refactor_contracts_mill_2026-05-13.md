# Checkpoint Tournament Contract Mill - 2026-05-13

Read-only code critique lane. This doc names the remaining artifact/data-flow
contract cuts after the first refactor pass.

## Current Data Flow

1. Checkpoint discovery reads the training Volume and returns concrete
   checkpoint refs.
2. The rating CLI normalizes those refs into a rating spec and writes
   `ratings/<rating_run_id>/config.json`.
3. The rating round writes `rounds/<round_id>/input.json` with scheduled pair
   specs.
4. Game or shard workers write game summaries, optional GIFs, and shard
   summaries under `battles/<battle_id>/`.
5. The reducer writes battle summaries, `battle_index.json`, `pair_history.json`,
   `scheduler_state.json`, round results, round ratings, and final `latest.json`.
6. The provisional writer scans shard summaries and writes
   `provisional_latest.json` plus a refreshed `battle_index.json`.
7. The website should read small committed artifacts first: `progress.json`,
   `latest.json`, `provisional_latest.json`, and `battle_index.json`.

## Findings

- The pure helper now has artifact filename constants and path helpers for
  `provisional_latest.json`, run-level `results.json`, and `pair_spec.json`.
  That is the right direction. Remaining cleanup is to make the Modal app use
  those helpers everywhere.
- Discovery still owns contract strings in the Modal app:
  `checkpoint_selection` values, missing reasons, schema id, and the
  `train/lightzero_exp*/ckpt/iteration_*.pth.tar` scan glob. These should move
  to a pure discovery contract module before discovery grows more modes.
- `rating_pool_hash` only identifies the checkpoint pool. It does not include
  evaluation context such as policy mode, max steps, decision timing,
  simulations, natural bonus setting, env/reward compatibility, or score formula.
  Reusing `pair_history.json` across a changed evaluator can silently mix
  evidence.
- `pair_history.json` and `scheduler_state.json` both validate only `pool_hash`.
  Add a separate rating context hash and require it when scheduling adaptive
  rounds.
- Request-path website fallbacks still can scan shard summaries or derive live
  pair rows when `battle_index.json` is missing. Keep these as recovery/debug
  paths, but make the normal product contract: website reads committed small
  artifacts only.
- Progress and source/status strings remain mostly implicit across Python and
  page JS: `games_running`, `all_games_seen`, `ratings_written`,
  `games_running_with_provisional_ratings`, `live_shard_tallies`,
  `checkpoint_round_input`, and `battle_index_missing`. These should become
  named constants or one small enum-like module.
- Cache keys are string-built in several places. A typo changes behavior without
  test visibility. Add helpers for `live-progress`, `provisional-rating`,
  `progress-refresh-spawn`, `battle-detail`, `gif-bytes`, and `json-bytes`.
- `_list_rating_runs` still constructs `latest.json`, `progress.json`, and
  `config.json` paths directly under the rating root. Use helper refs so the
  website and writer stay coupled to one path contract.
- Volume boundaries are sound: checkpoints come from `curvyzero-runs`; tournament
  artifacts go to `curvyzero-curvytron-tournaments`. The contract should still
  state that refs stored in artifacts are always Volume-relative refs, never
  mounted absolute paths.

## Exact Cleanup Cuts

1. Pure discovery contract:
   - add constants for discovery schema id, selection values, scan glob, and
     missing reasons;
   - return typed/validated discovery rows from a pure helper;
   - keep the Modal function as a thin Volume wrapper.
2. Artifact refs sweep:
   - replace the Modal-local `_rating_provisional_latest_ref` with
     `arena.rating_provisional_latest_ref`;
   - replace direct `"latest.json"`, `"progress.json"`, `"config.json"`, and
     `"pair_spec.json"` path joins with pure helper refs.
3. Rating context identity:
   - add `rating_context_hash(spec)` separate from `rating_pool_hash`;
   - write it into config, round input, pair history, scheduler state, snapshots,
     progress, and battle summaries;
   - reject adaptive/history reuse when either hash differs.
4. Website contract split:
   - normal APIs read only committed snapshots/indexes/progress;
   - expose shard-scan fallbacks behind explicit debug/recovery helpers or a
     query flag.
5. Status/source constants:
   - centralize progress phases, run statuses, result detail modes, and battle
     page source names;
   - make the JS labels consume the same names or generate them from Python.
6. Cache helper:
   - use small functions to build cache keys and document TTL intent;
   - include selected tournament/rating ids and `fresh`/Volume reload behavior in
     tests.

## Acceptance Checks

- Unit tests assert every public artifact ref is under
  `tournaments/curvytron/<tournament_id>` and passes
  `validate_tournament_artifact_ref`.
- A tiny rating smoke writes config, input, shard summary, battle index,
  provisional latest, progress, pair history, scheduler state, and final latest.
- A changed `max_steps` or policy mode causes adaptive history reuse to be
  rejected unless the caller explicitly starts a new rating run.
- Website API tests cover three states: no ratings yet, provisional ratings, and
  final latest ratings.
- A missing `battle_index.json` returns a clear debug source and does not become
  the steady-state path for normal standings.

## Rating Context Hash Cut

Smallest useful helper:

`rating_context_hash(rating_spec)` should normalize the rating spec and hash only
fields that change game evidence, scoring, or rating math. Keep
`rating_pool_hash(checkpoints)` separate.

Include these fields:

- `formula_version`
- `policy_mode`
- `collect_temperature`
- `collect_epsilon`
- `max_steps`
- `decision_ms`
- `decision_source_frames`
- `source_physics_step_ms`
- `num_simulations`
- `policy_batch_size`
- `natural_bonus_spawn`
- `policy_trail_render_mode`
- `trail_render_mode`
- `initial_rating`
- `base_k`
- `k_reference_games`
- `k_min`
- `k_max`
- `delta_clamp`
- `draw_score`
- `min_valid_fraction`

Do not include these fields:

- checkpoint identities, because `rating_pool_hash` owns them;
- `round_count`, `pair_selection`, `pairs_per_round`, `seed`,
  `ordered_pairs`, or `include_self_pairs`, because those schedule work but do
  not change what a completed game means;
- `games_per_pair`, because old 11-game battles can still be useful evidence
  when later rounds use 21 games per pair;
- `games_per_shard`, `reuse_policies_per_shard`, GIF/frame settings, and
  `action_trace_limit`, because those are execution/storage/review details;
- `stop_max_delta`, because it only controls loop stopping.

Store `rating_context_hash` in:

- `ratings/<run>/config.json`, next to the normalized `rating_spec`;
- `rounds/<round>/input.json`;
- every rating pair spec and derived `battle.json` when available;
- top-level `pair_history.json`;
- top-level `scheduler_state.json`;
- round `results.json`, round `ratings.json`, final `latest.json`, and
  `provisional_latest.json`;
- `progress.json` so the website can display/debug the active contract cheaply.

Mismatch rules:

- `pair_history_from_pair_results(...)` should keep the existing `pool_hash`
  check and also raise if `previous_pair_history.rating_context_hash` exists and
  differs from the current spec hash.
- `select_adaptive_v0_pair_slots(...)` should raise for both `pair_history` and
  `scheduler_state` if either stored context hash exists and differs.
- `rating_snapshot_from_pair_results(...)` should raise if
  `previous_snapshot.rating_context_hash` exists and differs.
- If a pair summary/battle summary has `rating_context_hash`, the reducer should
  reject it when it differs from the current rating spec.
- Legacy artifacts without `rating_context_hash` should be allowed for now, but
  every new artifact should write the field.

Tests to add:

- `test_rating_context_hash_changes_for_gameplay_and_rating_knobs`: verify hash
  changes for `max_steps`, decision timing, `policy_mode`,
  `num_simulations`, `natural_bonus_spawn`, and Elo/scoring knobs.
- `test_rating_context_hash_ignores_scheduler_and_artifact_knobs`: verify hash
  does not change for `seed`, `pair_selection`, `pairs_per_round`,
  `games_per_pair`, `games_per_shard`, GIF settings, or frame settings.
- `test_pair_history_rejects_rating_context_hash_mismatch`.
- `test_adaptive_v0_rejects_pair_history_or_scheduler_context_hash_mismatch`.
- `test_rating_snapshot_rejects_previous_snapshot_context_hash_mismatch`.
- `test_rating_round_outputs_write_rating_context_hash` for snapshot, pair
  history, scheduler state, and result artifacts.
- `test_legacy_artifacts_without_rating_context_hash_are_accepted` for one
  transition release.

## 2026-05-13 Doc Audit Addendum

Some findings above are now partly superseded by the current local diff. Cut 1 constants for checkpoint discovery/path refs have landed locally, and `rating_context_hash` is now written/validated for rating config, round input, pair history, scheduler state, and rating snapshots. The broader recommendation to carry it through progress, pair specs, battle summaries, and top-level round results remains aspirational unless those writers are updated.

The landed per-checkpoint battle index contract is `tournaments/curvytron/<tournament_id>/checkpoints/<checkpoint_id>/battle_index.json` with website source `checkpoint_battle_index`. Older notes proposing `checkpoint_battle_indexes/<checkpoint_id>.json`, a separate checkpoint-index schema id, richer opponent-specific rows, or no visible-row shard enrichment should be treated as unresolved design notes rather than implemented contract.

Still unresolved across docs: the Gibbs context-hash field proposal differs from the landed implementation on `policy_batch_size`, rating K/initial-rating fields, and explicit score/action/timeout contract fields; the canonical field set should be decided before more adaptive artifacts depend on it.
