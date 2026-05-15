# Contract Drift Audit - 2026-05-15

Scope: post-fix audit for the current `source_state_fixed_opponent` controlled-player contract. This is a docs-only audit; no runtime behavior was changed here.

## Current Contract

- Policy surface: `browser_lines + simple_symbols`.
- Reliable backend: `cpu_oracle`.
- Lab-only backend: scalar `jax_gpu`.
- Perspective: controlled-player view; the selected physical player is SELF.
- LightZero `to_play`: always `-1` for current non-board-game source-state rows.
- Tournament policy input: seat `N` receives `observation[0,N]`, `action_mask[0,N]`, and `to_play=-1`.

Primary source refs:

- `src/curvyzero/env/observation_surface_contract.py:16` defines the policy observation contract id.
- `src/curvyzero/env/observation_surface_contract.py:27` and `:28` pin `browser_lines` and `simple_symbols`.
- `src/curvyzero/env/observation_surface_contract.py:54` through `:65` name `cpu_oracle`, scalar `jax_gpu`, and the batched-GPU-not-scalar direction.
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:209` through `:215` restrict the current source-state env to the policy trail/bonus modes while still listing policy backends.
- `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1352` through `:1358` returns LightZero observations with `to_play=-1`.
- `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3469` through `:3475` passes per-seat visual/action-mask slices and `to_play=-1`.

## Remaining Drift Surfaces

1. Frozen checkpoint opponent provider still forwards seat id as `to_play`.

   `LightZeroCheckpointOpponentProvider.select_action(...)` receives `player_id` and `_policy_eval_forward(...)` sends `to_play=[int(player_id)]`.

   Refs:

   - `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:63` through `:87`
   - `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py:367` through `:388`
   - caller builds opponent-player visual slice at `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:1519` through `:1530`
   - historical two-seat smoke paths also preserve player-id `to_play` at `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py:2162` through `:2168`, `:2388` through `:2398`, `:2744` through `:2753`, and `:2843` through `:2849`; do not use those as precedent for current `source_state_fixed_opponent`

   Risk: training with a frozen learned opponent can evaluate that opponent under board-game-style player ids, while tournament/eval use `-1`. That is a real trainer/eval divergence if LightZero uses `to_play` in backup, value sign, or policy plumbing.

   Tiny safe patch recommendation: change `_policy_eval_forward` to pass `to_play=[-1]`, keep `player_id` only as sidecar/metadata, and add a provider unit test that spies on `eval_mode.forward`.

2. Backend is not carried into tournament/checkpoint identity.

   Training and env metadata can name `policy_observation_backend`, but tournament checkpoint normalization, rating pool hashing, rating roster rows, and checkpoint payload extraction only preserve trail/bonus modes, not backend.

   Refs:

   - backend is recorded in training command metadata at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3938` through `:3948`
   - env info records backend at `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:2132` through `:2140`
   - `normalize_checkpoint_spec` returns trail/bonus/contract but no backend at `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:226` through `:249`
   - `rating_pool_hash` hashes trail/bonus but no backend at `src/curvyzero/tournament/curvytron/contracts.py:205` through `:220`
   - `rating_roster_by_checkpoint` preserves trail/bonus but no backend at `src/curvyzero/tournament/curvytron/contracts.py:223` through `:240`
   - checkpoint metadata extraction reads trail/bonus only at `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:2644` through `:2718`
   - tournament loads a CPU `SourceStateGray64Stack4` regardless of checkpoint backend at `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3430` through `:3438`

   Risk: a scalar `jax_gpu` lab checkpoint and a `cpu_oracle` checkpoint can collapse into the same tournament/rating identity, while tournament evaluates both with CPU stack rendering. If the scalar GPU path is ever approximate or bugged, the rating result hides the training/eval mismatch.

   Tiny safe patch recommendation: add `policy_observation_backend` to checkpoint specs, rating roster rows, rating hash/context, checkpoint payload extraction, game summaries, and public leaderboard metadata. Reject non-`cpu_oracle` in production tournament unless the rating context explicitly marks a lab backend.

3. Readiness and surface gates do not validate all observation-surface metadata.

   `_extract_surface(...)` pulls backend and observation contract fields, but `_validate_visual_survival_surface(...)` checks only trail/bonus and older surface fields. The source-state fixed-opponent readiness gate also omits policy trail/bonus/backend and the observation contract.

   Refs:

   - extracted but not fully validated: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6295` through `:6301`
   - validation expected fields include trail/bonus but not backend/contract: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6415` through `:6420`
   - validation loop is driven only by that expected dict: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6477` through `:6480`
   - readiness expected fields omit observation surface: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:6484` through `:6518`

   Risk: command and compiled env can drift on backend/contract metadata without failing the local readiness proof. This is exactly where hidden fallback bugs tend to survive.

   Tiny safe patch recommendation: add `policy_observation_backend`, `policy_trail_render_mode`, `policy_bonus_render_mode`, `policy_observation_contract_id`, and selected `observation_contract` keys to both checks.

4. Scalar `jax_gpu` is still a normal accepted train flag, despite lab-only status.

   The current code validates it as one of the allowed policy observation backends and the local env can initialize it. Docs say it is a canary and too slow, but the launcher does not prevent `mode="train"` or subprocess use.

   Refs:

   - backend choices include CPU and GPU at `src/curvyzero/env/observation_surface_contract.py:54` through `:65`
   - launcher validates the backend choice at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:600` through `:607`
   - `_run_visual_survival_train` accepts it for train configs at `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py:3708` through `:3716`
   - env rejects non-policy surface for `jax_gpu` but otherwise initializes scalar GPU at `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:623` through `:640`

   Risk: an operator can launch scalar `jax_gpu` as if it were the production backend. Current docs say subprocess scalar GPU failed and CPU is faster; the code should make that status explicit.

   Tiny safe patch recommendation: require an explicit `policy_observation_backend_lab_canary=true` or profile-only mode for scalar `jax_gpu`, and fail fast for subprocess env managers until a batched production backend exists.

5. GIF/eval metadata is mostly separated, but a generic `trail_render_mode` alias remains.

   Tournament policy observations use per-seat `SourceStateGray64Stack4`; GIFs use full 704 RGB browser-lines frames for humans. That separation is intentional. The summary still emits a generic `trail_render_mode` alias beside `policy_trail_render_mode` and `gif_trail_render_mode`.

   Refs:

   - policy stack construction: `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3430` through `:3438`
   - policy input slicing: `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3469` through `:3485`
   - GIF render mode is fixed to browser lines at `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3411`
   - summary emits policy and GIF fields at `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3593` through `:3650`
   - generic alias: `src/curvyzero/tournament/curvytron_checkpoint_tournament.py:3623` through `:3628`

   Risk: downstream scripts can read `trail_render_mode` as policy or GIF depending on older habits. That can make eval/GIF artifacts look contract-compatible when only the human render is.

   Tiny safe patch recommendation: keep the alias only under an explicit legacy/deprecated key, or add a `trail_render_mode_alias_scope="policy_observation_legacy"` field and move new consumers to `policy_trail_render_mode` and `gif_trail_render_mode`.

6. Current docs still contain stale recommendations, but the active read path now points at this audit.

   Refs:

   - repaired during this audit: `docs/working/training/leaderboard_to_training_2026-05-13/tournament_eval_seat_perspective_audit_2026-05-15.md:20`, `:27`, `:63`, and `:64` now warn that player-id `to_play` is historical/provider drift, not the current contract.
   - stale current fast/body-circles language: `docs/working/training/leaderboard_to_training_2026-05-13/active_plan_2026-05-15.md:95` through `:113`, `:145`, `:152`, and `:206`
   - stale render/eval parity notes for the prior CPU fast lane: `docs/working/training/leaderboard_to_training_2026-05-13/render_eval_parity_audit_2026-05-15.md:7` through `:29` and `:45` through `:56`
   - stale proof/fallback body-circles language: `docs/working/training/leaderboard_to_training_2026-05-13/FULL_LOOP_PROOF.md:281` through `:288` and `:581` through `:590`
   - stale tournament-debug body-circles language: `docs/working/training/leaderboard_to_training_2026-05-13/TOURNAMENT_DEBUG.md:169`, `:183`, `:195`, `:208`, and `:260`

   Risk: a human copying from these docs can relaunch `body_circles_fast + simple_symbols` or reintroduce player-id `to_play` despite the current contract.

   Tiny safe patch recommendation: do not bulk-edit historical notes; add a short superseded banner to the most visible stale files and keep `README.md`, `NOW.md`, `OPERATING_PATTERN.md`, `policy_observation_perspective_contract_2026-05-15.md`, and this audit as the read path.

## Tests To Add

- Frozen provider `to_play` spy: assert both opponent player 0 and player 1 call `eval_mode.forward(..., to_play=[-1])`.
- Backend metadata round-trip: train command -> checkpoint discovery -> normalized checkpoint spec -> rating roster/hash -> game summary all preserve `policy_observation_backend`.
- Readiness gate negative test: mutate compiled env `policy_observation_backend` or `observation_contract.backend` and assert the gate fails.
- Tournament metadata test: assert policy surface, GIF surface, and deprecated generic aliases cannot be confused.
- Scalar GPU guard test: production train/subprocess launch rejects scalar `jax_gpu` without the explicit lab canary switch.
