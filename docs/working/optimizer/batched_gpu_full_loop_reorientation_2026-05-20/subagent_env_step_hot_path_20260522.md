# Env Step Hot Path Audit - 2026-05-22

Scope: read-only inspection of `src/curvyzero/training/source_state_hybrid_observation_profile.py`
and the CurvyTron vector env/runtime files it calls. I did not edit source or
revert shared workspace changes.

## Dataflow

Closed-loop hybrid manager:

1. `HybridBatchedObservationProfileManager.step()` validates `[B,P]` actions,
   partitions rows by `actor_count`, and calls each `InProcessHybridCurvyTronActor`
   sequentially. With `native_actor_buffer=True`, actors write compact scalar fields
   directly into parent arrays; otherwise they return copied payload objects.
   See `source_state_hybrid_observation_profile.py:510-584`.
2. Each actor calls `VectorMultiplayerEnv.step(action)` and times only that call
   as `actor_env_step_sec`. The non-native actor path then copies render state and
   batch arrays; native mode skips the payload object but still pays env runtime and
   batch construction. See `source_state_hybrid_observation_profile.py:306-417`.
3. `VectorMultiplayerEnv.step()` runs public-env checks, copies pre-step state,
   translates actions, ensures random tape headroom, advances source physics, builds
   reward/done/final observation/info, then calls `_batch()`. See
   `vector_multiplayer_env.py:1003-1116`.
4. Source-frame decisions are expanded into `decision_source_frames` calls to
   `vector_runtime.step_many()`. Current profile rows using source-like settings
   therefore multiply the runtime player loop by 12 source frames per high-level
   actor step unless configured otherwise. See `vector_multiplayer_env.py:1159-1219`.
5. `_batch()` always materializes debug metadata observation, action mask, reward,
   done, terminated, truncated, final row metadata, and a large copied public `info`
   dict. The hybrid actor consumes only a small subset. See
   `vector_multiplayer_env.py:2703-2766` and `_public_info()` at
   `vector_multiplayer_env.py:3052-3241`.

## Likely Hot Loops

- `vector_runtime._step_many_kernel()` loops players in reverse order for every
  source frame, doing movement, trail append, wrap, wall check, body collision,
  print-manager update, optional bonus catch, terminal score, and tick updates.
  See `vector_runtime.py:537-919`.
- The largest algorithmic suspect is `_body_collision_rows()`. For each player and
  source frame it chooses `scan_width = max(body_write_cursor[live_rows])`, then
  builds row-by-slot arrays over `row_count x scan_width` for positions, radii,
  active slots, ownership, own-age masking, and hit checks. See
  `vector_runtime.py:5647-5682`.
- In `profile_no_death`, death is disabled, but body collision still runs. The
  collision result is only used for counters and possible detected-hit metadata
  because `death_enabled` gates death application. See `vector_runtime.py:557`,
  `vector_runtime.py:758-862`. This is likely the best low-hanging profile-only
  optimization: if death is suppressed, test a guarded fast path that skips wall
  and body collision entirely.
- Immortal/profile rows may be a measurement trap: no-death plus `borderless=True`
  on reset keeps rows alive and lets body cursors grow toward capacity, so the
  collision scan can become a sustained high-watermark workload. Real training
  with deaths/resets may have shorter trails and different steady-state cost.
  See `vector_multiplayer_env.py:696-698`.

## Full-Capacity Scans And Copies

- Body collision is not full `body_capacity` every tick, but it is full current
  max cursor across live rows per actor shard. In long no-death runs this can
  approach `DEFAULT_BODY_CAPACITY=4096`, repeated for each live player and source
  substep.
- `VectorMultiplayerEnv.step()` copies several whole-row arrays before runtime:
  `pre_alive`, `pre_death_count`, `pre_active`, timer/disabled sidecars, plus
  action sidecar fields. See `vector_multiplayer_env.py:1026-1040`.
- `_public_info()` copies many arrays every step: `episode_id`, `round_id`,
  `map_size`, `present`, `alive`, scores, death arrays, reset/random-tape arrays,
  natural bonus timer arrays, `done`, `needs_reset`, winners, etc. This all lands
  inside actor `env.step()` time even though the hybrid actor does not need most of
  it. See `vector_multiplayer_env.py:3052-3241`.
- `_batch()` then copies observation/action mask/reward/done/terminated/truncated
  again. `action_mask` is also computed inside `_source_moves_and_action_sidecar()`,
  so the public path computes or copies mask-like data more than once. See
  `vector_multiplayer_env.py:2756-2766` and `vector_multiplayer_env.py:2845-2855`.
- Non-native hybrid actors add explicit copies of render state and scalar payloads.
  Native actor buffer removes that merge/payload layer, but not the `env.step()`
  packaging cost above. See `source_state_hybrid_observation_profile.py:322-350`,
  `source_state_hybrid_observation_profile.py:375-417`, and
  `source_state_hybrid_observation_profile.py:1782-1837`.
- Parent native mode still validates row coverage each step with a full
  `native_written_rows` bool vector and row range/duplicate checks. This is small
  relative to runtime/collision but avoidable in steady-state trusted profile mode.
  See `source_state_hybrid_observation_profile.py:531-551`.

## Actor Count / Process Topology

Current `actor_count` is not true parallelism in this profile. Actors are
in-process and stepped sequentially; `actor_step_wall_sec` is the wall time for the
loop, while `actor_step_sec` is the sum of each actor env/autoreset timer. See
`source_state_hybrid_observation_profile.py:531-575`.

Topology still matters:

- More actors means smaller local `row_count`, but repeats fixed overhead:
  `VectorMultiplayerEnv.step()` validation, `_runtime_step_state()` dict creation,
  `_public_info()` copies, `_batch()` allocation, and action-mask/observation
  materialization per actor.
- Body-collision scan cost roughly sums across actor shards as
  `sum(local_live_rows * local_max_cursor)`. If all shards have similar age, this
  is close to one big batch; if one shard has a high cursor and others do not,
  sharding can reduce wasted scan width.
- A real subprocess topology could reduce wall actor physics to approximately the
  slowest actor plus IPC, but it would add serialization/shared-buffer discipline.
  The current native buffer result should be read as "payload/merge cleanup", not
  proof that actor process parallelism has been tested.

## Next 3 Experiments

1. Split `actor_env_step_sec` inside `VectorMultiplayerEnv.step()` into runtime
   kernel, public packaging, `_public_info`, `_observe_array`, `_action_mask`, and
   action-sidecar buckets. This should confirm whether the large public batch is a
   measurement trap before changing physics.
2. Add a profile-only no-death fast-path canary that skips wall/body collision
   when `death_mode=profile_no_death`, with counters showing skipped scan slots.
   Compare B512/B1024 native actor-buffer rows at the same source-frame settings.
3. Run an actor topology grid with fixed B and source frames: `actor_count` in
   `{1,2,4,8,16,32}` and log body scan slots, max body cursor per actor, public
   packaging time, and actor wall/sum. This separates sharding/cache effects from
   true process parallelism and shows whether current A16 is helping or just
   multiplying per-actor public-env overhead.
