# Next GPU Experiment Wave

Date: 2026-05-21

Scope: profile-only GPU/hybrid optimizer lane. Do not touch live training runs,
trainer defaults, tournament defaults, checkpoint promotion, or active Modal
runs.

## Current Read

The C512 post-patch one-process batched manager is the clean anchor:

```text
C512 real render:         ~1439.84 scalar steps/s
C512 zero observation:    ~1805.22 scalar steps/s
remaining render ceiling: ~1.25x
```

Render is still a bottleneck, but it is no longer the whole Amdahl story at
C512. A perfect observation path would not create a 5-10x win from the current
one-process manager. C768 also failed to scale cleanly: real render stayed flat
and zero observation got worse, which points at topology, scheduling, or
policy/search/manager scalarization rather than a simple renderer wall.

The local hybrid scaffold is promising only as a topology probe. Its zero rows
at B64/A4, B256/A8, and B512/A16 reached roughly `15k`, `21.6k`, and `24.9k`
scalar timesteps/s, far above the stock-loop rows. That proves actor fan-in has
headroom before policy/search/training are reintroduced; it does not yet prove
real render, subprocess IPC, or LightZero integration.

## Questions This Wave Must Answer

1. Is render still the Amdahl bottleneck?
   - Yes if real C512/C768 remains far below zero after repeat and sim changes.
   - No, or only partly, if real stays within about `25-35%` of zero while
     policy/search/manager timing buckets dominate.

2. Does hybrid actor fan-in help?
   - Yes if real-render hybrid beats the one-process C512 zero ceiling
     (`~1805 steps/s`) by at least `20%` and compact payload costs stay small.
   - No if actor merge, render-service overhead, or scalar materialization eats
     the local zero-observation headroom before any stock bridge exists.

3. Does policy/search dominate after render is cheap?
   - Yes if sim4/sim8 rows drop throughput proportionally while real-vs-zero
     gaps stay small, or if policy forward plus MCTS exceeds observation time.
   - No if renderer/stack/pack still dominates at matched sim counts.

## Prioritized Minimal Grid

| Pri | Run | Shape | Purpose | Launch only if |
|---:|---|---|---|---|
| 1 | Read active saturation grid | C512/C768 x real/zero x sim2/sim4 | Decide whether C512 is saturated and whether sim4 moves wall to search | Already running; do not restart or modify active runs |
| 2 | One-process confirm row | C512 real/zero, no RND, sim2, same source steps as anchor | Verify the `1439.84` vs `1805.22` Amdahl read if active grid is noisy or incomplete | Needed only if active grid lacks clean matched counts |
| 3 | Search pressure pair | C512 real/zero, no RND, sim4 | Check whether policy/search dominates once render is cheap | Use same workload counts and no default changes |
| 4 | Hybrid real-render smoke | B256/A8 then B512/A16, direct dynamic JAX renderer injected from Modal/profile wrapper | Test whether actor fan-in plus central batched render beats one-process zero ceiling | Hybrid wrapper exists and reports exact backend identity |
| 5 | Hybrid payload stress | B512/A8 and B512/A16, real render, `--no-pickle-payload` paired with pickle-on | Split compact payload accounting from render/service timing | Only after row/player sentinel and terminal gates pass locally |
| 6 | Policy/search stub | Best hybrid real-render shape plus batched policy/search stand-in or explicit root-batch timing | Decide whether the next wall is search/model batching, not render | Only after hybrid real render beats one-process zero by `>=20%` |

Do not launch C768 as a default next row. Use C768 only to confirm a specific
saturation hypothesis from the active grid.

## Exact Metrics To Capture

Every one-process stock row must record:

- `profile_only` run id and exact `env_manager_type`;
- exact observation backend identity and fail-closed renderer backend name;
- env steps collected, raw compact env steps, MCTS search calls, MCTS root sum,
  replay sample calls, learner train calls;
- wall seconds, `steps_per_sec`, collector seconds, policy forward collect
  seconds, MCTS seconds, learner seconds;
- manager step seconds, vector/env step seconds, surface non-render seconds;
- renderer total seconds, renderer device seconds, host-to-device seconds,
  device-to-host seconds, pack/order seconds, stack update seconds;
- scalar timestep materialization seconds and payload bytes;
- GPU max utilization and max memory;
- RND counters only for RND rows, kept separate from renderer claims.

Every hybrid row from `scripts/profile_hybrid_batched_observation_manager.py`
or its Modal wrapper must record:

- `schema_id`, `impl_id`, `profile_only`, `calls_train_muzero`,
  `trainer_defaults_changed`, and `touches_live_runs`;
- `observation_mode`, `renderer_backend_name`, batch size, actor count,
  player count, measured steps, warmup steps;
- `steps_per_sec`, `physical_rows_per_sec`, `ready_count`,
  `timestep_count`, `live_physical_row_count`, terminal/autoreset counts;
- timing buckets: `actor_step_sec`, `actor_step_wall_sec`,
  `actor_idle_wait_sec`, `parent_send_receive_sec`, `gather_merge_sec`,
  `observation_sec`, `renderer_render_sec`, `renderer_device_render_sec`,
  `renderer_stack_update_sec`, `scalar_materialization_sec`,
  `compact_payload_pickle_sec`;
- `compact_payload_bytes_per_step`,
  `compact_payload_bytes_per_timestep`, `rendered_stack_bytes_per_step`,
  and `compact_vs_rendered_stack_ratio`;
- last observation/flat obs/target reward shapes;
- row-major mapping evidence: `policy_env_id`, `policy_env_row`,
  `policy_player`, and actor global row coverage.

## Pass / Fail Reads

Renderer is still the primary Amdahl wall if:

- C512 real render is more than `35%` slower than C512 zero on repeated matched
  rows;
- renderer plus stack/pack is the largest timed bucket at sim2 and sim4;
- policy forward plus MCTS does not grow into the dominant bucket under sim4.

Renderer is no longer the main next lever if:

- C512 real render remains within `25-35%` of zero;
- C768 stays flat or worse;
- zero rows spend most time in manager, policy forward, MCTS, scalarization, or
  learner plumbing.

Hybrid actor fan-in passes if:

- real-render hybrid B512/A16 exceeds `~2166 steps/s` (`1.2x` the C512
  one-process zero ceiling) with correct backend identity;
- compact payload bytes per timestep are far below rendered-stack bytes per
  timestep;
- actor idle/wait plus gather/merge does not dominate total wall;
- row/player/global-row sentinel checks and terminal/final-observation gates
  pass before interpreting speed.

Hybrid actor fan-in fails or pauses if:

- real-render hybrid cannot beat the one-process C512 zero ceiling;
- `parent_send_receive_sec`, `gather_merge_sec`, or scalar materialization
  dominates before subprocess IPC is even introduced;
- the path needs rendered stacks crossing actor boundaries;
- terminal final observations are still `None` or rendered after autoreset.

Policy/search becomes the next lane if:

- sim4/sim8 throughput drops while real-vs-zero observation gaps stay small;
- policy forward collect plus MCTS exceeds observation/render buckets;
- root/search counts match but search wall scales poorly with simulation count.

## Guardrails

- Profile-only rows only. No `train_muzero` calls from the hybrid harness.
- No trainer defaults, tournament defaults, checkpoint metadata, eval cadence,
  GIF jobs, or live run state changes.
- Do not kill, restart, or alter active Modal runs for this wave.
- Do not promote `policy_observation_backend=jax_gpu` or the current batched
  manager to Coach.
- Keep RND as a separate axis. RND meter rows are overhead/safety reads, not
  renderer reads.
- Treat deltas below `10%` as noise unless repeated with matching workload
  counts.
- Backend identity must fail closed. A GPU-profile row without the expected
  injected renderer is invalid, not a CPU fallback.
- Preserve row-major scalar order: `env_id = row * player_count + player`.
- Terminal visual `final_observation` must be captured before autoreset in any
  renderer-backed hybrid row used for architecture decisions.

## Recommended Launch Order

1. Let the existing detached C512/C768 real/zero sim2/sim4 grid finish and read
   it first.
2. If the grid confirms C512 saturation, launch the minimal Modal/profile-only
   hybrid real-render smoke: B256/A8, then B512/A16, with warmup and compact
   output.
3. If B512/A16 real render clears `~2166 steps/s`, run the paired
   pickle-on/pickle-off payload stress.
4. If hybrid render clears the payload stress, add a policy/search stub or
   sim-pressure row before designing any stock LightZero bridge.
5. If hybrid render does not clear the one-process zero ceiling, stop short of
   a stock bridge and decide between targeted render/stack cleanup and deeper
   device-resident env/search work.
