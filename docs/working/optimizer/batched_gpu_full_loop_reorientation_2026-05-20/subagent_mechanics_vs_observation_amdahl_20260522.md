# Mechanics vs Observation Amdahl Critique - 2026-05-22

Scope: read-only critique pass over the 2026-05-20 optimizer working docs and
the relevant profile source. I did not edit source and did not touch live
training runs. The only write from this pass is this note.

## Executive read

Game mechanics are not currently proven to be the wall.

The current repeated compact-loop wall is `env_step_sec`, but that label is too
broad. In the latest split rows, `env_step_sec` mostly means compact state
handoff, renderer/observation/stack update, root-batch observation ownership,
and related synchronization. The actual `VectorMultiplayerEnv` game runtime is
small in the best available split.

So the Amdahl target is not "rewrite game rules faster" yet. The target is
state residency and observation ownership: stop rebuilding, copying, reading,
and restacking the compact visual state when the next search consumes the same
shape.

## Evidence

The strongest current row is the matched H100 timing split after the render-state
filter:

| row | closed-loop roots/sec | env fraction | search fraction | actor render-state write | observation/stack | renderer render | production->compact | actual env runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1024/P2/sim16/h64/v16/loop16/native | `15,906` | `73.4%` | `5.3%` | `0.368s` | `0.988s` | `0.728s` | `0.517s` | `0.089s` |

That row says the inclusive env bucket is the wall, but the mechanics leaf is
not. Observation/renderer/state handoff leaves are each larger than actual env
runtime. The earlier smoke had the same shape: env fraction `80.4%`, search
`4.0%`, actor render-state write `1.049s`, observation/stack `0.979s`,
renderer render `0.723s`, and actual env runtime `0.085s`; it was a timing smoke,
not a matched speed row, but it correctly predicted where the matched split
would point.

The observation-refresh-off ceiling also prices the wall directly. With the
current compact env/search/replay shape but observation refresh skipped, sim16
went from `20.7k` to `48.6k` active roots/sec, about `2.35x`, and sim32 went
from `17.9k` to `32.1k`, about `1.80x`. That is large enough to prove the
observation/render-state wall is real. It is not large enough to claim a 10x
training answer by itself.

The fast visual compact-state adapter and no-copy root observation patch are
consistent with that read. Production-to-compact fell from roughly `0.37-0.52s`
to about `0.054-0.057s`, root-build fell to about `0.009s` in sim16 no-copy
rows, and the best refresh-on compact loop improved from about `20.7k` to
`26.6k` active roots/sec. Useful, but not an architecture break; the wall moved
inward to actor render-state writes and observation/stack ownership.

The actor-count falsifier argues against "just parallelize current mechanics in
this manager." B1024/P2/sim16/loop16/native was fastest at `actor_count=1`
(`16.42k` roots/sec) and slower at `4` (`13.15k`) and `16` (`11.92k`). That does
not disprove real subprocess/native actor parallelism, but it does disprove more
in-process shards as the next aggressive move.

Older scalar env-only render trajectory data points in the same direction but is
not the current denominator. In `local_dirty_cache_trajectory_profile_20260521`,
the 100/500/1000-step rows show observation at about `78%` of wall and render at
about `76-77%`, while `vector_step_sec` is much smaller. Treat this as
background evidence that observation used to dominate the scalar wrapper, not as
proof of the current compact-loop wall.

## Source read

`mctx_synthetic_benchmark.py` records the repeated closed-loop buckets. The
closed-loop `env_step_sec` starts before
`compact_visual_manager_for_replay.step(loop_joint_action)`, includes optional
resident stack update, and is then recorded beside `root_build_sec`, `h2d_sec`,
`search_sec`, `d2h_sec`, and `replay_index_sec`
(`src/curvyzero/infra/modal/mctx_synthetic_benchmark.py:2526`,
`:2539`, `:2611`, `:2621`). This is real wall time, but it is inclusive.

`source_state_hybrid_observation_profile.py` shows why inclusive `env_step_sec`
is not synonymous with mechanics. Actors time `VectorMultiplayerEnv.step(...)`
as `actor_env_step_sec` (`:369`, `:446`), while the manager separately times
observation update (`:733`), duplicates that label as `renderer_stack_update_sec`
(`:734`), and maps renderer telemetry including
`renderer_production_to_compact_sec` (`:749`, `:764`).

`VectorMultiplayerEnv.step(...)` now exposes useful internal leaves:
`public_prepare_sec`, `runtime_sec`, `post_runtime_bookkeeping_sec`,
`reward_sec`, `final_observation_sec`, `public_info_sec`, and `batch_pack_sec`
(`src/curvyzero/env/vector_multiplayer_env.py:1051`, `:1060`, `:1091`,
`:1095`, `:1101`, `:1105`, `:1137`). These are the right timers to use before
calling mechanics the wall.

The mechanics suspect is real but currently not convicted. The runtime loops
players and still calls `_body_collision_rows(...)`; that helper scans
`live_count * scan_width`, where `scan_width` is the max live
`body_write_cursor` prefix (`src/curvyzero/env/vector_runtime.py:763`, `:769`,
`:5647`, `:5665`, `:5681`). In `profile_no_death` rows, long immortal trails
can make this a measurement trap. The current split row, however, says this is
not the dominant leaf in the matched denominator.

The persistent renderer still performs the observation/state-handoff work that
matches the measured wall: production-to-compact conversion, H2D copies,
blocking persistent update, blocking compose, and optional D2H readback
(`src/curvyzero/infra/modal/source_state_batched_observation_boundary_profile.py:2855`,
`:2864`, `:2869`, `:2893`, `:2909`, `:2914`, `:2920`, `:2924`, `:2944`).

## Trustworthy timers and rows

Trust aggregate row-result timings and `experiment_log.md` summaries that use
closed-loop records. Do not compare Amdahl slices from runner printed last-step
telemetry.

Trust these top-level repeated closed-loop buckets as wall spans:

- `total_sec` and closed-loop roots/sec: best denominator for profile-only
  compact-loop throughput.
- `env_step_sec`: real inclusive wall from next compact step through observation
  or resident stack update. Trust it as "feed the next search" time, not as
  "physics" time.
- `search_sec`: real search wall in this profile because the code blocks on
  search outputs/readback before constructing next actions.
- `root_build_sec`, `h2d_sec`, `d2h_sec`, `replay_index_sec`: useful wall spans,
  with caveats below.

Trust these nested `next_step_timings_sec` fields as attribution, not as an
exclusive additive profile:

- `actor_step_wall_sec`: real wall for the in-process actor loop.
- `actor_env_*`: real internal timings from `VectorMultiplayerEnv.step(...)`;
  these are the mechanics/package split to watch.
- `observation_sec`: real wall around manager observation update plus reset
  observation handling.
- `renderer_render_sec`: inclusive renderer telemetry.
- `renderer_production_to_compact_sec`, `renderer_host_to_device_sec`,
  `renderer_persistent_update_sec`, `renderer_device_render_sec`, and
  `renderer_device_to_host_sec`: useful real spans; the GPU-side ones are
  synchronization points because the renderer blocks.

Do not add these blindly:

- `renderer_stack_update_sec` is currently equal to `observation_sec`; it is a
  duplicate label.
- `actor_idle_wait_sec` is residual math in an in-process manager, not true
  async idle.
- `resident_stack_update_sec` is real current wall, but its explicit
  `block_until_ready` can move wait time between buckets.
- `root_build_sec` can include validation or hot-loop root-observation
  materialization that resident modes do not semantically need every step.
- `replay_index_sec` is real but validation-heavy; in the fastest matched row it
  is about `0.3%`, so it is not the wall.

The old `local_dirty_cache_trajectory_profile_20260521` rows are trustworthy
for the scalar source-state wrapper timers (`observation_sec`, `vector_step_sec`,
`physical_loop_sec`, `render_sec`) because the scalar env code records those
around physical loop and `_lightzero_observation(...)`
(`src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py:907`,
`:908`, `:923`, `:1227`, `:1236`, `:1240`, `:1246`). They should not be used as
the active compact-loop denominator.

## Fresh falsifier

Run one profile-only matched H100 B1024/P2/sim16/h64/v16/loop16 grid after the
latest no-copy/fast-visual changes, with no live trainer and no semantic speed
claim:

| row | change | purpose |
| --- | --- | --- |
| baseline | current refresh-on compact loop | denominator |
| observation-off | step compact env but skip render-state write and observation refresh; feed previous/zero validated placeholder stack | price observation/render-state wall |
| mechanics-noop | keep render-state/renderer/stack/root/search/replay shape, but replace `VectorMultiplayerEnv.step(...)` with a frozen/no-op state advance or `decision_source_frames=0` profile canary | price game mechanics/runtime |
| both-off | observation-off plus mechanics-noop | sanity upper bound |

Required reporting:

- closed-loop roots/sec and total wall;
- `env_step_sec` fraction;
- `actor_env_runtime_sec`, `actor_env_public_prepare_sec`,
  `actor_env_public_info_sec`, `actor_env_batch_pack_sec`;
- runtime phase timers, especially `body_collision_sec`, `body_scan_slots`,
  max live cursor, and source-frame count;
- `actor_render_state_write_sec`, `observation_sec`,
  `renderer_production_to_compact_sec`, `renderer_device_to_host_sec`,
  `resident_stack_update_sec`;
- active roots, terminal rows, autoreset rows, and death mode.

Falsification rule:

- If mechanics-noop wins roughly as much as observation-off, or if
  `actor_env_runtime_sec`/`body_collision_sec` remains the largest leaf after
  observation is controlled, game mechanics are the wall.
- If observation-off wins much more than mechanics-noop, and actual env runtime
  stays small, the current diagnosis holds.
- If neither wins materially, the wall is broader loop/root/search/replay glue,
  not mechanics or observation alone.

Add one normal-death/autoreset companion row before promoting the conclusion.
The current profile uses no-death shapes in several places, and immortal trails
can overprice collision scans.

## What not to optimize next

Do not polish MCTX/search kernels next. Search is `5.3%` in the matched timing
split and usually single-digit percent in the repeated compact denominator.
Even deleting it cannot deliver the next large full-loop win.

Do not optimize replay-index construction next. It is about `0.3%` in the
fastest matched row and is already cheap enough for this question.

Do not add more in-process actor shards in this manager. The actor-count row
already says `actor_count=1` is fastest for the current topology.

Do not rewrite game mechanics or body collision first. Instrument and falsify
the mechanics hypothesis with the split above; the best available matched row
says actual env runtime is small.

Do not spend another pass on output assembly, direct CTree wrapper cleanup, or
other stock LightZero object polish as the main lane. Those remain compatibility
work, but the active compact-loop wall is observation/state ownership.

Do not call resident GPU stack a speed recommendation yet. Earlier resident rows
proved plumbing, not speed. Retest after no-copy root observation and fast visual
state changes, and judge on total closed-loop wall.
