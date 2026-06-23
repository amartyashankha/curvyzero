# Next Compact Closed-Loop Profile Grid

Date: 2026-05-22

Status: profile-only plan. Do not launch from this document. Do not touch live
Coach training runs, checkpoints, eval jobs, GIF jobs, tournaments, or active
Modal apps.

Latest read: this is now mainly a reproducibility grid, not a blind launch
plan. Fresh H100 rows already show that resident stack wins after root-copy
removal and that game mechanics are not the wall. Use this grid when the
environment changes again or when we need a same-denominator repeat before the
next state-ownership patch.

## Goal

Separate three things that were getting blurred together:

```text
real game mechanics
vs observation/render/stack handoff
vs MCTX search pressure
```

Use the same denominator for every row:

```text
HybridCompactBatch
-> compact env step
-> renderer/observation/stack update
-> CompactRootBatchV1
-> MCTX/JAX search
-> CompactSearchResultV1 / CompactReplayIndexRowsV1
```

These rows call `src/curvyzero/infra/modal/mctx_synthetic_benchmark.py`. They
do not call `train_muzero`.

## Fixed Defaults

Use the current fast visual/no-copy compact profile defaults explicitly:

```text
--observation-mode curvytron_hybrid_compact_visual_sample
--observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile
--batch-size 1024
--actor-count 1
--player-count 2
--body-capacity 4096
--hidden-dim 64
--max-depth 16
--rollout-steps 4
--closed-loop-steps 32
--warmup-runs 8
--steady-runs 12
--native-actor-buffer
--closed-loop-replay-index
--no-compact-root-copy-observation
--no-emit-full-json
```

Notes:

- `--batch-size 1024` is physical env rows; with two players this is 2048
  possible root rows.
- `--actor-count 1` is deliberate. The current in-process actor-count
  falsifier made `actor_count=1` fastest in this harness. This does not prove
  future subprocess/native actors are bad.
- `--compact-visual-observation-source host` is the host-stack path.
- `--compact-visual-observation-source resident_gpu` is the resident stack path.
- `--no-compact-root-copy-observation` keeps the no-copy root-batch behavior.
- Resident plus refresh-off is a profile-only ceiling/control. The benchmark
  now uses a zero resident stack in that case because observation refresh is
  intentionally skipped and there is no fresh renderer device output to read.
- `warmup-runs=8`, `steady-runs=12`, and `closed-loop-steps=32` are above smoke
  size so first compile, first render, and short-loop noise do not drive the
  read.

## Primary H100 Grid

Run H100 first if this grid needs to be repeated. This is the full 2 x 2 x 2
split:

| row | stack source | sims | refresh | falsifies |
| --- | --- | ---: | --- | --- |
| H1 | host | 16 | on | Current host-stack fast visual/no-copy anchor. If this drifts, debug measurement before interpreting the grid. |
| H2 | resident_gpu | 16 | on | Host stack/H2D/root observation handoff as the wall. Current evidence says resident should win. |
| H3 | host | 16 | off | Observation refresh as the wall. If refresh-off jumps while actor runtime stays small, game mechanics are not the wall. |
| H4 | resident_gpu | 16 | off | Resident-stack maintenance tax with observation refresh removed. |
| H5 | host | 32 | on | Search pressure under the current host observation path. |
| H6 | resident_gpu | 32 | on | Whether residency still helps when search is heavier. Current evidence says yes. |
| H7 | host | 32 | off | Search with observation mostly removed. If this is still env-bound, search is not the next wall. |
| H8 | resident_gpu | 32 | off | Resident/search control with observation refresh removed. |

### Command Template

Replace the three axis values for each row:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.mctx_synthetic_benchmark \
  --compute h100 \
  --observation-mode curvytron_hybrid_compact_visual_sample \
  --observation-renderer-backend jax_gpu_persistent_policy_framebuffer_profile \
  --compact-visual-observation-source <host|resident_gpu> \
  --batch-size 1024 \
  --actor-count 1 \
  --player-count 2 \
  --body-capacity 4096 \
  --hidden-dim 64 \
  --rollout-steps 4 \
  --num-simulations <16|32> \
  --max-depth 16 \
  --closed-loop-steps 32 \
  --warmup-runs 8 \
  --steady-runs 12 \
  --native-actor-buffer \
  --closed-loop-replay-index \
  <refresh-flag> \
  --no-compact-root-copy-observation \
  --no-emit-full-json
```

Use:

```text
--hybrid-refresh-observation-stack
```

for refresh-on rows, and:

```text
--no-hybrid-refresh-observation-stack
```

for refresh-off rows.

## Optional L4 Repeat

Only repeat on L4/T4 after H100 is understood. Start with four rows:

| row | change from H100 row | why |
| --- | --- | --- |
| L1 | H1 with `--compute l4` | Cheap host-stack anchor. |
| L2 | H2 with `--compute l4` | Checks whether resident stack still wins on cheaper hardware. |
| L3 | H5 with `--compute l4` | Only needed if H100 sim32 makes search decision-relevant. |
| L4 | H6 with `--compute l4` | Resident/search interaction sanity on L4/T4. |

Do not run the full L4 matrix unless H100 gives an ambiguous ordering.

## Manifest Shape

If this becomes a small runner, keep the manifest boring and explicit:

```json
{
  "grid_id": "opt-compact-closed-loop-split-h100-20260522a",
  "profile_only": true,
  "touches_live_runs": false,
  "calls_train_muzero": false,
  "module": "curvyzero.infra.modal.mctx_synthetic_benchmark",
  "base_args": {
    "observation_mode": "curvytron_hybrid_compact_visual_sample",
    "observation_renderer_backend": "jax_gpu_persistent_policy_framebuffer_profile",
    "batch_size": 1024,
    "actor_count": 1,
    "player_count": 2,
    "body_capacity": 4096,
    "hidden_dim": 64,
    "max_depth": 16,
    "rollout_steps": 4,
    "closed_loop_steps": 32,
    "warmup_runs": 8,
    "steady_runs": 12,
    "native_actor_buffer": true,
    "closed_loop_replay_index": true,
    "compact_root_copy_observation": false,
    "emit_full_json": false
  },
  "rows": [
    {"id": "H1", "compute": "h100", "compact_visual_observation_source": "host", "num_simulations": 16, "hybrid_refresh_observation_stack": true},
    {"id": "H2", "compute": "h100", "compact_visual_observation_source": "resident_gpu", "num_simulations": 16, "hybrid_refresh_observation_stack": true},
    {"id": "H3", "compute": "h100", "compact_visual_observation_source": "host", "num_simulations": 16, "hybrid_refresh_observation_stack": false},
    {"id": "H4", "compute": "h100", "compact_visual_observation_source": "resident_gpu", "num_simulations": 16, "hybrid_refresh_observation_stack": false},
    {"id": "H5", "compute": "h100", "compact_visual_observation_source": "host", "num_simulations": 32, "hybrid_refresh_observation_stack": true},
    {"id": "H6", "compute": "h100", "compact_visual_observation_source": "resident_gpu", "num_simulations": 32, "hybrid_refresh_observation_stack": true},
    {"id": "H7", "compute": "h100", "compact_visual_observation_source": "host", "num_simulations": 32, "hybrid_refresh_observation_stack": false},
    {"id": "H8", "compute": "h100", "compact_visual_observation_source": "resident_gpu", "num_simulations": 32, "hybrid_refresh_observation_stack": false}
  ],
  "optional_l4_repeat": {
    "row_ids": ["H1", "H2", "H5", "H6"],
    "compute": "l4"
  }
}
```

## Readout Fields

Extract aggregate compact summary fields first. Do not use only the last-step
sample:

```text
closed_loop.active_roots_per_sec
closed_loop.slowest_bucket
closed_loop.bucket_totals_sec.env_step_sec
closed_loop.bucket_totals_sec.root_build_sec
closed_loop.bucket_totals_sec.h2d_sec
closed_loop.bucket_totals_sec.search_sec
closed_loop.bucket_totals_sec.d2h_sec
closed_loop.bucket_totals_sec.replay_index_sec
closed_loop.bucket_fraction_of_total
closed_loop.next_step_bucket_totals_sec.actor_env_runtime_sec
closed_loop.next_step_bucket_totals_sec.actor_render_state_write_sec
closed_loop.next_step_bucket_totals_sec.observation_sec
closed_loop.next_step_bucket_totals_sec.renderer_production_to_compact_sec
closed_loop.next_step_bucket_totals_sec.renderer_persistent_delta_pack_sec
closed_loop.next_step_bucket_totals_sec.renderer_host_to_device_sec
closed_loop.next_step_bucket_totals_sec.renderer_device_render_sec
closed_loop.next_step_bucket_totals_sec.renderer_device_to_host_sec
closed_loop.next_step_bucket_totals_sec.renderer_stack_update_sec
closed_loop.next_step_bucket_totals_sec.stack_shift_sec
closed_loop.next_step_bucket_totals_sec.stack_latest_update_sec
closed_loop.next_step_bucket_totals_sec.resident_stack_update_sec
```

## Decision Rules

- Mechanics win only if refresh-off rows remain slow and
  `actor_env_runtime_sec` owns the wall.
- Observation wins if refresh-on rows are much slower than refresh-off rows and
  renderer/stack buckets explain the delta.
- Resident stack wins if resident refresh-on rows beat host refresh-on rows
  after no-copy root batches. Fresh evidence says this is true now.
- Search wins if sim32 raises `search_sec` enough to change the slowest bucket
  or materially lower active roots/sec in both refresh-on and refresh-off rows.
- If resident rows win but env_step remains the wall, the next target is compact
  render-state ownership before root batch/search, not replay-index rows and not
  raw GPU drawing.
