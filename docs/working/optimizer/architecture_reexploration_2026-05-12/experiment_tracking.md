# Experiment Tracking

Date: 2026-05-12

Purpose: keep 20-100 row optimizer profiling grids trackable without launching
or losing Modal jobs. This is a planning and bookkeeping contract for the
trusted stock LightZero path:

```text
lzero.entry.train_muzero
--mode train
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
```

The old custom `two-seat-selfplay` path is historical. Do not mix those rows
into this manifest except as explicitly labeled postmortem baselines.

## Manifest Row Fields

Each planned row should be represented before launch in a JSON/JSONL row, a
markdown table, or both. Use one row per attempted run/attempt, not one row per
family.

| Field | Meaning |
| --- | --- |
| `experiment_id` | Stable grid id, for example `opt-stock-frozen-scale-20260512j`. Same for every row in one grid. |
| `family` | Human group such as `collector_width`, `render_ablation`, `sim_ladder`, `death_mode`, `hardware_ladder`, or `opponent_ref`. |
| `run_id` | Modal/training run id. Must be unique across the volume. |
| `attempt_id` | Attempt id under the run. Include the row number and short cell label. |
| `command` | Full reviewed command text. This is the source of truth for launch reproduction. |
| `hardware` | Compute target and CPU shape, for example `gpu-l4-t4`, `gpu-l4-t4-cpu40`, or `gpu-h100-cpu40`. |
| `collectors` | `collector_env_num`; include `evaluator_env_num` separately if non-default. |
| `n_episode` | LightZero collector episode count per collection call. Match this to collector width for subprocess scale profiles unless intentionally testing otherwise. |
| `sims` | `num_simulations` used by MCTS. |
| `render_mode` | Source-state trail render mode recorded for historical/profile rows. Fresh production policy rows should use `browser_lines + simple_symbols` through `cpu_oracle`; `body_circles_fast` is control-only. |
| `death_mode` | `normal` or `death_off`; record the exact flag such as `disable_death_for_profile=true`. |
| `env_manager` | `base`, `subprocess`, or any future manager name. |
| `checkpoint_ref` | Learner checkpoint start/ref if used. Prefer immutable iteration refs; never use `latest`. |
| `opponent_ref` | Frozen opponent checkpoint ref and snapshot/state key if used. Prefer immutable `iteration_N.pth.tar`. |
| `expected_metrics` | Metrics this row is meant to answer, for example `steps_per_sec`, `collector_sec`, `mcts_sec`, `telem_obs`, `telem_opp`, `gpu_max_pct`. |
| `status` | One of `planned`, `approved`, `launched`, `running`, `complete`, `failed`, `superseded`, or `do_not_run`. |
| `result_path` | Expected or actual summary ref, usually `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/summary.json`. |

Recommended optional fields: `row_id`, `seed`, `batch_size`, `max_env_step`,
`max_train_iter`, `save_ckpt_after_iter`, `reward_variant`, `lightzero_eval_freq`,
`background_eval_enabled`, `background_gif_enabled`, `profile_cuda_sync_enabled`,
`profile_allow_auto_resume`, `notes`, and `owner`.

## Result Summary Fields

Summaries should keep denominators visible so comparisons do not silently mix
short episodes, long survival, or different search budgets.

| Field | Denominator or read |
| --- | --- |
| `ok` / `status` | Whether the run completed and whether the summary is trustworthy. |
| `run_id`, `attempt_id`, `result_path` | Join keys back to the manifest. |
| `steps` | Collected env steps. This is the denominator for throughput. |
| `wall_sec` | End-to-end train/profile wall time. |
| `steps_per_sec` | `steps / wall_sec`; primary coarse throughput metric. |
| `collector_sec` | Parent collector wall time, if available. Compare within same manager. |
| `mcts_sec` | Named MCTS/search time. Also compute `root_sims_per_sec` where possible. |
| `mcts_roots`, `mcts_sim_budget` | Search denominators. Record roots and root times `sims`. |
| `policy_collect_sec`, `policy_eval_sec` | Model forward time split by collect/eval path. |
| `learner_sec`, `replay_sec`, `checkpoint_sec`, `eval_sec` | Non-collector training overhead buckets. |
| `telemetry_sec`, `telemetry_rows`, `telemetry_stride` | Artifact/telemetry overhead denominator. |
| `telem_obs`, `telem_opp`, `telem_vec` | Worker-side sampled CPU seconds for subprocess env profiling. These are summed worker CPU seconds, not parent wall seconds. |
| `render_mode`, `death_mode`, `source_max_steps` | Episode-length and observation-cost context. |
| `collectors`, `n_episode`, `env_manager` | Parallelism denominator. |
| `batch_size`, `sims`, `root_batch_mean`, `recurrent_batch_mean` | Search/model batch context. |
| `gpu_max_pct`, `gpu_mem_mib`, `cuda_available`, `cuda_device_count` | Device utilization context. |
| `problem` | First error or warning, trimmed for table display. |

Use `scripts/summarize_curvytron_lightzero_profiles.py` for current profile
summaries. It already normalizes stock LightZero and old two-seat summaries, but
stock rows should be labeled and filtered before comparing them to historical
custom-adapter rows.

## Run ID Convention

Use sortable, unique, grep-friendly ids:

```text
<lane>-<path>-<family>-<hardware>-c<collectors>-nep<n_episode>-b<batch>-sim<sims>-<death>-<render>-s<seed>-<date><suffix>
```

Example:

```text
opt-stock-frozen-scale-l4cpu40-c32-nep32-b16-sim8-normal-browser-s304-20260512j
```

Rules:

- Prefix optimizer profiling rows with `opt-`.
- Include `stock-frozen` for the trusted `source_state_fixed_opponent` plus
  frozen checkpoint lane.
- Include row numbers in `attempt_id`, for example
  `profile-07-c32-nep32-b16-sim8`.
- Keep `run_id` immutable once launched. If a command changes, create a new
  suffix such as `20260512k`, not a reused id.
- Do not use mutable words such as `latest`, `best`, or `current` in checkpoint
  or opponent refs.
- Put the most important comparison dimensions in the id: hardware, collectors,
  `n_episode`, batch, sims, death mode, render mode, and seed.

## Artifact Hygiene

Profiling grids should produce timing summaries, not a museum of GIFs.

- Set `--lightzero-eval-freq 0` for optimizer profiles unless the row is
  explicitly an eval-cost row.
- Disable background artifact work on large grids:
  `--no-background-eval-enabled --no-background-gif-enabled`.
- Use high checkpoint intervals such as `--save-ckpt-after-iter 9999` for pure
  profiles unless checkpoint cost is the metric under test.
- Keep `--output-detail compact` for grid rows.
- If GIFs are needed for a sentinel row, keep one or two rows only and cap them
  with explicit `--background-gif-max-steps` and `--background-gif-frame-stride`.
- Keep profile telemetry sampled and named. Record `env_telemetry_stride` so
  worker-side `telem_obs`, `telem_opp`, and `telem_vec` have a denominator.
- Put expected result refs in the manifest before launch so missing summaries
  are visible as `planned` or `failed`, not forgotten.

The current dry-run manifest helper already emits artifact refs and command
text without launching Modal:

```text
python3 scripts/build_curvytron_stock_train_manifest.py --stdout-only
```

Review generated commands before launch. The generated shell artifact is a
review artifact, not permission to run a grid.

## Compare Results

Compare only rows that share the same trusted path and enough denominators:
stock `train_muzero`, env variant, opponent kind/ref, death mode, render mode,
source max steps, reward variant, and artifact settings.

Recommended workflow:

1. Freeze the manifest with every row in `planned` status and exact result refs.
2. Before launch, mark only reviewed rows `approved`; leave speculative rows
   `planned` or `do_not_run`.
3. After completion, summarize by exact `run_id:attempt_id`:

```text
python3 scripts/summarize_curvytron_lightzero_profiles.py \
  --attempt <run_id>:<attempt_id>
```

4. Paste or export the normalized rows into a results table keyed by
   `experiment_id`, `family`, `run_id`, and `attempt_id`.
5. Compare within families first:
   `collector_width` by `steps_per_sec` and wall growth;
   `sim_ladder` by `root_sims_per_sec`, `mcts_sec`, and `steps_per_sec`;
   `render_ablation` by `collector_sec`, `telem_obs`, and `steps_per_sec`;
   `hardware_ladder` by same work, same collectors, same sims.
6. Separate base-manager attribution rows from subprocess scale rows. Base rows
   expose render/opponent internals; subprocess rows expose scaling but need
   worker telemetry for env attribution.
7. Report the best row, the denominator row, and the first row where scaling
   bends. Do not report learning quality from optimizer profiles.

## Minimal Markdown Tables

Manifest table:

| row | experiment_id | family | run_id | attempt_id | hardware | collectors | n_episode | sims | render_mode | death_mode | env_manager | opponent_ref | status | result_path |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 01 | `opt-stock-frozen-scale-20260512j` | `collector_width` | `...c16...` | `profile-01-c16` | `gpu-l4-t4-cpu40` | 16 | 16 | 8 | `browser_lines` | `normal` | `subprocess` | `.../iteration_32.pth.tar` | `planned` | `training/.../summary.json` |

Result table:

| row | ok | steps | wall_sec | steps_per_sec | collector_sec | mcts_sec | policy_collect_sec | learner_sec | telem_obs | telem_opp | gpu_max_pct | problem |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 01 | true | 159 | 13.80 | 11.52 | 5.15 | 0.42 | 1.78 | 2.06 | - | - | - |  |
