# CurvyTron stale config analysis, Newton, 2026-05-13

## Data sources

Status snapshots:

- `artifacts/local/curvytron_pruning/status_chunks_20260513d/combined_status.json`
- `artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json`

Manifest roots:

- `artifacts/local/curvytron_opponent_mixture_manifests/`
- `artifacts/local/curvytron_survivaldiag_manifests/`

Snapshot `d` file mtime was `2026-05-13T14:33:39` local. Snapshot `e` file mtime was `2026-05-13T15:20:47` local. The gap was about 47.1 minutes.

## Method

For each row present in both snapshots, I used the latest visible training checkpoint iteration:

- Prefer max `checkpoints[].iteration`.
- Fall back to status `iteration` only if checkpoints are absent.

Classification:

- `advanced`: latest visible checkpoint iteration increased from snapshot `d` to `e`.
- `stale`: latest visible checkpoint iteration stayed unchanged.

This is a checkpoint-based stale test, not proof that the trainer process is dead. In this snapshot, all 55 stale rows still had `status_heartbeat_exists=true` and `background_poller_status=running`; 54 of 55 also had `progress_exists=true`. So "stale" here means "no new visible checkpoint over the 47 minute window."

Field mapping:

- `render`: `source_state_trail_render_mode`.
- `sim`: `num_simulations`.
- `collector`: `collector_env_num`. In these rows `n_episode` matches collector count.
- `learner`: `batch_size`, shown in run ids as `l32` or `l64`.
- `repeat`: `control_noise_profile_id`.
- `mixture`: compact opponent mixture from the manifest recipe.

## Short answer

Total rows compared: 212.

| State | Count |
|---|---:|
| advanced | 157 |
| stale | 55 |

Main findings:

- The current `curvy-mix3-currentckpt-20260513a` batch is mostly healthy: 102 of 126 rows advanced; 24 were stale.
- Follow-up clue: the six rows stuck at `iteration_0` are not explained by a high checkpoint cadence. Five sampled raw `progress_latest.json` files show `learner_train_iter` between 33998 and 175528 while the latest checkpoint remains `iteration_0`.
- In mix3, render is not the explanation. `body_circles_fast` and `browser_lines` each had 12 stale rows out of 63.
- In mix3, `sim16` is heavier and more stale-skewed: 7 stale out of 20, versus 17 stale out of 106 for `sim8`.
- In mix3, high action-repeat is also stale-skewed: 13 stale out of 52, versus 8 of 52 for medium repeat and 3 of 22 for no repeat.
- In mix3, the strongest recipe cluster is `blank100`: 4 stale out of 6. This is recipe-specific, not render-specific.
- The older `curvy-mix2-clean-20260513a` batch is much staler: 22 of 52 stale. Its stale rows are recipe-specific and not explained by heavy settings, because all mix2-clean rows are `sim8`, `learner32`, `collector32`.
- The survival diagnostic rows show expected heavy-setting pain in the compute sentinels, but those do not explain most mix3 stale rows.

## Follow-up: learner moved, checkpoint stayed at k0

The new clue changes the interpretation for the rows stuck at `iteration_0`.

Snapshot `e` has six non-dependency rows whose latest visible checkpoint is still `iteration_0`. All six were already `iteration_0` in snapshot `d`, all had `progress_exists=true`, all had exactly one checkpoint, and all had only `iteration_0` as the visible train checkpoint.

Raw status files under `artifacts/local/curvytron_pruning/status_chunks_20260513f/raw_status_files/` contain `progress_latest.json` for five of those six rows. Every available raw progress file has this shape:

- `event`: `checkpoint`
- `source`: `SaveCkptHook.__call__`
- `iteration`: `0`
- `checkpoint_name`: `iteration_0.pth.tar`
- `learner_train_iter`: far above zero

Read-only source inspection confirms the meaning: `_write_checkpoint_progress_latest` uses the newest actual file in `train/lightzero_exp/ckpt` as `iteration` when a checkpoint exists, and stores the learner's counter separately as `learner_train_iter`. So this is not just a display issue. It means the learner advanced, but no newer checkpoint file was visible in the checkpoint directory used by the status/poller path.

### k0 rows

| Run id | Matrix | Recipe | Render | Sim | Learner | Collector | Repeat | Save every | Raw learner iter | Latest ckpt |
|---|---|---|---|---:|---:|---:|---|---:|---:|---|
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | mix2-clean | `r50-scr50` | fast | 8 | 32 | 32 | high | 10000 | 175528 | `iteration_0` |
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | mix2-clean | `r50-scr50` | browser | 8 | 32 | 32 | none | 10000 | 33998 | `iteration_0` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | mix3 | `r40-blank20-mid20-scr20` | fast | 8 | 32 | 32 | none | 10000 | 65588 | `iteration_0` |
| `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | mix3 | `r50-blank50` | browser | 16 | 32 | 32 | high | 10000 | 97749 | `iteration_0` |
| `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011` | mix3 | `r75-blank25` | fast | 16 | 32 | 32 | medium | 10000 | 100411 | `iteration_0` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s16-c32-l32-repM-k10-c1-s2306011` | mix3 | `r40-blank20-mid20-scr20` | fast | 16 | 32 | 32 | medium | 10000 | not in local raw sample | `iteration_0` |

The five sampled rows crossed the configured checkpoint cadence by 3.4x to 17.6x:

| Run id | `learner_train_iter / save_ckpt_after_iter` |
|---|---:|
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | 3.4 |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | 6.6 |
| `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | 9.8 |
| `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011` | 10.0 |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | 17.6 |

### k0 config comparison

The k0 rows do not share one obvious heavy setting.

| Axis | k0 rows | All rows in value | k0 % |
|---|---:|---:|---:|
| `save_ckpt_after_iter=10000` | 6 | 178 | 3.4 |
| `save_ckpt_after_iter=15000` | 0 | 33 | 0.0 |
| `body_circles_fast` | 4 | 107 | 3.7 |
| `browser_lines` | 2 | 105 | 1.9 |
| `sim8` | 3 | 187 | 1.6 |
| `sim16` | 3 | 24 | 12.5 |
| `learner32` | 6 | 187 | 3.2 |
| `learner64` | 0 | 24 | 0.0 |
| `collector32` | 6 | 187 | 3.2 |
| `collector64` | 0 | 24 | 0.0 |
| repeat none | 2 | 40 | 5.0 |
| repeat medium | 2 | 89 | 2.2 |
| repeat high | 2 | 78 | 2.6 |

Recipe split:

| Recipe | k0 rows | All rows in recipe | k0 % |
|---|---:|---:|---:|
| `r40-blank20-mid20-scr20` | 2 | 18 | 11.1 |
| `r50-scr50` | 2 | 24 | 8.3 |
| `r75-blank25` | 1 | 18 | 5.6 |
| `r50-blank50` | 1 | 24 | 4.2 |

This says:

- Cadence is not too high: all k0 rows use `save_ckpt_after_iter=10000`, and five sampled rows passed that boundary multiple times.
- Render is not the cause: both fast and browser rows appear.
- Batch/collector size are not the cause: all k0 rows are `learner32` and `collector32`; no `learner64` or `collector64` row is stuck at k0.
- `sim16` is overrepresented, but it only explains half of the k0 rows.
- Repeat is not a clean explanation: the k0 rows split evenly across none, medium, and high repeat.
- Recipe may matter as a trigger or correlate, but it is not sufficient: the same recipes also have checkpointed rows.

### Launch and checkpoint path fields

I compared the six k0 rows against the 206 rows whose latest visible checkpoint is above zero.

| Field checked | k0 rows | Checkpointed rows | Read |
|---|---:|---:|---|
| `save_ckpt_after_iter` | `10000`: 6/6 | `10000`: 172/206; `15000`: 33/206; missing in one old manifest row | Cadence is not too high for k0; sampled learner counters passed 10000 by 3.4x to 17.6x. |
| `max_train_iter` | `300000`: 6/6 | `300000`: 205/206; missing in one old manifest row | No k0 row is near the train cap. |
| `max_env_step` | `30000000`: 6/6 | `30000000`: 205/206; missing in one old manifest row | No k0-specific env cap. |
| `lightzero_eval_freq` | `0`: 6/6 | `0`: 205/206; missing in one old manifest row | Stock LightZero in-loop eval is off in both groups. |
| `eval_offline` / offline eval keys | 0/6 rows have such a key | 0/206 rows have such a key | No offline eval switch in these manifests. |
| `background_eval_enabled` / `background_gif_enabled` | `true` / `true`: 6/6 | `true` / `true`: 205/206; missing in one old manifest row | Background eval/GIF is not disabled for k0. |
| `lightzero_multi_gpu` | `false`: 6/6 | `false`: 205/206; missing in one old manifest row | No multi-GPU split in k0. |
| `rank`, `distributed`, `world_size`, `local_rank` keys | 0/6 rows | 0/206 rows | No manifest or command rank/distributed setting found. |
| `mode` | `train`: 6/6 | `train`: 206/206 | Same run mode. |
| `canonical_launcher` | canonical LightZero CurvyTron launcher: 6/6 | same: 206/206 | Same launcher. |
| `calls_stock_train_muzero` | `true`: 6/6 | `true`: 206/206 | Same stock train path claim. |
| deployed app / train function / poller function | expected app and functions: 6/6 | expected app and functions: 205/206; one old row lacks launch record | No k0-only app/function difference. |
| spawn order | `poller, train`: 6/6 | `poller, train`: 205/206; one old row lacks launch record | No k0-only launch order difference. |
| `profile_allow_auto_resume` | `false`: 6/6 | `false`: 205/206; missing in one old manifest row | No manifest setting enabling a special profile resume mode. |
| `profile_volume_commit` | `false`: 6/6 | `false`: 205/206; missing in one old manifest row | No profile volume-commit mode. |
| `stop_after_learner_train_calls` | `0`: 6/6 | `0`: 205/206; missing in one old manifest row | No stop-after-training-call cap. |
| `poller_kwargs.exp_name_ref` pattern | expected `.../<run_id>/attempts/<attempt_id>/train/lightzero_exp`: 6/6 | expected pattern: 205/206; one old row lacks the manifest field | No alternate LightZero experiment directory for k0. |
| manifest checkpoint mirror root | expected `.../<run_id>/checkpoints/lightzero`: 6/6 | expected pattern: 206/206 | No alternate checkpoint mirror root for k0. |
| status progress / heartbeat / poller refs | expected run and attempt path: 6/6 for each ref | expected run and attempt path: 206/206 for each ref | Status files are being read from the expected attempt path. |

The one healthy row with missing manifest fields is an older diagnostic row. It has visible checkpoints, so it does not explain the k0 failure mode.

I did not find a manifest field that disables checkpoint writing. The checkpoint roots also look consistent:

- Stable mirror root: `training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero`
- Training exp root: `training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/lightzero_exp`
- Progress checkpoint ref: `<training exp root>/ckpt/iteration_0.pth.tar`

For the five k0 rows with local raw heartbeat files, `command.auto_resume.enabled=true`, `found=true`, `checkpoint_iteration=0`, `resume_state_found=true`, and `resume_state_source_kind=run_resume_state_mirror`. The raw `progress_latest.json` files for those same rows point at `<training exp root>/ckpt/iteration_0.pth.tar` while reporting high `learner_train_iter`.

That is not a differing launch-manifest field. It is direct evidence that these rows saw or resumed from `iteration_0`, then the learner counter moved forward without a newer durable checkpoint becoming visible.

The more precise statement is: these k0 rows resumed from or saw `iteration_0`, then the learner counter advanced, but the durable checkpoint file set did not advance beyond `iteration_0`.

### k0 interpretation

The k0 rows are not slow-start rows. They are learner-progress rows with missing durable checkpoints.

The current best factual explanation is a checkpoint publication failure or checkpoint naming/discovery failure after learner advancement. The evidence does not identify whether the missing piece is inside LightZero checkpoint save, CurvyZero's wrapper around `SaveCkptHook`, resume interaction after preemption, or the Modal volume publication path. It does rule out the simple explanations that checkpoint cadence is too high, checkpointing is disabled in the launch manifest, or all k0 rows are just browser/heavy-batch rows.

## By matrix

| Matrix | Total | Advanced | Stale | Stale % |
|---|---:|---:|---:|---:|
| `curvy-mix3-currentckpt-20260513a` | 126 | 102 | 24 | 19.0 |
| `curvy-mix2-clean-20260513a` | 52 | 30 | 22 | 42.3 |
| `curvy-survive-bonus-large-20260513b` | 33 | 25 | 8 | 24.2 |
| `survivaldiag-v1b-20260513h` | 1 | 0 | 1 | 100.0 |

Advanced rows moved by normal checkpoint-sized jumps:

| Matrix | Advanced rows | Min delta | Median delta | Max delta |
|---|---:|---:|---:|---:|
| `curvy-mix3-currentckpt-20260513a` | 102 | 10000 | 20000 | 30000 |
| `curvy-mix2-clean-20260513a` | 30 | 10000 | 20000 | 20000 |
| `curvy-survive-bonus-large-20260513b` | 25 | 15000 | 15000 | 30000 |

## Overall config axes

These counts include all 212 rows.

| Axis | Value | Total | Advanced | Stale | Stale % |
|---|---|---:|---:|---:|---:|
| render | `body_circles_fast` | 107 | 82 | 25 | 23.4 |
| render | `browser_lines` | 105 | 75 | 30 | 28.6 |
| sim | 8 | 187 | 142 | 45 | 24.1 |
| sim | 16 | 24 | 15 | 9 | 37.5 |
| learner | 32 | 187 | 138 | 49 | 26.2 |
| learner | 64 | 24 | 19 | 5 | 20.8 |
| collector | 32 | 187 | 138 | 49 | 26.2 |
| collector | 64 | 24 | 19 | 5 | 20.8 |
| repeat | `none` | 40 | 29 | 11 | 27.5 |
| repeat | `policy_action_repeat_medium` | 89 | 72 | 17 | 19.1 |
| repeat | `policy_action_repeat_high` | 78 | 51 | 27 | 34.6 |
| repeat | `policy_action_repeat_low` | 5 | 5 | 0 | 0.0 |

One `survivaldiag-v1b` row had missing sim/learner/collector fields in the joined manifest view, so the sim/learner/collector totals above are 211 plus that one unmatched field row.

## Current mix3 batch

Matrix: `curvy-mix3-currentckpt-20260513a`.

Total: 126 rows. Advanced: 102. Stale: 24.

### Mix3 simple axes

| Axis | Value | Total | Advanced | Stale | Stale % |
|---|---|---:|---:|---:|---:|
| render | `body_circles_fast` | 63 | 51 | 12 | 19.0 |
| render | `browser_lines` | 63 | 51 | 12 | 19.0 |
| sim | 8 | 106 | 89 | 17 | 16.0 |
| sim | 16 | 20 | 13 | 7 | 35.0 |
| learner | 32 | 106 | 85 | 21 | 19.8 |
| learner | 64 | 20 | 17 | 3 | 15.0 |
| collector | 32 | 106 | 86 | 20 | 18.9 |
| collector | 64 | 20 | 16 | 4 | 20.0 |
| repeat | `none` | 22 | 19 | 3 | 13.6 |
| repeat | `policy_action_repeat_medium` | 52 | 44 | 8 | 15.4 |
| repeat | `policy_action_repeat_high` | 52 | 39 | 13 | 25.0 |

### Mix3 render and sim

| Render | Sim | Total | Advanced | Stale | Stale % |
|---|---:|---:|---:|---:|---:|
| `body_circles_fast` | 8 | 53 | 45 | 8 | 15.1 |
| `browser_lines` | 8 | 53 | 44 | 9 | 17.0 |
| `body_circles_fast` | 16 | 10 | 6 | 4 | 40.0 |
| `browser_lines` | 16 | 10 | 7 | 3 | 30.0 |

This supports the simpler read: `sim16` is heavier; render is not the main stale split inside mix3.

### Mix3 recipes and opponent mixtures

| Recipe | Mixture | Total | Advanced | Stale | Stale % |
|---|---|---:|---:|---:|---:|
| `blank100` | `blank100` | 6 | 2 | 4 | 66.7 |
| `r50-blank50` | `recent50+blank50` | 18 | 13 | 5 | 27.8 |
| `r75-blank25` | `recent75+blank25` | 18 | 14 | 4 | 22.2 |
| `r50-mid25-old25` | `recent50+mid25+old25` | 18 | 15 | 3 | 16.7 |
| `r50-scr50` | `recent50+scripted_wall_avoidant50` | 18 | 15 | 3 | 16.7 |
| `old100` | `old100` | 6 | 5 | 1 | 16.7 |
| `recent100` | `recent100` | 6 | 5 | 1 | 16.7 |
| `scr100` | `scripted_wall_avoidant100` | 6 | 5 | 1 | 16.7 |
| `r40-blank20-mid20-scr20` | `recent40+blank20+mid20+scripted_wall_avoidant20` | 18 | 16 | 2 | 11.1 |
| `mid100` | `mid100` | 6 | 6 | 0 | 0.0 |
| `r25-blank75` | `recent25+blank75` | 6 | 6 | 0 | 0.0 |

### Mix3 seed/copy check

| Copy index | Total | Advanced | Stale | Stale % |
|---:|---:|---:|---:|---:|
| 1 | 73 | 57 | 16 | 21.9 |
| 2 | 19 | 14 | 5 | 26.3 |
| 3 | 5 | 5 | 0 | 0.0 |
| 4 | 8 | 8 | 0 | 0.0 |
| 5 | 21 | 18 | 3 | 14.3 |

There is no single seed-only failure visible. Copy 2 is somewhat worse, but copy coverage is uneven and the stale rows are better explained by recipe plus heavier settings.

## Older mix2-clean batch

Matrix: `curvy-mix2-clean-20260513a`.

Total: 52 rows. Advanced: 30. Stale: 22.

All mix2-clean rows are `sim8`, `learner32`, `collector32`, so these stale rows are not explained by the known heavy settings.

| Axis | Value | Total | Advanced | Stale | Stale % |
|---|---|---:|---:|---:|---:|
| render | `body_circles_fast` | 26 | 17 | 9 | 34.6 |
| render | `browser_lines` | 26 | 13 | 13 | 50.0 |
| repeat | `none` | 14 | 6 | 8 | 57.1 |
| repeat | `policy_action_repeat_medium` | 24 | 18 | 6 | 25.0 |
| repeat | `policy_action_repeat_high` | 14 | 6 | 8 | 57.1 |

Recipe clusters:

| Recipe | Mixture | Total | Advanced | Stale | Stale % |
|---|---|---:|---:|---:|---:|
| `r50-scr50` | `recent50+scripted_wall_avoidant50` | 6 | 0 | 6 | 100.0 |
| `r50-blank25-scr25` | `recent50+blank25+scripted_wall_avoidant25` | 6 | 1 | 5 | 83.3 |
| `r50-mid50` | `recent50+mid50` | 6 | 1 | 5 | 83.3 |
| `r50-old50` | `recent50+old50` | 6 | 2 | 4 | 66.7 |
| `r50-blank20-mid15-scr15` | `recent50+blank20+mid15+scripted_wall_avoidant15` | 6 | 5 | 1 | 16.7 |
| `r50-mid25-old25` | `recent50+mid25+old25` | 6 | 5 | 1 | 16.7 |
| `r50-blank50` | `recent50+blank50` | 6 | 6 | 0 | 0.0 |
| `blank100` | `blank100` | 2 | 2 | 0 | 0.0 |
| `mid100` | `mid100` | 2 | 2 | 0 | 0.0 |
| `old100` | `old100` | 2 | 2 | 0 | 0.0 |
| `recent100` | `recent100` | 2 | 2 | 0 | 0.0 |
| `scr100` | `scripted_wall_avoidant100` | 2 | 2 | 0 | 0.0 |

This is the clearest recipe-specific stale cluster in the data. It is not just browser rendering, and it is not sim16.

## Survival diagnostic rows

Matrix: `curvy-survive-bonus-large-20260513b`.

Total: 33 rows. Advanced: 25. Stale: 8.

| Axis | Value | Total | Advanced | Stale | Stale % |
|---|---|---:|---:|---:|---:|
| render | `body_circles_fast` | 17 | 14 | 3 | 17.6 |
| render | `browser_lines` | 16 | 11 | 5 | 31.2 |
| sim | 8 | 29 | 23 | 6 | 20.7 |
| sim | 16 | 4 | 2 | 2 | 50.0 |
| learner | 32 | 29 | 23 | 6 | 20.7 |
| learner | 64 | 4 | 2 | 2 | 50.0 |
| collector | 32 | 29 | 22 | 7 | 24.1 |
| collector | 64 | 4 | 3 | 1 | 25.0 |
| repeat | `none` | 4 | 4 | 0 | 0.0 |
| repeat | `policy_action_repeat_low` | 5 | 5 | 0 | 0.0 |
| repeat | `policy_action_repeat_medium` | 12 | 10 | 2 | 16.7 |
| repeat | `policy_action_repeat_high` | 12 | 6 | 6 | 50.0 |

Recipe clusters:

| Recipe | Total | Advanced | Stale | Stale % |
|---|---:|---:|---:|---:|
| `h20_blank_canvas_all_stochasticity_large_repeat` | 9 | 9 | 0 | 0.0 |
| `h21_blank_canvas_medium_high_extra_repeat` | 4 | 2 | 2 | 50.0 |
| `h22_passive_immortal_dirty_control_large` | 8 | 7 | 1 | 12.5 |
| `h23_blank_canvas_small_compute_sentinels` | 12 | 7 | 5 | 41.7 |

This supports the expected rule that heavy sentinels can be slow or stale. It does not explain the current mix3 batch as a whole.

## Stale examples

Representative stale rows:

| Matrix | Recipe | Run id | Render | Sim | Learner | Collector | Repeat | Seed | Iter d -> e |
|---|---|---|---|---:|---:|---:|---|---:|---|
| mix3 | `blank100` | `curvy-mix3cur-blank100-rf-s8-c32-l32-rep0-k10-c2-s2308021` | fast | 8 | 32 | 32 | none | 2308021 | 60000 -> 60000 |
| mix3 | `blank100` | `curvy-mix3cur-blank100-rb-s8-c32-l32-repM-k10-c1-s2308011` | browser | 8 | 32 | 32 | medium | 2308011 | 80000 -> 80000 |
| mix3 | `r50-blank50` | `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | browser | 16 | 32 | 32 | high | 2302011 | 0 -> 0 |
| mix3 | `r75-blank25` | `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repH-k10-c1-s2303011` | fast | 16 | 32 | 32 | high | 2303011 | 50000 -> 50000 |
| mix3 | `r40-blank20-mid20-scr20` | `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s16-c32-l32-repM-k10-c1-s2306011` | fast | 16 | 32 | 32 | medium | 2306011 | 0 -> 0 |
| mix2 | `r50-scr50` | `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-rep0-k10-c3-s2104031` | fast | 8 | 32 | 32 | none | 2104031 | 20000 -> 20000 |
| mix2 | `r50-scr50` | `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-repH-k10-c2-s2104021` | browser | 8 | 32 | 32 | high | 2104021 | 130000 -> 130000 |
| mix2 | `r50-mid50` | `curvy-mix2clean-r50-mid50-rb-s8-c32-l32-rep0-k10-c2-s2102021` | browser | 8 | 32 | 32 | none | 2102021 | 100000 -> 100000 |
| survive | compute sentinel | `curvy-survive-bonus-blank-browser-heavy-search16-r290-s1141774` | browser | 16 | 32 | 32 | high | 1141774 | 165000 -> 165000 |
| survive | compute sentinel | `curvy-survive-bonus-blank-fast-heavy-batch64-r299-s1141671` | fast | 8 | 64 | 32 | high | 1141671 | 105000 -> 105000 |

## Interpretation

The stale rows are not one single failure mode.

For the current mix3 batch, the strongest facts are:

1. Render is not the cause: both render modes have exactly 12 stale rows.
2. `sim16` is a real heavier-setting risk.
3. High action-repeat is a real stale-skewed setting.
4. `blank100` is the strongest recipe cluster.
5. `learner64` and `collector64` are not stale-skewed in this window.

For the older mix2-clean batch, the stale rows are more recipe-specific than compute-specific:

1. `r50-scr50` is 6 of 6 stale.
2. `r50-blank25-scr25` and `r50-mid50` are each 5 of 6 stale.
3. All mix2-clean rows are `sim8`, `learner32`, `collector32`, so this is not a heavy-knob explanation.

The practical read is: known heavy settings explain some stale rows, especially `sim16` and high repeat, but the bigger pattern is recipe/opponent-mixture specific. The current batch should be monitored by recipe as much as by render or compute knobs.
