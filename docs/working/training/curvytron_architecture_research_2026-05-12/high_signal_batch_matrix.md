# CurvyTron High-Signal Batch Matrix

> Historical warning: this is a launch ledger and evidence record, not the
> current next-matrix plan. Early sections still describe recent frozen rows
> and search/batch sweeps before the v1d readout showed outcome saturation and
> weak survival. Outcome reward rows here are historical probes, not serious
> candidates for the next diagnostic lane. Use
> [current_source_of_truth.md](current_source_of_truth.md) for active survival
> plus bonus guidance with outcome reward off/zero.

Date: 2026-05-12

Purpose: record the reviewed CurvyTron stock `train_muzero` high-signal matrix
and its launch-surface guardrails. This started as a no-launch plan; the launch
ledger below records the generated matrix that was later spawned.

## Short Verdict

Run 8 primary jobs, all on cheap L4/T4, all stock LightZero `train_muzero`,
all labeled as fixed-opponent or frozen-opponent checks. Do not include the old
custom two-seat trainer in this batch. Do not include centralized joint-action
diagnostics until their source-state env config path is fixed and canary-tested.
Do not include the native two-seat bridge as a training lane until the actual
trainer feeds `GameSegment` / `MuZeroGameBuffer`.

Current launch artifact:

- generator: `scripts/build_curvytron_stock_train_manifest.py`
- manifest: `artifacts/local/curvytron_stock_train_manifests/stock-long-v1c.json`
- commands: `artifacts/local/curvytron_stock_train_manifests/stock-long-v1c.commands.sh`
- guardrails: `mode=train`, no `two-seat-selfplay`, immutable frozen checkpoint
  refs only, source-state render mode explicit.

Current launch surface update:

- the stock source-state fast renderer flag is
  `--source-state-trail-render-mode body_circles_fast`;
- `fast_gray64_direct` is not a stock source-state env choice and belongs to
  the old custom two-seat surface;
- every generated row uses `--mode train` through
  `curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train`;
- the trainer path imports and calls canonical `lzero.entry.train_muzero`;
- checkpoint eval and GIF generation are enabled for every row through the
  external checkpoint poller path, not through LightZero's in-loop evaluator;
- stock LightZero in-loop eval is disabled at the command surface with
  `--lightzero-eval-freq 0`, which the launcher maps to
  `policy.eval_freq=max_train_iter+1`;
- the generated commands contain no `two-seat-selfplay`, no `--two-seat-*`
  flags, and no `fast_gray64_direct`.

The batch should answer one question first:

```text
Can stock LightZero train_muzero produce any CurvyTron learning curve when the
opponent source is explicit and replay/targets stay native?
```

## Common Settings

Use these defaults unless a row overrides them:

- `mode=train`
- `compute=gpu-l4-t4`
- `env_manager_type=base`
- `collector_env_num=1`
- `evaluator_env_num=1`
- `max_train_iter=3000`
- `max_env_step=262144`
- `source_max_steps=256`
- `save_ckpt_after_iter=100`
- `lightzero_eval_freq=0`
- `background_eval_enabled=true`
- `background_eval_launch_kind=poller`
- `background_eval_seed_count=8`
- `background_eval_max_steps=4096`
- `background_eval_num_simulations=8`
- `background_eval_batch_size=64`
- `background_gif_enabled=true`
- `background_gif_max_steps=2048`
- `background_gif_frame_stride=4`
- `source_state_trail_render_mode=body_circles_fast`

Reasoning:

- L4/T4 is enough for these stock-loop proofs and keeps cost low.
- `env_manager_type=base` avoids the known frozen-checkpoint CUDA-in-subprocess
  hazard from the passed GPU canary.
- Checkpoint every 100 iterations gives roughly 30 post-initial curve points over
  3000 iterations without saving every loop.
- Stock LightZero in-loop eval stays off; the CurvyZero checkpoint poller should
  own the comparable eval panel.

## Naming

Run id:

```text
curvytron-stock-stock-high-signal-v1b-<row>-<label>
```

Attempt id:

```text
stock-high-signal-v1b-attempt-<row>-<label>
```

Never use `latest` or `ckpt_best` as a frozen opponent ref for this matrix.

## Primary Matrix

| row | lane | env variant | opponent | reward | batch | sims | seed | GIF | What it answers |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 01 | fixed-straight control | `source_state_fixed_opponent` | fixed straight | `sparse_outcome` | 32 | 8 | 410 | yes | Can the clean stock loop learn sparse win/loss against the simplest env-owned opponent? |
| 02 | reward control | `source_state_fixed_opponent` | fixed straight | `dense_survival_plus_outcome` | 32 | 8 | 410 | yes | Does dense survival create faster signal or just reward hacking against the same trivial opponent? |
| 03 | frozen core | `source_state_fixed_opponent` | recent frozen checkpoint | `dense_survival_plus_outcome` | 32 | 8 | 411 | yes | Primary practical route: stock ego policy against a recent frozen opponent with shaped survival signal. |
| 04 | frozen reward ablation | `source_state_fixed_opponent` | same recent checkpoint as row 03 | `sparse_outcome` | 32 | 8 | 411 | yes | Is sparse outcome alone enough once the opponent is not trivial? |
| 05 | opponent age | `source_state_fixed_opponent` | mid checkpoint | `dense_survival_plus_outcome` | 32 | 8 | 412 | yes | Does a less-current opponent give easier curriculum signal than the recent checkpoint? |
| 06 | opponent age | `source_state_fixed_opponent` | old checkpoint | `dense_survival_plus_outcome` | 32 | 8 | 413 | yes | Does an older/distinct opponent add signal or teach a brittle exploit? |
| 07 | search sensitivity | `source_state_fixed_opponent` | same recent checkpoint as row 03 | `dense_survival_plus_outcome` | 32 | 16 | 414 | yes | Is weak search the bottleneck, or does doubling sims not move the early curve? |
| 08 | batch sensitivity | `source_state_fixed_opponent` | same recent checkpoint as row 03 | `dense_survival_plus_outcome` | 64 | 8 | 415 | yes | Does a larger learner batch stabilize targets enough to change the curve? |

Rows 09/10 from `stock-high-signal-v1` were centralized joint-action
diagnostics. They failed at env-config validation before collection with
`opponent_policy_kind=none_centralized_joint_action`. They are quarantined from
`stock-high-signal-v1b`; relaunch them only after a focused canary proves the
joint-action env accepts its no-opponent configuration.

## Launch Ledger

2026-05-12 13:55 EDT: launched the generated `stock-high-signal-v1` matrix.

Validation before launch:

- `python3 -m py_compile` passed for the manifest generator, train launcher,
  run status tool, and eval harness.
- `uv run ruff check` passed for the touched launcher/status/eval/test files.
- `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py
  tests/test_curvytron_run_status.py
  tests/test_curvyzero_source_state_visual_survival_lightzero_env.py -q`
  reported `53 passed, 1 skipped`.

Launch command source:

```text
zsh artifacts/local/curvytron_stock_train_manifests/stock-high-signal-v1.commands.sh
```

Rows spawned:

| row | run id | attempt id |
| ---: | --- | --- |
| 01 | `curvytron-stock-stock-high-signal-v1-01-fixed-straight-sparse-b32-sim8` | `stock-high-signal-v1-attempt-01-fixed-straight-sparse-b32-sim8` |
| 02 | `curvytron-stock-stock-high-signal-v1-02-fixed-straight-dense-b32-sim8` | `stock-high-signal-v1-attempt-02-fixed-straight-dense-b32-sim8` |
| 03 | `curvytron-stock-stock-high-signal-v1-03-frozen-recent-dense-b32-sim8` | `stock-high-signal-v1-attempt-03-frozen-recent-dense-b32-sim8` |
| 04 | `curvytron-stock-stock-high-signal-v1-04-frozen-recent-sparse-b32-sim8` | `stock-high-signal-v1-attempt-04-frozen-recent-sparse-b32-sim8` |
| 05 | `curvytron-stock-stock-high-signal-v1-05-frozen-mid-dense-b32-sim8` | `stock-high-signal-v1-attempt-05-frozen-mid-dense-b32-sim8` |
| 06 | `curvytron-stock-stock-high-signal-v1-06-frozen-old-dense-b32-sim8` | `stock-high-signal-v1-attempt-06-frozen-old-dense-b32-sim8` |
| 07 | `curvytron-stock-stock-high-signal-v1-07-frozen-recent-dense-b32-sim16` | `stock-high-signal-v1-attempt-07-frozen-recent-dense-b32-sim16` |
| 08 | `curvytron-stock-stock-high-signal-v1-08-frozen-recent-dense-b64-sim8` | `stock-high-signal-v1-attempt-08-frozen-recent-dense-b64-sim8` |
| 09 | `curvytron-stock-stock-high-signal-v1-09-joint-diagnostic-b32-sim8` | `stock-high-signal-v1-attempt-09-joint-diagnostic-b32-sim8` |
| 10 | `curvytron-stock-stock-high-signal-v1-10-joint-diagnostic-b32-sim16` | `stock-high-signal-v1-attempt-10-joint-diagnostic-b32-sim16` |

Immediate health check:

- Modal apps were detached and active shortly after launch.
- Train directories and `status_heartbeat.json` existed for row 01.
- First status-table check was too early: no `progress_latest.json`,
  checkpoints, or eval manifests yet.

2026-05-12 14:19 EDT monitor:

- Rows 01-08 are active/healthy as stock training jobs: each has
  `status_heartbeat.json`, a running checkpoint poller, and LightZero
  `lightzero_exp/ckpt/iteration_*.pth.tar` checkpoints.
- Rows 09-10 stopped quickly with summaries. Both called stock
  `train_muzero`, but failed before training because
  `source_state_joint_action` passed `opponent_policy_kind=none_centralized_joint_action`
  into a config path that only accepts `fixed_straight` or
  `frozen_lightzero_checkpoint`.
- The compact run-status tool is not useful for these stock rows yet because it
  expects custom `progress_latest.json`; use `status_heartbeat.json`,
  `checkpoint_eval_poller.json`, and `lightzero_exp/ckpt/` for this batch.
- Checkpoint eval and GIF jobs are being spawned, but current eval/GIF artifacts
  are failing with
  `_build_visual_survival_configs() missing 1 required keyword-only argument:
  'opponent_use_cuda'`. Treat eval/GIF curves as unhealthy until that is fixed;
  do not read failed zero-step evals as policy performance.

Follow-up health check:

- Rows 01-08 trained and produced checkpoints quickly, up to roughly
  `iteration_1100` through `iteration_2300` depending on row.
- Rows 09-10 stopped before collection because
  `source_state_joint_action` rejected
  `opponent_policy_kind=none_centralized_joint_action`.
- The v1 checkpoint eval/GIF workers failed on every checkpoint with
  `_build_visual_survival_configs() missing ... opponent_use_cuda`. That was an
  observability bug, not proof that training itself failed.
- The run status tool initially looked empty because stock run IDs were checked
  without attempt IDs. It now reads `latest_attempt.json` and falls back to the
  stock LightZero `train/lightzero_exp/ckpt` directory when counting
  checkpoints.
- Old v1 detached apps were stopped after v1b launch. Their volume artifacts
  were kept.

2026-05-12 14:14 EDT: launched patched `stock-high-signal-v1b`.

Validation before v1b launch:

- `python3 -m py_compile` passed for the manifest generator, eval harness, and
  run status tool.
- `uv run ruff check` passed for the manifest generator, eval harness, run
  status tool, and focused tests.
- `uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py
  tests/test_curvytron_run_status.py -q` reported `37 passed, 1 skipped`.

v1b changes:

- rows 09/10 removed until joint-action canary is fixed;
- checkpoint eval/GIF config now passes `opponent_use_cuda`;
- status tool can resolve latest stock attempts automatically;
- checkpoint counting handles the stock LightZero directory
  `train/lightzero_exp/ckpt`;
- every v1b command still uses `--mode train`, `body_circles_fast`,
  `--lightzero-eval-freq 0`, external checkpoint eval/GIF pollers, and no
  custom two-seat flags.

Immediate v1b health check:

- 8 detached v1b apps were active shortly after launch.
- Rows 01-05 and 07-08 had already produced checkpoints by the first status
  check; row 06 had a heartbeat and was still warming up.
- Row 02 `iteration_100` numeric checkpoint eval wrote a valid manifest:
  `ok=true`, `mean_steps=13.5`, `max_steps=24`, `failure_count=0`,
  outcome histogram `loss=5`, `win=3` across 8 seeds.
- Row 02 `iteration_100` GIF inspection wrote both views:
  `raw.gif` for greedy eval and `collect_t1.gif` for collect-mode
  temperature/epsilon sampling. Both were `ok=true`; collect-mode action
  distribution was not single-action collapsed.
- Row 02 wrote `show_in_gif_browser.flag`, so the deployed GIF browser can pick
  up the run from the volume.
- The background poller status file may show `completed_count=0` while child
  eval/GIF artifacts are already present; interpret artifact files as ground
  truth until the poller counter is tightened.
- Current focused validation after all patches:
  `57 passed, 1 skipped` for source-state env, run status, and live checkpoint
  eval plumbing tests.

2026-05-12 14:33 EDT status snapshot:

- 8 v1b detached apps were still active.
- Checkpoints ranged from `iteration_1300` to final-ish `iteration_3001`.
- Latest evaluated mean survival remained low:
  row 01 `13.5`, row 02 `10.0`, row 03 `8.0`, row 04 `8.0`,
  row 05 `6.25`, row 06 `11.375`, row 07 `6.375`, row 08 `6.25`.
- Latest eval collapse flags were true for rows 04 and 06, false for the rest.
- Row 01 reached `iteration_3001`, so v1b is a short canary, not an overnight
  training wave.
- The optimizer handoff file
  `docs/working/optimizer/coach_next_training_run_recommendations_2026-05-12.md`
  is now explicitly superseded and historical. Do not use its old
  `--mode two-seat-selfplay` / `fast_gray64_direct` matrix for Coach learning
  runs.
- Current optimizer pivot is
  `docs/working/optimizer/stock_frozen_optimizer_pivot_2026-05-12.md`:
  trusted lane is stock `train_muzero` with
  `env_variant=source_state_fixed_opponent`, frozen checkpoint opponent on CPU,
  and optimizer profiling still treats `body_circles_fast` mainly as a speed /
  attribution lens rather than a final visual-surface decision.

2026-05-12 14:53 EDT status snapshot:

- Modal showed 4 active v1b train apps and 4 stopped v1b train apps.
- Rows 01, 02, 05, and 06 completed checkpoint eval/GIF polling with
  `outstanding=0`.
- Rows 03, 04, 07, and 08 were still running with recent poller heartbeats,
  checkpoints through `iteration_2900`, `2900`, `2900`, and `2400`.
- Latest completed evals were still short-horizon: row 01 `13.5` at
  `iteration_3001`, row 02 `11.75` at `iteration_3000`, row 05 `6.125` at
  `iteration_3000`, row 06 `10.75` at `iteration_3002`.
- Completed rows wrote valid eval manifests and `raw.gif` plus `collect_t1.gif`
  GIF summaries for the latest checkpoint. Browser marker fetches succeeded for
  rows 01-08.

2026-05-12 17:00 EDT v1b final read:

- All 8 v1b rows completed their short `max_train_iter=3000` canary.
- Every row produced checkpoints plus external checkpoint eval/GIF artifacts.
- Survival stayed low and did not show a convincing learning curve.
- Latest evaluated mean survival by row:
  row 01 `13.5`, row 02 `11.75`, row 03 `8.0`, row 04 `8.0`,
  row 05 `6.125`, row 06 `10.75`, row 07 `7.0`, row 08 `6.25`.
- Row 01 had the best-looking early point around `15.875` at `iteration_1000`,
  but this did not turn into a sustained improvement.
- Interpretation: v1b proved the stock path can run and emit observability, but
  it did not prove learning. The next useful test is a longer/wider stock run,
  not more renderer speculation.

2026-05-12 17:13 EDT launched `stock-long-v1c`.

Purpose:

- keep stock LightZero `train_muzero` as the learning path;
- test whether v1b was too short and too narrow;
- use wider subprocess collectors on CPU-rich L4/T4 while keeping the learner
  and search policy on GPU and the frozen opponent on CPU;
- keep external CurvyZero checkpoint eval/GIF on and stock LightZero in-loop
  eval off.

Launch command source:

```text
zsh artifacts/local/curvytron_stock_train_manifests/stock-long-v1c.commands.sh
```

v1c settings:

- `compute=gpu-l4-t4-cpu40`
- `mode=train`
- `env_variant=source_state_fixed_opponent`
- `env_manager_type=subprocess`
- main fast rows: `body_circles_fast`, `collector_env_num=32`,
  `n_episode=32`, `batch_size=32`
- browser sentinel rows: `browser_lines`, `collector_env_num=16`,
  `n_episode=16`, `batch_size=32`
- `max_train_iter=20000`
- `max_env_step=2000000`
- `save_ckpt_after_iter=250`
- `lightzero_eval_freq=0`
- background checkpoint eval/GIF poller enabled for every row

Rows:

| row | run id | purpose |
| ---: | --- | --- |
| 01 | `curvytron-stock-stock-long-v1c-01-c32-fast-fixed-sparse-b32-sim8` | fixed-straight sparse control |
| 02 | `curvytron-stock-stock-long-v1c-02-c32-fast-fixed-dense-b32-sim8` | fixed-straight dense control |
| 03 | `curvytron-stock-stock-long-v1c-03-c32-fast-frozen-recent-dense-b32-sim8` | main frozen recent dense row |
| 04 | `curvytron-stock-stock-long-v1c-04-c32-fast-frozen-recent-sparse-b32-sim8` | frozen recent sparse ablation |
| 05 | `curvytron-stock-stock-long-v1c-05-c32-fast-frozen-old-dense-b32-sim8` | frozen old curriculum row |
| 06 | `curvytron-stock-stock-long-v1c-06-c32-fast-frozen-recent-dense-b32-sim16` | higher-search frozen recent row |
| 07 | `curvytron-stock-stock-long-v1c-07-c16-browser-frozen-recent-dense-b32-sim8` | browser-lines fidelity sentinel |
| 08 | `curvytron-stock-stock-long-v1c-08-c16-browser-fixed-sparse-b32-sim8` | browser-lines fixed control |

Immediate health check:

- Modal showed 8 new detached CurvyZero apps.
- Each app had 2 tasks, matching one training function plus one poller.
- The first status-table pass showed heartbeat files and running pollers.
- Rows 01 and 02 had already written `iteration_0`; other rows were still
  warming up.
- No eval/GIF artifacts existed yet, which is expected before the first
  `save_ckpt_after_iter=250` checkpoint.

2026-05-12 17:20 EDT v1c status:

- All 8 v1c apps were still detached and active.
- Checkpoints were arriving quickly: rows ranged from about `iteration_2000`
  through `iteration_3000` by the second status pass.
- External eval/GIF artifacts were being written for every row.
- Latest evaluated mean survival was still low:
  row 01 `10.0`, row 02 `13.375`, row 03 `8.0`, row 04 `7.75`,
  row 05 `11.75`, row 06 `7.0`, row 07 `8.0`, row 08 `14.125`.
- Interpretation: throughput is good enough to let this cook, but there is no
  convincing survival learning signal yet.
- Tooling note: `--output curve-summary` still shows `points=0` for these stock
  rows because it reads the missing custom `progress_latest` curve. For stock
  runs, use checkpoint count plus eval manifests/GIF summaries as ground truth
  until the curve tool is extended to summarize eval manifests directly.

2026-05-12 17:22 EDT curve tooling update:

- `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py` now lets
  `--output curve-summary` fall back to checkpoint eval manifests when the
  custom progress curve is absent.
- `python3 -m py_compile` and `uv run ruff check` passed for the status tool.
- First usable v1c eval-curve summary still showed no learning:
  - best fixed-straight rows were around `13-14.5` mean steps;
  - frozen-recent rows stayed around `7-8` mean steps;
  - fixed/browser sentinel row 08 was around `14.375` best mean steps;
  - several rows had some checkpoint with action-collapse flags.
- Interpretation did not change: v1c is running quickly and observability works,
  but the survival curve is still flat/weak this early.

Parallel monitor note:

- `docs/working/training/curvytron_architecture_research_2026-05-12/stock_long_v1c_monitor_critique_2026-05-12.md`
  independently checked the v1c commands, manifest, sampled heartbeats, and
  optimizer pivot alignment.
- Main risks from that note: checkpoint/eval/GIF artifact volume at 80
  checkpoints per row, reliance on the current default `opponent_use_cuda=false`
  rather than an explicit CLI flag, and a possible long-survival LightZero
  probability-size issue if agents eventually live much longer.

2026-05-12 17:34 EDT v1c curve status:

- All 8 v1c apps were still detached and active.
- Checkpoints were around `iteration_6500` through `iteration_9000`.
- Eval/GIF pollers continued writing artifacts.
- Fixed-straight controls showed only weak movement:
  - row 01 best `17.375` mean steps at `iteration_4000`, latest `14.25`;
  - row 02 best `14.5`, latest `13.125`;
  - row 08 browser-lines fixed sentinel best `16.625`, latest `16.5`.
- Frozen-recent rows remained flat:
  - row 03 latest `8.0`;
  - row 04 latest `8.0`;
  - row 06 latest `7.0`;
  - row 07 browser-lines frozen sentinel latest `8.0`.
- Frozen-old dense row 05 had a small bump, best `14.5`, latest `11.25`.
- Interpretation: there may be a tiny fixed-opponent survival bump, but there
  is still no convincing frozen-opponent learning signal.

Long-survival failure note:

- A background check traced the known LightZero
  `ValueError: 'a' and 'p' must have same size` to a death-disabled stock
  frozen subprocess profile in
  `docs/working/optimizer/stock_frozen_optimizer_pivot_2026-05-12.md`.
- Most likely trigger: terminal/truncated observations can expose all-zero
  action masks. Normal-death v1c is less exposed while episodes are short, but
  if policies start surviving much longer this should be tested with a focused
  low-`source_max_steps` truncation smoke.

2026-05-12 17:58 EDT launched `stock-tensor-v1d`.

Why:

- Waiting five hours on only the 8-row v1c slice was too narrow.
- v1d keeps the same trusted stock LightZero `train_muzero` path but launches a
  broader experiment tensor so the long wait produces useful comparisons.

Launch artifacts:

- manifest:
  `artifacts/local/curvytron_stock_train_manifests/stock-tensor-v1d.json`
- commands:
  `artifacts/local/curvytron_stock_train_manifests/stock-tensor-v1d.commands.sh`
- generated rows: `32`

Global settings:

- `mode=train`
- `compute=gpu-l4-t4-cpu40`
- `env_variant=source_state_fixed_opponent`
- `env_manager_type=subprocess`
- `max_train_iter=100000`
- `max_env_step=10000000`
- `save_ckpt_after_iter=2000`
- stock LightZero in-loop eval off: `lightzero_eval_freq=0`
- CurvyZero background checkpoint eval/GIF poller on
- no command contains `two-seat-selfplay`, `source_state_joint_action`, or
  `fast_gray64_direct`

Axes covered:

- opponents: fixed straight, recent frozen checkpoint, mid frozen checkpoint,
  old frozen checkpoint;
- rewards: sparse outcome and dense survival plus outcome;
- search: `num_simulations=8` and `16`;
- collector width: `C16`, `C32`, and `C64`;
- learner batch: `B32` and `B64`;
- visual surface: main `body_circles_fast`, plus `browser_lines` sentinels;
- source episode cap: `source_max_steps=256` and selected `1024` rows;
- simple stochasticity check: two rows with
  `ego_action_straight_override_probability=0.05`.

Initial dispatch check:

- Command file launched all 32 rows with detached Modal entrypoints.
- Row launch summaries confirmed `opponent_use_cuda=false`,
  background eval/GIF enabled, stock eval disabled, and expected run/attempt
  IDs.
- Immediate `modal app list` showed many new CurvyZero apps from 17:56-17:58
  EDT. Some were still settling with one task; do a cleaner health check after a
  short delay before the long wait.

2026-05-12 22:08 EDT v1c/v1d readout:

- Added a lightweight `eval-summary` mode to
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py` because the old
  curve summary was spending too much time scanning GIF summaries. Use
  `--output eval-summary` for survival curves across many runs.
- v1c reached roughly `20000` iterations. v1d reached roughly `100000`
  iterations for many rows; several rows were still detached/running or had
  stopped early, so read checkpoint/eval artifacts as ground truth.
- Main fact: fixed-straight rows show weak but real survival movement. Frozen
  recent and frozen mid checkpoint rows are mostly flat at about `8` steps.
  Frozen old rows move somewhat, but not enough to call the route solved.

Compact v1d survival facts:

| row group | best mean steps | latest mean steps | read |
| --- | ---: | ---: | --- |
| fixed sparse/dense, fast | `23.0` / `17.375` | `17.375` / `17.25` | weak positive signal |
| fixed dense sim16 | `18.5` | `17.25` | sim16 does not obviously solve it |
| fixed dense C64 | `21.0` | `17.375` | more collectors did not give a clean jump |
| fixed sparse with straight override `0.05` | `21.25` | `19.25` | best latest fixed row so far |
| browser fixed sentinels | `17.375` / `22.25` | `16.375` / `16.25` | browser surface is not wildly different |
| recent frozen rows | mostly `8.0`, one `9.25` | mostly `8.0` | essentially flat |
| mid frozen rows | mostly `8.0`, one `8.75` | mostly `8.0` | essentially flat |
| old frozen rows | `15.25` to `18.125` | `10.125` to `14.125` | some movement, not stable |
| max1024 fixed rows | `17.375` / `18.25` | `16.25` / `17.125` | higher cap did not change early story |

Specific v1d rows worth remembering:

- `v1d-31-c32-fast-fixed-sparse-straight005-b32-sim8`: best `21.25`, latest
  `19.25`, latest max `36`.
- `v1d-01-c32-fast-fixed-sparse-b32-sim8`: best `23.0`, latest `17.375`.
- `v1d-22-c16-browser-fixed-dense-b32-sim8`: best `22.25`, latest `16.25`.
- `v1d-03-c32-fast-recent-sparse-b32-sim8`: completed around `100020`,
  latest `8.0`.
- `v1d-32-c32-fast-recent-dense-straight005-b32-sim8`: completed around
  `100062`, latest `8.0`.

Interpretation to keep honest:

- Yes, there is some learning signal against the simple fixed-straight opponent.
- No, there is not yet a convincing practical CurvyTron training signal against
  a recent or mid frozen checkpoint opponent.
- The old-opponent bump suggests curriculum difficulty or opponent quality may
  matter, but it could also be a brittle exploit or eval noise.
- Action-collapse flags still appear often. Do not promote a row just because
  best mean survival moved once.
- The next investigation should focus on why frozen-recent/mid rows stay pinned
  at the starting survival floor while fixed and old-opponent rows can move.

Follow-up critique notes:

- The frozen-opponent provider is probably wired, because old/recent/mid share
  the same provider path but behave differently. The checkpoint contents or
  deterministic opponent mode are more suspicious than the provider itself.
- Frozen opponents currently use deterministic/eval-mode policy actions. If a
  recent or mid checkpoint has a narrow early-kill behavior, the learner may
  see a repetitive short episode distribution around `8` steps.
- Before another broad training matrix, characterize the fixed, old, mid, and
  recent opponents directly: first 16-step action histograms, terminal causes,
  survival against fixed/random ego, and short GIFs.
- The status/readout path should move to a cached survival table. See
  `docs/working/training/curvytron_architecture_research_2026-05-12/survival_readout_cache_plan_v1d_2026-05-12.md`.

Outcome-readout clarification:

- The eval manifests include learner-centric `win/loss/draw/cap` outcome
  counts. The compact `eval-summary` table now prints first, best, and latest
  outcomes, and it prefers the manifest aggregate outcome histogram instead of
  recomputing from terminal reason strings.
- Corrected readout: fixed and old-opponent rows often start with losses, then
  move toward mostly wins. Example fixed rows start around `loss:7,win:1` and
  later reach `win:7` or `win:8`.
- Recent frozen rows are different: many are already `win:8` at the first
  checkpoint and remain around `8` survival steps. That means the outcome signal
  is already saturated while survival does not improve.
- This supports the current suspicion: recent/mid frozen opponents are not
  useful pressure for the next overnight batch. They may simply die too quickly
  or deterministically, leaving no meaningful outcome gradient.
- For this phase, read progress as survival curve plus action/terminal
  diagnostics. Treat win count as secondary and often saturated.

Next-opponent design note:

- A one-player/wall-avoidance sanity lane is reasonable: make opponent pressure
  non-limiting so the policy learns to survive, not just to outlive a weak
  opponent.
- Fastest clean version is not true one-player `player_count=1`; the current
  public env assumes 2+ players.
- Safer diagnostic knobs to add before the next overnight batch:
  - `opponent_death_mode=normal|immortal`, implemented as an ego-only death
    mode so player 0 can still die but player 1 cannot.
  - `opponent_trail_mode=normal|none`, implemented by preventing player 1 from
    printing body/trail geometry, not merely hiding it in the renderer.
- Do not reuse global `profile_no_death`: it disables death for the ego too and
  changes the task.

Joint-action note:

- The source-state env now accepts
  `opponent_policy_kind=none_centralized_joint_action` only for
  `source_state_joint_action`, with focused tests passing.
- Do not mix those diagnostics back into the main v1b learning matrix. If
  needed, launch them as a separate canary after the fixed/frozen rows have a
  readable curve.

Launch command source:

```text
zsh artifacts/local/curvytron_stock_train_manifests/stock-high-signal-v1b.commands.sh
```

## Variables To Keep Fixed

- Render path: keep source-state visual input. For v1c, `body_circles_fast` is
  the main speed lane and two small `browser_lines` rows are fidelity sentinels.
- Death mode/source mechanics: keep normal death.
- Opponent provider: use the strict frozen LightZero checkpoint provider only
  for frozen rows.
- Checkpoint refs: explicit immutable `iteration_*.pth.tar` refs only.
- No current-policy two-seat custom replay.
- No action-repeat/no-op skip stochasticity.
- No dynamic checkpoint-pool refresh unless that route is already integrated
  and has its own canary.

## Eval Panel

Every checkpoint eval should report, at minimum:

- survival steps: mean, median, max, and per-seed traces;
- sparse outcome and dense helper components separately;
- terminal reason counts and draw/win/loss counts;
- action distribution for collect-mode and greedy/eval-mode;
- opponent metadata: kind, checkpoint ref, strict-load status, device;
- stock path proof: `called_train_muzero=true` and native buffer class used.

For frozen rows, the panel should include:

- training opponent;
- fixed-straight opponent;
- at least one held-out old checkpoint;
- at least one held-out recent checkpoint when available.

Do not compare raw reward totals across reward variants without normalizing by
episode count and row count.

## Lane Interpretations

`fixed-straight control` is a low-bar stock-loop sanity check. If it cannot move,
the problem is probably not opponent curriculum.

`frozen core` is the practical near-term route. It is not true live self-play.
It asks whether a named frozen/recent opponent can create a useful learning
curve while stock LightZero owns collection, replay, targets, and learning.

`opponent age` separates curriculum difficulty from frozen-opponent plumbing.
Recent may be more realistic; old/mid may produce clearer early signal.

`reward ablation` separates sparse competitive learning from survival-shaped
learning. Dense reward is useful only if sparse outcome and held-out survival
also move.

Historical warning: this reward-ablation framing predates the corrected v1d
readout. Do not use it to justify outcome reward in the next diagnostic lane;
current guidance is survival plus bonus reward with outcome reward off/zero.
See [current_source_of_truth.md](current_source_of_truth.md).

`search/batch sensitivity` checks the two cheapest MuZero knobs before changing
architecture again.

`centralized joint action` is not competitive self-play. It is a clean control
for one-real-physical-tick replay under stock `train_muzero`.

`native two-seat bridge` has zero rows in this matrix. Tiny target parity passed,
but it should enter a training batch only after the actual two-seat trainer uses
native `GameSegment` / `MuZeroGameBuffer` in the learning loop.

## Stop / Promote Rules

Stop reading a row as useful if its summary does not prove
`called_train_muzero=true`, strict checkpoint loading for frozen rows, and
checkpoint artifacts at the planned cadence.

Promote a lane only if at least two adjacent checkpoints improve held-out
survival or outcome beyond seed noise and the action distribution is not a
single-action collapse in both collect and eval views.

If rows 01 and 02 are flat, do not scale frozen rows. Inspect stock env/reward
and target telemetry first.

If rows 03 through 08 are flat but rows 09 or 10 move, the blocker is likely the
single-ego frozen-opponent formulation, not visual MuZero in general.

If rows 09 and 10 are also flat, the next work should be an instrumentation
audit of targets, reward support, reset randomness, and observations before any
larger batch.
