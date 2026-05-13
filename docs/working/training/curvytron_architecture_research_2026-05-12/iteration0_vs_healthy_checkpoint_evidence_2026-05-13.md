# Iteration-0 Stale Rows vs Healthy Checkpointing Rows

Date: 2026-05-13

Scope: read-only investigation. Source code was not edited. Evidence came from local snapshots under `artifacts/local/curvytron_pruning/status_chunks_20260513e`, richer local raw files under `status_chunks_20260513f`, `live_bug_probe_20260513`, and narrow read-only `modal volume ls` checks for checkpoint and resume-state sidecar directories.

## Status Semantics Caution

`train_status=running`, `status_heartbeat.status=running`, and a fresh `progress_latest.timestamp` are weak liveness signals. They can mean the wrapper or hook path is still writing metadata, but they do not prove durable checkpoint health. For these rows, the stronger artifact-health signals are:

- the highest visible `iteration_*.pth.tar` in `train/lightzero_exp/ckpt`;
- the highest visible `iteration_*.resume_state.pkl` sidecar in `checkpoints/lightzero_resume_state`;
- eval/GIF advancement beyond `live_checkpoint_iteration_0`;
- whether `progress_latest.iteration` / `checkpoint_name` tracks a newer checkpoint or stays pinned to `iteration_0`.

## Evidence Table

| Class | Run id | Train status | `progress_latest` timestamp | `learner_train_iter` | Highest checkpoint | Highest sidecar | Eval/GIF state | Config cadence |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| Stale k0 | `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | `running` in snapshot `e`; heartbeat stage `auto_resume_checked` in raw `f` | `2026-05-13T19:53:50Z` from `live_bug_probe`; `2026-05-13T19:40:33Z` in raw `f` | 180751 live; 175528 in raw `f` | `iteration_0.pth.tar` only by snapshot `e` and live `modal volume ls`; `ckpt_best` also present | `iteration_0.resume_state.pkl` only by live `modal volume ls` | Snapshot `e`: 1 eval manifest, 1 GIF, latest eval/GIF `iteration_0`; GIF terminal `round_survivor_win`; greedy collapse warning true | `save_ckpt_after_iter=10000`, `save_ckpt_after_run=True` |
| Stale k0 | `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | `running` in snapshot `e`; heartbeat stage `auto_resume_checked` in raw `f` | `2026-05-13T19:40:11Z` in raw `f` | 33998 | `iteration_0` only in snapshot `e` | raw heartbeat auto-resumed from matching `iteration_0.resume_state.pkl`; highest sidecar not live-listed for this row | Snapshot `e`: 1 eval manifest, 1 GIF, latest eval/GIF `iteration_0`; GIF terminal `round_survivor_win` | `save_ckpt_after_iter=10000`, `save_ckpt_after_run=True` |
| Stale k0 | `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | `running` in snapshot `e`; heartbeat stage `auto_resume_checked` in raw `f` | `2026-05-13T19:51:34Z` from `live_bug_probe`; `2026-05-13T19:43:26Z` in raw `f` | 69261 live; 65588 in raw `f` | `iteration_0` only in snapshot `e` and raw `progress_latest.checkpoint_name` | raw heartbeat auto-resumed from matching `iteration_0.resume_state.pkl`; highest sidecar not live-listed for this row | Snapshot `e`: 1 eval manifest, 1 GIF, latest eval/GIF `iteration_0`; GIF terminal `round_survivor_win` | `save_ckpt_after_iter=10000`, `save_ckpt_after_run=True` |
| Stale k0 | `curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011` | `running` in snapshot `e`; heartbeat stage `auto_resume_checked` in raw `f` | `2026-05-13T19:44:19Z` in raw `f` | 97749 | `iteration_0` only in snapshot `e` and raw `progress_latest.checkpoint_name` | raw heartbeat auto-resumed from matching `iteration_0.resume_state.pkl`; highest sidecar not live-listed for this row | Snapshot `e`: 1 eval manifest, 1 GIF, latest eval/GIF `iteration_0`; GIF terminal `round_survivor_win` | `save_ckpt_after_iter=10000`, `save_ckpt_after_run=True` |
| Stale k0 | `curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011` | `running` in snapshot `e`; heartbeat stage `auto_resume_checked` in raw `f` | `2026-05-13T19:43:27Z` in raw `f` | 100411 | `iteration_0` only in snapshot `e` and raw `progress_latest.checkpoint_name` | raw heartbeat auto-resumed from matching `iteration_0.resume_state.pkl`; highest sidecar not live-listed for this row | Snapshot `e`: 2 eval manifests, 1 GIF, latest eval/GIF `iteration_0`; GIF terminal `round_survivor_win` | `save_ckpt_after_iter=10000`, `save_ckpt_after_run=True` |
| Stale k0 | `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s16-c32-l32-repM-k10-c1-s2306011` | `running` in snapshot `e` | not in local raw `f` sample | not in local raw `f` sample | `iteration_0` only in snapshot `e` | not sampled live | Snapshot `e`: 2 eval manifests, 1 GIF, latest eval/GIF `iteration_0`; GIF terminal `round_survivor_win` | `save_ckpt_after_iter=10000` in same sampled config family |
| Healthy | `curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021` | `running` in snapshot `e` | `2026-05-13T19:14:40Z` in snapshot `e` | snapshot `e` progress iteration 190000; direct learner counter not sampled in raw `f` | Snapshot `e`: `iteration_190000`; live `modal volume ls`: `iteration_210000.pth.tar` | live `modal volume ls`: `iteration_210000.resume_state.pkl` | Snapshot `e`: 13 eval manifests, 19 GIFs, latest eval/GIF `iteration_190000`; GIF terminal `round_survivor_win` | `save_ckpt_after_iter=10000`, `save_ckpt_after_run=True` |
| Healthy | `curvy-survive-bonus-blank-browser-steady-base-r034-s1110171` | `running` in snapshot `e` | `2026-05-13T09:38:30Z` in snapshot `e` and `live_bug_probe`; progress file itself stale/initial-looking despite healthy artifact stream | `0` in sampled progress file, showing this field can be stale/misleading for some healthy rows | Snapshot `e`: `iteration_255000`; live `modal volume ls`: `iteration_270000.pth.tar` | live `modal volume ls`: `iteration_270000.resume_state.pkl` | Snapshot `e`: 13 eval manifests, 16 GIFs, latest eval/GIF `iteration_240000`; GIF terminal `round_survivor_win` | `save_ckpt_after_iter=15000`, `save_ckpt_after_run=True` |

## Snapshot-Level k0 Context

Snapshot `status_chunks_20260513e/summary.json` reported:

- `checked_run_count=212`, all `train_status_counts.running=212`;
- `iteration_zero_count=6`;
- all six k0 rows had `progress_exists=true`;
- the six k0 rows were unchanged since snapshot `d`, with checkpoint ages around 255-450 minutes;
- each k0 row had only `iteration_0` as the latest checkpoint, and only `iteration_0` as latest eval/GIF checkpoint.

## Conclusions

1. The stale k0 rows are not merely dead rows with absent progress. Five sampled rows had fresh `progress_latest.timestamp` values and high `learner_train_iter` values, up to 180751, while the visible checkpoint and sidecar state remained at `iteration_0`.
2. The stale k0 rows are also not explained by an overly sparse checkpoint cadence. Their sampled configs use `save_ckpt_after_iter=10000`, and every sampled `learner_train_iter` is multiple checkpoint intervals past zero.
3. Healthy rows under the same artifact scheme show model checkpoints and resume-state sidecars advancing together (`iteration_210000` for healthy mix2, `iteration_270000` for healthy survival in live volume reads), plus eval/GIF artifacts advancing across many checkpoints.
4. Therefore, run health should not be inferred from `train_status=running`, heartbeat status, or freshness of `progress_latest` alone. A safer health predicate is: newest durable checkpoint and newest resume-state sidecar advance at expected cadence, and eval/GIF artifacts follow those checkpoints within the poller lag budget.
5. The best concise diagnosis for the k0 rows is: training code was alive enough to run checkpoint-hook metadata writes and increment learner counters, but durable checkpoint/sidecar artifacts did not advance beyond the initial `iteration_0` state.
