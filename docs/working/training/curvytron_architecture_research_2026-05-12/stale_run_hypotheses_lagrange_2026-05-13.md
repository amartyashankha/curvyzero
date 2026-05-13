# Stale Run Hypotheses - Lagrange - 2026-05-13

Read-only investigation note. No source code changes.

## Scope

Investigate CurvyTron Modal runs where most preserved rows advance, but a stale tail still reports `running` without new checkpoints.

Primary sources read:

- `current_source_of_truth.md`
- `run_health_check_2026-05-13.md`
- `run_prune_decision_2026-05-13.md`
- `opponent_mixture_lane_2026-05-13.md`
- `opponent_mixture_batch_plan_2026-05-13.md`
- `run_inventory_2026-05-13.md`
- `eval_gif_status_note_codex.md`
- `artifacts/local/curvytron_pruning/status_chunks_20260513e/summary.json`
- `artifacts/local/curvytron_pruning/status_chunks_20260513d/summary.json`
- `artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json`

## Current Read

- Preserved surface: 212 rows, all reporting `train_status=running`.
- Snapshot `status_chunks_20260513e`: 157 rows advanced since the prior chunked snapshot, 55 did not.
- Unchanged rows by bucket: mix2 22, mix3 24, survival 8, v1b dependency 1.
- Stale by checkpoint age: 36 rows over 3h, 10 rows over 6h.
- Six live non-dependency rows remained at `iteration_0` after hours: 2 mix2, 4 mix3.
- Eval/GIF is not globally broken: 211/212 eval manifests, 212/212 GIF artifacts.
- The stale tail is patterned, not random:
  - mix2 stale rows concentrate in `r50-scr50`, `r50-old50`, and `r50-blank25-scr25`.
  - mix3 stale rows concentrate in heavier probes such as `s16`, `c64`, and `l64`.
  - survival stale rows include heavy/search/batch/collect variants.
- Render backend is unlikely to be the sole cause. Stale rows are split across fast/browser, and docs show browser only modestly slower than fast in matched reads.
- `train_status=running` is not sufficient proof of useful progress. Some rows have fresh-looking status/progress metadata while checkpoint count and checkpoint mtime remain fixed.

## Follow-up Evidence

New read-only evidence from the follow-up wave changes the ranking:

- Some stale rows refreshed `progress_latest` at 19:40+.
- Those refreshed rows report live-looking `learner_train_iter` values such as 34k, 65k, 97k, 100k, and 175k.
- The same rows still report `checkpoint_name=iteration_0`.
- Some train logs show multiple container IDs under one Modal FunctionCall.
- Healthy rows can also have preemption/multiple containers, so preemption alone is not sufficient to explain the stale tail.

Implication: for rows with high `learner_train_iter`, the trainer is not simply dead or waiting forever in collection. The primary failure surface moves downstream of learner advancement: checkpoint naming, checkpoint save cadence, checkpoint publication to the Modal volume, or resume/preemption interaction with those paths.

## Re-ranked Failure Modes

| Rank | Failure mode | Movement | Why | Next read-only check |
| --- | --- | --- | --- | --- |
| 1 | Checkpoint publication/persistence failure after learner advancement | More likely | `learner_train_iter` reaches tens/hundreds of thousands while `checkpoint_name` and visible checkpoint files remain at `iteration_0`. That means useful training state can advance without a corresponding durable checkpoint signal. | For high-iter stale rows, inspect existing train logs around expected checkpoint cadences. Look for save-hook messages, checkpoint path, volume write/commit errors, skipped saves, or writes to an unexpected run root. |
| 2 | Progress/status checkpoint fields are stale or only record last durable checkpoint | More likely as a symptom | Fresh `progress_latest` can coexist with stale `checkpoint_name=iteration_0`; this means progress freshness is not equivalent to checkpoint freshness. | Compare `learner_train_iter`, `checkpoint_name`, checkpoint directory listing, checkpoint mtimes, eval manifests, and GIF summaries for the same stale rows. |
| 3 | Preemption/resume interaction with checkpoint publication | More likely as a cofactor, not sole cause | Multiple container IDs under one FunctionCall show preemption/resume is in play, and a resume path could fail to restore checkpoint publishing correctly. | Compare stale preempted rows with healthy preempted rows. Check whether healthy rows emit a post-preemption checkpoint and stale rows do not. Inspect run root/path metadata immediately after container transitions. |
| 4 | Modal volume write/commit/layer issue affecting train checkpoints | More likely than before | If learner iterations advance but durable checkpoint files do not, the volume publication path is suspect. Existing docs already mention Modal volume pressure, commit storms, and `DataLossError`, though mostly in eval/GIF workers. | Search existing logs/artifacts for train-side volume write, commit, reload, `DataLossError`, or `ResourceExhausted` messages near stale rows. Check for partial checkpoint artifacts. |
| 5 | Opponent mixture recipe pathology | Narrower | The mix2 stale concentration in scripted/old recipes still matters, but high learner iterations argue against a pure collection deadlock. It may be a trigger or correlate for checkpoint-publish failure rather than the direct stall. | Group high-iter stale rows by mixture recipe and compare against healthy rows with the same mixture, same render, and similar compute knobs. |
| 6 | Config-driven slow training in heavy rows | Less likely as primary | High `learner_train_iter` values contradict a simple "too slow to reach checkpoint" explanation for sampled stale rows. Heavy configs may still explain some lower-iter rows. | Split stale rows into high-learner-iter vs low-learner-iter cohorts before attributing any row to slow compute. |
| 7 | Zombie/stale `running` status or dead container | Less likely for high-iter rows | Fresh `progress_latest` plus high learner iterations means at least those trainers are alive enough to update progress. | Keep this hypothesis only for stale rows whose heartbeat/progress mtimes also stop advancing. |
| 8 | Poller/eval/GIF artifact masking | Still low for primary failure | Poller issues can hide eval/GIF freshness but cannot explain absent train checkpoints when learner iterations advance. | Continue separating train checkpoint freshness from eval/GIF/poller counters. |
| 9 | Prune/cancel accidentally interrupted survivors | Lower | A canceled/dead survivor does not fit fresh high learner iterations. Earlier prune evidence still shows preserved rows were not broadly killed. | Still cross-check stale survivor call IDs against cancel manifests before any destructive action. |
| 10 | Status snapshot/chunking artifact | Lower | The new per-row progress evidence is specific and stronger than an aggregate snapshot-count concern. | Use small stale-only refreshes to validate the same mismatch repeats. |

## Earlier Hypotheses

| Possible cause | Supporting evidence | Contradicting evidence | Next read-only check |
| --- | --- | --- | --- |
| Config-driven slow training in heavy rows | mix3 stale rows include many `s16`, `c64`, `l64` probes; survival stale rows include `heavy`, `batch64`, `search16`, or `collect64` variants; some stale rows have multiple prior checkpoints, implying they can run but may be slow. | mix2 stale rows are mostly not heavy-compute variants; several rows remain at `iteration_0` for hours, which is more severe than a simple 20-30 minute cadence stretch. | From saved status artifacts, compute checkpoint gap distributions by token: `s16`, `c64`, `l64`, `batch64`, `search16`, `collect64`, compared with matched base rows. |
| Opponent mixture recipe creates pathological collection or learning stalls | mix2 stale rows concentrate in scripted/old-opponent recipes: `r50-scr50`, `r50-old50`, `r50-blank25-scr25`; opponent mixture docs confirm strict per-episode frozen/scripted opponent selection with no fallback. | Other scripted/current/old mixture rows did advance, and mix2 canaries reached k10/k20 artifacts; no row-level exception is visible in the status summaries. | Compare stale vs healthy rows by mixture recipe, selected opponent component metadata, eval episode lengths, terminal causes, and checkpoint intervals using existing manifests only. |
| `running` status is stale or misleading after useful training stops | All 212 rows report `running` even though 55 did not advance; sample stale rows show `event=checkpoint`, old latest checkpoint, and fixed checkpoint count; docs already warn to use checkpoint mtime/eval/GIF freshness rather than `train_status` alone. | 157 rows advanced across the same period, so the status mechanism is not universally stale; fresh progress timestamps may mean some processes are alive but slow. | For stale rows, compare `status_heartbeat.json` mtime/timestamp, `progress_latest.json` mtime/timestamp, and latest checkpoint mtime. If heartbeat/progress keeps moving while checkpoints do not, status is alive-but-not-progressing. |
| Modal FunctionCall is alive but stuck inside training after a checkpoint | Stale rows can retain `running` status and large `elapsed_sec` while checkpoint count stays fixed; train functions have long timeouts, so stuck calls can persist for hours. | Not proven from volume artifacts alone; slow data collection can look similar. | Read-only inspect a small sample of stale train FunctionCalls via call graph/status/log APIs. Check for still-running containers, retry state, timeout state, or terminal failures. Do not call cancel/get-waiting APIs. |
| Poller/eval/GIF issues are masking artifact freshness rather than blocking training | Docs note old pollers can hit Modal volume `DataLossError`, and poller completed counters stay zero until poller exit; 54 unchanged rows also did not add eval/GIF artifacts. | Primary stale signal is missing new train checkpoints, which poller failure cannot explain. Eval/GIF presence is broadly healthy: 211/212 eval manifests and 212/212 GIFs. | For stale rows, separate train checkpoint freshness from eval/GIF freshness. Read poller status/error JSON for `DataLossError`, retry labels, and latest eval/GIF checkpoint. |
| Prune/cancel accidentally interrupted survivors | Staleness was observed after aggressive FunctionCall cancellation and volume cleanup work. | Prune docs record 212 preserved rows, 1720 killed train/poller calls canceled with 0 failures, no missing preserved roots, and healthy survivor samples; stale pattern is config-correlated, not random. | Cross-check stale survivor train/poller call IDs from grouped launch artifacts against prune cancel manifests/results. Any intersection would be a serious prune bug. |
| Modal volume layer/commit pressure delays or drops checkpoint writes | Cleanup hit Modal volume rate/layer limits; docs mention commit storms and `DataLossError` in checkpoint eval/GIF workers. | 157 rows advanced with no checkpoint regressions; documented commit failures mostly affect eval/GIF/poller paths, not the train checkpoint save path. | Search existing row artifacts and logs for volume write/commit errors near stale rows. Check for partial checkpoint files, missing metadata, or repeated save failures. |
| Function timeout or platform termination is leaving old status behind | Long-running functions have finite train/poller timeouts; older/manual stopped runs may not write final completed/failed status. | Current stale rows appear to have stalled well before the 16h train timeout in several cases; status summaries do not show failed/timeout terminal state. | Read-only FunctionCall status/call graph for stale rows and compare launch time to timeout budget. Look specifically for timeout/terminated states that did not make it into volume status. |
| Status snapshot artifact mismatch or chunking artifact | Earlier all-212 status reads hit 300s timeout; chunked reads are preferred; docs and summary have small count differences such as `<=30 min` bucket. | The core stale signal is stable across chunked snapshots: 55 unchanged, 36 over 3h, 10 over 6h, no regressions. | Rerun read-only status for only the stale row IDs, in small chunks, and compare checkpoint paths/mtimes against `status_chunks_20260513e`. |

## Working Interpretation

The strongest current hypothesis is still a mixed tail, but the center of gravity has moved:

1. For rows with high `learner_train_iter` and `checkpoint_name=iteration_0`, the learner is advancing but durable checkpoint publication is not.
2. Preemption/resume is plausible as a cofactor, but not sufficient by itself because healthy rows can also preempt.
3. Mixture recipe and heavy compute settings remain useful stratifiers, but they are less likely to be the whole explanation for high-iter stale rows.
4. `running` and fresh `progress_latest` prove liveness better than before, but they still do not prove checkpoint durability.

Treat `latest_checkpoint` mtime, checkpoint count, eval/GIF manifest freshness, and read-only FunctionCall state as the decisive signals. Do not treat `train_status=running` alone as evidence that a row is still learning.

## Recommended Read-Only Checks

1. For high-`learner_train_iter` stale rows, inspect logs around expected checkpoint intervals and container transitions.
2. Compare stale preempted rows to healthy preempted rows under the same FunctionCall/container pattern.
3. Build a stale-row-only status refresh in small chunks and compare against `status_chunks_20260513e`.
4. For the six `iteration_0` rows and the ten `>6h` rows, inspect heartbeat/progress/checkpoint mtimes side by side.
5. Cross-check stale survivor FunctionCall IDs against the prune cancel manifest/results.
6. Read-only inspect call graph/status/log state for a tiny stale sample:
   - one mix2 scripted/old row
   - one mix3 heavy row
   - one survival heavy/search row
7. Compare checkpoint gap distributions by recipe and heavy token before canceling or deleting any survivor roots.
