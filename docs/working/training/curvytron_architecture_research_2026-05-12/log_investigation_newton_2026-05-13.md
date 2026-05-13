# CurvyTron Modal Log Investigation - 2026-05-13

Scope: read-only investigation of CurvyTron rows that report
`train_status=running` while checkpoint files are stale. No source code was
edited. This note only records Modal logs, Modal volume metadata, local launch
metadata, and status artifacts. This document itself is the only file written.

## Inputs Checked

- `docs/working/training/curvytron_architecture_research_2026-05-12/run_health_check_2026-05-13.md`
- `docs/working/training/curvytron_architecture_research_2026-05-12/current_source_of_truth.md`
- `artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json`
- `artifacts/local/curvytron_pruning/status_chunks_20260513e/summary.json`
- Launch manifests:
  - `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-clean-20260513a.grouped_submit_launch.json`
  - `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_launch.json`
  - `artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513b.grouped_submit_launch.json`
- Modal app: `curvyzero-lightzero-curvytron-visual-survival-train`
- Modal volume: `curvyzero-runs`

## Commands Run

Representative local/status commands:

```bash
sed -n '1,240p' docs/working/training/curvytron_architecture_research_2026-05-12/run_health_check_2026-05-13.md
sed -n '1,260p' docs/working/training/curvytron_architecture_research_2026-05-12/current_source_of_truth.md
jq '.unchanged_rows[0:20]' artifacts/local/curvytron_pruning/status_chunks_20260513e/summary.json
jq -r '.rows[] | select(.run_id as $rid | ["curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011","curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031","curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051","curvy-survive-bonus-blank-fast-heavy-batch64-r299-s1141671","curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021","curvy-mix3cur-r25-blank75-rf-s8-c32-l32-rep0-k10-c1-s2301011","curvy-survive-bonus-blank-fast-medium-base-r109-s1112151"] | index($rid)) | {run_id,train_status,train_stage,timestamp,latest_checkpoint,latest_checkpoint_mtime:(.latest_checkpoint_mtime|strftime("%Y-%m-%dT%H:%M:%SZ")),checkpoint_count,progress_exists,eval_manifest_count:(.eval_checkpoints|length),gif_artifact_count:(.gif_artifacts|length? // .gif_artifact_count // null),latest_eval_checkpoint,latest_eval_created_at,latest_gif_checkpoint}' artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json
```

Representative launch metadata command:

```bash
jq -r '.records[] | select(.run_id|IN("curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011","curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031","curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021")) | [.run_id,.train_function_call_id,.poller_function_call_id,.artifact_refs.progress_latest,.artifact_refs.background_eval_status,.artifact_refs.summary] | @tsv' artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-clean-20260513a.grouped_submit_launch.json
```

Representative Modal log/volume commands:

```bash
modal app logs curvyzero-lightzero-curvytron-visual-survival-train --function-call fc-01KRGJQ7YY2W1BMGR05RSR1NYP --timestamps --show-function-call-id --show-container-id --tail 1000
modal app logs curvyzero-lightzero-curvytron-visual-survival-train --function-call fc-01KRGJQ7YY2W1BMGR05RSR1NYP --timestamps --show-function-call-id --show-container-id --tail 500 --search error
modal app logs curvyzero-lightzero-curvytron-visual-survival-train --function-call fc-01KRGJQ7YY2W1BMGR05RSR1NYP --timestamps --show-function-call-id --show-container-id --tail 500 --search traceback
modal app logs curvyzero-lightzero-curvytron-visual-survival-train --function-call fc-01KRGJQ7YY2W1BMGR05RSR1NYP --timestamps --show-function-call-id --show-container-id --tail 500 --search timeout
modal app logs curvyzero-lightzero-curvytron-visual-survival-train --since 2026-05-13T05:00:00-04:00 --until 2026-05-13T19:45:00-04:00 --timestamps --show-function-call-id --show-container-id --search preempt --tail 200
modal volume get curvyzero-runs <progress_latest_ref> - | sed -n '1,40p'
modal volume get curvyzero-runs <checkpoint_eval_poller_ref> - | sed -n '1,80p'
modal volume ls curvyzero-runs <train/lightzero_exp/ckpt path> --json
uv run --extra modal python - <<'PY'
import modal
for fc in ["fc-01KRGJQ7YY2W1BMGR05RSR1NYP", "fc-01KRGJQBFF1JC8MJGTA0D0RXNW", "fc-01KRGRHAZ3ZVRBVM5YNJN1RNNN", "fc-01KRGJPR52EN5XH3RYZ3TBEHXW"]:
    call = modal.FunctionCall.from_id(fc)
    print(fc, call.get_dashboard_url(), call.get_call_graph())
PY
```

## Representative Rows

| Role | Run id | Status snapshot checkpoint | Snapshot latest checkpoint mtime | Train call | Poller call |
| --- | --- | ---: | --- | --- | --- |
| stale mix2, stuck at k0 | `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | `iteration_0` | `2026-05-13T11:53:38Z` | `fc-01KRGJQ7YY2W1BMGR05RSR1NYP` | `fc-01KRGJQ7V9RHFKTWT2N8NRPGRF` |
| stale mix2, stuck at k20 | `curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031` | `iteration_20000` | `2026-05-13T12:38:44Z` | `fc-01KRGJQBFF1JC8MJGTA0D0RXNW` | `fc-01KRGJQBC5P5S05641VEB2STBP` |
| stale mix3, stuck at k0 | `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | `iteration_0` | `2026-05-13T15:06:01Z` | `fc-01KRGRHAZ3ZVRBVM5YNJN1RNNN` | `fc-01KRGRHASW04PTAHP7AG86HQBM` |
| stale survival comparator | `curvy-survive-bonus-blank-fast-heavy-batch64-r299-s1141671` | `iteration_105000` | `2026-05-13T14:18:10Z` | `fc-01KRGB5CFMSFG672NBCRA270VN` | `fc-01KRGB5CCJTHHKQ50MXFECTC44` |
| healthy mix2 comparator | `curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021` | `iteration_190000` at snapshot, later `iteration_200000` in volume | `2026-05-13T19:04:40Z` at snapshot | `fc-01KRGJPR52EN5XH3RYZ3TBEHXW` | `fc-01KRGJPR1GWNZGCQGNA6DA75BH` |
| healthy mix3 comparator | `curvy-mix3cur-r25-blank75-rf-s8-c32-l32-rep0-k10-c1-s2301011` | `iteration_160000` | `2026-05-13T18:56:14Z` | `fc-01KRGRFAPK9VPNK2NMSCXFKAAM` | `fc-01KRGRFAJBZ3GZAS94MK7E7NJ4` |
| healthy survival comparator | `curvy-survive-bonus-blank-fast-medium-base-r109-s1112151` | `iteration_285000` | `2026-05-13T19:10:34Z` | `fc-01KRGB3TQY01EJC3SSC2FX100S` | `fc-01KRGB3TMHW4D2V47ENYKF3DR3` |

## Modal Findings

1. The stale rows are not simply missing launch metadata. The grouped launch
   manifests contain train and poller FunctionCall IDs for the representative
   stale rows.

2. `modal.FunctionCall.from_id(...).get_call_graph()` returned one input for
   each checked train call with status `InputStatus.PENDING`, including healthy
   comparators. This did not distinguish stale from healthy rows. Dashboard URLs
   were available, for example `https://modal.com/id/fc-01KRGJQ7YY2W1BMGR05RSR1NYP`.

3. Per-train-call Modal logs for the representative stale calls did not show a
   direct Python exception, CUDA error, timeout, or explicit crash in the
   checked output. The only per-call startup log line seen for those train calls
   was the recurring warning:
   `not found transformer, please install it using: pip install transformers`.
   The same kind of sparse train-call logging appears on healthy calls too.

4. App-level logs do show broad worker preemption during the afternoon. Example
   line:
   `Runner interrupted due to worker preemption. Your Function will be restarted with the same input.`
   This was common across the app, not limited to stale rows.

5. Some representative stale-row containers were explicitly preempted:
   - stale mix2 k0 first container `ta-01KRGJXPPJAGP9RTC9C3JWM0XF` at
     `2026-05-13 08:07:55-04:00`;
   - stale mix2 k0 second container `ta-01KRGKV53HB281B19VZWX40FHA` at
     `2026-05-13 08:33:10-04:00`;
   - stale mix2 k20 first container `ta-01KRGJYC5M24F5ZXF2QQDY18HD` at
     `2026-05-13 08:49:30-04:00`;
   - stale mix2 k20 second container `ta-01KRGP83DVW0YSM8A9WN1Y08GF` at
     `2026-05-13 11:06:56-04:00`;
   - stale mix3 k0 first container `ta-01KRGXXXG8SNB29JVKQZWG84XV` at
     `2026-05-13 11:30:59-04:00`;
   - stale mix3 k0 second container `ta-01KRGZCWJNA2E43JR7HYYQSV5X` at
     `2026-05-13 13:20:27-04:00`.

6. Preemption alone is not sufficient to explain staleness. The healthy
   survival comparator also had a container preempted:
   `ta-01KRGCTC2D5F10MN7BJJNXJ0AN` at `2026-05-13 11:06:54-04:00`, and that row
   still reached `iteration_285000` in the status snapshot.

7. App-level searches did find many `Traceback` entries, but the visible
   examples were mostly background eval/GIF related, including `DataLossError:
   failed to publish commit to server`, checkpoint-load failures from apparent
   partially written/corrupt checkpoint files, and `KeyboardInterrupt` around
   subprocess env reset. I did not tie those aggregate tracebacks directly to
   the representative stale train FunctionCall IDs above.

## Volume / Progress Findings

The strongest signal is a mismatch between learner progress and checkpoint file
creation.

### Stale mix2 k0

Run:
`curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011`

Remote checkpoint directory contained only:

- `iteration_0.pth.tar`, modified `2026-05-13 07:53 EDT`;
- `ckpt_best.pth.tar`, modified `2026-05-13 07:53 EDT`.

But remote `progress_latest.json` was fresh at `2026-05-13T19:35:34.273824Z`
and reported:

- `event`: `checkpoint`;
- `source`: `SaveCkptHook.__call__`;
- `iteration`: `0`;
- `learner_train_iter`: `173663`;
- `elapsed_sec`: `25051.352271`.

The poller state for the same row was stale: `heartbeat_at` was
`2026-05-13T12:01:51.162589Z`, `scheduled_count`/`seen_count` was `1`, and
`completed_count` was `0`.

Plain read: the train call continued far past k0 by learner iteration, but the
new checkpoint files were not appearing. The status row's `running` state and
fresh progress timestamp are therefore not evidence of healthy checkpoint
progress.

### Stale mix2 k20

Run:
`curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031`

Remote checkpoint directory contained only:

- `iteration_0.pth.tar`, modified `2026-05-13 07:56 EDT`;
- `iteration_10000.pth.tar`, modified `2026-05-13 08:17 EDT`;
- `iteration_20000.pth.tar`, modified `2026-05-13 08:38 EDT`;
- `ckpt_best.pth.tar`, modified `2026-05-13 07:56 EDT`.

Remote `progress_latest.json` was fresh at `2026-05-13T19:42:49.921169Z` and
reported:

- `event`: `checkpoint`;
- `source`: `SaveCkptHook.__call__`;
- `iteration`: `20000`;
- `learner_train_iter`: `144146`;
- `elapsed_sec`: `16545.370727`.

The poller state had `heartbeat_at` `2026-05-13T12:59:58.204464Z`,
`last_scan_count` `3`, `gif_scheduled_count` `3`, `outstanding_count` `6`, and
`completed_count` `0`.

Plain read: this row crossed far beyond learner iter 20k but kept advertising
the same latest checkpoint. The missing eval manifest in the health check is
consistent with the poller being stuck/not completing, but the train-side
checkpoint file production is also stale.

### Stale mix3 k0

Run:
`curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051`

Remote checkpoint directory contained only:

- `iteration_0.pth.tar`, modified `2026-05-13 11:06 EDT`;
- `ckpt_best.pth.tar`, modified `2026-05-13 11:05 EDT`.

Remote `progress_latest.json` was fresh at `2026-05-13T19:43:26.614776Z` and
reported:

- `event`: `checkpoint`;
- `source`: `SaveCkptHook.__call__`;
- `iteration`: `0`;
- `learner_train_iter`: `65588`;
- `elapsed_sec`: `8338.8368`.

The current poller state had a fresh `heartbeat_at`
`2026-05-13T19:24:22.782878Z`, but `last_scan_count` was `0`, `scheduled` was
empty, and `seen_count` was `0`. This differs from the status snapshot, which
had already seen an `iteration_0` checkpoint/eval/GIF. It suggests a restarted
poller view that is not a reliable complete history by itself.

### Stale survival comparator

Run:
`curvy-survive-bonus-blank-fast-heavy-batch64-r299-s1141671`

Remote checkpoint directory ended at `iteration_105000.pth.tar`, modified
`2026-05-13 10:18 EDT`, even though `progress_latest.json` was fresh at
`2026-05-13T19:25:30.043822Z` and reported:

- `iteration`: `105000`;
- `learner_train_iter`: `200496`;
- `elapsed_sec`: `15508.350003`.

Again, learner iteration had advanced well past the advertised checkpoint
iteration while no newer checkpoint files were present.

### Healthy mix2 comparator

Run:
`curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021`

Remote checkpoint directory contained a regular sequence from `iteration_0` to
`iteration_200000`. Recent examples:

- `iteration_180000.pth.tar`, modified `2026-05-13 14:39 EDT`;
- `iteration_190000.pth.tar`, modified `2026-05-13 15:04 EDT`;
- `iteration_200000.pth.tar`, modified `2026-05-13 15:28 EDT`.

Remote `progress_latest.json` reported:

- `iteration`: `200000`;
- `learner_train_iter`: `203003`;
- `timestamp`: `2026-05-13T19:36:32.047033Z`.

The healthy row has the same broad app shape, same GPU function family
(`lightzero_curvytron_visual_survival_gpu_cpu40`), same compute label in launch
metadata (`gpu-l4-t4-cpu40`), and also lives under the same deployed app, but
its checkpoint files advance normally.

## Interpretation

The representative stale rows are live enough to refresh
`progress_latest.json`, and Modal still reports the train FunctionCall input as
pending/running-like. They are not live enough to produce new checkpoint files.
The key failure signature is:

```text
progress_latest.timestamp is fresh
progress_latest.learner_train_iter is high
progress_latest.iteration/checkpoint_name is old
train/lightzero_exp/ckpt has no newer iteration_*.pth.tar files
poller state is often stale or incomplete
```

Modal logs show frequent preemption across the app, and some stale rows were
preempted and restarted. Healthy rows were also preempted and recovered, so
preemption is a plausible trigger/amplifier but not a complete explanation.

I did not find direct per-train-call evidence of CUDA failure, explicit timeout,
or Python traceback for the selected stale train calls. The aggregate app log
does contain many background eval/GIF failures and preemptions, so artifact
health should continue to be read separately from trainer checkpoint health.

## Practical Follow-Up Signals

For monitoring these live rows, `train_status=running` and a fresh
`progress_latest.json` should not be used as the primary liveness proof. A row
should be considered stale when learner iteration has moved far beyond the
latest checkpoint iteration but the checkpoint directory has not gained a new
`iteration_*.pth.tar` file in the expected cadence window.

Useful fields for the next read-only check:

- latest checkpoint file mtime from `train/lightzero_exp/ckpt`;
- latest canonical checkpoint file mtime under `checkpoints/lightzero`;
- `progress_latest.learner_train_iter`;
- `progress_latest.iteration`;
- poller `heartbeat_at`, `seen_count`, `scheduled_count`, and `completed_count`;
- per-container preemption timestamps for the train call.

## Follow-Up: Progress Writes Without New Iteration Checkpoints

Question investigated: why can `progress_latest.json` show fresh timestamps and
high `learner_train_iter` values while `_latest_lightzero_iteration_checkpoint`
and run status still see only old `iteration_*.pth.tar` files?

### Commands and files checked

Local code/docs:

```bash
rg -n "progress_latest|SaveCkptHook|save_checkpoint|_latest_lightzero_iteration_checkpoint|ckpt_best|auto_resume|resume_state" src scripts tests docs/working/training/curvytron_architecture_research_2026-05-12
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '1800,1935p'
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '1935,2135p'
nl -ba src/curvyzero/infra/modal/lightzero_curvytron_run_status.py | sed -n '805,945p'
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '5125,5235p'
```

Upstream DI-engine source checked because local `ding` was not importable:

```bash
uv run python - <<'PY'
import ding
PY
uv run --extra modal python - <<'PY'
import ding
PY
curl -L --silent https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/worker/learner/learner_hook.py | nl -ba | sed -n '1,180p'
curl -L --silent https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/worker/learner/base_learner.py | nl -ba | sed -n '378,405p'
```

Representative Modal volume checks:

```bash
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempts/try-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/train/progress_latest.json - | jq '{iteration, learner_train_iter, checkpoint_name, checkpoint_ref, source, timestamp, elapsed_sec}'
modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempts/try-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/train/lightzero_exp/ckpt --json
modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/checkpoints/lightzero --json
modal volume ls curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempts/try-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/train/lightzero_exp/lightzero_resume_state --json
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempts/try-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempt.json - | jq '{attempt_id, status, auto_resume: .config.auto_resume}'
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempts/try-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/status_heartbeat.json - | jq '{stage, status, exp_name_ref}'
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempts/try-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/train/lightzero_exp/formatted_total_config.py - | rg -n "save_ckpt|exp_name|ckpt_name|load_ckpt|auto_resume|hook" -C 2
modal volume get curvyzero-runs training/lightzero-curvytron-visual-survival/curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/attempts/try-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011/train/lightzero_exp/log/learner/learner_logger.txt - | rg -n "save|ckpt|checkpoint|iteration|ERROR|Exception|Traceback|preempt|timeout"
```

The same progress/checkpoint/config/learner-log commands were also run for the
stale k20 mix2 row and the healthy mix2 comparator.

### Code evidence

`_latest_lightzero_iteration_checkpoint(exp_name)` only scans
`exp_name / "ckpt"` for files named `iteration_*.pth.tar`. It ignores
`ckpt_best.pth.tar`, any `latest`-style name, and any checkpoint outside that
directory.

`_write_checkpoint_progress_latest(...)` then records two different concepts:

- `learner_train_iter` is read from `learner.train_iter`;
- `iteration` comes from the highest visible `iteration_*.pth.tar` when one
  exists, and only falls back to `learner_train_iter` when no iteration
  checkpoint exists.

That means a fresh progress write can truthfully say
`learner_train_iter=178018` while still advertising
`checkpoint_name=iteration_0.pth.tar`.

The BaseLearner wrapper is not the writer observed in these stale examples.
The stale progress files have `source="SaveCkptHook.__call__"`. The relevant
wrapper calls DI-engine's original `SaveCkptHook.__call__` first, then calls
`_save_lightzero_resume_sidecar_state(...)`, then writes
`progress_latest.json`. The progress write therefore proves the hook returned,
but it does not prove that the hook created a new checkpoint file.

`_save_lightzero_resume_sidecar_state(...)` also does not force an error when a
matching checkpoint is missing. It computes
`exp_name/ckpt/iteration_<learner.train_iter>.pth.tar`; if that exact file does
not exist, it returns
`{"saved": False, "reason": "matching_iteration_checkpoint_not_found"}`. The
SaveCkptHook wrapper ignores that return value and still writes
`progress_latest.json`.

Upstream DI-engine evidence from `learner_hook.py`:

- `SaveCkptHook.__call__` saves only when
  `engine.rank == 0 and engine.last_iter.val % self._freq == 0`;
- when it saves, it uses `engine.ckpt_name` if set, otherwise
  `iteration_<engine.last_iter.val>.pth.tar`;
- it writes through `save_file(path, state_dict)` and logs
  `learner save ckpt in ...`.

Upstream DI-engine evidence from `base_learner.py`:

- `BaseLearner.save_checkpoint(ckpt_name=None)` temporarily sets
  `self.ckpt_name` only when a caller provides a `ckpt_name`;
- it calls the registered `save_ckpt_after_run` hook and then resets
  `self.ckpt_name = None`.

Together, this makes the most direct explanation: CurvyZero's SaveCkptHook
wrapper is called after every learner iteration hook call, but DI-engine's
underlying SaveCkptHook only creates an `iteration_*.pth.tar` at the configured
frequency and only when its iteration/modulo condition is met. The wrapper's
progress writer currently refreshes progress even on hook calls where the
underlying hook skipped saving.

That normal modulo gating alone does not explain hours of staleness. With
`save_ckpt_after_iter=10000`, a row at learner iteration 178018 should have
crossed many save boundaries. The evidence narrows the failure to one of these
conditions after the last visible checkpoint:

- DI-engine's SaveCkptHook is being called but its save condition is not being
  satisfied for the learner counter it uses;
- the hook is saving under a different name because `engine.ckpt_name` is set;
- the hook is saving somewhere outside the `exp_name/ckpt` path checked by
  status;
- the save attempt is not completing, but no exception reaches the wrapper
  because the original hook returns.

The checked volume state rules out several of those for the representative
rows, below.

### Representative stale rows

Stale k0 mix2 row:
`curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011`

Fresh progress at recheck:

- `iteration`: `0`;
- `learner_train_iter`: `178018`;
- `checkpoint_name`: `iteration_0.pth.tar`;
- `checkpoint_ref`: the attempt-local
  `train/lightzero_exp/ckpt/iteration_0.pth.tar`;
- `source`: `SaveCkptHook.__call__`;
- `timestamp`: `2026-05-13T19:47:00.152701Z`.

The attempt-local checkpoint directory contained exactly:

- `iteration_0.pth.tar`, modified `2026-05-13 07:53 EDT`;
- `ckpt_best.pth.tar`, modified `2026-05-13 07:53 EDT`.

No `latest.pth.tar`, no later non-iteration checkpoint, and no later
`iteration_*.pth.tar` was visible in that checked directory. The canonical
`checkpoints/lightzero` directory did not exist for this run, so status would
fall through to the same attempt-local path referenced by progress.

Stale k20 mix2 row:
`curvy-mix2clean-r50-blank25-scr25-rf-s8-c32-l32-rep0-k10-c3-s2106031`

Fresh progress at recheck:

- `iteration`: `20000`;
- `learner_train_iter`: `145986`;
- `checkpoint_name`: `iteration_20000.pth.tar`;
- `source`: `SaveCkptHook.__call__`;
- `timestamp`: `2026-05-13T19:47:01.443757Z`.

The attempt-local checkpoint directory contained only:

- `iteration_0.pth.tar`;
- `iteration_10000.pth.tar`;
- `iteration_20000.pth.tar`;
- `ckpt_best.pth.tar`.

The corresponding `lightzero_resume_state` directories, both attempt-local and
canonical, contained only sidecars through `iteration_20000`. That matches the
code path where sidecar save returns
`matching_iteration_checkpoint_not_found` after later learner iterations
because no exact `iteration_<learner_train_iter>.pth.tar` exists.

Healthy mix2 comparator:
`curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021`

Fresh progress at recheck:

- `iteration`: `200000`;
- `learner_train_iter`: `208610`;
- `checkpoint_name`: `iteration_200000.pth.tar`;
- `source`: `SaveCkptHook.__call__`;
- `timestamp`: `2026-05-13T19:49:53.228539Z`.

Its attempt-local checkpoint directory contained the expected sequence
`iteration_0.pth.tar`, `iteration_10000.pth.tar`, ..., through
`iteration_200000.pth.tar`, plus `ckpt_best.pth.tar`. This comparator uses the
same configured `save_ckpt_after_iter=10000` and the same visible
`source="SaveCkptHook.__call__"` progress-writing path, but the iteration files
advance normally.

### Answers to the specific hypotheses

Are non-iteration checkpoint names being written?

For the checked stale rows, only `ckpt_best.pth.tar` was present as a
non-iteration checkpoint name. No `latest.pth.tar` or later non-iteration name
was visible in `train/lightzero_exp/ckpt`. `ckpt_best.pth.tar` was old in the
k0 row and does not explain fresh high learner iterations.

Are saves failing after the wrapper?

The BaseLearner and SaveCkptHook wrappers both call the original save path
before writing progress. If the original raises, the subsequent progress write
would not run. I did not find learner-log or per-call-log evidence of an
exception, traceback, timeout, or checkpoint write error in the checked rows.
However, `_save_lightzero_resume_sidecar_state` can fail silently with
`matching_iteration_checkpoint_not_found`; that failure is not logged and does
not block the progress write.

Is `save_checkpoint` being called only for `ckpt_best`/latest?

The observed stale progress is not from the BaseLearner
`save_checkpoint` wrapper; it is from `SaveCkptHook.__call__`. Upstream
DI-engine can use `engine.ckpt_name` when `BaseLearner.save_checkpoint(ckpt_name)`
has set it, which is how best-checkpoint style names can be produced. But the
checked stale directories do not show a fresh mutable `ckpt_best` or `latest`
file being updated after the last iteration checkpoint.

Could `exp_name/ckpt` differ from the status path?

No evidence of that for the checked stale rows. The generated
`formatted_total_config.py` has `exp_name` equal to the attempt-local
`.../train/lightzero_exp`, and the stale progress `checkpoint_ref` points under
that same `.../train/lightzero_exp/ckpt` directory. For the checked stale rows,
canonical `checkpoints/lightzero` was absent, so `_checkpoint_summary` would use
the same attempt-local `exp_name/ckpt` directory. There is still a general
status-reader hazard when a canonical directory exists but is stale, because
status checks canonical first, then attempt-local; that was not the explanation
for these sampled stale rows.

Does auto-resume/preemption reuse a path incorrectly?

The stale k0 `attempt.json` showed auto-resume selected
`source_kind=current_attempt_lightzero_exp`,
`checkpoint_iteration=0`, and an attempt-local checkpoint ref under
`train/lightzero_exp/ckpt/iteration_0.pth.tar`; it also found a matching
`iteration_0.resume_state.pkl`. This is expected reuse of the same attempt/run
path after restart, not direct evidence of a wrong path. It can still amplify
the problem: after preemption, if the latest visible checkpoint is old, resume
starts from that old checkpoint and future progress writes can remain fresh
even while no new checkpoint files appear.

### Current best reading

The fresh progress/high `learner_train_iter` clue is explained by the progress
writer's semantics: it records current learner iteration separately from the
latest checkpoint discovered on disk. In the stale rows, the wrapper keeps
writing progress from `SaveCkptHook.__call__`, but the visible
`exp_name/ckpt` directory does not gain the expected `iteration_10000` cadence
files. For sampled rows, this is not a status-path mismatch and not merely
newer non-iteration checkpoint names. The next code-level question is why the
underlying DI-engine SaveCkptHook stops producing `iteration_*.pth.tar` at its
configured frequency for some resumed/preempted runs while it continues to do
so for healthy rows with the same wrapper and cadence.

## Narrow Recheck: Iteration-0-Only Stale Rows

This pass focused only on rows where `progress_latest.json` is fresh, the
learner counter is high, and the visible checkpoint set is still only
`iteration_0.pth.tar` plus `ckpt_best.pth.tar`.

### Selection command

```bash
jq -r '.rows[] | select((.progress_exists==true) and (.latest_checkpoint=="iteration_0") and ((.iteration // 0)==0)) | [.run_id,.attempt_id,.timestamp,.elapsed_sec,.checkpoint_count,.latest_checkpoint,.train_status] | @tsv' artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json | head -60
```

The artifact snapshot returned six matching rows. I sampled three stale rows
and one healthy comparator:

- stale: `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011`;
- stale: `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021`;
- stale: `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051`;
- healthy: `curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021`.

### Exact volume commands

For each sampled run, I used the same command shapes:

```bash
modal volume get curvyzero-runs <run>/attempts/<attempt>/train/progress_latest.json - | sed '/^✓ Finished/d' | jq '{iteration, learner_train_iter, checkpoint_name, checkpoint_ref, source, timestamp, elapsed_sec}'
modal volume ls curvyzero-runs <run>/attempts/<attempt>/train/lightzero_exp/ckpt --json | jq -r '"count=\(length)", (.[] | [.Filename,."Created/Modified",.Size] | @tsv)'
modal volume ls curvyzero-runs <run>/attempts/<attempt>/train/lightzero_exp/lightzero_resume_state --json | jq -r '"count=\(length)", (.[] | [.Filename,."Created/Modified",.Size] | @tsv)'
modal volume ls curvyzero-runs <run>/checkpoints/lightzero_resume_state --json | jq -r '"count=\(length)", (.[] | [.Filename,."Created/Modified",.Size] | @tsv)'
modal volume ls curvyzero-runs <run>/checkpoints/lightzero --json
modal volume get curvyzero-runs <run>/attempts/<attempt>/attempt.json - | sed '/^✓ Finished/d' | jq '{attempt_id,status,auto_resume:.config.auto_resume}'
```

Local code was re-read with:

```bash
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '1804,1866p'
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '2043,2124p'
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '2127,2205p'
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '5128,5365p'
nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py | sed -n '3528,3562p'
nl -ba src/curvyzero/infra/modal/run_management.py | sed -n '256,280p'
```

### Stale rows: current evidence

`curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011`

- Progress recheck: `iteration=0`, `learner_train_iter=180035`,
  `checkpoint_name=iteration_0.pth.tar`,
  `source=SaveCkptHook.__call__`,
  `timestamp=2026-05-13T19:52:10.276693Z`.
- Attempt-local `ckpt` count was `2`: `iteration_0.pth.tar` modified
  `2026-05-13 07:53 EDT`, and `ckpt_best.pth.tar` modified
  `2026-05-13 07:53 EDT`.
- Attempt-local `lightzero_resume_state` count was `1`:
  `iteration_0.resume_state.pkl`, modified `2026-05-13 08:38 EDT`.
- Canonical `checkpoints/lightzero_resume_state` count was also `1`:
  `iteration_0.resume_state.pkl`, modified `2026-05-13 08:38 EDT`.
- Canonical `checkpoints/lightzero` did not exist.
- `attempt.json` reported `status=running`, auto-resume
  `checkpoint_iteration=0`, `source_kind=current_attempt_lightzero_exp`,
  `resume_state_found=true`, and `resume_state_source_kind=run_resume_state_mirror`.

`curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021`

- Progress recheck: `iteration=0`, `learner_train_iter=36415`,
  `checkpoint_name=iteration_0.pth.tar`,
  `source=SaveCkptHook.__call__`,
  `timestamp=2026-05-13T19:48:41.130933Z`.
- Attempt-local `ckpt` count was `2`: `iteration_0.pth.tar` modified
  `2026-05-13 07:55 EDT`, and `ckpt_best.pth.tar` modified
  `2026-05-13 07:55 EDT`.
- Attempt-local `lightzero_resume_state` count was `1`:
  `iteration_0.resume_state.pkl`, modified `2026-05-13 13:50 EDT`.
- Canonical `checkpoints/lightzero_resume_state` count was also `1`:
  `iteration_0.resume_state.pkl`, modified `2026-05-13 13:50 EDT`.
- Canonical `checkpoints/lightzero` did not exist.
- `attempt.json` reported `status=running`, auto-resume
  `checkpoint_iteration=0`, `source_kind=current_attempt_lightzero_exp`,
  `resume_state_found=true`, and `resume_state_source_kind=run_resume_state_mirror`.

`curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051`

- Progress recheck: `iteration=0`, `learner_train_iter=69261`,
  `checkpoint_name=iteration_0.pth.tar`,
  `source=SaveCkptHook.__call__`,
  `timestamp=2026-05-13T19:51:34.129990Z`.
- Attempt-local `ckpt` count was `2`: `iteration_0.pth.tar` modified
  `2026-05-13 11:06 EDT`, and `ckpt_best.pth.tar` modified
  `2026-05-13 11:05 EDT`.
- Attempt-local `lightzero_resume_state` count was `1`:
  `iteration_0.resume_state.pkl`, modified `2026-05-13 13:24 EDT`.
- Canonical `checkpoints/lightzero_resume_state` count was also `1`:
  `iteration_0.resume_state.pkl`, modified `2026-05-13 13:24 EDT`.
- Canonical `checkpoints/lightzero` did not exist.
- `attempt.json` reported `status=running`, auto-resume
  `checkpoint_iteration=0`, `source_kind=current_attempt_lightzero_exp`,
  `resume_state_found=true`, and `resume_state_source_kind=run_resume_state_mirror`.

### Healthy comparator

`curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021`

- Progress recheck: `iteration=200000`, `learner_train_iter=208610`,
  `checkpoint_name=iteration_200000.pth.tar`,
  `source=SaveCkptHook.__call__`,
  `timestamp=2026-05-13T19:49:53.228539Z`.
- Attempt-local `ckpt` count was `22`, including `iteration_0.pth.tar`
  modified `2026-05-13 07:51 EDT`, `iteration_10000.pth.tar` modified
  `2026-05-13 08:12 EDT`, `iteration_200000.pth.tar` modified
  `2026-05-13 15:28 EDT`, and `ckpt_best.pth.tar`.
- Attempt-local `lightzero_resume_state` count was `21`, including
  `iteration_0.resume_state.pkl`, `iteration_10000.resume_state.pkl`, and
  `iteration_200000.resume_state.pkl`.
- Canonical `checkpoints/lightzero_resume_state` count was also `21`, with the
  same sampled sidecars through `iteration_200000`.
- Canonical `checkpoints/lightzero` did not exist for this running comparator
  either, so the healthy/stale difference is not the canonical checkpoint
  mirror. The difference is that the healthy attempt-local `ckpt` and sidecar
  dirs advanced every 10k iterations.
- `attempt.json` reported auto-resume `found=false` for this healthy row, so it
  did not begin by resuming from an existing `iteration_0` checkpoint.

### Resume/progress code reading

The progress writer uses `runs.write_json(..., mode="wb")`, so
`progress_latest.json` is a persistent volume file that can remain visible
after a Modal process exits or is preempted. A stale old progress file can
therefore survive across Modal restarts.

However, a *fresh* `progress_latest.timestamp` cannot advance by itself. In
the code path checked here, fresh checkpoint progress is written only from
inside an active training process after the progress hooks are installed:

- `_install_lightzero_full_resume_state_hooks(...)` wraps
  `SaveCkptHook.__call__`;
- the wrapper first calls DI-engine's original save hook;
- it then calls `_save_lightzero_resume_sidecar_state(...)`;
- it finally calls `_write_checkpoint_progress_latest(...)` with
  `source="SaveCkptHook.__call__"`.

The sidecar code requires an exact matching checkpoint file:
`exp_name/ckpt/iteration_<learner.train_iter>.pth.tar`. If that file is absent,
it returns `matching_iteration_checkpoint_not_found` and does not write a new
sidecar. This is exactly consistent with the stale samples: progress was
refreshed at high learner counters, but sidecar dirs still contain only
`iteration_0.resume_state.pkl`.

Auto-resume scans the current attempt `lightzero_exp/ckpt`, prior attempts, and
the canonical checkpoint mirror; it selects the highest numbered
`iteration_*.pth.tar` and explicitly ignores `ckpt_best`. For all three stale
samples, `attempt.json` shows auto-resume selected only `iteration_0` and found
only an `iteration_0` resume sidecar. That makes the sampled rows look like
resumed/restarted processes that continued training far beyond learner iter 0,
kept refreshing progress from live hook calls, but never produced the first
post-resume `iteration_10000.pth.tar` or matching sidecar.

### Narrow conclusion

For the iteration-0-only stale rows checked here, fresh progress must have come
from an alive training process at the time of the progress timestamp, not from
the mere survival of a volume file across restarts. The persisted progress file
can survive restarts, but it cannot update its own timestamp or
`learner_train_iter`.

The strongest concrete evidence is the paired absence of both later checkpoint
files and later sidecar files:

```text
stale: progress learner_train_iter 36k/69k/180k, ckpt only iteration_0, sidecar only iteration_0
healthy: progress learner_train_iter 208k, ckpt through iteration_200000, sidecar through iteration_200000
```

This points away from a status-reader path mismatch for these sampled rows and
toward a live post-resume training path where `SaveCkptHook.__call__` keeps
running but the underlying save hook does not create the expected
`iteration_10000.pth.tar` cadence checkpoint.
