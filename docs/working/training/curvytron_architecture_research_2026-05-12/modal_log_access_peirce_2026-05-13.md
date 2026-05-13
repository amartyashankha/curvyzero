# Modal Log Access - 2026-05-13

Purpose: record the local Modal CLI workflow for inspecting CurvyTron app,
function-call, and container logs. This was a read-only check. No source code,
Modal apps, volumes, or function calls were modified.

## Local CLI Surface

The local command is:

```bash
uv run --extra modal modal ...
```

Useful help commands:

```bash
uv run --extra modal modal --help
uv run --extra modal modal app --help
uv run --extra modal modal app logs --help
uv run --extra modal modal container --help
uv run --extra modal modal container logs --help
```

Important result: this installed CLI does not have a top-level
`modal function` command:

```bash
uv run --extra modal modal function --help
```

That fails with:

```text
No such command 'function'.
```

Function-call logs are fetched through `modal app logs --function-call fc-*`,
not through a separate `modal function logs` command.

## App ID Lookup

List apps as JSON:

```bash
uv run --extra modal modal app list --json
```

Current trainer app found:

```text
App ID: ap-2mvquK3ZZJvDqleyHS088M
Description: curvyzero-lightzero-curvytron-visual-survival-train
State: deployed
Tasks at check time: about 509
```

Reliable local extraction:

```bash
APP_ID="$(
  uv run --extra modal modal app list --json |
    jq -r '.[] | select(.Description=="curvyzero-lightzero-curvytron-visual-survival-train") | ."App ID"'
)"
```

Caveat: using the displayed description as the app identifier did not work here:

```bash
uv run --extra modal modal app logs curvyzero-lightzero-curvytron-visual-survival-train \
  --function-call fc-01KRGRFNESEWMN25TV125DAQ88 \
  --tail 40 \
  --timestamps \
  --show-function-call-id \
  --show-container-id
```

It returned:

```text
No App with name 'curvyzero-lightzero-curvytron-visual-survival-train' found in the 'main' environment.
```

Use the App ID for reliable log fetching.

## Finding Function Calls For Rows

The grouped launch artifacts are the cleanest source of original train/poller
function-call IDs:

```bash
jq -r '.records[] | [.row_id,.run_id,.train_function_call_id,.poller_function_call_id] | @tsv' \
  artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_launch.json |
  head -n 5
```

Example output shape:

```text
r001  curvy-mix3cur-r25-blank75-rf-s8-c32-l32-rep0-k10-c1-s2301011  fc-01KRGRFAPK9VPNK2NMSCXFKAAM  fc-01KRGRFAJBZ3GZAS94MK7E7NJ4
r002  curvy-mix3cur-r25-blank75-rb-s8-c32-l32-rep0-k10-c1-s2301011  fc-01KRGRFAXVR2QYY4XEN2PP1CWG  fc-01KRGRFATJZPQK63ME58N6FPP0
```

The prune plan is useful when choosing stale candidates because it includes
status fields:

```bash
jq -r '.kill_rows[] |
  select(.batch=="mix3" and (.train_root_exists==false)) |
  [.row_id,.run_id,.attempt_id,.train_function_call_id,.poller_function_call_id,
   .train_status,.train_stage,.latest_checkpoint,.train_root_exists,.status_heartbeat_exists] |
  @tsv' \
  artifacts/local/curvytron_pruning/curvytron_prune_plan_20260513a.json |
  head -n 30
```

## Fetching A Specific Function Call

Use App ID plus function-call ID:

```bash
uv run --extra modal modal app logs "$APP_ID" \
  --function-call "$FUNCTION_CALL_ID" \
  --since 8h \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

For concise failure checks, add a source filter:

```bash
uv run --extra modal modal app logs "$APP_ID" \
  --function-call "$FUNCTION_CALL_ID" \
  --since 8h \
  --source stderr \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

Observed working example for a stale-looking mix3 train call:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --function-call fc-01KRGRFNESEWMN25TV125DAQ88 \
  --since 8h \
  --source stderr \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

Returned one stderr line:

```text
2026-05-13 10:39:46-04:00 fu-qnPwXhmTXuzKA5j46JxeqG fc-01KRGRFNESEWMN25TV125DAQ88 ta-01KRGWEPTZF36DQWH2EEX71KWE [05-13 14:39:46] WARNING  not found transformer, please install it using: pip install transformers language_transformer.py:9
```

The paired poller function call for that row:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --function-call fc-01KRGRFNB99TYS3TN4V4SVYKA9 \
  --since 8h \
  --source system \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

returned no output in this check. No output is not proof that the row never ran;
it only means no matching retained log lines were returned for that exact
function-call/source filter.

Observed working example for a healthy mix3 train call:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --function-call fc-01KRGRFB6QJHFYPYR7XFYC89HR \
  --since 8h \
  --source stderr \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

Returned the same transformer warning shape at `2026-05-13 09:46:28-04:00`.
The train-call logs themselves were sparse; health still needs artifact/status
reads, not train-call logs alone.

## Search By Run ID

To find child eval/GIF calls or later log lines that mention a row, search by
run id:

```bash
uv run --extra modal modal app logs "$APP_ID" \
  --search "$RUN_ID" \
  --since 8h \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id
```

For the stale-looking row
`curvy-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051`, a broad
`--search` found later eval/GIF JSON logs and artifact refs around
`2026-05-13 10:47:38-04:00`, including `iteration_0` eval artifacts. That is a
useful signal that a stale snapshot can be outdated, and that child worker logs
may be more informative than the original train/poller call logs.

Caveat: broad search output can be enormous because a single JSON log event is
displayed as many physical lines. Prefer `--source stderr` or add local
guards while exploring:

```bash
uv run --extra modal modal app logs "$APP_ID" \
  --search "$RUN_ID" \
  --since 8h \
  --source stderr \
  --timestamps \
  --show-function-call-id \
  --show-container-id
```

or, for a quick glance:

```bash
uv run --extra modal modal app logs "$APP_ID" \
  --search "$RUN_ID" \
  --since 8h \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 80
```

`head` only limits local display; it is not a Modal-side log limit.

## Container Logs

When `modal app logs` is run with `--show-container-id`, it prints `ta-*`
container IDs. Those can be passed to:

```bash
uv run --extra modal modal container logs "$CONTAINER_ID" \
  --tail 20 \
  --timestamps
```

Observed working example:

```bash
uv run --extra modal modal container logs ta-01KRGWEPTZF36DQWH2EEX71KWE \
  --tail 20 \
  --timestamps
```

returned the same transformer warning as the app/function-call query.

Container logs are less precise than app logs for row triage because they lose
the function-call prefix and can include very large JSON events from whatever
ran in that container. Use container logs after `app logs` has identified a
specific `ta-*`, not as the first pass.

List currently running containers for the app:

```bash
uv run --extra modal modal container list \
  --app-id ap-2mvquK3ZZJvDqleyHS088M \
  --json |
  jq 'length, .[0:5]'
```

At check time this returned `491` running containers for the trainer app.

## Error Searches

Broad app-level searches work, but they are noisy:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search Traceback \
  --since 8h \
  --tail 20 \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 80
```

This returned multiple `Traceback (most recent call last):` lines around
`2026-05-13 10:12-10:19-04:00`, plus later JSON error summaries.

Targeted DataLoss retry search:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search DataLossError \
  --since 8h \
  --tail 20 \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 80
```

This returned retry-labelled volume commit lines such as:

```text
2026-05-13 15:37:25-04:00 fc-01KRHD1GX5RZR21W40X6YWZGG7 ta-01KRHD1DKPY3V8BCKV3E756GZJ {"attempt": 1, "error": "failed to publish commit to server", "error_type": "DataLossError", "event": "modal_volume_commit_retry", "label": "checkpoint_eval_and_inspect", "next_delay_sec": 1.205}
```

Those are useful for current volume-pressure diagnostics, but they are app-wide
and should not be attributed to a specific training row unless the line or a
nearby filtered query includes the row id/run id.

## Follow-up: Stale Restart Hypothesis

Hypothesis tested: some stale rows may still be training or restarting, while no
later `iteration_*.pth.tar` checkpoints are visible.

Follow-up used App ID:

```text
ap-2mvquK3ZZJvDqleyHS088M
```

At this point a newer local prune snapshot existed:

```bash
jq '{created_at, keep_count, kill_count, counts_by_batch_keep, counts_by_batch_kill}' \
  artifacts/local/curvytron_pruning/curvytron_prune_plan_20260513c.json
```

It reported `keep_count=212`, `kill_count=609`, with mix3 split as
`126` keep and `174` kill. The exact stale sample below comes from that newer
snapshot, not from the earlier `20260513a` example.

Stale sample selected:

```text
row_id: r040
run_id: curvy-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051
attempt_id: try-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051
train_function_call_id: fc-01KRGRFNESEWMN25TV125DAQ88
poller_function_call_id: fc-01KRGRFNB99TYS3TN4V4SVYKA9
snapshot fields: latest_checkpoint=null, train_root_exists=false,
status_heartbeat_exists=false, reason=pruned_duplicate_or_stale
```

Metadata extraction:

```bash
jq '.kill_rows[] |
  select(.run_id=="curvy-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051") |
  {row_id, batch, run_id, attempt_id, train_function_call_id,
   poller_function_call_id, latest_checkpoint, train_root_exists,
   status_heartbeat_exists, reason}' \
  artifacts/local/curvytron_pruning/curvytron_prune_plan_20260513c.json
```

Exact train-call query:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --function-call fc-01KRGRFNESEWMN25TV125DAQ88 \
  --since 12h \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id \
  --tail 20
```

Observed output was only the import warning:

```text
2026-05-13 10:39:46-04:00 fu-qnPwXhmTXuzKA5j46JxeqG fc-01KRGRFNESEWMN25TV125DAQ88 ta-01KRGWEPTZF36DQWH2EEX71KWE [05-13 14:39:46] WARNING  not found transformer, please install it using: pip install transformers language_transformer.py:9
```

Exact poller-call query:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --function-call fc-01KRGRFNB99TYS3TN4V4SVYKA9 \
  --since 12h \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id \
  --tail 20
```

returned no output.

Exact train-container checks for the stale row also returned no matching
preemption, timeout, or checkpoint evidence:

```bash
uv run --extra modal modal container logs ta-01KRGWEPTZF36DQWH2EEX71KWE \
  --search preempt \
  --since 12h \
  --timestamps \
  --tail 20

uv run --extra modal modal container logs ta-01KRGWEPTZF36DQWH2EEX71KWE \
  --search timeout \
  --since 12h \
  --timestamps \
  --tail 20

uv run --extra modal modal container logs ta-01KRGWEPTZF36DQWH2EEX71KWE \
  --search checkpoint \
  --since 12h \
  --timestamps \
  --tail 20
```

Checkpoint-path search for the stale row found `iteration_0` child eval/GIF
logs, but not a later `iteration_10000` checkpoint reference:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search 'curvy-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051/attempts/try-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051/train/lightzero_exp/ckpt/iteration_0.pth.tar' \
  --since 12h \
  --timestamps \
  --show-function-call-id \
  --show-container-id \
  --tail 10 |
  head -n 80
```

Evidence returned around `2026-05-13 10:47:38-04:00` on child worker
`fc-01KRGWRZ463M8EFPGYFYD9H4DX` / `ta-01KRGTB4C5HAWPR822QC7QFFW6`, including:

```text
"checkpoint_label": "iteration_0"
"checkpoint_ref": "training/lightzero-curvytron-visual-survival/curvy-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051/attempts/try-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051/train/lightzero_exp/ckpt/iteration_0.pth.tar"
"eval_id": "live_checkpoint_iteration_0"
"failure": null
```

Later checkpoint search:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search 'curvy-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051/attempts/try-mix3cur-r50-blank50-rf-s8-c32-l32-rep0-k10-c5-s2302051/train/lightzero_exp/ckpt/iteration_10000.pth.tar' \
  --since 12h \
  --timestamps \
  --show-function-call-id \
  --show-container-id \
  --tail 10 |
  head -n 80
```

returned no output.

Healthy comparison sample:

```text
row_id: r003
run_id: curvy-mix3cur-r25-blank75-rb-s8-c32-l32-rep0-k10-c2-s2301021
train_function_call_id: fc-01KRGRFB6QJHFYPYR7XFYC89HR
snapshot fields: latest_checkpoint=iteration_20000, train_root_exists=true,
status_heartbeat_exists=true
```

The exact train-call log for that healthy row was also sparse, returning only
the same transformer warning shape:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --function-call fc-01KRGRFB6QJHFYPYR7XFYC89HR \
  --since 12h \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id \
  --tail 20
```

Checkpoint-path searches did find healthy later checkpoints through child
eval/GIF worker logs:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search 'curvy-mix3cur-r25-blank75-rb-s8-c32-l32-rep0-k10-c2-s2301021/attempts/try-mix3cur-r25-blank75-rb-s8-c32-l32-rep0-k10-c2-s2301021/train/lightzero_exp/ckpt/iteration_20000.pth.tar' \
  --since 12h \
  --timestamps \
  --show-function-call-id \
  --show-container-id \
  --tail 10 |
  head -n 80
```

Evidence returned around `2026-05-13 10:41:02-04:00` on child worker
`fc-01KRGWEJXGQRK0NDJN1R596PWP` / `ta-01KRGVP2N9H5S2BTA80JXCZE00`, including:

```text
"checkpoint_label": "iteration_20000"
"checkpoint_ref": "training/lightzero-curvytron-visual-survival/curvy-mix3cur-r25-blank75-rb-s8-c32-l32-rep0-k10-c2-s2301021/attempts/try-mix3cur-r25-blank75-rb-s8-c32-l32-rep0-k10-c2-s2301021/train/lightzero_exp/ckpt/iteration_20000.pth.tar"
"eval_id": "live_checkpoint_iteration_20000"
"failure": null
```

Broad term searches used for the hypothesis:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search save_checkpoint \
  --since 12h \
  --tail 20 \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 80

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search progress_latest \
  --since 12h \
  --tail 20 \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 80

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search timeout \
  --since 12h \
  --tail 20 \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 80

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search resched \
  --since 12h \
  --tail 20 \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 80
```

Results:

- `save_checkpoint` matched unrelated historical failure JSON fields such as
  `"learner_save_checkpoint": null`; it did not expose a current stale mix3
  save failure.
- `progress_latest` returned no output.
- `timeout` returned no output.
- `resched` returned no output.

App-wide preemption search did return real restart evidence:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --search 'Runner interrupted' \
  --since 12h \
  --tail 20 \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id |
  head -n 80
```

Example lines:

```text
2026-05-13 15:29:04-04:00 ta-01KRHCT3XP014TC8KZTP0HBX0D Runner interrupted due to worker preemption. Your Function will be restarted with the same input...
2026-05-13 15:30:28-04:00 ta-01KRGK4K0AG0CVV3098H4AR33N Runner interrupted due to worker preemption. Your Function will be restarted with the same input...
```

For container `ta-01KRGK4K0AG0CVV3098H4AR33N`, a container-context query mapped
the preemption to a specific function call:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --container ta-01KRGK4K0AG0CVV3098H4AR33N \
  --since 12h \
  --tail 40 \
  --timestamps \
  --show-function-call-id \
  --show-container-id |
  head -n 120
```

Observed context:

```text
2026-05-13 07:56:55-04:00 fc-01KRGJQE6XZGCEKECSAHTW6H8H ta-01KRGK4K0AG0CVV3098H4AR33N [05-13 11:56:55] WARNING  not found transformer, please install it using: pip install transformers language_transformer.py:9
2026-05-13 15:30:28-04:00 ta-01KRGK4K0AG0CVV3098H4AR33N Runner interrupted due to worker preemption. Your Function will be restarted with the same input...
```

Local manifest/prune mapping identified
`fc-01KRGJQE6XZGCEKECSAHTW6H8H` as mix2 row `r087`,
`curvy-mix2clean-r50-blank25-scr25-rb-s8-c32-l32-repM-k10-c3-s2106031`,
with `latest_checkpoint=iteration_50000`, `train_root_exists=true`, and
`status_heartbeat_exists=true`. Querying the function call showed the same call
coming back on a new container:

```bash
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --function-call fc-01KRGJQE6XZGCEKECSAHTW6H8H \
  --since 12h \
  --timestamps \
  --show-function-id \
  --show-function-call-id \
  --show-container-id \
  --tail 30
```

Observed context:

```text
2026-05-13 07:56:55-04:00 fu-qnPwXhmTXuzKA5j46JxeqG fc-01KRGJQE6XZGCEKECSAHTW6H8H ta-01KRGK4K0AG0CVV3098H4AR33N [05-13 11:56:55] WARNING  not found transformer, please install it using: pip install transformers language_transformer.py:9
2026-05-13 15:30:45-04:00 fu-qnPwXhmTXuzKA5j46JxeqG fc-01KRGJQE6XZGCEKECSAHTW6H8H ta-01KRHD3FBA2JTJZ37BWSKW4XP9 [05-13 19:30:45] WARNING  not found transformer, please install it using: pip install transformers language_transformer.py:9
```

Interpretation:

- The sampled stale mix3 row had evidence for `iteration_0` child evaluation,
  but no retained log evidence of later checkpoint refs, preemption,
  reschedule, timeout, checkpoint save failure, progress-writer failure, or
  ongoing restart.
- The healthy comparison row showed the same sparse train-call log shape, but
  checkpoint-path searches did find later checkpoint refs through child workers.
- Modal preemption/restart is real in this app. The concrete mapped example was
  a healthy/kept mix2 row with `iteration_50000`, not the sampled stale mix3 row.
- This follow-up does not support the exact hypothesis for the sampled stale
  row. It does support using checkpoint-path searches plus exact function-call
  and container checks before attributing app-wide preemption lines to stale
  rows.

Extra caveats:

- App-wide preemption lines often omit function-call IDs. Attribute them only
  after a container-context query or direct function-call query gives a `fc-*`.
- Original train-call logs do not reliably print checkpoint names. Child
  eval/GIF logs were a better source for visible `iteration_*.pth.tar` refs.
- No output from `modal app logs` is absence in the retained/queryable logs, not
  proof the event never happened.

## Recommended Workflow

1. Resolve the App ID with `modal app list --json`; do not rely on the displayed
   app description as a log identifier.
2. Resolve `run_id`, `train_function_call_id`, and `poller_function_call_id`
   from the grouped launch artifact or prune/status plan.
3. Query the exact train call with `modal app logs "$APP_ID" --function-call
   "$TRAIN_FC" --since ... --source stderr --timestamps --show-function-id
   --show-function-call-id --show-container-id`.
4. Query the exact poller call the same way. Try `--source system`, `stdout`,
   and no source filter if needed.
5. If exact train/poller logs are sparse, search exact checkpoint paths such as
   `.../ckpt/iteration_0.pth.tar`, `iteration_10000.pth.tar`, and the latest
   expected checkpoint. Child eval/GIF workers may expose checkpoint refs even
   when the train call does not.
6. Search by `run_id` to find child eval/GIF workers and artifact refs, but keep
   output bounded with `head` because JSON log entries can display as many
   physical lines.
7. Use `container logs` only after app logs gives a useful `ta-*` container id.
   App-wide preemption lines should not be attributed to a row until a
   container-context or function-call query maps them to a `fc-*`.
8. Keep output bounded. `--tail` counts log entries, not displayed physical
   lines; one JSON entry can produce thousands of lines.
