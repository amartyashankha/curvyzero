# CurvyTron Modal Artifact Liveness Audit

Date: 2026-05-13

Scope: read-only audit of Modal volume artifacts and captured Modal logs for sampled
CurvyTron training rows. No source files were changed.

## Question

Are the rows that look stale in the status table actually alive but not
checkpointing, or are the status signals stale/misleading?

Short answer: both, depending on the signal. The sampled stale rows are very
likely still executing training code, because `progress_latest.json` continues
to be rewritten with increasing `learner_train_iter`. However, the checkpoint,
eval/GIF, and poller signals are stale/misleading for these rows: no new
`iteration_*.pth.tar` checkpoint files appear after `iteration_0`, while the
status table continues to report `train_status=running` and the poller JSON can
show old `status=running` heartbeats.

## Commands Run

Status snapshot and raw-file inventory:

```bash
pwd
rg --files artifacts/local/curvytron_pruning | sed -n '1,160p'
ls -la docs/working/training/curvytron_architecture_research_2026-05-12
find artifacts/local/curvytron_pruning/status_chunks_20260513e -maxdepth 2 -type f | sort | sed -n '1,120p'
find artifacts/local/curvytron_pruning/status_chunks_20260513f/raw_status_files -maxdepth 2 -type f | sort | sed -n '1,160p'
git status --short
```

Snapshot row extraction:

```bash
jq '.rows[0] | keys' artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json
jq '.rows[] | select((.run_id // "") | test("curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021|curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051|curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011|curvy-mix3cur-r50-blank50-rb-s16-c32-l32-repH-k10-c1-s2302011|curvy-mix3cur-r75-blank25-rf-s16-c32-l32-repM-k10-c1-s2303011"))' artifacts/local/curvytron_pruning/status_chunks_20260513e/combined_status.json
for f in artifacts/local/curvytron_pruning/status_chunks_20260513f/raw_status_files/*/*.json; do jq -c '.' "$f" | cut -c1-2000; done
```

Raw status summaries:

```bash
for d in artifacts/local/curvytron_pruning/status_chunks_20260513f/raw_status_files/curvy-*; do
  run=$(basename "$d")
  jq '{run_id, event, iteration, learner_train_iter, elapsed_sec, timestamp, updated_at, checkpoint_name, checkpoint_ref}' "$d/progress_latest.json"
  jq '{run_id, stage, status, heartbeat_at, started_at, modal_task_id, command_auto_resume:.command.auto_resume, command_max_env_step:.command.max_env_step, command_max_train_iter:.command.max_train_iter, command_save_ckpt_after_iter:.command.save_ckpt_after_iter}' "$d/status_heartbeat.json"
  jq '{run_id, status, started_at, heartbeat_at, last_scan_count, seen_count, scheduled_count, completed_count, gif_scheduled_count, gif_completed_count, outstanding_count, train_done, stable_polls}' "$d/checkpoint_eval_poller.json"
done
```

Captured Modal-log searches:

```bash
rg -n "curvy-mix2clean-r50-scr50-rb|curvy-mix3cur-r40-blank20|iteration_0|iteration_[1-9]|Traceback|PytorchStreamReader|ERROR|WARNING|auto_resume|SaveCkptHook|learner_train_iter|checkpoint" artifacts/local/curvytron_pruning/status_chunks_20260513e/log_*.txt
modal app logs --help | sed -n '1,180p'
modal app logs ap-2mvquK3ZZJvDqleyHS088M --function-call fc-01KRGJQ8XDR9X2YWXG9QGTV1VA --timestamps --show-function-call-id --show-container-id --tail 80
modal app logs ap-2mvquK3ZZJvDqleyHS088M --function-call fc-01KRGRHAZ3ZVRBVM5YNJN1RNNN --timestamps --show-function-call-id --show-container-id --tail 80
modal app logs ap-2mvquK3ZZJvDqleyHS088M --function-call fc-01KRGRHASW04PTAHP7AG86HQBM --timestamps --show-function-call-id --show-container-id --tail 120
modal app logs ap-2mvquK3ZZJvDqleyHS088M --function-call fc-01KRGJPR52EN5XH3RYZ3TBEHXW --timestamps --show-function-call-id --show-container-id --tail 80
modal app logs ap-2mvquK3ZZJvDqleyHS088M --function-call fc-01KRGJPR1GWNZGCQGNA6DA75BH --timestamps --show-function-call-id --show-container-id --tail 120
```

Live Modal volume checks, representative form:

```bash
RUN=curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021
ATT=try-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021
BASE=training/lightzero-curvytron-visual-survival/$RUN
modal volume get curvyzero-runs $BASE/attempts/$ATT/train/progress_latest.json - | jq '{iteration, learner_train_iter, checkpoint_name, checkpoint_ref, source, timestamp, elapsed_sec}'
modal volume ls curvyzero-runs $BASE/attempts/$ATT/train/lightzero_exp/ckpt --json
modal volume ls curvyzero-runs $BASE/checkpoints/lightzero --json
modal volume get curvyzero-runs $BASE/attempts/$ATT/train/status_heartbeat.json - | jq '{stage,status,heartbeat_at,started_at,modal_task_id,auto_resume:.command.auto_resume}'
modal volume get curvyzero-runs $BASE/attempts/$ATT/train/checkpoint_eval_poller.json - | jq '{status,started_at,heartbeat_at,last_scan_count,seen_count,scheduled_count,completed_count,gif_scheduled_count,gif_completed_count,outstanding_count,train_done,stable_polls}'
```

The same live volume pattern was run for:

- `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051`
- `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011`
- healthy comparator `curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021`

## Snapshot Findings

### Stale row: mix2 scrub recent/blank, rb

Run:
`curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021`

`status_chunks_20260513e/combined_status.json` reported:

- `train_status`: `running`
- `train_stage`: `auto_resume_checked`
- `latest_checkpoint`: `iteration_0`
- `checkpoint_count`: 1
- `latest_checkpoint_mtime`: 1778673350.2821896, corresponding to 2026-05-13 07:55 EDT
- `progress_exists`: true
- one live eval and one GIF, both for `live_checkpoint_iteration_0`
- eval/GIF action summary was not collapsed: top action fraction about 0.60

`status_chunks_20260513f/raw_status_files/.../progress_latest.json` showed a
fresh file relative to the snapshot:

- `timestamp`: `2026-05-13T19:40:11.147121Z`
- `learner_train_iter`: 33998
- `iteration`: 0
- `checkpoint_name`: `iteration_0.pth.tar`
- `source`: `SaveCkptHook.__call__`

Live Modal volume check later showed the same pattern had advanced:

- `timestamp`: `2026-05-13T19:48:41.130933Z`
- `learner_train_iter`: 36415
- `iteration`: 0
- `checkpoint_name`: `iteration_0.pth.tar`

Checkpoint directory contents remained only:

- `iteration_0.pth.tar`, modified `2026-05-13 07:55 EDT`, size `91.7 MiB`
- `ckpt_best.pth.tar`, modified `2026-05-13 07:55 EDT`, size `61.2 MiB`

The canonical mirror path `.../checkpoints/lightzero` returned `No such file or
directory` in the live check.

`status_heartbeat.json` was not fresh. It had:

- `status`: `running`
- `stage`: `auto_resume_checked`
- `started_at`: `2026-05-13T17:50:16.388769Z`
- `heartbeat_at`: `2026-05-13T17:50:26.236607Z`
- `modal_task_id`: `ta-01KRH7BDHA2MAK9WNZWBTTCD0N`
- auto-resume from `iteration_0.pth.tar` and matching `iteration_0.resume_state.pkl`

`checkpoint_eval_poller.json` was older still:

- `started_at`: `2026-05-13T11:49:39.506702Z`
- `heartbeat_at`: `2026-05-13T11:49:39.506729Z`
- `status`: `running`
- `last_scan_count`: 0
- `seen_count`: 0
- no scheduled/completed eval or GIF work

Exact Modal train logs for function call
`fc-01KRGJQ8XDR9X2YWXG9QGTV1VA` only showed repeated container starts/warnings:

```text
2026-05-13 07:53:15-04:00 fc-01KRGJQ8XDR9X2YWXG9QGTV1VA ta-01KRGJXS8MHDAS037PE8BHQB32 [05-13 11:53:14] WARNING  not found transformer, please install it using: pip install transformers
2026-05-13 08:14:23-04:00 fc-01KRGJQ8XDR9X2YWXG9QGTV1VA ta-01KRGM4GGAQ6GKY6NQS1BFP0X3 [05-13 12:14:23] WARNING  not found transformer, please install it using: pip install transformers
2026-05-13 13:19:09-04:00 fc-01KRGJQ8XDR9X2YWXG9QGTV1VA ta-01KRH5J9KKH36X25SYWMPJZM1Z [05-13 17:19:09] WARNING  not found transformer, please install it using: pip install transformers
2026-05-13 13:50:16-04:00 fc-01KRGJQ8XDR9X2YWXG9QGTV1VA ta-01KRH7BDHA2MAK9WNZWBTTCD0N [05-13 17:50:16] WARNING  not found transformer, please install it using: pip install transformers
```

Interpretation: the row is very likely alive at the training-process level
because `progress_latest.learner_train_iter` increased from 33998 to 36415
between checks. It is not healthy as an artifact-producing row: checkpoint,
eval/GIF, status heartbeat, and poller artifacts are stale or stuck around
`iteration_0`.

### Stale row: mix3 current/recent/blank/mid/scr, rf

Run:
`curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051`

`status_chunks_20260513e/combined_status.json` reported:

- `train_status`: `running`
- `train_stage`: `auto_resume_checked`
- `latest_checkpoint`: `iteration_0`
- `checkpoint_count`: 1
- `latest_checkpoint_mtime`: 1778684761.20305, corresponding to 2026-05-13 11:06 EDT
- one eval and one GIF, both for `live_checkpoint_iteration_0`
- eval/GIF action collapsed to action `1`

`status_chunks_20260513f/raw_status_files/.../progress_latest.json` showed:

- `timestamp`: `2026-05-13T19:43:26.614776Z`
- `learner_train_iter`: 65588
- `iteration`: 0
- `checkpoint_name`: `iteration_0.pth.tar`
- `source`: `SaveCkptHook.__call__`

Live Modal volume check later showed:

- `timestamp`: `2026-05-13T19:51:34.129990Z`
- `learner_train_iter`: 69261
- `iteration`: 0
- `checkpoint_name`: `iteration_0.pth.tar`

Checkpoint directory contents remained only:

- `iteration_0.pth.tar`, modified `2026-05-13 11:06 EDT`, size `91.7 MiB`
- `ckpt_best.pth.tar`, modified `2026-05-13 11:05 EDT`, size `61.2 MiB`

`status_heartbeat.json` was stale relative to the fresh progress file:

- `status`: `running`
- `stage`: `auto_resume_checked`
- `started_at`: `2026-05-13T17:22:09.410476Z`
- `heartbeat_at`: `2026-05-13T17:24:27.763357Z`
- `modal_task_id`: `ta-01KRH5R16841D0BDF5RXQDBC4T`

`checkpoint_eval_poller.json` looked freshly restarted but not useful:

- `started_at`: `2026-05-13T19:24:22.782849Z`
- `heartbeat_at`: `2026-05-13T19:24:22.782878Z`
- `status`: `running`
- `last_scan_count`: 0
- `seen_count`: 0
- no scheduled/completed eval or GIF work

Exact Modal train logs for function call
`fc-01KRGRHAZ3ZVRBVM5YNJN1RNNN`:

```text
2026-05-13 11:05:33-04:00 fc-01KRGRHAZ3ZVRBVM5YNJN1RNNN ta-01KRGXXXG8SNB29JVKQZWG84XV [05-13 15:05:33] WARNING  not found transformer, please install it using: pip install transformers
2026-05-13 11:31:24-04:00 fc-01KRGRHAZ3ZVRBVM5YNJN1RNNN ta-01KRGZCWJNA2E43JR7HYYQSV5X [05-13 15:31:24] WARNING  not found transformer, please install it using: pip install transformers
2026-05-13 13:22:10-04:00 fc-01KRGRHAZ3ZVRBVM5YNJN1RNNN ta-01KRH5R16841D0BDF5RXQDBC4T [05-13 17:22:10] WARNING  not found transformer, please install it using: pip install transformers
```

The poller function call `fc-01KRGRHASW04PTAHP7AG86HQBM` returned no retained
log lines with `--tail 120`.

Interpretation: same pattern as the named mix2 stale row. The learner appears
to be advancing, but the artifact surface that status consumers care about is
stale. The poller is especially misleading because it can say `running` while
having scanned zero checkpoints.

## Healthy Comparator

Run:
`curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021`

`status_chunks_20260513e/combined_status.json` reported:

- `iteration`: 190000
- `checkpoint_count`: 20
- `latest_checkpoint`: `iteration_190000`
- `background_poller_seen_count`: 19
- `eval_manifest_count`: 13
- `gif_artifact_count`: 19

Live Modal volume check showed the row continuing normally:

- `timestamp`: `2026-05-13T19:54:47.805276Z`
- `learner_train_iter`: 210603
- `iteration`: 210000
- `checkpoint_name`: `iteration_210000.pth.tar`
- `source`: `SaveCkptHook.__call__`

Checkpoint directory had a regular sequence:

```text
iteration_210000.pth.tar  modified 2026-05-13 15:53 EDT
iteration_200000.pth.tar  modified 2026-05-13 15:28 EDT
iteration_190000.pth.tar  modified 2026-05-13 15:04 EDT
iteration_180000.pth.tar  modified 2026-05-13 14:39 EDT
iteration_170000.pth.tar  modified 2026-05-13 14:14 EDT
...
iteration_30000.pth.tar   modified 2026-05-13 08:52 EDT
iteration_20000.pth.tar   modified 2026-05-13 08:31 EDT
iteration_10000.pth.tar   modified 2026-05-13 08:12 EDT
iteration_0.pth.tar       modified 2026-05-13 07:51 EDT
ckpt_best.pth.tar         modified 2026-05-13 07:51 EDT
```

Poller status was imperfect but at least saw the checkpoint stream:

- `started_at`: `2026-05-13T19:23:30.361209Z`
- `heartbeat_at`: `2026-05-13T19:24:05.367864Z`
- `last_scan_count`: 20
- `seen_count`: 20
- `scheduled_count`: 20
- `gif_scheduled_count`: 20
- `outstanding_count`: 40

Exact Modal train logs for function call
`fc-01KRGJPR52EN5XH3RYZ3TBEHXW` only showed the standard transformer warning:

```text
2026-05-13 07:49:23-04:00 fc-01KRGJPR52EN5XH3RYZ3TBEHXW ta-01KRGJPSPJ0R9TQWMAENR8KR4M [05-13 11:49:23] WARNING  not found transformer, please install it using: pip install transformers
```

The comparator proves that the same `progress_latest` writer can coexist with a
healthy iteration checkpoint sequence. Therefore a fresh `progress_latest` file
alone is not proof that the artifact pipeline is healthy.

## Cross-Signal Readout

| Signal | Stale sampled rows | Healthy comparator | Reliability |
| --- | --- | --- | --- |
| `progress_latest.timestamp` | Fresh at 19:48-19:51Z | Fresh at 19:54Z | Useful as weak process-liveness signal |
| `progress_latest.learner_train_iter` | Increasing, e.g. 33998 to 36415 and 65588 to 69261 | Increasing to 210603 | Useful as learner-loop progress signal |
| `progress_latest.iteration` / `checkpoint_name` | Stuck at `iteration_0` | Tracks `iteration_210000` | Good artifact-health signal |
| `lightzero_exp/ckpt` files | Only `iteration_0` and `ckpt_best`, both old | Regular `iteration_*.pth.tar` sequence | Strong artifact-health signal |
| status heartbeat | `running`, but stale by hours | Not checked deeply in this pass | Misleading unless freshness is evaluated |
| poller heartbeat | `running`, but often stale or zero-scan | Sees many checkpoints | Misleading unless `heartbeat_at`, `seen_count`, and `last_scan_count` are considered |
| eval/GIF presence | Present only for `iteration_0` | Present across many checkpoints | Good downstream artifact-health signal |
| Modal app logs | Mostly only startup warnings; no retained error lines for sampled calls | Same | Weak for this question unless paired with volume state |

## Conclusion

The named rows are not simply "alive but between checkpoint intervals." They
are alive enough to rewrite progress metadata and advance `learner_train_iter`,
but they are not producing the expected `iteration_10000`, `iteration_20000`,
... checkpoint files even after learner iterations far beyond the configured
`save_ckpt_after_iter=10000`.

The current status signals are therefore misleading if read as training health:

- `train_status=running` can persist with stale status heartbeats.
- `checkpoint_eval_poller.status=running` can persist with zero scans or old
  heartbeats.
- `progress_latest.timestamp` can be fresh even while
  `progress_latest.checkpoint_name` points at an old checkpoint.

For run-health triage, treat the strongest liveness/artifact-health proof as:

1. Fresh `progress_latest.timestamp` plus increasing `learner_train_iter` means
   the learner process is probably still executing.
2. New `iteration_*.pth.tar` files, advancing `progress_latest.iteration`, and
   matching live eval/GIF artifacts mean the row is artifact-healthy.
3. Fresh progress with a stuck checkpoint sequence means "alive but not
   checkpointing"; do not rank it as healthy.
4. Heartbeat fields should always be paired with freshness thresholds, because
   `status=running` by itself is stale-prone.

## Follow-Up Checks

- Add a liveness classifier that separates `process_alive`,
  `checkpoint_stream_alive`, `eval_stream_alive`, and `poller_alive`.
- In status rows, expose heartbeat age for both train and poller artifacts.
- Flag rows where `learner_train_iter - progress_latest.iteration >= 2 *
  save_ckpt_after_iter`.
- Investigate why the DI-engine checkpoint hook returns often enough to rewrite
  `progress_latest.json` but does not emit interval checkpoint files for these
  rows.
