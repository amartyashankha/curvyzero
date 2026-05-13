# Modal Log Timeline - Current CurvyTron Training App - 2026-05-13

Scope: read-only investigation of the deployed Modal app
`curvyzero-lightzero-curvytron-visual-survival-train`, focusing on crashes,
`DataLossError`, checkpoint corruption, preemption, and missing-checkpoint
symptoms. Source code was not edited. The only file written for this pass is
this note.

Current app metadata observed at `2026-05-13T20:05:28Z`:

- App ID: `ap-2mvquK3ZZJvDqleyHS088M`
- State: `deployed`
- Tasks: `460`
- Created at: `2026-05-13 05:35:08-04:00`

## Existing Inputs Read

- `run_health_check_2026-05-13.md`
- `log_investigation_newton_2026-05-13.md`
- `modal_log_access_peirce_2026-05-13.md`
- `artifact_liveness_audit_agent_2026-05-13.md`
- Focused context from `delegation_log.md`, local pruning/status artifacts, and
  grouped launch manifests.

## Targeted Commands Run

Representative read-only commands:

```bash
uv run --extra modal modal app list --json
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M --search DataLossError --since 16h --timestamps --show-function-call-id --show-container-id --tail 80
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M --search preempt --since 16h --timestamps --show-function-call-id --show-container-id --tail 80
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M --search PytorchStreamReader --since 16h --timestamps --show-function-call-id --show-container-id --tail 80
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M --search Traceback --since 16h --timestamps --show-function-call-id --show-container-id --tail 80
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M --search KeyboardInterrupt --since 16h --timestamps --show-function-call-id --show-container-id --tail 30
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M --search "No such file" --since 16h --timestamps --show-function-call-id --show-container-id --tail 30
uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M --search matching_iteration_checkpoint_not_found --since 16h --timestamps --show-function-call-id --show-container-id --tail 30
uv run --extra modal modal volume get curvyzero-runs <progress_latest.json> -
uv run --extra modal modal volume ls curvyzero-runs <train/lightzero_exp/ckpt>
```

The `"No such file"` and `matching_iteration_checkpoint_not_found` app-log
searches returned no retained matching lines in this pass. That is a fact about
the searched log surface, not proof those conditions never occurred internally.

## Timeline

| Time EDT | Signal | Fact |
| --- | --- | --- |
| `05:35` | App deployment | App `ap-2mvquK3ZZJvDqleyHS088M` was created/deployed for the current trainer app. |
| `07:14` | Checkpoint corruption | A checkpoint selfplay/GIF worker failed loading `iteration_30000.pth.tar` for `curvy-survive-bonus-blank-browser-light-base-r074-s1111171` with `PytorchStreamReader failed reading zip archive: failed finding central directory`. |
| `07:49` | Checkpoint corruption | A second selfplay/GIF worker failed loading `iteration_60000.pth.tar` for `curvy-survive-bonus-blank-fast-medium-base-r177-s1120291` with the same PyTorch zip/central-directory error. |
| `08:39` | Checkpoint corruption | `curvy-mix2clean-old100-rb-s8-c32-l32-rep0-k10-c1-s2111011` failed loading `iteration_0.pth.tar` in a selfplay/GIF worker with the same corruption signature. |
| `08:07-13:20` | Preemption on sampled stale rows | Prior targeted checks found multiple stale-row containers preempted and restarted, including stale mix2 and mix3 examples. |
| `10:06-10:19` | Traceback burst | App-level `Traceback` search returned many tracebacks across different function/container IDs. The visible search output alone did not tie those tracebacks to the sampled stale train calls. |
| `10:52` | Checkpoint corruption | `curvy-survive-bonus-blank-fast-medium-batch64-r245-s1140631` failed loading `iteration_120000.pth.tar` with the same PyTorch zip/central-directory error. |
| `11:04` | Checkpoint corruption repeats | Another worker for the same batch64 run/checkpoint failed loading `iteration_120000.pth.tar` with the same error. |
| `11:04` | Runtime interruption | App logs showed `KeyboardInterrupt` inside DI-engine `subprocess_env_manager.py` during env reset for function call `fc-01KRGRFCZAMJ3S8HS2AA26H781`. This is an interruption/crash symptom; I did not prove it is the cause of missing checkpoints. |
| `15:24-16:00` | Broad preemption burst | App logs show many containers interrupted due to worker preemption. Most lines say Modal will restart the same input; a few later lines omit the restart phrase. |
| `15:28-16:01` | `DataLossError` commit storm | App logs show dense `modal_volume_commit_retry` events with `error_type=DataLossError`, labels mostly `checkpoint_eval_and_inspect` and `checkpoint_selfplay_gif`, and error `failed to publish commit to server`. |
| `20:04-20:05` | Missing checkpoint symptoms still live | Fresh volume reads show sampled stale rows still rewriting `progress_latest.json` while their checkpoint directories remain stuck at `iteration_0.pth.tar` plus `ckpt_best.pth.tar`. |

## Current Missing-Checkpoint Samples

These are current reads from the Modal volume, not just old status snapshots.

| Run | Fresh progress timestamp | `learner_train_iter` | Advertised checkpoint | Visible checkpoint files |
| --- | --- | ---: | --- | --- |
| `curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011` | `2026-05-13T20:04:12Z` | `184538` | `iteration_0.pth.tar` | only `iteration_0.pth.tar`, `ckpt_best.pth.tar` |
| `curvy-mix2clean-r50-scr50-rb-s8-c32-l32-rep0-k10-c2-s2104021` | `2026-05-13T20:04:00Z` | `40621` | `iteration_0.pth.tar` | only `iteration_0.pth.tar`, `ckpt_best.pth.tar` |
| `curvy-mix3cur-r40-blank20-mid20-scr20-rf-s8-c32-l32-rep0-k10-c5-s2306051` | `2026-05-13T20:05:37Z` | `75672` | `iteration_0.pth.tar` | only `iteration_0.pth.tar`, `ckpt_best.pth.tar` |
| Healthy comparator `curvy-mix2clean-r50-blank50-rf-s8-c32-l32-rep0-k10-c2-s2101021` | `2026-05-13T19:54:47Z` | `210603` | `iteration_210000.pth.tar` | visible late files include `iteration_180000`, `190000`, `200000`, `210000` |

Interpretation: the stale rows are not merely between checkpoint intervals.
They have crossed many expected `save_ckpt_after_iter=10000` boundaries while
the durable `iteration_*.pth.tar` stream remains stuck.

## Strongest Findings

1. Fact: The dominant missing-checkpoint signature is fresh learner progress
   with stale durable checkpoints: `progress_latest.timestamp` refreshes and
   `learner_train_iter` rises, but `progress_latest.iteration` and
   `train/lightzero_exp/ckpt/iteration_*.pth.tar` do not advance.
2. Fact: The issue is not global. A healthy comparator in the same deployed app
   reaches `iteration_210000` and has the expected checkpoint file sequence.
3. Fact: App-level logs contain real checkpoint corruption symptoms. Multiple
   eval/GIF workers hit PyTorch zip central-directory failures while loading
   checkpoint files.
4. Fact: App-level logs contain a broad Modal preemption burst, and prior
   targeted checks tied preemptions to some stale-row containers.
5. Fact: Preemption is not sufficient as a root cause. Prior checks found at
   least one healthy survival comparator was also preempted and recovered.
6. Fact: `DataLossError` appears as volume commit retry noise around eval/GIF
   jobs, especially `checkpoint_eval_and_inspect` and `checkpoint_selfplay_gif`,
   not as a direct sampled train-call checkpoint-save traceback.
7. Fact: Per-train-call logs for sampled stale rows remain sparse, usually only
   the transformer import warning. I did not find direct retained Python
   tracebacks, CUDA errors, or explicit train-call timeout lines for those train
   FunctionCall IDs.
8. Hypothesis: preemption plus high concurrent eval/GIF volume commits can
   amplify checkpoint and poller inconsistency, but it does not by itself
   explain rows that keep training for tens of thousands of learner iterations
   without emitting later `iteration_*.pth.tar`.
9. Hypothesis: the checkpoint corruption errors are likely downstream workers
   reading incomplete or otherwise invalid checkpoint files, but the log search
   does not prove whether the bad file came from interrupted save, volume
   consistency, concurrent read-before-complete, or another writer-path issue.
10. Hypothesis: the progress writer currently overstates checkpoint health:
    `SaveCkptHook.__call__` can refresh `progress_latest.json` even when the
    latest visible `iteration_*.pth.tar` has not changed. This explains the
    misleading liveness surface, but not the original reason later checkpoint
    files are missing.

## What To Trust

Trust, in order:

1. Actual `iteration_*.pth.tar` files in `train/lightzero_exp/ckpt` or the
   canonical `checkpoints/lightzero` mirror.
2. `progress_latest.iteration` and `checkpoint_name`.
3. Eval/GIF artifacts for the same checkpoint label.
4. `progress_latest.learner_train_iter` as process-liveness only.
5. `train_status=running` and poller `status=running` only when paired with
   fresh heartbeat age and nonzero scan progress.

The current evidence supports classifying the sampled stale rows as
`process_alive_but_checkpoint_stream_dead`, not as healthy running jobs.
