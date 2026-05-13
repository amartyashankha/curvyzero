# Modal Runtime Failure Classes - 2026-05-13

Scope: read-only investigation of Modal logs and local docs/source references.
No source code was edited. The target app was
`ap-2mvquK3ZZJvDqleyHS088M`
(`curvyzero-lightzero-curvytron-visual-survival-train`), with a small check of
the deployed checkpoint tournament app `ap-TozUPLAp5EAgC3GeA3i2AW`.

## Short Read

The timestamped checkpoint directory bug remains the clean explanation for many
rows that appear stuck at `iteration_0`: LightZero/DI-engine can write to
`train/lightzero_exp_YYMMDD_HHMMSS/ckpt` after restart while CurvyZero status,
poller, mirror, and some callers still look at `train/lightzero_exp/ckpt`.

That is separate from the other failures in the logs:

- `Runner interrupted due to worker preemption`: real worker interruptions,
  often with Modal saying the function will restart with the same input.
- DI-engine `KeyboardInterrupt` during env reset: real training-hot-path
  interruptions, seen inside subprocess env reset/render code.
- `DataLossError: failed to publish commit to server`: volume commit failures,
  concentrated in checkpoint eval/inspection and GIF artifact workers.
- `PytorchStreamReader failed reading zip archive`: checkpoint load failures in
  GIF/eval workers after a checkpoint path was found.
- `matching_iteration_checkpoint_not_found`: not observed in the last 6h sample.

So the current split is: timestamped paths explain bad checkpoint discovery and
stale status; `DataLossError` and `PytorchStreamReader` mostly explain missing
or failed eval/GIF artifacts; preemption and env reset interrupts are the real
training-hot-path failures found in this pass.

## Error Classes

| Class | Example evidence | Affects | Training crash evidence? | Readout |
| --- | --- | --- | --- | --- |
| Timestamped `lightzero_exp` dirs | Existing docs show stale fixed path `train/lightzero_exp/ckpt/iteration_0.pth.tar` while timestamped siblings have later checkpoints, e.g. `lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar`. | Status, checkpoint poller, eval/GIF discovery, resume, any fixed checkpoint refs; tournaments only if callers pass fixed refs. | Not by itself. Training can continue while readers look in the wrong directory. | Main cause of many stale `iteration_0` rows. |
| Worker preemption | Last 6h rough count from main thread: about 362 `Runner interrupted` lines. Sample lines around `2026-05-13 15:26-16:26 EDT` often say the function will restart with the same input. | Training, eval/GIF, pollers, tournaments depending on which function was interrupted. | Yes, as interruption/restart pressure. Not proof of permanent failure. | Important cofactor. It can trigger restarts, and restarts can create timestamped `lightzero_exp_*` dirs. |
| DI-engine env reset `KeyboardInterrupt` | `2026-05-13 11:04:31 EDT`, function `fc-01KRGRFCZAMJ3S8HS2AA26H781`: `Env 19 reset has exceeded max retries(1)` after `KeyboardInterrupt` in `curvyzero_source_state_visual_survival_lightzero_env.py:725`, `vector_multiplayer_env.py`, and `vector_source_random.py`. Another at `2026-05-13 16:13:14 EDT`, `fc-01KRGJQGTKVRNT8H4W130YVC2Y`, interrupted during `_lightzero_observation` / `render_source_state_canvas_gray64`. | Training hot path: DI-engine subprocess env reset and observation rendering. | Yes. These are real train-function interruptions separate from checkpoint path discovery. | They look like interruption symptoms, likely tied to worker shutdown/preemption, but they still matter because they can kill/restart collection and lose in-memory work since the last good checkpoint. |
| Modal volume `DataLossError` commit failures | Last 6h rough count from main thread: about 1210 `failed to publish commit` lines. Label count from a 6h log sample: `checkpoint_eval_and_inspect` 633, `checkpoint_selfplay_gif` 582. Source commit wrapper logs `modal_volume_commit_retry` at `lightzero_curvyzero_stacked_debug_visual_survival_train.py:6487` and `lightzero_curvytron_visual_survival_eval.py:116`. | Eval/inspection/GIF artifact publication; website freshness; status of background artifact jobs. | No direct learner crash evidence in the sampled labels. | Real artifact-publishing failure. It can make eval/GIF/status artifacts stale or missing without proving training stopped. |
| PyTorch checkpoint read failure | Last 6h rough count from main thread: about 201 `PytorchStreamReader failed reading zip archive` lines. Examples: `curvy-survive-bonus-blank-fast-medium-batch64-r245-s1140631`, `iteration_120000`, at `2026-05-13 10:52:56` and `11:04:29 EDT`; `curvy-mix3cur-recent100-rb-s8-c32-l32-rep0-k10-c2-s2307021`, `iteration_110000`, at `2026-05-13 15:46:56 EDT`. The GIF worker calls `eval_mod._torch_load(checkpoint_path)` at `lightzero_curvyzero_stacked_debug_visual_survival_train.py:7529`; `_torch_load` calls `torch.load` at `lightzero_curvytron_visual_survival_eval.py:212`. | GIF/eval checkpoint loading. Also a tournament risk if a tournament reads the same bad checkpoint. | Not proof the trainer crashed. It proves a reader found a checkpoint file that PyTorch could not load. | Likely incomplete/corrupt checkpoint visibility or a race with checkpoint publication. Treat as separate from fixed-path discovery. |
| Tournament shutdown tracebacks | Deployed tournament app `ap-TozUPLAp5EAgC3GeA3i2AW` had sampled `Traceback` lines in `threading.py`, line 1590, `_shutdown`, between `13:44` and `15:40 EDT`. | Tournaments only. | No training effect. | Needs separate tournament review if tournament reliability matters; not evidence of training failure. |
| CUDA OOM | Exact search for `CUDA out of memory` returned no lines in this pass. | None observed. | No. | No CUDA OOM evidence found in sampled logs. |

## Training vs Artifact Path

Training-hot-path evidence found:

- Worker preemptions are frequent.
- DI-engine env reset `KeyboardInterrupt` tracebacks are real training-function
  interruptions.
- At least one interrupted train function call later showed a new container,
  consistent with Modal restart behavior.

Artifact/status evidence found:

- `DataLossError` commit retries are heavily concentrated in background
  checkpoint eval/inspect and self-play GIF workers.
- `PytorchStreamReader` failures are emitted by the checkpoint self-play GIF
  path after `_wait_for_visible_checkpoint` returns a checkpoint path.
- These artifact failures can make the website, eval summaries, GIFs, and
  poller status misleading without proving the learner stopped.

Checkpoint-discovery evidence:

- The explicit string `matching_iteration_checkpoint_not_found` had zero lines
  in the last 6h rough count.
- That does not clear the timestamped path bug; it means the observed stale rows
  are not currently being explained by that exact logged sidecar failure string.
  Existing volume/docs evidence still shows actual checkpoints under
  `lightzero_exp_*` while fixed-path readers remain stale.

## Tournament Footgun

`src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` does the right
broad scan when it discovers checkpoints from a run root:

```text
train/lightzero_exp*/ckpt/iteration_*.pth.tar
```

The footgun is outside that broad scan: any caller or manifest that passes a
fixed `train/lightzero_exp/ckpt/iteration_N.pth.tar` ref can miss the real
timestamped checkpoint stream. Tournaments can also fail if they select a
checkpoint file that exists but is not loadable by PyTorch.

## Evidence Of Real Training Crashes

Yes, but limited:

- The DI-engine `KeyboardInterrupt` / `Env reset has exceeded max retries(1)`
  traces are real train-function failures or interruptions.
- Worker preemption is real and frequent.

No evidence found in this pass for:

- CUDA out-of-memory.
- A persistent Python exception directly inside the learner update loop.
- `DataLossError` from the learner's own checkpoint-save path. The sampled
  `DataLossError` labels were artifact workers, not the learner.

Best current wording: training is being interrupted and restarted, while a large
part of the visible badness is artifact/checkpoint-reader fallout. The strongest
single stale-status bug is still timestamped checkpoint directory discovery.

## Commands Used

```bash
uv run --extra modal modal app list

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --since 6h --search "failed to publish commit" \
  --timestamps --show-function-call-id --show-container-id --tail 2000 \
  | grep -o '"label": "[^"]*"' | sort | uniq -c

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --since 6h --search "PytorchStreamReader failed reading zip archive" \
  --timestamps --show-function-call-id --show-container-id --tail 80

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --since 6h --search "Runner interrupted" \
  --timestamps --show-function-call-id --show-container-id --tail 80

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --since 6h --search "matching_iteration_checkpoint_not_found" \
  --timestamps --show-function-call-id --show-container-id --tail 40

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --since 6h --search "KeyboardInterrupt" \
  --timestamps --show-function-call-id --show-container-id --tail 120

uv run --extra modal modal app logs ap-2mvquK3ZZJvDqleyHS088M \
  --since 6h --search "CUDA out of memory" \
  --timestamps --show-function-call-id --show-container-id --tail 50

uv run --extra modal modal app logs ap-TozUPLAp5EAgC3GeA3i2AW \
  --since 6h --search "Traceback" \
  --timestamps --show-function-call-id --show-container-id --tail 80

rg -n "timestamp|DataLoss|PytorchStreamReader|KeyboardInterrupt|preempt|matching_iteration_checkpoint_not_found|lightzero_exp" \
  docs/working/training/curvytron_architecture_research_2026-05-12 \
  docs/working/training docs/working/optimizer

nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  | sed -n '6470,6535p'

nl -ba src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  | sed -n '7450,7738p'

nl -ba src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py \
  | sed -n '108,155p;208,220p;1500,1538p'

nl -ba src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  | sed -n '245,260p;390,404p'
```
