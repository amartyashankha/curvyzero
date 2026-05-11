# Modal Background Spawn Probe

## Question

Can a local Modal entrypoint launch a cheap background job with
`Function.spawn`, return a function call id, and still have the remote job write
progress and done markers to the `curvyzero-runs` Volume after the entrypoint
returns?

## Code Path

Probe module:

`src/curvyzero/infra/modal/background_spawn_probe.py`

The local entrypoint calls:

```python
background_sleep_and_mark.spawn(...)
```

The remote function writes these refs:

```text
training/modal-background-spawn-probe/<run_id>/attempts/<attempt_id>/probe_progress/latest.json
training/modal-background-spawn-probe/<run_id>/attempts/<attempt_id>/probe_progress/progress_000.json
training/modal-background-spawn-probe/<run_id>/attempts/<attempt_id>/probe_progress/progress_001.json
training/modal-background-spawn-probe/<run_id>/attempts/<attempt_id>/probe_progress/progress_002.json
training/modal-background-spawn-probe/<run_id>/attempts/<attempt_id>/probe_progress/progress_003.json
training/modal-background-spawn-probe/<run_id>/attempts/<attempt_id>/done.json
training/modal-background-spawn-probe/<run_id>/attempts/<attempt_id>/summary.json
```

It commits the Volume after the start marker, each progress marker, and the
final done/summary markers.

## Run 1: Spawn Without Detach

Command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.background_spawn_probe \
  --run-id bg-spawn-probe-20260510-toy \
  --attempt-id sleep6-commit2-a \
  --steps 3 \
  --sleep-sec 2
```

Launch result:

```text
function_call_id: fc-01KR7NY1EP961CN5Z7BEP6KQN7
root_ref: training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-a
```

The local entrypoint returned and Modal printed:

```text
Stopping app - local entrypoint completed.
```

Verification command:

```bash
uv run --extra modal modal volume ls curvyzero-runs \
  training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-a
```

Result after waiting longer than the requested 6 seconds:

```text
No such file or directory
```

App logs only showed image build and app stop. No Volume directory appeared for
this attempt.

## Run 2: Spawn With Detach

Command:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.background_spawn_probe \
  --run-id bg-spawn-probe-20260510-toy \
  --attempt-id sleep6-commit2-detached-a \
  --steps 3 \
  --sleep-sec 2
```

Modal printed this detach note:

```text
running a local entrypoint in detached mode only keeps the last triggered Modal function alive
```

Launch result:

```text
function_call_id: fc-01KR7NZMEXMNEQ64SDS7ZQABCY
root_ref: training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-detached-a
```

Verification command:

```bash
uv run --extra modal modal volume ls curvyzero-runs \
  training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-detached-a
```

Result:

```text
training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-detached-a/probe_progress
training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-detached-a/summary.json
training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-detached-a/done.json
```

Progress verification:

```bash
uv run --extra modal modal volume ls curvyzero-runs \
  training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-detached-a/probe_progress
```

Result:

```text
progress_000.json
progress_001.json
progress_002.json
progress_003.json
latest.json
```

Done marker fetch:

```bash
uv run --extra modal modal volume get curvyzero-runs \
  training/modal-background-spawn-probe/bg-spawn-probe-20260510-toy/attempts/sleep6-commit2-detached-a/done.json -
```

Key fields:

```json
{
  "ok": true,
  "phase": "completed",
  "steps": 3,
  "sleep_sec": 2.0,
  "modal_task_id": "ta-01KR7NZMSEY6GHKKKK8ACE6FJ7",
  "started_at": "2026-05-10T00:53:21.077739Z",
  "ended_at": "2026-05-10T00:53:33.061858Z"
}
```

Latest progress marker had `step: 3` and `phase: running`. The done marker had
`phase: completed`.

## Claim

For this toy worker, the working pattern is:

```text
modal run --detach ... local_entrypoint calls Function.spawn(...)
```

That pattern returned a `function_call_id` from the local entrypoint and still
produced progress plus done markers in the `curvyzero-runs` Volume after the
entrypoint returned.

The local entrypoint should print refs and return. Monitoring should happen via
Volume refs, not by waiting for the local `modal run` process to return the
remote function result.

## Non-claim

`Function.spawn` by itself was not enough in the plain `modal run` test above.
The non-detached run returned a call id but produced no Volume markers.

This does not prove long GPU LightZero training will complete. It only validates
the cheap background launch and Volume marker pattern.

This does not prove policy quality, checkpoint cadence, or LightZero learner
health.

No pytest was run.

## Implication For LightZero Training

Launch long LightZero training with both pieces:

```bash
uv run --extra modal modal run --detach -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-l4-t4 \
  ...
```

The wrapper should keep using `Function.spawn` for `mode=train`.

After launch, treat the printed `function_call_id` and Volume refs as the
control surface. Poll progress JSON and checkpoints in `curvyzero-runs`; do not
depend on the local CLI session for the training result.
