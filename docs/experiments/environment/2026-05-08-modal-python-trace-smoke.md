# 2026-05-08 Modal Python Trace Smoke

## Question

Can a small Modal Function run the existing Python toy-v0 trace helper, write a
JSON trace/fingerprint artifact to the shared `curvyzero-runs` Volume, commit
it, and return the exact immutable run/attempt ref?

## Setup

- Modal app: `curvyzero-env-fidelity-smoke`.
- Entrypoint: `curvyzero.infra.modal.fidelity_smoke`.
- Volume: `curvyzero-runs`.
- Mounted path in the container: `/runs`.
- Artifact layout:

```text
experiments/<run_id>/attempts/<attempt_id>/fidelity-smoke/trace_fingerprint.json
```

This is Python-only. It calls `curvyzero.env.tracing.trace_scripted_actions`
once with a deterministic toy action script. It does not call the JS oracle.

## Commands

### Local Checks

```sh
uv run python -m py_compile src/curvyzero/infra/modal/fidelity_smoke.py
uv run --extra dev ruff check src/curvyzero/infra/modal/fidelity_smoke.py
```

### Remote Python Trace Smoke

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.fidelity_smoke --run-id env-fidelity-smoke-20260508-python-trace
```

## Results

Local checks:

```text
py_compile: passed
ruff: All checks passed!
```

Remote Modal run: `ap-muerWyWt71VmTbAhIc5DLL`.

Returned metadata:

```json
{
  "app_name": "curvyzero-env-fidelity-smoke",
  "artifact_path": "/runs/experiments/env-fidelity-smoke-20260508-python-trace/attempts/attempt-20260508T181410Z-486f8f56551c/fidelity-smoke/trace_fingerprint.json",
  "artifact_ref": "experiments/env-fidelity-smoke-20260508-python-trace/attempts/attempt-20260508T181410Z-486f8f56551c/fidelity-smoke/trace_fingerprint.json",
  "attempt_id": "attempt-20260508T181410Z-486f8f56551c",
  "bytes": 4584,
  "client_elapsed_ms": 3302.533,
  "committed": true,
  "remote_elapsed_ms": 1012.759,
  "run_id": "env-fidelity-smoke-20260508-python-trace",
  "schema": "curvyzero_modal_python_trace_smoke_result/v1",
  "sha256": "274e5dfe098cdc91e74e7ca8d16b0da005af9a4e12473308cdc9cafdfb650460",
  "trace_fingerprint": "517d035de684424ccda362eb92846d01c87ad6c5c137e748e6dedce54c08a6e9",
  "volume_mount": "/runs",
  "volume_name": "curvyzero-runs"
}
```

## Interpretation

The Python-only Modal trace smoke works end to end. The Function copied the
local `src` tree into the image, installed `numpy`, ran one deterministic
toy-v0 trace, wrote the JSON artifact with exclusive file creation, committed
the Volume, and returned a stable `artifact_ref`.

This is not a full environment fidelity proof. It does not compare against the
JS oracle yet.

## Artifacts

- Volume: `curvyzero-runs`.
- Artifact ref:
  `experiments/env-fidelity-smoke-20260508-python-trace/attempts/attempt-20260508T181410Z-486f8f56551c/fidelity-smoke/trace_fingerprint.json`.

## Blockers

None for this smoke.
