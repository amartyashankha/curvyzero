# 2026-05-08 Modal Artifact Smoke

## Question

Can a tiny Modal Function write a JSON artifact to the shared `curvyzero-runs`
Volume under an immutable run/attempt path, commit it, and return enough metadata
for a local caller to fetch the exact artifact later?

## Setup

- Modal app: `curvyzero-artifact-smoke`.
- Entrypoint: `curvyzero.infra.modal.artifact_smoke`.
- Volume: `curvyzero-runs`.
- Mounted path in the container: `/runs`.
- Artifact layout:

```text
experiments/<run_id>/attempts/<attempt_id>/artifact-smoke/artifact.json
```

The smoke is intentionally generic. It is meant to become the durable artifact
pattern for future headless probes: environment fidelity checks, package/device
smokes, benchmark manifests, profiler summaries, and later training/evaluation
summaries. It does not implement the CurvyTron oracle or any browser hosting.

## Commands

### Local Checks

```sh
uv run --extra modal python -m py_compile src/curvyzero/infra/modal/artifact_smoke.py
uv run --extra modal --extra dev ruff check src/curvyzero/infra/modal/artifact_smoke.py
```

### Remote Artifact Smoke

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.artifact_smoke --run-id modal-artifact-smoke-20260508
```

### Exact Artifact Fetch

```sh
uv run --extra modal modal volume get curvyzero-runs \
  experiments/modal-artifact-smoke-20260508/attempts/attempt-20260508T174948Z-063ca38f0317/artifact-smoke/artifact.json \
  /private/tmp/curvyzero-artifact-smoke.json \
  --force
```

## Results

Local checks:

```text
py_compile: passed
ruff: All checks passed!
```

Remote Modal run: `ap-ippEnXYKtxMBhr7lC1zaQZ`.

Returned metadata:

```json
{
  "app_name": "curvyzero-artifact-smoke",
  "artifact_path": "/runs/experiments/modal-artifact-smoke-20260508/attempts/attempt-20260508T174948Z-063ca38f0317/artifact-smoke/artifact.json",
  "artifact_ref": "experiments/modal-artifact-smoke-20260508/attempts/attempt-20260508T174948Z-063ca38f0317/artifact-smoke/artifact.json",
  "attempt_id": "attempt-20260508T174948Z-063ca38f0317",
  "bytes": 485,
  "client_elapsed_ms": 2818.862,
  "committed": true,
  "remote_elapsed_ms": 601.358,
  "run_id": "modal-artifact-smoke-20260508",
  "schema": "curvyzero_modal_artifact_smoke_result/v1",
  "sha256": "43dbd2d38af738396d65a0e7a29e0ed141980a10b5db85ceaaed06c14a68220c",
  "volume_mount": "/runs",
  "volume_name": "curvyzero-runs"
}
```

Fetched artifact:

```json
{
  "app_name": "curvyzero-artifact-smoke",
  "attempt_id": "attempt-20260508T174948Z-063ca38f0317",
  "created_at": "2026-05-08T17:49:48.369350Z",
  "environment": {
    "modal_task_id": "ta-01KR4BBD6J956DHSR60PAJ0FS2",
    "platform": "Linux-4.4.0-x86_64-with-glibc2.36",
    "python": "3.11.12"
  },
  "message": "generic Modal Volume artifact smoke",
  "run_id": "modal-artifact-smoke-20260508",
  "schema": "curvyzero_modal_artifact_smoke/v1",
  "volume_name": "curvyzero-runs"
}
```

## Interpretation

The artifact smoke works end to end. A Modal Function can create a unique
attempt path, write a JSON artifact with exclusive file creation, commit the
Volume, and return a stable `artifact_ref` that can be fetched exactly.

This is only an artifact contract smoke. It is not a simulator fidelity test,
oracle job, training job, or hosted demo.

## Artifacts

- Volume: `curvyzero-runs`.
- Artifact ref:
  `experiments/modal-artifact-smoke-20260508/attempts/attempt-20260508T174948Z-063ca38f0317/artifact-smoke/artifact.json`.
- Local verification copy: `/private/tmp/curvyzero-artifact-smoke.json`.

## Follow-ups

- Reuse the same immutable path and returned-manifest shape for future headless
  fidelity probes.
- Add richer probe payloads only after the probe itself is defined.
- Keep full browser/demo hosting deferred.
