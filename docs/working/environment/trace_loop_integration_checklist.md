# Trace Loop Integration Checklist

Status: Fast checklist for the workers landing the scenario trace loop.

Target architecture:

```text
scenario JSON -> JS trace -> Python trace -> first-mismatch diff -> Modal artifact
```

Artifact hygiene: follow
`docs/design/environment/artifact_policy.md`. Commit scenarios, code, docs, and
short experiment logs. Keep local traces under `artifacts/local/` or `tmp/`.
Keep complete run payloads in the Modal Volume under `/runs/...`.

## Expected Files Once Code Lands

- `scenarios/fidelity/smoke.json`
- `scenarios/fidelity/canonical.json`
- `tools/reference_oracle/*scenario*.js` or equivalent JS scenario runner
- `src/curvyzero/env/tracing.py` updated to read the same scenario JSON
- `scripts/env_fidelity_diff.py`
- `src/curvyzero/infra/modal/environment_fidelity.py`
- `scripts/modal_run_env_fidelity.py`
- `scripts/modal_fetch_curvyzero_artifact.py`

Modal artifact shape:

```text
/runs/fidelity/<run_id>/
  manifest.json
  scenario_set.json
  batches/batch-000/
    manifest.json
    scenarios/<scenario_id>/
      input.json
      js/trace.jsonl
      js/events.jsonl
      python/trace.jsonl
      python/events.jsonl
      diff/report.json
      diff/first_mismatch.json
      diff/summary.txt
    complete.json
```

## Exact Commands To Expect

Fast local checks:

```sh
uv run --extra dev pytest
node tools/reference_oracle/headless_probe.js
uv run python scripts/benchmark_env.py --episodes 100 --max-steps 500
```

Python-only Modal boundary smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.fidelity_smoke --run-id env-fidelity-smoke-YYYYMMDD-001
```

Stage probes for the new loop:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind js-probe --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind python-probe --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind diff --run-id env-fidelity-YYYYMMDD-001 --batch-id batch-000
```

Preferred end-to-end command:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind batch --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
```

Canonical run after smoke passes:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind batch --scenario-set scenarios/fidelity/canonical.json --run-id env-fidelity-YYYYMMDD-002
```

Deployed path after the app is stable:

```sh
uv run --extra modal modal deploy -m curvyzero.infra.modal.environment_fidelity
uv run python -m scripts.modal_run_env_fidelity --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
uv run python -m scripts.modal_fetch_curvyzero_artifact --run-id env-fidelity-YYYYMMDD-001 --ref fidelity/env-fidelity-YYYYMMDD-001/manifest.json --output tmp/env-fidelity-YYYYMMDD-001-manifest.json
```

Fetch exact failure files only:

```sh
uv run python -m scripts.modal_fetch_curvyzero_artifact --run-id env-fidelity-YYYYMMDD-001 --ref fidelity/env-fidelity-YYYYMMDD-001/batches/batch-000/scenarios/wall-hit/diff/report.json --output tmp/wall-hit-report.json
uv run python -m scripts.modal_fetch_curvyzero_artifact --run-id env-fidelity-YYYYMMDD-001 --ref fidelity/env-fidelity-YYYYMMDD-001/batches/batch-000/scenarios/wall-hit/diff/first_mismatch.json --output tmp/wall-hit-first-mismatch.json
```

## Top 5 Failure Points

1. JS runner does not read scenario JSON.
   Quick test: run the JS probe against `scenarios/fidelity/smoke.json` and confirm each `scenario_id` gets `js/trace.jsonl`.

2. Python trace uses hidden defaults instead of the scenario file.
   Quick test: change a smoke scenario seed or action script and confirm the Python trace changes.

3. Trace schemas drift.
   Quick test: run local diff on one tiny JS/Python pair and check that `report.json` names missing fields clearly.

4. Modal writes partial artifacts.
   Quick test: require `complete.json`, scenario manifests, and run `manifest.json` before treating a batch as valid.

5. First mismatch is missing or too vague.
   Quick test: force one known mismatch and confirm `diff/first_mismatch.json` includes scenario id, tick, field, JS value, Python value, and previous tick context.

Pass rule: command exit code is not enough. The run passes only when the manifest says `complete`, every scenario has JS/Python traces, every diff report exists, and failed scenarios have `first_mismatch.json`.
