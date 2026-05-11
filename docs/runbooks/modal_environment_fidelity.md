# Modal Environment Fidelity Runbook

Status: Draft

This runbook describes the command shape we should aim for. Some commands are
future commands; they name the intended files and entry points before the code
exists.

## Rule Of Thumb

Use Modal for whole scenarios or batches. Do not call Modal once per game tick.
Do not make full CurvyTron browser/server hosting part of the first fidelity
loop.

The normal flow is:

```text
pick scenario set
run one Modal batch job
inside the job:
  read scenario JSON
  write JS trace
  write Python trace from the same scenario JSON
  write first-mismatch diff
  write artifacts and manifests
fetch only the small manifest or one exact failure file
debug locally
repeat
```

## Current Checks

These commands exist today and are safe first checks from the repo root:

```sh
uv run --extra dev pytest
uv run python scripts/benchmark_env.py --episodes 1000
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind tests
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind benchmark --episodes 25 --max-steps 500
uv run --extra modal modal run -m curvyzero.infra.modal.smoke --kind gpu
uv run --extra modal modal run -m curvyzero.infra.modal.fidelity_smoke --run-id env-fidelity-smoke-YYYYMMDD-001
```

These are not full fidelity proofs. They prove that local tests, remote tests,
remote benchmarking, GPU visibility, and one Python-only trace artifact write
are working.

## Python-Only Fidelity Smoke

Run this command when checking the Modal wrapper for the Python trace helper:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.fidelity_smoke --run-id env-fidelity-smoke-YYYYMMDD-001
```

This runs one Python trace helper call inside one Modal Function. It does not
call the JS oracle yet. It writes one JSON trace/fingerprint artifact to the
`curvyzero-runs` Volume, commits the Volume once, and returns the exact artifact
ref:

```text
experiments/env-fidelity-smoke-YYYYMMDD-001/attempts/<attempt_id>/fidelity-smoke/trace_fingerprint.json
```

Keep this smoke at the job boundary. The trace helper may step the environment
inside the remote process, but local code must not call Modal once per tick.

## Future One-Off Fidelity Commands

The `environment_fidelity` module named here is planned, not present in this
repo yet. Use one-off `modal run` while that app is still changing:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind batch --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
```

When debugging one stage, keep the unit of work as a whole scenario set or
batch, not one game tick:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind js-probe --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind python-probe --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind diff --run-id env-fidelity-YYYYMMDD-001 --batch-id batch-000
```

Preferred path:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind batch --scenario-set scenarios/fidelity/canonical.json --run-id env-fidelity-YYYYMMDD-001
```

That command should run scenario JSON -> JS trace -> Python trace ->
first-mismatch diff -> artifacts inside one Modal job. It should use headless
reference probes. It should not need to expose the original web server.

## Future Deployed Commands

Deploy when the batch job is stable:

```sh
uv run --extra modal modal deploy -m curvyzero.infra.modal.environment_fidelity
```

Then use small local client scripts:

```sh
uv run python -m scripts.modal_run_env_fidelity --scenario-set scenarios/fidelity/smoke.json --run-id env-fidelity-YYYYMMDD-001
uv run python -m scripts.modal_run_env_fidelity --scenario-set scenarios/fidelity/canonical.json --run-id env-fidelity-YYYYMMDD-002 --max-containers 4
uv run python -m scripts.modal_fetch_curvyzero_artifact --run-id env-fidelity-YYYYMMDD-001 --ref fidelity/env-fidelity-YYYYMMDD-001/manifest.json --output tmp/env-fidelity-YYYYMMDD-001-manifest.json
```

The client script should call `modal.Function.from_name`. It should fail loudly
if the deployed app or Function is missing. These client scripts are future
command shapes until matching files are added.

## Artifact Locations

Active Modal artifacts live in the `curvyzero-runs` Volume:

```text
/runs/fidelity/<run_id>/
  manifest.json
  scenario_set.json
  batches/
    <batch_id>/
      manifest.json
      scenarios/
        <scenario_id>/
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

The returned command summary should include exact refs, for example:

```text
fidelity/env-fidelity-YYYYMMDD-001/manifest.json
fidelity/env-fidelity-YYYYMMDD-001/batches/batch-000/scenarios/wall-hit/diff/report.json
fidelity/env-fidelity-YYYYMMDD-001/batches/batch-000/scenarios/wall-hit/diff/first_mismatch.json
```

Fetch exact refs. Do not download the whole run tree by default.

Future fetch examples:

```sh
uv run python -m scripts.modal_fetch_curvyzero_artifact --run-id env-fidelity-YYYYMMDD-001 --ref fidelity/env-fidelity-YYYYMMDD-001/batches/batch-000/scenarios/wall-hit/diff/report.json --output tmp/wall-hit-report.json
uv run python -m scripts.modal_fetch_curvyzero_artifact --run-id env-fidelity-YYYYMMDD-001 --ref fidelity/env-fidelity-YYYYMMDD-001/batches/batch-000/scenarios/wall-hit/diff/first_mismatch.json --output tmp/wall-hit-first-mismatch.json
```

## What Runs Locally

Run these locally first:

- Python unit tests.
- Small simulator golden tests.
- Local benchmark smoke.
- Trace schema changes.
- Diff logic against a tiny fixture.
- Drafting and inspecting scenario JSON.
- Human inspection of fetched diff reports.

Local commands should be quick:

```sh
uv run --extra dev pytest
uv run python scripts/benchmark_env.py --episodes 100 --max-steps 500
uv run python -m scripts.env_fidelity_diff --js-trace tmp/js.jsonl --python-trace tmp/python.jsonl --output tmp/diff.json
```

`scripts.env_fidelity_diff` is a future command. It should use the same diff
code as Modal.

## What Runs On Modal

Run these on Modal:

- Full scenario batches that read scenario JSON.
- Headless JS reference probes that write JS traces.
- Python clone probes that write Python traces in the same runtime.
- First-mismatch trace diffs over Volume artifacts.

Use Modal when the exact runtime matters or when the output should be stored as
a shared artifact.

Later demo-only Modal work:

- Host the original CurvyTron browser/server.
- Run Playwright against the browser.
- Capture screenshots or videos.
- Compare browser websocket behavior after state traces are useful.

Full browser hosting remains deferred. Do not treat these as blockers for
environment reconstruction.

## Batch Rules

Each batch job should:

1. Read one scenario set.
2. Create `/runs/fidelity/<run_id>/batches/<batch_id>/`.
3. For each scenario, read scenario JSON, write the JS trace, then write the
   Python trace from the same scenario JSON.
4. Write diff reports, including `first_mismatch.json` for failures.
5. Write scenario manifests.
6. Write `complete.json` after all payload files exist.
7. Commit the Volume.
8. Return a compact summary.

The summary should include:

- `run_id`
- `batch_id`
- pass count
- fail count
- first failing scenario id
- exact Volume refs for `manifest.json` and first mismatch files

## Failure Handling

If a batch fails before `complete.json`, treat the batch as incomplete.

On retry:

- Prefer a new `attempt_id` path if the previous output is partial.
- Reuse completed artifacts only when their manifest status is `complete`.
- Never overwrite old traces in place.
- Write the final manifest last.

## First Scenario Set

Start with this smoke set:

- straight movement for 60 ticks
- left turn for 60 ticks
- wall hit from a known position
- opponent trail hit
- same-tick double death

Then add:

- self-collision latency threshold
- trail print/hole boundary
- bonus catch changes speed
- one full fixed replay

Do not add browser/server hosting to this first scenario set. Keep it headless.

## Pass Criteria

A fidelity batch passes only when:

- every scenario has JS and Python artifacts, unless the scenario is marked
  Python-only
- every diff report exists
- exact fields match: alive/dead, terminal flags, death cause, score events, and
  event order
- numeric fields are within scenario tolerances
- the first mismatch file exists for every failed scenario
- the run manifest says `complete`

Command exit code alone is not enough. Check the manifest and artifact refs.
