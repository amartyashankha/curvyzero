# Modal Fidelity Jobs

Status: Draft

This page defines the Modal shape for environment fidelity work. It builds on
the comparison plan in `fidelity_comparison.md` and the hot-loop rule in
`../modal_architecture.md`.

## Goal

Use Modal for repeatable remote runs and artifact storage. Do not use Modal for
single game ticks.

A Modal job should run one full scenario or a batch of scenarios. The next loop
is:

```text
scenario JSON -> JS trace -> Python trace -> first-mismatch diff -> artifacts
```

Inside that job, normal local code reads each scenario JSON file, runs the
headless JavaScript reference probe, runs the Python clone probe, compares the
two traces, and writes the artifacts.

The goal is not to host the original CurvyTron browser/server on Modal. Full
browser or websocket hosting is later demo work. It must not block environment
reconstruction.

## Job Structure

Use one Modal app for this lane:

```text
curvyzero-env-fidelity
```

Use these coarse Functions:

| Function | Runs | Writes | Notes |
| --- | --- | --- | --- |
| `js_reference_probe` | One scenario batch against the headless CurvyTron JS reference. | JS traces and event logs. | Uses a Node-capable image. It should run probe scripts locally inside the container. It does not need to host the public game server. |
| `python_clone_probe` | The same scenario batch against CurvyZero Python. | Python traces and event logs. | Uses the normal CurvyZero CPU image. It should share the scenario schema with the JS probe. |
| `trace_diff` | A completed JS/Python artifact pair. | Diff reports and pass/fail summaries. | Reads files from the Volume, compares locally, and writes one report per scenario. |
| `fidelity_batch` | Full end-to-end batch. | All artifacts plus batch manifest. | Preferred command path. It calls local helpers inside one Function, not Modal Functions per tick. |
| `package_artifacts` | Completed batch refs. | Small export bundle or bucket manifest. | Optional later step for long-lived archives. |

The first implementation can make `fidelity_batch` the only public Function.
The smaller Functions are useful when we need to rerun only the JS probe, only
the Python probe, or only the diff.

`fidelity_batch` is the main shape: one Modal job reads scenario JSON, produces
JS traces, produces Python traces, writes first-mismatch diffs, then writes
artifacts and manifests.

## No Per-Step Modal Calls

The simulator loop must stay inside one process:

```text
good:
  modal function starts
  load scenario batch
  for each scenario:
    read scenario JSON
    run JS ticks locally
    run Python ticks locally
    diff local trace files and record the first mismatch
  write artifacts and manifests
  commit Volume
  return compact summary

bad:
  local script calls Modal for tick 1
  local script calls Modal for tick 2
  local script calls Modal for tick 3
```

Do not use `modal.Queue`, `modal.Dict`, `.remote()`, `.map()`, or web requests
inside `env.step()`, the JS tick loop, or the trace diff loop. Use Modal only
around whole jobs, shards, or batches.

Use `.map()` or `.starmap()` only at the scenario-batch level:

```text
batch A: straight movement, left turn, wall hit
batch B: self collision, opponent trail, same-tick double death
batch C: trail print/hole boundary, bonus catch, full replay
```

## Scenario Input

Each scenario should be a small JSON file:

```text
scenarios/<scenario_id>.json
```

Required fields:

- `scenario_id`
- `ruleset_id`
- `player_count`
- `seed`
- `initial_state`, if the probe supports forced state
- `action_script`
- `time_policy`, such as fixed 60 Hz or source elapsed milliseconds
- `trace_schema_version`
- `tolerances`
- `provenance`, such as `source-derived`, `source-inspired`, `v0-choice`, or
  `unresolved`

The scenario file is the contract. JS and Python should both read it. The diff
should not need hidden settings. The expected trace path is scenario JSON to JS
trace, the same scenario JSON to Python trace, then a first-mismatch diff and
artifact write.

## Artifact Paths

Use one Modal Volume first:

```text
curvyzero-runs mounted at /runs
```

Put fidelity outputs under:

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
          js/
            trace.jsonl
            events.jsonl
            stdout.txt
            manifest.json
          python/
            trace.jsonl
            events.jsonl
            stdout.txt
            manifest.json
          diff/
            report.json
            first_mismatch.json
            summary.txt
            manifest.json
      complete.json
  exports/
    <export_id>/
      manifest.json
```

Keep files medium sized. Do not write one file per tick. Use JSONL traces,
compressed JSONL, Parquet, NPZ, or another chunked format once traces get large.

## Manifest Contract

Every manifest is small JSON. It tells a reader what exists and how it was made.

Run manifest:

```json
{
  "run_id": "env-fidelity-20260508-001",
  "created_at": "2026-05-08T00:00:00Z",
  "modal_app": "curvyzero-env-fidelity",
  "code_ref": "<git-sha-or-package-version>",
  "scenario_set_ref": "scenario_set.json",
  "batches": ["batches/batch-000/manifest.json"],
  "status": "complete"
}
```

Batch manifest:

```json
{
  "batch_id": "batch-000",
  "run_id": "env-fidelity-20260508-001",
  "scenarios": ["straight-60", "left-turn-60"],
  "artifact_root": "fidelity/env-fidelity-20260508-001/batches/batch-000",
  "status": "complete"
}
```

Scenario manifest:

```json
{
  "scenario_id": "straight-60",
  "ruleset_id": "curvytron-v1-reference",
  "source_target": "curvytron-v1-reference",
  "js_trace": "js/trace.jsonl",
  "python_trace": "python/trace.jsonl",
  "diff_report": "diff/report.json",
  "first_mismatch": "diff/first_mismatch.json",
  "trace_schema_version": 1,
  "tolerances": {
    "position_abs": 0.000001,
    "angle_abs": 0.000001
  },
  "status": "pass"
}
```

Write payload files first. Write `manifest.json` and `complete.json` last. If a
job retries, it should write a new attempt or detect that a complete artifact
already exists.

## Storage Rules

- Use a Modal Volume for active fidelity runs.
- Use immutable run, batch, scenario, and attempt paths.
- Commit the Volume after each batch or after a large scenario completes.
- Return only a compact summary to the caller: run id, batch ids, pass/fail
  counts, and exact artifact refs.
- Promote old or large outputs to a bucket later with `package_artifacts`.
- Keep `modal.Dict` for tiny pointers only, such as latest good run. Do not put
  traces in a Dict.

## Local Versus Modal

Run locally:

- Fast Python unit tests.
- Single-scenario debugging.
- Editing and inspecting trace schemas.
- Small golden cases that do not need the JS reference runtime.
- Local diff development against already fetched artifacts.

Run on Modal:

- Headless JS reference probes that need the pinned Node image.
- Python clone probes that should match the remote runtime.
- Full scenario batches.
- Trace diffs over remote artifacts.
- Reproducibility checks before promoting a fidelity claim.

Use local runs for speed. Use Modal when the runtime, artifact storage, or
batch repeatability matters.

Later demo-only Modal work:

- Hosting the original CurvyTron browser/server.
- Playwright checks against the browser.
- Screenshot or video capture from the browser UI.
- Websocket protocol comparison beyond what a headless probe records.

These can help demos and UI confidence later. Full browser hosting remains
deferred and is not required for the first source-fidelity loop.

## Image Split

Use at least two images:

- `reference_image`: Node plus the old CurvyTron source and small headless
  probe scripts.
- `python_env_image`: Python 3.11 plus CurvyZero source, tests, and trace tools.

The first `fidelity_batch` can use one combined image if that makes the first
probe simpler. Split images once build time, dependency conflicts, or cache
behavior becomes painful.

## Python-Only Smoke Skeleton

The repo has `src/curvyzero/infra/modal/fidelity_smoke.py` and
`src/curvyzero/env/tracing.py`. Use that Python-only smoke as the first safe
Modal trace job. Keep it smaller than the future `environment_fidelity` job:

- One Modal app, for example `curvyzero-env-fidelity-smoke`.
- One public Function that runs one Python trace helper call.
- No JS oracle, no trace diff, and no browser/server hosting.
- One Volume write to `curvyzero-runs`, then one Volume commit.
- Return only run id, attempt id, artifact ref, byte count, and hash.

The Function should write a single JSON trace/fingerprint artifact under an
immutable path:

```text
/runs/fidelity-smoke/<run_id>/attempts/<attempt_id>/trace_fingerprint.json
```

That JSON should identify the helper name, helper version or code ref, scenario
id, seed, rules hash, trace schema version, fingerprint hash, and the compact
trace payload or trace ref. It is allowed to be Python-only because this smoke
checks the Modal job boundary and artifact contract, not source fidelity.

Do not implement the trace loop inside the Modal file. The Modal Function should
call the local helper once and let the helper own trace fields.

## Future Commands

Aim for these commands. Only the Python-only smoke module is present in this
repo right now; the `environment_fidelity` module and client scripts below are
future command shapes until matching files are added.

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.fidelity_smoke --run-id <run_id>
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind js-probe --scenario-set scenarios/fidelity/smoke.json --run-id <run_id>
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind python-probe --scenario-set scenarios/fidelity/smoke.json --run-id <run_id>
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind diff --run-id <run_id> --batch-id batch-000
uv run --extra modal modal run -m curvyzero.infra.modal.environment_fidelity --kind batch --scenario-set scenarios/fidelity/smoke.json --run-id <run_id>
```

After the app is stable, deploy it and call named Functions from small client
scripts:

```sh
uv run --extra modal modal deploy -m curvyzero.infra.modal.environment_fidelity
uv run python -m scripts.modal_run_env_fidelity --scenario-set scenarios/fidelity/smoke.json --run-id <run_id>
uv run python -m scripts.modal_fetch_curvyzero_artifact --run-id <run_id> --ref fidelity/<run_id>/manifest.json --output tmp/<run_id>-manifest.json
```

## First Build Order

1. Add local scenario JSON files and a trace schema.
2. Make the JS reference probe read scenario JSON and write a JS trace.
3. Make the Python trace helper read the same scenario JSON and write a Python
   trace.
4. Add a local diff that writes `report.json` and `first_mismatch.json`.
5. Add artifact paths and manifests for the full loop.
6. Keep the Python-only `fidelity_smoke.py` as a small Modal boundary check.
7. Add `fidelity_batch` on Modal as one coarse job for scenario batches.
8. Make `fidelity_batch` run scenario JSON -> JS trace -> Python trace ->
   first-mismatch diff -> artifacts inside the job.
9. Add artifact fetch and exact-ref debugging commands.
10. Defer full browser/server hosting until there is a separate demo need.
