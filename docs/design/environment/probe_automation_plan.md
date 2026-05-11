# Probe Automation Plan

Status: draft

This page names the next small automation step for environment probes and
replays. It only covers local probe automation. Modal batch jobs stay later.

## Current Tools

- `tools/reference_oracle/scenario_runner.js` is the main JS oracle path. It
  loads the original CurvyTron source in a VM, reads one scenario JSON file, runs
  each step, and writes a source-style trace with game state, avatar state, and
  events.
- `tools/reference_oracle/headless_probe.js` is still useful as a hard-coded
  smoke probe. It proves the reference source can be loaded and stepped without
  the browser.
- `tools/reference_oracle/border_probe.js` is a hard-coded behavior probe for
  normal wall death versus borderless wrap.
- `tools/run_fidelity_loop.py` runs one local scenario end to end: JS oracle,
  Python runner, first-mismatch diff, and `summary.json`. It uses common-trace
  diff by default; use `--raw-diff` only for debugging runner output.
- `tools/run_fidelity_batch.py` runs a JSON list of scenarios through the same
  one-scenario loop and writes one compact batch `summary.json`.
- `tools/fidelity_diff.py` reports the first mismatch. In common-trace mode, it
  first projects JS and Python outputs into the small shared trace shape.

## Next Small Steps

1. Done: add a local scenario list runner.
   - Input: one text or JSON list of scenario paths.
   - For each scenario, call the existing one-scenario loop.
   - Keep the same per-scenario artifact folders.
   - Write one batch summary at the artifact root.
   - Return non-zero only for runner crashes or blocked diffs by default.
   - Use `--fail-on-mismatch` when strict mismatch failure is needed.

2. Done: make JSON summaries easy to scan.
   - Keep the existing per-scenario `summary.json`.
   - Add a batch `summary.json` with counts for `pass`, `fail`, and `blocked`.
   - Include the first mismatch path and message for each failed scenario.
   - Include paths to JS, Python, diff, and stderr files.
   - Do not add a new report format yet.

3. Done: make common-trace comparison the default.
   - Use common trace as the normal mode for local fidelity runs.
   - Keep raw exact diff behind an explicit flag for debugging.
   - Record the mode in every scenario and batch summary.
   - This matches the current schema reality: raw JS and raw Python artifacts do
     not have the same shape.

4. Fold hard-coded probes into scenarios one at a time.
   - Keep `headless_probe.js` as a smoke check.
   - Convert border behavior into scenario fixtures before adding more special
     JS probe scripts.
   - Prefer one shared scenario schema over new one-off probes.

5. Add Modal batch later.
   - First prove local list runs, batch summaries, and common-trace defaults.
   - Then wrap the same local runner in one Modal batch job.
   - Do not call Modal per tick or per env step.

## First Batch Shape

The first local batch can be small:

```text
scenarios/environment/source_kinematics_straight_step.json
scenarios/environment/source_kinematics_left_turn_step.json
scenarios/environment/source_kinematics_right_turn_step.json
scenarios/environment/forced_two_player_turn_step.json
```

Expected command shape:

```text
python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics
```

The batch runner should be thin. It should reuse `run_fidelity_loop.run_loop`
instead of copying JS, Python, or diff logic.

Latest local result on 2026-05-08:

- command: `uv run python tools/run_fidelity_batch.py scenarios/environment/source_kinematics_batch.json --python-runner source-kinematics --artifact-root /private/tmp/curvy-source-kinematics-batch`
- result: `4` pass, `0` fail, `0` blocked
- artifact root: `/private/tmp/curvy-source-kinematics-batch`

## Batch Summary Shape

Use one compact JSON object:

```json
{
  "schema": "curvyzero_local_fidelity_batch/v1",
  "artifact_root": "artifacts/local/fidelity",
  "diff_mode": "common-trace",
  "counts": {
    "pass": 0,
    "fail": 0,
    "blocked": 0
  },
  "scenarios": [
    {
      "scenario_id": "forced_two_player_turn_step",
      "status": "fail",
      "summary_path": "artifacts/local/fidelity/forced_two_player_turn_step/summary.json",
      "first_mismatch": {
        "path": "$.steps[0].players[0].x",
        "message": "First mismatch at $.steps[0].players[0].x."
      }
    }
  ]
}
```

Keep this summary small. Large traces stay in the per-scenario artifact folders.

## Not Yet

- Do not build Modal batch first.
- Do not add broader tolerance logic before a real numeric mismatch needs it.
- Do not expand browser, websocket, or screenshot replay work in this step.
- Do not add another hard-coded JS probe when a scenario fixture can express the
  same setup.
