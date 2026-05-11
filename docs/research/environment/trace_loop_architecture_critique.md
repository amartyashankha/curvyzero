# Trace Loop Critique

Status: Draft
Date: 2026-05-08

Current loop:

```text
scenario JSON -> JS trace -> Python trace -> common trace -> diff -> artifacts
```

The loop is right. The problem is that the contract is loose. Too many files
describe overlapping shapes, and the tools still accept several aliases.

## Top 5 Fixes

1. Freeze one scenario shape.

   Use `scenario_id`, `players`, `steps`, `time`, `world`, and `comparison`.
   Treat `id`, `ticks`, `action_script`, `source_setup`, and `initial_state` as
   old compatibility names only.

2. Stop diffing raw traces.

   JS raw output is source-shaped. Python raw output is toy-runner-shaped. They
   should not match. Only compare normalized common traces.

3. Make normalization explicit.

   Write `common_trace.json` for JS and Python before diffing. The normalizer
   owns frame alignment, player id mapping, field names, and dropping the Python
   reset frame.

4. Make the differ stricter.

   The differ should read the scenario comparison block and return `pass`,
   `fail`, or `blocked`. If Python says `source_fidelity: false`, a source
   fidelity check is `blocked`, not failed.

5. Keep Modal boring.

   Modal should run whole scenario batches and store artifacts. It should not
   know physics, trace fields, normalization rules, or diff policy.

## Cut Scope

Do now:

- one scenario schema
- JS runner
- Python runner
- normalizer
- first-mismatch diff
- local artifact writer
- one Modal batch wrapper

Do later:

- browser hosting
- websocket protocol checks
- screenshots
- video
- bonus-heavy fixtures
- extra Modal entry points
