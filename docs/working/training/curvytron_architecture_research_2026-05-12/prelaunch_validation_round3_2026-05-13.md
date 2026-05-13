# Prelaunch Validation Round 3 - 2026-05-13

Scope: local validation after the survivaldiag manifest/docs cleanup. No Modal
launches and no large matrix launch.

## What Changed Before This Round

- Active docs were reconciled with the current survivaldiag manifest shape.
- `scripts/build_curvytron_survivaldiag_manifest.py` now uses `row_note` for
  executable dirty-control and sim16 sentinel rows.
- Only future non-commanded specs use `gate`, `status=gated_not_commanded`, and
  `command_omitted=true`.
- The manifest artifacts were regenerated under
  `artifacts/local/curvytron_survivaldiag_manifests/`.

## Commands Run

```bash
uv run python scripts/build_curvytron_survivaldiag_manifest.py
```

Result: passed. It emitted `50` executable review rows, `25` logical render
pairs, `10` gated specs, `dry_run_only=true`, `launches_modal=false`, and
`current_launch_approved=false`.

```bash
uv run pytest tests/test_curvytron_survivaldiag_manifest.py tests/test_eval_curves.py tests/test_curvytron_run_status.py -q
```

Result: passed, `20 passed`.

```bash
uv run pytest tests/test_curvyzero_source_state_visual_survival_lightzero_env.py tests/test_vector_multiplayer_env.py tests/test_vector_runtime.py -q
```

Result: passed, `209 passed`.

```bash
uv run pytest tests/test_curvytron_live_checkpoint_eval_plumbing.py tests/test_curvytron_gif_browser.py -q
```

Result: passed, `67 passed, 10 skipped`.

```bash
uv run ruff check scripts/build_curvytron_survivaldiag_manifest.py tests/test_curvytron_survivaldiag_manifest.py scripts/analyze_curvytron_eval_curves.py src/curvyzero/analysis tests/test_eval_curves.py
```

Result: passed.

## Gate Read

Locally cleared:

- survivaldiag manifest shape and dry-run safety;
- row naming and `row_note` semantics for executable dirty/sentinel rows;
- local eval-curve and run-status parsing;
- source-state reward, blank-canvas, action-repeat, vector runtime, and wrapper
  behavior;
- local live checkpoint/GIF plumbing tests.

Still not cleared locally:

- high-cap live runtime canary with `source_max_steps=65536`;
- real live survivaldiag status/eval snapshot showing rich readout fields;
- exact passive-immortal dirty-control canary, if those rows stay in the first
  executable matrix;
- scripted/random/checkpoint opponent rows.

Verdict: local validation is green for the current dry-run manifest and clean
blank-canvas first-wave core. The launch remains blocked on live canary/readout
gates.
