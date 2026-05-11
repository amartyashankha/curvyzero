# 2026-05-08 Trace Normalization

## Question

Can the local diff compare the current JS and Python runner outputs without
failing first on runner metadata noise?

## Result

Yes. `curvyzero.env.trace_compare.project_common_trace` projects both payloads
into `curvyzero_common_trace/v1`. `tools/fidelity_diff.py --common-trace` runs
the existing first-mismatch diff on those projections.

The projection keeps only:

- scenario id
- step index
- step ms
- player id
- x, y, angle, alive
- score and roundScore when present

## Intentional Mismatches

- JS `loadedSources` is ignored.
- Python toy-v0 labels such as `runner` and `source_fidelity: false` are not used
  in the common trace comparison.
- Python still has a reset frame; the projection drops it for scenario-runner
  payloads with one reset frame plus one frame per action.
- Movement values are still expected to differ for the current fixture.

## Checks

```sh
uv run --extra dev pytest tests/test_trace_compare.py
uv run --extra dev ruff check src/curvyzero/env/trace_compare.py tools/fidelity_diff.py tests/test_trace_compare.py
```
