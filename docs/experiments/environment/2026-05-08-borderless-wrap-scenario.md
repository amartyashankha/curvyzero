# 2026-05-08 Borderless Wrap Scenario

## Question

Can the shared scenario loop reproduce the source borderless wrap case from
`tools/reference_oracle/border_probe.js`?

## Scenario

- Fixture: `scenarios/environment/source_borderless_wrap_step.json`
- Source behavior: p0 starts at x `87.35` in an 88-unit arena, moves straight
  for `100` ms, crosses to x `88.95`, then source borderless wraps it to x `0`.
- Scope: source movement plus borderless wrap state/events only. No body
  collision, trails, bonus stack timing, or full game rules.
- Batch: `scenarios/environment/source_border_batch.json`

## Commands And Results

```sh
node tools/reference_oracle/scenario_runner.js scenarios/environment/source_borderless_wrap_step.json
```

Result: JS reference emitted p0 at `[0, 44]`, p0 `alive: true`, p1 at
`[42.4, 44]`, p1 `alive: true`, scores `[0, 0]`.

```sh
uv run python tools/run_fidelity_loop.py scenarios/environment/source_borderless_wrap_step.json --artifact-root /private/tmp/curvy-borderless-wrap-toy-loop
```

Result: expected toy-v0 mismatch. First mismatch:

```text
$.steps[0].players[0].alive: JS true, toy-v0 false
```

```sh
uv run python tools/run_fidelity_loop.py scenarios/environment/source_borderless_wrap_step.json --python-runner source-borderless-wrap --artifact-root /private/tmp/curvy-borderless-wrap-loop-source-runner
```

Result: `match: true`, `diff_status: pass`, `first_mismatch: null`.

```sh
uv run python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --artifact-root /private/tmp/curvy-source-border-events-batch
```

Result: `3` pass, `0` fail, `0` blocked. `source_borderless_wrap_step` had
`match: true`, `diff_status: pass`, and `first_mismatch: null` with
`comparison.include_events: true`.

## Interpretation

The shared loop now covers three forced border state/event cases: normal-wall
single death, normal-wall same-frame draw, and source borderless wrap. This
locks in the correction that borderless is source-specific edge teleporting, not
a clean torus and not the default rule.
