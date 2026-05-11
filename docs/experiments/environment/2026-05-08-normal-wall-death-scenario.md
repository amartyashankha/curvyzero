# 2026-05-08 Normal-Wall Death Scenario

## Question

Can the shared scenario loop reproduce the source normal-wall death case from
`tools/reference_oracle/border_probe.js`?

## Scenario

- Fixture: `scenarios/environment/source_normal_wall_death_step.json`
- Source behavior: p0 starts at x `87.35` in an 88-unit arena with radius `0.6`,
  moves straight for `100` ms, reaches x `88.95`, and dies by normal-wall death.
- Scope: source movement plus normal-wall death state/events only. No body
  collision, trails, bonuses, or full scoring model.

Second fixture:

- Fixture: `scenarios/environment/source_normal_wall_same_frame_draw_step.json`
- Source behavior: p0 and p1 start just inside opposite walls, both move outward
  for `100` ms, both die in the same `Game.update` frame, and no one receives
  survivor score.
- Batch: `scenarios/environment/source_normal_wall_batch.json`

## Commands And Results

```sh
node tools/reference_oracle/scenario_runner.js scenarios/environment/source_normal_wall_death_step.json
```

Result: JS reference emitted p0 at `[88.95, 44]`, p0 `alive: false`, p1 at
`[42.4, 44]`, p1 `alive: true`, final p1 score `1`.

```sh
uv run python tools/run_fidelity_loop.py scenarios/environment/source_normal_wall_death_step.json --artifact-root /private/tmp/curvy-normal-wall-death-loop
```

Result: expected toy-v0 mismatch. First mismatch:

```text
$.steps[0].players[0].x: JS 88.95, toy-v0 88.3499984741211
```

```sh
uv run python tools/run_fidelity_loop.py scenarios/environment/source_normal_wall_death_step.json --python-runner source-normal-wall --artifact-root /private/tmp/curvy-normal-wall-death-loop-source-runner
```

Result: `match: true`, `diff_status: pass`, `first_mismatch: null`.

```sh
uv run python tools/run_fidelity_loop.py scenarios/environment/source_normal_wall_same_frame_draw_step.json --python-runner source-normal-wall --artifact-root /private/tmp/curvy-normal-wall-draw-loop-source-runner
```

Result: `match: true`, `diff_status: pass`, `first_mismatch: null`.

```sh
uv run python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_batch.json --python-runner source-normal-wall --artifact-root /private/tmp/curvy-source-normal-wall-batch
```

Result: `2` pass, `0` fail, `0` blocked.

```sh
uv run python tools/run_fidelity_batch.py scenarios/environment/source_border_batch.json --python-runner source-border-rules --artifact-root /private/tmp/curvy-source-border-events-batch
```

Result: `3` pass, `0` fail, `0` blocked. The two normal-wall fixtures each had
`match: true`, `diff_status: pass`, and `first_mismatch: null` with
`comparison.include_events: true`.

```sh
uv run python tools/run_fidelity_batch.py scenarios/environment/source_normal_wall_multiplayer_batch.json --python-runner source-border-rules --fail-on-mismatch --artifact-root /private/tmp/curvy-source-normal-wall-multiplayer-batch-terminal-draw-final
```

Result: `3` pass, `0` fail, `0` blocked; `diff_mode: common-trace`;
`source_normal_wall_3p_two_die_one_survivor_step`,
`source_normal_wall_4p_ordered_deaths_survivor_score`, and
`source_normal_wall_4p_two_prior_then_same_frame_terminal_draw` each had
`first_mismatch: null`.

Full verification commands for this doc update:

```sh
uv run --extra dev pytest
```

Result: `64 passed in 0.40s`.

```sh
uv run --extra dev ruff check .
```

Result: `All checks passed!`.

## Interpretation

The shared JS runner, scenario files, common trace, and Python source-normal-wall
runner now agree for one single-wall death case and one same-frame wall draw
case. The mixed `source-border-rules` batch also verifies the narrow normal-wall
event contract for those cases. The multiplayer batch verifies the narrow 3P/4P
normal-wall scoring, death-order, and 4P terminal draw canaries only. The runner
is deliberately narrow; broader body, trail, bonus, and replay events are still
outside this experiment.
