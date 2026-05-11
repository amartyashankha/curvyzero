# Vector Lifecycle Reset/Spawn Boundary Plan

Status on 2026-05-09: a small reset/spawn composition helper exists, with
optional 1v1 delayed-start scheduling metadata when the timer arrays are
present, but full vector lifecycle still does not.

## Implemented Boundary

`src/curvyzero/env/vector_lifecycle.py::reset_and_spawn_round_rows(...)` is the
current production-shaped fast-path seam between reset and spawn:

```text
reset_and_spawn_round_rows(
    target,
    reset_template,
    row_mask,
    *,
    player_count,
    reset_seed,
    reset_source,
    snapshot_array_names=None,
) -> metadata
```

The helper does exactly this:

1. Validate the row mask.
2. Preflight that `target` and `reset_template` can compose.
3. Call `vector_reset.reset_arrays(...)` for selected rows.
4. Call `vector_spawn.spawn_round_rows(...)` for the same selected rows.
5. Optionally stamp 1v1 round flags and delayed `3000 ms`
   PrintManager-start timer metadata when all lifecycle/timer arrays are
   present. This metadata still reports `full_lifecycle=false`.
6. Return reset metadata, spawn metadata, schedule metadata, and a top-level
   `terminal_transition_snapshot` alias from the reset metadata.

The selected rows' row-local random tape comes from the reset template because
the reset copy happens before spawn. Spawn then advances only selected
`random_tape_cursor` and `random_tape_draw_count` rows.

## Array Composition Report

Before mutation, the helper checks the exact array names needed for composition.
It requires:

- Reset rows: `episode_id`, `episode_step`, `env_active`, `reset_pending`,
  `done`, `terminated`, `truncated`, `terminal_reason`, `reset_seed`,
  `reset_source`.
- Spawn rows: `pos`, `heading`, `alive`, `present`, `map_size`,
  `random_tape_values`, `random_tape_length`, `random_tape_cursor`,
  `random_tape_draw_count`.
- Matching `target` and `reset_template` key sets, because `reset_arrays(...)`
  copies template rows by array name.

If required arrays are absent or the key sets differ, the helper returns
`can_compose=False` without mutating `target`. The returned metadata includes:

- `missing_target_arrays`
- `missing_reset_template_arrays`
- `missing_target_required_arrays`
- `missing_reset_template_required_arrays`
- `target_only_arrays`
- `reset_template_only_arrays`

Shape, dtype, value, tape-exhaustion, and spawn geometry errors still come from
the underlying `vector_reset` and `vector_spawn` validators.

## Test Coverage

`tests/test_vector_lifecycle.py` covers:

- selected rows reset from the template and then spawn from template-local tape;
- optional 1v1 delayed-start timer scheduling metadata while
  `full_lifecycle=false`;
- terminal snapshots survive the reset and later spawn mutation;
- skipped rows and their random cursors remain untouched;
- missing/mismatched arrays return exact metadata and do not mutate target rows.

## Still Missing

This helper is not full lifecycle. It does not:

- schedule warmup, warmdown, next-round, or game-start timers;
- advance or fire the optional delayed-start timer rows it can stamp;
- emit lifecycle events;
- insert or mutate world bodies;
- generate seeds or maintain seed history;
- build observations, rewards, final observations, replay surfaces, or trainer
  API output;
- autoreset after transition construction;
- implement scoring, match end, round end, or next-round behavior.

Those pieces still need separate source-backed tests before any claim that
vector rows reproduce the promoted lifecycle fixtures end to end.
