# CurvyTron accumulated replay audit - 2026-05-10

## Scope

Audited:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py`

No pytest was run. I used compile and synthetic introspection only.

## Verdict

Pass after one tiny summary fix.

The learner path itself is wired correctly for the accumulated replay patch:
default behavior remains `current_iteration`, accumulated mode makes each
learner update see all replay rows collected so far, and `learner_sample_size`
sampling preserves the metadata arrays required by the survival target adapter.

I found and fixed one small reporting bug: the top-level `replay.sample` summary
previously summarized all collected rows even when `replay_scope` was the
default `current_iteration`. That could make a multi-iteration current-iteration
run look accumulated in the compact summary. The training update path was
already using the correct scoped rows; the fix makes the top-level summary use
the final learner-visible replay set and labels `row_count` as total collected
rows.

## Checks

Default behavior stays current iteration:

- `run_curvytron_two_seat_lightzero_train_smoke(..., replay_scope=REPLAY_SCOPE_CURRENT_ITERATION)`
- local CLI `--replay-scope` defaults to `REPLAY_SCOPE_CURRENT_ITERATION`
- Modal function and local entrypoint default `replay_scope="current_iteration"`
- accumulated mode is opt-in.

Accumulated mode samples all collected replay so far:

- after each collection phase, `iteration_replay_rows` are extended into
  `replay_rows`
- when `replay_scope == "accumulated"`, `learner_replay_rows = replay_rows`
- when `learner_sample_size` is unset or at least the available row count,
  `_select_replay_rows` returns every row in collection order.

`learner_sample_size` preserves metadata arrays:

- `_sample_replay_batch` selects complete row dictionaries, then builds
  `iteration_batch`, `env_row_id_batch`, `player_id_batch`, and
  `decision_index_batch` from the selected rows
- `_learn_mode_batches` receives the sampled batch and `_target_value_batch`
  stays on the `discounted_survival_return` path when those keys are present.

Determinism is limited to reproducible learner batch draw:

- deterministic sampling uses `_replay_sample_rng(seed, iteration, update_index)`
  only for `_select_replay_rows`
- environment construction still receives the run seed normally; accumulated
  replay does not pin a fixed environment row or replay the same environment
  trajectory by itself.

Summaries are now clear:

- top-level `replay.row_count` is total collected replay rows
- top-level `replay.replay_rows_available`, `learner_batch_size`, and `sample`
  describe the final learner-visible replay rows under the selected scope
- per-iteration summaries still show both `iteration_sample` and
  `learner_sample`.

Modal wiring:

- remote function accepts and forwards `replay_scope` and `learner_sample_size`
- local Modal entrypoint accepts and passes both args to `.remote(...)`
- the module docstring command includes both flags.

## Verification

Ran:

```bash
python3 -m py_compile \
  src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py \
  src/curvyzero/infra/modal/lightzero_curvytron_two_seat_train_smoke.py
```

Result: clean.

Ran AST/default introspection and synthetic replay sampling. The synthetic
sample with `learner_sample_size=3` reported:

```text
sample 6 3 [0, 2, 5] (3,) (3,) (3,) (3,)
payload_replay 6 total collected replay rows 2 2 final learner-visible replay rows
```

That confirms metadata arrays survive sampling and the summary fix separates
total collected rows from learner-visible rows.

## Residual Risks

Bounded `learner_sample_size` can still truncate future rows from a trajectory.
The metadata target path remains active, but discounted returns are computed
from the sampled rows, not an unsampled full replay table.

This remains the local two-seat smoke adapter, not LightZero's upstream
GameBuffer, replay priorities, distributed collector, or full self-play trainer.

The whole accumulated table is kept in memory for the smoke run. That is fine
for bounded experiments but is not a durable replay-store design.
