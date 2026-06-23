# Operating Docs Review - 2026-05-21

Scope: docs-only review of the optimizer working-memory folder. No live
training runs, Modal jobs, production code, checkpoints, evals, GIFs, or
tournaments were touched.

Reviewed:

- `README.md`
- `orchestration.md`
- `task_board.md`
- `world_model.md`
- `experiment_log.md`
- nearby notes: `actual_training_speed_read_20260521.md`,
  `next_experiment_grid.md`, `subagent_real_consumer_canary_plan_20260521.md`,
  `subagent_real_consumer_impl_critique_20260521.md`

## Plain Current Read

The docs contain the right facts, but the word "current" now means too many
different things:

1. live Coach/default training truth;
2. stock full-loop profile truth;
3. profile-only batched observation truth;
4. profile-only resident/real-consumer truth;
5. historical renderer-only truth.

That makes the folder easy to misread. The safest cleanup is not to delete most
history. Instead, add a short current snapshot at the top of each main doc and
push old "current" blocks behind clear labels like "Historical anchor" or
"Superseded by real-consumer canary."

## Stale Or Confusing Statements

### `README.md`

- The opening "Plain Current Truth" says the promising path is not
  trainer-integrated yet. That is still true for production training, but it
  hides the newer fact that a profile-only real LightZero collect-forward
  consumer now exists and passed smoke.
- The "Current next-canary plan" still points first to vector-facade and Amdahl
  docs. It should now say the immediate gate is the medium real-consumer rows:
  H100/L4, B512/A16/sim8, scalar edge off/on.
- The doc mixes "do not recommend to Coach yet" with older renderer speed wins.
  Add one top table with denominators:
  `actual training`, `stock full-loop profile`, `profile-only hybrid`,
  `renderer-only/surface`.

### `orchestration.md`

- The active-wave section is useful, but several named sub-agent statuses are
  stale. Example: the real-consumer canary plan is listed as active even though
  the gate now exists and smoke passed.
- Hooke appears twice with active/completed wording. Use one owner row per task.
- "Next rows" are named in prose, but there is no active-run registry that tells
  the next main agent exactly what is running, how to poll it, and what field
  names to extract.

### `task_board.md`

- There is an unchecked item to implement the next real-consumer canary followed
  by checked items saying it was implemented and smoked. Mark the old item as
  superseded or remove it.
- Add a new unchecked item for ingesting the medium real-consumer rows into
  `experiment_log.md`, `world_model.md`, and `orchestration.md`.
- Add a new unchecked item for the real-consumer critique guardrails:
  require/label uint8 + direct64, clarify `to_play`, and assert/filter zero-mask
  terminal roots before any normal-death row.

### `world_model.md`

- The newest "Real Consumer Canary Status" section is clear and should be moved
  closer to the top. It is now the best summary of the actual Amdahl question.
- Add a decision rule for the medium rows:
  if collect-forward collapses versus synthetic resident rows, the next wall is
  the public LightZero collect/search boundary, not rendering.
- Keep the wording "not device-resident MCTS" prominent. It prevents the common
  mistake of saying "we moved MCTS to GPU" when the current LightZero path still
  crosses through CPU tree/search internals.

### `experiment_log.md`

- It has strong results, but it needs a "Latest Active / Pending Rows" block
  near the top. Without that, future agents must infer run state from scattered
  prose.
- The real-consumer smoke section is good. After medium rows return, add a
  small table beside the synthetic resident table so the drop is obvious.
- Keep inclusive timer caveats near every policy/MCTS table:
  `policy_forward_collect` includes nested model/search work, so do not sum it
  with MCTS as independent wall time.

### Nearby Docs

- `actual_training_speed_read_20260521.md` is doing the right thing by keeping
  actual Coach speed separate from profile speed. Link it from every Coach-facing
  recommendation.
- `next_experiment_grid.md` is stale at the top: it still frames the next work
  mainly around batched GPU manager gates. Add a 2026-05-21 addendum saying the
  real-consumer medium rows now decide whether to keep pushing stock
  collect-forward or pivot toward deeper search batching / MCTX-style research.
- `subagent_real_consumer_impl_critique_20260521.md` should be linked from the
  task board. Its findings are practical and should not get lost.

## Missing Active-Run Registry Convention

Add either `active_run_registry.md` or a top section in `orchestration.md` with
one row per active profile job.

Recommended fields:

| field | meaning |
| --- | --- |
| `owner` | main agent or sub-agent responsible |
| `status` | queued, running, returned, failed, superseded |
| `run_family` | human run family / artifact family |
| `session_id` | local exec session id if there is one |
| `modal_run_url_or_id` | Modal app/run id if detached |
| `command_doc` | file or note containing exact command |
| `purpose` | one sentence: what question this row answers |
| `denominator` | actual training, full-loop profile, hybrid profile, renderer-only |
| `shape` | compute, B/C/A, sims, steps, warmup, scalar edge, RND/death |
| `expected_fields` | required result fields before the row is trusted |
| `result_artifact` | local artifact path or "stdout only" |
| `next_action` | poll, summarize, rerun, ignore, or debug |

Rules:

- No detached profile row should exist without a registry row.
- If a row is killed or superseded, mark it explicitly; do not leave it as
  "running."
- Record the denominator before any speed number. This prevents mixing
  learner iterations/hour, env steps/sec, roots/sec, and renderer surface time.
- For profile-only rows, state `touches_live_runs=false`.

## Cleanest Next Wave After Medium Rows Return

First, ingest the four medium real-consumer rows:

```text
H100 scalar edge off
H100 scalar edge on
L4/T4 scalar edge off
L4/T4 scalar edge on
```

Compare them against the synthetic resident anchors:

```text
H100 synthetic scalar-off: ~10.98k roots/sec
H100 synthetic scalar-on:  ~7.62k roots/sec
L4 synthetic scalar-off:   ~5.84k roots/sec
L4 synthetic scalar-on:    ~4.13k roots/sec
```

Then branch:

1. If real collect-forward keeps most of the synthetic speed, run a small
   normal-death/RND guardrail next. That means the public LightZero boundary may
   be good enough for a profile-only bridge.
2. If real collect-forward is much slower than synthetic, do not keep chasing
   renderer-only work. Split the wall with two rows:
   real MuZero `initial_inference` only, and real collect-forward at sims
   `1/2/4/8`. That separates model forward from CPU tree/search.
3. If scalar edge still halves throughput, keep scalar timesteps as an edge
   artifact only. Do not design the hot loop around scalar materialization.
4. If H100 barely beats L4, bigger GPU is not the next answer. The problem is
   host/search/orchestration shape.
5. If terminal/zero-mask or `to_play` semantics are unclear, fix those
   profile-only guardrails before running normal-death rows.

Recommended next sub-agent split:

- **Results ingester:** poll/summarize medium rows, update the registry and
  experiment log.
- **Canary hardener:** apply the real-consumer critique as docs/tasks first:
  uint8/direct contract, `to_play`, zero-mask roots, timing labels.
- **Search splitter:** plan the initial-inference-only and sim sweep rows.
- **World-model editor:** keep README/world_model/orchestration in sync after
  the medium table lands.

## Concrete Recommended Edits

Small edits that should happen soon:

1. Add `active_run_registry.md` or an equivalent top block in
   `orchestration.md`.
2. Move the real-consumer status block to the top of `world_model.md`.
3. In `task_board.md`, remove or supersede the unchecked "implement
   real-consumer canary" line.
4. Add a checked "first LightZero collect-forward smoke passed" line and an
   unchecked "medium rows ingestion" line near the top of `task_board.md`.
5. Link `subagent_real_consumer_impl_critique_20260521.md` from
   `README.md` and `task_board.md`.
6. Add a "Denominator glossary" to `README.md`:
   actual training, full-loop profile, hybrid profile, renderer-only,
   roots/sec, env steps/sec, learner iterations/hour.

No production-code edits are recommended from this docs review alone.
