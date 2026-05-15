# Handoff: Leaderboard-To-Training Loop

## Read This First

Start here:

1. `README.md`
2. `current_state.md`
3. `implementation_log.md`
4. `closed_loop_spec.md`
5. `gaps_and_tests.md`

## Plain Status

Current restart decision:

- The active lane is all-v2: `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`,
  `curvyzero-curvytron-control-v2`,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-checkpoint-events-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- The recreated all-v2 loop is proven at canary scale by
  `curvy-e2e-allv2-canary-20260515a`: trainer checkpoint -> v2 intake ->
  v2 tournament -> public leaderboard -> assignment materialization ->
  v2 control pointer -> same running trainer refresh -> provider-ok env rows.
- Do not use old non-v2 assignments/checkpoints as current launch inputs unless
  they are explicitly copied or rematerialized into the all-v2 lane.
- The next manifest should use `random_per_episode`, a control-volume refresh
  pointer, all-v2 objects from `src/curvyzero/contracts/curvytron.py`, and at
  least about `20%` blank/immortal pressure with some higher-immortal variants.
  Do not repeat the previous weak `5%` wall recipes as the main plan.
- The restart18 builder now fails closed for real launch shape: explicit
  ratings snapshot required, default opponent source is immutable assignment,
  assignment and refresh pointer refs are `control:`, and the default refresh
  interval is `2000` learner train iterations.
- The grouped submitter now rejects app-name mismatches. A manifest row, the
  selected app, and `curvytron_train_app_name()` must all agree on the current
  `-v2` trainer app.
- Latest all-v2 source audit: current v2 tournament storage contains only the
  all-v2 canary leaderboard. It is not a production-quality restart source
  because it has only `4` active rows and the top row is `iteration_0.pth.tar`.
  Any old leaderboard/champion source must be explicitly copied or rerated into
  v2, and every referenced checkpoint file must exist in `curvyzero-runs-v2`,
  before building a real restart18 launch.
- Modal operating pattern: use deployed apps for durable services, Volume JSON
  as truth, Dict/Queue for coordination only, stop stale detached apps, and
  avoid broad reload-dependent behavior.

The all-v2 canary proves wiring, not production-quality promotion or survival
improvement. Its promotion intentionally allowed provisional rows, and the
champion slot was `iteration_0.pth.tar`; that is acceptable for wiring proof
only. Survival is still partial/noisy and must be measured on a larger run.

New design direction:

- Modal Dict may hold the desired slot recipe for each training run id.
- That Dict can grow into a run-control record with both opponent slot settings
  and reward settings.
- Reward settings should be explicit: survival weight, bonus weight, and final
  outcome weight. Current safe profiles map to the existing reward variants:
  `sparse_outcome`, `dense_survival_plus_outcome`, and
  `survival_plus_bonus_no_outcome`.
- Reward settings are not like live slot preferences. Freeze the chosen reward
  recipe into launch or new-attempt artifacts and record its hash.
- That recipe is mutable operator intent, not training truth.
- The materializer turns the recipe plus a verified leaderboard snapshot into
  immutable `assignment.json`, `audit.json`, and refresh records on Volume.
- The trainer still consumes only assignment refs at safe boundaries.

Latest local rule:

- The tournament service defaults to a top-100 active pool.
- Mature rows below rank 100 are marked `retired`, not deleted.
- Retired rows stay in rating history and public leaderboard snapshots, but are
  not scheduled for future games.
- New or under-tested rows stay `provisional` even if their temporary rank is
  below 100, so they can still receive placement games.
- Pointer summaries count retired rows separately from provisional rows.

What was proven:

1. A leaderboard rating can be published into a public leaderboard snapshot.
2. A public leaderboard snapshot can be converted into an assignment.
3. An assignment can be written to the training Volume.
4. A trainer can consume that assignment and run.
5. That trainer writes checkpoints.
6. Tournament discovery can see those checkpoints.
7. Intake can enqueue and drain those checkpoints.
8. A tiny rating run can complete.
9. The new rating can be published into a new leaderboard.
10. A new assignment can be selected from that leaderboard.
11. A second trainer smoke can consume that new assignment and run.
12. A running trainer can use the direct assignment-refresh path:
    `opponent_assignment_refresh_ref` is checked at a collect boundary,
    every collector env is reset/proven, refresh JSONL is written, and later
    telemetry rows carry the new assignment.

The direct refresh path itself now has local tests and a recreated all-v2 Modal
same-trainer proof. Remaining automation work is launch hardening, run-control
policy, production-scale tournament gates, cleanup, and survival measurement.

Latest tiny remote smokes:

- `leaderboard-pointer-repair` republished
  `current:curvytron-latest212-smoke-20260513` from immutable Volume snapshots.
- `arena-oneframe-public-publish-smoke-20260514b /
  elo-oneframe-public-publish-smoke-20260514b` ran a two-checkpoint one-frame
  rating smoke with `active_pool_limit=100`.
- `curvytron-oneframe-public-smoke-20260514b` published that smoke as a
  non-diagnostic public leaderboard with two active rows.
- `inspector-online-continuation-proof-20260514c /
  elo-oneframe-online-continuation-proof` proved the bounded online path:
  old 3 checkpoint refs -> submit 3 new refs -> repair missing Queue events
  from manifest -> stale-claim takeover for the proof -> continued rating wrote
  `round-000001` with 6 rows, 9 pairs, 9 games, and one-frame timing.
- `refresh-e2e-smoke-20260514/train-refresh-d` proved direct running-trainer
  assignment refresh: assignment A at launch, assignment B pending, one
  `applied` refresh event, all-env ready proof for B, and telemetry rows that
  switched from A to B after LightZero startup collection.

## Coach / Tournament Job Boundary

Coach owns training:

- launch/runtime config;
- exact checkpoint writing;
- immutable `assignment.json` consumption;
- eval/GIF using the same assignment as training;
- deciding whether leaderboard evidence is good enough to steer training.

The long-running tournament job owns tournament observability:

- checkpoint discovery from the training Volume;
- subscriber/intake Dict and Queue coordination;
- tournament games, GIF samples, Elo/rating artifacts, pair history, and
  scheduler state;
- public leaderboard snapshot publishing.

The clean exchange is:

```text
Coach -> tournament job: exact checkpoint refs on the training Volume
tournament job -> Coach: public leaderboard snapshot, then immutable assignment.json
```

The trainer must not poll live tournament state or Modal Dict while learning.
Modal Dict/Queue are coordination helpers. Volume JSON snapshots are the
durable truth.

Hard boundary:

- No live Elo reads inside the trainer step loop.
- No Modal Dict or Queue reads inside env steps or learner updates.
- A small run-control Dict read is allowed only at a clean collect boundary
  once that path is implemented; the current working path uses a direct pending
  assignment ref instead.
- No tournament browser/API reads inside the trainer step loop.
- Assignment refresh is allowed only at a clean boundary: launch, resume,
  explicit operator refresh, or a checkpoint boundary if that is implemented
  deliberately.

Subscriber watch rule:

- run IDs or run prefixes are live watches;
- explicit checkpoint refs are frozen seeds;
- continuation into an existing rating run must be explicit
  (`continue_from_latest=True`) and must keep the full known checkpoint pool.

Tournament service rule:

- `intake-seed` is the admin/configure path for scheduler/evaluator policy.
- `tournament-submit` / `intake-submit` is the normal candidate path.
- Submit accepts candidate checkpoint refs or run IDs, not scheduler knobs.
- Drain uses the manifest policy by default. Ad-hoc rating overrides are an
  explicit internal/operator escape hatch, not the product contract.
- When continuation adds new refs, old checkpoint ids must be reused from
  `latest.json`. Otherwise old ratings and pair history can detach from the
  same checkpoint. This now has local regression coverage.
- Non-detached `modal run` is not a safe parent for background tournament
  workers. If a launch spawns child game/rating work and should keep running
  after the command returns, use `modal run --detach` or wait for the children
  to finish. Verify `latest.json` advanced and completed game summaries exist;
  do not count "round scheduled" as success.

Checkpoint discovery foot gun:

- discovery must scan `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
- DI-engine can create timestamped folders like `lightzero_exp_260513_123802`;
- scanning only `train/lightzero_exp/ckpt/...` misses real checkpoints.

Publication boundary:

- The tournament job owns public leaderboard snapshots and the compact live pointer;
- Coach owns assignment selection, assignment refresh timing, and launch policy;
- Modal Dict may point at the latest public snapshot, but the durable handoff is
  the Volume JSON snapshot plus a selected immutable `assignment.json`.

## Important Smoke Runs

Assignment write:

```text
leaderboard-assignment-smoke / assignment-smoke
leaderboard-assignment-smoke-b / assignment-smoke-b
```

Training smokes:

```text
leaderboard-assignment-train-smoke-20260513a
leaderboard-assignment-train-smoke-20260513c
```

Tiny tournament:

```text
arena-closed-loop-smoke-20260513b / elo-closed-loop-smoke
```

Smoke leaderboard:

```text
curvytron-closed-loop-smoke-20260513b
```

## Tests That Passed

Latest focused local verification:

```text
uv run pytest tests/test_curvytron_checkpoint_tournament.py -q
139 passed, 12 skipped

uv run pytest \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_opponent_leaderboard.py \
  tests/test_lightzero_checkpoint_opponent_provider.py -q
70 passed, 1 skipped

uv run ruff check \
  src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py \
  src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py \
  src/curvyzero/training/opponent_leaderboard.py \
  src/curvyzero/training/lightzero_checkpoint_opponent_provider.py \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py \
  tests/test_opponent_leaderboard.py \
  tests/test_lightzero_checkpoint_opponent_provider.py
All checks passed

git diff --check
clean
```

Latest extra local coverage:

- `tests/test_curvytron_checkpoint_intake_repair.py`
- `tests/test_curvytron_tournament_scheduler_fairness.py`
- `tests/test_curvytron_opponent_leaderboard_pointer_repair.py`
- no-active-row leaderboard publish guard in
  `tests/test_curvytron_checkpoint_tournament.py`
- one-frame leaderboard publish guard in
  `tests/test_curvytron_checkpoint_tournament.py`

Latest tiny remote smoke coverage:

- pointer repair rebuilt the live Dict pointer from immutable leaderboard
  snapshots;
- one-frame rating recorded `decision_source_frames=1`,
  `decision_ms=16.666666666666668`, and `active_pool_limit=100`;
- one-frame leaderboard publish wrote snapshot/latest refs and moved the live
  pointer with `active_count=2`.

Combined focused tournament check:

```text
uv run pytest \
  tests/test_curvytron_checkpoint_tournament.py \
  tests/test_curvytron_checkpoint_intake_repair.py \
  tests/test_curvytron_tournament_scheduler_fairness.py \
  tests/test_curvytron_opponent_leaderboard_pointer_repair.py -q
151 passed, 12 skipped
```

Earlier focused smoke regression:

```text
uv run pytest \
  tests/test_lightzero_checkpoint_opponent_provider.py \
  tests/test_opponent_leaderboard.py \
  tests/test_opponent_registry.py \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_artifact_writer_stores_assignment_and_audit \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_opponent_assignment_ref_resolves_to_existing_mixture_contract \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_resolves_assignment_ref \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_function_accepts_assignment_ref_and_resolves_command \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_checkpoint_eval_poller_command_rejects_assignment_and_inline_mixture \
  tests/test_curvytron_live_checkpoint_eval_plumbing.py::test_local_launcher_passes_gif_config_to_poller_and_prints_enabled \
  tests/test_curvytron_checkpoint_tournament.py::test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer -q
```

Result:

```text
27 passed
```

Lint/compile:

```text
ruff: all checks passed
py_compile: passed
```

## Code Added Or Changed

New:

- `src/curvyzero/tournament/checkpoint_intake_service.py`
- `src/curvyzero/training/opponent_leaderboard.py`
- `tests/test_opponent_leaderboard.py`
- `tests/test_lightzero_checkpoint_opponent_provider.py`
- `scripts/materialize_curvytron_leaderboard_assignment.py`

Changed:

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
- `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py`
- `tests/test_curvytron_checkpoint_tournament.py`
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`

## Important Bug Found And Fixed

The trainer failed when a leaderboard-selected checkpoint had a different
LightZero support head shape than the current trainer default.

Fix:

```text
lightzero_checkpoint_opponent_provider.py
```

now infers reward/value support sizes from checkpoint state dict head shapes
before building the MuZero model.

## What Still Needs Work

Do next:

1. Audit the exact larger launch manifest for all-v2 refs, refresh pointer,
   reward/slot settings, observation surface, checkpoint cadence, and no stale
   non-v2 inputs.
2. Run a production-shaped but bounded tournament/assignment validation with
   real active-row gates; do not use the canary's provisional relaxations as a
   production-quality leaderboard gate.
3. Keep the background eval/GIF poller path separate unless the launch depends
   on it; the all-v2 canary proves training refresh, not broad eval/GIF.
4. Clean or hide stale apps/arenas only after confirming they are not needed for
   current proof, old champion extraction, or diagnostics.
5. Launch the next deliberately named larger run only from all-v2 inputs, then
   measure survival with numbers before claiming learning improvement.

Recent focused fixes after critique:

- the real checkpoint eval/GIF poller now accepts and forwards
  `opponent_assignment_ref`;
- the assignment artifact writer has direct regression coverage;
- the pure assignment selector excludes retired rows when provisional rows are
  allowed, and sorts rows before choosing the champion.
- the trainer-facing leaderboard publisher refuses provisional ratings unless
  explicitly allowed, commits the Volume before moving the live pointer, and
  leaves the pointer unchanged if commit fails.
- checkpoint intake now has regression coverage for live run-id watches versus
  frozen explicit checkpoint refs.
- checkpoint intake drain now has regression coverage that `spawn_if_existing`
  does not touch an existing rating run unless `continue_from_latest=True`.
- continuation specs use the full `seen_checkpoint_refs` pool, so adding a new
  checkpoint does not silently drop older rated checkpoints.
- `stable_slots_v1` is now the production-direction materializer. It writes
  ordinary assignment entries, uses nested `recency.latest_for_run`, supports
  blank or wall-avoidant immortal sentinel entries, dedupes checkpoint id/ref,
  and records per-slot evidence in the audit.

## Do Not Overstate

Do not say production closed-loop training is done.

Say this:

```text
The deployed all-v2 feedback loop works at canary scale. It proves wiring, not
production-quality ranking or survival improvement. Production automation still
needs launch-manifest audit, active-row quality gates, cleanup, and scale tests.
```
