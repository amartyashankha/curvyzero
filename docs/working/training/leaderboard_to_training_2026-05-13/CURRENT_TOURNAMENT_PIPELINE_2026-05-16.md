# Current CurvyTron Tournament Pipeline

Date: 2026-05-16.

This is the current path. Older dated docs in this directory are historical
unless they are cited here.

## General Contracts

- `docs/working/training/curvytron_feedback_loop/POLICY_OBSERVATION_CONTRACT.md`
- `docs/working/training/curvytron_feedback_loop/OBSERVABILITY_CONTRACT.md`

## Current IDs

- Current tournament arena: `curvy-r18fresh-live-bounded-dsf1-20260516b`
- Current rating run: `elo-r18fresh-live-bounded-dsf1-20260516b`
- Training-candidate assignment bank run:
  `curvy-r18fresh-training-candidate-assignments`
- Training-candidate assignment bank attempt:
  `try-r18fresh-training-candidate-assignments`
- Current GIF run prefix default: `curvy-r18fresh-`

These ids are also the defaults in `src/curvyzero/contracts/curvytron.py`.

For a local static readout of the current code contract:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.curvyzero_checkpoint_tournament --mode current
```

This prints app names, Volume names, current tournament/rating ids, scheduled
tick intervals, manual steps, and tournament defaults. It does not spawn rating
or trainer work.

## v2 Modal Names

Apps:

- Trainer: `curvyzero-lightzero-curvytron-visual-survival-train-v2`
- Tournament/intake/controller: `curvyzero-checkpoint-tournament-v2`
- GIF browser: `curvyzero-curvytron-gif-browser-v2`

Volumes:

- Runs/checkpoints: `curvyzero-runs-v2`
- Tournaments/ratings: `curvyzero-curvytron-tournaments-v2`
- Control/assignments: `curvyzero-curvytron-control-v2`

Coordination objects:

- Intake Dict: `curvyzero-curvytron-checkpoint-intake-v2`
- Intake Queue: `curvyzero-curvytron-checkpoint-events-v2`
- Trainer-facing leaderboard Dict:
  `curvyzero-curvytron-opponent-leaderboard-live-v2`

Non-v2 names and pre-purge v2 objects are historical.

## Current Loop

```text
trainers write checkpoints to curvyzero-runs-v2
-> scheduled subscriber scans active run-id watches
-> scheduled drain claims events and spawns/continues rating
-> rating writes latest.json under the current tournament/rating id
-> training-candidate refresh publishes a trainer-facing snapshot
-> refresh writes immutable assignments and three recipe refresh pointers
-> trainers launched with those pointers consume the new assignment at a coarse boundary
```

Volume JSON is durable truth. Dicts and Queues coordinate current work, but do
not own history.

## Automated Now

- Scheduled intake subscriber: scans active manifests/run-id watches for new
  checkpoints.
- Scheduled intake drain: drains checkpoint events and starts rating work.
- Rating continuation: the drain continues from current rating state instead of
  restarting from scratch.
- Scheduled training-candidate refresh tick: every 30 minutes, reads the
  current arena/rating defaults and writes the trainer-facing snapshot,
  assignment files, and refresh pointers.
- Trainer assignment refresh: each trainer launched with a refresh pointer and
  nonzero refresh interval checks the pointer at coarse train-iteration
  boundaries and applies newer immutable assignments.

## Manual Now

- Launching trainer batches.
- Initial seeding/config of active intake manifests, run-id watches, scheduler
  policy, and first assignments.
- Deploying/redeploying apps.
- Choosing or auditing launch manifests.
- Cleanup/purge/hide operations for old apps, arenas, Volume trees, Dicts,
  Queues, and browser visibility.
- Repair/debug commands such as direct rating, reduce, provisional, visibility,
  or exact submit outside the normal scheduled path.

Current-lane operator modes now default to the shared current arena/rating when
`--tournament-id` is omitted. Fresh one-off experiments should pass explicit
new ids.

## GIF Truth

- Source defaults are GIF-on for tournament generation:
  `DEFAULT_SAVE_GIF=True` and `DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR=5`.
- New tournament GIFs use `800` fps with `1ms` minimum frame duration.
- The current active fast feedback arena was seeded with persisted
  `save_gif=false`; its completed rounds remain no-GIF. `gif_sample_games_per_pair=5`
  in that config is inert while `save_gif=false`.
- The intake config has been corrected for future current arenas. Verify the
  persisted intake/rating artifact, not only source defaults.
- Old completed rounds that were seeded no-GIF remain no-GIF unless rerun or
  regenerated.

## Validation State

Mechanical loop: proven.

- Generation 9: tournament output from the current corrected live lane became
  training-candidate assignments and all `18/18` running trainers consumed the
  new assignment shas with provider-ok frozen-checkpoint rows.
- Generation 10: a later tournament snapshot refreshed assignments again; all
  `18/18` running trainers consumed the gen10 shas with provider-ok rows.
- Generation 12: the deployed scheduled refresh tick published from a later
  tournament round; all still-running `15/15` trainers consumed the gen12 shas
  with provider-ok rows. The other 3 trainers had already completed.

Learning quality: modest.

- The `r18fresh` batch found better mid-run checkpoints in all `18/18` rows.
- Latest checkpoint quality often regressed. Current readout is roughly first /
  best / latest survival `159.9 / 246.0 / 175.4`, with only `10/18` latest
  checkpoints above first and only `4/18` latest checkpoints near their own
  best.
- Treat current tournament output as useful for preserving candidate policies,
  not as proof that the latest trainer checkpoints are strong.

## Perspective And Metadata Truth

- Training uses `learner_seat_mode=random_per_episode` by default. The learner
  sees the controlled-player view for whichever physical player it controls in
  that episode; fixed player-0/player-1 modes are diagnostics.
- Tournament eval uses `seat_order_mode=balanced_random`. Each policy receives
  the controlled-player view for its actual physical seat in that game.
- Fresh checkpoint artifacts are expected to carry their policy observation
  surface: `policy_trail_render_mode=browser_lines`,
  `policy_bonus_render_mode=simple_symbols`, and
  `policy_observation_backend=cpu_oracle`.
- Actual tournament policy loading now fails if that metadata is missing or if
  it names a non-current surface/backend. Older helper/display paths may still
  normalize raw refs for labels and scheduling, but the eval path must not
  silently run a policy on an assumed surface.

## Cleanup Scope

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  is still a large Modal orchestration file, about 13k lines. It mixes app
  setup, CLI defaults, training invocation, checkpoint metadata, assignment
  refresh, resume hooks, background eval/GIF, and status tooling.
- Do not start a broad split while validating live learning behavior. The safe
  cleanup path is staged: first extract current defaults/profile construction
  and manifest recipe rules, then split Modal entrypoint wrappers from plain
  helper modules. Keep behavior-preserving tests around each extraction.

## Historical Markers

- `curvy-r18fresh-live-bounded-20260516a` /
  `elo-r18fresh-live-bounded-20260516a` is historical: it completed, but lacked
  explicit `decision_source_frames=1` metadata and was refused by the
  training-candidate controller.
- `curvy-r18fresh-live-20260516a` /
  `elo-r18fresh-live-20260516a` is historical/dirty: overlapping workers and
  unbounded all-pairs artifacts polluted that tree.
- Older proof lanes, top100 lanes, v2real18 lanes, pre-purge refs, and non-v2
  app/volume names are audit history only.
# Current Tournament Pipeline

General contracts now live outside this archived planning folder:

- `docs/working/training/curvytron_feedback_loop/POLICY_OBSERVATION_CONTRACT.md`
- `docs/working/training/curvytron_feedback_loop/OBSERVABILITY_CONTRACT.md`
