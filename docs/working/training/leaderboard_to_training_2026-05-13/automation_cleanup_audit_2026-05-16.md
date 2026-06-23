# Automation Cleanup Audit - 2026-05-16

Scope: read-only audit of the current CurvyTron multi-trainer/tournament loop.
No runtime state was changed. One read-only Modal inventory was checked during
this audit.

## Short Answer

The current path is mostly automated after it is seeded and after trainers are
launched. It is not fully operator-free.

The active deployed lane is all-v2:

- App `curvyzero-checkpoint-tournament-v2`, deployed, app id
  `ap-u9u1TpA2BPQjyviJtR9bj1`, `318` tasks at read time.
- App `curvyzero-lightzero-curvytron-visual-survival-train-v2`, deployed, app
  id `ap-FTBsuB0JXLZoA5MYhadNYv`, `30` tasks at read time.
- App `curvyzero-curvytron-gif-browser-v2`, deployed, app id
  `ap-efdj9SpiGjGwC4dBdSClSQ`, `1` task at read time.
- Active Curvy volumes are only `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`.
- Active Curvy coordination objects are
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-opponent-leaderboard-live-v2`, and queue
  `curvyzero-curvytron-checkpoint-events-v2` with `2` partitions and `931`
  queued items at read time.

There is also one live detached tournament worker:
`ap-X0DUzy51Yzb3wWIBUx42jf`, app `curvyzero-checkpoint-tournament-v2`,
created `2026-05-16 05:42:41-04:00`, `5` tasks. Treat it as current live work
unless a targeted log/artifact check proves otherwise.

## What Is Automated Now

### Checkpoint Subscriber And Intake

Automated/scheduled:

- `curvytron_checkpoint_intake_subscriber_tick` in
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` is scheduled
  with `modal.Period(seconds=DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS)`.
  `DEFAULT_CHECKPOINT_INTAKE_SCAN_SECONDS = 10` in
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_settings.py`.
  It calls `curvytron_checkpoint_intake_tick.local({})`.
- `curvytron_checkpoint_intake_drain_tick` in the same file is also scheduled
  every `10` seconds. It loops over active intake manifests, calls
  `curvytron_checkpoint_intake_drain.local(...)`, and passes
  `spawn_rating=True`.

Seed/config still required:

- `curvytron_checkpoint_intake_seed` creates the active manifest, scan policy,
  rating defaults, Queue partition, and optional first rating spawn.
- `curvytron_checkpoint_intake_submit` accepts later exact refs/run ids, but it
  intentionally rejects scheduler knobs through
  `checkpoint_intake_service.validate_submit_payload`.

Plain meaning: once an active intake manifest exists, scanning and draining are
scheduled. Creating or changing that manifest is still an operator/controller
action.

### Rating Continuation

Automated through intake drain, not by a separate cron:

- `curvytron_checkpoint_intake_drain_tick` schedules rating continuation by
  calling `curvytron_checkpoint_intake_drain` with `spawn_rating=True`.
- `curvytron_checkpoint_intake_drain` decides `continue_from_latest` from the
  manifest defaults or live run-id watch, repairs missing queue events, claims a
  rating run, and spawns `curvytron_rating_loop`.
- `curvytron_rating_loop` writes config, starts
  `curvytron_rating_provisional_loop.spawn(...)`, and runs new rounds from the
  latest existing state via `_rating_loop_start_state`.
- `curvytron_rating_round`, `curvytron_rating_reduce`,
  `curvytron_rating_progress`, and `curvytron_rating_provisional` are deployed
  Modal functions, but only the intake ticks/provisional loop are scheduled.

Plain meaning: continuation is service-driven once the intake manifest is live.
Manual `mode=rating`, `mode=reduce`, or `mode=provisional` are repair/debug
paths, not the intended current loop.

### Training-Candidate Refresh

Automated/scheduled:

- `curvytron_training_candidate_refresh_tick` in
  `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` is scheduled
  with `modal.Period(seconds=TRAINING_CANDIDATE_REFRESH_SECONDS)`.
- `TRAINING_CANDIDATE_REFRESH_SECONDS` comes from
  `DEFAULT_CURVYTRON_TRAINING_CANDIDATE_REFRESH_SECONDS = 30 * 60` in
  `src/curvyzero/contracts/curvytron.py`.
- The tick calls `curvytron_training_candidate_refresh`, records state in
  `checkpoint_intake_state`, and uses the shared defaults:
  current tournament/rating ids, assignment-bank run/attempt, and the three
  control refresh pointers from `src/curvyzero/contracts/curvytron.py`.

Deployed status is visible in code and runtime inventory:

- The scheduled function lives in deployed app
  `curvyzero-checkpoint-tournament-v2`.
- `NOW.md` says the scheduled controller fired and published generation 12
  from `round-000011`; later manual proof refreshed generation 20/21 from
  `round-000029`.

Plain meaning: this is the clearest automated piece now.

### Trainer Assignment Refresh

Automated inside each running trainer, if the run was launched with a refresh
pointer:

- `lightzero_curvytron_visual_survival_*` train functions accept
  `opponent_assignment_refresh_interval_train_iter` and
  `opponent_assignment_refresh_ref`.
- `_install_lightzero_opponent_assignment_refresh_hook` patches LightZero
  collection. It checks the pointer at coarse train-iteration buckets, loads a
  pending assignment, and applies it through
  `_apply_opponent_assignment_refresh_to_collector_env`.
- `_opponent_assignment_refresh_due`,
  `_opponent_assignment_refresh_bucket`, and
  `_opponent_assignment_refresh_reset_param` are the small core helpers.
- `CURVYTRON_ASSIGNMENT_REFRESH_INTERVAL_TRAIN_ITER = 2_000` in
  `src/curvyzero/contracts/curvytron.py`; the manifest builder passes this for
  refresh-enabled runs.

Not global/scheduled:

- There is no Modal schedule that forces every trainer to refresh.
- A trainer without `opponent_assignment_refresh_ref` or with interval `0` will
  not consume tournament-fed assignments.

Plain meaning: trainer refresh is automated per launched job, not a fleet-wide
service.

### GIF And Browser Visibility

Deployed browsers:

- Tournament browser: `curvytron_tournament_browser` is an ASGI app inside
  deployed app `curvyzero-checkpoint-tournament-v2`.
- GIF browser: `gif_browser` in
  `src/curvyzero/infra/modal/curvytron_gif_browser.py` is an ASGI app inside
  deployed app `curvyzero-curvytron-gif-browser-v2`.
- `curvytron_gif_browser_hide_run` can hide a trainer run from the GIF picker.

Visibility mechanics:

- Tournament visibility is marker-file based. `_write_tournament_marker` writes
  `show_in_tournament_browser.flag`; `_update_tournament_visibility` lists,
  hides, shows, or `hide_except`s those markers.
- `curvytron_rating_loop`, `curvytron_checkpoint_intake_seed`, and tournament
  run paths write tournament markers automatically.
- Manual visibility cleanup is still through `mode=visibility`.

GIF production:

- Tournament defaults still say `DEFAULT_SAVE_GIF=True`,
  `DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR=5`, and `DEFAULT_GIF_FPS=800.0`.
- The current fast proof lane intentionally has persisted `save_gif=false`, so
  it will not emit tournament GIF samples even though the code default supports
  them.
- Trainer self-play GIFs are produced by
  `lightzero_curvytron_visual_survival_checkpoint_selfplay_gif`, usually
  scheduled by `lightzero_curvytron_visual_survival_checkpoint_eval_poller` or
  checkpoint hooks when `background_gif_enabled=True`.

Plain meaning: browsers are deployed. Visibility is partly automatic, but
cleanup/hiding is manual. Current fast tournament visual evidence is off unless
the persisted intake/rating config enables GIFs.

## What Still Requires Manual Operator Commands

- Deploying or redeploying apps:
  `modal deploy -m curvyzero.infra.modal.curvyzero_checkpoint_tournament`,
  trainer app, and GIF browser app.
- Building a launch manifest:
  `scripts/build_curvytron_tonight18_manifest.py`.
- Submitting trainer rows:
  `scripts/submit_curvytron_survivaldiag_manifest.py`, which uses
  `modal.Function.from_name(...).spawn(...)`.
- Seeding the intake service for a fresh arena:
  `mode=intake-seed`.
- Exact manual candidate submission outside the watched run ids:
  `mode=tournament-submit` or `mode=intake-submit`.
- Manual repair/debug:
  `mode=intake-status`, `mode=intake-drain`, `mode=progress`,
  `mode=provisional`, `mode=reduce`, `mode=leaderboard-pointer-repair`, and
  `scripts/curvytron_tournament_debug_bundle.py`.
- Manual visual cleanup:
  `mode=visibility` for tournament arenas and
  `curvytron_gif_browser_hide_run` for GIF runs.
- Manual app cleanup:
  `modal app stop ...` for detached proof workers after artifact/log checks.
- Manual status reads:
  `curvyzero.infra.modal.lightzero_curvytron_run_status` currently shows up as
  short-lived stopped helper apps in runtime inventory, not as the main current
  deployed service.

## Confusing Stale Or Proof Lanes

Do not delete these from this audit. Just stop promoting them as current.

- Older docs in this folder still name many superseded arenas:
  `curvy-v2real18-*`, `curvy-v2refresh18*`, `curvy-looplive-*`,
  `curvy-restart18-source-rerate-*`, `curvy-r18fresh-live-20260516a`, and
  `curvy-r18fresh-live-bounded-20260516a`.
- `post_purge_current_truth_2026-05-16.md` says pre-`2026-05-16 00:54 EDT`
  v2 state was purged and must be treated as historical.
- Current code defaults point at
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`. That is the current arena/rating
  pair in `src/curvyzero/contracts/curvytron.py`.
- Runtime inventory no longer shows old non-v2 Curvy volumes. Current Curvy
  storage is the three v2 volumes only.
- The one active detached tournament app,
  `ap-X0DUzy51Yzb3wWIBUx42jf`, looks like current proof/live work because it is
  a v2 tournament app created after the current proof lane started. Do not stop
  it from a name-only cleanup.
- The many stopped `curvyzero-lightzero-curvytron-run-status` apps are
  observation helpers. They are noise in Modal app lists, not current durable
  services.
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  is still the current trainer app despite the stale "stacked_debug" filename.
  The module docstring admits this. The name is now misleading.
- Current fast proof tournament config has `save_gif=false`. That is a proof
  lane setting, not proof that GIF capture is broken.
- `curvyzero-lightzero-curvytron-visual-survival-eval` appears in older
  cleanup notes, but was not present as a current deployed app in the latest
  app inventory. Do not keep calling it part of the current path unless it is
  redeployed or a current owner says it is needed.

## Minimal Cleanup Plan

1. Declare one current path in one file.
   Use `src/curvyzero/contracts/curvytron.py` as the code source of truth and
   one short doc as the human source of truth. Everything else links to it or
   says "historical".

2. Rename the current trainer module.
   Move or alias
   `lightzero_curvyzero_stacked_debug_visual_survival_train.py` to a plain name
   like `curvytron_train.py`. Keep a temporary import shim only if needed for
   deployed command compatibility, and mark it deprecated.

3. Split proof lanes from current lanes in docs and dashboards.
   In `README.md`, `NOW.md`, and `OPERATING_PATTERN.md`, keep only:
   current arena/rating id, current app names, current volumes, current
   scheduled functions, current launch command, current debug command.
   Move old arena lists into an `archive/` subfolder or prepend
   "historical, do not launch from this".

4. Make one launch command the obvious path.
   Wrap manifest build plus submit plus intake seed in a single
   `scripts/launch_curvytron_current.py --dry-run/--apply` command. It should
   print the app names, volumes, tournament id, rating id, refresh pointers,
   and whether tournament GIFs are enabled before it does anything.

5. Make one status command the obvious path.
   Promote the debug/status bundle into one command that reports:
   trainer rows, latest checkpoints, intake manifest, queue length, latest
   rating, scheduled training-candidate generation, trainer refresh proof, and
   GIF/browser config. Stop making operators stitch this together from repeated
   Modal helpers.

6. Make visual intent explicit.
   Add a current-lane field such as `tournament_gif_enabled` to launch output
   and status output. If a lane is marked "current visual arena", fail launch
   when persisted `save_gif=false`.

7. Hide, do not delete, old visible arenas first.
   Use `mode=visibility --visibility-action hide_except` after confirming the
   active arena id. Deletion can be a later exact-path storage cleanup.

8. Add a cleanup checklist for detached apps.
   Only stop app ids after checking logs and the rating/artifact ids they are
   writing. Never stop based on app description alone.

## Code-Level Simplifications

- Collapse the local-entrypoint modes in
  `curvyzero_checkpoint_tournament.py`. Keep current operator modes; move
  older smoke/debug modes to explicit scripts or test helpers.
- Make scheduled-service functions tiny wrappers around pure controller
  functions. `curvytron_checkpoint_intake_drain` is carrying too many repair,
  claim, queue, and spawn branches in one body.
- Separate "seed intake", "submit refs", and "service tick" into different
  modules. The current file mixes web UI, rating, intake, visibility,
  leaderboard publish, and training-candidate writes.
- Remove fresh-path fallback aliases for observation fields. The current
  policy surface should be named once:
  `browser_lines + simple_symbols`, `cpu_oracle`, one source frame.
- Make trainer refresh impossible to miss in manifests: require
  `opponent_assignment_refresh_ref` for any row whose opponent source is
  `assignment`, unless the row is explicitly marked `static_no_refresh`.
- Rename "training-candidate" to something plainer in operator docs, such as
  "trainer assignment export". The code can keep the old schema id if changing
  artifacts is too expensive.
- Keep `materialize_curvytron_leaderboard_assignment.py` as a local migration
  or archive tool, not part of the current happy path. The deployed controller
  now does the current assignment export.
- Add a startup self-check in deployed apps that prints current app/volume/id
  contract values once. This makes bad env overrides obvious in logs.
