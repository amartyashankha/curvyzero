# NOW: Coach / Tournament Control Panel

Last organized: 2026-05-15.

Read this first. This page is the current plain truth, not a full history.

## 2026-05-15 All-V2 Reset

- Current operator decision completed: deleted and recreated the active v2
  storage/control lane, then redeployed the CurvyTron apps against that lane.
  Do not launch new training from the old hybrid namespace.
- Current app names:
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`,
  `curvyzero-checkpoint-tournament-v2`, and
  `curvyzero-curvytron-gif-browser-v2`.
- Current Modal Volumes, all opened with `version=2`:
  `curvyzero-runs-v2`, `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`.
- Current Modal Dict/Queue objects:
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-checkpoint-events-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Historical proof in old/non-v2 or hybrid storage remains useful only as
  evidence about the mechanics. It is not a valid launch input for the new
  all-v2 lane unless explicitly copied or rematerialized there.
- The shared source of truth is
  `src/curvyzero/contracts/curvytron.py`. If a script or app has a different
  default, treat that as a bug.
- Verified at about 2026-05-15 14:10 EDT: v2 trainer/tournament/GIF apps are
  deployed; the old non-v2 Curvy trainer/tournament deployments are stopped.

## 2026-05-15 E2E Validation Phase

- Current phase: deployed end-to-end validation and hardening. Stop broad
  design churn unless it removes an immediate proof or launch blocker.
- Latest source-leaderboard audit: the recreated all-v2 tournament volume
  currently contains only the tiny all-v2 canary arena
  `curvy-e2e-allv2-canary-live-20260515a` and one public leaderboard
  `e2e-allv2-canary-live-r3-20260515a`. That snapshot is valid wiring proof
  only: `25` rows, `4` active rows, relaxed/provisional gates, and rank 1 is
  `iteration_0.pth.tar`. Do not use it as the production source for restart18.
- Current launch blocker is now plain: we need either a fresh
  production-shaped all-v2 rating/leaderboard snapshot, or an explicit
  rematerialization/rerate step that copies selected old strong checkpoint refs
  into v2 storage and rates them there. Old/non-v2 snapshots are not launch
  inputs until their referenced checkpoint files exist in `curvyzero-runs-v2`.
- Fresh all-v2 canary is proven at wiring scale:
  `curvy-e2e-allv2-canary-20260515a` /
  `try-e2e-allv2-canary-20260515a`; poller
  `fc-01KRPDXH9FW4DNNMW41G9XC3EY`; trainer
  `fc-01KRPDXHDF4C5RS17DR1Z18EVX`. The seed checkpoint was copied into
  `curvyzero-runs-v2` before launch, and the submitter now opens the v2 control
  volume with the shared VolumeFS-version helper. Proof result: the trainer
  wrote checkpoints through at least `iteration_5300.pth.tar`; v2 intake and
  tournament completed `round-000003` with `18/18` games, `0` failures, and
  `stable=true`; promotion wrote assignment sha
  `0597bceb176580d19d658fd513f752a47a7d4e0f5c9094d5c0f58f60f422c2e0`; the same
  running trainer applied that sha at train iter `5061` with all `8` envs ready;
  a later refresh check stayed on the promoted sha at train iter `5372`;
  `env_steps.jsonl` had `1836` provider-ok rows using the promoted sha on the
  latest post-refresh fetch.
- What this still does not prove: production-quality promotion, a learned
  nonzero checkpoint as champion, background eval/GIF poller behavior, or
  survival improvement. The canary used relaxed/provisional gates and is a
  wiring proof only.
- Post-proof focused local regression passed; latest launch-readiness rerun
  after fail-closed manifest/app-guard patches:
  `tests/test_curvytron_tonight18_manifest.py`,
  `tests/test_curvytron_survivaldiag_submitter.py`,
  `tests/test_curvytron_shared_contracts.py`, and
  `tests/test_promote_curvytron_rating_round.py` -> `26 passed`.
- Broader E2E-adjacent local regression passed after the all-v2 proof and
  fail-closed launch patch:
  checkpoint tournament, live eval plumbing, source-state env, opponent
  leaderboard, opponent mixture, opponent registry, and GIF browser slices ->
  `343 passed, 24 skipped`.
- Current deployed trainer/tournament namespace after the latest redeploy is
  all-v2: `curvyzero-runs-v2`, `curvyzero-curvytron-control-v2`, and
  `curvyzero-curvytron-tournaments-v2`.
- Modal VolumeFS version check, 2026-05-15 14:08 EDT: verified with
  `Volume.from_name(name, version=2).info()` for `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`. Do not infer filesystem version from the
  name suffix alone; the code uses the explicit shared-contract map.
- Important correction from the latest remote smoke: the old
  `curvy-e2e-clean-canary-long-20260515c` assignment points at checkpoints in
  `curvyzero-runs`, so it is historical proof only. It must not be used as an
  input to the new all-v2 lane unless rematerialized into v2 storage.
- Pre-reset current-namespace poller smoke passed after writing a tiny control
  assignment in the then-current control volume that pointed at an actual
  checkpoint in `curvyzero-runs-v2`
  (`optimizer-gpuobs-canary-20260515/.../iteration_0.pth.tar`). Result:
  `curvy-e2e-poller-v2-namespace-smoke-20260515a` completed with
  `status=completed`, `seen_count=0`, and no assignment/checkpoint resolution
  error. This is useful historical evidence; the recreated all-v2 lane was
  later proven by `curvy-e2e-allv2-canary-20260515a`.
- Current proof status: trainer writes a fresh checkpoint -> intake/subscriber
  admits it -> tournament rates it -> public leaderboard/assignment is updated
  -> the same running trainer refreshes to the new assignment -> later trainer
  telemetry proves the refreshed opponent loaded and was used. This passed for
  the recreated all-v2 lane on `curvy-e2e-allv2-canary-20260515a`.
- The old `controlrun2` proof is real, but it predates the latest seat/slot
  cleanup. Treat it as a historical template. The current active proof is the
  all-v2 canary `curvy-e2e-allv2-canary-20260515a` above.
- Fresh restart/canary runs should use fresh all-v2 tournament/rating ids. The
  v2 queue was recreated empty, but fresh ids still keep proof artifacts
  readable and avoid accidental reuse of stale durable tournament folders.
- Canary shape: frequent checkpoints, short enough to observe quickly, deployed
  trainer/tournament apps, `control:` assignment pointer,
  `commit_on_checkpoint=true`, and `random_per_episode` learner seat.
- Pass/fail must stay written here and in `FULL_LOOP_PROOF.md` with exact run
  ids, assignment shas, checkpoint refs, and telemetry counts.
- First canary
  `curvy-e2e-clean-canary-20260515a` proved checkpoint production, live run-id
  intake, tournament rating, public leaderboard publication, assignment
  materialization, and control-pointer rewrite. It missed same-process refresh
  because the pointer was rewritten after the last refresh event and just before
  `max_train_iter=4000`.
- Long canary:
  `curvy-e2e-clean-canary-long-20260515c` /
  `try-e2e-clean-canary-long-20260515c`, train call
  `fc-01KRP6J8GDR05P6MYRZKXVQ9QK`, starting from short-canary champion
  `iteration_1000.pth.tar`, with `max_train_iter=20000`.
- Long-canary tournament/promotion is done:
  `curvy-e2e-clean-canary-long-live-20260515c` /
  `elo-e2e-clean-canary-long-live-20260515c`, `2` pairs / `6` games /
  `0` failures, assignment sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30`.
- Bug found and fixed: promotion wrote a duplicate pointer at literal
  `control:training/...` instead of replacing `training/...`. The script is
  patched, regression-tested, and the live pointer was manually repaired for the
  long canary.
- Current deployed proof result: passed on the long canary after pointer repair.
  The same running trainer applied sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30` at train
  iter `2750`, all `8` envs were ready, and `env_steps.jsonl` contained `1107`
  rows using that sha with provider load OK.
- Current remaining validation/hardening:
  keep the pointer-prefix regression test; record and later fix the
  duplicate-rating race; watch `/runs` inode pressure before any larger launch;
  separately remote-smoke background eval/GIF control-ref handling if the next
  launch depends on those pollers.
- Fresh hardening result after the canary:
  broad E2E-adjacent local suite passed with `385 passed, 14 skipped`;
  background eval poller control-ref assignment-file reload bug is patched and
  unit-tested; old long-canary training app was stopped after proof. The
  current active trainer app is now the all-v2 deployment
  `ap-TyUzvtvYjsO0YFrNx275CE` with `0` tasks.
- Fresh remote poller smoke found the next hardening gap: the poller can now
  read the `control:` assignment file, but if that assignment points at a
  `runs:`/`/runs` frozen checkpoint ref, the separate poller process may still
  miss the checkpoint file until the runs volume is explicitly reloaded before
  resolving/loading frozen opponents. This is a background eval/GIF poller
  blocker, not a contradiction of the same-trainer refresh proof above.
- Local patch after that smoke: assignment resolution now has a separate,
  poller-only nested-checkpoint reload flag. The active trainer refresh path
  still avoids broad `/runs` reload while training. Focused validation:
  `tests/test_curvytron_live_checkpoint_eval_plumbing.py` -> `85 passed,
  3 skipped`; ruff passed for the touched trainer/test files.
- Broad E2E-adjacent validation after the poller hardening passed:
  `386 passed, 14 skipped, 16 warnings`. This is pre-reset history; the fresh
  all-v2 deployed canary has now passed. The next remote step is launch
  hardening and larger-run readiness, not another hybrid smoke.

## 2026-05-15 Seat / Slot Recheck

- Current local truth: the trainer default is `random_per_episode`, fixed-seat
  modes are diagnostics, and old `ego_player_index` config is rejected.
- Current local truth: public opponent mixtures/assignments use
  `opponent_immortal`; raw public `opponent_death_mode` is rejected. Runtime
  selection adds the env-facing `opponent_death_mode` only after a slot is
  chosen.
- Current local truth: `blank_canvas_noop` public slots must set
  `opponent_immortal=true`; the blank sentinel in both leaderboard selectors
  now advertises that truth instead of relying on hidden env behavior.
- Current local truth: tournament eval specs pin policy observation surfaces
  with `policy_trail_render_mode` and `policy_bonus_render_mode`, default to
  eval/greedy policy mode, and use balanced physical seats.
- Current local truth: trainer run metadata writes the same policy observation
  surface onto checkpoints/run summaries (`policy_trail_render_mode`,
  `policy_bonus_render_mode`, `policy_observation_contract_id`, and
  `observation_contract`). Tournament checkpoint normalization reads that
  metadata so a checkpoint carries the observation surface it was trained on.
- Focused validation: `79 passed` for opponent slots, random learner seat,
  no-op/straight action contract, tonight18 manifest, and tournament eval
  parity slices; ruff passed for touched files.
- Side-agent red reports were rechecked against current code: checkpoint intake
  repair is now green locally (`17 passed`), and trainer surface metadata is
  now green locally (`11 passed`). Do not treat older failing snapshots of
  those suites as current.
- Checkpoint observation-surface readback was also spot-checked:
  `test_checkpoint_spec_reads_policy_render_mode_from_observation_contract`
  passed.
- Earlier cleanup inventory that said v2 durable objects were absent is now
  stale: the exact v2 Volumes/Dicts/Queue were deleted and recreated at about
  2026-05-15 14:08 EDT. Treat old cleanup statements as history, not launch
  truth.
- Fresh Modal inventory recheck after all-v2 cleanup/redeploy: tournament
  service `curvyzero-checkpoint-tournament-v2` is deployed, trainer service
  `curvyzero-lightzero-curvytron-visual-survival-train-v2` is deployed with
  `0` tasks, GIF browser `curvyzero-curvytron-gif-browser-v2` is deployed with
  `0` tasks, and the old non-v2 Curvy trainer/tournament deployments are
  stopped.
- Current deployed storage is all-v2. Non-v2 storage remains present only as
  historical evidence and must not be used for the next launch unless an
  explicit migration copies the needed artifact into the all-v2 lane.
- Fresh deployed Modal loop after the all-v2 reset is proven at canary scale by
  `curvy-e2e-allv2-canary-20260515a`: trainer checkpoint -> subscriber/intake
  -> tournament -> public leaderboard -> assignment refresh -> trainer uses the
  promoted opponent. Older non-v2 canaries remain historical mechanics proof
  only.

## 2026-05-15 Tournament File Cleanup

- `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` was too large
  and mixed too many jobs in one file.
- First safe split is landed locally. The public Modal entrypoint and function
  names stay the same.
- New modules:
  - `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_settings.py`
  - `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_runtime.py`
  - `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament_browser_render.py`
- The Modal image already copies all of `src/` into `/repo/src`, so these helper
  modules will deploy with the existing app packaging.
- Validation: focused browser slice `27 passed`; broader
  tournament/intake/leaderboard slice `153 passed, 11 skipped`; ruff passed.
- Next cleanup should move discovery, intake-manifest helpers, and rating
  artifact/progress helpers. Keep actual Modal functions and queue/dict mutation
  in the entrypoint until the helper modules are boring.

## 2026-05-15 Shared Contract Cleanup

- Current CurvyTron defaults now live in one file:
  `src/curvyzero/contracts/curvytron.py`.
- The old compatibility shim
  `src/curvyzero/infra/modal/curvytron_volume_names.py` is deleted. Do not
  reintroduce it.
- Current storage/app defaults are explicit shared-contract objects:
  `curvyzero-runs-v2` for trainer checkpoints/runs,
  `curvyzero-curvytron-tournaments-v2` for tournament artifacts,
  `curvyzero-curvytron-control-v2` for assignment/control files,
  `curvyzero-curvytron-checkpoint-intake-v2`,
  `curvyzero-curvytron-checkpoint-events-v2`, and
  `curvyzero-curvytron-opponent-leaderboard-live-v2`.
- Tournament UI "current" defaults also live in the shared contract. Current
  code points at the fresh all-v2 canary proof:
  `curvy-e2e-allv2-canary-live-20260515a` /
  `elo-e2e-allv2-canary-live-20260515a`. Older loop18/v2real18 arenas are
  historical/forensic until a fresh real launch id is chosen.
- `modal_volume_kwargs_for_name(...)` uses an explicit map of verified VolumeFS
  versions. It currently requests version 2 for `curvyzero-runs-v2`,
  `curvyzero-curvytron-tournaments-v2`, and
  `curvyzero-curvytron-control-v2`. Old non-v2 volumes are not current launch
  targets.
- Current app/Volume/Dict/Queue getters reject non-`-v2` overrides. If a shell
  or manifest tries to route the current CurvyTron lane to old names, that is a
  launch-time error, not a quiet compatibility path.
- Current training/tournament defaults are explicit: one source frame per
  action, `source_max_steps=1048576`, `save_ckpt_after_iter=10000`,
  `commit_on_checkpoint=true`, `learner_seat_mode=random_per_episode`, and
  policy observations are `browser_lines + simple_symbols`.
- The restart18 manifest builder now fails closed for real launches: it
  requires an explicit ratings/leaderboard snapshot, defaults to
  `opponent-source=assignment`, writes immutable assignments to the v2 control
  volume, creates per-recipe control-volume refresh pointers, and uses a coarse
  default refresh interval of `2000` learner train iterations. Inline-mixture
  manifests are diagnostic-only and must explicitly disable refresh.
- A diagnostic dry-run manifest was built from the all-v2 canary leaderboard to
  verify the builder/submitter fields. It is not launchable as the real batch:
  the source has only `4` active rows and rank 1 is `iteration_0.pth.tar`.
- Seat/perspective wording now has a single local contract:
  `policy_observation_perspective_contract_2026-05-15.md`. Coach/training owns
  learner-seat selection; Optimizer owns fast backends for the same
  controlled-player observation view.
- Learner perspective is now a first-class training setting. The current
  default is `random_per_episode`; each reset chooses the learned policy's
  physical seat deterministically from the episode/reset seed, and telemetry
  records both learner and opponent player indices.
- The action space is still exactly three actions: `left`, `straight`,
  `right`. `straight` is the explicit no-turn/no-op-in-turn-space action; do
  not add a fourth action unless the game contract itself changes.
- Opponent slots express immortality with public `opponent_immortal`. Fresh
  public mixture entries may not hand-author `opponent_death_mode`; episode
  selection derives that lower-level env switch. `blank_canvas_noop` entries
  must state `opponent_immortal=true` because the runtime treats them as
  inert/immune.
- `16.666666666666668ms` is not an arbitrary per-test magic number. It is
  `SOURCE_PHYSICS_STEP_MS * 1`, because the current contract is one action per
  source frame.
- Render parity work is deferred to the optimizer lane. The current launcher
  contract should not use `body_circles_fast` as the training policy surface;
  that path remains historical/diagnostic evidence only.
- Fresh tournament/rating specs now use `policy_trail_render_mode` and
  `policy_bonus_render_mode` for policy observations. Old aliases like
  `observation_*`, `source_state_*`, and generic `trail_render_mode` are not
  accepted as fresh spec inputs; old checkpoint metadata translation remains an
  explicit repair boundary.
- Focused validation after this cleanup:
  `uv run pytest tests/test_curvytron_shared_contracts.py
  tests/test_curvytron_survivaldiag_manifest.py
  tests/test_curvytron_opponent_mixture_manifest.py
  tests/test_curvytron_tonight18_manifest.py
  tests/test_curvytron_checkpoint_tournament.py::test_rating_context_hash_changes_for_evaluator_not_roster
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif
  tests/test_multiplayer_source_state_trainer_surface.py::test_body_circles_fast_is_rejected_by_current_trainer_surface
  -q` -> `35 passed`.
- Ruff passed on the touched contract, trainer, tournament, manifest, and
  focused test files.
- Focused validation after the blank/immortal public-slot cleanup:
  `uv run pytest tests/test_opponent_leaderboard.py tests/test_opponent_mixture.py
  tests/test_opponent_registry.py
  tests/test_source_state_visual_survival_learner_seat_regression.py
  tests/test_env_contract.py tests/test_curvytron_tonight18_manifest.py
  tests/test_curvytron_checkpoint_tournament.py::test_build_game_specs_randomizes_balanced_seat_order_by_default
  tests/test_curvytron_checkpoint_tournament.py::test_rating_counts_wins_from_each_games_actual_seat_order
  tests/test_curvytron_checkpoint_tournament.py::test_tournament_render_contract_pins_policy_surface_and_full_gif
  tests/test_curvytron_checkpoint_tournament.py::test_tournament_rejects_legacy_policy_surface
  tests/test_curvytron_checkpoint_tournament.py::test_checkpoint_game_uses_per_seat_policy_modes_and_full_rich_gif
  -q` -> `79 passed`.
- Ruff passed on the touched opponent leaderboard/mixture/registry tests and
  source files.

## 2026-05-15 Stop / Restart Decision

- 2026-05-15 10:45 EDT cleanup pass:
  - `opponent_leaderboard.py` no longer emits public stable-slot entries with
    `opponent_death_mode`; it emits `opponent_immortal` and lets episode
    selection derive the runtime env switch.
  - Focused validation: `uv run pytest tests/test_opponent_leaderboard.py
    tests/test_opponent_mixture.py tests/test_curvytron_tonight18_manifest.py
    -q` -> `48 passed`; ruff passed for touched assignment/mixture/manifest
    files.
  - Broader validation: `uv run pytest tests/test_opponent_leaderboard.py
    tests/test_opponent_mixture.py tests/test_curvytron_tonight18_manifest.py
    tests/test_source_state_visual_survival_learner_seat_regression.py
    tests/test_curvytron_live_checkpoint_eval_plumbing.py
    tests/test_curvytron_checkpoint_tournament.py
    tests/test_curvytron_tournament_scheduler_fairness.py -q` ->
    `271 passed, 14 skipped, 4 warnings`; ruff passed across the touched
    trainer/tournament/assignment files.
  - Older opponent-mixture manifest builder was also cleaned so `rf` means
    `body_circles_fast`, public mixtures use `opponent_immortal`, command/dry
    run kwargs satisfy the current grouped submitter, and public manifests no
    longer persist derived `opponent_death_mode`.
  - Final combined validation with the older mixture builder included:
    `282 passed, 14 skipped, 4 warnings`; ruff passed.
  - `cleanup_lane_2026-05-15.md` now treats v2 storage/control objects as exact
    purge candidates, not current evidence.
  - Exact v2 storage/control purge completed and verified absent: the v2 runs,
    tournament, control volumes, intake/leaderboard dicts, and checkpoint queue
    are gone. Non-v2 Curvy storage remains.
- Current decision: the live v2 real18 run is invalid enough to stop, not just
  label as weak evidence. It is useful only as smoke/history for diagnosing the
  failure modes.
- Current code direction: new training defaults to
  `learner_seat_mode=random_per_episode`; `fixed_player_0` and
  `fixed_player_1` are explicit diagnostics only. Old `ego_player_index`
  config is rejected.
- Current opponent-slot direction: slot recipes express death immunity with
  `opponent_immortal`. Policy kind, runtime mode, and immortality are separate
  ideas. `opponent_death_mode` may still appear as the env/runtime switch
  derived at episode selection, but it is not accepted as clean public slot
  intent.
- Tournament eval balanced physical seating is implemented locally by the
  tournament lane and covered by focused tests. Treat it as required for the
  next tournament deploy/restart.
- Do not launch new training until these are implemented and tested:
  randomized learner seat/perspective handling, no-op/straight action checks,
  tournament eval parity with the trainer surface, and stale-app/workspace
  cleanup.
- Next manifest should use `random_per_episode` learner seat/perspective
  handling. Do not repeat the previous seat-0-only shape.
- Next manifest should globally include at least about `20%` blank/immortal
  pressure, with some variants using higher immortal pressure. Do not reuse the
  previous weak `5%` wall-avoidant recipes as the main pressure plan.
- Modal operating rule for the restart: prove durable behavior from deployed
  apps, use Volume JSON as truth, use Dict/Queue only for coordination, stop
  stale detached apps before trusting dashboards, and avoid designs that depend
  on broad Volume reloads during active file reads.

## 2026-05-15 09:40 EDT Reorientation

- New P0 risk: player perspective may be wrong or incomplete. We need to know
  whether the trainer ever trains the learned policy as seat 1 / player 2, and
  whether tournament games evaluate both seats in a way that matches training.
  Until this is audited, treat the corrected rerate as smoke evidence, not final
  proof that rankings are valid.
- Corrected 16.6667ms rerate
  `elo-v2real18-rerate67-allpairs-16ms-20260515a` is still useful as a live
  worker/timing/render smoke. Do not publish new training assignments from it
  until the perspective audit clears or a fix lands.
- The user is open to purging and relaunching, but the next launch should not
  repeat an invalid setup. First inventory the live trainers, checkpoints,
  survival/outcome metrics, leaderboard state, and weak-run intervention state.
- Weak-run immortal/blank intervention has not yet been proven applied. This is
  now a tracked lane, not a chat-only idea.
- Parallel agents now own five bounded audits: training player perspective,
  tournament evaluation perspective, live metrics, workspace cleanup, and
  weak-run immortal intervention.
- Main-thread rule for the rest of this incident: if a blocker appears, start a
  smaller honest parallel proof while debugging, but also stop or relabel any
  work that cannot answer the real question.
- Live metrics audit at this point: `21` tracked rows, `14` running, `7`
  failed, `215` checkpoints, max checkpoint `iteration_160000`. Old live
  leaderboard still has only `53` rows and max admitted iteration `30000`;
  corrected 16ms rerate is still running with successful game logs but no final
  corrected leaderboard yet.

## 2026-05-15 09:26 EDT Current Truth

- The v2 real18 trainers are alive enough to matter: latest tracked status had
  `21` rows (`18` original plus `3` replacements), `17` running, `4` failed
  originals, `90` durable checkpoints total, and latest checkpoint up to
  `iteration_60000`.
- The small deployed v2 loop is still the strongest full-loop proof: trainer
  checkpoint -> intake/tournament -> promoted assignment -> same running
  trainer refresh was observed with concrete assignment sha
  `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770`.
- The current 67-checkpoint rerate
  `elo-v2real18-rerate67-allpairs-20260515a` is useful only as a liveness
  smoke. Do not promote it as final evidence. Review found that the tournament
  game runtime used `20.0ms` source ticks while the trainer/checkpoints use
  `16.666666666666668ms` source ticks.
- Immediate fix: patch the tournament contract so runtime timing mismatch is
  rejected, carry `policy_bonus_render_mode` explicitly beside
  `policy_trail_render_mode`, redeploy `curvyzero-checkpoint-tournament-v2`,
  and launch a fresh 16.6667ms rerate for the discovered v2real18 checkpoints.
- Local fix is now implemented and tested:
  `uv run pytest tests/test_curvytron_checkpoint_tournament.py
  tests/test_curvytron_gif_browser.py -q` -> `154 passed, 21 skipped`.
  Deploy and corrected rerate launch are next.
- Corrected tournament app is deployed. Corrected detached rerate is running as
  app `ap-MKU8vQNXqZWCqX6Dle0ztG`, function call
  `fc-01KRNXDVC9552230KK0KCBYZQ1`, rating id
  `elo-v2real18-rerate67-allpairs-16ms-20260515a`.
- Corrected rerate persisted input was verified: `67` checkpoints,
  `2,211` pairs / `46,431` games, `decision_source_frames=1`,
  `decision_ms=16.666666666666668`,
  `source_physics_step_ms=16.666666666666668`,
  `policy_trail_render_mode=body_circles_fast`,
  `policy_bonus_render_mode=simple_symbols`, `max_steps=1048576`,
  GIFs on with `5` samples per pair.
- Wrong-tick rerate app `ap-uIXpEjsU0Iy0lM0NHs8qEk` was stopped so it cannot
  waste compute or be mistaken for final evidence.
- Survival improvement is still not proven for the real18 lane. The best
  current proxy is weak: GIF physical-step mean moved about `121.6 -> 133.3`
  across comparable original rows, with latest greater than first in `8/16`
  rows. This is not a clean eval metric.

## 2026-05-15 06:35 EDT V2 Real18 State

- Current v2 training batch:
  `curvy-v2real18-20260515a`, launched on
  `curvyzero-lightzero-curvytron-visual-survival-train-v2`.
- Current v2 tournament:
  `curvy-v2real18-live-20260515a` /
  `elo-v2real18-live-20260515a` on
  `curvyzero-checkpoint-tournament-v2`.
- Corrected tournament round `round-000001` completed cleanly:
  `231/231` pairs, `4,851/4,851` games, `0` failed games,
  `22` active rows, `0` provisional rows. `stable=false` only means ratings
  are still moving; the active-row gate is now passing.
- The earlier provisional-row issue was self-inflicted: `placement_min_games`
  was set to `420`, which is impossible for a 17-player all-pairs pool because
  each checkpoint can only get `16 * 21 = 336` games. The corrected run let the
  tournament choose the feasible active gate.
- Fresh per-recipe assignment refresh was published from the active v2
  leaderboard:
  - `blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35` ->
    sha `9717c8b00d1e4a030026ca4188611f04d961b6d6a6f477f8758f11489d8f8d45`
  - `blank20-wall5-rank1_75` ->
    sha `e348714b7c960ea62423fd5a8cedaf20427778f764957a4142d5968bc2080f36`
  - `blank5-wall5-rank2_25-rank1_65` ->
    sha `4db8fe399ce6d423f50cb30d8269c2d18bbf1b7025f8c40ffb8163972604fb5a`
- Real-batch refresh proof is partial and live:
  status snapshot `run_status_after_refresh.json` showed `4/18` rows had
  already logged `decision=applied` with one of those new assignment hashes.
  `15/18` original rows were still running; `3/18` had failed.
- Failure diagnosis is concrete:
  the failed rows were using old deployed code where
  `lightzero_target_config.td_steps = source_max_steps = 1048576`. That value
  is a MuZero replay target horizon, not the game cap, and it triggered
  LightZero replay sampling failures: `ValueError: 'a' and 'p' must have same
  size`.
- Fix is implemented and deployed:
  the trainer no longer writes `td_steps` from `source_max_steps` for
  `source_state_fixed_opponent`, so LightZero keeps its stock `td_steps=5`
  while the environment can still allow long games.
- Replacements launched for the three failed rows from refreshed manifest
  `curvy-v2real18-refresh-r1-20260515a`:
  rows `r008`, `r009`, and `r011`. They use new run ids under
  `curvy-v2real18-refresh-r1-20260515a-*`, start from the current tournament
  rank-1 checkpoint, and use the same control refresh pointers.
- Still unproven for the real batch:
  all live/replacement rows applying refreshed assignments, replacement rows
  surviving past the old replay crash point, background eval summaries finishing,
  and a quantified survival-improvement trend.

## 2026-05-15 06:57 EDT Live Check

- Current tournament latest snapshot is now `round-000003`: `40` rows, all
  `active`, `0` provisional. `stable=false` means ratings are still moving, not
  that the active-row gate is failing.
- Current tracked trainer set is `21` unique rows: `18` original real18 rows
  plus `3` replacements. Latest status read shows `17` running and `4` failed
  original rows.
- Checkpoint files discovered from those `21` run ids with
  `checkpoint_selection=all`: `67` refs across `20` runs; one replacement had
  no checkpoint at discovery time.
- The trainer side is ahead of the tournament side: status sees `61`
  checkpoint files, discovery sees `67`, but the tournament latest ranks only
  `40` rows. Immediate action is to submit the exact discovered refs to the
  current v2 intake in small batches, then run a detached drain/continuation and
  watch whether the tournament row count catches up.
- Assignment refresh uptake is real: `15/21` tracked rows have applied one of
  the three current tournament-derived assignment hashes at least once. The six
  without applied refresh are the three old failed originals plus the three
  replacement rows.
- Replay-crash replacement evidence is improving but not complete: replacement
  `r008` has reached `iteration_10000`; the other two replacements were still
  at startup checkpoints on this read.
- Survival improvement is still not proven for this real18 lane. Background
  eval manifest count is still `0`; GIF artifacts exist, but they are not a
  survival metric.

## 2026-05-15 07:12 EDT V2 Rerate Decision

- The first 67-ref continuation under the old rating id failed correctly:
  `latest snapshot context_hash does not match rating spec`. Cause: the manual
  seed changed the rating context from the old `policy_trail_render_mode=
  body_circles_fast` to null. We should not mix those histories under one
  rating id.
- The anti-shrink patch worked: after redeploy and reseed, both Dict and Volume
  intake manifests stayed at `67` refs with `placement_min_games=null` in the
  seed output. A later status check also showed the old latest moved from `40`
  to `53` active rows, but the old rating id is now historical for this
  purpose.
- Historical diagnostic rerate lane: for the exact 67 discovered refs, the old
  bounded v2real18 smoke explicitly set
  `policy_trail_render_mode=body_circles_fast`, high `max_steps=1048576`, GIFs
  on, and `all_pairs`. Keep this as timing/current-CPU fallback evidence only;
  do not copy it into the restart target.

## 2026-05-15 07:20 EDT Live Check

- Fresh clean rerate is running under rating id
  `elo-v2real18-rerate67-allpairs-20260515a`, app id
  `ap-uIXpEjsU0Iy0lM0NHs8qEk`.
- The Volume `progress.json` for that rerate is stale at the initial
  `0/46,431` games read, but Modal logs are live and show successful game
  workers at pair indices in the hundreds. Sampled game logs show
  `ok=true`, `error_type=null`, and `max_steps=1048576`.
- Trainer status snapshot over `21` tracked rows:
  `17` running, `4` failed originals, `90` total durable checkpoint files,
  checkpoint range `0..7` per row, max latest checkpoint `iteration_60000`.
- Assignment refresh uptake is still real: `15/21` rows have applied one of
  the three current tournament-derived assignment hashes. The latest decision
  is usually `unchanged` because those rows already applied the same sha.
- Current applied assignment shas are evenly split across the three recipes:
  `4db8fe39...`, `9717c8b0...`, and `e348714b...`, five rows each.
- Eval/survival proof is still missing for this v2 real18 lane:
  `eval_manifest_count` is still `0`, and background eval completions are only
  `7` total. Do not claim survival improvement from checkpoint count or GIFs.

## 2026-05-15 05:09 EDT Reorientation

- The small `/control`-volume loop is the strongest proof we have: a trainer
  wrote checkpoints, intake accepted them, tournament rated them, promotion
  wrote a new assignment on `/control`, and the already-running trainer applied
  that new assignment.
- The longer behavior proof is not still cooking. Modal now shows
  `ap-ciAzi7ByfRueLxZLtqxuEf` stopped at `2026-05-15 05:05:35 EDT`. That run
  wrote checkpoints through at least `iteration_2000`, but it did not receive a
  second promoted assignment before the rating lane stalled.
- The longer proof's trainer side looked healthy. The earlier read that the
  tournament worker was stalled was stale. Fresh progress checks now show the
  long-proof ratings completed games.
- Cleanup agent Noether cataloged live Modal apps and stopped nothing. Reason:
  several app ids contain mixed stale and current tournament activity, so
  app-level stopping would be too blunt without a finer target.
- Current next direct test: run the smallest direct rating smoke with two exact
  checkpoints and `--wait`. If that completes, the game workers are okay and
  the bug is in intake/claim/background scheduling. If it stalls too, debug the
  rating/game worker path itself before launching more training.
- Direct rating separator result:
  `curvy-looplive-directrating-smoke-20260515a` /
  `elo-looplive-directrating-smoke-20260515a` completed. It rated
  `iteration_0` vs `iteration_3000` from
  `curvy-looplive-proof-controllong-20260515d`, played `3` games, wrote
  `latest.json`, and ended `stable=true`. This proves fresh direct rating and
  game workers can run. The remaining tournament blocker is narrower:
  intake/claim/background scheduling or stale round artifacts.
- Next separator:
  run the same two-checkpoint shape through `intake-seed` / `intake-drain`
  with `spawn_rating` and `wait`, using a fresh tournament/rating id.
- Intake separator result:
  `curvy-looplive-intake-smoke-20260515a` /
  `elo-looplive-intake-smoke-20260515a` completed through the intake path:
  `1/1` pair, `3/3` games, `0` failures, `stable=true`.
- Long proof rating correction:
  `elo-looplive-controllong-proof-fresh-20260515e` is complete:
  `10/10` pairs, `210/210` games, `0` failures. The stale
  `game_map_started` snapshot was written before the blocking map returned;
  it was not final state.
- Real remaining gap:
  the long trainer finished before we used the completed long-proof rating to
  publish a new assignment and refresh that same running trainer. We need a new
  behavior proof or real launch that keeps the trainer alive long enough for
  the tournament result to come back in.
- New behavior proof launch:
  `curvy-looplive-proof-controlrun2-20260515f` /
  `try-looplive-proof-controlrun2-20260515f` is now spawned on deployed trainer
  app `curvyzero-lightzero-curvytron-visual-survival-train-v2`, function call
  `fc-01KRNF20A3YCKANZEH8PV0G33F`. It uses the control assignment from the
  successful controlfast proof, starts from controlfast `iteration_135`, writes
  checkpoints every `50` train iterations with `commit_on_checkpoint=true`, and
  refreshes from a `/control` pointer every `25` train iterations. Goal: rate a
  fresh checkpoint and promote it back into this same still-running trainer.
- Controlrun2 behavior proof passed, 2026-05-15 05:30 EDT:
  checkpoint `iteration_400` from `curvy-looplive-proof-controlrun2-20260515f`
  was seeded through intake with the controlfast champion anchor. Rating
  `elo-looplive-controlrun2-proof-r0-20260515f` completed `1` pair / `3` games /
  `0` failures, `stable=true`. Promotion wrote `/control` assignment
  `looplive-controlrun2-proof-r0-assignment-20260515f`, sha
  `3ff1af447117e4e90cd1e82277530063d20ba14086d180df5474e7d5309dfa9d`, and
  rewrote the same running trainer's `/control` pointer. The trainer then
  logged `decision=applied` at train iter `1798`, `env_ready_report.ok=true`,
  and later env telemetry rows used the new sha with
  `opponent_provider_load_ok=true`. This proves the small deployed live loop.
- First post-refresh behavior read, 2026-05-15 05:34 EDT:
  `controlrun2` had `315` env telemetry rows and `67` terminal samples. Before
  refresh (`4fbc8...`) terminal samples were `49`, mean return `162.92`,
  median return `206`, win/loss `29/20`. After refresh (`3ff1...`) terminal
  samples were `18`, mean return `212.44`, median return `270`, win/loss
  `13/5`. This is a good canary sign, but the sample is too small to call
  survival improvement proven.
- V2 storage proof passed, 2026-05-15 05:56 EDT:
  `curvy-v2-looplive-proof3-20260515a` wrote v2 checkpoints, direct v2 rating
  `elo-v2-looplive-proof3-direct-r0-20260515a` completed `1` pair / `3` games /
  `0` failures, promotion wrote assignment sha
  `adb04ed3905fb9c8984e5e213a9261079f0e4be188315912d12ae5290d55b770` on the
  v2 control Volume, and the already-running v2 trainer applied it at train
  iter `1904`. Later env rows used that sha with
  `opponent_provider_load_ok=true`, including a slot pointing at the fresh
  proof checkpoint `iteration_300`.
- V2 intake recheck passed too: the intake-spawned rating
  `elo-v2-looplive-proof3-r0-20260515a` later showed `status=complete`,
  `1` pair / `3` games / `0` failures, `ratings_written=true`, `stable=true`.
  The earlier stuck read was another stale progress snapshot.
- First v2 post-refresh behavior read, 2026-05-15 05:58 EDT:
  `curvy-v2-looplive-proof3-20260515a` had `238` env rows and `50` terminal
  samples on the old assignment sha `d881...`, versus `117` env rows and `25`
  terminal samples on the refreshed sha `adb04...`. Mean terminal return moved
  from `118.24` to `134.72`; mean terminal length moved from `159.20` down to
  `144.96`. That proves the new assignment is being used, but it is not a clean
  survival-improvement claim.

## Current Live Objects

| Item | Value |
| --- | --- |
| Training app | `curvyzero-lightzero-curvytron-visual-survival-train-v2` |
| Tournament app | `curvyzero-checkpoint-tournament-v2` |
| GIF browser app | `curvyzero-curvytron-gif-browser-v2` |
| Current all-v2 proof tournament | `curvy-e2e-allv2-canary-live-20260515a` |
| Current all-v2 proof rating | `elo-e2e-allv2-canary-live-20260515a` |
| Current all-v2 proof training run | `curvy-e2e-allv2-canary-20260515a` |
| Historical clean tournament target | `curvy-loop18-live-main-20260514f` |
| Historical clean rating target | `elo-loop18-live-main-adaptive417-20260515b` |
| Historical training prefix | `curvy-n18conn-` |
| Stale manifest footgun | `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/curvy-v2refresh18p-20260514b.json` is not the current run source for checkpoint injection |
| Manifest builder | `scripts/build_curvytron_tonight18_manifest.py` |

## Current Truth

- Live reorientation, 2026-05-15:
  - Clean rating job `elo-loop18-live-main-adaptive417-20260515b` completed
    `300` pairs / `6,300` games and wrote `latest.json`.
  - `stable=false` means this is a usable bounded first round, not final
    convergence.
  - Promotion published the public leaderboard and wrote assignment
    `loop18-main-adaptive417-r0-assignment-20260515b`.
  - Consume smoke now passes with the immutable tournament winner as the
    initial policy checkpoint.
  - Latest verified smoke:
    `loop18-main-adaptive417-consume-smoke-20260515e`.
  - The trainer loaded
    `iteration_240000.pth.tar` from the rank-1 row into the MuZero model with
    `matching_shape`, skipped four mismatched reward/value head tensors, and
    preserved a fresh optimizer.
  - Smoke artifact verification reports:
    `initial_checkpoint_loaded=true`, `meaningful_model_load=true`,
    `fresh_optimizer_preserved=true`, `provider_ok_row_count=359`,
    `env_step_row_count=359`, `smoke_passed=true`.
  - This proves: clean tournament result -> public leaderboard snapshot ->
    immutable assignment -> trainer launch consumes assignment and starts from
    the tournament winner.
  - Updated proof, 2026-05-15:
    `curvy-looplive-proof-controlrun2-20260515f` proves a new checkpoint from a
    running trainer can be seeded through intake, rated, published/materialized,
    promoted to a `/control` assignment, refreshed into that same running
    trainer, and used by env workers.
  - The later all-v2 canary proves the v2 storage namespace is clean at wiring
    scale. Meaningful survival improvement over a long window and
    production-scale tournament quality remain separate evidence gaps.
  - Important correction: the existing 18 `curvy-n18conn-*` training runs were
    launched with fixed `opponent_assignment_ref` files only. Their manifest
    has no `initial_policy_checkpoint_ref`, no
    `opponent_assignment_refresh_interval_train_iter`, and no
    `opponent_assignment_refresh_ref`. They cannot prove the automatic live
    refresh loop.
  - Historical storage namespace footgun: the clean adaptive417 proof artifacts live in
    the non-v2 storage names:
    `curvyzero-runs`, `curvyzero-curvytron-tournaments`,
    `curvyzero-curvytron-control`, intake dict
    `curvyzero-curvytron-checkpoint-intake-v0`, and leaderboard dict
    `curvyzero-curvytron-opponent-leaderboard-live`.
    This was superseded by the all-v2 reset and all-v2 canary proof. Do not use
    non-v2 artifacts as current launch inputs unless explicitly copied into v2
    storage.
  - Trainer refresh patch, 2026-05-15:
    `opponent_assignment_refresh_ref` may now be either an immutable assignment
    JSON or a `curvyzero_opponent_assignment_refresh_pointer/v0` JSON containing
    `assignment_ref` and `assignment_sha256`.
  - Refresh pointer reload patch, 2026-05-15:
    the running trainer now calls `runs_volume.reload()` before reading the
    mutable refresh pointer. Without this, overwriting the pointer on the
    Volume might not become visible inside an already-running trainer.
  - Focused trainer tests after the pointer patch:
    `tests/test_curvytron_live_checkpoint_eval_plumbing.py` -> `79 passed,
    3 skipped`.
  - Historical trainer app redeployed after pointer patch with explicit non-v2
    proof storage env vars. The current active deployment is the all-v2 app
    namespace.
  - Live-loop proof failure found, 2026-05-15:
    `curvy-looplive-proof-fast-20260515a` wrote numbered checkpoints,
    intake discovered them, tournament rating completed, leaderboard was
    published, assignment was materialized, and the refresh pointer was
    overwritten. The running trainer did not apply the new assignment because
    `runs_volume.reload()` failed while the process cwd was inside `/runs`.
    A cwd-only patch was not enough: Modal also refuses to reload a Volume when
    LightZero has a TensorBoard event file open under that same Volume. Correct
    design is now separated: training logs/checkpoints stay on `/runs`, while
    mutable assignment pointers and refreshed assignments live on `/control`
    and use `control:` refs.
  - Control-volume patch, 2026-05-15:
    trainer functions mount both `/runs` and `/control`; `control:` refs reload
    only the control Volume; assignment writer can write to `target_volume=control`.
    Focused tests now pass with `83 passed, 3 skipped`.
  - Control-volume live proof, 2026-05-15:
    `curvy-looplive-proof-controlfast-20260515c` has now proven the full small
    loop. The trainer wrote checkpoints; intake accepted `129` refs; proof
    rating `elo-looplive-controlfast-proof-20260515c` completed `10` pairs /
    `210` games / `0` failures; promotion wrote a `/control` assignment with
    sha `4fbc8ef9d621ed5848a474d63f0cec900d6a97dd7827b5c1e81e17de0e12d462`;
    the already-running trainer refreshed at train iter `1130` with
    `decision=applied`, `env_ready_report.ok=true`, and later env rows used the
    same sha with provider load OK.
  - Remaining caveat:
    this is a small proof, not evidence that survival is improving in the large
    training batch. Quantify survival/progress before claiming learning is
    healthy.
  - Follow-up behavior proof:
    `curvy-looplive-proof-controllong-20260515d` is no longer running. It
    proved that checkpoints were being written and refresh checks were firing,
    but it did not prove a second full promotion-and-refresh cycle before the
    trainer stopped.
    It wrote checkpoints through `iteration_3021.pth.tar`.
  - Modal CLI must use `--env shankha-dev`; defaulting to `main` makes app log
    checks falsely report missing apps.
  - Four stale detached tournament apps from older stress/all-pairs attempts
    were asked to stop:
    `ap-hgRfLZZa1y9JS6U06hIH9k`, `ap-cIdYnGnowhAaDtn1VRQJw5`,
    `ap-uJTlKnuoh3q4uoa6UeOReu`, `ap-fiuernb2RQSnfIz9yQC0gW`.
  - Preserve for now: deployed tournament app, deployed trainer app, GIF
    browser, and fresh detached proof app `ap-eXVq2pDG90HQgKiHMcEew6`.
- Latest Coach handoff, 2026-05-14 02:14 EDT:
  - Trusted path:
    `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train --env-variant source_state_fixed_opponent`.
  - Recommended next run shape:
    `reward_variant=survival_plus_bonus_no_outcome`, `num_simulations=8`,
    `batch_size=32`, `collector_env_num=32` main, `64` bounded probe.
  - Render correction, 2026-05-15: the production policy observation surface is
    CPU `cpu_oracle` `browser_lines + simple_symbols`.
  - GPU `browser_lines + simple_symbols` remains lab/profiling-only until the
    trainer-visible contract parity gates pass.
  - `body_circles_fast + simple_symbols` is historical CPU ablation/control
    evidence only. Do not copy it as the new-run destination.
  - Do not use `batch64`, multi-GPU, or broad `sim16`; `sim16` is only a labeled
    sentinel.
  - Checkpoint cadence should be around `save_ckpt_after_iter=5000-10000`, lower
    for canaries.
  - Best practical aggressive compute probe: `C256/H100/sim8` with the target
    observation surface.
  - Best cheaper wide compute probe: `C384/L4/sim8` with the target observation
    surface.
  - Keep `C32/C64` as learning-safe core because survival/tournament evidence
    supports those better than very wide collector counts.
- The main goal is still the full loop:

  ```text
  trainer writes checkpoint
  -> subscriber sees checkpoint
  -> tournament rates checkpoint
  -> public leaderboard updates
  -> Coach materializes assignment
  -> trainer consumes assignment as frozen opponents
  -> survival improves or clearly fails
  ```

- The loop is validated at all-v2 canary scale: a concrete checkpoint was
  followed through every link above. Production scale and survival improvement
  remain separate, still-open claims.
- Current tournament rankings are still not final proof. The old `51`-row
  symptom is no longer the live API shape. The active loop arenas now show 18
  standings rows from `curvy-n18conn-*` runs, while `clean3` has a huge
  375-checkpoint all-pairs round planned and `main` has the clean 18-way round.
- Do not use the stale `curvy-v2refresh18p-*` manifest for new checkpoint
  injection. Live discovery against those run ids returns zero checkpoint refs.
  Live discovery against the `curvy-n18conn-*` run ids from the `main`
  standings returns `417` checkpoint refs across 18 runs, max iteration
  `306755`.
- Injection status, 2026-05-15:
  - old `elo-loop18-live-main-20260514f` manifest now has all 417 exact refs
    seen/queued, proving the old `51` cap is not the live intake blocker;
  - that old rating manifest still has bad defaults for this purpose:
    `pair_selection=all_pairs` and `max_steps=8000`;
  - clean replacement rating manifest is
    `elo-loop18-live-main-adaptive417-20260515b`, with 417 refs,
    `adaptive_v0`, `pairs_per_round=300`, `active_pool_limit=100`,
    `games_per_pair=21`, `max_steps=1048576`,
    `decision_source_frames=1`, `decision_ms=16.6667`, GIFs on, 5 samples.
- Current training rows use H100 compute, but that does not mean GPU observation
  rendering. The active manifest used CPU-side
  `body_circles_fast + simple_symbols` observations on H100 compute; keep that
  fact as historical CPU-control evidence only.
- The production target is CPU `cpu_oracle` `browser_lines + simple_symbols`.
  GPU `browser_lines + simple_symbols` is lab/profiling-only until the
  trainer-visible contract parity gates pass.
- The running 18-row batch shows partial learning signal, but not a clean
  monotonic win.

## Survival Signal Snapshot

| Group | Rows | First mean | Best mean | Latest mean | Latest - first | Best - first | Latest up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All rows | 18 | 132.5 | 182.3 | 141.7 | +9.2 | +49.8 | 10/18 |
| Outcome only | 6 | 123.3 | 184.2 | 144.4 | +21.1 | +60.9 | 4/6 |
| Survival + bonus | 6 | 130.4 | 170.9 | 132.7 | +2.3 | +40.5 | 3/6 |
| Survival + bonus + outcome | 6 | 143.9 | 191.8 | 148.0 | +4.2 | +47.9 | 3/6 |
| Clean | 9 | 124.3 | 177.6 | 142.5 | +18.2 | +53.3 | 6/9 |
| 10% noise/skip | 9 | 140.7 | 187.0 | 140.8 | +0.2 | +46.3 | 4/9 |

Source: read-only `eval-summary` over the 18 `curvy-n18conn-*` rows on
2026-05-15 after the connected tournament/consume smoke work.

Plain read: best checkpoints improved strongly, but latest checkpoints are
noisy. The current batch does not show clean monotonic survival improvement.
Almost every row has at least one collapsed-action eval checkpoint, so stronger
policy-quality checks are still needed.

## P0 Incidents

| Incident | Current read | Required action |
| --- | --- | --- |
| v0/v2 storage split | Resolved for the active lane: exact v2 objects were deleted, recreated, verified with `version=2`, redeployed, and proven by the all-v2 canary. Non-v2 artifacts remain historical evidence only. | Keep using shared contract defaults and explicit v2 VolumeFS mapping. Audit launch manifests so stale non-v2 refs cannot sneak into a real batch. |
| Tournament ranking count | Old `51`/`90` `v2refresh18p` readings and old loop18 readings are historical/forensic, not the next launch target. The shared-contract default now points at the all-v2 canary proof until a fresh real tournament id is chosen. | Choose fresh all-v2 tournament/rating ids for the next real launch, then update the shared contract and dashboards in the same patch. |
| Tournament observation mismatch | Historical CPU diagnostic lane used `body_circles_fast + simple_symbols`, but the restart target is CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU `browser_lines + simple_symbols` stays lab-only until contract parity passes. | Align trainer and tournament on the production observation surface before trusting new rankings. |
| Hidden fallback soup | Too many names can silently repair or change render settings. | Urgent patch may keep compatibility fallbacks; follow-up refactor must create one explicit observation-surface contract. |
| Running trainer refresh proof | Proven for wiring: the all-v2 canary proved clean v2 rating -> publish -> assignment -> same-trainer refresh -> provider-ok env rows. | Track scale/survival separately in `FULL_LOOP_PROOF.md`; do not reopen the wiring question unless the contract changes. |
| Old artifacts/apps clutter | Old arenas and apps make the dashboards hard to use. | Cleanup lane must preserve current lane and old champion-anchor source only. |

## Active Side Experiments

- Weak-run immortal intervention: pick the five weakest current runs, inspect
  their current slot probabilities, and raise blank/immortal exposure to about
  `50%` only for those runs.
- Old champion anchors: find the strongest prior full-sweep tournament winners,
  inject roughly top five exact checkpoint refs into the current clean
  tournament, and monitor acceptance/rating.
- Current batch marker: Tournament Arena must clearly default to and label the
  current arena.
- GIF speed/cap: GIFs at `80 fps` can look short even when not truncated; do
  not combine million-step max games with every-frame GIF capture without an
  explicit safe sampling policy.

## Dashboard State

- Tournament website defaults in current code point at the all-v2 canary proof
  ids. Redeploy the tournament app after the next real tournament/rating ids
  are chosen so the dashboard does not default to old loop18/v2real18 lanes.
- GIF browser “Current batch” in current code is no longer `curvy-v2real18-*`;
  it points at the restart prefix `curvy-r18v2-*`. Redeploy the GIF browser
  when the fresh batch is ready.
- Older deployed-dashboard verifications that mention `curvy-n18conn-*` or
  `curvy-loop18-live-main-20260514f` are historical and should not steer the
  next launch.

## Next Move

1. Audit/build the fresh restart18 manifest from an explicit all-v2
   leaderboard snapshot. Do not use default `/private/tmp` snapshots or old
   loop18/v2real18 ids.
2. Once fresh real tournament/rating ids are chosen, update
   `src/curvyzero/contracts/curvytron.py`, redeploy tournament/GIF apps, and
   verify the dashboards show the fresh ids/prefixes.
3. For weak-run immortal intervention, use the five rows and current mixes in
   `TRAINING_CONTROL.md`; do not mutate the live pointer until the exact audited
   assignment-writer path is selected.
4. The live feedback loop is now proven in the active all-v2 lane by
   `curvy-e2e-allv2-canary-20260515a`. `controlrun2` remains historical
   pre-reset proof.
5. Immediate plan: harden launch readiness, audit stale non-v2 refs, keep
   cleanup organized, and launch the next deliberately named larger run only
   from all-v2 inputs.
6. Keep `TODO.md` and `FULL_LOOP_PROOF.md` live as facts change.
