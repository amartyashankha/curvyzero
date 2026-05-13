# CurvyTron Training To-Do List

Purpose: active checklist. Keep this shorter and more current than the long
research docs.

## Immediate

- [ ] Finish the pruned-run volume cleanup. The kill-list train/poller
  FunctionCalls are canceled (`1720` canceled, `0` failures), but Modal volume
  deletion is rate-limited. Latest read after waiting: `212` survivor roots
  present, `452` kill-list roots still visible, `0` missing preserved roots.
  Retry only after a cooldown and always recount after each wave. A later
  25-root mounted wave and a 50-root mounted wave both reported success without
  changing the immediate listing, and direct volume API deletion is still
  hitting "too many layers." Do not change the preserve/kill policy unless the
  run inventory changes materially. Website markers are already clean: killed
  roots are hidden from the GIF browser and survivor roots remain visible.
- [x] Create a clean run inventory instead of relying on Modal dashboard task
  counts. Current catalog:
  [run_inventory_2026-05-13.md](run_inventory_2026-05-13.md).
- [x] Stop the ugly 50-row `survivaldiag-v1b` batch. Keep artifacts, but do not
  continue that launch lane.
- [x] Replace the 50-row drip with the clean 300-row
  `curvy-survive-bonus-large-20260513a` manifest.
- [x] Fix run naming. Names now look like
  `curvy-survive-bonus-blank-fast-steady-base-r001-s1110011`.
- [x] Add grouped Modal submitter. The next batch must use one deployed app and
  one poller call plus one train call per row.
- [x] Fix the web UI `speed unknown` bug for future runs by writing
  `train/progress_latest.json` on checkpoint save.
- [x] First clean 300-row grouped launch failed because train kwargs were
  incomplete; stop it and delete the broken 300 run roots.
- [x] Patch grouped train kwargs and submitter preflight so missing required
  trainer settings fail before launch.
- [x] Redeploy the trainer app and GIF browser.
- [x] Relaunch the clean 300-row batch as
  `curvy-survive-bonus-large-20260513b`.
- [x] Monitor the clean large batch for liveness, checkpoint/eval cadence, and
  first survival/reward signal. Fresh all-row read on 2026-05-13 09:30 EDT:
  300/300 running with progress, checkpoints, evals, and GIFs.
  Fresh all-row read on 2026-05-13 10:25 EDT: 300/300 running with progress,
  heartbeats, trainer roots, and pollers. Latest checkpoints are mostly
  `iteration_105000` to `iteration_135000`; latest eval mean across 300 rows is
  about 89.9, with max 155.875.
- [x] Keep checking sampled late rows until they have trainer-owned files, not
  only poller files. This is now closed for `curvy-survive-bonus-large-20260513b`.
- [x] Make recent-checkpoint opponent mixture the next immediate batch lane.
  Use [opponent_mixture_batch_plan_2026-05-13.md](opponent_mixture_batch_plan_2026-05-13.md).
- [x] Replace the stale one-base mixture batch with a compact base grid. The
  main variable is still opponent mixture, but the next manifest must include
  paired fast/browser render rows and a few named search/stochasticity probes.
- [x] Run the mixture component critique checklist before launch: each component
  needs a reason, a risk, a local proof that it is actually selected, and an
  artifact field that records it.
- [x] Implement episode-level opponent mixture in the trusted stock path:
  sample per reset, log selected component, reject missing refs, no silent
  fixed-straight fallback.
- [x] Build the first mixture dry-run manifest. It is now stale because it used
  only `body_circles_fast` and `save_ckpt_after_iter=15000`.
- [x] Deploy the trainer app with current mixture eval/GIF code.
- [x] Launch the first 3-row mixture canary:
  `curvy-mix-recent-canary-20260513a`. Treat it as a plumbing check because it
  used the stale checkpoint cadence.
- [x] Prove the first mixture canary failed before trainer startup:
  Modal FunctionCall returned `_run_visual_survival_train() got an unexpected
  keyword argument 'opponent_mixture_spec'`.
- [x] Redeploy the trainer app after the current mixture/progress patches.
- [x] Regenerate the mixture manifest with `save_ckpt_after_iter=10000` and the
  compact base grid. Current core artifact:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix2-core-20260513a.json`.
- [x] Build corrected `curvy-mix2` artifacts: 6-row canary and 228-row review
  manifest.
- [x] Decide whether to launch all 228 rows or prune sentinel rows after the
  corrected canary. The chosen mix2 launch surface was the 156-row clean
  sim8/C32/B32 manifest; hold passive rows and sim16/C64/B64 sentinel rows for
  later waves.
- [x] Prove `curvy-mix2-canary-20260513a` failed before trainer startup:
  command metadata said fixed-straight while env config said weighted mixture.
- [x] Patch the mixture relation mismatch and readiness summary locally; focused
  pytest, ruff, py_compile, and `git diff --check` pass.
- [x] Redeploy the trainer app after the mixture relation fix.
- [x] Regenerate and launch a fresh six-row mixture canary name, avoiding the
  failed `curvy-mix2-canary-20260513a` roots.
- [x] Monitor `curvy-mix2-canary-20260513b`: trainer-owned files,
  `called_train_muzero=true`, `iteration_0` checkpoint, eval/GIF summaries,
  selected mixture component fields, and first `k10` checkpoint.
- [x] Resolve the passive-row launch question. Decision: prune passive from the
  first full mixture launch; keep passive only as canary/dirty-control evidence
  unless explicitly accepted later.
- [x] Decide and generate the exact pruned subset from the 180-row core artifact.
  The chosen mix2 launch surface was the 156-row
  `curvy-mix2-clean-20260513a` manifest: no passive rows, 7 main recipes,
  5 controls, paired fast/browser render, three repeat levels, `k10`.
- [x] Recheck the fidelity/cadence split before launching the core batch. Latest
  sampled 300b rows do not show browser rendering as the main slowdown, but the
  fresh mixture canary took about 31-38 minutes to reach first `k10`.
- [x] Canary the corrected mixture manifest remotely before launching the full
  mixture batch.
- [x] Launch the mixture batch only after local tests, ruff,
  grouped dry-run, corrected canary, and eval/GIF summary checks pass.
- [x] Monitor `curvy-mix2-clean-20260513a` for train roots, `iteration_0`,
  eval/GIF summaries, and first `k10` checkpoint across early/middle/late rows.
  Fresh read: 155/156 rows running, 105 rows at `k10` or later, 64 rows with
  evals, 149 rows with GIFs, and early survival signal.
  Fresh read on 2026-05-13 10:25 EDT: 156/156 running, 154 progress rows, all
  trainer roots and heartbeats, many rows at `iteration_40000` to
  `iteration_70000`, 111 eval rows, and 154 GIF rows.
- [x] Check whether eval/GIF artifacts are actually appearing for
  `curvy-mix2-clean-20260513a`. They are; do not use live
  `background_poller_completed_count=0` as the artifact-health field.
- [x] For `curvy-mix2-clean-20260513a`, compare first `k10` checkpoint timing
  by render mode only after matched fast/browser rows have actually started.
  Latest matched read: 42 matched pairs at `k10`; browser rows are modestly
  slower on checkpoint-gap time, but not enough to explain liveness.
- [x] Write down v1d axis projection using corrected outcome curves.
- [x] Add [user_priority_snapshot.md](user_priority_snapshot.md) so current
  priorities and operating pattern are explicit.
- [x] Pull fresh v1d eval-summary from Modal and write the first simple table.
- [x] Finish manual old-run analysis with subagent cross-check.
- [x] Run eval-curve tooling on old v1d artifacts and compare it with the
  manual table.
- [x] Refine the future tensor design after old-run analysis; do not treat the
  current matrix sketch as a launch plan.
- [x] Confirm minimal `opponent_death_mode=immortal` support and focused tests.
- [x] Pin down what immortal opponents do after wall and trail collisions.
- [x] Design repeat-copy seed policy for random/stochastic opponent rows.
- [x] Design or prototype a scripted wall-avoidant opponent baseline.
- [x] Decide whether `opponent_trail_mode=none` is safe enough for this batch or
  should be a canary-only lane.
- [x] Add focused blank-canvas design note.
- [x] Decide first representation for random frozen opponents: generated random
  LightZero checkpoint, explicit random-policy opponent kind, or both.
- [ ] Implement or script immutable random-init / `iteration_0` checkpoint
  generation and manifest fields if random learned opponents enter the matrix.
- [x] Draft the next overnight matrix before launch; use blank-canvas/no-op as
  the anchor, `survival_plus_bonus_no_outcome` as the main reward, and no
  scripted/random rows until wired and canaried.
- [x] Build the current survivaldiag dry-run manifest generator, separate from
  the historical stock generator.
- [x] Pass live gates in [launch_gate_checklist.md](launch_gate_checklist.md)
  for the first-wave exact lanes.
- [x] Run the final local manifest/test/lint pass before any 12-hour batch.
- [x] Build or delegate eval-curve tooling that can compare many runs by metric.
- [x] Ensure local eval/status export carries reward components, bonus
  counts, terminal-cause histograms, action entropy, and eval health into
  `eval_checkpoints`.
- [x] Check the rich eval/status export on a real live survivaldiag eval
  snapshot.
- [x] Rerun the high-cap live canary using
  `--stop-after-learner-train-calls`; `max_train_iter` can overshoot within one
  LightZero collect/update block.
- [x] Update stale docs so other agents know the current lane.
- [x] Merge current subagent results from [delegation_log.md](delegation_log.md).
- [x] Review [operating_patterns.md](operating_patterns.md) before long waits or
  launches.

## Code Changes Under Consideration

- [x] Add ego-only death mode: player 0 can die, player 1 cannot.
- [ ] Add opponent no-trail mode: player 1 does not create collision body/trail.
- [x] Add survival-only or survival-plus-bonus reward variant for stock
  source-state LightZero.
- [x] Confirm whether bonus catch counts are exposed to wrapper telemetry/info.
- [x] Wire `bonus_catch_count_step` into
  `survival_plus_bonus_no_outcome` trainer reward.
- [x] Add CLI flags and manifest fields for the new opponent/reward knobs.
- [x] Expose stock source-state action-repeat knobs through the trusted
  launcher and canary them remotely.
- [x] Add vector/runtime `disabled_player_mask` for clean blank-canvas no-op
  opponent; do not fake this with `remove_player`.
- [ ] Add focused tests for passive-immortal side effects: out-of-bounds motion,
  trail phasing, learner death on opponent trail, metadata, and trainer config
  plumbing.
- [x] Decide and implement the next opponent behavior for current mixtures:
  blank canvas plus proactive wall-avoidant scripted opponent; passive remains
  dirty-control evidence only.
- [x] Iterate scripted wall-avoidant/reflection probes and pick the first
  integration candidate with data.
- [ ] Optional later: implement a separate proactive force-field wall-avoidant
  opponent with margin `20`. This is not the current launched scripted
  opponent; current launches use `proactive_wall_avoidant`.
- [x] Wire the first hand-designed opponent variant into the stock source-state
  trainer: `proactive_wall_avoidant`. Keep `lazy_weave`,
  `jitter_force_field`, and `wall_follower` as later variants.
- [x] Canary scripted wall-avoidant rows end to end before adding them to the
  overnight matrix. Remote GIF summaries prove scripted selection and clean GIF
  generation.
- [x] Keep passive immortal separate from scripted wall-avoidance: immortal
  means death suppression only, while wall avoidance must use legal actions to
  steer away before hitting the wall.
- [x] Add opponent mixture support as a first-class stock-path feature. It mixes
  blank canvas, passive immortal, proactive wall-avoidant, and static frozen
  checkpoint refs per episode/reset.
- [x] Implement `blank_canvas_noop` opponent runtime mode if the clean
  wall-avoidance lane remains first.
- [x] Add render masking for blank-canvas mode so player 1 is public
  `present/alive` but invisible to observations.
- [x] Check repo-side LightZero support sizing for
  `survival_plus_bonus_no_outcome` at `source_max_steps=65536`; avoid enormous
  reward/value heads.
- [x] Confirm Modal/runtime LightZero accepts the capped shared support config
  for `survival_plus_bonus_no_outcome`.
- [x] Run a real tiny stock `train_muzero` canary with blank-canvas plus
  survival-plus-bonus.
- [x] Inspect e2e canary volume artifacts and confirm both eval and GIF outputs
  are discoverable by the browser/tooling.
- [x] Run matched browser-lines and normal-opponent canaries before deciding
  the next matrix.

## Tooling

- [ ] Create a local/volume survival-outcome curve cache.
- [x] Support multiple metrics per curve locally: outcome win rate, mean survival,
  reward, action collapse, terminal reasons.
- [ ] Add simple curve scoring functions: latest, best, early slope, late slope,
  peak then crash, monotonic-ish improvement, and collapse flags.
- [ ] Export a compact local snapshot for subagents so they do not repeat Modal
  volume scans.
- [x] Block the historical stock manifest generator from accidental current
  launch use.

## Run Planning

- [x] Use high episode cap by default, likely `65536`.
- [x] Pair fast render and browser render for important rows.
- [x] Add repeated copies for important stochastic/random-opponent rows.
- [x] Sweep stochasticity more seriously.
- [x] Use `save_ckpt_after_iter=15000` for the already-running clean 300-row
  batch. Fresh mtimes show it is about 28-31 minutes on sampled rows that reach
  `iteration_15000`.
- [x] Use `save_ckpt_after_iter=10000` for the next mixture batch unless the
  active cadence audit finds a stronger split by render or baseline knob.
  Decision: mix2-clean and mix3-currentckpt use `k10`.
- [x] If checkpoint timing is used as evidence in a future manifest, interleave
  or randomize fast/browser row order so startup lag does not look like render
  speed. Done for `curvy-mix3-currentckpt-20260513a` with alternating render
  lead.
- [x] Refine the candidate 300-row next mixture matrix after the current
  156-row batch reaches first `k10` checkpoints. Launched as
  `curvy-mix3-currentckpt-20260513a`: 180 main mixture rows, 60 pure controls,
  60 narrow compute probes.
- [x] Implement the draft `curvy-mix3-nextwave-20260513a` manifest profile
  with balanced fast/browser launch order and grouped-submit dry-run.
- [x] Decide whether the next matrix uses `save_ckpt_after_iter=10000` or
  `7500` after measuring healthy matched-row checkpoint gaps. Decision:
  keep `k10` for `curvy-mix3-currentckpt-20260513a`.
- [x] Decide whether scripted opponent rows stay in the next matrix after GIF
  and eval summaries prove the scripted component is being selected and behaves
  normally. Decision: keep scripted rows; remote GIF summaries prove scripted
  selection and clean GIF generation.
- [x] Regenerate the next-wave 300-row mixture manifest with current
  `curvy-survive-bonus-large-20260513b` checkpoint refs instead of stale v1b
  refs. Current launch surface:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.json`.
- [x] Launch `curvy-mix3-currentckpt-20260513a` through the grouped app.
  Launch artifact:
  `artifacts/local/curvytron_opponent_mixture_manifests/curvy-mix3-currentckpt-20260513a.grouped_submit_launch.json`.
- [x] Get the first `curvy-mix3-currentckpt-20260513a` startup read. Fresh
  2026-05-13 10:13 EDT read: 187 train roots, 180 running pollers, 36 live
  trainer heartbeats, 35 rows with checkpoints, 3 rows at `iteration_10000`, 8
  eval rows, and 26 GIF rows. Follow-up 10:22 EDT read: 190 train roots, 186
  running pollers, 33 rows at `iteration_10000`, 1 row at `iteration_20000`, 28
  eval rows, and 34 GIF rows. This proves startup/liveness, not full maturity.
  Fresh 10:25 EDT read: 190 train roots, 186 running pollers, 37 live trainer
  heartbeats, 34 rows at `iteration_10000`, 1 at `iteration_20000`, 32 eval
  rows, and 34 GIF rows.
- [ ] Continue monitoring `curvy-mix3-currentckpt-20260513a` until most rows
  have trainer heartbeats, `iteration_0`, eval/GIF summaries, and first `k10`
  checkpoints.
- [x] Patch the checkpoint eval/GIF volume commit storm locally. Current code
  reduces eval commits, disables inner eval commits inside the checkpoint
  worker, and wraps trainer/eval/GIF commits in retry/backoff with jitter.
- [x] Redeploy the trainer and eval apps after the commit-storm patch.
- [ ] Recheck Modal logs after old checkpoint workers drain. New code emits
  `modal_volume_commit_retry`; old pre-redeploy workers can still fail at old
  direct `runs_volume.commit()` lines until they exit.
- [ ] If current pre-redeploy pollers keep missing artifacts, launch a
  current-code sidecar/backfill for the live batches instead of assuming the
  old pollers picked up the redeploy.
- [x] Add a repeatable matched-render status analyzer for mixture batches:
  `scripts/analyze_curvytron_mixture_status.py`.
- [x] Use the matched-render analyzer on later `curvy-mix2-clean-20260513a`
  status snapshots before changing render or checkpoint cadence.
- [x] Correct matched-render analyzer to use checkpoint file mtimes, not latest
  progress elapsed seconds.
- [ ] Wait for `iteration_10000` eval manifests, not only `iteration_10000`
  checkpoints/GIFs, before ranking mixture learning quality.
- [x] Build the aggressive staged matrix using
  [aggressive_matrix_scale_plan.md](aggressive_matrix_scale_plan.md), not the
  stale 48-row sketch.
- [x] Treat matched fast/browser render twins as paired diagnostics with the
  same logical seed/copy, not independent evidence.
- [ ] Keep search/collector/learner sweeps small unless v1d projection says
  otherwise.
- [ ] Run at least 12 hours for the next serious batch unless a canary fails.

## Later

- [ ] Characterize possible frozen opponents before using them as curriculum.
- [ ] Promote stable opponent mixture conclusions into the main docs after the
  100-run mixture batch readout.
- [ ] Revisit current-policy two-seat self-play only after survival diagnostic
  lane is understood.
- [ ] Promote stable conclusions into design docs after the overnight readout.
