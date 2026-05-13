# CurvyTron Training To-Do List

Purpose: active checklist. Keep this shorter and more current than the long
research docs.

## Immediate

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
- [ ] Do not launch yet. When everything else is ready, sleep for 30 minutes
  before the actual launch step.
- [ ] After the hold, deploy the trainer app once and submit the full 300 rows
  with `scripts/submit_curvytron_survivaldiag_manifest.py --allow-launch`.
- [ ] Monitor the clean large batch for liveness, checkpoint/eval cadence, and
  first survival/reward signal.
- [ ] Keep the scripted wall-avoidant opponent lane moving in parallel, but do
  not let it block the clean 300-row batch.
- [ ] Keep ancestor checkpoint controls as a tiny side wave after one exact
  canary; do not mix them into the main clean repeat wave.
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
- [ ] Decide and implement the next opponent behavior: passive canary,
  reflecting wall behavior, no-trail blank, or scripted wall-avoidant.
- [x] Iterate scripted wall-avoidant/reflection probes and pick the first
  integration candidate with data.
- [ ] Implement proactive force-field wall-avoidant opponent with margin `20`
  as the first scripted opponent candidate.
- [ ] Wire hand-designed opponent variants into the stock source-state trainer,
  starting with `proactive_force_field`; keep `lazy_weave`,
  `jitter_force_field`, and `wall_follower` as later variants.
- [ ] Canary scripted wall-avoidant rows end to end before adding them to the
  overnight matrix.
- [ ] Keep passive immortal separate from scripted wall-avoidance: immortal
  means death suppression only, while wall avoidance must use legal actions to
  steer away before hitting the wall.
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
- [x] Use `save_ckpt_after_iter=15000` for the clean large batch. The stopped
  50-row batch showed `5000` was about one checkpoint every 10 minutes; `15000`
  should be about 30 minutes.
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
- [ ] Revisit current-policy two-seat self-play only after survival diagnostic
  lane is understood.
- [ ] Promote stable conclusions into design docs after the overnight readout.
