# Launch Gate Checklist

Purpose: decide when a large overnight CurvyTron survival diagnostic batch is
allowed to start.

Current launch status: rescue relaunch. The ugly 50-row `survivaldiag-v1b`
batch was stopped and kept as the only old batch worth preserving. The first
clean 300-row `curvy-survive-bonus-large-20260513a` grouped-app launch was also
stopped because trainer calls crashed immediately from incomplete grouped
`train_kwargs`; only pollers wrote files. Tiny stock `train_muzero` canaries
still clear the reward/support, blank-canvas, passive-immortal dirty-control,
sim16, checkpoint eval/GIF, and live rich-status plumbing gates. The next action
is to patch/test/redeploy/relaunch the same 300-row batch fresh.

## Hard Gates

- [x] Stock trainer path only: `--mode train`, stock `train_muzero`.
- [x] Opponent canary passes for every exact lane being launched. Blank-canvas,
  passive-immortal dirty-control, and sim16 fast/browser strict-stop canaries
  have passed.
- [x] Local survival plus bonus reward tests pass: survival and bonus pickup reward
  appear in trainer rewards, and outcome reward is absent/zero.
- [x] `survival_plus_bonus_no_outcome` is the exact reward variant in the
  trainer config, not only a row label.
- [x] Repo-side LightZero support sizing for
  `survival_plus_bonus_no_outcome` is practical at high cap; reward/value heads
  are not accidentally enormous.
- [x] Modal/runtime LightZero dry gate confirms the capped support config.
- [x] Bonus pickup reward canary passes if bonus reward is included.
- [x] High-cap canary retry uses `source_max_steps=65536` and
  `--stop-after-learner-train-calls`; summary proves bounded learner train
  calls and no config/runtime failure.
- [x] Tiny live canary stop behavior is controlled with
  `--stop-after-learner-train-calls`; do not rely on `max_train_iter` alone for
  very short canaries.
- [x] Checkpoint/eval/GIF poller still writes artifacts.
- [x] Eval-curve tooling can read a local snapshot or canary output.
- [x] Local status/export code can preserve or derive the fields the curve
  tooling now understands: reward components, bonus counts, terminal-cause
  histograms, action histograms/entropy, failure rate, and eval health.
- [x] A real survivaldiag eval/status snapshot includes those rich fields.
- [x] Dry-run matrix manifest names encode reward, opponent death, render,
  stochasticity, seed/copy, and core compute knobs.
- [x] Dry-run matrix manifest separates `training_seed`, reset seed/strategy,
  `opponent_policy_seed`, `opponent_behavior_seed`, `eval_seed`, and `copy_id`
  for rows where those differ.
- [x] Dry-run matrix manifest records reward weights/components:
  `reward_survival_weight`, `reward_bonus_weight`, `reward_outcome_weight`, and
  the actual emitted reward components.
- [x] Dry-run matrix manifest records opponent contract:
  `opponent_runtime_mode`, `opponent_trail_mode`, `opponent_death_mode`,
  `opponent_collision_effect`, and `opponent_visibility_mode`.
- [x] Dry-run matrix manifest records render pairing: `logical_pair_id`,
  `render_pair_role`, and matched seed/copy guarantee.
- [x] Dry-run matrix manifest records stochasticity definition: type, probability,
  affected actor, and action override rule.
- [x] Stock source-state action-repeat stochasticity is exposed and canaried:
  `policy_action_repeat_min`, `policy_action_repeat_max`, and
  `policy_action_repeat_extra_probability` are CLI/config/manifest fields for
  the trusted `--mode train` lane.
- [x] Telemetry/status can report action histogram/entropy,
  straight/left/right rates, terminal cause, bonus count, reward components,
  eval health, and GIF health on live artifacts.
- [x] Scripted/random/checkpoint opponent rows are not included as executable
  rows in the first-wave manifest. They remain gated specs or future work.
- [x] Large batch names are readable and encode opponent, render,
  stochasticity, compute, row number, and seed.
- [x] Large batch uses one deployed Modal app with poller+train calls per row,
  not one `modal run` app per row.
- [ ] Grouped-app train kwargs are complete and verified by manifest and
  submitter preflight checks.
- [x] Current 300-row checkpoint cadence uses `save_ckpt_after_iter=15000`.
  Live mtimes show about 28-31 minutes on sampled rows that reach
  `iteration_15000`.
- [ ] Rescue relaunch shows real trainer artifacts on sampled rows, not only
  `checkpoint_eval_poller.json`.

## Soft Gates

- [x] Browser render canary is not obviously broken.
- [ ] Fast render and browser render produce comparable starting observations.
- [x] `blank_canvas_noop` mode exists and passes tests. Do not substitute
  passive immortal or generic no-trail wording for this anchor lane.
- [x] Blank-canvas implementation uses a real disabled/no-op player path or an
  equivalent tested mechanism; `remove_player` and passive immortal are not
  accepted substitutes.
- [ ] Scripted wall-avoidant opponent survives real-env probes long enough to be
  a real trail-maker lane. If not, keep it out of the main matrix.
- [ ] Sim16/C64/B64 sentinels are defined only after the core matrix is clear.

## Blocking Conditions

- Do not launch a large batch if the immortal opponent also makes ego immortal.
- Do not launch a blank-canvas batch if player 1 can move, render, catch
  bonuses, or leave collidable/visual trail.
- Do not launch if survival reward is missing from train rewards.
- Do not launch if outcome reward is nonzero in the diagnostic lane.
- Do not launch if bonus pickup is named in the row but is not present in the
  actual trainer reward and telemetry.
- Do not launch random learned frozen rows if the opponent checkpoint identity
  and seed are not immutable and visible in the manifest.
- Do not launch if checkpoint eval/GIF artifacts are not discoverable.
- Do not launch if the manifest does not make axes obvious from the run name.
- Do not launch a large batch through per-row `modal run` commands.
- Do not launch if grouped train kwargs are missing required trainer settings.
- Do not launch mixture rows if command metadata and env config disagree about
  the opponent relation. The expected relation for episode-level mixtures is
  `learner_vs_weighted_episode_opponent_mixture`.
- Do not launch if action-collapse telemetry is missing for a large matrix.
- Do not clear a gate from `status_heartbeat.json` or
  `checkpoint_eval_poller.json` after a forced stop; verify Modal app state and
  completed summary/eval/GIF artifacts.
- Do not treat poller-only files as trainer startup. Verify sampled run roots
  include trainer-owned files such as `run.json`, `latest_attempt.json`,
  `attempt.json`, and `status_heartbeat.json`.
- Do not launch scripted/random/checkpoint opponent rows from design notes alone;
  they need first-class wiring and canaries.

## Latest Runtime Gate

2026-05-13: Modal dry run for `source_state_fixed_opponent` with
`survival_plus_bonus_no_outcome`, `source_max_steps=65536`, `num_simulations=1`,
and stock LightZero config returned `ok=true` with no readiness problems. This
does not prove learning, but it clears the reward-support/config-constructor
gate for the stock trainer path.

2026-05-13: Modal dry run for the same stock path plus
`--opponent-runtime-mode blank_canvas_noop` returned `ok=true` with no
readiness problems. This clears the config surface for blank-canvas mode, not a
learning claim.

2026-05-13: local validation suite passed for vector runtime, vector env,
source-state wrapper, live checkpoint plumbing, and eval-curve tooling:
`261 passed, 1 skipped`.

Latest e2e state: the first real canary failed because LightZero v0.2.0 shares
`model.support_scale`; reward/value supports were mismatched (`5` vs `601`
bins). After the shared `support_scale=300` fix, tiny stock `train_muzero`
canaries passed for `blank_canvas_noop/body_circles_fast`,
`blank_canvas_noop/browser_lines`, and `normal/body_circles_fast`.

2026-05-13: tiny Modal stock repeat smoke passed with
`policy_action_repeat_min=1`, `policy_action_repeat_max=3`,
`policy_action_repeat_extra_probability=0.25`,
`ego_action_straight_override_probability=0.1`,
`opponent_runtime_mode=blank_canvas_noop`, and
`reward_variant=survival_plus_bonus_no_outcome`. It completed with
`ok=true`, `called_train_muzero=true`, and 46 telemetry rows. This proves the
flag/config path and tiny runtime path, not learning.

2026-05-13: local status/export bridge test passed. Status rollup can now carry
survivaldiag reward, bonus, terminal, action, entropy, failure-rate, and
eval-health fields into `eval_checkpoints` when the checkpoint eval manifests
contain them. This clears local export logic, not live artifact presence.

2026-05-13: the historical stock manifest generator is blocked by default. It
only emits with `--allow-historical-matrix`, marks itself
`current_launch_approved=false`, and is not the launch surface for the current
survivaldiag matrix.

2026-05-13: local validation round 2 passed relevant suites:
`526 passed, 14 skipped`. This covers source-state reward/env behavior,
blank-canvas masking, action-repeat plumbing, live checkpoint/eval plumbing,
run status, GIF browser tests, eval-curve tooling, and adjacent vector/runtime
tests. It does not replace the high-cap runtime canary or live volume snapshot
check.

2026-05-13: new dry-run survivaldiag manifest generator passed local validation.
It emits `50` executable review rows and `25` render pairs:
`4` exact preflight rows, `32` blank-canvas core rows, `8` blank-canvas
extra-repeat rows, `4` passive-immortal dirty-control rows, and `2` sim16
sentinels. It also records `10` gated specs that are not emitted as commands:
`4` survival-only ablations because `survival_only` is not a first-class stock
source-state reward variant, and `6` ancestor-checkpoint sentinels because
those lanes still need exact-lane canaries and immutable identity review. The
manifest remains `current_launch_approved=false`.

2026-05-13: launcher-compatibility audit found no hard blocker in the default
generated manifest: `gpu-l4-t4-cpu40` is accepted, all emitted command flags
map to `main(...)`, background eval/GIF flags are real, and generated commands
use stock `--mode train` with no stale two-seat path. The manifest now uses
`row_note` for executable dirty/sentinel rows so those rows are not confused
with non-commanded gated specs.

2026-05-13: local validation round 3 passed after manifest/docs cleanup:
manifest/eval/status tests `20 passed`, source-state/vector/runtime tests
`209 passed`, live-checkpoint/GIF tests `67 passed, 10 skipped`, and ruff
passed for the manifest/eval tooling slice. This clears local validation for
the dry-run manifest and blank-canvas first-wave core.

2026-05-13: strict-stop live canaries passed for the remaining exact launch
families. The passive-immortal fast/browser pair, sim16 fast/browser pair, and
the earlier blank-canvas fast/browser pair all completed with final heartbeat,
one checkpoint, completed checkpoint poller, eval artifact, GIF artifact, and
live status fields for reward components, bonus count/reward, terminal cause,
action histogram/entropy, failure rate, eval health, and GIF health. This is a
plumbing/readout claim, not a learning claim. Passive immortal remains a dirty
control.

Previous launch action: final manifest generation plus focused tests and ruff
passed on the current tree, then the 50-row `survivaldiag-v1b-20260513h` batch
was launched from generated review commands. That batch later got stopped
because its names were unreadable and it created one Modal app per row. Keep
its artifacts as evidence, but do not continue that launch lane.

## Overnight Defaults

- Duration target: at least 12 hours.
- Episode cap: `65536`.
- Train caps: `max_train_iter=300000`, `max_env_step=30000000`.
- Checkpoint cadence: `save_ckpt_after_iter=15000` for the already-running
  300-row batch; `save_ckpt_after_iter=10000` working default for the next
  mixture batch.
- Background poller lifetime: `64800` seconds. Stock train Modal functions have
  a 16-hour timeout, and the checkpoint poller function has a 20-hour timeout.
- Attempt ID prefix: `try-...`, with human-readable row names kept below the
  96-character run-management limit.
- Main compute: L4/T4 unless a canary shows a strong reason otherwise.
- Main search/collector/learner: `sim8`, `C32`, `B32`.
- Render: matched `body_circles_fast` and `browser_lines` for serious rows.
- Stochasticity: no, low, medium, high.

## First Morning Readout

Use curve tooling to rank by:

- latest mean survival;
- best mean survival;
- survival best-delta;
- reward best-delta;
- late-bloom flag;
- peak-then-crash flag;
- action collapse;
- eval health.

Then manually inspect top candidates, late bloomers, and surprising failures.
