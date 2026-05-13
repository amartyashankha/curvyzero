# Current Source Of Truth

Date: 2026-05-12/13 overnight

Purpose: keep the active CurvyTron training worldview in one place. Older docs
are useful history, but this is the current decision surface.

## North Star

Train CurvyTron well enough that we trust the training setup, observability, and
analysis loop before returning to richer self-play. The immediate learning
target is not "beat a weak opponent"; it is "learn to survive longer from visual
input."

## Current Read

The stock LightZero path can move some metrics, but the last v1d matrix did not
prove a useful CurvyTron training setup.

Corrected outcome read:

- Fixed-straight and old frozen opponents often started with losses, then moved
  toward wins. That is a real but low-bar signal.
- Recent and mid frozen opponents were often already wins at the first
  checkpoint, while survival stayed around the floor. That means outcome reward
  was already saturated and gave little useful pressure.
- The next serious batch should use survival plus bonus reward. Outcome reward
  should be off/zero for the diagnostic lane and remain an eval metric only.

## Trusted Training Lane

Use stock LightZero `train_muzero` through:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py --mode train
```

Avoid the old custom `--mode two-seat-selfplay` lane for learning claims.

## Current Phase

Hold launch. The ugly `survivaldiag-v1b` 50-row batch was stopped because the
names were unreadable and each row created a separate ephemeral Modal app. Keep
the artifacts, but do not use those run names for the next launch.

The next launch candidate is now a single clean 300-row manifest:
`curvy-survive-bonus-large-20260513a`.

Current work is:

- use old v1d projection only for outcome/score lessons;
- use the new 300-row manifest with readable run names;
- submit through one deployed trainer app, not one `modal run` app per row;
- keep checkpoint cadence at `save_ckpt_after_iter=15000`, about one
  checkpoint/GIF every 30 minutes on active sim8 rows;
- keep future opponent work gated unless it has first-class wiring and canaries;
- sleep for 30 minutes before the actual launch step, per latest user request.

New blocker found: current `opponent_death_mode=immortal` only suppresses
player 1 death. It does not keep the opponent inside the arena or remove its
trail. Treat it as a canary or raw ingredient, not as the main opponent design.

Reward/canary gate status: `survival_plus_bonus_no_outcome` is implemented,
locally tested, Modal dry-gated, and tiny e2e-gated. It gives survival plus
same-step bonus pickup and excludes outcome from trainer reward. The first real
canary exposed that LightZero v0.2.0 uses shared `model.support_scale`; after
the fix, reward and value heads are both effective `601` bins from
`support_scale=300`. Tiny stock `train_muzero` canaries now pass for
`blank_canvas_noop/body_circles_fast`, `blank_canvas_noop/browser_lines`, and
`normal/body_circles_fast`.

Control stochasticity status: the stock trainer path now exposes and records
the source-state action-repeat knobs:
`policy_action_repeat_min`, `policy_action_repeat_max`, and
`policy_action_repeat_extra_probability`. This is held-action repeat inside one
LightZero env step, not a separate no-op transition. Local tests cover the env
accounting, and a tiny Modal stock `train_muzero` repeat smoke completed with
`policy_action_repeat_min=1`, `policy_action_repeat_max=3`,
`policy_action_repeat_extra_probability=0.25`, blank canvas, and 46 telemetry
rows.

Status/export status: the local status rollup now preserves and derives richer
survivaldiag fields into `eval_checkpoints` when checkpoint eval manifests have
them: reward totals/components, bonus counts, terminal-cause histograms, action
histograms/entropy, failure rate, and eval health. This has local test coverage.
The live reward/status gate is now cleared for the first-wave families:
strict-stop canaries write completed heartbeat/status, completed poller state,
eval/GIF artifacts, `eval_reward_variant=survival_plus_bonus_no_outcome`,
positive survival reward instead of sparse `-1.0` outcome reward, and explicit
zero-valued bonus fields. Reward components mean trainer reward terms only:
`survival` plus `bonus`. Sparse outcome is separate telemetry.

Latest live canaries:

| Family | Fast | Browser | Launch read |
| --- | --- | --- | --- |
| Blank canvas no-op | passed | passed | main anchor cleared |
| Passive immortal dirty control | passed | passed | plumbing cleared, still conceptually dirty |
| Blank canvas sim16 sentinel | passed | passed | sim16 sentinel cleared |

Live canary stop semantics: for very short train-mode canaries, do not rely on
`max_train_iter` as an exact checkpoint count. LightZero checks that cap after a
collect/update block, and one block can run many learner updates. Use
`--stop-after-learner-train-calls` for strict small canaries. A forced Modal
stop can leave heartbeat/poller JSON saying `running`; check Modal app state
directly when deciding whether anything is still alive.

Future successful runs now write a final `status_heartbeat.json` with
`status=completed` or `failed`; the stale-heartbeat warning mainly applies to
older runs and manually stopped runs.

Manifest safety status: the old
`scripts/build_curvytron_stock_train_manifest.py` generator is historical-only
and now refuses to emit by default. It can only be used with
`--allow-historical-matrix` to inspect old May 12 controls.

Current manifest status: `scripts/build_curvytron_survivaldiag_manifest.py`
defaults to `large_ready`, a 300-row dry-run manifest for the next launch. It
uses stock `--mode train`, `survival_plus_bonus_no_outcome`, high cap `65536`,
separated seed/copy fields, matched render pairs, reward/opponent/stochasticity
contract fields, and background eval/GIF enabled. Shape:

| Block | Rows |
| --- | ---: |
| Blank canvas all-level repeats | 160 |
| Blank canvas medium/high extra repeats | 40 |
| Passive immortal fixed-straight dirty controls | 40 |
| Blank canvas compute sentinels: search16, collect64, batch64 | 60 |

The generated run names are human-readable, for example
`curvy-survive-bonus-blank-fast-steady-base-r001-s1110011`. The manifest also
stores the exact `train_kwargs` and `poller_kwargs` needed by the grouped
submitter. It still sets `current_launch_approved=false` because generation is
a review artifact; actual launch goes through
`scripts/submit_curvytron_survivaldiag_manifest.py --allow-launch` after the
manual hold. Current default caps are overnight-sized:
`max_train_iter=300000`, `max_env_step=30000000`,
`save_ckpt_after_iter=15000`, and
`background_eval_poller_max_runtime_sec=64800`.
The stock train Modal functions have a 16-hour timeout; the checkpoint poller
function has a 20-hour timeout so it can keep watching after a 12-hour-plus
training run.
The launched batch uses short attempt IDs with the `sdv1bh-a` prefix because
run-management IDs must be at most 96 characters.

Current live batch: no new clean large batch is running yet. The ugly 50-row
batch was stopped. The next action after the hold is to deploy the trainer app
once, then submit the 300 manifest rows into that app with one poller call and
one train call per row.

Speed display bug: the web UI shows `speed unknown` when
`attempts/<attempt>/train/progress_latest.json` is absent. Future trainer code
now writes this file on each checkpoint save with `iteration` and `elapsed_sec`;
local plumbing tests cover it.

Launch hygiene note: an attempted dispatch from the old
`survivaldiag-v1-blank-core-*` review commands partially spawned a few detached
apps before the attempt-id length bug was fixed. Those apps were stopped. Do not
use those partial rows for learning claims. The later ugly 50-row `v1b` batch
was also stopped. The clean launch candidate is now
`curvy-survive-bonus-large-20260513a`.

Modal dashboard cleanliness: the next launch must use one deployed app plus many
function calls. The grouped submitter mirrors the local entrypoint by spawning
the checkpoint poller first and the train function second. Do not go back to
one `modal run` per row. See
[modal_grouped_app_launcher_followup_2026-05-13.md](modal_grouped_app_launcher_followup_2026-05-13.md).

Manifest/launcher audit status: the default generated commands use accepted
compute `gpu-l4-t4-cpu40`, all emitted flags map to the launcher CLI surface,
background eval/GIF flags are real, and no command uses `two-seat-selfplay`.
Bad manual `--compute` overrides are still not prevalidated by the generator,
so keep review commands generated from known defaults unless adding explicit
validation.

## Next Training Concept

Add diagnostic opponent settings so the learner cannot get easy reward from a
weak opponent dying:

- `opponent_death_mode=immortal`: ego can die; opponent cannot. Current behavior
  is passive death immunity: the opponent can keep moving out of bounds and
  still leaves normal trail/body points.
- `opponent_trail_mode=none`: possible later/canary lane where the opponent
  leaves no collision trail. Do not bundle this with immortal death until the
  separate effects are understood.
- repeated copies: important stochastic/random-opponent rows should run several
  times with different seeds so one lucky seed does not drive the conclusion.
- seed meanings must stay separate: training seed, reset seed, opponent policy
  seed, opponent behavior seed, and eval seed are different axes, not one vague
  `seed`.
- scripted wall-avoidant opponent: possible stronger baseline that turns away
  from walls and creates useful trails.
- random learned frozen opponent: not a first-class source-state opponent yet.
  Prefer immutable random-init or `iteration_0` LightZero checkpoints through
  the existing frozen-checkpoint path, with explicit opponent policy seeds.

Audit result: current `opponent_death_mode=immortal` is not a clean main lane
by itself. It suppresses player 1 death; it does not reflect, stop, remove
trail, or make the opponent avoid walls.

Those gates are now clear for the running first-wave blank-canvas-heavy batch.
For later opponent families, repeat the same rule: no scripted, random, or
checkpoint-opponent rows enter a serious matrix until their exact lane is wired,
manifest-addressable, and canaried.

## Opponent Feature Status

| Feature | Current state | Launch stance |
| --- | --- | --- |
| Blank canvas no-op | Implemented and locally/e2e canaried. Player 1 keeps the two-player shape but is inert, hidden, no-trail, no-collision, and no-bonus. | Main anchor for first survivaldiag matrix. |
| Passive immortal | Implemented as player-1 death suppression. It does not reflect, avoid walls, stop out-of-bounds motion, or remove trail. | Tiny dirty control only. Do not treat as clean opponent design. |
| Opponent no-trail mode | Design exists as a concept, but broad moving no-trail opponent is not implemented as a first-class trainer lane. | Keep out unless separately wired and canaried. Blank canvas already covers the clean no-trail/absent case. |
| Scripted wall-avoidant | Probe/design exists. Best first policy is `proactive_force_field` with safe margin `20`; variants include `lazy_weave`, `jitter_force_field`, and `wall_follower`. These use legal left/straight/right actions and do not bounce or teleport. | Second-wave lane only after stock-trainer wiring and e2e canary. |
| Random learned/frozen | Frozen checkpoint path exists; random-init/iteration-0 checkpoint generation and immutable opponent seed fields are not launch-ready. | Controls only after immutable refs/seeds are in the manifest. |

Do not let the manifest work erase the feature backlog. The first launch can be
blank-canvas-heavy, but the next useful feature lane is probably scripted
wall-avoidant opponent wiring, starting with `proactive_force_field`.
That work is not first-wave launch-blocking: the current policy exists as a
probe/design only and needs first-class source-state trainer wiring, metadata,
tests, and a tiny e2e canary before it can join a matrix.

Feature audit status: the clean first-wave core is implemented and live-tested as
`blank_canvas_noop` + explicit `survival_plus_bonus_no_outcome` + held-action
repeat + eval/GIF/status plumbing. Do not rely on the launcher default
`reward_variant=auto` for this lane; current review commands set the reward
explicitly, and live checkpoint eval defaults to that same training reward
unless explicitly overridden. Moving no-trail, scripted wall-avoidant,
random-init, and ancestor checkpoint families remain separate gated feature
work.

## Matrix Priorities

| Axis | Current stance |
| --- | --- |
| Reward | Survival plus bonus pickup. Outcome reward off/zero; not a serious next-lane candidate. |
| Opponent | Main clean lane should start with blank canvas/no-op once implemented. Passive immortal is a dirty control. Scripted wall-avoidant is the stronger trail-maker lane once validated. |
| Repeated copies | Important stochastic/random-opponent rows should have repeated seeds, about five copies when the row matters. |
| Episode cap | Do not sweep. Set high, e.g. `65536`. If runs get long, that is the signal. |
| Render | For serious settings, run both `body_circles_fast` and `browser_lines` as matched rows. |
| Stochasticity | Sweep meaningfully, including more aggressive levels than `0.05`. |
| Search / collectors / learner batch | Do not over-sweep until projected from v1d. Use a small focused set. |

## Future Tensor Rule

The next tensor should be assembled as staged blocks, not a single crossed
product and not the old 48-row sketch.

Block order should roughly be: mechanics canaries, blank-canvas wall-avoidance,
fixed immortal trail-maker, random-policy immortal opponent repeat groups,
ancestor-checkpoint controls, stochasticity ladder, tiny reward ablation, and
projection sentinels.

Use about five repeated copies for important random/stochastic opponent rows.
Use fewer copies for ancestor checkpoint controls unless they become the main
claim. Keep render twins matched by logical seed/copy so render differences can
be read cleanly.

Aggressive scale is allowed and expected after gates pass. See
[aggressive_matrix_scale_plan.md](aggressive_matrix_scale_plan.md). Rough
targets: about 50 runs to prove the clean survival lane is alive, about 100
runs to compare opponent families and seed stability, 200+ runs for variance
and confirmation, and about 300 runs only if the extra rows are repeats or
clearly different opponent families.

Do not launch from any historical dry-run generator or old launch shell script.
The next launch surface must be generated from a current survivaldiag manifest
that records reward, opponent, render pairing, stochasticity, and separate
seed/copy fields. The current dry-run generator does this locally, but the
commands are still not launch-approved.

## Old-Run Projection Rule

For the v1d runs, project knobs using the corrected outcome/score curve:
first, best, and latest `win/loss/draw`. Do not use those old survival curves
to choose the next reward objective, because those runs did not train on the
survival-first objective we now care about.

For the next runs, the primary curve should be survival/reward. Outcome remains
a secondary eval metric, not a training reward.

## Active Docs

- [todo.md](todo.md)
- [user_priority_snapshot.md](user_priority_snapshot.md)
- [hypotheses_and_evidence.md](hypotheses_and_evidence.md)
- [v1d_axis_projection.md](v1d_axis_projection.md)
- [v1d_fresh_eval_summary_2026-05-13.md](v1d_fresh_eval_summary_2026-05-13.md)
- [next_overnight_matrix_plan.md](next_overnight_matrix_plan.md)
- [next_matrix_manifest_design.md](next_matrix_manifest_design.md)
- [aggressive_matrix_scale_plan.md](aggressive_matrix_scale_plan.md)
- [opponent_diagnostic_design.md](opponent_diagnostic_design.md)
- [blank_canvas_noop_opponent_lane.md](blank_canvas_noop_opponent_lane.md)
- [scripted_wall_avoidant_opponent_baseline_2026-05-13.md](scripted_wall_avoidant_opponent_baseline_2026-05-13.md)
- [source_state_reward_wiring_audit_2026-05-13.md](source_state_reward_wiring_audit_2026-05-13.md)
- [prelaunch_validation_round3_2026-05-13.md](prelaunch_validation_round3_2026-05-13.md)
- [live_highcap_canary_plan_2026-05-13.md](live_highcap_canary_plan_2026-05-13.md)
- [delegation_log.md](delegation_log.md)
- [eval_curve_tooling_plan.md](eval_curve_tooling_plan.md)
- [operating_patterns.md](operating_patterns.md)
- [archive_and_stale_docs.md](archive_and_stale_docs.md)
- [instruction_digest_2026-05-13.md](instruction_digest_2026-05-13.md)

## Do Not Forget

- We need tooling that turns many run artifacts into comparable eval curves.
- Curves should support different metrics: outcome score now, survival and
  reward later, and possibly weighted combinations later.
- Analysis should be cautious about false negatives: weak or late-learning runs
  should not be thrown away too early.
- The main thread should plan, delegate, and synthesize. Subagents can implement
  tools, inspect code, analyze axes, and clean docs.
- Future tensors should be staged blocks, not one giant crossed product.
