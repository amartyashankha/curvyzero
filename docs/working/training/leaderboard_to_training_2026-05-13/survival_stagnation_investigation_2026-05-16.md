# Survival Stagnation Investigation, 2026-05-16

Status: active research. Last updated: 2026-05-16 after correcting the TSV
column parsing bug.

## Question

The full loop is now wired, but survival does not obviously trend upward. This
doc tracks the evidence before we change anything.

## Current Setup Under Test

- Batch manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18fresh-allv2-20260516a/curvy-r18fresh-allv2-20260516a.json`
- Training matrix: 18 runs = 3 reward variants x 3 opponent recipes x 2 control
  variants.
- Current tournament:
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`
- Latest tournament snapshot at first check: `round-000011`, `398` rated rows,
  max checkpoint iteration `290000`, `300` bounded pairs, `6,300` games.
- Deployed loop proof: scheduled controller generation 12 was consumed by every
  still-running trainer, with provider failures equal to zero.

## Corrected Quantitative Read

Eval summaries from the current `r18fresh` batch show real mid-run learning and
real late-run regression. A fresher full status pull after the TSV pass nudged
the latest rows slightly, so the current read is:

- `18/18` runs have a best mean survival above their iteration-0 mean.
- `10/18` runs have latest mean survival above their iteration-0 mean.
- Only `4/18` latest checkpoints are within 10% of their own best survival.
- Mean survival: first `159.9`, best `246.0`, latest `175.4`.
- Mean best delta: `+86.1` steps. Mean latest delta: `+15.5` steps.
- Mean drop from best to latest: `70.6` steps.
- Action collapse appears in `13/18` rows at some point in each checkpoint
  ladder, but the latest eval checkpoints are not collapsed by the 0.95
  top-action threshold. This is intermittent collapse plus late regression, not
  every latest policy being stuck on one action.
- Best checkpoints are often mid-run, not final-run. The tournament top 30 also
  heavily favors mid-run checkpoints such as `iteration_40000` through
  `iteration_160000`, with some `iteration_0` and early checkpoints still
  competitive.

Split by reward variant:

| Reward variant | Latest improved | Collapsed | Mean first | Mean best | Mean latest | Mean latest delta | Mean drop from best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sparse_outcome` | 2/6 | 5/6 | 157.7 | 260.4 | 160.2 | +2.5 | 100.2 |
| `survival_plus_bonus_no_outcome` | 3/6 | 5/6 | 154.2 | 235.5 | 166.4 | +12.2 | 69.1 |
| `survival_plus_bonus_plus_outcome` | 5/6 | 3/6 in history, 0/6 latest | 167.7 | 242.0 | 203.3 | +35.6 | 38.7 |

Split by control/noise variant:

| Control/noise variant | Latest improved | Collapsed | Mean first | Mean best | Mean latest | Mean latest delta | Mean drop from best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 4/9 in the fresher status pull | historical collapse present | 166.3 | 238.4 | 166.8 | +0.5 | 72.2 |
| `straight_override_p10_repeat_p10` | 6/9 in the fresher status pull | historical collapse present | 153.4 | 253.0 | 184.0 | +30.6 | 69.0 |

Split by opponent recipe:

| Recipe | Latest improved | Collapsed | Mean first | Mean best | Mean latest | Mean latest delta | Mean drop from best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `blank10-wall10-rank2_25-rank1_55` | 5/6 | 5/6 | 135.5 | 225.9 | 171.8 | +36.4 | 54.0 |
| `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5` | 2/6 | 5/6 | 157.5 | 219.1 | 143.7 | -13.8 | 75.4 |
| `blank20-wall5-rank1_70-rank1imm5` | 2/6 | 3/6 | 186.6 | 293.0 | 206.4 | +19.8 | 86.6 |

Plain reading: this is not "nothing learns." It is "the batch often finds a
better policy and then drifts/collapses later." The `survival_plus_bonus_plus_outcome`
and `so10rep10` slices look least bad at latest, but even there the sample is
small and most latest rows are below their own best.

Important separation of concerns:

- The original `r18fresh` tournament-attached batch did make learning progress
  in the sense that every row found a better checkpoint at some point.
- The full feedback loop also worked mechanically: tournament-ranked
  checkpoints were published into assignments and consumed by the same running
  trainers as frozen opponents with provider-load-ok rows and zero provider
  load failures in the proven generation checks.
- The unresolved issue is learning quality and retention: latest checkpoints
  are often much worse than each row's own best checkpoint. This is why the
  main suspects are training-regime issues (support scale, live opponent
  nonstationarity, collect/batch cadence, shallow search, action noise
  semantics), not a dead subscriber/tournament path.
- The tournament should remain the source of truth for "best policy so far" and
  for selecting frozen opponents/champions. Per-run survival curves answer a
  different diagnostic question: whether each trainer keeps generating better
  latest candidates. If a mid-run checkpoint beats a later checkpoint, the
  tournament should rank the mid-run policy higher; that is good tournament
  behavior and also evidence of trainer regression/retention issues.

The raw aggregate payload for this pass is in:
`/tmp/r18fresh-survival-aggregate.json`. The fresher full-status digest is in:
`/tmp/r18fresh-status-digested.json`.

### Parallel Deep-Dive Update, 2026-05-16

Kant re-ran the active eval/status pull using the current manifests. The newer
numbers keep the same diagnosis:

| Set | Rows | Eval points | Survival mean first/best/latest | Best improved | Latest improved | Latest near best | Collapse |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `r18fresh` | 18 | 523 | `160.2 / 247.0 / 181.6` | 18/18 | 12/18 | 2/18 within 90% | 13/18 ever, 0/18 latest |
| `r18nofb` selected static-control | 3 | 5 | `156.4 / 160.4 / 151.5` | 1/3 | 1/3 | 2/3 within 90% | 0/3 ever/latest |

Additional signal:

- `r18fresh` has `762` visible checkpoints total, min/max per row `25/66`,
  with `7` rows completed and `11` still running.
- Latest action distributions are not collapsed. Combined `r18fresh` latest
  eval actions were roughly `0:45.3%`, `1:49.5%`, `2:5.2%`; mean row
  top-action fraction `0.579`.
- Available learner stderr tails exist for only `7` `r18fresh` rows and none of
  the queried static-control rows. Those tails do show nontrivial learner
  signals: policy entropy near `1.094`, target policy entropy around `0.864`,
  nonzero policy loss around `6.45`, and nonzero grad norm around `1.53`. This
  suggests the learner is fitting nonconstant targets where observable, but the
  machine-readable loss/time-series observability is not good enough yet.
- Current `progress_latest.json` is checkpoint-progress shaped, not
  learner-metric shaped. Missing observability: durable per-checkpoint learner
  loss curves, consistent `model_parameters_changed`, and complete stderr tail
  snapshots for all active rows.

### Observability Patch, 2026-05-16 11:23 EDT

This gap is patched for future jobs. The trainer records lightweight
machine-readable learner metrics through a passive `BaseLearner.train` wrapper:

- `learner_metrics.jsonl` and `learner_metrics_latest.json` live beside
  `progress_latest.json` in the train artifact root.
- Checkpoint progress embeds `last_learner` when a learner row exists.
- `lightzero_curvytron_run_status.py` falls back to
  `learner_metrics_latest.json` and reports train-call index, train iter
  before/after, collector envstep, elapsed time, and numeric learner metrics.
- The wrapper is non-blocking and preserves the original train return value or
  exception behavior.

The same deploy fixes the own-latest control refresh path. The old own-latest
selected3 launch hit one `kept_previous` event per row because it tried to
`Volume.reload()` the runs volume while TensorBoard had an event file open.
Fresh own-latest jobs no longer reload the same runs volume to read a
same-process pointer/checkpoint; external tournament/control refreshes still
reload external state. Local verification: `156 passed, 3 skipped`; ruff clean.

Fresh proof lane launched as `curvy-ownlatest-staticmix-20260516b`, selected
rows `r007/r009/r011`, launch record
`artifacts/local/curvytron_tonight18_manifests/curvy-ownlatest-staticmix-20260516b/curvy-ownlatest-staticmix-20260516b.selected3.launch.json`.
Startup poll immediately after launch found pollers running but no checkpoints,
progress, refreshes, or learner metrics yet. Next proof gate: observe a nonzero
checkpoint writing a run-local pointer, a later applied refresh event,
provider-ok env rows, and usable learner metrics.

## Concrete Runtime Facts

From a real completed run summary:

- The trainer calls stock LightZero MuZero, but the runtime relation is
  `learner_vs_weighted_episode_opponent_mixture`, not true current-policy
  two-seat self-play.
- The command explicitly records `current_policy_self_play=false`,
  `trusted_current_policy_self_play=false`,
  `current_policy_two_seat_action_collection=false`, and
  `two_seat_self_play_status=not_two_seat_self_play`.
- `learner_seat_mode=random_per_episode` is active, so the controlled policy
  does see both player perspectives over episodes.
- `num_simulations=8`, `collector_env_num=256`, `n_episode=256`,
  `batch_size=32`, `lightzero_eval_freq=0`, and background eval is the main
  checkpoint ladder.
- Actual sparse-row target audit example:
  `collector_collect_calls=29`, `game_segments_seen=7479`,
  `replay_sample_calls=308600`, and `action_observability_rows=1234490`.
  That is roughly the expected LightZero `replay_ratio=0.25` shape, not an
  obvious runaway replay loop.
- The `source_max_steps` cap is `1,048,576`. Dense reward targets are capped
  hard at model support scale `300`:
  - `sparse_outcome`: reward/value requested scale `1`, not capped.
  - `survival_plus_bonus_no_outcome`: value requested scale `2,097,152`, capped
    to `300`.
  - `survival_plus_bonus_plus_outcome`: reward requested scale `1,048,578` and
    value requested scale `3,145,728`, both capped to `300`.

This support cap is now a top suspect for dense-reward weirdness. It is also a
good reason to run a short-horizon or reward-scaled control before launching a
large new dense run.

## Current Leading Hypotheses

1. Dense-reward target/support mismatch is a top concrete suspect. The code
   knowingly caps very large requested dense value/reward ranges to support
   scale `300` while using `discount_factor=1.0`.
2. Late-regression after mid-run best is the strongest behavioral symptom.
   It fits the observed "best improves, latest often falls" shape.
3. The current opponent/tournament feedback may be too nonstationary for this
   short run length. A static-opponent control should separate "MuZero cannot
   hold the skill" from "the live opponent curriculum is destabilizing it."
4. Eval noise is significant: background eval uses only `8` seeds, while the
   tournament measures direct checkpoint-vs-checkpoint games. Treat per-row
   latest deltas as noisy; trust broad patterns more than individual rows.
5. Reward/objective mismatch is still possible. Tournament Elo rewards wins
   against checkpoints, while the user's visual question is mostly survival
   time and mechanics. Bonus/outcome variants need reward-component checks.
6. A mechanics mismatch is still possible: observation surface, player
   perspective, no-op/action mapping, bonus ownership/effects, immortality, or
   seat order.
7. The tournament scheduler/backlog issue is real but separate: a previous live
   continuation accidentally scheduled unbounded all-pairs over hundreds of
   refs. That explains stale/backlogged leaderboard symptoms, not the fact that
   individual latest checkpoints can regress relative to their own best.

## Evidence To Gather

- Aggregate survival deltas by reward variant, opponent recipe, and noise/control
  variant. Done for the current pass; rerun after any fresh control.
- Compare latest, best, and tournament-ranked checkpoints for each run.
- Inspect action distributions and collapse by checkpoint.
- Compare current training config against the original `mu0light0pat` or closest
  known baseline.
- Confirm tournament eval uses the same observation/action semantics as training
  checkpoints.
- Check whether bonus pickup and reward components are nonzero and sensible.
- Check whether the trainer is actually learning from the intended rewards,
  not just surviving random seed/opponent variance.

## Parallel Audit Docs

- Curves/signals audit:
  `survival_signal_curves_audit_2026-05-16.md`
- Baseline/config delta audit:
  `survival_baseline_delta_audit_2026-05-16.md`
- Tournament/mechanics audit:
  `survival_tournament_mechanics_audit_2026-05-16.md`
- Static control-run plan:
  `survival_static_control_run_plan_2026-05-16.md`
- Mechanics/reward audit:
  `survival_mechanics_reward_audit_2026-05-16.md`

## External Baseline Notes

Primary-source checks in progress:

- MuZero's paper frames the learned model around reward, policy, and value for
  planning, so reward scaling/ownership and action-selection collapse are valid
  first-class debugging targets:
  <https://arxiv.org/abs/1911.08265>
- AlphaZero's paper is the clean self-play reference point: it starts from
  random play and uses only game rules, which supports the user's point that a
  perfect starting leaderboard is not required to test the loop:
  <https://arxiv.org/abs/1712.01815>
- LightZero's own Gumbel MuZero docs/source warn that larger
  `update_per_collect` is more off-policy and show eval action selection is
  deterministic/no-root-noise. That makes update/replay ratio, exploration, and
  eval-vs-collect action mode important checks:
  <https://opendilab.github.io/LightZero/_modules/lzero/policy/gumbel_muzero.html>

## Subagent Updates, 2026-05-16

Mechanics/reward audit:

- No evidence found that bonus pickup is credited to the wrong player.
- No evidence found that reward components read the wrong seat.
- No evidence found that tournament eval feeds policies the wrong
  seat/perspective observation.
- The largest actionable risk remains dense support scaling: active dense
  survival/bonus rewards can imply million-scale returns while the model support
  is capped at `300`.
- Footgun noted: plus-outcome naming says source-step count, while the current
  implementation uses the wrapper physical-step index. This is clean only
  because the current contract is `decision_source_frames=1`.

Baseline/config delta audit:

- The closest stock baseline is LightZero Atari MuZero, not a local
  `mu0light0pat` identifier.
- Current r18fresh differs materially from that baseline: `num_simulations=8`
  instead of stock Atari `50`, `collector_env_num=256` instead of `8`,
  `batch_size=32` instead of `256`, huge fixed-opponent collection waves, dense
  support cap `300`, `td_steps=5`, and stock root-noise/temperature against a
  three-action game.
- The current lane is single-agent fixed-opponent MuZero with randomized learner
  seat, not true two-seat self-play. That is acceptable as an experiment, but do
  not label it as trusted current-policy self-play.
- The best historical CurvyTron signal was a static matched frozen-opponent run,
  not live tournament feedback. The useful old row was `s92` against its matched
  frozen opponent, with survival rising roughly `151.8 -> 417.0 -> 500.4`.
  That proves the simpler local learning setup can work, but it does not prove
  broad generalization or live-curriculum stability.
- Current `r18fresh` is deliberately much more aggressive than the old/simple
  shape: live assignment refresh every `2000` train iterations, `256` collector
  envs, `batch_size=32`, `num_simulations=8`, stock eval mostly off, and dense
  reward/value support capped at `300`. The highest-probability explanation for
  "best improves, latest falls back" is a combination of nonstationary opponents,
  stale large collection waves, shallow search targets, small learner batches,
  and saturated dense targets.
- Stochastic override/repeat rows add a real requested-action versus executed-
  action mismatch for MuZero replay. Keep them out of clean diagnosis until the
  deterministic rows can hold latest performance near best.

Static/no-tournament control plan:

- Existing builder can make a static frozen/blank/opponent mixture by setting
  `--opponent-source mixture` and `--assignment-refresh-interval-train-iter 0`.
- The cleanest first control is only 2-3 selected rows, not another full matrix.
  The first launch candidate is the no-feedback survival+bonus-no-outcome clean
  slice: rows `r007`, `r009`, and `r011`.
- This control separates "live tournament curriculum destabilizes learning" from
  "the local MuZero target/reward setup is itself unstable."
- Attempted old `restart96` refs are stale after the latest cleanup: both
  current v2 and old-source Modal audits reported `96/96` missing. Do not use
  those old refs for a current control launch without a fresh rematerialization
  source.
- Current control source now uses four nonzero exact checkpoint refs extracted
  from active bounded tournament snapshot
  `curvy-r18fresh-live-bounded-dsf1-20260516b` /
  `elo-r18fresh-live-bounded-dsf1-20260516b`.
- Built control manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18nofb-staticmix-20260516a/curvy-r18nofb-staticmix-20260516a.json`.
- Ref audit passed against `curvyzero-runs-v2`: `4/4` refs exist, `0` missing.
- Grouped dry-run for rows `r007`, `r009`, `r011` passed with
  `assignment_write_count=0` and `refresh_pointer_write_count=0`.
- Launched rows `r007`, `r009`, and `r011` from this manifest. Launch record:
  `artifacts/local/curvytron_tonight18_manifests/curvy-r18nofb-staticmix-20260516a/selected3.launch.json`.
  Train calls: `fc-01KRRJTH7NN6000VN6JK1PPGSF`,
  `fc-01KRRJTHF6R6K65K9CZSSGVTQA`,
  `fc-01KRRJTHP18HR02ZR6HPRY8XCX`. Poller calls:
  `fc-01KRRJTH3E457D6JRZKTNV115J`,
  `fc-01KRRJTHBAHYP8XB1W20D0MB7J`,
  `fc-01KRRJTHJFF7V1MVECRGANHBYN`.
- Early status poll for the no-feedback control is healthy. All `3/3`
  selected rows are `train_status=running`, all `3/3` have
  `status_heartbeat_exists=true`, all `3/3` background eval pollers are
  `running`, and all `3/3` pollers have seen/scheduled the initial checkpoint.
  Assignment refresh is disabled as intended:
  `assignment_refresh_applied_count=0` and
  `assignment_refresh_event_count=0` for all three rows. Initial evals already
  completed for two rows: `112.625` mean steps for `r007` and `183.75` mean
  steps for `r011`; `r009` had scheduled but not finished the initial eval at
  the poll.
- Observability hardening found during that poll: `progress_latest.json` can be
  read mid-write in already-running jobs because the old shared JSON writer
  overwrote files directly. Local code now writes non-exclusive JSON through a
  temp file and atomic replace in `run_management.write_json`; focused tests
  pass. This fix applies to future deployments, not already-running containers.
- Follow-up no-feedback poll: all `3/3` rows are still running with refresh
  counts at zero. Row `r011`
  (`curvy-r18nofb-survbonusnoout-blank20-wall5-rank1_70-rank1imm5-clean...`)
  has reached `iteration_10000`; its initial eval was `183.75` mean steps and
  its `iteration_10000` eval was `136.25`. This is too early to call, but it
  means the static control can already distinguish "not enough time" from "local
  MuZero/reward/cadence regression is present even without tournament refresh."
- Latest no-feedback poll: all `3/3` rows now have `iteration_10000`
  checkpoints and assignment refresh remains exactly zero. Current early eval
  deltas are mixed: `r007` moved `112.625 -> 124.625` mean steps, `r011` moved
  `183.75 -> 136.25`, and `r009` has scheduled/seen `iteration_10000` but its
  latest completed eval is still `iteration_0` at `193.625`.

## Current Diagnosis

The feedback loop is proven as a system path, but the learning signal is not
healthy enough to trust another large sweep without a smaller diagnostic. The
most useful next experiment is a static/no-refresh control plus one target-scale
control, because the strongest concrete bug-risk is not tournament ingestion; it
is that the model target support may be too small for the reward/value scale.

More precise current read: this is not "no learning." It is "learns something,
then fails to retain it." Static/no-refresh rows should answer whether live
tournament-fed opponent nonstationarity is the main cause. If static rows also
learn-then-regress, the next suspect moves to target/support scaling and
LightZero cadence (`collector_env_num`, `batch_size`, `num_simulations`,
`td_steps`).

Local implementation updates ready for the next deploy:

- Shared Modal JSON artifacts now use temp-file plus atomic replace for
  non-exclusive writes, so hot files like `progress_latest.json` and
  `status_heartbeat.json` are less likely to be observed half-written.
- The tonight18 manifest builder now exposes the core diagnostic knobs instead
  of hard-coding them: `collector_env_num`, `n_episode`, `num_simulations`,
  `batch_size`, `model_support_cap`, `td_steps`, and background eval sizing.
  Focused tests/ruff pass: `107 passed, 3 skipped`; shared-contract slice also
  passed (`4 passed`).
- Smoke-built a diagnostic manifest using the new knobs from the current static
  top-four refs: `collector_env_num=64`, `n_episode=64`,
  `num_simulations=25`, `batch_size=128`, `model_support_cap=2048`, and
  `td_steps=50`. This was only a local build under `/tmp`, not a launch.
- Same-run/own-latest control scope: current code can consume refresh pointers,
  and now has a local producer ready for the next deploy. After writing a
  nonzero exact `iteration_N.pth.tar`, the trainer can write an immutable
  assignment pointing at that same-run checkpoint and then update a run-local
  `runs:` refresh pointer. The manifest builder has
  `--own-checkpoint-opponent-refresh` for this lane; a smoke manifest confirmed
  it creates no assignment bank/control pointer and sets
  `own_checkpoint_opponent_refresh_enabled=true`. Keep it labeled as "learner
  versus refreshed frozen own previous checkpoint," not true current-policy
  self-play. Do not use mutable `latest.pth.tar` refs.
- Verification after the own-latest/local-tooling patch:
  `146 passed, 3 skipped`; ruff clean for the touched trainer, manifest, and
  test files. The trainer app was redeployed at 2026-05-16 11:03 EDT, so new
  jobs use the patch; already-running jobs continue on their prior image.
- Launched a small own-latest moving-control lane after the deploy:
  `curvy-ownlatest-staticmix-20260516a`, selected rows `r007`, `r009`, `r011`.
  Manifest and launch record live under
  `artifacts/local/curvytron_tonight18_manifests/curvy-ownlatest-staticmix-20260516a/`.
  It deliberately has no assignment bank and no shared control refresh pointer.
  The trainer should create a run-local `runs:` pointer only after each row
  writes its first nonzero exact checkpoint.

## Open Next Actions

- Poll the new live-state and baseline/control comparison subagents and fold
  any fresh findings into this doc. Done for Kant, Feynman, and Mendel; keep
  following new subagents only if they have a fresh, bounded question.
- Continue monitoring `curvy-r18nofb-*` rows until nonzero checkpoints and evals
  exist, then compare survival retention against the tournament-feedback rows.
- Monitor `curvy-ownlatest-*` rows: first gate is iteration-0 startup; second
  gate is first nonzero checkpoint; third gate is a run-local own-checkpoint
  pointer plus applied refresh events and provider-ok env rows.
- Decide whether to patch launch-time overrides for target support scale,
  `source_max_steps`, `num_simulations`, and collect/batch cadence before the
  next real 18-run batch.
- Candidate next clean launch knobs if the no-feedback control supports the
  diagnosis: clean rows only, no stochastic override/repeat, static assignments
  or slower refresh, `collector_env_num` closer to `32` or `64`,
  `batch_size=128` or `256`, and `num_simulations=25` or `50`.
