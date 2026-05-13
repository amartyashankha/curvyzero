# CurvyTron Frozen Checkpoint Mix Plan

Date: 2026-05-12

## Superseded Scope

This plan describes frozen-opponent mixing inside the custom two-seat adapter.
It is historical/prototype guidance, not the current trusted learning path.

The useful idea that survives is different: stock `train_muzero` ego-vs-recent
frozen opponent may be a practical route if wired through the source-state
fixed-opponent path and labeled honestly.

## Current Goal

Keep the canonical CurvyTron two-seat LightZero trainer as the main path.
Add an optional opponent mix where most env rows remain current-policy self-play,
and some rows use a frozen older checkpoint as one opponent.

The key rule: only current-policy seats create learner replay rows. Frozen
checkpoint actions are environment actions only; they must not become learner
labels.

## First Safe Primitive

- Default stays pure current-policy self-play.
- Optional knob chooses the fraction of env rows that are current policy versus
  a frozen checkpoint opponent.
- The frozen opponent is chosen per env row episode/reset, not per step.
- Mixed rows use one frozen seat and one current-policy seat.
- Replay, return targets, and learner samples only include current-policy rows.
- Progress and records must say which rows were self-play and which were
  current-vs-frozen.

Implemented in the canonical path:

- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`

Current knobs:

- `--two-seat-frozen-opponent-probability`
- `--two-seat-frozen-opponent-checkpoint-ref` or the existing
  `--opponent-checkpoint-ref` fallback
- `--two-seat-frozen-opponent-player-id`
- `--two-seat-frozen-opponent-num-simulations`
- `--two-seat-frozen-opponent-use-cuda`

Safety tests now cover:

- Default self-play behavior still works.
- Mixed rows use the frozen checkpoint only for env actions.
- Frozen-controlled seats do not enter replay.
- Replay sampling rejects non-current-policy rows if they ever leak in.
- Modal payload forwards the frozen-opponent knobs.

## Reward Sanity Check

The current trainer reward is not only terminal win/loss:

- Alive step helper: default `+0.01`.
- Bonus pickup helper: default `+0.05` on the pickup step.
- Terminal outcome: env sparse outcome times `0.01 * episode_step_count`.
  Current env terminal winner rows use winner `+1` and loser `-1`, so this is
  `+0.01*T` for the winner and `-0.01*T` for the loser.

Open question: in pure current-policy self-play, symmetric policies may still
produce weak or brittle terminal signal. The frozen-checkpoint mix is partly a
diversity experiment for that reason, not just a performance tweak.

Research note from the 2026-05-12 critique:

- Self-play against the latest copy is not broken, but identical policies can
  make the terminal signal weak or cyclic in symmetric simultaneous games.
- Dense survival reward helps the agent stop dying immediately, but if it gets
  too large it can change the objective into "delay the end" instead of "win."
- Frozen checkpoint opponents are a reasonable way to break mirror-policy loops
  and test robustness against older strategies.

## Next Layer

Once the primitive is tested, add checkpoint-pool variants:

- Sweep coarsely across the important axes, not finely along one knob.
- Opponent mix ratio: include a no-frozen baseline, a small mix, a medium mix,
  and one aggressive mix.
- Checkpoint age: recent, halfway-back, and older/distinct checkpoints.
- Base config: mostly use the best clean base config, with only a small number
  of variants from learning-rate/reward/noise if current artifacts justify them.
- Use deterministic bucket counts per batch rather than noisy per-row sampling.
- Name runs clearly so the website and run list are readable.

The exact values are not settled. The batch should explore the space enough to
learn something tomorrow without making the artifact review impossible.

Current run-analysis note:

- Best clean bases from the 2026-05-12 matrix look like the `lr_1e-4` run and
  the default B64/sim8 seed family.
- The action-repeat/no-op run is interesting as an exploratory opponent, but it
  should not be the main base because that variant has extra accounting caveats.

## Recommended Batch After Canary

Launch one waited Modal canary first. Continue only if the canary shows strict
checkpoint load, nonzero mixed-row counts, no frozen-controlled replay rows, and
normal current-policy action/reward telemetry.

### Canary Result

Latest waited canary: `mixpast-canary-p30-lr1e4opp-i100-20260512c`.

Status: passed as a plumbing canary.

What it proved:

- The canonical Modal launcher can accept a frozen checkpoint ref and resolve it
  inside the remote container where `/runs` is mounted.
- The frozen opponent checkpoint loaded from explicit row18
  `iteration_100.pth.tar`.
- Training reached `ok=true`, saved `iteration_0`, `iteration_5`, and
  `iteration_6` checkpoints, and wrote a completed summary.
- `lightzero_policy_model_device` was `cuda:0`.
- Learner updates ran and changed model parameters.
- Opponent mix telemetry was present with both
  `current_policy_selfplay` and `current_policy_vs_frozen_checkpoint` rows.
- Replay stayed learner-side only: the canary's replay row count tracked current
  policy rows, and frozen-controlled seats were only env actions.
- Fresh policy actions were not collapsed in this short canary; both players had
  nontrivial action entropy in the training collection path.

Two launch bugs were found and fixed during the canary loop:

- Do not check Modal Volume checkpoint refs in the local entrypoint. Resolve and
  verify frozen checkpoint refs inside `_run_two_seat_selfplay_payload`, where
  `/runs` is mounted.
- Summary array stripping must handle string metadata arrays such as rollout
  labels. Numeric array summaries still include min/max/mean; string arrays now
  keep shape/dtype plus a short sample.

Additional pre-launch critique fixes after the passed canary:

- Reset resampling no longer forces at least one frozen row whenever a single
  env row resets. Initial batch assignment can still use a rounded bucket, but
  autoreset rows now get independent probability draws so configured `p=0.25`
  does not drift toward all-frozen under asynchronous deaths.
- Replay sampling is now strict: rows must explicitly say
  `action_source=current_policy` and `learner_controlled=true`.
- Frozen opponent refs must be immutable `iteration_*.pth.tar` checkpoints for
  two-seat mixed runs. Mutable `latest.pth.tar` and `ckpt_best.pth.tar` refs are
  rejected because they can silently change.
- Snapshot-backed opponent sidecars now include provider load metadata after
  lazy checkpoint load, so inspection can see which checkpoint state loaded and
  on which device.

Verification checks passed after the fixes:

- `uv run ruff check ...`
- `uv run pytest tests/test_curvytron_two_seat_render_mode.py tests/test_curvytron_live_checkpoint_eval_plumbing.py -q`
  -> latest focused result: `48 passed, 1 skipped`.

Remaining caveat: this is a correctness/plumbing canary, not learning evidence.
The next decision should be based on a small number of longer mixed runs plus
the existing overnight40 signal, not on this six-iteration run.

Second waited canary: `mixpast-canary-resetstress-p25-r27i200-20260512a`.

Purpose: stress reset behavior after the reset-resampling fix. This used short
`max_ticks=12` to force frequent deaths/resets, background eval/GIF disabled,
row27 `iteration_200` as the frozen checkpoint, and configured frozen ratio
`p=0.25`.

Status: passed as a reset/plumbing canary.

Readout:

- `ok=true`, no problems, learner updated, model parameters changed.
- Checkpoints saved at `iteration_0`, `iteration_10`, and `iteration_20`.
- Across all 20 iterations, frozen rows were `302 / 1280` row-steps
  (`23.6%`) for configured `25%`.
- Mean frozen rows at iteration end was `1.8 / 8` (`22.5%`).
- Latest replay action source counts were only `current_policy`; no frozen
  checkpoint rows entered learner replay.
- Latest replay rollout counts included both self-play and current-vs-frozen
  current-policy rows.

Interpretation: the reset skew bug is fixed for this stress case. Individual
iterations can still be noisy with `B=8` because a tiny batch can temporarily
have `0` or `5` frozen rows, but the aggregate ratio stayed near the configured
probability under frequent resets.

Mirrored audit canaries before the V1 launch:

- `mixpast-audit-p25-r27i250-lr1e4-f1-20260512a`
- `mixpast-audit-p25-r27i250-lr1e4-f0-20260512a`

Both used row27 `iteration_250`, configured frozen ratio `p=0.25`,
`num_simulations=8`, `lr=1e-4`, background eval/GIF disabled, and mirrored the
frozen seat.

Status: passed as a pre-launch audit.

Readout:

- Both runs finished `ok=true` with no problems.
- Both loaded the explicit immutable row27 `iteration_250.pth.tar` frozen
  checkpoint on `cuda:0`.
- Both ran learner updates and changed model parameters.
- Both wrote checkpoints at `iteration_0`, `iteration_6`, and `iteration_12`.
- Fresh current-policy collection did not show action collapse in either seat.
- Progress JSON now includes stratified replay/sample counters by rollout kind,
  player id, and action source.
- The learner-visible replay rows all had `action_source=current_policy`.
- With frozen player `1`, mixed learner rows appeared only as
  `current_policy_vs_frozen_checkpoint|player_0|current_policy`.
- With frozen player `0`, mixed learner rows appeared only as
  `current_policy_vs_frozen_checkpoint|player_1|current_policy`.

Interpretation: the static frozen-opponent primitive is wired well enough for
the V1 matrix. This still does not prove learning benefit; it only proves the
training data is not obviously polluted by frozen-policy labels.

## Launch Constraints From Critique

Treat the frozen mix as an opponent-diversity experiment, not as "more
self-play data."

Concrete constraints for the first serious batch:

- Keep batch/search/render/reward settings mostly fixed. The first readout should
  isolate frozen-opponent ratio and checkpoint age/source.
- Use explicit `iteration_*.pth.tar` frozen refs only.
- Read frozen ratio carefully. A `25%` env-row mix is only about `14%` of learner
  rows because self-play rows create two learner seats while current-vs-frozen
  rows create one.
- Pair important runs across frozen player seat `0` and `1`, or at least include
  mirrored seat sentinels. A fixed frozen seat can create false progress from
  seat/color/spawn artifacts.
- Include near-lag frozen opponents, not only widely separated checkpoints. A
  frozen opponent that is close to the current/source run's latest durable
  checkpoint may be a stronger and more relevant sparring partner than a very
  old checkpoint. For the one-checkpoint primitive, represent this with
  `iteration_200`/latest durable refs from strong rows where available, plus
  mid/old contrast runs.
- Current frozen checkpoint actions use the existing snapshot-backed LightZero
  eval-mode path. This is acceptable for a clearly labeled first probe, but it
  is not the same as sampled collect-mode opponent behavior. A collect-mode
  frozen opponent provider is a follow-up if this lane looks useful.
- Metrics must be stratified by rollout kind and frozen checkpoint ref. Raw
  reward sums are not comparable across frozen ratios without row-count context.

## Mixpast V1 Launch

Launch script:

```text
scripts/launch_curvytron_mixpast_20260512.zsh
```

All real runs keep CurvyZero checkpoint eval and GIF generation enabled through
the background poller at checkpoint cadence. Stock LightZero in-loop eval stays
off (`--lightzero-eval-freq 0`). The no-GIF/no-eval setting was only for
canaries.

Common settings:

- `gpu-l4-t4`, `fast_gray64_direct`, normal death.
- `batch_size=64`, `num_simulations=8`,
  `collect_steps_per_iteration=64`, `updates_per_iteration=4`,
  `learner_sample_size=256`.
- `max_train_iter=3000`, checkpoint every `50` iterations.
- No action skip/repeat stochasticity in this wave.
- Frozen opponent provider is the existing snapshot-backed LightZero eval-mode
  path. Names/docs should not imply sampled collect-mode league play.

This is still the one-checkpoint primitive, not the full dynamic "pick a
checkpoint from earlier in the same run" pool. It is useful because it tests
whether adding historical opponents from the current overnight40 population
helps or hurts, but it is not the final league design.

In this static one-checkpoint batch, "near" means the latest durable checkpoint
from the source lineage, not a moving same-run lag of the learner. A true
near-past opponent requires checkpoint-pool or dynamic lag selection.

Planned rows:

| id | run suffix | base | frozen p | frozen ref | frozen seat | purpose |
|---:|---|---|---:|---|---:|---|
| 01 | `baseline-r27-noobs-p0` | row27-like no obs noise | `0` | none | `1` | fresh baseline for row27-like base |
| 02 | `baseline-default-p0` | default | `0` | none | `1` | fresh default baseline |
| 03 | `r27near250-p10-f1` | row27-like | `0.10` | row27 `iteration_250` | `1` | near-lag low mix |
| 04 | `r27near250-p25-f1` | row27-like | `0.25` | row27 `iteration_250` | `1` | primary near-lag mix |
| 05 | `r27near250-p25-f0` | row27-like | `0.25` | row27 `iteration_250` | `0` | seat-balance pair |
| 06 | `r27near250-p50-f1` | row27-like | `0.50` | row27 `iteration_250` | `1` | aggressive mix stress |
| 07 | `r27mid100-p25-f1` | row27-like | `0.25` | row27 `iteration_100` | `1` | mid checkpoint |
| 08 | `r27mid100-p25-f0` | row27-like | `0.25` | row27 `iteration_100` | `0` | mid seat-balance pair |
| 09 | `r27old50-p25-f1` | row27-like | `0.25` | row27 `iteration_50` | `1` | older checkpoint contrast |
| 10 | `r27old50-p25-f0` | row27-like | `0.25` | row27 `iteration_50` | `0` | old checkpoint seat pair |
| 11 | `default-r04near150-p25-f1` | default | `0.25` | row04 `iteration_150` | `1` | default-family near-lag |
| 12 | `default-r04near150-p25-f0` | default | `0.25` | row04 `iteration_150` | `0` | default-family seat pair |
| 13 | `default-r05near150-p25-f1` | default | `0.25` | row05 `iteration_150` | `1` | default seed replicate |
| 14 | `default-r05near150-p25-f0` | default | `0.25` | row05 `iteration_150` | `0` | default seed seat pair |
| 15 | `default-r04mid100-p25-f1` | default | `0.25` | row04 `iteration_100` | `1` | default mid contrast |
| 16 | `baseline-r18-lr1e4-p0` | row18-like lr `1e-4` | `0` | none | `1` | fresh lr baseline |
| 17 | `r18near150-p25-f1` | lr `1e-4` | `0.25` | row18 `iteration_150` | `1` | continuity near-lag |
| 18 | `r18near150-p25-f0` | lr `1e-4` | `0.25` | row18 `iteration_150` | `0` | continuity seat pair |
| 19 | `r18near150-p10-f1` | lr `1e-4` | `0.10` | row18 `iteration_150` | `1` | continuity low mix |
| 20 | `r18near150-p50-f1` | lr `1e-4` | `0.50` | row18 `iteration_150` | `1` | continuity aggressive mix |

Launch state:

- Launched all 20 rows at `2026-05-12T07:44Z` with
  `scripts/launch_curvytron_mixpast_20260512.zsh`.
- Launch log: `logs/curvytron_mixpast-v1_launch_20260512.log`.
- All 20 Modal local-entrypoint commands returned `status=spawned`.
- Real runs keep CurvyZero background checkpoint eval/GIF enabled. Canaries used
  `--no-background-eval-enabled --no-background-gif-enabled`.
- Immediate volume check showed early rows had started before later rows. A
  delayed check confirmed sampled later rows, including rows 10, 16, and 20,
  had `show_in_gif_browser.flag` and initial `iteration_0` checkpoints.

Early monitor, `2026-05-12T08:08Z`:

- All 20 rows have browser markers and are visible to the GIF browser API.
- 19/20 rows had at least one iteration record in `progress.jsonl`; row 13 had
  a start record and browser artifacts but no iteration record in the sampled
  pull yet.
- Every row with an iteration record reported `problem_count=0`.
- Frozen-mix rows showed both current-policy self-play rows and
  current-policy-vs-frozen rows in replay. The learner sample/action-source
  counters still only train on `current_policy` rows.
- Fresh collection action summaries did not report collapse warnings in the
  sampled iteration records.
- Some progress files can contain a later `start` line after an iteration line
  when a detached function restarts; monitor by highest recorded `iteration`,
  not blindly by the final JSONL line.

Browser redeploy, `2026-05-12T08:09Z`:

- Redeployed `src/curvyzero/infra/modal/curvytron_gif_browser.py`.
- Verified API URL:
  `https://modal-labs-shankha-dev--curvyzero-curvytron-gif-browser--bada8e.modal.run/api/summaries?fresh=1&limit=1&run_id=mixpast-v1-01-baseline-r27-noobs-p0-20260512`.
- API check returned `mixpast_runs=20`.
- Checked run `mixpast-v1-01-baseline-r27-noobs-p0-20260512` had both
  `eval_greedy`/`raw.gif` and `collect_t1`/`collect_t1.gif` present for the
  sampled checkpoint.

Readout:

- Compare each mix against the fresh baseline for the same base config and the
  original overnight40 source row.
- Treat `p=0.25` as the candidate default only if it improves held-out evals and
  does not hurt current-policy self-play.
- Treat `p=0.50` as a stress test, not the default.
- Seat-pair disagreements are red flags, not noise to average away.
- Row30/action-repeat opponents are deferred. This V1 batch keeps the main
  comparisons cleaner by using the same source lineages for near/mid/old
  opponent age.

## Open Checks

- Literature/research check: opponent pools, fictitious self-play, league
  training, and AlphaZero-style checkpoint opponents.
- Follow-up code check: extend from one frozen checkpoint to a checkpoint pool or
  dynamic same-run historical opponent selection once the static primitive has
  evidence.

## Stop Note

At `2026-05-12T14:21Z`, all active detached
`curvyzero-lightzero-curvytron-visual-survival-train` Modal apps were stopped
along with the scheduled collect GIF subscriber. This includes the mixpast V1
jobs. Artifacts were intentionally kept in the `curvyzero-runs` Modal volume for
later inspection; only compute was stopped. The GIF browser deployment was left
running.
