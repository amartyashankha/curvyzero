# Findings

Use this file for evidence-backed conclusions from the r18fresh postmortem. Keep raw notes and partial observations in `EXPERIMENT_LOG.md` until they are checked.

## Confirmed

### Observability, not effort, was the bottleneck

- Evidence: parallel critiques agreed that r18fresh had many nearby signals:
  eval summaries, checkpoint artifacts, assignment refresh events, tournament
  rating rows, battle summaries, and env telemetry. The missing piece was a
  single stitched readout joining those signals by checkpoint and assignment
  generation. The smallest concrete version is an append-only
  `lineage_events.jsonl` with one row per boundary transition.
- Interpretation: the batch was hard to reason about because proving the full
  loop required manual artifact chasing. That made it too easy to confuse
  dashboard row counts, latest checkpoints, active tournament rows, assignment
  exports, and trainer consumption.
- Impact: the next batch needs a durable lineage table before scale, not just
  more monitoring. Each checkpoint should be traceable from trainer write to
  intake, tournament play, leaderboard/export generation, assignment SHA, and
  provider-load evidence in the trainer.
- Follow-up: build or run a compact full-loop proof command that prints one
  row per checkpoint/export generation, then use it as the launch gate for
  any large batch.

### Operator surfaces need one lifecycle story

- Evidence: the tournament/GIF browser questions kept recurring around the
  same ambiguity: which arena is current, why raw row counts exceed active
  rows, which rating round is latest, which export generation trainers see,
  and whether a checkpoint is missing, queued, scheduled, played, active,
  exported, or loaded.
- Interpretation: the UI exposed fragments of truth without one state machine.
  That made dashboards useful for noticing problems but poor for explaining
  them.
- Impact: add one operator status readout or endpoint that stitches trainer
  writes, intake, tournament rounds, exports, assignments, and trainer
  provider-load proof. Both tournament and GIF browser should link/filter by
  the same checkpoint identity.
- Follow-up: implement the health strip and per-checkpoint lifecycle view
  described in `OBSERVABILITY_PLAN.md`.

### Learning signal and loop proof are separate questions

- Evidence: r18fresh found stronger mid-run checkpoints in every row, while
  latest checkpoints often regressed. Separately, the code/tests can prove
  that checkpoints, tournament exports, and trainer assignments are wired, but
  that proof does not say the policies are improving.
- Interpretation: "the loop works" means data flows through the system. "The
  experiment works" means that flow produces better policies. These must be
  measured separately.
- Impact: future status updates should always split mechanical loop health from
  learning quality. A green pipeline with flat survival is still a failed
  experiment; a promising survival curve with missing tournament/export proof
  is still an unproven loop.
- Follow-up: use `SIGNALS.md` as the checklist for both halves before launch,
  during long sleeps, and after wake-up.

### The tournament display is not the trainer top-100 source

- Evidence: the current rating latest has `564` rows, `100` active, `0`
  provisional, and `464` retired. The public/trainer snapshot read at
  generation 22 had `563` rows but also exactly `100` active rows.
- Interpretation: the website/API can show raw rating rows up to its page cap.
  Trainer bootstrap should filter active rows, not use every displayed row.
- Impact: preserve top candidates from active rows. Do not treat 500 visible
  rows as a broken top-100 truncation by itself.
- Follow-up: use the old-overnight rank-1 checkpoint as the shared initial
  policy seed for every next-batch trainer; keep raw/deduped top lists only as
  audit/opponent material.

### Matched eval comparison changes the learning readout

- Evidence: all 18 runs have evals through `iteration_240000`. At that matched
  point, mean survival is `189.6`; `sparse_outcome` is `197.6`,
  `survival_plus_bonus_plus_outcome` is `196.0`, and
  `survival_plus_bonus_no_outcome` is `175.1`.
- Interpretation: latest-only comparisons were too noisy. Sparse and plus
  outcome are close on matched survival, while no-outcome is weaker.
- Impact: next-batch reward choices should not be based only on latest
  checkpoint survival.
- Follow-up: compute AUC and retention on the common grid for each run.

### The strongest recipe signal is `blank20-wall5-rank1_70-rank1imm5`

- Evidence: at matched `iteration_240000`, that recipe has mean eval survival
  `239.6` and normalized AUC `198.5`, versus `173.1`/`161.3` for
  `blank10-wall10-rank2_25-rank1_55` and `156.0`/`154.2` for the ladder
  recipe.
- Interpretation: this recipe is the clearest survival-favored candidate.
- Impact: keep it in the next batch and expand around it with fresh seeds.
- Follow-up: compare its tournament strength after deduping sibling checkpoints
  from the same run.

### Plus-outcome has the best matched-grid survival slice among reward arms

- Evidence: at the 0..240k matched grid, plus-outcome has AUC `179.5`, latest
  mean `215.1`, and common-grid drop `-45.0`. Sparse has AUC `169.9`, latest
  `162.2`, and drop `-46.6`. No-outcome has AUC `164.6`, latest `181.7`, and
  drop `-60.4`.
- Interpretation: sparse is competitive at exactly 240k, but plus-outcome has
  the better integrated curve and tournament top-band alignment.
- Impact: plus-outcome should be a main next-batch arm. Sparse should stay as a
  diagnostic. No-outcome should not be primary without a new reason.
- Follow-up: inspect reward/value support scaling before expanding dense reward
  variants.

### Own reward confirms the retention problem

- Evidence: latest reward is within 10% of each run's best reward in only
  `1/18` runs. Latest survival is within 90% of best in only `3/18` runs.
- Interpretation: this is not just a tournament-ranking artifact. The policies
  usually pass through a better intermediate checkpoint and then get worse
  under their own reward/eval signal.
- Impact: next training should promote or preserve best-so-far checkpoints, not
  latest checkpoints.
- Follow-up: add best-checkpoint retention gates and inspect learner instability
  around the point where reward drops.

### Bonus reward did not matter in this batch

- Evidence: bonus AUC is near zero for all variants; no-outcome average bonus
  AUC is `0.023`, plus-outcome average bonus AUC is `0.009`, and sparse is zero
  by design.
- Interpretation: the runs were mostly learning survival/outcome behavior, not
  meaningful bonus pickup behavior.
- Impact: do not infer much from the bonus component in this batch. If bonuses
  matter for the next experiment, they need better observability or curriculum
  pressure.
- Follow-up: decide whether bonus mechanics should remain in the reward variant
  set or become a later focused diagnostic.

### Tournament rank is only moderately aligned with eval survival

- Evidence: joining current `round-000035` rating rows to eval checkpoints gives
  `572` matched rows. Rating correlation is `0.431` with eval survival across
  all rows and `0.302` among active rows. Rating correlation with own reward is
  near zero overall and `0.066` for plus-outcome.
- Interpretation: tournament strength is not reducible to either survival eval
  or own reward. It is selecting head-to-head behavior that may survive well
  enough and win matchups, even when the own reward scalar is volatile.
- Impact: preserve tournament-best, survival-best, and own-reward-best for
  audit and possible opponent seeding. For initial policy weights, the current
  launch decision is simpler: seed every run from the tournament rank-1
  checkpoint.
- Follow-up: keep the diverse candidate list separate from
  `initial_policy_checkpoint_ref`.

### Tournament top strength favors plus-outcome and mid-run checkpoints

- Evidence: current tournament rank 1 in `round-000035` is a
  `iteration_260000` checkpoint from
  `survival_plus_bonus_plus_outcome / blank20-wall5-rank1_70-rank1imm5 /
  so10rep10`. Plus-outcome owns all current top-10 rows in the latest detailed
  readout, and r018 alone contributes most of the top band.
- Interpretation: the tournament is rescuing best-so-far checkpoints from runs
  whose latest checkpoints often regress.
- Impact: next promotion should use tournament-selected or heldout-best
  checkpoints, not latest-only.
- Follow-up: do not mix initial-policy seeds for the next launch; use the
  rank-1 checkpoint everywhere and vary the opponent curriculum.

### The current champion run is r018

- Evidence: r018
  `curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423`
  has best tournament rank `#1` at `iteration_260000`, latest tournament rank
  `#6` at `iteration_270000`, and `16` checkpoints in the top 100. Its eval
  curve has best survival `287 @ 190k`, latest eval `290 @ 270k`, and a current
  tournament rating `1663.5` for the rank-1 checkpoint.
- Interpretation: this is the strongest single trajectory in the batch, and it
  did not collapse at latest in the same way many other runs did.
- Impact: preserve it as the launch seed and as a champion candidate. Do not
  let its sibling checkpoints silently become different initial-policy seeds.
- Follow-up: if we use siblings or challengers, use them as explicit opponent
  material or audit rows, not hidden bootstrap variation.

### The current ranking is useful but not stable

- Evidence: rating latest read at `round-000035` reports `stable=false` and
  `max_abs_delta=87.80`.
- Interpretation: active coverage is good enough for exploratory seeding, but
  the Elo ordering can still move.
- Impact: use current top candidates as bootstrap material, not final truth.
- Follow-up: pin the exact snapshot used for any next-batch seed.

### Tournament game duration increased over rounds

- Evidence: battle-summary aggregation across rounds `0..35` gives weighted
  mean duration `131.16` physical steps in round 0 and `162.20` in round 35.
  The first five rounds average `131.59`, the last five average `159.78`, and
  round index versus duration correlation is `0.945`.
- Interpretation: the tournament pool is moving toward longer games over time,
  even though many individual trainer latest checkpoints regress.
- Impact: tournament selection is preserving some useful intermediate policies.
  Do not collapse the analysis to latest trainer checkpoints only.
- Follow-up: compare tournament duration trend against active-pool composition
  after the top-1 shared seed is verified and opponent materials are pinned.

### The main training failure mode is retention, not total absence of signal

- Evidence: mean survival rises from `160.2` at iteration `0` to `189.6` at the
  matched `240k` checkpoint, but mean latest survival is only `186.4` while the
  mean per-run best is `251.3`. Latest survival is within 90% of best in only
  `3/18` runs; latest own reward is within 10% of best in only `1/18` runs.
- Interpretation: most runs discover a better intermediate policy and then
  regress later.
- Impact: analysis should separate best-so-far, latest, and tournament-selected
  checkpoints. Treat latest-only conclusions as suspect.
- Follow-up: keep trend summaries in `TREND_ANALYSIS.md` and use the manifest
  fields for all matched comparisons.

### The strongest single knob-level effect is the `b20/r1imm` opponent recipe

- Evidence: matched-grid survival AUC is `199.1` for `b20/r1imm`, versus
  `161.0` for `r2/r1` and `154.3` for `ladder`. At `240k`, `b20/r1imm` averages
  `239.6` survival, versus `173.1` and `156.0`.
- Interpretation: the opponent mixture matters more than the average reward or
  noise effects in this batch.
- Impact: do not average this effect away when comparing reward variants.
- Follow-up: maintain one-knob comparisons in `TREND_ANALYSIS.md`.

### The own-latest control shows the same retention shape, but is still early

- Evidence: selected `curvy-ownlatest-staticmix-20260516b` rows r007/r009/r011
  have only `5`, `5`, and `8` eval points so far. Their best survival values
  are `200.9`, `183.2`, and `202.6`, while their latest survival values are
  `169.5`, `134.2`, and `148.0`.
- Interpretation: the control lane has not run long enough for a final
  comparison, but it already shows better intermediate checkpoints followed by
  worse latest checkpoints.
- Impact: tournament feedback is not the only obvious suspect for regression.
  Keep the own-latest control separate and let it mature before treating it as
  a decisive control.
- Follow-up: rerun the control eval pull after more checkpoints and compare it
  to the r18fresh no-outcome clean rows at matched iterations.

## Candidates To Validate

- Per-run learning quality and failure modes.
- Tournament ranking stability and matchup-specific weaknesses.
- Whether the 18-run batch is still producing useful signal.
- Whether `ownlatest` `20260516b` remains a useful running control.
- Which next-batch design changes are supported by evidence.

## Finding Template

### Short finding title

- Evidence:
- Interpretation:
- Impact:
- Follow-up:
