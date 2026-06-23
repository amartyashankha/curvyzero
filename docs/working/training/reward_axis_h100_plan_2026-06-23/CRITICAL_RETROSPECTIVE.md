# Critical Retrospective

Status: synthesis from prior docs and current repo read. This is intentionally
blunt so we do not repeat old mistakes at higher scale.

## Past Lessons That Matter Now

### 1. Observability Was Often The Bottleneck

The r18fresh postmortem says the batch had many nearby artifacts but lacked one
stitched readout joining checkpoint write, intake, tournament, leaderboard
export, assignment hash, and trainer consumption. That made it too easy to
confuse activity with loop proof.

Implication: Wave A needs lane-local artifact ledgers. A live run without a
row-to-checkpoint-to-eval path is not an experiment yet; it is GPU activity.

Reference: `docs/working/training/r18fresh_postmortem_2026-05-16/FINDINGS.md`

### 2. The Learning Failure Was Usually Retention

r18fresh and CZ26 both show the same shape: many rows find better intermediate
checkpoints and then latest regresses. r18fresh mean survival rose from `160.2`
at iteration 0 to `189.6` at 240k, while per-run best averaged `251.3`. CZ26
Grid A had first/best/latest survival `199.0 / 276.3 / 157.5`; Grid B had
`199.9 / 336.2 / 167.9`.

Implication: latest checkpoint is not the promotion object. Best-so-far,
AUC, latest-vs-best retention, and tournament-selected checkpoints must be
tracked separately.

References:

- `docs/working/training/r18fresh_postmortem_2026-05-16/TREND_ANALYSIS.md`
- `docs/working/training/cz26_analysis_2026-05-18/DEEP_ANALYSIS.md`

### 3. Plus-Outcome Is Promising, Not Proven

r18fresh favored `survival_plus_bonus_plus_outcome` on integrated survival and
tournament top-band alignment, while no-outcome remained the clean control and
sparse stayed competitive at some matched horizons. CZ26 complicated the story:
`out67` looked best around 170k, but not at 300k; `out100` had tournament
signals but sparse exposure.

Implication: plus-outcome should be a main arm, but alpha/support/cadence
controls are mandatory. A plus-outcome win at one horizon can still be an early
peak with harder decay.

References:

- `docs/working/training/r18fresh_postmortem_2026-05-16/FINDINGS.md`
- `docs/working/training/cz26_analysis_2026-05-18/FINDINGS.md`

### 4. Raw Reward Misled Us

Own reward has different semantics by variant. Sparse reward is tiny and noisy;
no-outcome reward mostly tracks survival; plus-outcome reward includes a scaled
terminal residual that can wipe out survival reward. Bonus pickup barely moved
in r18fresh.

Implication: raw reward is a within-variant sanity signal, not the cross-variant
scoreboard. For plus-outcome, always inspect inferred terminal residual.

Reference:
`docs/working/training/r18fresh_postmortem_2026-05-16/REWARD_BREAKDOWN_ANALYSIS.md`

### 5. Opponent Recipe Can Dominate Reward

The clearest r18fresh survival effect was the `blank20-wall5-rank1_70-rank1imm5`
recipe. CZ26 also showed recipe effects that did not collapse to one simple
winner. Pure rank1 looked weak in Grid B; mixed recipes with blank/wall support
remained more interesting.

Implication: reward comparisons must not average away opponent recipe. The
static 18-row reward isolate is not waste; it is how we avoid a false reward
winner caused by one recipe.

References:

- `docs/working/training/r18fresh_postmortem_2026-05-16/TREND_ANALYSIS.md`
- `docs/working/training/cz26_analysis_2026-05-18/DEEP_ANALYSIS.md`

### 6. Tournament Is Useful And Easy To Overread

Tournament rows rescued some mid-run policies, and tournament game duration rose
over r18fresh rounds. But tournament rank was only moderately aligned with
survival, often had sparse exposure, and CZ26 raw top rows were polluted by
shared `iteration_0` seeds unless learned rows were separated.

Implication: tournament is a promotion input only after exposure is visible and
seed checkpoints are labeled or excluded.

Leaderboard implication for the new campaign: tournament/leaderboard comes
later. It should select and rank nonzero checkpoints after static training has
produced candidates; it should not be the first curriculum source or a Wave A
launch blocker.

References:

- `docs/working/training/leaderboard_to_training_2026-05-13/tournament_elo_trajectory_2026-05-16.md`
- `docs/working/training/cz26_analysis_2026-05-18/FINDINGS.md`

### 7. RND Was Real, But Not Promoted

RND was implemented in the trainer process, with `none`, `rnd_meter_v0`, and
`rnd_replay_target_v0`. The old blank sweep launched a 9-point ladder, then a
fast-checkpoint replacement reached trained checkpoints, evals, GIFs, and valid
JSONL RND metrics. Focused tests and a broader RND-adjacent suite passed.

The reason it did not get promoted was not "RND missing." It was:

- positive RND changed replay targets but lacked retained-quality proof
- running/global normalization was unresolved
- RND resume/checkpoint state was incomplete
- one seed was not enough
- blank-canvas survival was not game-strength proof
- latest snapshot metrics had a writer issue, even though JSONL was valid

Implication: reopen RND aggressively, but call it an experimental lane. The old
"blocked" label should apply to promotion, not to controlled H100 exploration.

Reference:
`docs/working/training/exploration_bonus_rnd_2026-05-19/RND_BLANK_SWEEP_RUN_2026-05-19.md`

### 8. Speed Progress Was Real, But Separate

The compact speed lane made real support progress, but the accepted baseline is
still OPT-104 at `12689.38 env/s`. The fastest single support row reached
`15852.67 env/s`, about `1.25x`, and repeat/long evidence was more modest. That
is not a repeatable 2x and not a learning-quality result.

Implication: compact speed, compact reward contract, and RND must stay separate
in claims. A faster support row does not make RND safer or plus-outcome better.

Reference:
`docs/working/optimizer/reorientation_2026-05-23/CLEANUP_COMMIT_PLAN_2026-06-23.md`

## Mistakes To Not Repeat

- Calling implementation proof a learning result.
- Treating "running" or "checkpoint exists" as health complete.
- Treating latest checkpoint as the policy, when best and tournament-selected
  checkpoints tell a different story.
- Comparing raw reward across reward variants.
- Letting a broad sweep answer no clean question.
- Launching a partial row subset by accident.
- Trusting tournament rank without exposure counts.
- Reading shared `iteration_0` seeds as learned policies.
- Treating stale docs as current truth because they are detailed.
- Blocking all progress on one imperfect lane when a smaller honest proof can
  run in parallel.

## Reorientation From The Past

The old cautious RND posture made sense when compute was scarce and the goal was
not to contaminate the reward-axis read. With broad, timeboxed H100 capacity,
and narrower 2-hour-plus and 8-hour-plus caps, the better move is different:

- keep RND isolated
- include stock and meter controls
- replicate enough to see seed noise
- read RND metrics and survival together
- promote only after fixed-opponent and retention checks

The old reward-axis caution also needs updating. We should not wait for a tiny
three-row isolate before preparing broad lanes. We should prepare broad lanes
whose internal controls are clean.

## What Success Would Look Like

Good near-term success is not "we found the final reward." Good near-term
success is:

- RND sweep tells us whether low positive weights deserve fixed-opponent tests.
- Static reward isolate tells us whether plus-outcome still beats no-outcome
  and sparse under no-refresh controls.
- Cadence/support panel tells us whether regression is reward or training
  dynamics.
- Static exact-ref recipe lanes tell us whether production-style curricula are
  alive before leaderboard feedback is reintroduced.
- The docs can tell a future operator exactly what was launched, what was
  healthy, what learned, what regressed, and what got promoted.

That is the standard before spending the next wave.
