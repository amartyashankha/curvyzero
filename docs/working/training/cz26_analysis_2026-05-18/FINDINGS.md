# Findings

This file records only findings from the current `cz26` analysis. It should not
repeat the entire r18fresh postmortem.

## Current State

CZ26 has a first deep local analysis pass. The canonical plain-language
synthesis is now:

```text
docs/working/training/cz26_analysis_2026-05-18/DEEP_ANALYSIS.md
```

The canonical generated report is:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
```

## Confirmed Setup Facts

- `cz26-full-20260517a` has `136` rows.
- Grid A has `96` rows.
- Grid B has `40` rows.
- There is no separate control/canary row inside the 136-run manifest.
- Grid B pure controls are:
  - `b100`;
  - `w100`;
  - `r1`.
- The separate `cz26c` canary is not part of the 136-run analysis.
- Every Grid A/B row uses the same pinned old r18fresh rank-1 initial checkpoint.
- A manifest-only analyzer pass succeeded and wrote:
  - `artifacts/local/cz26_analysis_2026-05-18/manifest_only_analysis.json`;
  - `artifacts/local/cz26_analysis_2026-05-18/manifest_only_analysis.md`.
- The current local workspace contains eval/status, tournament rating, joined
  analysis, matched contrasts, and a second-pass deep report.

## Tournament Snapshot Pull

Pulled from the tournament Volume:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_rating_latest.json
artifacts/local/cz26_analysis_2026-05-18/cz26_rating_progress.json
```

Facts from `cz26_rating_latest.json`:

- Latest completed rating snapshot: `round-000049`.
- Total rating rows: `4512`.
- CZ26 rating rows: `4508`.
- CZ26 distinct runs represented: `136 / 136`.
- Every CZ26 run has a latest checkpoint represented in the rating snapshot.
- Best CZ26 checkpoint reached tournament rank `2`.
- CZ26 checkpoint rows in top bands:
  - top 10: `6`;
  - top 30: `26`;
  - top 100: `96`.

Facts from the live-loop status snapshot:

- Current loop status: `rating_game_batch_active`.
- Latest completed rating has `4512` checkpoints.
- Active game batch has `4636` checkpoints, `300` pairs, and `6300` games.
- Active game batch was about `22.6%` complete at the sampled status.
- There were `181` checkpoints not yet in the latest completed rating at that
  moment.

Interpretation:

- The rating snapshot is suitable for the current learned-only tournament pass.
- It is not the final word because a newer active batch was running at sample
  time.
- Eval/status has now been pulled and joined, so survival and reward analysis
  can use local artifacts.

## Tournament-Only First Read

Using `scripts/analyze_curvytron_cz26_grid.py` against the manifest plus
`cz26_rating_latest.json`:

```text
artifacts/local/cz26_analysis_2026-05-18/rating_only_analysis.json
artifacts/local/cz26_analysis_2026-05-18/rating_only_analysis.md
```

Tournament-only signals so far:

- The top CZ26 entries in the raw rating table are mostly `iteration 0` seed
  checkpoints. Those are shared starting policies, not learned policies.
- For learning-quality analysis, use the learned-only tournament columns. In
  this file, learned means `iteration > 0`.
- Learned CZ26 checkpoint rows:
  - `4371` learned rows;
  - all `136` runs have learned checkpoints in the rating snapshot;
  - best learned CZ26 rank is `34`;
  - learned top 10: `0` rows;
  - learned top 30: `0` rows;
  - learned top 100: `36` rows.
- Grid A learned best ranks by reward setting:
  - `out0`: best learned rank `40`, `4` runs hit learned top 100;
  - `out33`: best learned rank `68`, `3` runs hit learned top 100;
  - `out67`: best learned rank `34`, `5` runs hit learned top 100;
  - `out100`: best learned rank `56`, `6` runs hit learned top 100.
- Grid A learned best ranks by noise:
  - `n20`: best learned rank `34`, `10` runs hit learned top 100;
  - `n0`: best learned rank `58`, `5` runs hit learned top 100;
  - `n10`: best learned rank `40`, `3` runs hit learned top 100.
- Grid A learned best ranks by recipe:
  - `b20w05r1`: best learned rank `34`, `3` runs hit learned top 100;
  - `b10w05r1`: best learned rank `40`, `5` runs hit learned top 100;
  - `b20w05top2`: best learned rank `60`, `5` runs hit learned top 100;
  - `b20w10r1`: best learned rank `56`, `5` runs hit learned top 100.
- Grid A leaderboard-immortal axis:
  - `imm0`: best learned rank `34`, `12` runs hit learned top 100;
  - `imm10`: best learned rank `56`, `6` runs hit learned top 100.
- Grid B produced no learned top-30 CZ26 rows in this snapshot. Its best learned
  run reached rank `65`.
- Grid B learned recipe readout is mixed:
  - `b25w25r1` has the best learned rank, `65`;
  - `b100` has best learned rank `78`;
  - `b20w05lad4` has best learned rank `87`;
  - `r1` is weak here, best learned rank `479`.

Do not over-read this yet. Tournament rank needs to be paired with survival
curves, reward curves, and completion horizons before making training-setting
recommendations.

## Joined Eval + Tournament Pull

Pulled:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_eval_status_latest.json
```

Joined output:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_joined_analysis.json
artifacts/local/cz26_analysis_2026-05-18/cz26_joined_analysis.md
```

Completeness:

- Eval rows: `136 / 136`.
- Tournament rows: `136 / 136` runs represented.
- Training status: `132` completed, `4` still running.
- Eval manifest count range: `10` to `191`.
- Common eval horizon:
  - all 136 rows: `30000`;
  - Grid A all rows: `170000`;
  - Grid A completed-only rows: `300000`;
  - Grid B all rows: `30000`;
  - Grid B rows with at least 300k eval coverage: `38 / 40`.

High-level survival shape:

- The average row improves from its first eval to its best eval, then drops by
  the latest eval.
- Grid A survival:
  - first mean `199.0`;
  - best mean `276.3`;
  - latest mean `157.5`;
  - latest/best retention `0.60`.
- Grid B survival:
  - first mean `199.9`;
  - best mean `336.2`;
  - latest mean `167.9`;
  - latest/best retention `0.55`.

So the pattern is not "nothing learned." It is "many rows find a better
intermediate policy, but the latest checkpoint often regresses."

Grid A survival at the all-row matched horizon (`170000`):

- Reward axis:
  - `out67` is highest matched survival: `211.8`;
  - `out100`: `189.2`;
  - `out0`: `175.9`;
  - `out33`: `157.2`.
- Noise axis:
  - `n10`: `186.8`;
  - `n0`: `182.5`;
  - `n20`: `181.3`.
- Immortality axis:
  - `imm0`: `191.8`;
  - `imm10`: `175.2`.
- Recipe axis:
  - `b20w10r1`: `194.6`;
  - `b10w05r1`: `183.8`;
  - `b20w05top2`: `183.1`;
  - `b20w05r1`: `172.7`.

Grid A survival at completed-only `300000` horizon:

- Reward axis:
  - `out0`: `160.7`;
  - `out100`: `157.5`;
  - `out33`: `153.2`;
  - `out67`: `148.1`.
- Noise axis:
  - `n10`: `167.4`;
  - `n20`: `159.4`;
  - `n0`: `137.7`.
- Immortality axis is basically flat:
  - `imm0`: `155.5`;
  - `imm10`: `154.5`.
- Recipe axis:
  - `b10w05r1`: `163.6`;
  - `b20w05r1`: `155.1`;
  - `b20w05top2`: `154.2`;
  - `b20w10r1`: `147.1`.

This is a real example of why endpoint choice matters: `out67` looks best at
170k but not at 300k. That can mean it peaks earlier and decays harder, not
that either read is "wrong."

Grid B survival:

- The all-row matched horizon is only `30000` because
  `cz26b-r028-out50-n10-imm10-b20w05r1` has only 30k eval coverage.
- On the richer 38-row 300k readout, the strongest survival recipes are:
  - `b20w05r1`: `242.2` across 3 covered rows;
  - `b30w05r1`: `211.4` across 3 covered rows;
  - `b50r1`: `208.9` across 4 rows;
  - `b20w05lad4`: `207.8` across 4 rows.
- The weakest pure recipe in Grid B is `r1`:
  - 300k survival `121.3`;
  - learned best tournament rank `479`.

Reward readout:

- Reward generally mirrors survival when the reward definition is survival-like.
- Do not compare raw reward across different outcome alphas as if it were one
  scale.
- Still, reward reinforces the broad pattern: best reward is much higher than
  latest reward, so many rows peak and then decay.
- Grid A reward latest/best retention is `0.45`.
- Grid B reward latest/best retention is `0.44`.

Tournament versus eval:

- Best eval survival has a weak but real relationship with learned tournament
  rank: correlation about `-0.22`, where lower rank is better.
- Best eval reward has a slightly stronger relationship with learned tournament
  rank: correlation about `-0.32`.
- Latest eval survival has almost no relationship with learned tournament rank:
  correlation about `-0.02`.

Interpretation: the tournament is often rewarding an intermediate checkpoint,
not the final checkpoint. That matches the visible peak-then-regress training
shape.

## Matched Contrasts

Generated:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_matched_contrasts.json
artifacts/local/cz26_analysis_2026-05-18/cz26_matched_contrasts.md
```

These compare rows where all other listed knobs match.

Grid A reward-alpha contrasts:

- At the all-row matched horizon, `out67` beats `out0` on survival by `+35.9`
  steps on average, winning `17 / 24` matched pairs.
- `out100` beats `out0` on matched survival by `+13.3`, winning `13 / 24`
  pairs.
- `out33` is worse than `out0` on matched survival by `-18.8`.
- Tournament learned-rank contrasts point differently:
  - `out67` is better than `out0` by `39.9` rank positions on average;
  - `out100` is better than `out0` by `111.8` rank positions on average;
  - `out33` is worse than `out0`.

Grid A noise contrasts:

- `n10` beats `n0` on matched survival by `+4.3` and latest survival by
  `+17.9`.
- `n20` is not better than `n0` on matched survival, but it is better on learned
  tournament rank versus both `n0` and `n10`.
- This says noise is not one clean monotonic knob: `n10` looks better in eval
  stability, while `n20` produced more tournament top-100 learned checkpoints.

Grid A leaderboard-immortal contrasts:

- `imm10` is worse than `imm0` on matched survival by `-16.6`.
- `imm10` is worse than `imm0` on learned tournament rank by `172.8` rank
  positions on average.
- Current evidence does not support 10% leaderboard immortality for Grid A.

Grid A recipe contrasts:

- Versus baseline `b20w05r1`, matched survival favors:
  - `b10w05r1` by `+11.1`;
  - `b20w10r1` by `+21.9`, but with an even `12 / 24` pair split;
  - `b20w05top2` by `+10.4`.
- Learned tournament rank favors `b20w05top2` versus baseline by `116.1` rank
  positions on average, despite weaker top-band headline rank.
- Recipe signals are therefore not settled by one table; survival and tournament
  are not ranking recipes identically.

Grid B matched contrasts:

- Pure `r1` is weak. Every tested alternative beats it on matched survival in
  most or all 4 matched blocks.
- Versus `r1`, learned tournament rank improves for:
  - `b20w05lad4` by `496.0` rank positions;
  - `b30w05r1` by `474.5`;
  - `b50r1` by `433.8`;
  - `w100` by `402.5`;
  - `b100` by `302.3`.
- `imm10` is worse than `imm0` on Grid B learned tournament rank as well.

Current read:

- Outcome reward is not simply bad. `out67` and `out100` both show useful
  tournament signals, while latest survival can still decay.
- The strongest consistent negative signal so far is leaderboard-opponent
  immortality at 10%.
- The strongest Grid B slot conclusion is that pure rank-1 opponents are not
  enough by themselves.
- A second-pass completed-only / exact-horizon report now exists because the
  all-row Grid B common horizon is dragged down to 30k by one low-coverage run.

## Second-Pass Deep Report

Generated:

```text
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.json
artifacts/local/cz26_analysis_2026-05-18/cz26_deep_report.md
```

This report adds:

- reward components and inferred outcome residual;
- exact-horizon tables at 30k, 170k, and 300k;
- exact-horizon matched contrasts;
- tournament exposure for top learned checkpoints;
- action-collapse watchlist;
- 136-row per-run table.

Plain read:

- The batch made real intermediate progress, then many latest checkpoints
  regressed.
- `out67` looks good around 170k but not at 300k.
- `out100` has useful learned-tournament signal but not stable survival.
- `out33` is weak in this batch.
- `n10` is the cleanest eval-survival noise setting.
- `n20` has some learned-tournament signal but worse survival stability.
- Grid A does not support `imm10`; Grid B is mixed but not a clean endorsement.
- Pure `r1` is weak in Grid B.
- Tournament rank fine ordering is not trustworthy yet because many top learned
  checkpoints have only 1-6 battles.
- Action collapse is common enough to track: 8 latest checkpoints collapsed,
  and 63 runs had at least one fully collapsed checkpoint.

## Remaining Questions

- Why do many rows peak and then regress?
- Are the top learned tournament rows actually strong, or are some ranks sparse
  exposure spikes?
- Does tournament feedback make training harder over time in a way that hides
  progress in raw reward?
- Should future loops preserve best intermediate checkpoints more aggressively
  instead of relying on latest checkpoints?
- Which Grid B recipes stay good after controlling for tournament exposure and
  the two low-coverage rows?
