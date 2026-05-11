# Training Evaluation

Status: Partially implemented for dummy survival, Tiny Line Duel, and Pong baselines
Date: 2026-05-08

## Short Answer

Evaluate progress with small, repeatable opponent ladders and fixed seed sets.
For the current coach lane, the core question is not "is this agent strong?"
but "is training improving at a believable rate without hiding bugs?"

Use:

- deterministic eval artifacts for every training run;
- random and sticky-random floors;
- one or two scripted heuristics;
- fixed checkpoint opponents once policies exist;
- old-vs-new checkpoint comparisons only after single-baseline evals are sane.

Do not build a league, Elo service, exploitability suite, or coverage dashboard
yet. Those are useful later, but the first evaluator should be a boring table
that catches regressions in survival rate, win rate, truncations, collision
causes, action collapse, seat bias, and wall-clock/sample progress.

Current implementation notes:

- EVAL1 exists as `scripts/run_dummy_survival_eval.py`.
- EVAL2 exists as `scripts/run_dummy_line_duel_eval.py`.
- Both write `summary.json` and `episodes.jsonl` artifacts.
- Learned-checkpoint loading exists for both dummy harnesses via explicit
  `--checkpoint-policy learned:path/to/checkpoint.npz` inputs.
- Pong scoreboards, self-play replay, and self-play training exist locally, but
  the current Pong artifacts are plumbing and negative evidence, not policy
  quality. Generation 2 lost to its parent and won 0 games against
  `track_ball`.
- Next eval step is fixed-baseline eval around the critique decision: repair
  the crude self-play trainer or switch to a known simple baseline/curriculum.
  It is not ratings, a league, or more generations by default.

## Source Takeaways

- MuZero reports progress as evaluation curves through training, not just a
  final score. Its model predicts reward, policy, and value for planning, and
  it was evaluated across Atari plus Go/chess/shogi. For CurvyZero, mirror the
  habit of checkpointed evaluation curves, not the scale of the benchmark.
  Source: [MuZero arXiv](https://arxiv.org/abs/1911.08265) and
  [Nature](https://www.nature.com/articles/s41586-020-03051-4).
- AlphaZero is the relevant self-play pattern: MCTS-guided self-play, trained
  from game outcomes and search policies, with performance tracked against
  strong external/reference players and Elo-style curves. For CurvyZero, fixed
  opponents and fixed seeds are enough before Elo. Source:
  [AlphaZero arXiv](https://arxiv.org/abs/1712.01815).
- AlphaGo Zero used old-vs-new network evaluation before promoting a new
  self-play generator. That is a useful later checkpoint-promotion pattern, but
  too much ceremony before we even trust random/scripted baselines. Source:
  [AlphaGo Zero Nature](https://www.nature.com/articles/nature24270).
- OpenSpiel is a good reminder that exploitability, AlphaRank, and game-theory
  metrics exist for multi-agent games, including simultaneous and general-sum
  settings. For this project, keep those as future tools; they are not needed
  to tell whether dummy survival or Tiny Line Duel is learning. Sources:
  [OpenSpiel paper](https://arxiv.org/abs/1908.09453) and
  [OpenSpiel docs](https://openspiel.readthedocs.io/en/latest/intro.html).
- Later large-scale multiplayer systems such as AlphaStar used populations,
  league diversity, and targeted evaluation. The useful idea is opponent
  diversity, not a v0 league implementation. Source:
  [AlphaStar Nature](https://www.nature.com/articles/s41586-019-1724-z).

## Current Critique / Corrections

Date: 2026-05-09

Hard critique: the current eval lane is useful infrastructure, but it is not
yet strong evidence that training is improving the policy. The dummy survival
smokes show exactly the failure mode the evaluator must guard against:
`one_step_safe` already solves the fixed seed-123 smoke, and the learned
checkpoint is executed through the same deterministic `DummyPlanner` with
`epsilon=0.0` and `explore_unknown=False`. After the safety-aware planner patch,
early checkpoints matched `one_step_safe` on only 10 eval episodes, while later
checkpoints degraded. That proves checkpoint loading and sweep mechanics work,
but it does not prove the model update learned a robust survival policy.

The correction is to treat every learned-checkpoint number as a decomposition:

- planner prior / hand-coded safety signal;
- learned table/model contribution;
- seed-set luck;
- checkpoint-selection luck.

Until those are separated, say "checkpoint plus planner scored X on this eval
split", not "training learned X".

Immediate changes to the eval contract:

- Add a `planner_only` or `untrained_model_same_planner` baseline wherever a
  learned checkpoint is evaluated. If the checkpoint cannot beat that baseline,
  the eval is measuring planner priors, not learning.
- Add a `learned_no_safety_tiebreak` or equivalent ablation before claiming that
  the model update matters. This can be later if it is annoying, but the docs
  and summaries should mark the gap.
- Report improvement over both `random_uniform` and `one_step_safe`. Beating
  random is a floor; approaching or beating the scripted heuristic is the first
  useful toy-learning gate.
- Keep action histograms and terminal causes mandatory. A survival/win-rate bump
  with action collapse, timeout farming, or seat skew is not a clean improvement.
- For Tiny Line Duel, aggregate paired seat groups, not just individual
  matchups. Same seed with swapped seats is the unit that catches seat-order
  bugs.

### Episodes And Seeds

Use small episode counts only for plumbing smokes:

| Use | Minimum now | What it can support |
| --- | ---: | --- |
| CLI/artifact smoke | 5-10 episodes per policy/match | The command runs and writes sane rows. No quality claim. |
| Baseline sanity | 25-50 episodes per policy/match | Random/scripted behavior is roughly stable; obvious regressions show up. |
| Checkpoint selection candidate | 100 paired seed groups or 100 solo episodes | Pick a candidate for more eval; do not publish as robust. |
| Strong toy claim | 400+ games/episodes on selection split plus held-out confirmation | "This checkpoint is better on this toy task" with uncertainty reported. |

For binary survival/win rates, 10 episodes are especially misleading. Seeing
10/10 successes only says the failure rate might be low on those 10 seeds; by
the rough "rule of three", zero observed failures in `n` trials leaves an
approximately `3/n` upper bound on the failure rate. That is about 30% at
`n=10`, 6% at `n=50`, and 3% at `n=100`. Use exact/binomial or bootstrap
intervals in summaries once we make claims.

For training-run comparisons, separate environment episode count from
independent training seeds. One training seed with many eval episodes can show
a local regression or candidate. It cannot establish algorithmic improvement.
Use at least 3 independent training seeds for internal decisions and 5-10+ for
claims about one training method beating another.

### Seed Splits

Every eval artifact should name its split, seed list/hash, and intended use.
Do not reuse one fixed `seed=123` eval for every purpose.

- `train`: used for rollout, exploration, curriculum, replay, and online
  training noise. Never used for checkpoint selection or final reporting.
- `monitor`: tiny fixed canary set used during training for curves and
  regression detection. It is allowed to become familiar and biased; label it
  that way.
- `selection`: fixed dev/eval set used to choose `best` checkpoint labels. Do
  not tune planner safety, hyperparameters, or eval thresholds repeatedly on the
  same set without rotating or versioning it.
- `heldout`: fixed but untouched confirmation set run after selecting a
  checkpoint. Use it to decide whether the selection result was seed luck.
- `debug`: hand-picked replayable failures. Never blend these into aggregate
  score claims.

The current dummy survival seed-123 sweep should be reclassified as `monitor`
or `debug`, not `heldout`, because it has already influenced interpretation and
planner changes.

### Best-Checkpoint Selection

Best-checkpoint selection is necessary because final checkpoints already
degrade in dummy survival, but it can overfit just like hyperparameter tuning.
Use a two-stage rule:

1. Rank periodic checkpoints on the `selection` split using a predeclared score:
   survival/win rate first, then mean steps/reward, then action/terminal sanity.
2. Freeze the chosen checkpoint and run `heldout` once. Report both selection
   and held-out tables. If held-out does not confirm the ranking, keep the run
   as inconclusive instead of fishing for another checkpoint.

Avoid making `best_vs_random`, `best_vs_heuristic`, and `best_overall` all from
the same tiny seed set. If multiple labels matter, either use separate
selection splits or make one explicit scalar score. Keep `latest` in every table
so checkpoint cherry-picking is visible.

### Next Contract Before Serious Runs

Local is for tiny debug only. Before spending on serious Modal jobs, the
evaluator should produce this minimum contract:

```text
run_id
checkpoint_id/checkpoint_path
policy_execution_spec: planner id, planner config, epsilon, temperature,
                       search simulations, safety masks, action tie-breaks
train_seed_spec: split id, base seed, generated seed hash
eval_seed_spec: split id, base seed or explicit list hash, episode count
opponent_specs: policy/checkpoint ids, versions, deterministic/stochastic flags
rules/schemas: ruleset id/hash, observation schema, reward schema, action schema
per_episode_rows: seed, seat, opponent, outcome, reward, length, terminal cause,
                  truncation flag, action histogram
aggregate_summary: point estimates plus intervals where claims are made
selection_record: selection split, ranking metric, selected checkpoint,
                  held-out confirmation path
```

Immediate before-large-run gates:

- Survival: learned checkpoint plus planner beats `untrained_model_same_planner`
  and random on `selection`, and the selected checkpoint remains above those
  floors on `heldout`.
- Survival: the selected checkpoint is compared to `one_step_safe`; matching it
  on a 10-episode monitor set is not enough.
- Tiny Line Duel: `one_step_safe` beats random/sticky on paired seats; a learned
  checkpoint beats random/sticky without seat bias, timeout farming, or action
  collapse on `selection`, then confirms on `heldout`.
- All summaries include split ids and enough policy execution metadata that a
  future reader can tell whether the eval used raw policy logits, a planner, a
  safety mask, MCTS simulations, or a scripted fallback.

Source alignment:

- AlphaGo Zero promoted a stronger self-play generator only after old-vs-new
  evaluation, using a win-rate threshold to avoid selecting on noise. We should
  borrow the idea of a promotion gate, not the scale. Source:
  [AlphaGo Zero Nature](https://www.nature.com/articles/nature24270).
- AlphaZero and MuZero report progress through training with checkpointed
  evaluation curves and external/reference comparisons, not only final
  checkpoint scores. Sources: [AlphaZero arXiv](https://arxiv.org/abs/1712.01815)
  and [MuZero arXiv](https://arxiv.org/abs/1911.08265).
- OpenSpiel's AlphaZero implementation separates actors, learner,
  evaluators, checkpoints, config, and machine-readable logs; that maps well to
  CurvyZero's desired artifact contract. Source:
  [OpenSpiel AlphaZero docs](https://openspiel.readthedocs.io/en/latest/alpha_zero.html).
- General RL empirical-design guidance warns that random seeds,
  implementation details, hyperparameters, and weak statistics can make
  improvements look real when they are not. Sources:
  [Deep RL That Matters](https://arxiv.org/abs/1709.06560) and
  [Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html).

## 2026-05-09 Deeper Review: AlphaZero/MuZero-Style Eval For Pong

This update uses local notes and the source links already captured above; no
new web fetch was needed. The practical question is whether each Pong training
attempt should evaluate only against fixed baselines, or also against older
generations.

Short answer: yes, latest-vs-old-generation eval is a good idea, but only as a
secondary regression and promotion check. It should not replace fixed baselines
or heldout seeds. For the current Pong lane, the main scoreboard should remain
`random_uniform` and `track_ball` on fixed seed sets. Add older checkpoints once
there are real periodic policy checkpoints, not while the learned policy is
still mostly imitation/value-target plumbing.

Current artifact correction: the local self-play loop exists, but gen2 failed.
It lost to the parent and got 0 wins against `track_ball`. Treat it as a
hypothesis/scaffold. Plumbing that writes replay, checkpoints, Modal refs, or
scoreboards does not prove the policy is better.

### What The AlphaZero/MuZero Pattern Suggests

At a high level:

- AlphaGo Zero used old-vs-new network games as a promotion gate before a new
  network became the self-play generator. The useful idea is "do not let a worse
  generator take over just because it is newer."
- AlphaZero and MuZero report progress through checkpointed evaluation curves
  against reference opponents or benchmark scores. The useful idea is "track
  progress over training checkpoints under a stable eval protocol."
- MuZero-style training makes policy execution part of the eval object. The
  same checkpoint can behave differently with a different search budget,
  temperature, action mask, or fallback. Eval rows must record that execution
  spec.
- Later league/population systems use opponent pools and ratings when pairwise
  tables get too large or policies cycle. That is not the v0 need here.

For CurvyZero, borrow the habits, not the ceremony:

- save periodic checkpoints;
- evaluate them on named seed splits;
- always show fixed baselines;
- show `latest` next to `selected_best`;
- use old-vs-new only after the baseline table is trustworthy.

### Latest Vs Older Checkpoints

Use latest-vs-old when:

- self-play quality depends on choosing the next generator;
- final checkpoints sometimes regress, as dummy survival already showed;
- a policy may forget how to beat random/scripted opponents while improving
  against its recent self;
- the fixed baseline table is already sane and stable;
- the comparison uses a fixed seed set, a predeclared sample count, and paired
  seats for two-player games.

Do not use latest-vs-old as the first or only metric. A new policy can beat an
older weak policy and still be worse than `track_ball`, worse on heldout seeds,
or worse than the selected best checkpoint.

For Pong, a monitor improvement alone does not promote a child checkpoint. A
child is only a candidate if it beats the parent and preserves or improves the
fixed-baseline rows. Heldout confirmation is required before a quality claim.

Risks:

- Cyclic or non-transitive strategies: A beats B, B beats C, and C beats A.
- Overfitting to the checkpoint pool: training learns to exploit stale
  opponents, not to play better Pong.
- Noisy small samples: a tiny seed set can make promotion look real.
- Latest is not always best: the final checkpoint can be worse than an earlier
  selected checkpoint.
- Moving target confusion: changing search budget, max steps, reward schema, or
  seed split can look like policy progress.
- Pool bias: if the pool contains only weak or similar policies, beating the
  pool says little.

Mitigation is simple for now: keep the fixed baseline table as the scoreboard,
run older-checkpoint rows only for `previous`, `selected_best`, and maybe one
age-spaced checkpoint, and require heldout confirmation before calling a
selected checkpoint better.

### Seed Sets

Use three seed roles for Pong:

- `monitor`: small fixed set run on every attempt. It catches wiring regressions
  and obvious behavior shifts. It is allowed to become familiar.
- `selection`: larger fixed set used to pick `selected_best` among periodic
  checkpoints. The metric must be chosen before the sweep.
- `heldout`: separate fixed set run after selection. It confirms or rejects the
  selection result.

Do not retune policy code or thresholds repeatedly against `heldout`. Once a
heldout result changes a design choice, rotate/version the split before using it
as fresh evidence again.

### Simple Pong Eval Table For Every Training Attempt

For the current Pong lane, every training attempt should run this scoreboard
table. Keep the exact episode counts adjustable by cost, but keep the rows
stable.

| Row | Split | Episodes per seating | Pair group | Purpose |
| --- | --- | ---: | --- | --- |
| 1 | `monitor` | 32 | `track_ball` vs `random_uniform` | Baseline sanity and environment drift check. |
| 2 | `monitor` | 32 | candidate vs `random_uniform` | Floor: does the candidate score against a weak opponent? |
| 3 | `monitor` | 32 | candidate vs `track_ball` | Main near-term gate: can it do better than timeout/lose against the known scripted baseline? |
| 4 | `monitor` | 32 | `track_ball` vs `track_ball` | Timeout/geometry canary; expected to truncate in current setup. |
| 5 | `selection`, when periodic checkpoints exist | 100 | `latest`, `selected_best`, and candidate vs `random_uniform` | Check whether the new candidate clears the easy floor. |
| 6 | `selection`, when periodic checkpoints exist | 100 | `latest`, `selected_best`, and candidate vs `track_ball` | Pick best checkpoint with the real current Pong gate. |
| 7 | `selection`, when policy checkpoints are nontrivial | 100 paired | candidate vs `previous` and candidate vs `selected_best` | Regression/promotion check, not primary strength. |
| 8 | `heldout`, after selecting a best checkpoint | 200+ | selected candidate, `latest`, `track_ball`, and `random_uniform` rows above | Confirm the selected result once, without fishing. |

For now, "candidate" can mean the newly trained policy checkpoint. If the run
only produced a value checkpoint that cannot act, it does not get scoreboard
rows; report it under debug metrics instead.

The current pass/fail reading should stay modest:

- Must beat `random_uniform` more often than it loses on monitor before spending
  more eval budget.
- Must improve candidate-vs-`track_ball` wins, losses, truncations, or score
  margin before claiming strategic progress.
- Matching `track_ball` by timing out is not enough.
- Beating an older learned checkpoint is interesting only if the candidate also
  holds the fixed-baseline rows.
- Do not build leagues until a simple learner improves for an inspectable
  reason.

### Scoreboard Metrics Vs Debug Metrics

Scoreboard metrics are the numbers allowed in progress claims:

- win/loss/draw or score event outcome by pair group;
- score margin or mean terminal reward by policy;
- truncation/timeout rate;
- mean and median episode length, mainly as a guardrail;
- baseline deltas versus `random_uniform`, `track_ball`, `latest`, and
  `selected_best`;
- heldout confirmation of the selected checkpoint.

Observability/debug metrics explain why a scoreboard row changed. They should
be saved and inspected, but not promoted into "the policy improved" claims by
themselves:

- action histograms and action collapse;
- action histograms by seat;
- entropy or collapse metrics;
- terminal causes and failure examples;
- last-hit counts and hit ownership;
- off-center contact counts and top/center/bottom impact distribution;
- ball/paddle positions, ball velocity, and raster frames;
- value-target MSE, policy imitation accuracy, reward-target distributions, and
  checkpoint metadata;
- contact-outcome probe results;
- per-step traces, frame joins, terminal causes, and replay examples.

Example: an angle-control probe that makes many off-center contacts is a useful
debug success. It becomes scoreboard progress only when those contacts produce
better win rate, score margin, or lower loss/truncation rate against
`track_ball` on the named splits.

## What To Measure Now

### Dummy Survival

Primary signal: the trained checkpoint survives longer and crashes less often
than an untrained/random policy on fixed and held-out seeds.

Report:

- mean terminal reward;
- survival rate or crash rate;
- median and percentile episode length;
- action histogram;
- learned state count / transition edge count for the dummy tabular model;
- eval episodes/sec and training wall time;
- config, seed range, checkpoint id, and artifact paths.

Compare against:

- random/untrained policy;
- previous checkpoint;
- a simple one-step-safe or clearance heuristic if it is cheap to add.

Do not call this "MuZero quality." This is only the actor -> replay -> update
-> checkpoint -> evaluator circuit proving that it can show a directional
learning signal.

### Tiny Line Duel

Primary signal: ego policy improves against random and simple scripted
opponents without creating seat bias, timeout farming, or action collapse.

Report:

- win/loss/draw/timeout rate;
- mean terminal reward;
- paired-seat win-rate delta;
- mean and percentile episode length;
- wall, trail, same-cell, cross-swap, and timeout cause counts;
- action histogram by player and by checkpoint;
- per-opponent table, not one blended score.

Compare against:

- `random_uniform`;
- `random_sticky`;
- `one_step_safe`;
- `lookahead_clearance` or equivalent scripted heuristic;
- fixed older checkpoints once learned policies exist.

Use paired seeds wherever possible: same seed, both seat assignments. Until
reset variety exists, still report player-specific outcomes so update-order
bugs are visible.

### CurvyTron Later

Primary signal: the same evaluator matrix works on the no-bonus training
ruleset with source-relevant telemetry. Add richer media only when debugging
needs it.

Report everything from Tiny Line Duel plus:

- ruleset id/hash and observation/reward schema ids;
- spawn/arena variant id;
- death tick, death cause, tie group, rank/outcome;
- fixed debug replay seeds for surprising losses;
- training steps, environment steps, and search simulations per decision if
  MuZero/Mctx is active.

## Baseline Ladder

Use this order:

1. Random stress: deterministic seeds, plausible episode lengths, no seat bias.
2. Scripted-vs-random: at least one simple heuristic beats random.
3. Learned-vs-random: checkpoint curve improves on fixed eval seeds and then
   holds on held-out seeds.
4. Learned-vs-scripted: policy beats or approaches the simple heuristic.
5. Old-vs-new: latest checkpoint beats earlier checkpoints without forgetting
   random/scripted opponents.
6. Small fixed pool: latest, recent, best-so-far, random, sticky-random,
   heuristic.

Only add Elo-style ratings after there are enough recurring checkpoints or
opponents that a pairwise table is hard to read. Before that, Elo is polish
over a tiny matrix and can hide the seed/opponent details we need for debugging.

## Old-Vs-New Without A League

Checkpoint comparison is a guardrail, not the main path:

- Always save periodic checkpoints.
- Keep `latest`, `best_vs_random`, `best_vs_heuristic`, and a few age-spaced
  checkpoints.
- Evaluate new checkpoints against random/scripted baselines first.
- Then evaluate new vs previous/best on fixed paired seeds.
- Mark "best" labels only when confidence intervals clear the previous
  checkpoint and no baseline metric regresses badly. Do not use this as a
  reason to build a league.

This captures the useful AlphaGo Zero idea of guarding self-play quality while
avoiding a league scheduler.

## What Not To Build Yet

Avoid for EVAL0-EVAL2:

- a persistent Elo/rating service;
- AlphaRank, NashConv, exploitability, PSRO, or coverage metrics;
- full round-robin over every checkpoint;
- league matchmaking, exploiters, main agents, or PBT;
- video generation for every eval episode;
- dense dashboards before raw JSON/CSV summaries are stable;
- joint-action search evaluation before ego-vs-random and ego-vs-scripted work.

Future notes:

- Exploitability makes sense for tiny matrix/normal-form reductions, not for
  first CurvyTron training runs.
- AlphaRank/PSRO-style analysis may help once policies cycle or non-transitive
  checkpoint pools appear.
- Coverage metrics may matter once the environment has curriculum/domain
  randomization; for now, fixed and held-out seed sets are enough.

## Staged Plan

| Stage | Harness | Purpose | Minimum Output | Pass Signal |
| --- | --- | --- | --- | --- |
| EVAL0 | `scripts/run_toy_baseline.py` and `scripts/run_dummy_survival_train.py` | Prove evaluator artifacts and summaries are stable. | `summary.json`, per-iteration metrics, final eval table. | Re-running fixed seeds produces comparable summaries; no quality claim. |
| EVAL1 | Dummy survival evaluator | Track whether the dummy learner improves over its own random/untrained floor. | Reward, survival/crash, steps, action histogram, eval speed by checkpoint. | Later checkpoints improve on fixed seeds and do not collapse on held-out seeds. |
| EVAL2 | Tiny Line Duel baseline matrix | Validate simultaneous two-player eval before CurvyTron. | Random/scripted/learned table with paired seats, collision causes, truncations. | Heuristic beats random; learned policy beats random without seat bias or timeout farming. |
| EVAL3 | Fixed checkpoint pool | Detect self-play regression and forgetting. | Latest/recent/best-vs-baseline table; old-vs-new paired seeds. | New checkpoint improves against baselines and is not worse than best-so-far within noise. |
| EVAL4 | CurvyTron no-bonus ruleset | Reuse the same evaluator shape on the real training target. | Rules/schema hashes, outcome telemetry, fixed debug replays, held-out eval table. | Baseline ladder clears random and scripted gates before serious MuZero claims. |
| EVAL5 | Rating/future analysis | Only if pairwise tables become unreadable. | Optional Elo or AlphaRank-style report. | Adds clarity over the raw matrix; otherwise skip. |

## Recommendation

Build the evaluator as a small artifact-producing job, not as a research
platform. The first durable contract should be:

```text
checkpoint + eval_config + opponent_specs + seed_set
  -> per_episode_rows + aggregate_summary + small replay/failure sample
```

That contract fits local dummy survival, Tiny Line Duel, Modal coarse jobs, and
later CurvyTron. Once this is stable, MuZero progress becomes a straightforward
question: does search/model training improve the same tables faster or farther
than the simpler policy-only baselines?
