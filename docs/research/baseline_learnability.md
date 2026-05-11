# Baseline Learnability Before MuZero

Status: Proposed
Date: 2026-05-08

## Short Answer

Do not start serious MuZero work until the v0 environment passes three cheap gates:

1. Random-vs-random stress shows deterministic, balanced, fast episodes with sane telemetry.
2. A simple heuristic reliably beats random on fixed and held-out seeds.
3. A small PPO or imitation-then-PPO baseline beats random on held-out seeds using the same observation and reward contract planned for search.

If these fail, MuZero will only make the failure harder to diagnose. The baseline work should stay deliberately narrow: 1v1, no bonuses, one round per episode, fixed config, terminal win/loss reward, standard wrappers, and automatic debug artifacts for failed evaluations.

## Scope

Target `curvyzero-v0` first:

- 1v1 only.
- No bonuses or powerups.
- One round equals one episode.
- Simultaneous actions for both players.
- Fixed arena, speed, trail width, turn rate, action repeat, and spawn policy.
- Terminal rewards: win `+1`, loss `-1`, same-tick tie/draw `0` unless the v0 scoring decision says otherwise.
- Timeout truncation is separate from game termination and should be logged as a failure signal, not quietly treated as a normal draw.

The baseline gate proves that the environment, observations, rewards, wrappers, seeding, and evaluation harness are sane enough for MuZero. It is not meant to produce the final strong agent.

## Experiment Ladder

Run the smallest experiments in this order.

| Step | Experiment | Purpose | Pass condition |
| --- | --- | --- | --- |
| 0 | Static env checks | Catch invalid spaces, reset/step shape drift, nondeterministic seeds, bad terminal flags. | Gymnasium/PettingZoo checks pass where applicable; repeated seed/action traces match byte-stable summaries. |
| 1 | Random-vs-random stress | Establish the environment floor before any agent logic. | No crashes or memory growth across at least 100k local episodes; seat win rates near 50/50 on symmetric settings; timeout and tie rates explained. |
| 2 | Heuristic-vs-random | Prove the game has exploitable structure under the current observation and action repeat. | Heuristic beats random by a statistically meaningful margin on fixed and held-out seeds. Initial target: at least 65 percent win rate in 1v1. |
| 3 | Heuristic imitation, if possible | Test whether the observation contains enough information for a small network to copy a competent policy. | Behavior-cloned policy beats random or at least matches most heuristic actions on held-out heuristic trajectories. |
| 4 | PPO-vs-random | Prove learning from reward works without search. | PPO beats random on held-out seeds and shows monotonic-ish improvement across checkpoints. Initial target: at least 60 percent win rate, then raise to 70 percent. |
| 5 | PPO-vs-heuristic challenge | Decide whether baseline is merely exploiting random or learning robust survival/control. | PPO approaches or beats the heuristic only after it has cleared random. Failure here should not block MuZero if earlier gates are clean, but it informs curriculum. |

Keep the first pass local CPU. Use Modal only after the harness is stable or when local throughput blocks iteration.

## Random Baseline

Random is a test instrument more than an opponent.

Implement:

- `random_uniform`: sample uniformly from the legal action set for each alive player.
- `random_sticky`: keep the previous action with probability such as `0.8`, otherwise resample. This catches whether the environment only rewards twitchy randomness.
- `mirror_random`: same random action stream with swapped seats, useful for seat-bias checks.

Log:

- Episodes/sec and physics ticks/sec.
- Episode length distribution.
- Win/loss/tie/timeout rates by seat.
- Collision cause: wall, own trail, opponent trail, head-head, timeout.
- Mean and percentile decision count per episode.
- Seed, ruleset hash, observation schema hash, action repeat, wrapper type.

Red flags:

- One seat wins materially more often under symmetric random play.
- Most episodes end by timeout.
- Same seed/action trace changes between runs.
- Random survives longer than the heuristic after enough trials.
- Vectorized and single-env stepping disagree for the same seeds/actions.

## Heuristic Baseline

The first heuristic should use privileged simulator state if that makes implementation faster. The goal is to test the environment's strategic signal, not to be a fair learned policy. Add an observation-only version later if useful for imitation.

Minimum heuristic:

- For each legal action, simulate or approximate the next `K` physics ticks.
- Reject actions that immediately collide with walls or trails.
- Prefer the action with the largest forward clearance.
- Break ties by preferring the action that increases distance from the nearest wall/trail and avoids turning into the opponent.
- Add a small turn penalty only as a tie-breaker, not as a reward-shaping claim.

Recommended variants:

- `wall_avoid`: uses only wall distance. It should beat random if movement/control are implemented sanely.
- `ray_clearance`: uses egocentric ray distances to walls/trails. This is the best first non-privileged heuristic.
- `one_step_safe`: enumerates legal actions and picks any action that avoids immediate death.
- `lookahead_clearance`: evaluates short rollouts for `K in {5, 10, 20}` physics ticks.

Use the heuristic to generate failure artifacts. Every heuristic death against random should be replayable with seed, action trace, observations, collision cause, and a small rendered trajectory.

## Imitation Baseline

Imitation is optional, but it is the cheapest way to test observation sufficiency before reward learning.

Use it if the heuristic can produce trajectories quickly:

1. Roll out `ray_clearance` or `lookahead_clearance` against random.
2. Store `(observation, action, legal_action_mask, outcome, seed, tick)` for the ego player.
3. Train a small MLP for ray features or a tiny CNN for local rasters with supervised cross entropy.
4. Evaluate the cloned policy against random and against the heuristic on held-out seeds.

Pass signals:

- Held-out action accuracy is well above the majority-action baseline.
- The cloned policy beats random without access to privileged state.
- Mistakes are interpretable in replays: missing trail information, opponent information, or horizon.

Failure signals:

- High training accuracy but poor held-out action accuracy means seed/layout overfit or too little data diversity.
- Good action accuracy but poor win rate means the heuristic labels are brittle or covariate shift is severe.
- Low action accuracy and low win rate means the observation likely omits important state or is too noisy/expensive to learn from.

Do not build a large offline-RL pipeline for this. A local PyTorch training script over `.npz` or parquet-ish chunks is enough.

## Learned PPO Baseline

PPO is the right first learned RL baseline because it is robust, on-policy, simple to instrument, and widely supported. The first version should be ego-agent training against a scripted opponent, not full self-play.

Training setup:

- Wrap v0 as a single-agent Gymnasium env: ego observes from its perspective, action controls ego, opponent is `random_uniform`, `random_sticky`, or heuristic.
- Rotate ego seat across episodes or vector env slots.
- Use terminal reward only at first. Add dense shaping only after diagnosing a concrete learning failure.
- Start with many parallel CPU envs and small networks.
- Save checkpoints often enough for evaluation curves, for example every fixed number of environment steps.

Suggested progression:

1. PPO ego versus `random_uniform`.
2. PPO ego versus `random_sticky`.
3. PPO ego versus a mixed random opponent.
4. PPO ego versus the heuristic, only after beating random.
5. Optional shared-policy self-play after the single-agent wrapper is proven.

Initial hyperparameter posture:

- Keep defaults close to the library's PPO defaults for the first smoke.
- Prefer `n_steps * n_envs` large enough for stable on-policy batches.
- Use entropy bonus early because left/right control can otherwise collapse into repetitive turning.
- Track value loss and explained variance, but judge success by held-out win rate.

## Library Choice For The Baseline

Keep the simulator independent of the library. The project-owned core should expose `reset_many`, `step_many`, and deterministic trajectory replay; wrappers adapt that core to training tools.

| Tool | Use first when | Advantages | Watch-outs |
| --- | --- | --- | --- |
| Stable-Baselines3 | Fastest path to local PPO against scripted opponents. | Mature PPO, vectorized env utilities, simple checkpoint/eval callbacks. | Its VecEnv API differs from modern Gymnasium reset/step semantics; handle `terminal_observation` and auto-reset carefully. Mostly single-agent, so use an ego-vs-scripted wrapper first. |
| CleanRL | Best when debugging every PPO detail matters. | Single-file style is easy to copy, instrument, and modify for unusual observations or logging. Includes seeding, TensorBoard, video, and experiment hooks. | It is intentionally not a modular import library; expect to adapt a script rather than plug into a framework. |
| RLlib | Use after the small local baseline if native multi-agent scaling becomes useful. | Strong multi-agent concepts, policy mapping, scalable EnvRunner/Learner architecture, PPO and imitation/offline algorithms. | Heavier dependency and operational surface. Current docs note multi-agent setups are not vectorizable in the same way as single-agent envs, so it may be overkill for the first gate. |
| Hand-rolled PyTorch PPO | Use only if SB3/CleanRL wrappers fight the environment. | Maximum control and easiest to align with custom batched env API. | Easy to introduce PPO bugs; use as a last resort for the baseline gate. |

Default recommendation: start with SB3 PPO for the shortest learnability smoke, keep a CleanRL-style script as the inspectable fallback, and postpone RLlib until PettingZoo/multi-agent self-play is required.

## Observation Choices

Start compact. Observation failure is one of the main reasons a baseline will not learn.

Recommended first observation: egocentric ray features.

- Ray distances to wall/trail/occupied cells at fixed angles around the heading.
- Optional separate channels for own trail, opponent trail, walls, and opponent head.
- Opponent relative bearing and distance in ego coordinates.
- Ego heading encoded as `sin/cos` only if absolute orientation is needed for debugging; prefer heading-relative features for learning.
- Alive/time features only if timeout or delayed scoring needs them.
- Legal action mask, even if all actions are usually legal, so future wrappers do not change shape.

Second observation: local egocentric raster.

- Fixed crop centered on ego and rotated to ego heading.
- Channels for walls, own trail, opponent trail, opponent head, and optionally recent trail age.
- Small first, for example `32x32` or `48x48`, before larger crops.
- Measure crop/rotation cost separately from physics.

Avoid first:

- Full-board raw pixels.
- Absolute coordinates as the primary signal.
- Observation schemas that require Python object dictionaries inside the hot loop.
- Hidden privileged simulator state for learned policies, except in an explicitly labeled oracle/debug baseline.

Run the same heuristic/PPO evaluation with ray features before rasters. If rays fail but the heuristic with privileged state succeeds, inspect whether rays miss side/rear trail information, opponent head motion, or enough lookahead.

## Evaluation Protocol

Evaluation must be separate from training and must produce tables, not anecdotes.

Use three seed sets:

- `fixed_debug`: small canonical set, for example 32 seeds, reused every run for quick regression and replay comparison.
- `fixed_eval`: larger stable set, for example 512 or 1,024 seeds, reused across checkpoints and algorithms.
- `heldout_eval`: generated from a disjoint seed range and never used for training, model selection, or heuristic tuning.

Suggested seed convention:

- Training seeds: `0` to `9_999_999`.
- Fixed debug seeds: `10_000_000` to `10_000_031`.
- Fixed eval seeds: `20_000_000` to `20_001_023`.
- Held-out eval seeds: `30_000_000` upward.

Evaluate each checkpoint with both seat assignments. For seed `s`, run `(ego=player_0, opponent=player_1)` and the swapped assignment. Aggregate by paired seed so seat bias is visible.

Report:

- Win rate, loss rate, tie rate, timeout rate.
- Mean terminal reward with confidence interval.
- Median and percentile episode length.
- Collision cause breakdown.
- Seat-specific win rates and swapped-seat delta.
- Evaluation wall time and episodes/sec.
- For learned agents: train steps, environment steps, checkpoint id, config hash.

Use Wilson or bootstrap confidence intervals for win rate. As a rough first gate, do not call a policy better than random unless the lower confidence bound is above 50 percent on held-out seeds.

## Fixed Seeds vs Held-Out Seeds

Fixed seeds are for comparability and debugging. Held-out seeds are for claims.

Use fixed seeds to:

- Compare checkpoints over time.
- Reproduce regressions.
- Generate stable videos.
- Detect seat bias and wrapper changes.

Use held-out seeds to:

- Decide whether the environment is learnable.
- Decide whether to start MuZero.
- Compare observation schemas after tuning on fixed seeds.

Never tune heuristic thresholds, PPO hyperparameters, action repeat, or observation schemas directly against held-out results. If held-out results drive a design change, allocate a new held-out range for the next decision.

## Metrics And Artifacts

Every baseline run should write:

- `config.json`: ruleset, observation schema, reward, action repeat, library versions, seed ranges.
- `metrics.jsonl`: per-training-iteration metrics.
- `eval.csv`: one row per evaluated episode.
- `summary.md`: win-rate table and interpretation.
- `checkpoints/`: learned policy checkpoints, if any.
- `replays/`: a small curated set of wins, losses, ties, timeouts, and surprising collisions.
- `failure_cases/`: seeds and action traces for deterministic reproduction.

Minimum summary table:

| Agent | Opponent | Obs | Seeds | Episodes | Win | Tie | Timeout | Mean reward | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_uniform | random_uniform | none | fixed_eval | 2,048 | ~50% | tracked | tracked | ~0.0 | seat-balance check |
| wall_avoid | random_uniform | privileged/rays | fixed_eval | 2,048 | target >=65% | tracked | tracked | >0.0 | heuristic gate |
| PPO | random_uniform | rays | heldout_eval | 2,048 | target >=60% | tracked | tracked | >0.0 | learned gate |

## Failure Diagnosis

When a baseline fails, diagnose in this order.

| Symptom | Likely cause | Next check |
| --- | --- | --- |
| Random-vs-random is seat-biased | Spawn asymmetry, wall inclusivity, update order, tie resolution, RNG split bug. | Run swapped-seat paired seeds; replay first divergent symmetric seeds. |
| Random episodes mostly timeout | Arena too large, speed too slow, trail gaps too generous, collision not writing trails. | Plot episode length distribution and trail occupancy over time. |
| Heuristic cannot beat random | Collision/action semantics wrong, action repeat too coarse, observation/control horizon too short, heuristic bug. | Test `one_step_safe`; inspect deaths with rendered replay. |
| Heuristic beats random but imitation fails | Observation omits state the heuristic uses, labels are inconsistent, network too small, data not diverse. | Train on privileged features as an oracle; compare action confusion by ray sector. |
| Imitation works but PPO fails | Sparse reward/horizon issue, PPO config issue, reward sign bug, normalization/autoreset bug. | Overfit PPO on tiny fixed seeds; inspect returns and terminal observations. |
| PPO learns fixed seeds but fails held-out | Overfitting to spawn layouts or fixed eval leakage. | Increase spawn diversity; remove absolute coordinates; evaluate more seeds. |
| PPO wins by timeout or degenerate circling | Reward or truncation handling is wrong for the target. | Separate termination from truncation; report timeout as failure unless explicitly accepted. |
| Training curves improve but eval does not | Evaluation wrapper mismatch, deterministic/stochastic action mismatch, checkpoint load bug. | Run policy in training env with eval seed and compare raw action traces. |
| Learned policy collapses to one action | Entropy too low, action mapping bug, observation normalization issue. | Log action histogram by tick and seed; test policy on mirrored observations. |

Do not patch these with dense shaping until the underlying cause is understood. Prefer smaller arenas, shorter horizons, clearer observations, or scripted curricula before adding reward terms that MuZero would later inherit.

## Promotion Gate

Promote baseline learnability to a design decision only when these are true:

- Random stress passes determinism, throughput, and seat-balance checks.
- Heuristic beats random on fixed and held-out seeds with replayable failures.
- PPO or imitation-then-PPO beats random on held-out seeds.
- Evaluation artifacts include config hashes, seed ranges, raw per-episode rows, and representative replays.
- Known failure modes are either fixed or documented as non-blocking for MuZero v0.

MuZero can start after this gate with much lower ambiguity: if MuZero fails, the team can focus on search, replay, targets, model capacity, and batching instead of wondering whether the game is learnable at all.

## Sources

- `curvytron_muzero_modal_handoff.md`
- `docs/investigation_plan.md`
- `docs/design/deterministic_environment.md`
- `docs/design/rulesets.md`
- `docs/research/training_architecture_notes.md`
- `docs/research/performance_vectorization.md`
- Stable-Baselines3 vectorized environment docs: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- CleanRL overview: https://docs.cleanrl.dev/
- RLlib algorithms and multi-agent docs: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html and https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
- PettingZoo Parallel API docs: https://pettingzoo.farama.org/api/parallel/
