# Pong Self-Play Training Plan - 2026-05-09

Status: historical custom dummy Pong scaffold under critique. This is not the
current source of truth for quality or next strategy, and it is not the
official Atari LightZero lane or the CurvyTron repo-native lane.

## North Star

Build the smallest real Pong training loop:

```text
policy games -> replay rows -> true score plus survival telemetry -> learner update
-> checkpoint -> scoreboard eval -> repeat
```

This is an implemented scaffold, not a locked decision. The critique wave in
`docs/working/pong_training_critique_wave_2026-05-09.md` may replace it with a
simpler known baseline or curriculum. The old `track_ball` imitation,
lookahead, and self-play runs are useful plumbing and negative evidence. They
do not prove policy quality.

## Correct Framing

- This doc is about custom dummy Pong only. It does not describe official
  Atari Pong control results, and it does not describe CurvyTron.
- Pong is a two-player toy for the training stack.
- The learner should improve through self-play or policy rollouts, not by
  copying `track_ball` forever.
- `track_ball` is an eval baseline and a possible curriculum opponent. It is
  not the teacher we are trying to imitate as the final behavior.
- The scoreboard must stay honest: learned checkpoints versus `random_uniform`,
  `track_ball`, older checkpoints, and best-so-far checkpoints on fixed seed
  splits.
- Imitation, lookahead, angle probes, and contact probes are diagnostics. Keep
  them when they explain a failure; do not let them define progress.
- Use survival-first reporting: true score, survival mean/median/p90/std,
  shaped loss-delay telemetry, reward timing, action distribution, opponent
  split, reset split, checkpoint ref, and manifest ref.

## Historical Shaped Training Return V0

Stale/current-status note: this shaped target was useful for the old toy
learner scaffold, but it is not the current default objective. Keep it as a
historical ablation design unless a future run is explicitly labeled
`shaped-objective`. Current default rule is sparse true score for environment
reward and promotion, with survival/loss-delay as telemetry.

Keep the environment and eval score simple:

```text
ego scores:       +1.0
opponent scores:  -1.0
no score event:    0.0
```

The old first self-play trainer used a separate episode-level training target
that gave partial credit for lasting longer when the agent lost:

```text
survival_fraction = episode_steps / max_steps

if ego wins:
    shaped_return = +1.0
elif ego loses:
    shaped_return = -1.0 + 0.5 * survival_fraction
else:
    shaped_return = 0.0
```

This meant a fast loss was close to `-1.0`, a long loss could approach `-0.5`,
and a win was still clearly best. Treat this as historical scaffold logic, not
the active promotion rule.

Log both values:

- `score_return`: the raw score/win signal.
- `shaped_return`: the training target used by the old toy learner.

Never report `shaped_return` as the scoreboard win metric.

For any revived shaped-objective ablation, also log the distribution, not only
the mean:

- mean, median, p90, and max survival steps;
- survival standard deviation;
- shaped-return standard deviation;
- rare wins or rare long rallies.

Variance can be useful for exploration. Early on, if two checkpoints have
similar mean score and survival, keep the one with a wider useful tail as a
secondary candidate. This is not the same as changing `PongEnv.step()` reward,
and it must be dropped if it rewards stalling, random actions, or worse
baseline performance.

## Loop V0

Historical/current-status note: this loop shape ran and exposed plumbing plus
failure modes. Do not restart it just because the scaffold exists. A revived
run needs a concrete learner/objective bug fix or a clearly separate baseline
choice.

1. Start from a policy checkpoint. The first one can be random or the existing
   raster imitation checkpoint.
2. Generate self-play games on raster observations. Use the same policy for
   both seats at first, with small action noise so games do not become identical.
3. Write one replay row per ego player per step:
   `raster_grid`, `ego_agent`, `action_taken`, `joint_action`, `score_return`,
   final `shaped_return`, terminal metadata, and policy/checkpoint ids.
4. Train a tiny visual policy/value learner from those rows. A simple first
   update is enough: increase probability of actions from above-average
   shaped-return episodes and decrease it for below-average episodes; fit value
   to `shaped_return`.
5. Save periodic checkpoints.
6. Run the checkpoint scoreboard against `random_uniform`, `track_ball`, older
   checkpoints, and best-so-far.
7. Treat monitor rows as a guardrail only. A child checkpoint becomes a
   candidate only if it beats its parent and preserves or improves the fixed
   baseline rows against `random_uniform` and `track_ball`.
8. Run heldout only after selection. Heldout is required before any quality
   claim.

## What We Expect To See

Historical note: these were expectations for the old shaped-return scaffold.
Current promotion still requires true score/baseline improvement; shaped or
survival gains are explanatory unless the run was labeled as an ablation.

First signal:

- mean `shaped_return` trends upward in self-play or rollout eval;
- losses last longer before the policy wins more often;
- checkpoint-vs-checkpoint rows show newer checkpoints beating older weak ones.

Better signal:

- learned checkpoints beat `random_uniform` reliably;
- learned checkpoints lose less often or force more pressure against
  `track_ball`;
- eventually, learned checkpoints beat a target ladder that has first been
  proven scoreable. Default `track_ball` is not that target from normal resets.

Failure signal:

- shaped return rises only because games timeout;
- the policy stops trying to score;
- checkpoint-vs-`track_ball` stays flat while shaped return rises.

If that happens, reduce the longevity weight, add timeout penalties, or use the
raw score return for promotion while keeping shaped return only as an auxiliary
training target.

## Current Correction

Generation 2 lost to the parent and won 0 games against `track_ball`. The later
beatability probe changes the interpretation: old self-play was not merely
failing; it was being asked to beat an impossible default target. Do not keep
adding generations, leagues, or Modal scale because the loop exists.

The later 512-game Modal fresh-replay audit also missed the useful survival
bar: its best checkpoint was worse than repair ckpt25 by the
survival/loss-delay proxy.
Repair ckpt25 remains the best old baseline. The old self-play trainer is
stopped as the main lane unless a concrete learner/objective bug is found.

The next decision is narrow:

- make a small, concrete bug fix to the old trainer, or
- start a separate known simple baseline or curriculum such as PPO, actor-critic,
  CEM, or another small policy-gradient path on a weaker/changed target ladder.

Either path must run fixed-baseline evals first. Do not build leagues until a
simple learner improves for an inspectable reason.

The small `policy_grad = probs.copy()` aliasing repair is done. Treat it as
trainer hygiene only; it does not change the gen2 conclusion or justify blind
scaling.

## Current Next Actions

1. Do not add more old self-play generations by default.
2. If this lane is revived, first state the concrete bug fix or baseline being
   tested and label any shaped objective before launch.
3. Run fixed-baseline eval before and after the change, with survival-first
   reporting beside true score.
4. Keep custom dummy Pong claims separate from official Atari LightZero and
   CurvyTron.

## Historical Implementation Tasks

This list records what the self-play scaffold already did. It is not the active
next-action list.

1. Add a Pong self-play replay builder. Done once for `random_uniform` and
   `learned:<checkpoint>` policies.
2. Add the shaped-return fields above to replay summaries. Done.
3. Add a small self-play policy/value trainer. Done once. Do not call it
   MuZero yet.
4. Reuse the existing checkpoint scoreboard for eval. Done once.
5. Add a local runbook command once the loop works. Done.
6. Run a second-generation manual loop: collect from the latest self-play
   checkpoint with exploration, train from that checkpoint, and score old versus
   new. Done once; do not promote gen2.
7. Decide whether to repair this trainer or replace it with a simpler known
   baseline/curriculum. Fixed-baseline evals come first either way.
8. If this trainer is repaired further, keep action diversity visible,
   normalize advantages per ego/seat or per game, and gate child checkpoints
   against the parent plus fixed baselines.
9. Do not add more generations by default. The active decision is repair this
   trainer only for a concrete bug, or switch to a separate simpler
   baseline/curriculum.

## First Smoke Result

`docs/experiments/2026-05-09-dummy-pong-selfplay-smoke.md` records the first
end-to-end smoke.

What worked:

- self-play replay wrote 498 rows from 16 random-vs-random games;
- `score_return` and `shaped_return` were present on replay rows;
- the trainer wrote epoch 25 and epoch 50 checkpoints;
- the existing checkpoint scoreboard loaded those checkpoints.

What did not work yet:

- the first value-learning default was too high and diverged; default is now
  `0.001`;
- the first self-play checkpoints did not score against `track_ball`;
- `selfplay50` only tied random on the tiny monitor row, though it beat
  `selfplay25` in learned-vs-learned.

Interpretation: the loop shape is real, but quality is weak. This result has
now been superseded by the generation-2 smoke and critique wave below.

## Generation 2 Result

`docs/experiments/2026-05-09-dummy-pong-selfplay-gen2-smoke.md` records the
manual generation-2 loop.

What worked:

- replay from `selfplay50` plus epsilon exploration wrote 1,392 rows;
- training initialized from `selfplay50` and wrote epoch 25/50/75 checkpoints;
- scoreboard compared parent, children, random, and `track_ball`.

What did not work:

- no generation-2 checkpoint beat `track_ball`, which is now known to be an
  impossible default target from normal resets;
- generation-2 lost head-to-head against the parent checkpoint;
- the final generation-2 checkpoint narrowed its action use and stopped
  predicting `down`.

Interpretation: do not promote generation 2. Do not assume more generations are
the right next step. The active decision is repair this crude trainer or switch
to a simpler known baseline/curriculum.

Every future run should emit more than a scoreboard:

- iteration metrics;
- action histograms by seat;
- entropy or collapse metrics;
- terminal causes and truncation reasons;
- a few failure examples;
- heldout results after selection, not before.

## Modal Role

Modal should run whole jobs and store files. Local is only for tiny debug and
artifact-shape checks. Once the learner path is chosen and worth running, the
serious train/eval attempt belongs on Modal.

Do not spend time on multi-node or large GPUs until the learner has a real
training signal. When it does, scale by adding more games, more parallel actors,
and more checkpoint eval games before reaching for stronger hardware.

## Historical Work To Keep In Its Place

- `track_ball` imitation proved raster observations, checkpoint save/load, and
  learned-policy eval.
- The scoreboard proved baseline and checkpoint comparisons.
- Angle/contact probes proved the paddle angle mechanic and showed why
  off-center contact rate alone is not progress.
- One-step, loss-delay, and depth-2 lookahead relabeling did not beat
  `track_ball`; keep those as diagnostics, not the main path.
