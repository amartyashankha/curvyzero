# Pong Lessons For CurvyTron Native Path - 2026-05-10

Scope: docs-only extraction from older Pong notes. No code or pytest. Purpose:
keep only lessons relevant to the CurvyTron native LightZero path, not
a Pong result recap.

## Short Verdict

Pong's strongest lesson is to prefer the native LightZero spine wherever the
game can be expressed as:

```text
visual frame stack -> discrete action -> reward -> done
```

CurvyTron can be expressed that way for single-ego fixed/frozen/checkpoint-
lagged opponent training. The native path should own collector, GameBuffer,
learner, checkpoints, and eval. Custom code should shrink to wrappers,
opponent providers, metadata, and later one joint-action collector.

## Lessons

1. Working Pong evidence came from native or stock-ish LightZero surfaces.

Official visual Pong used installed `LightZero==0.2.0`, ALE Pong, conv MuZero,
stacked frames, strict checkpoint loading, normal `iteration_*` checkpoints,
and same-run `iteration_0` comparisons. Wave11 normal Pong showed late
survival/return curves under stock-only strict eval.

CurvyTron implication: scale the registered native `train_muzero` single-ego
visual env first because it exercises LightZero's real trainer machinery.

2. Custom dummy Pong was useful plumbing, not quality evidence.

Dummy Pong proved custom env registration, one-ego wrappers, opponent in-env
control, target sidecars, action histograms, truncation metadata, and
scorecards. It also repeatedly produced weak or collapsed policies. Its MLP
features, tiny sims, custom telemetry, and scripted opponent made it
non-comparable to official Atari Pong.

CurvyTron implication: keep custom CurvyTron machinery as a diagnostic harness
only when it answers a native-path gap. Do not let a private trainer become the
main learning lane just because it writes prettier artifacts.

3. Do not compare across lanes or runs.

Older confusion came from comparing one run's final checkpoint to another
run's `iteration_0`, or mixing official Atari Pong, custom dummy Pong, and
CurvyTron claims. The reliable Pong read was same-run, same-eval-contract,
`iteration_0` versus later normal checkpoint.

CurvyTron implication: compare same-run checkpoints under the same strict eval
seed panel, cap, sims, opponent, reward schema, and observation schema.

4. Survival is required telemetry, not automatically the training objective.

Pong docs converged on sparse/true reward as the default claim metric, with
survival, loss-delay, reward timing, action entropy, terminal cause, and
timeout rate as readout. Survival-shaped Pong was a side lane, not stock proof.

CurvyTron implication: label reward schemas loudly. Survival-time wrappers may
be valid debug objectives, but promotion needs anti-stall checks, action
histograms, terminal reasons, and held-out seed panels.

5. Custom target/replay code is where drift hides.

Dummy Pong exposed hazards around root-visit targets, executed actions, support
scale, replay sidecars, horizon mismatch, eval config mismatch, and checkpoint
load parity. Some trainer telemetry looked alive while scorecards stayed weak.

CurvyTron implication: native LightZero target construction should remain the
default. If CurvyTron needs custom replay metadata, add the smallest bridge and
prove equivalence against native behavior before scaling.

6. The CurvyTron blocker is joint action, not "not Pong."

Pong and single-ego CurvyTron fit the LightZero one-action-per-env-row shape.
True two-seat self-play does not: one current policy must choose all active
seat actions before one simultaneous source step.

CurvyTron implication: fixed, scripted, frozen, or checkpoint-lagged opponents
belong on native `train_muzero`. True current-policy two-seat play needs one
carefully bounded custom collector, not a whole private trainer.

7. Native CurvyTron is already mechanically viable but not learning-positive.

Existing native CurvyTron visual-survival runs complete, publish checkpoints,
strict-load, and eval over random seed panels. Current curves are flat or weak.
That says the current native fixed-opponent setup has not yet produced reliable
learning, not that LightZero cannot train CurvyTron.

CurvyTron implication: keep the native lane, improve the contract, and judge by
curves. Do not switch to custom infrastructure unless the missing abstraction
is precisely identified.

## Practical Rule

Default to native LightZero for anything single-ego:

```text
registered env + native collector + GameBuffer + learner + checkpoint cadence
+ strict eval + same-run curves
```

Customize only the irreducible CurvyTron pieces:

```text
source-faithful wrapper, observation/reward schema, opponent provider,
metadata, seed panels, seat swaps, and later one joint-action collector
```

Claim template:

```text
This proves <native single-ego / fixed-opponent / frozen-opponent / custom
two-seat collector> behavior under <reward schema> and <eval contract>.
It does not prove <current-policy self-play / source-fidelity visual skill /
win-loss mastery> unless those are explicitly in the contract.
```
