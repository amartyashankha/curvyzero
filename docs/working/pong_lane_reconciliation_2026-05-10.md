# Pong Lane Reconciliation - 2026-05-10

Scope: docs-only reconciliation. No training, pytest, or code changes were run.

## Decision

Atari-style means a LightZero-compatible visual environment shape: stacked image
frames, discrete actions, conv model path, reward/done/info, reset/seed, and
eval/checkpoint discipline. It does not mean "must use ALE."

Official Atari Pong is the primary LightZero reproduction/control lane. It uses
ALE because it is real Atari Pong. ALE is the Atari emulator for stock Atari ROM
control runs; it is not a requirement for CurvyTron.

Use custom dummy Pong only as a bridge/debug lane. It is useful for plumbing,
telemetry, and small controlled ablations, but it is not a competing quality
lane. Do not compare dummy Pong scores to Atari Pong scores. Do not use dummy
Pong collapse or success as the main verdict on LightZero's Atari Pong setup.

## Why Custom Dummy Pong Exists

Custom dummy Pong exists because it is small, owned by this repo, and easy to
instrument.

It lets us test things that are harder to see inside stock Atari:

- whether a LightZero custom env can import, reset, step, and train;
- whether a single-ego wrapper can hide a scripted opponent inside
  `env.step`;
- whether checkpoint, scorecard, target-replay, truncation, and seed metadata
  are being written clearly;
- whether shaped-objective experiments change telemetry without pretending to
  be stock Atari reward.

That makes it a bridge from LightZero to project-owned games such as CurvyTron.
It also makes it a debug harness for target and eval reporting.

## Why It Differs From Atari-Style LightZero Pong

The two lanes do not play the same problem.

Official Atari Pong uses LightZero's Atari path: ALE Pong, visual observations,
conv MuZero, frame stacking, Atari action count, Atari wrapper semantics, and
stock-ish LightZero evaluator behavior.

Custom dummy Pong uses a project toy env. LightZero controls one ego paddle.
The wrapper supplies the other paddle's scripted action. The default
observation is a 10-float `tabular_ego` row, with an optional flat 15-by-9
raster. The config is patched from CartPole into an MLP MuZero setup with a
3-action space.

Those differences are the point for debugging, but they make quality claims
non-transferable.

## What Custom Dummy Pong Has Taught Us

It proved that the custom env path can be wired into LightZero without forking
LightZero.

It gave us useful observability patterns:

- env-side episode rows;
- explicit `terminated` versus `truncated` metadata;
- action histograms and collapse detection;
- target replay sidecars;
- same-run checkpoint scorecards;
- support-scale and config-surface checks.

It also showed that shaped survival/loss-delay rewards can improve some
survival telemetry while still failing to produce convincing Pong skill.

## Why It Is Not Currently Successful

The current dummy results are not quality evidence.

The shaped `epochs24` run collapsed to always choosing `up`. The earlier shaped
smoke collapsed to always choosing `down`. Raw scores did not show convincing
skill against the fixed opponents. Survival sometimes moved, but the policy did
not become robust.

The likely issue is not one simple bug. The lane is deliberately tiny and
non-Atari:

- tiny MLP features instead of stacked Atari frames;
- scripted opponent hidden inside the wrapper;
- small MCTS defaults in many debug runs;
- short horizons and truncation behavior;
- shaped reward ablations that change the objective;
- scorecard machinery richer than the training discipline around it.

So the honest read is: useful plumbing, weak policy evidence.

## What Would Make It More Atari-Disciplined

Keep dummy Pong only if it gets stricter.

Minimum improvements before treating another dummy run as more than a debug
run:

- sparse score reward by default, with shaped rewards labeled as ablations;
- fixed train/eval split ids and same-run `iteration_0` baselines;
- strict checkpoint load with no fallback;
- survival-first reporting before return or win/loss;
- explicit action-collapse checks;
- compiled config proof for model surface, support scale, MCTS sims, unroll,
  discount, and segment length;
- no hidden changes to reward, horizon, opponent, reset profile, or feature
  schema;
- a visual/raster history path if the goal is to move closer to Atari-style
  discipline, instead of only 10-float tabular state;
- larger search and training budgets only after the small run passes the
  reporting and config checks.

Even with those fixes, dummy Pong remains a bridge/debug lane unless someone
deliberately brings it closer to visual frame-stack discipline and labels that
new purpose. The primary LightZero reproduction/control question stays with
official Atari Pong.

## Claim / Non-Claim Rule

Every run doc must say the claim and non-claim in plain words before the result
table.

Examples:

- Official Atari Pong claim: "This checkpoint improved against its same-run
  `iteration_0` baseline under strict eval."
- Official Atari Pong non-claim: "This does not prove solved Pong, exact
  upstream reproduction, or CurvyTron readiness."
- Custom dummy Pong claim: "This run proves wrapper, target, scorecard, or
  shaping telemetry behavior in the toy env."
- Custom dummy Pong non-claim: "This is not official Pong quality evidence and
  its scores must not be compared to ALE Pong scores."

## Source Pointers

- Source map:
  `docs/working/pong_official_vs_custom_source_map_2026-05-09.md`
- Active board:
  `docs/working/training_coach_active_board_2026-05-10.md`
- State index:
  `docs/working/training_state_index_2026-05-09.md`
- Official Atari wrapper:
  `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- Official dry config wrapper:
  `src/curvyzero/infra/modal/lightzero_pong_dry_config_smoke.py`
- Custom dummy config builder:
  `src/curvyzero/infra/modal/lightzero_dummy_pong_config_import_smoke.py`
- Custom dummy LightZero env:
  `src/curvyzero/training/lightzero_dummy_pong_env.py`
- Custom dummy env:
  `src/curvyzero/training/dummy_pong.py`
- Custom dummy features:
  `src/curvyzero/training/lightzero_dummy_pong_features.py`
