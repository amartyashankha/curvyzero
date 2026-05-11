# LightZero Dummy Pong Visual Parity Bridge - 2026-05-09

Scope: compare current project dummy Pong LightZero surfaces against stock
LightZero Atari Pong, then pick the smallest practical bridge toward real
visual-control work and CurvyTron. No training and no pytest were run.

## Decision

Do not promote `raster_flat` as a learning lane.

Keep `tabular_ego` as the main custom dummy Pong diagnostic lane, and mark
`raster_flat` as a feature-fit smoke only until it has temporal information.
The next bridge should be a new stacked raster mode, not an incremental claim
on the existing flat raster:

```text
raster_stack4_ego
observation: 4 recent local raster frames, channel-first
model: conv only after the shape is large enough for LightZero's conv trunk
fallback falsifier: raster_stack4_ego + MLP only as a wiring smoke
```

This is intentionally a plan, not a code patch. A correct stacked visual bridge
needs env-side frame history, schema ids, reset padding, observation-space
changes, config validation, checkpoint adapter shape matching, and a dry smoke.
That is small enough for the next focused implementation pass, but too easy to
do halfway while pretending it matches stock Atari.

## Comparison

| Surface | Stock LightZero Atari Pong | `tabular_ego` dummy Pong | `raster_flat` dummy Pong |
| --- | --- | --- | --- |
| Observation | stacked grayscale frames, commonly `(4, 64, 64)` | 10 normalized state floats | one categorical `9x15` grid flattened to `135` |
| History / velocity | frame stack provides motion history | explicit `ball_vx_forward`, `ball_vy`, step fraction | none; a single frame cannot infer velocity |
| Model | conv MuZero | MLP MuZero | MLP MuZero |
| Action space | Atari Pong action map, usually 6 actions | 3 actions: `up`, `stay`, `down` | same 3 actions |
| Reward | Atari clipped game reward over long episodes | sparse terminal `+1/-1`, timeout `0` | same |
| Support scale | stock configs tolerate broad MuZero supports with larger visual runs | requested support ranges can be logged, but compiled `support_scale=300` is still a known mismatch candidate | same mismatch |
| MCTS sims | normal Atari configs use about 50 sims; tiny smokes use 2 only as plumbing checks | historical train attempts used 2/8/16; 2 has shown bad policy targets | same |
| CurvyTron relevance | visual temporal control pattern, but Atari-specific ROM/ALE semantics | good control/debug falsifier, not visual | weak visual smoke; not enough for trail/history-heavy CurvyTron |

## Why `tabular_ego` Stays First

`tabular_ego` is not visually faithful, but it is the least confusing training
debugger. It includes both paddles, ball relative position, ball velocity, ball
row, seat geometry, and horizon fraction. If this lane cannot learn with sane
MCTS targets and support scale, raster will mostly add noise.

The known highest-signal custom Pong fixes remain:

- keep train-time `num_simulations >= 8`; prefer `16` for diagnosis;
- reserve `num_simulations=2` for import/dry smokes only;
- set `td_steps` to the fixed Pong episode horizon for sparse terminal tests;
- use `discount_factor=1.0`;
- patch or at least expose the actual compiled MuZero `support_scale`, not only
  requested `reward_support_range` / `value_support_range`;
- keep independent checkpoint scorecards as the quality gate.

## Why `raster_flat` Should Be Retired For Quality Claims

`raster_flat` currently looks visual but is missing the important Atari-like
property: temporal history. One frame shows where the ball is, not where it is
going. Feeding that frame to an MLP also removes the main stock visual pattern:
convolution over stacked images.

Use `raster_flat` only for:

- env/import feature-fit checks;
- checkpoint shape plumbing;
- confirming the LightZero adapter can carry non-tabular observations.

Do not use `raster_flat` for:

- "visual Pong" claims;
- deciding MuZero cannot learn from images;
- CurvyTron observation design;
- comparison against stock Atari visual performance.

## Smallest Next Bridge

Add a separate `raster_stack4_ego` feature mode.

Practical shape:

```text
raw grid: height=9, width=15
history: 4 frames
shape: (4, 9, 15) for a first wiring smoke
schema id: dummy_pong_lightzero_raster_stack4_ego_v0
reset history: repeat reset frame 4 times
step history: append post-step frame after each env step
```

Then decide whether LightZero's stock conv model can handle that spatial size.
If not, do not contort the trainer first. Use one of these explicit bridges:

1. resize/pad to a stock-like image shape such as `(4, 64, 64)`, then use conv;
2. keep `(4, 9, 15)` and run MLP only as a shape/history smoke;
3. add compact velocity channels to raster and call it a falsifier, not Atari parity.

The preferred visual bridge is option 1, because it matches the official
Atari convention most honestly. The cheapest falsifier is option 2.

## CurvyTron Implication

CurvyTron will need temporal occupancy, headings, trails, deaths/gaps, and
multi-player context. A single flat categorical image is even less adequate
there than in dummy Pong. The useful transferable shape is:

```text
stacked local raster/history
explicit ego orientation or ego-aligned crop
separate semantic channels, not magic integer pixels
fixed action ids for left/straight/right
terminal/survival rewards logged separately from shaping
```

Dummy Pong should therefore use stacked raster as a bridge to CurvyTron
observation plumbing, not as an endpoint.

## Commands

Keep using the tabular lane for the next meaningful custom Pong diagnostic:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_train_attempt \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --opponent-policy lagged_track_ball_1 \
  --max-env-step 1024 \
  --pong-episode-max-steps 120 \
  --max-train-iter 16 \
  --num-simulations 16 \
  --batch-size 32 \
  --update-per-collect 8 \
  --n-evaluator-episode 4 \
  --collector-env-num 1 \
  --evaluator-env-num 1 \
  --n-episode 2 \
  --game-segment-length 50 \
  --td-steps 120 \
  --num-unroll-steps 5 \
  --discount-factor 1.0 \
  --reward-support-min -5 \
  --reward-support-max 6 \
  --reward-support-delta 1 \
  --value-support-min -5 \
  --value-support-max 6 \
  --value-support-delta 1 \
  --seed 10 \
  --run-id lz-dpong-tabular-h120-s10 \
  --attempt-id train-1024x16-sims16
```

Keep current `raster_flat` only as a dry feature-fit smoke:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
  --mode feature-fit \
  --env dummy_pong_lag1 \
  --feature-mode raster_flat \
  --seed 0 \
  --max-env-step 64 \
  --num-simulations 2 \
  --batch-size 8
```

Next implementation pass, if approved:

```text
1. Add `raster_stack4_ego` schema and encoder.
2. Add frame-history state to `DummyPongLightZeroEnv`.
3. Extend config import smoke validation to expect `(4, 9, 15)`.
4. Run dry/feature-fit only.
5. Only then decide resize-to-64 conv vs small-grid MLP smoke.
```

## One-Line Bridge Policy

`tabular_ego` is the learning debugger; `raster_flat` is retired to smoke-only;
`raster_stack4_ego` is the next honest visual bridge.
