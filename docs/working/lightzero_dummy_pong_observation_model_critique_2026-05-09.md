# LightZero Dummy Pong Observation/Model Critique - 2026-05-09

Role: model-capacity / observation critic. No pytest.

## Short Verdict

The current `tabular_ego` setup is probably **not missing the basic Markov
state needed for dummy Pong**. It includes both paddles, ball-relative position,
ball velocity, ball row, and step fraction. A tiny MLP should be able to learn
a sane paddle controller from it if the training target is learnable.

The current `raster_flat` setup is much weaker. It is a single unstacked
`9x15` grid flattened into an MLP input. It can see object positions, but not
velocity, timestep, ego identity as a channel, reset/curriculum metadata, or
opponent behavior. It is a useful compatibility smoke, not a fair visual
control setup.

So: observation/model capacity can explain some weakness, especially for
`raster_flat`, but it is not the leading explanation for eval action collapse
in `tabular_ego`. The leading explanation remains weak/tied MCTS roots from a
sparse, undertrained target, plus deterministic eval tie-breaking. There is
also a config footgun: requested `reward_support_range` / `value_support_range`
is recorded in our patch surface, but compiled LightZero MuZero still has
`model.support_scale=300`.

## What Each Encoder Contains

`tabular_ego`, from
`src/curvyzero/training/lightzero_dummy_pong_features.py`:

| Needed signal | Present? | Notes |
| --- | --- | --- |
| Ego paddle position | Yes | `ego_paddle_y`, scaled by legal top range. |
| Opponent paddle position | Yes | `opponent_paddle_y`. |
| Paddle side / seat | Mostly | `ego_paddle_x`, `opponent_paddle_x`, and ego-relative ball x leak side; no explicit `ego_agent` one-hot. |
| Ball x/y position | Yes | `ball_dx_forward`, `ball_dy_from_ego_center`, and absolute `ball_y`. |
| Ball velocity | Yes | `ball_vx_forward`, `ball_vy`. |
| Step / horizon | Yes | `step / max_steps`; denominator changes if `PongConfig.max_steps` changes. |
| Reset profile metadata | No | Only in terminal telemetry, not observation. Safe only if a run uses one fixed reset profile. |
| Opponent policy metadata | No | Only in env config/telemetry. Safe only if a run uses one fixed opponent policy. |
| Contact angle / hit offset | Derivable locally | At contact, impact offset is `ball_y - paddle_center_y`; after contact, outgoing `ball_vy` carries the angle. No explicit `last_hit_impact`. |

`raster_flat`, from `PongEnv.raster_observation()`:

| Needed signal | Present? | Notes |
| --- | --- | --- |
| Paddle and ball positions | Yes | Values `1`, `2`, `3/4` in a single grid. |
| Ball velocity | No | A single frame cannot distinguish incoming/outgoing or vertical direction. |
| Ego/seat | Weak | The fixed values identify physical players, but the model gets no explicit "I am player_0/player_1" channel. |
| Step / horizon | No | No time feature. |
| Reset/opponent metadata | No | Not in grid. |
| Contact angle | Partly | Current offset is visible only when ball/paddle geometry is near contact; outgoing angle is invisible without next/previous frame. |

The raster result matches this: the raster bridge trained and strict-loaded, but
independent MCTS still chose zero `down`: aggregate `[424,146,0]`.

## Model And Official Pattern Comparison

Our custom dummy Pong config starts from official CartPole MuZero and patches:

- `model_type=mlp`
- observation shape `10` for `tabular_ego` or `135` for `raster_flat`
- action size `3`
- often tiny CPU settings: `num_simulations=2/8/16`, small batches, one
  collector/evaluator, and historically `update_per_collect=1`

Official LightZero CartPole is MLP and state-based, so `tabular_ego` is a
reasonable bootstrap pattern. But official Atari Pong is not a flattened
single-frame MLP: it is a convolutional visual MuZero over stacked grayscale
frames, with much larger budgets. If we want a real visual dummy Pong lane,
`raster_flat` should be at least frame-stacked, preferably channelized, and
eventually conv-based.

Capacity itself is not the obvious blocker for `tabular_ego`: LightZero's MLP
defaults use latent state dim `128` and small policy/value heads. That is plenty
for a 10-float toy state. It may be awkward for a 135-cell raster because the
MLP must learn spatial parsing and velocity inference without temporal input.

## Top Issues

1. **`raster_flat` is missing velocity.**
   Single-frame raster cannot know whether the ball is incoming, outgoing, or
   what vertical bounce is unfolding. This is the most direct observation
   deficiency. Do not use raster results to indict MuZero until a 2-frame or
   velocity-augmented raster falsifier has run.

2. **`tabular_ego` omits metadata that becomes necessary when distributions are mixed.**
   Fixed `opponent_policy` and fixed `pong_reset_profile` are fine without
   metadata. But if training mixes `default` and `contact_pressure`, or mixes
   random/lagged/track opponents, the same physical state can imply different
   opponent dynamics. Add metadata before claiming a mixed curriculum failed.

3. **Contact angle is available to `tabular_ego`, not explicitly labeled.**
   Contact probes show top/center/bottom contacts change outgoing `ball_vy`.
   `tabular_ego` can compute hit offset from ball row and paddle center near
   contact. It does not see `last_hit_impact`, but that is not needed for a
   Markov state if current ball velocity is present.

4. **Support-range knobs look misleading.**
   The patch surface can show `reward_support_range=[-5,6,1]` and
   `value_support_range=[-5,6,1]`, but a local compile check still showed
   `model.support_scale=300`. LightZero MuZero uses `support_scale` to build
   `DiscreteSupport(-300,300)`. This can dilute tiny `-1/0/+1` targets and
   should be treated as a real config bug candidate.

5. **Eval collapse is better explained by weak roots than missing tabular information.**
   Debug rows showed tiny logits and tied/near-tied visit counts; deterministic
   eval picks argmax visit counts and tie-breaks toward lower indices. Collapse
   direction changes by run (`up`, `stay`, `down`), which argues against a
   single missing feature or action-map bug.

6. **Horizon is both task setting and feature scale.**
   `step/max_steps` is useful, but changing `max_env_step` as a training budget
   previously also changed `PongConfig.max_steps`. Explicit
   `pong_episode_max_steps` should remain fixed for comparisons.

## Smallest Falsifying Experiments

1. **Tabular supervised oracle check.**
   Train a tiny supervised policy on generated labels from `track_ball`,
   `lagged_track_ball_1`, and angle-control/contact-pressure action sweeps
   using only the 10 `tabular_ego` features. If it cannot learn sensible
   `up/stay/down` decisions, the tabular observation is suspect. If it can,
   MuZero training/search is the suspect.

2. **Raster velocity falsifier.**
   Run the same raster smoke with either two stacked frames or raster plus
   `(ball_vx_forward, ball_vy, step_fraction)` appended. Keep checkpoint eval
   matched to that schema. If zero-`down` disappears quickly while single-frame
   raster stays collapsed, raster information was the blocker.

3. **Metadata ablation for curriculum.**
   On `contact_pressure` plus `lagged_track_ball_1`, compare:
   `tabular_ego` vs `tabular_ego + reset_profile + pressure_agent +
   contact_distance + opponent_policy_id`. If only the metadata version learns,
   the mixed distribution is partially hidden-state.

4. **Support-scale patch control.**
   Patch actual `policy.model.support_scale` to a small value matching the
   outcome scale, or set the corresponding model support size correctly if
   LightZero requires that path. Run a tiny paired control against the same
   config that only records `reward_support_range`. If roots become less tied,
   the current support-range surface was decorative.

5. **Capacity-only control.**
   Double MLP latent/head widths for `tabular_ego` without changing data,
   support scale, or eval sims. If action collapse remains, stop blaming MLP
   size. If it fixes collapse alone, then capacity was real.

6. **Observation-action debug table.**
   For one collapsed checkpoint, log 64 heldout observations with
   `ball_dy_from_ego_center`, `ball_vx_forward`, `ball_vy`, seat, logits,
   root visits, and selected action. If states requiring `down` have sane
   features but tied/bad visits, the observation is exonerated.

## Recommendation

Keep `tabular_ego` as the main diagnostic lane. It is compact, mostly Markov,
and intentionally removes visual parsing as a confounder.

Treat `raster_flat` as a bridge smoke only until it has temporal information.

Before another scale run, fix or explicitly falsify the support-scale issue and
run the tabular supervised oracle check. Those two are cheaper and more
diagnostic than another large MuZero run with the same observations.
