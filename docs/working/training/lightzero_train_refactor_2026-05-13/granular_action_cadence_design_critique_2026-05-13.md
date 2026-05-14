# Granular Action Cadence Design Critique

Date: 2026-05-13

## Bottom Line

The source-state wrapper can express the desired trusted-lane behavior: current
local defaults are `decision_source_frames=1`, `decision_ms=SOURCE_PHYSICS_STEP_MS`,
and policy action repeat defaults to exactly one wrapper step. In that narrow
configuration, one LightZero `env.step(action)` maps to one granular source
physics frame.

The fragile part is not the happy path. The fragile part is that the codebase
still has two ways to stretch a policy action across physics time:

- `decision_ms` / `decision_source_frames` inside the source-state wrapper and
  public vector env;
- `policy_action_repeat_min/max/extra_probability` inside the LightZero wrapper.

If the redesign only changes the default `decision_ms` to one source frame, it
will improve the trusted default but leave enough ambiguity for stale configs,
background eval, tests, or future launchers to silently reintroduce multi-frame
actions.

## Facts From The Current Code

- The source-state wrapper default is now one source frame:
  `DEFAULT_DECISION_SOURCE_FRAMES = 1` and `DEFAULT_DECISION_MS =
  SOURCE_PHYSICS_STEP_MS * DEFAULT_DECISION_SOURCE_FRAMES` in
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`.
- The public `VectorMultiplayerEnv` still has a separate direct-use default:
  when `decision_source_frames` is absent, it falls back to `decision_ms=300.0`
  and `source_frame_decision=False`.
- The wrapper derives frames from `decision_ms` if `decision_source_frames` is
  not supplied. A stale `decision_ms` therefore still means "hold the same
  action for N source frames".
- `_build_visual_survival_configs` passes `decision_ms` but does not pass
  `decision_source_frames`, so the trainer surface does not make the intended
  frame count first-class.
- The wrapper sets `_max_source_ticks = _max_ticks * _decision_source_frames`
  and passes that as the underlying vector env `max_ticks`. This makes
  `source_max_steps` behave like "policy decisions" when
  `decision_source_frames > 1`, despite the name.
- The wrapper samples one opponent action before the repeat loop and reuses the
  same joint action for all explicit repeats.

## Main Risks

### 1. `source_max_steps` will change real-time episode length

If the old default bundled 12 source frames and `source_max_steps=256`, the
underlying source runtime could run about `256 * 12` physics frames before
timeout. With one-frame decisions, the same `source_max_steps=256` means 256
physics frames.

That may be exactly the new desired contract, but it is a major behavioral
change: survival horizon, collision opportunity, natural bonus exposure,
discount horizon, value support scale, and `td_steps` all move at once. The
current name `source_max_steps` is only coherent if it means source physics
frames. The implementation only preserves that meaning when
`decision_source_frames == 1`.

Recommendation: pin the trusted-lane contract as:

```text
source_max_steps == max source physics frames in the episode
max_ticks == underlying source physics frame limit
policy transitions per episode <= source_max_steps
```

Then reject or loudly mark any `decision_source_frames > 1` path as a legacy
decision-bundle mode, because `_max_ticks * _decision_source_frames` otherwise
turns `source_max_steps` back into policy-step count.

### 2. `decision_ms` is a dangerous cadence knob

`decision_ms` sounds like wall-clock timing, but in the wrapper it is also an
action-hold knob. `_source_frame_decision_config` converts it into an integer
source-frame count. That means an old CLI argument, manifest value, eval payload,
or test fixture with `decision_ms=300.0` silently changes one policy action into
many physics frames.

Recommendation: make `decision_source_frames` the explicit config field for any
non-default cadence and treat `decision_ms` as derived metadata in the trusted
lane. If `decision_ms` is accepted, the surface should report both:

```text
decision_source_frames
source_ticks_per_lightzero_step
```

and reject a trusted train config where either is not `1`.

### 3. Rewards are only unambiguous in the one-frame default

The reward schemas claim alignment to "reward_t_plus_1_after_one_source_tick".
That is true when `decision_source_frames=1` and action repeat is disabled.

With `decision_source_frames > 1`, dense survival pays once per vector env step,
not obviously once per source physics frame. Bonus pickup reward uses
`bonus_catch_count_step`, which may aggregate catch events inside the vector
step, but the reward schema still says one source tick. With explicit
`policy_action_repeat`, the wrapper sums rewards across repeated vector steps,
but each repeated vector step may itself contain multiple source frames.

Recommendation: keep reward schema claims conditional on
`decision_source_frames=1`. If bundled decisions remain supported, their info
payload should use a different reward alignment label such as
`reward_after_bundled_source_frames`.

### 4. Natural bonus timing will shift with the default

Natural bonus timers advance by the step's `timer_advance_ms`. Under granular
cadence this is one source frame per LightZero step. Under old bundled cadence
it was many source frames per LightZero step.

This affects both frequency of bonus spawns and learning target scale. With a
short default `source_max_steps`, the agent may see far fewer bonus events than
before. Tests that force `decision_ms=300.0` for profile/no-death bonus coverage
are not representative of the trusted granular lane.

Recommendation: after the default change, add one trusted-config test that
asserts the expected bonus timer progress in source-frame units, and keep the
long-`decision_ms` profile tests labeled as legacy/profiling fixtures.

### 5. Wall collision and heading semantics depend on using source-frame mode

The good path is the source-frame loop in `VectorMultiplayerEnv`: it calls
`step_many` once per source frame and applies source moves on the first substep.
With `decision_source_frames=1`, heading updates, wall collisions, trail
printing, and death checks are granular.

The bad path is direct `VectorMultiplayerEnv` construction without
`decision_source_frames`, where a single `step_many` receives a large
`decision_ms`. That is not equivalent to granular stepping and can alter
collision and movement behavior.

Recommendation: do not infer trusted-lane behavior from direct vector-env
defaults. The source-state wrapper should always pass `decision_source_frames`,
and tests should assert the underlying batch reports
`source_physics_substeps_executed == [1]`.

### 6. Opponent action cadence is still coupled to ego cadence

The fixed/proactive/frozen opponent action is chosen once before the wrapper's
repeat loop. If `policy_action_repeat > 1`, the opponent also repeats for that
duration. If `decision_source_frames > 1`, both players' actions are held across
the source-frame substeps.

For the trusted granular default this is fine: both players produce one action
per source tick. For any explicit action repeat, however, the opponent timing is
not independent. A wall-avoidant or checkpoint opponent can react only at the
same repeated cadence as the ego action.

Recommendation: document this as "joint action repeat", not just "ego action
repeat". If the intent is ego-only stickiness while the opponent reacts every
source tick, the current architecture does not do that.

### 7. Action-repeat knobs are not yet conceptually clean

`decision_source_frames` and `policy_action_repeat_*` are both action-repeat
mechanisms from the policy's point of view:

- `decision_source_frames=N` holds the same joint action inside one public
  vector step;
- `policy_action_repeat=N` loops multiple public vector steps before returning
  the next LightZero observation.

The info payload distinguishes them, but the trainer config only exposes
`decision_ms` and `policy_action_repeat_*`, which makes the first one too easy
to miss.

Recommendation: for the trusted lane, assert all of these together:

```text
decision_source_frames == 1
decision_ms == SOURCE_PHYSICS_STEP_MS
policy_action_repeat_min == 1
policy_action_repeat_max == 1
policy_action_repeat_extra_probability == 0.0
source_physics_substeps_executed == [1]
policy_action_repeat_executed == 1
```

If an experiment wants action repeat, prefer the explicit
`policy_action_repeat_*` knobs and label the resulting data as repeated-action
data. Avoid using `decision_ms` as the repeat interface.

## Test Gaps To Close

- A config-surface regression test for `_build_visual_survival_configs` proving
  the trusted `--mode train` env config exposes one source frame per LightZero
  step, not just a small `decision_ms`.
- An env-boundary regression test that steps the trusted-config wrapper once and
  checks both wrapper info and underlying vector info:
  `decision_source_frames=1`, `source_physics_substeps_executed=[1]`,
  `physical_step_index=1`, and `public_env_info.tick_index=1`.
- A timeout semantics test proving `source_max_steps=N` ends after N source
  physics frames in the trusted config.
- A negative or explicit-legacy test showing that `decision_ms` larger than one
  source frame is either rejected for trusted train or labeled as bundled
  source-frame decision mode.
- A repeat semantics test that makes clear repeat affects the joint action
  cadence, including the opponent action.

## Suggested Contract Language

For the trusted stock LightZero training lane:

```text
One LightZero env step advances exactly one CurvyTron source physics frame.
The scalar LightZero action and the env-selected opponent action are applied
simultaneously for that one source frame. `source_max_steps` is counted in
source physics frames. Any multi-frame hold must be requested explicitly through
an action-repeat or legacy decision-bundle field and must be visible in config,
info, telemetry, and run summaries.
```

That contract is small enough to test and strong enough to prevent the old
12-frame bundle from sneaking back in through a friendly-looking timing field.
