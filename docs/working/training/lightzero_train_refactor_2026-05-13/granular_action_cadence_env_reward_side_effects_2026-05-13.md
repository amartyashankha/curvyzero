# Granular Action Cadence Env/Reward Side-Effect Audit

Date: 2026-05-13

## Scope

This audit covers env and reward semantics for the trusted stock LightZero
`source_state_fixed_opponent` train path after the cadence change.

It reads the refactor notes and inspects only the source-state LightZero wrapper,
the vector env, and nearby eval/test surfaces needed to answer the side-effect
questions.

No source code was edited for this audit.

## Bottom Line

The trusted path now has a clear one-policy-action-per-source-physics-frame
contract.

That is good, but it is not behavior-neutral versus old bundled-step runs. The
same numeric `source_max_steps`, survival return, timeout timing, and eval
`steps_survived` now mean source physics frames in the trusted lane. Old runs
with hidden multi-frame `decision_ms` are not directly comparable unless reports
also show `decision_source_frames`, `source_physics_step_ms`, and reward variant.

## Facts

- The wrapper default is now `DEFAULT_DECISION_SOURCE_FRAMES = 1`, and
  `DEFAULT_DECISION_MS = SOURCE_PHYSICS_STEP_MS`.
- The trusted trainer config writes `decision_source_frames = 1`,
  `source_physics_step_ms`, and `source_max_steps_semantics =
  source_physics_steps`.
- Trusted `--mode train` and `--mode dry` reject stale bundled `decision_ms`
  values for `source_state_fixed_opponent`. The error says to use
  `policy_action_repeat_*` for explicit repeat.
- The wrapper still accepts larger `decision_ms` in profile-like paths. That is
  useful for diagnostics, but those runs are outside the trusted train/dry
  contract.
- The vector env supports two modes:
  - no `decision_source_frames`: one vector step uses one larger `decision_ms`;
  - with `decision_source_frames`: one vector step loops source physics substeps.
- In source-frame mode, source turn input is applied only on substep 0. The
  resulting current angular velocity persists through later substeps.
- The wrapper calls the vector env once per policy repeat and passes
  `timer_advance_ms = self._decision_ms`.
- The wrapper increments `_physical_step_index` once per vector step, not once
  per LightZero policy transition. With default repeat 1, this equals source
  ticks. With explicit repeat N, one LightZero transition may advance N source
  ticks.
- The vector runtime increments `state["tick"]` once per source physics substep.
- The vector env increments `episode_step` once per public vector env step, not
  once per internal substep.

## `source_max_steps` And Max Ticks

Current trusted semantics:

- Trainer config sets both `source_max_steps` and env `max_ticks` to the same
  number.
- Wrapper stores `_max_ticks = max_ticks/source_max_steps` and computes
  `_max_source_ticks = _max_ticks * decision_source_frames`.
- With trusted `decision_source_frames = 1`, underlying vector `max_ticks`
  equals configured `source_max_steps`.
- Timeout is marked after runtime stepping, when source `tick >= max_ticks`.

Side effect:

- `source_max_steps=65536` now means 65,536 source physics frames.
- Under the old hidden 12-frame default, the same configured number could become
  786,432 underlying source ticks. Old and new runs with the same
  `source_max_steps` are therefore not horizon-equivalent.

Risk:

- Dashboards or eval tables that only show `source_max_steps` may make old and
  new runs look comparable when they are not.
- `max_env_step` and LightZero collector step counters count policy transitions,
  not necessarily source ticks when explicit `policy_action_repeat_*` is used.

Recommended tests/smokes:

- Keep the existing test that `source_max_steps=3` times out after 3 source
  ticks in the default trusted wrapper.
- Add/keep a config-surface assertion that `source_max_steps_semantics` is
  present in train, background eval, GIF, and manifests.
- Add one report-table smoke that renders both `steps_survived` and
  `decision_source_frames` side by side.

## Natural Bonus Timers

Current trusted semantics:

- Natural bonus spawning is enabled by default in the wrapper config.
- The wrapper passes one source-frame worth of `timer_advance_ms` per default
  step.
- In vector source-frame mode, timer budget is split across substeps. With
  `decision_source_frames = 1`, there is only one substep.
- Bonus timer state is exposed in info as natural bonus fields, including
  `natural_bonus_pop_count`, remaining timer, and source pop time metadata.

Side effect:

- Bonus spawning is still based on source elapsed time, not policy count in the
  abstract. But the policy now gets observations and reward opportunities every
  source frame, so it can react around bonuses at a finer cadence.
- For the same configured `source_max_steps`, trusted runs now cover fewer
  source milliseconds than old hidden 12-frame runs. That can reduce the number
  of natural bonus pops per episode unless `source_max_steps` is increased.

Risk:

- `survival_plus_bonus_no_outcome` runs may see different bonus pickup density
  per episode because the horizon changed.
- Profile tests still use large `decision_ms`; those are good timer stress tests
  but should not be treated as trusted train semantics.

Recommended tests/smokes:

- Add a deterministic no-death natural-bonus smoke with
  `decision_source_frames=1` and enough `source_max_steps` to force at least one
  pop.
- Add a paired diagnostic that compares natural bonus pop count after the same
  source tick count, not after the same policy step count, for one-frame and
  explicit-repeat configs.

## Reward Scale

Current trusted semantics:

- `sparse_outcome` uses the vector env terminal reward map.
- `dense_survival_plus_outcome` adds `1.0` while ego is alive after each wrapper
  physical step, plus terminal sparse outcome.
- `survival_plus_bonus_no_outcome` adds `1.0` while ego is alive after each
  wrapper physical step, plus `1.0` per same-step bonus catch, and ignores
  terminal outcome for training reward.
- With default repeat 1, dense survival reward is now one reward unit per source
  physics frame.
- With explicit policy repeat N, the wrapper loops and sums reward components,
  so one LightZero transition can return up to N survival units plus bonus.
- The trainer target config scales `td_steps` from `source_max_steps`. For
  `survival_plus_bonus_no_outcome`, requested value support is
  `source_max_steps * (1 + bonus_reward)`, then capped by the configured model
  support cap.

Side effect:

- Per-source-second dense survival reward is much larger than old bundled
  transition reward if old runs emitted only one survival unit per 12 source
  frames.
- Total possible episode return is now aligned with source ticks, but the value
  support cap can hide the full theoretical scale for large horizons.

Risk:

- Learning curves, value targets, and checkpoint comparisons can change simply
  because the reward clock changed.
- If explicit action repeat is used later, reward per LightZero transition is
  no longer bounded by the default per-frame reward even though source cadence is
  still granular internally.

Recommended tests/smokes:

- Keep the existing explicit-repeat test that repeat 3 returns survival reward
  3.0 and reports `physical_decision_ms_total = 3 * SOURCE_PHYSICS_STEP_MS`.
- Add a tiny train-loop smoke that runs at least one learner update with
  `survival_plus_bonus_no_outcome` and confirms no reward/value support shape
  errors.
- Add a manifest/report check that surfaces reward variant, reward schema id,
  `decision_source_frames`, support cap, and whether reward/value support was
  capped.

## Survival Metrics

Current trusted semantics:

- Eval counts one action loop iteration as one `steps_survived`.
- In the trusted default, that equals source physics frames.
- Telemetry includes `physical_step_index`, `source_tick_index`,
  `decision_source_frames`, `source_physics_step_ms`, and policy repeat fields.
- `survival_length_is_eval_metric` is true, but survival length is telemetry for
  reward schemas that are not pure sparse outcome.

Side effect:

- `steps_survived` is now finer-grained. A policy surviving the same wall-clock
  game time will report about 12x more steps than a legacy 12-frame policy-step
  metric, if old reports counted bundled decisions.
- If old eval paths infer `decision_source_frames` from checkpoint metadata, old
  and new checkpoints can coexist but must be labeled.

Risk:

- Leaderboards or checkpoint selectors may rank mixed-cadence policies by raw
  `steps_survived` without normalization.

Recommended tests/smokes:

- Add an eval aggregation test where two rows with different
  `decision_source_frames` keep their cadence fields in the aggregate output.
- Add a leaderboard/report guard that refuses to merge raw survival metrics
  across cadence unless it labels or normalizes them.

## Wall And Trail Collision Timing

Current trusted semantics:

- Vector runtime movement, visual/body trail writes, wall checks, body collision
  checks, bonus catches, terminal scoring, and source tick increment happen in
  the source runtime step.
- With trusted `decision_source_frames = 1`, each LightZero default step runs
  exactly one source runtime step.
- Wall deaths and body/trail deaths can now be observed at the first source tick
  where they happen.
- Trail collision latency is stored in source body/tick units. The trusted
  default now exposes every unit to the policy.

Side effect:

- The policy receives observations before intermediate wall/trail events that
  used to be hidden inside a bundled decision.
- Collision timing is not delayed by the wrapper in the trusted default.
- Old hidden bundled runs may have held a turn through several collision checks
  before the policy could change action.

Risk:

- Policies trained under old cadence may over-turn or under-react when evaluated
  at one-frame cadence.
- Wall-avoidant or trail-avoidant opponent behavior may change because it is
  queried once per source frame in the trusted default.

Recommended tests/smokes:

- Add a deterministic near-wall fixture that proves a turn on the next source
  tick can avoid or cause a wall death as expected.
- Add a trail-collision fixture that reports `death_tick`, `source_tick_index`,
  and `decision_source_frames=1`.
- Keep old bundled/product-fidelity tests clearly marked outside the trusted
  train contract if they intentionally use multi-frame decisions.

## Action Repeat Semantics

Current trusted semantics:

- Hidden repeat through stale `decision_ms` is rejected in train/dry.
- Explicit repeat lives in `policy_action_repeat_min`,
  `policy_action_repeat_max`, and `policy_action_repeat_extra_probability`.
- Explicit repeat repeats the same joint action across multiple vector steps.
  With trusted default frame config, each repeat is one source physics frame.
- The wrapper returns one LightZero timestep for the whole repeat and sums the
  reward components over executed repeats.
- Repeat stops early if the episode ends or times out.

Side effect:

- Explicit repeat preserves source-frame mechanics but reduces policy
  observation frequency.
- LightZero transition reward can become a sum over several source ticks.

Risk:

- Config or telemetry text still says "one policy action per source frame" unless
  repeat fields are read too. The true statement is: default trusted repeat is
  one frame; explicit repeat is allowed and reported.

Recommended tests/smokes:

- Keep the existing repeat cap test where requested repeat 3 executes only 2
  steps because `source_max_steps=2`.
- Add a telemetry row assertion that explicit repeat records both
  `policy_action_repeat_requested` and `policy_action_repeat_executed`.

## Opponent Cadence

Current trusted semantics:

- LightZero provides one scalar ego action.
- The wrapper computes one opponent action before the repeat loop.
- That same joint action is used for every executed repeat.
- With default repeat 1, the opponent acts once per source physics frame.
- With explicit repeat N, the opponent action is also held for N source frames.
- `blank_canvas_noop` disables player-1 movement/trail/collision/bonus side
  effects while keeping public player-1 present/alive metadata.
- `opponent_death_mode=immortal` protects the opponent from death but does not
  protect ego.
- Proactive wall-avoidant opponent geometry uses the wrapper decision duration.
  In trusted default that is one source physics frame.

Side effect:

- Fixed straight and proactive opponents now update at source-frame cadence in
  trusted default runs.
- Frozen LightZero opponents are queried at the same cadence as ego in default
  trusted runs.
- Opponent-mixture comparisons across old and new cadence need labels, because
  the same frozen checkpoint may behave differently under a different
  environment cadence.

Risk:

- Opponent policy sidecars and tournament reports can understate cadence if they
  show opponent kind but not decision frame fields.

Recommended tests/smokes:

- Add a fixed-opponent cadence smoke asserting one opponent action per default
  source tick and held opponent action under explicit repeat.
- Add a frozen-opponent smoke that records `decision_source_frames=1` in the
  opponent policy sidecar or surrounding episode metadata.
- Add a tournament/eval guard for mixed cadence checkpoint pairs.

## Highest-Priority Validation Before Serious Runs

1. Run a tiny real stock `train_muzero` smoke after this patch, not just config
   tests.
2. Run a tiny eval/GIF smoke and inspect that `decision_source_frames`,
   `source_physics_step_ms`, `source_max_steps_semantics`, reward variant, and
   repeat fields appear in artifacts.
3. Run a deterministic env smoke for timeout, near-wall death, trail death, and
   one natural bonus pop under `decision_source_frames=1`.
4. Add a report/leaderboard guard so raw `steps_survived` is not silently mixed
   across old bundled cadence and new source-frame cadence.

## Useful Existing Coverage

- Wrapper default cadence test: one default step advances one source frame.
- Wrapper `source_max_steps` cap test: cap is granular source ticks.
- Explicit repeat test: repeat is one LightZero transition with summed reward
  and reported repeat fields.
- Repeat cap test: repeat stops at timeout.
- Config plumbing test: train config writes one-frame cadence and source-step
  semantics.
- Stale `decision_ms` rejection test for trusted config.
- Background eval/GIF config tests include cadence fields.

## Open Gaps

- No fresh real stock train-loop smoke is recorded in the side-effect notes.
- Mixed-cadence leaderboard/report behavior still needs an explicit guard.
- Natural bonus behavior has profile stress coverage, but the trusted one-frame
  train path should have a small deterministic pop-count smoke.
- Near-wall and trail-death timing should be locked with source-frame fixtures,
  because these are the mechanics most likely to reveal hidden cadence mistakes.
