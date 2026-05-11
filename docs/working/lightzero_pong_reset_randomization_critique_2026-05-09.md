# LightZero Pong Reset Randomization Critique - 2026-05-09

Scope: critique only. No pytest. No code edits.

## Short Read

The repeated `episode_seed=2` problem was a seed-plumbing problem first, not a
call for broad randomization. Fix dynamic episode seeds before the next run and
judge that run with seed histograms plus an independent held-out scorecard.

Dummy Pong already has a tiny reset distribution: seeded ball row, serve
direction, and vertical ball direction. What it does not have is a named reset
profile, paddle-position variation, or hard contact/curriculum starts. Add
those deliberately later, not as anonymous noise in the next LightZero rerun.

## What Repeated

In `DummyPongLightZeroEnv.reset()`, the same `episode_seed` is passed to both:

- `PongEnv.reset(seed=episode_seed)`;
- `opponent_policy.reset(episode_seed, opponent_agent)`.

So, for the 4096/64 run dominated by `episode_seed=2`, both things repeated
when the opponent was `random_uniform`:

- the initial Pong state sampled by reset: ball `y`, ball `vx`, and ball `vy`;
- the random opponent action stream, because `RandomUniformPolicy.reset()`
  reseeds its RNG from base seed plus `episode_seed` plus agent offset.

The fixed parts also repeated: both paddles start centered and the ball starts
at center `x`. For deterministic opponents such as `track_ball` and
`lagged_track_ball_1`, there is no separate random opponent stream; repeating
the seed mainly repeats the reset state, and the opponent then follows the
state deterministically.

Important nuance: the ego policy was changing during training, so full
trajectories were not necessarily byte-identical. But the task slice was
still one repeated initial sample plus one repeated random-opponent stream.
That is enough to make trainer-side wins look much better than held-out
checkpoint quality.

## Is Dynamic Seeding Enough?

Enough for the next run: yes.

Dynamic seeds fix the immediate trust bug. They make the existing reset
distribution actually vary across episodes, and they prevent one easy seed from
dominating the sidecar summary. The next run should prove:

- `episode_seed` is not dominated by one value;
- trainer telemetry is labeled as sidecar telemetry, not final quality;
- independent MCTS scorecard uses a fresh recorded eval-wave seed list and the
  same horizon.

Enough as a long-term robustness plan: no.

Dynamic seeds only sample the current narrow reset distribution. They do not
create a broader training family. The research notes make the split clear:
standard MuZero can train on reset-time variation, but variation must be named,
seeded, logged, and scored separately from the clean baseline.

## Dummy Pong Reset Plan

For the immediate post-seed-fix LightZero run, do not add new reset randomness.
Keep the task shape fixed so the run answers one question: did dynamic seed
handling remove the repeated-seed artifact?

After that, add a named reset profile before adding more knobs. Suggested order:

1. `canonical_v0`: current behavior. Paddles centered, ball center `x`, seeded
   ball `y`, seeded serve direction, seeded `vy`.
2. `mild_reset_v1`: same rules, but add small paddle-y jitter. Keep both
   paddles legal and not pinned to the top/bottom at reset.
3. `curriculum_contact_v1`: separate training/eval profile with near-contact
   starts, used only when the goal is paddle-contact learning.

Safe first knobs:

- Ball `y`: already safe and already present. Keep it.
- Serve direction / ball `vx`: already safe and already present. Keep it.
- Ball `vy`: already safe for `-1/+1`. Adding `0` is probably okay later, but
  it changes early geometry and should be a profile change.
- Paddle `y`: safest new knob, if kept mild. It teaches recovery from noncenter
  paddles without changing rules.

Knobs that change the task more:

- Wide paddle-y ranges at reset: useful later, but can turn the early problem
  into recovery-from-bad-placement instead of learning normal Pong geometry.
- Near-contact starts: useful as curriculum or diagnostic data, not canonical
  training. They overrepresent rare tactical states.
- Random ball `x` away from center: useful later, but it changes timing and can
  create immediate-contact or unfair short-horizon episodes.
- Random paddle height, board size, speed, or bounce mechanics: domain
  variation, not reset hygiene.

## Stochasticity To Wait On

Wait on sticky actions, action noise, mechanics noise, random frame skip, and
domain randomization until clean dynamic-seed Pong is learnable and scorecards
move.

These are useful robustness tools, but they answer a later question: can a
policy survive nuisance variation? They should be named profiles with metadata,
not blended into canonical training.

Suggested later order:

- observation/replay augmentation first, if visual overfit is the issue;
- mild reset/domain profiles second, one parameter family at a time;
- sticky/frozen controls as scorecard rows before training with them;
- mechanics noise, random frame skip, paddle/ball speed sweeps, and bounce-rule
  variation only after the baseline is stable.

None of this requires Stochastic MuZero at first. Standard MuZero is fine for
reset-time variation and wrapper noise as data distribution changes. Revisit
Stochastic MuZero only when search must branch over unresolved future chance
events, such as random items, hazards, trail gaps, or strategically important
hidden random outcomes.

## Carryover To CurvyTron

The CurvyTron version of this lesson is:

- fix seed plumbing before claiming learning;
- keep `curvyzero-v0` deterministic and clean for fidelity/debug gates;
- add robust reset/domain profiles only as named lanes;
- log rules hash, variation profile, sampled parameters, opponent policy, and
  all random streams in replay and eval.

CurvyTron reset variation should start with spawn geometry and orientation
variation inside source-faithful rules. Later profiles can vary arena size,
speed, turn rate, trail width, colors, observation noise, and opponent mixture.
Harder source mechanics such as bonuses, trail gaps, hazards, and stochastic
events should wait until the deterministic trainer and clean scorecard are
credible.

The same warning applies: a fixed seed can make a policy look good at one spawn
or one opponent path. A broad randomization profile can also hide failures if
it is not labeled. CurvyTron needs both: deterministic canonical eval and
separate robustness rows.

## Smallest Next Implementation

After the seed fix, the smallest recommended implementation is not new physics
randomization. It is auditability:

1. run the modest post-seed-fix LightZero attempt with `dynamic_seed=True`;
2. fail the run if one `episode_seed` dominates the sidecar;
3. run the independent held-out MCTS scorecard;
4. record the current reset profile name in telemetry, even if it is just
   `canonical_v0`.

Only after that passes should dummy Pong add `mild_reset_v1` with small
paddle-y jitter. Near-contact starts and sticky/action/mechanics noise should
stay separate later profiles.

## Sources Read

- `docs/research/robustness_randomization_for_muzero.md`
- `docs/research/training/domain_variation_for_robustness.md`
- `docs/research/stochastic_muzero.md`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `docs/working/lightzero_trainer_scorecard_mismatch_2026-05-09.md`
- `docs/working/lightzero_pong_checkpoint_diagnostics_2026-05-09.md`
- `docs/working/lightzero_pong_post_seed_fix_run_plan_2026-05-09.md`
