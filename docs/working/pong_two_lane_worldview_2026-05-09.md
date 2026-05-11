# Pong Two-Lane Worldview - 2026-05-09

Role: worldview/discrepancy critic. No code changes and no pytest.

## Short Answer

The repo now has two Pong lanes because they answer different questions.

Official Atari Pong asks:

```text
Can stock LightZero run its normal visual Atari Pong stack on Modal?
```

Custom dummy Pong asks:

```text
Can CurvyZero control a small game, trainer adapter, telemetry, checkpoint
scoreboard, opponent setup, and sparse-reward debugging path before moving
those ideas to CurvyTron?
```

Those are not duplicates. Official Atari Pong is the outside reference lane.
Custom dummy Pong is the project-owned lab lane.

The confusion came from calling both "Pong" while they have different
worldviews:

| Question | Official Atari Pong | Custom dummy Pong |
| --- | --- | --- |
| Who owns the environment? | LightZero/Gym/ALE | CurvyZero |
| What is the input? | stacked Atari image frames | 10 tabular floats or tiny raster |
| What model shape? | convolutional MuZero | MLP MuZero or small project policies |
| What actions? | Atari 6-action surface | 3 actions: up, stay, down |
| What is being proved? | stock visual trainer mechanics and baseline reproduction | custom env/training/eval plumbing and sparse-game diagnosis |
| CurvyTron relevance | proves the library can do visual Atari control | proves we can own the game contract before CurvyTron |

## Why Custom Dummy Pong Was Created

Custom dummy Pong was created because official Atari Pong was the wrong first
thing for the CurvyZero problem.

The ADR in `docs/decisions/0005-main-pong-repository-library-choice.md` says
the immediate goal was a `LightZero custom dummy Pong MuZero smoke`: adapt a
CurvyZero-owned env to LightZero, produce a checkpoint, and write project-owned
telemetry. It explicitly says not to start from Atari Pong because the ROM/ALE
setup had not been solved yet and Atari carried image/segment assumptions that
were not needed for the first custom-env trainer smoke. That is historical: the later
official/control Pong work did get ALE-backed `PongNoFrameskip-v4` running.
Dummy Pong still remains bridge/debug evidence, not CurvyTron quality evidence
and not a reason to make CurvyTron depend on ALE.

The source matches that intention. `src/curvyzero/training/dummy_pong.py`
describes a "tiny deterministic two-player Pong-like environment" built to
exercise simultaneous two-player controls, ego observations, ball dynamics,
terminal scoring rewards, and time-limit truncation. The LightZero wrapper in
`src/curvyzero/training/lightzero_dummy_pong_env.py` then turns that into a
single-ego LightZero env: LightZero controls one paddle, and the wrapper supplies
the opponent action.

So dummy Pong was not meant to be a fake Atari benchmark. It was a cheap,
inspectable bridge:

- from stock trainer control to custom environment control;
- from single-agent examples to opponent-in-the-env games;
- from generic reward logs to CurvyZero score/survival/action telemetry;
- from "LightZero can run" to "LightZero can run our game contract."

That is close to the CurvyTron problem. CurvyTron is also project-owned,
competitive, step-based, and sensitive to simulator/eval contracts. Atari Pong
is useful, but it cannot answer whether CurvyZero can define and audit its own
game.

## What Custom Dummy Pong Tests That Official Atari Cannot

Custom dummy Pong tests the project-owned seams.

First, it tests a custom environment adapter. The repo has to expose
`reset()`, `step()`, action masks, `to_play`, observation schemas, seed
handling, and LightZero-compatible config surfaces. Official Atari Pong already
has all of that inside LightZero/DI-engine. It proves less about whether
CurvyZero can wire CurvyTron later.

Second, it tests opponent handling. Dummy Pong can run learner-vs-scripted
opponents such as `random_uniform`, `lagged_track_ball_1`, and `track_ball`,
and it now has frozen-checkpoint opponent plumbing. Official Atari Pong has the
Atari opponent/game baked into ALE. That is a cleaner benchmark, but not a
test of our future multiplayer/self-play staging.

Third, it tests sparse reward instrumentation. Dummy Pong keeps the environment
reward honest:

```text
ego scores: +1
ego loses:  -1
otherwise:  0
timeout:    0 with truncation telemetry
```

Then it reports survival steps and shaped loss-delay as telemetry. That split
is important for CurvyTron, where rewarding "not dying" inside `env.step()`
could teach passive play. Official Atari Pong gives Atari rewards, but it does
not test our reward-schema discipline.

Fourth, it tests action-collapse diagnosis. The dummy lane found a real MuZero
target issue: LightZero trains policy logits toward MCTS root visit
distributions, not the exploratory action that was eventually executed. So a
collector can execute `down`, while the stored policy target still gives `down`
zero mass. Official Atari Pong can show whether stock LightZero learns under
stock settings, but it cannot isolate the three-action toy failure as cleanly.

Fifth, it tests independent scoreboards. The dummy lane has fixed baselines,
paired seats, action histograms, raw score, survival, truncation, shaped
loss-delay, and checkpoint loading paths. Official Atari eval is necessary, but
its stock logs are not a substitute for the CurvyZero scoreboard protocol we
need for CurvyTron.

Finally, it tests controllable reset/curriculum hypotheses. `contact_pressure`
starts, lagged opponents, raster-vs-tabular encoders, and policy-head-vs-MCTS
eval can all be manipulated in dummy Pong. ALE Pong is intentionally less
editable.

## What Custom Dummy Pong Misses Compared To Official Atari

Custom dummy Pong misses most of the official visual-control problem.

It is not stock Atari Pong. It does not use ALE physics, ROMs, Atari scoring,
Atari wrappers, or the true 6-action Atari control surface.

It is not the official visual model path. Official LightZero Atari Pong uses
stacked grayscale frames and a convolutional MuZero model. Dummy `tabular_ego`
uses 10 floats. Dummy `raster_flat` uses a tiny single-frame grid flattened
into an MLP, and current notes correctly call that a weak bridge because it
lacks velocity/history.

It is not a scale or quality benchmark. Official Atari configs use much larger
budgets, commonly 50 MCTS simulations, larger batches, replay ratios, and long
episode/data scales. Many dummy runs used 2, 8, or 16 simulations as diagnostic
caps. That is useful for finding bugs, but not comparable to official Atari
quality.

It is not full self-play. The current LightZero dummy env is a single-ego
wrapper over a two-player game. The opponent is scripted or frozen. That is a
reasonable staging lane, but it is not true current-policy two-seat self-play.

It may hide visual and partial-observation issues. `tabular_ego` is probably
Markov enough for this toy, but CurvyTron will likely need richer spatial
observations. A dummy result can debug target semantics, but it cannot prove the
visual stack.

It also has failure evidence. The dummy sparse runs repeatedly showed action
collapse, especially zero `down` in held-out scorecards. The latest diagnosis
is not "run it longer"; it is root-visit target quality, support/value scale,
and telemetry gaps around replay targets.

## Why Official Atari Still Matters

Official Atari Pong is the reality check.

The repo now has positive official-lane mechanics:

- ROM-enabled ALE env reset/step works through LightZero/DI-engine.
- The stock visual trainer path runs on Modal.
- Checkpoint-load and eval mechanics work.
- The GPU1024 control produced a small real signal: final 256-step eval
  improved same-cap return from `-5` to `-3` and saw one `+1` reward.

That does not solve Atari Pong. But it proves the official visual LightZero
stack is alive. It is the guardrail against overfitting the project to a toy
adapter that LightZero never had to support upstream.

In simple terms: official Atari tells us whether the trainer can behave in its
native habitat. Dummy Pong tells us whether we can make the trainer behave in
ours.

## Why Dummy Pong Is Still Worth Keeping

Keep custom dummy Pong, but shrink its authority.

It is still worth keeping because CurvyTron is not an ALE game. CurvyTron needs
exactly the things dummy Pong exercises: custom deterministic stepping,
simultaneous/opponent behavior, project-owned observations, sparse competitive
reward, seed control, replay metadata, independent scoreboards, and artifact
discipline.

But dummy Pong should no longer be treated as "almost official Pong" or as a
quality benchmark. It is a diagnostic environment. Its job is to make bugs
small enough to see before those bugs become expensive in CurvyTron.

The best current use is narrow:

- use `tabular_ego` to debug LightZero target semantics, support scale, seed
  plumbing, checkpoint loading, and action collapse;
- use `raster_flat` only as a compatibility bridge until it has frame stacking
  or velocity channels;
- use fixed baselines and scoreboards as protocol tests;
- use contact/angle resets as bounded diagnostics, not as a new main game;
- keep survival/loss-delay as telemetry, not environment reward.

## Recommendation

Recommendation: keep the two lanes, merge only the reporting protocol, and
retire parts of custom dummy Pong as progress claims.

Keep:

- Official Atari Pong as the stock LightZero/ALE reproduction lane.
- Custom dummy Pong as the CurvyZero-owned diagnostic lane.
- The `tabular_ego` dummy LightZero adapter for target-quality and action
  collapse debugging.
- The independent dummy Pong scoreboard protocol, including action histograms,
  raw score, survival, truncation, shaped loss-delay, seed info, and checkpoint
  refs.
- Frozen-checkpoint opponent plumbing as a staged bridge toward self-play.

Merge:

- The language and report shape across lanes. Every report should say plainly
  whether it is `official Atari Pong` or `custom dummy Pong`.
- The artifact/checkpoint discipline: run ids, attempt ids, config surface,
  checkpoint refs, hashes, eval mode, and no-fallback status.
- The evaluation habit: do not trust trainer-side telemetry alone; use
  independent eval where possible.

Retire or demote:

- Retire "dummy Pong as visual Pong parity" language. It is not parity.
- Demote `raster_flat` single-frame MLP results to bridge-smoke status only.
- Stop longer same-config sparse dummy runs until target telemetry shows MCTS
  root visits and policy targets are sane.
- Stop promoting trainer-side wins without held-out scoreboards.
- Stop using `track_ball` as a universal score target when probes show some
  reset/opponent combinations are not scoreable.

In one sentence:

```text
Keep dummy Pong as the microscope; keep official Atari Pong as the calibration
target; do not confuse the microscope image with the benchmark.
```

## Mapping To CurvyTron

CurvyTron should inherit the custom dummy Pong discipline, not the dummy Pong
game itself.

Useful carryovers:

- Project-owned environment contract with explicit ruleset, observation,
  action, reward, and reset schemas.
- Sparse true competitive reward as the main environment reward.
- Survival ticks, crash cause, pressure events, distance-to-death, rank, and
  timeout as telemetry.
- Independent scoreboards against fixed baselines, older checkpoints, and
  selected best checkpoints.
- Seed diversity checks and replay target telemetry.
- A staged path from scripted opponents to frozen-checkpoint opponents to real
  current-policy self-play.

Things not to carry over blindly:

- Do not copy dummy Pong's single-ego wrapper and call it CurvyTron self-play.
  It can be a staging shape, but true CurvyTron needs explicit multiplayer
  semantics.
- Do not copy single-frame `raster_flat` as the main visual observation.
  CurvyTron needs an observation design that preserves motion, trail geometry,
  walls, opponents, and likely frame/history or velocity-like information.
- Do not copy survival shaping into `env.step()`. Use survival for curriculum
  and selection tie-breaks only after true payoff is tied.
- Do not assume low-simulation MCTS targets are good enough. CurvyTron will be
  at least as sensitive to root-visit quality as dummy Pong.

The healthy path is:

1. Use official Atari Pong to keep LightZero visual training honest.
2. Use custom dummy Pong to debug CurvyZero-owned training contracts cheaply.
3. Move only the proven contracts into CurvyTron: schema discipline, telemetry,
   scoreboards, checkpoint handling, target audits, and staged opponent design.
4. Leave behind dummy-specific hacks once CurvyTron has its own source-derived
   environment and eval ladder.

## Sources Read

- `README.md`
- `docs/investigation_plan.md`
- `docs/decisions/0005-main-pong-repository-library-choice.md`
- `docs/working/training_state_index_2026-05-09.md`
- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/working/training_experiment_backlog.md`
- `docs/working/lightzero_official_visual_pong_pattern_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_official_parity_gap_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_observation_model_critique_2026-05-09.md`
- `docs/working/lightzero_dummy_pong_eval_ladder_2026-05-09.md`
- `docs/working/lightzero_muzero_target_semantics_2026-05-09.md`
- `docs/research/reward_shaping_for_pong_curvy_muzero.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-env-smoke.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-tiny-train-smoke.md`
- `docs/experiments/2026-05-09-modal-lightzero-pong-gpu1024-control.md`
- `docs/experiments/2026-05-09-lightzero-dummy-pong-implementation-inspection.md`
- `src/curvyzero/training/dummy_pong.py`
- `src/curvyzero/training/lightzero_dummy_pong_env.py`
- `src/curvyzero/training/lightzero_dummy_pong_features.py`
- `src/curvyzero/training/dummy_pong_eval.py`
- `src/curvyzero/training/lightzero_dummy_pong_policy.py`
- `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py`
- `src/curvyzero/infra/modal/lightzero_atari_rom_image.py`
