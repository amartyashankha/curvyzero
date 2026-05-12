# CurvyTron Observability Inspector Brief - 2026-05-11

Purpose: give the next inspector agent a clean starting point for CurvyTron
evals and observability. Keep this broad. The inspector can turn it into code.

## Plain Situation

CurvyTron is a two-player survival game. Each player leaves a trail. A player
dies by hitting a wall, its own trail, or the other player's trail. The simple
training target is survival time: how long the policy stays alive.

The current training problem is not yet "prove final learning." The current
problem is more basic:

- Do the games end for sensible reasons?
- Are trained checkpoints surviving longer over time?
- Are policies doing varied actions, or are they collapsing to one action?
- Are evals comparing the same timestep, same game settings, same checkpoint,
  and same opponent setup?
- Can we inspect a few bad episodes and understand why they died?

Right now, the answer is: not cleanly enough. We have scattered logs, progress
files, eval manifests, and docs, but not one reliable view that explains a run.

## Current Learning Reality

The current best CurvyTron two-seat run is:

```text
run_id: curvytron-two-seat-selfplay-live12-playerpersp-episodefix-clean-long-20260511
attempt_id: live12-playerpersp-episodefix-clean-long-20260511
```

It uses current-policy self-play in the two-seat custom path. Both players are
controlled by the current policy/search, actions are combined into one joint
step, and the same policy is updated.

Important recent fixes:

- Player-perspective visual input: each policy row now sees "self" and "other"
  consistently.
- `to_play` should not use CurvyTron public player ids `0/1`; use `-1` for
  single-agent/bot-style rows unless a tested board-game `1/2` contract exists.
- Survival return targets now use episode ids, so targets are not accidentally
  cut at outer training iteration boundaries.

Current status from the latest docs: the run changes model weights and does not
show a hard one-action collapse, but survival has not shown a clear lift yet.

## Important Mechanic

At `decision_ms=300`, one action is held for a large chunk of physical time. A
weak or simple policy dying after about 10-20 decisions can be normal for the
current game scale. This does not prove the environment is broken.

The inspector should report both:

- decisions survived
- approximate wall-clock game time survived: `decisions * decision_ms / 1000`

Do not compare survival curves unless `decision_ms`, map size, reward, max
ticks, eval path, and opponent setup are recorded together.

## What The Inspector Should Build First

Build a read-only report first. Do not start by redesigning training.

Inputs:

- `run_id`
- `attempt_id`
- optional eval manifest path
- optional checkpoint list

Output one report with:

- run identity: config, timestep, reward, max ticks, self-play/fixed-opponent
  status, checkpoint cadence
- checkpoint list: which checkpoints exist and which have evals
- training curve: iteration, mean survival, max survival, episode count, action
  mix, model changed, problem count
- eval curve: survival by checkpoint over a common random seed set
- action check: action histogram and top-action fraction
- death summary: wall/body/timeout/draw when available
- trace samples: a few short episodes from first, middle, latest, and best
  checkpoints
- warnings: missing files, mixed timesteps, no eval for latest checkpoint,
  possible action collapse, fixed-opponent result presented as self-play

## What Is Missing

The biggest missing field is death explanation. We often know that an episode
ended, but not clearly enough why it ended.

The inspector should either surface or help add:

- death cause: wall, own trail, opponent trail, draw, timeout
- player that died
- player position and heading before death
- action taken before death
- nearest wall/trail distance before death, if easy
- reset/spawn settings
- `decision_ms`, map size, radius, speed, turn amount per decision

This should be enough to answer: "Did the agent die because it drove into a wall,
looped into itself, hit the opponent, timed out, or because the eval wrapper is
wrong?"

## First Useful Questions

1. Are simple baselines sane?
   Compare straight, random, wall-avoid, and ray-clearance policies at the same
   `decision_ms` as training.

2. Are trained checkpoints beating those simple baselines?
   Use survival time first. Score is secondary.

3. Is the policy collapsing?
   Look for one action taking almost all decisions.

4. Are runs comparable?
   Reject comparisons that mix timestep, map size, eval path, or opponent type.

5. Are deaths understandable?
   For a few short episodes, show enough trace detail to explain the crash.

## Relevant Files

- `src/curvyzero/training/curvytron_baseline_eval.py`
- `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`
- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py`
- `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py`
- `src/curvyzero/env/vector_multiplayer_env.py`
- `src/curvyzero/env/vector_trainer_env.py`
- `src/curvyzero/env/vector_runtime.py`
- `src/curvyzero/env/trainer_observation.py`
- `docs/working/training/archive_2026-05-12_two_seat_purge/coach_current_state_2026-05-10.md`
- `docs/working/training/curvytron_no_learning_investigation_2026-05-11.md`
- `docs/working/training/archive_2026-05-12_two_seat_purge/curvytron_background_training_ledger_2026-05-10.md`

## Non-Claims

- We have not proven stable CurvyTron learning yet.
- We should not treat fixed-opponent results as proof of self-play progress.
- We should not call 10-step episodes broken unless the timestep, death cause,
  and baseline behavior say they are broken.
