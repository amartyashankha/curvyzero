# Coach Handoff: LightZero / CurvyTron Interface Reality Check

Date: 2026-05-11

## Purpose

This is a short handoff for the training coach. No experiments were run and no
training claim was made. The work was an interface sanity check: read the
LightZero/CurvyTron training notes against the source-fidelity notes and clarify
what the coach is really asking for.

## What I Did

- Read the coach/training notes around LightZero, self-play, CurvyTron wrappers,
  shared reporting, and repo-native actor-loop shape.
- Read the trainer-facing CurvyZero env code and local LightZero-shaped smoke
  adapter.
- Rechecked those against the source-map/environment docs after the user
  correctly challenged the phrase "CurvyTron advances after a full joint
  action."

## Main Correction

Do not treat the trainer abstraction as the source game truth.

Source CurvyTron is a real-time server game:

- players hold or change input/control state;
- source controls resolve to move values `-1`, `0`, or `1`;
- the server advances by elapsed milliseconds, not by an inherent RL action
  step;
- avatar update order, collisions, trails, scoring, and events happen inside
  that source frame/lifecycle model.

The current `curvyzero-v0` and LightZero-shaped adapter expose discrete
left/straight/right decisions. That can be a useful training wrapper, but it is
an imposed decision cadence/control abstraction. It should be labeled that way.

## Coach Takeaway

The coach is not asking whether LightZero contains MuZero. It does. The useful
question is what exact decision process we impose between LightZero and a
real-time CurvyTron-like environment.

Before a CurvyTron LightZero run is treated as meaningful, the run should say:

- what decision cadence is used;
- whether actions mean held controls, control changes, or toy-v0 turn commands;
- what elapsed-ms policy is used per trainer step;
- which ruleset is active: toy `curvyzero-v0` or source-derived CurvyTron;
- what opponent/controller policy supplies non-ego controls;
- what replay metadata records control state, elapsed time, source events,
  terminal cause, final observations, and reward maps.

LightZero can consume a discrete ego-control interface. It should not be allowed
to define the source game semantics.

## Relevant Files

- `docs/working/training_coach_handoff_2026-05-09.md`
- `docs/working/lightzero_curvytron_wrapper_verbose_brief_2026-05-09.md`
- `docs/working/lightzero_selfplay_capability_explainer_2026-05-09.md`
- `docs/working/shared_training_reporting_contract_2026-05-09.md`
- `docs/working/training_lessons_for_curvytron_2026-05-09.md`
- `docs/working/repo_native_ppo_learner_boundary_2026-05-09.md`
- `docs/working/optimizer/actor_loop_architecture_2026-05-09.md`
- `docs/research/curvytron_source_map/facts_index.md`
- `docs/research/curvytron_source_map/movement_controls.md`
- `docs/design/rulesets.md`
- `docs/design/deterministic_environment.md`
- `docs/design/environment/training_interface_contract.md`
- `docs/working/environment/per_tick_elapsed_support_plan.md`
- `docs/working/environment/full_environment_spec_2026-05-09.md`
- `src/curvyzero/env/core.py`
- `src/curvyzero/env/trainer_contract.py`
- `src/curvyzero/env/trainer_observation.py`
- `src/curvyzero/training/curvyzero_lightzero_smoke.py`
- `tests/test_curvyzero_lightzero_smoke.py`

## Non-Claims

- Not a LightZero evaluation.
- Not a CurvyTron fidelity proof.
- Not a recommendation to scale training.
- Not proof that the current toy-v0 action API is wrong; only that it is a
  training abstraction, not native source semantics.
