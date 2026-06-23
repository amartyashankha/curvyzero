# Working Memory

Use this folder for scratch notes, temporary synthesis, and raw agent handoffs that are too messy for top-level docs.

Working notes should either be promoted into `docs/design`, `docs/decisions`, `docs/research`, or deleted once they stop being useful.

Use `raw-handoffs/` for undigested handoff text and agent context dumps.

## Active Lane Spines

- `../../goal.md`: optimizer front door for speed, training-loop ownership,
  CPU/GPU split, current baseline, and next gates.
- `optimizer/README.md`: optimizer folder index. It points back to `goal.md`
  before any historical optimizer notes.
- `optimizer/reorientation_2026-05-23/CURRENT_STATE.md`: current optimizer
  truth after `goal.md`.
- `optimizer/reorientation_2026-05-23/TASK_BOARD.md`: current optimizer task
  board after `goal.md` and `CURRENT_STATE.md`.
- `optimizer/batched_gpu_full_loop_reorientation_2026-05-20/README.md`:
  historical/supporting optimizer investigation for batched GPU observations,
  RND compatibility, host-overhead/dataflow, and full-loop Amdahl. Do not use
  it as the current optimizer spine when it disagrees with `goal.md`.
- `optimizer/lane_contract_2026-05-10.md`: short boundary between Optimizer,
  Coach, and Environment/RAM reconstruction.
- `optimizer/runtime_verdict_2026-05-10.md`: historical compact verdict. Use
  only for audit if it disagrees with the current optimizer README.
- `environment/active_lanes.md`: environment/source-fidelity lane map.
- `training/r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md`: current
  broad CurvyTron launch defaults and CZ26 control-plane truth.
- `training/training_loop_extension_refactor_2026-05-19/CURRENT_PHASE.md`:
  current trainer-refactor truth.

## Current Training Working Docs

Current spine. Read these first:

- `training/r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md`: current
  broad CurvyTron defaults, despite the historical folder name.
- `training/training_loop_extension_refactor_2026-05-19/CURRENT_PHASE.md`:
  current trainer code-shape/refactor truth.
- `training/exploration_bonus_rnd_2026-05-19/README.md`: RND/exploration-bonus
  lane, including the fact that positive `rnd_replay_target_v0` has been
  launched experimentally but is not production-settled.
- `training/curvytron_feedback_loop/POLICY_OBSERVATION_CONTRACT.md`: controlled
  player policy-observation contract.
- `../runbooks/training_smokes.md`: commands; check command age before copying.
- `../design/training_eval_protocol.md`: stable eval rules.
- `../experiments/README.md`: dated evidence index.

Historical/supporting notes. Do not let their old next-action lists compete:

- `training_state_index_2026-05-09.md`: old map. Useful only if a newer doc
  explicitly points back to it.
- `training_coach_handoff_2026-05-09.md`: old restart packet.
- `training_experiment_backlog.md`: old ledger.
- `training_loop_agenda.md`: broader current state and historical queue.
- `training_coach_packet.md`: older purpose/priorities packet; prefer the
  state index plus handoff for current navigation.
- `pong_training_critique_wave_2026-05-09.md`: repair-vs-baseline critique.
  Useful context, but current lane order is in the state index and handoff.
- `pong_selfplay_training_plan_2026-05-09.md`: older Pong self-play hypothesis
  and gen2 failure correction. Frozen-checkpoint LightZero self-play is now the
  immediate staged path; true live current-policy self-play remains separate.
- `pong_training_plan.md`: historical Pong visual-toy plan plus completed
  plumbing notes. Do not use its old imitation-first sections as current
  direction.
- `training_coach_reorientation_2026-05-09.md`: drift correction and north
  star reset.

Supporting working notes:

- `pong_angle_learning_next_steps_2026-05-09.md`: angle/contact diagnostics,
  not the main scoreboard.
- `pong_next_signal_experiment_2026-05-09.md`: next Pong signal probes after
  one-step lookahead failures.
- `pong_survival_target_recovery_2026-05-09.md`: loss-delay training target
  recovery note and guardrails.
- `pong_state_critique_2026-05-09.md`: critique of stale Pong claims.
- `dummy_survival_degradation_diagnosis_2026-05-09.md`: why survival is now a
  diagnostic lane.
