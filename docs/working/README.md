# Working Memory

Use this folder for scratch notes, temporary synthesis, and raw agent handoffs that are too messy for top-level docs.

Working notes should either be promoted into `docs/design`, `docs/decisions`, `docs/research`, or deleted once they stop being useful.

Use `raw-handoffs/` for undigested handoff text and agent context dumps.

## Active Lane Spines

- `optimizer/README.md`: optimizer front door for speed, profiling, Amdahl,
  CPU/GPU split, Modal boundaries, and setup measurement.
- `optimizer/lane_contract_2026-05-10.md`: short boundary between Optimizer,
  Coach, and Environment/RAM reconstruction.
- `optimizer/runtime_verdict_2026-05-10.md`: compact current CurvyTron
  source path, CPU/GPU boundary, profile numbers, and near-term optimizer
  verdict.
- `environment/active_lanes.md`: environment/source-fidelity lane map.
- `training_state_index_2026-05-09.md`: coach/training lane map.

## Current Training Working Docs

Active spine. Read these first:

- `training_state_index_2026-05-09.md`: compact map of the training docs
  hierarchy. Use this first when deciding which detailed doc to open.
- `training_coach_handoff_2026-05-09.md`: compact restart packet. Read this
  first after memory wipe; it states the truth, operating pattern, reward
  rules, active research lanes, and shorthand.
- `training_experiment_backlog.md`: active lane ledger and newest run-lineage
  synthesis. It is long; use the top current-truth sections first.
- `../runbooks/training_smokes.md`: commands; self-play commands are
  reproduction only.
- `../design/training_eval_protocol.md`: stable eval rules.
- `../experiments/README.md`: dated evidence index.

Historical/supporting notes. Do not let their old next-action lists compete:

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
