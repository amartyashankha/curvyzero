# CurvyTron Training Architecture Research - 2026-05-12

Purpose: understand why the recent CurvyTron training work did not learn, why
we moved between stock LightZero and custom two-seat paths, and what must be
true before another large learning claim.

This folder is the working area. Short stable conclusions should flow upward to
the design docs and the Coach index.

> Supersession note, 2026-05-16: this folder is now historical architecture
> research. Current broad launch defaults are centralized in
> `../r18fresh_postmortem_2026-05-16/CURRENT_LAUNCH_DEFAULTS.md` and
> `src/curvyzero/contracts/curvytron.py`. Older references here to
> `body_circles_fast`, `fast_gray64_direct`, `browser_sprites`, batch32, or H100
> are evidence from earlier lanes, not instructions for the current broad lane.

## Current Source

Start here for active work:

- [current_source_of_truth.md](current_source_of_truth.md)
- [user_priority_snapshot.md](user_priority_snapshot.md)
- [todo.md](todo.md)
- [hypotheses_and_evidence.md](hypotheses_and_evidence.md)
- [v1d_axis_projection.md](v1d_axis_projection.md)
- [v1d_fresh_eval_summary_2026-05-13.md](v1d_fresh_eval_summary_2026-05-13.md)
- [next_overnight_matrix_plan.md](next_overnight_matrix_plan.md)
- [opponent_diagnostic_design.md](opponent_diagnostic_design.md)
- [blank_canvas_noop_opponent_lane.md](blank_canvas_noop_opponent_lane.md)
- [scripted_wall_avoidant_opponent_baseline_2026-05-13.md](scripted_wall_avoidant_opponent_baseline_2026-05-13.md)
- [source_state_reward_wiring_audit_2026-05-13.md](source_state_reward_wiring_audit_2026-05-13.md)
- [aggressive_matrix_scale_plan.md](aggressive_matrix_scale_plan.md)
- [launch_gate_checklist.md](launch_gate_checklist.md)
- [eval_curve_tooling_plan.md](eval_curve_tooling_plan.md)
- [delegation_log.md](delegation_log.md)
- [operating_patterns.md](operating_patterns.md)
- [archive_and_stale_docs.md](archive_and_stale_docs.md)

The older docs below remain useful evidence, but may contain historical claims
that have been superseded.

## Current Plain Read

CurvyTron did not fail because visual MuZero is impossible here.

The scaled May 12 path failed to prove learning because it was not the trusted
stock LightZero `train_muzero` path. It solved current-policy two-seat action
collection, but it also replaced LightZero's collector, replay buffer, target
builder, and training loop with repo-owned code.

Plain wording for the main mismatch: stock LightZero gives the env one action
at a time. CurvyTron physics wants both players' actions before one game tick.
Fixed/frozen opponent wrappers avoid that mismatch by letting the env choose
the other player's action. Custom two-seat code tried to make the live policy
choose both players' actions, but then it accidentally took over too much of
training.

The stock LightZero CurvyTron paths still exist:

- `source_state_fixed_opponent`: stock `train_muzero`, one learner-controlled
  ego player, env-owned fixed/frozen-style opponent.
- `source_state_joint_action`: stock `train_muzero`, one scalar action chooses
  both player actions, one real physical tick per replay row.
- `source_state_turn_commit`: stock plumbing/profile only, not trainable today
  because fake pending steps enter replay.

The custom `--mode two-seat-selfplay` path is useful as a collector prototype
and profiler, but it is not currently a trusted learning lane.

Latest proof point: `stock-frozen-canary-source-state-s304-20260512` completed
as a tiny CPU stock `train_muzero` run against a strict frozen LightZero
checkpoint opponent. `stock-frozen-gpu-base-canary-source-state-s304-20260512b`
then completed the same kind of tiny proof on an L4 GPU with
`env_manager_type=base`. These prove the stock frozen-opponent route can
execute with real checkpoint loading. They do not prove learning yet. The next
useful claim needs a curve: survival, sparse outcome, reward components, and
action distribution across checkpoints.

Latest two-seat replay status: the tiny native bridge projects physical ticks
into two seat-local `GameSegment`s, pushes them into `MuZeroGameBuffer`, and
passes deterministic reward/value/policy target assertions inside the
Modal/LightZero runtime. This proves the tiny bridge contract, not the old
custom trainer.

Latest architecture read: the custom two-seat path diverges from stock at the
collector, `GameSegment`, replay buffer, learner, and checkpoint seams. The
intended change was simultaneous action collection, but the scaled path also
changed the training contract.

## Research Threads

- [orchestration_plan.md](orchestration_plan.md): live task split, who is
  checking what, and how findings should be merged.
- [path_matrix.md](path_matrix.md): one table comparing every CurvyTron
  LightZero path.
- [history_timeline.md](history_timeline.md): how the project moved from
  fixed/frozen controls to custom two-seat scaling.
- [architecture_questions.md](architecture_questions.md): open questions about
  collectors, replay, self-play, GPU/CPU split, and observability.
- [open_questions_and_hypotheses.md](open_questions_and_hypotheses.md):
  tracked hypotheses and unresolved questions.
- [known_wrong.md](known_wrong.md): blunt list of confirmed mistakes.
- [cleanup_targets.md](cleanup_targets.md): stale docs/defaults/scripts to fix
  or archive.
- [next_gates.md](next_gates.md): concrete gates before another large run.
- [frozen_recent_opponent_route.md](frozen_recent_opponent_route.md): practical
  route for stock `train_muzero` against recent frozen checkpoints.
- [pong_lessons_for_curvytron.md](pong_lessons_for_curvytron.md): historical
  analogy from custom Pong failures to stock LightZero Pong signal.
- [muzero_training_pitfalls_literature.md](muzero_training_pitfalls_literature.md):
  cited checklist of MuZero/RL pitfalls mapped to our local failure modes.

## Stable Docs

- [postmortem](../postmortems/2026-05-12-curvytron-no-learning.md)
- [stock LightZero loop contract](../../../design/training/lightzero_stock_loop_contract.md)
- [simultaneous replay bridge contract](../../../design/training/simultaneous_replay_bridge_contract.md)
- [CurvyTron learning gates](../../../design/training/curvytron_learning_gates.md)

## Rule For New Claims

A CurvyTron training result is not a learning claim unless the artifact states:

- which trainer path ran;
- whether stock `train_muzero` was called;
- whether LightZero `GameSegment` / `MuZeroGameBuffer` targets were used;
- what "self-play" means in that run;
- how survival and outcome moved across checkpoints;
- what opponent source was used.
