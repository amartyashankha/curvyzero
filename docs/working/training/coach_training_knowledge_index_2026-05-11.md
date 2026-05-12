# Coach Training Knowledge Index - 2026-05-11

Purpose: one short index for coach training findings so the current gate,
replication evidence, failure notes, and optimizer references stay findable.
Keep details in the linked docs.

No pytest was run for this docs-only index.

## Start Here

- 2026-05-12 canonical CurvyTron Coach launcher:
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  with `--mode two-seat-selfplay`. Stock LightZero in-training eval stays off;
  CurvyZero checkpoint eval, inspection, and GIF generation stay on; checkpoint
  cadence defaults to `100` iterations.
- Current read at 2026-05-11 14:08 EDT: stock-like visual Pong has real
  survival learning across multiple rows. `s122` is strongest, but `s114`,
  `s120`, `s121`, `s142`, and exact controls `s113/s123` all show later
  survival gains. The active lesson is to judge by same-run survival curves
  after enough horizon, not by early score-only reads.
- Cleanup at 2026-05-11 14:51 EDT: active Modal Pong jobs were stopped after
  Pong replication passed the basic survival-signal gate. Pong checkpoints and
  eval manifests remain on the Volume. CurvyTron remains active.
- Current CurvyTron correction: stop treating
  `env_variant=source_state_turn_commit` as trainable/default or
  learning-quality current-policy self-play. It is useful stock LightZero
  smoke/profile plumbing only:
  player 0 records a pending action with no physics advance and reward `0`;
  player 1 commits the real joint action and gets survival reward. Stock
  GameSegment storage can let value targets credit player 0 states for player 1
  survival, so this is a reward-credit risk until directly fixed or disproven.
  Keep survival time as the lead metric and label turn-commit claims narrowly.
- Env-wrapper audit correction: safe plumbing fixes are being made to align
  source-state env knobs, metadata, telemetry, stack schema hash, and make
  render non-mutating with respect to the observation stack.
- 2026-05-11 21:08 EDT: `source_state_turn_commit` stock LightZero plumbing
  smoke passed after cleanup. It called `train_muzero`, ran GPU model/search,
  MCTS, replay sample, one learner step, copied `iteration_0`, and wrote
  telemetry with pending and physical-commit rows. This is still plumbing
  evidence only because reward credit remains untrusted.
- 2026-05-11 21:18 EDT: target/replay audit confirmed the reward-credit
  blocker. GameSegments stored fake pending rows as normal scalar transitions
  (`0,1,0,1...`), and sampled value targets backed commit rewards through
  pending rows. `source_state_turn_commit` is now blocked for
  `mode=train`; keep it as profile/smoke only.
- Turing recommendation, candidate/control only until tested: a 9-action
  centralized joint-action wrapper: one scalar action -> `(p0,p1)`, one real
  CurvyTron tick, one reward, `to_play=-1`, `action_space_size=9`. Loud caveat:
  centralized control, not true competitive self-play.
- Reward-design note: shared survival reward is acceptable as a short-term
  diagnostic. Keep sparse outcome and shaped survival logged separately; the
  long-loss-vs-short-win scale issue is recorded for later shaping, not the
  current blocker.
- 2026-05-11 20:10 EDT: two-seat policy-action-repeat/dropout smoke passed on
  Modal with per-seat repeats visible in logs. This is plumbing evidence only,
  not a learning claim.
- [CurvyTron training stochasticity knobs](../environment/training_stochasticity_knobs_2026-05-10.md):
  current Coach decision on self-play path choice, fixed-opponent controls,
  trainer-layer action repeat/dropout, per-seat/per-row schedules, and logging
  requirements.
- [Active board](../training_coach_active_board_2026-05-10.md): current
  decision, live gates, reporting rules, and the canonical two-seat self-play
  entrypoint. Survival steps remain the lead metric.
- [Canonical two-seat handoff](curvytron_canonical_two_seat_handoff_2026-05-12.md):
  current launcher, old-wrapper deletion note, and observability defaults.
- [LightZero Pong replication monitor](../lightzero_pong_replication_monitor_2026-05-11.md):
  live Pong control status, stock64 survival curves, and next eval cadence.
- [Eval speed investigation](eval_speed_investigation_2026-05-11.md):
  why Pong/CurvyTron eval bundles are slow, which paths are parallel vs serial,
  and what timing future evals must emit. Top-line correction: old Pong
  artifacts on Modal L4/T4 showed serious 50-search stock eval around
  `90-104s` for only `833-848` action steps. This measures that eval mode
  only; it does not measure direct policy-head action speed, tiny-search speed,
  batched search speed, H100 speed, or a deployment serving path.
- [CurvyTron checkpoint eval + inspection handoff](curvytron_checkpoint_eval_inspection_handoff_2026-05-11.md):
  background checkpoint poller, eval artifact layout, inspector report layout,
  and proof-run refs for checkpoint-level eval/inspection plumbing.

## Finding Map

- [Pong replication failure audit](pong_replication_failure_audit_2026-05-11.md):
  which Pong lanes should have worked, what signal they showed, and what still
  needs checking.
- [Pong discrepancy action plan](pong_discrepancy_action_plan_2026-05-11.md):
  short action plan for why older stock-ish runs failed or were inconclusive,
  and what must be checked before CurvyTron.
- [Stock64 signal comparison](pong_stock64_signal_comparison_2026-05-11.md):
  short survival-first comparison of installed 0.2.0 stock64 runs such as
  `s114`, `s120`, `s121`, `s122`, and repeats.
- [Archived two-seat purge notes](archive_2026-05-12_two_seat_purge/README.md):
  historical May 10/11 notes that predate the canonical launcher. Use them only
  as context, not as launch guidance.
- [Modal training lifecycle footguns](modal_training_run_lifecycle_footguns_2026-05-11.md)
  (in progress): safe launch, Volume, and checkpoint verification pattern for
  long Modal training jobs.
- [Pong replication follow-up queue](pong_replication_followup_queue_2026-05-11.md)
  (in progress): which old stock/near-stock runs to monitor, relaunch, ignore,
  or archive.
- [Non-LightZero controls](../non_lightzero_control_scout_2026-05-11.md):
  OpenSpiel/Mctx/MiniZero scout status and what each control does or does not
  prove.
- [Optimizer framework reassessment](../optimizer/framework_reassessment_2026-05-11.md):
  current framework worldview for CurvyTron MuZero, simultaneous action
  modeling, and migration criteria.
- [Optimizer replication controls](../optimizer/framework_replication_controls_2026-05-11.md):
  framework examples to reproduce before trusting a migration path.
- [Optimizer to Coach handoff](../optimizer/coach_handoff_2026-05-11.md):
  native CurvyTron LightZero trainer path, setup evidence, boundaries, and
  optimizer-supplied starting knobs.
- [Optimizer CurvyTron native profile](../optimizer/curvytron_native_lightzero_profile_2026-05-11.md):
  source-state visual trainer profile, renderer fix, CUDA/model-device
  evidence, collector-width sweeps, and speed caveats.
- [Optimizer docs home](../optimizer/README.md): entry point for profiling,
  framework, and training-loop optimizer docs.

## Maintenance Rule

Update this index when a finding becomes a gate, a gate is retired, or a
planned path becomes a real document. Do not turn this into another run ledger.
