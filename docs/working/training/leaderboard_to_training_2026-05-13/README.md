# Leaderboard-To-Training Working Memory

Date: 2026-05-13

## Purpose

This directory is the coordination layer for turning CurvyTron tournament
leaderboards into training inputs and for deciding the next overnight runs.

The core problem:

```text
training checkpoints -> tournament/intake -> Elo leaderboard -> opponent assignment -> next training run
```

The system is partly implemented. A tiny manual closed-loop smoke works, but the
production loop is not automated yet. These docs separate what works from what
still needs tests and wiring.

## If You Are New Here

- **Volume**: durable file storage. Checkpoints and JSON snapshots live here.
- **Modal Dict / Queue**: short-lived coordination state. Useful, but not truth.
- **Intake subscriber**: scans for new checkpoints.
- **Intake drain**: consumes checkpoint events and starts rating work.
- **Rating snapshot**: tournament result JSON, usually `latest.json`.
- **Public leaderboard snapshot**: frozen JSON derived from ratings, meant for
  training to consume indirectly.
- **Assignment**: small frozen opponent list that the trainer reads.
- **One-frame**: current trusted training cadence, one source physics frame per
  policy action.
- **Closed loop**: training creates checkpoints; tournament ranks them; a new
  assignment feeds selected checkpoints back into training.

## Read Order

1. `current_state.md`
2. `closed_loop_spec.md`
3. `dataflow.md`
4. `gaps_and_tests.md`
5. `tournament_seeding_and_anchors.md`
6. `non_neural_opponent_contracts.md`
7. `seeded_roster_design.md`
8. `optimizer_speed_axis.md`
9. `overnight_run_decision.md`
10. `launch_readiness_checklists.md`
11. `implementation_log.md`
12. `operator_runbook.md`
13. `caveats.md`
14. `subagent_lanes.md`

## Existing Source Docs

- `docs/working/training/lightzero_train_refactor_2026-05-13/current_source_of_truth.md`
- `docs/working/training/lightzero_train_refactor_2026-05-13/todo.md`
- `docs/working/training/lightzero_train_refactor_2026-05-13/opponent_leaderboard_interface.md`
- `docs/working/training/checkpoint_tournament_orchestration_2026-05-13.md`
- `docs/working/training/checkpoint_tournament_active_threads_2026-05-13.md`
- `docs/working/training/checkpoint_tournament_intake_runbook_2026-05-13.md`
- `docs/working/training/checkpoint_tournament_public_leaderboard_working_memory_2026-05-13.md`
- `docs/working/training/curvytron_architecture_research_2026-05-12/fair_comparison_212_run_investigation_2026-05-13.md`
- `docs/working/optimizer/README.md`
- `docs/working/optimizer/current_plate_map_2026-05-13.md`

## Current Big Read

- Tournament rating artifacts exist on the tournament Volume.
- Modal Dict/Queue intake exists for checkpoint candidate coordination.
- Public leaderboard snapshot publishing works in a remote smoke.
- Trainer consumption of a leaderboard-derived assignment works in a remote
  tiny train smoke.
- Intake V0 is a guarded batch launcher, not a full online Elo service.
- One-frame training cadence is patched for the trusted train lane, but
  tournament evaluator parity for the newest one-frame contract must be checked
  before a new public leaderboard launch.
- Non-neural policies are training-mixture concepts today. Tournament/rating
  remains checkpoint-player-centric until general participant specs are designed
  and tested.
- Optimizer/speed recommendations are throughput constraints, not policy-quality
  conclusions.
- A tiny manual closed-loop smoke worked:
  trainer assignment -> train checkpoints -> discovery/intake -> rating ->
  public leaderboard -> new assignment -> trainer smoke.
- What is still missing is automation: online continuation, scheduled refresh,
  pointer repair, and production runbooks.
- Read `caveats.md` before trusting any launch plan.

## Launch Posture

There are two viable near-term launch postures:

1. **Static manifest now**: use current survival/leaderboard recommendations and
   manually mirror intended frozen-opponent assignments. This is still the safer
   overnight path.
2. **Manual leaderboard-fed smoke path**: works for tiny runs and can be used for
   further testing.
3. **Automated leaderboard-fed launch later**: first implement refresh policy,
   intake continuation, pointer repair, and production safeguards.

The static path is faster. The leaderboard-fed path is the durable product
direction, but it still needs automation before long overnight runs depend on it.
