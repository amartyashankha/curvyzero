# Leaderboard-To-Training Working Memory

Date: 2026-05-13

## Purpose

This directory is the coordination layer for turning CurvyTron tournament
leaderboards into training inputs and for deciding the next overnight runs.

The core problem:

```text
training checkpoints -> tournament/intake -> Elo leaderboard -> opponent assignment -> next training run
```

The system is implemented far enough to prove the deployed loop at canary
scale. A fresh post-cleanup canary has shown a trainer checkpoint entering
intake, being rated, producing a public snapshot and assignment, and then being
picked up by the same running trainer. These docs separate that proven path
from the remaining hardening work needed before larger runs.

## If You Are New Here

- Start with `NOW.md`, `TODO.md`, and `OPERATING_PATTERN.md`. Older dated
  pages below are historical working notes unless they explicitly say they are
  current.
- **Volume**: durable file storage. Checkpoints and JSON snapshots live here.
- **Modal Dict / Queue**: short-lived coordination state. Useful, but not truth.
- **Intake subscriber**: scans for new checkpoints.
- **Intake drain**: consumes checkpoint events and starts rating work.
- **Rating snapshot**: tournament result JSON, usually `latest.json`.
- **Public leaderboard snapshot**: frozen JSON derived from ratings, meant for
  training to consume indirectly.
- **Assignment**: small frozen opponent list that the trainer reads. Assignment
  entries are the slots.
- **Trainer refresh**: the trainer actually starts using a newer assignment
  through a control-volume pointer at a coarse refresh boundary.
- **stable_slots_v1**: Coach-owned materializer that turns one verified
  leaderboard snapshot into `assignment.json` + `audit.json`.
- **One-frame**: current trusted training cadence, one source physics frame per
  policy action.
- **Controlled-player view**: the policy observation is rendered from the
  physical player that the policy row controls. Coach/training chooses that
  player; renderer backends do not randomize it.
- **Closed loop**: training creates checkpoints; tournament ranks them; a new
  assignment feeds selected checkpoints back into training. Today this is
  proven at canary scale for a running trainer, but the production controller
  path still needs hardening before a large batch.
- **Modal launch lifetime**: if a local `modal run` starts child tournament
  workers and then exits, those children may be killed unless the run was
  detached or the parent waited for them.

## Plain Words For Current Work

- **Online tournament continuation**: new checkpoints can be added to an existing
  tournament without starting over. Keep the old scores and old match history,
  then add new games for the new checkpoints.
- **Remote proof**: run the path on Modal, not only in local unit tests.
- **Queue-loss repair**: if a Queue event disappears, the next scan can recreate
  it from the durable manifest.
- **Stale-claim repair**: if one drain says "I am starting the rating job" and
  then dies, a later drain can safely take over without double-counting work.
- **One-frame tournament validation**: prove the tournament games use the same
  one-action timing as the current training line.
- **Assignment command path**: the human or Coach script writes frozen
  `assignment.json` and `audit.json` files from a trusted leaderboard snapshot.
  The trainer reads those files, not live tournament state.
- **Safe refresh**: a long training run only swaps opponents at a clean boundary,
  such as launch, resume, or checkpoint time.

Current product target:

```text
new checkpoint arrives
-> add it to the tournament
-> schedule many games against existing strong checkpoints
-> run those games in parallel
-> update ratings without losing old evidence
-> publish a leaderboard snapshot Coach can later use
```

Concrete near-term shape:

- active pool defaults to the top 100 mature checkpoints;
- new checkpoints should get placement games against many active checkpoints;
- if 10 new checkpoints arrive, their games should be scheduled together, not
  one checkpoint at a time;
- Modal Queue/Dict can wake the system up, but Volume files remain the truth.

Current restart decision:

- The current v2 real18 run is invalid enough to stop. Keep its artifacts as
  smoke/diagnostic evidence, not as a production candidate.
- Randomized learner seat/perspective, no-op/straight action checks, tournament
  eval parity, public slot immortality, and the fresh deployed full-loop canary
  are now test/proof covered. The next phase is broad E2E-adjacent regression,
  live cleanup decisions, and then a deliberately named larger run.
- The next manifest should use `random_per_episode` and globally include at
  least about `20%` blank/immortal pressure, with some higher-immortal variants.
  Do not repeat the previous weak `5%` wall recipes as the main pressure plan.

Modal operating pattern:

- Deployed apps are the durable service shape.
- Volume JSON is truth.
- Dict/Queue coordinate; they do not own history.
- Stop stale detached apps before trusting dashboards or restarting.
- Avoid broad reload-dependent behavior during active file reads.

## Read Order

1. `NOW.md`
2. `TODO.md`
3. `OPERATING_PATTERN.md`
4. `policy_observation_perspective_contract_2026-05-15.md`
5. `contract_drift_audit_2026-05-15.md`
6. `current_state.md`
7. `closed_loop_spec.md`
8. `full_loop_validation_matrix.md`
9. `dataflow.md`
10. `gaps_and_tests.md`
11. `slot_architecture_feedback.md`
12. `run_slot_control_design.md`
13. `run_reward_control_design.md`
14. `coarse_opponent_refresh_design.md`
15. `tournament_seeding_and_anchors.md`
16. `non_neural_opponent_contracts.md`
17. `seeded_roster_design.md`
18. `optimizer_speed_axis.md`
19. `overnight_run_decision.md`
20. `launch_readiness_checklists.md`
21. `implementation_log.md`
22. `operator_runbook.md`
23. `modal_skills_pointers.md`
24. `tournament_control_invariants.md`
25. `caveats.md`
26. `subagent_lanes.md`

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
- A narrow submit surface now exists locally: configure policy with
  `intake-seed`, then submit candidate refs/run IDs with `tournament-submit` /
  `intake-submit`. Submit does not carry scheduler knobs.
- Public leaderboard snapshot publishing works in a remote smoke.
- Public leaderboard publish now fails closed unless the rating snapshot is
  one-frame and has active rows, unless an explicit diagnostic/legacy path is
  used.
- Public leaderboard Dict pointer repair exists, rebuilds from immutable Volume
  snapshots, has passed a tiny remote smoke, and has a minimal operator
  runbook.
- Trainer consumption of a leaderboard-derived assignment works in remote tiny
  train smokes and in the current post-cleanup running-trainer canary.
- Intake V0 is a guarded batch launcher, not a full always-on tournament
  service.
- A small bounded online continuation proof passed on Modal:
  old checkpoints were rated, new checkpoint refs entered through intake,
  missing Queue events were rebuilt from the manifest, and the next rating
  continued from `latest.json`.
- Modal launch lifetime is now a known foot gun: non-detached `modal run` is not
  a safe parent for background tournament game/rating workers. Use
  `modal run --detach` for fire-and-return work, or wait for spawned child work
  to finish before the command exits. Verify `latest.json` advanced and game
  summaries exist; "round scheduled" alone is not success.
- Local tests now protect the checkpoint-id stability needed by that path: old
  refs keep old ids when new refs are added.
- One-frame training cadence is patched for the trusted train lane. A tiny
  two-checkpoint one-frame rating/publish smoke passed; still do a larger
  current-source validation before trusting a new public leaderboard launch.
- Non-neural policies are training-mixture concepts today. Tournament/rating
  remains checkpoint-player-centric until general participant specs are designed
  and tested.
- Optimizer/speed recommendations are throughput constraints, not policy-quality
  conclusions.
- A tiny manual closed-loop smoke worked, including the `stable_slots_v1`
  materializer path:
  trainer assignment -> train checkpoints -> discovery/intake -> rating ->
  public leaderboard -> new assignment -> trainer smoke.
- The first stable-slot smoke exposed a metadata problem: `recent_strong` can
  only be trustworthy if tournament rating rows preserve checkpoint recency
  metadata. Local code/tests now preserve run id, attempt id, iteration, mtime,
  and `latest_for_run` across fresh discovery, explicit refs, intake
  continuation, rating rows, and public leaderboard rows. The all-v2 canary
  proves automatic refresh wiring; production-quality `recent_strong` selection
  still needs real active-row gates before a large launch depends on it.
- New slot-control design direction: a Modal Dict may hold the desired slot
  recipe for each training run id, but the trainer still consumes immutable
  assignment files. See `run_slot_control_design.md`.
- New reward-control design direction: the run-scoped Dict can grow into a
  broader run-control record with an explicit reward recipe. Reward recipes
  should be frozen into launch/attempt artifacts; do not read live reward
  weights inside the learner loop. See `run_reward_control_design.md`.
- Coarse opponent refresh direction: mutable slot intent should live in
  run-control, while the trainer applies concrete assignment changes only at a
  coarse LightZero collection boundary. The deployed canary proves the pointer
  mechanism; the exact production refresh interval remains a launch knob, not a
  hidden fallback.
- What is still missing is hardening around the online path: duplicate rating
  claim cleanup, queue/claim repair at larger scale, inode/storage pressure
  cleanup, and enough survival telemetry to know whether the loop improves
  play.
- Current loop closure is proven only when a run is launched with a
  control-volume refresh pointer. A checkpoint reaching the leaderboard still
  does not magically affect trainers that were not configured for refresh.
- Debugging this by hand with repeated `modal volume get` and `jq` is too
  brittle. Keep a small status-bundle tool in the loop so every investigation
  starts from the same manifest/config/latest/progress summary.
- `slot_rules_v0` is not the production direction. Purge that path from launch
  guidance instead of growing a slot language.
- Read `caveats.md` before trusting any launch plan.

## Launch Posture

Current near-term launch posture:

1. **All-v2 refresh-enabled run**: this is now the durable direction. The
   deployed all-v2 canary proves the feedback wiring at canary scale.
2. **Production-shaped validation before scale**: the next larger run still needs
   manifest audit, active-row quality gates, stale non-v2 input checks, cleanup,
   and survival measurement.
3. **Static/manual assignments**: keep only as a fallback if the refresh contract
   changes or a fresh proof fails.

The leaderboard-fed path is no longer just a manual smoke path. It works at
canary scale, but it is not yet proof of production-quality ranking or learning
improvement.
