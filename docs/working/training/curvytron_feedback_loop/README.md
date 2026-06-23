# CurvyTron Feedback Loop

Created: 2026-05-16

This is the general source of truth for the CurvyTron trainer/checkpoint/
tournament/export/trainer feedback loop. Batch-specific folders such as
`r18fresh_postmortem_2026-05-16` are case studies and evidence, not the general
architecture contract.

## Current Contracts

- `CURRENT_RESEARCH_PHASE.md`: current scheduler research hub, active lanes,
  open questions, and gates.
- `ORCHESTRATION.md`: how the main thread and subagents are being coordinated.
- `SCHEDULER_SIMULATION_PLAN.md`: toy and high-fidelity simulation plan for
  bounded online tournament scheduling.
- `CANDIDATE_SCHEDULERS.md`: current design synthesis and candidate V1
  scheduler shape.
- `TOURNAMENT_ARCHITECTURE_CRITIQUE.md`: high-level critique of scheduler,
  worker fanout, warm pools, Volume overhead, and reducer boundaries.
- `EXPERIMENT_LOG.md`: append-only notes from local scheduler experiments.
- `TOURNAMENT_REFACTOR_BRIDGE.md`: how to clean up tournament code without
  mixing cleanup with behavior changes.
- `POLICY_OBSERVATION_CONTRACT.md`: what a policy sees during training and
  tournament evaluation.
- `POLICY_PERSPECTIVE_UNIFORMITY_INVESTIGATION.md`: active audit of whether
  player 0 and player 1 policy observations are the same semantic language.
- `OBSERVABILITY_CONTRACT.md`: what artifacts must exist to prove the loop.
- `BATCH_CONSTRUCTION_INVESTIGATION.md`: how launch rows, opponent slot
  mixtures, and learner minibatches are actually constructed. Current contract:
  recipes are integer slot-count bags, not percentages; the real broad lane
  materializes them over 256 collector envs while learner `batch_size=64`
  remains replay sampling.
- `TRAINING_CODE_INVENTORY.md`: plain inventory of the current LightZero
  training code shape, bloated files, and cleanup boundaries.
- `TRAINING_LOOP_MODULARITY_PLAN.md`: focused plan for making training-loop
  algorithm changes, such as exploration bonuses or side networks, modular
  without refactoring the whole system.
- `TASK_BOARD.md`: current general implementation lanes.

## North Star

The system should be able to prove, from durable artifacts:

`checkpoint written -> intake accepted -> tournament played/rated -> export
written -> trainer assignment applied -> provider loaded exact opponent ->
learning metrics moved`

Dashboards may explain this chain, but Volume artifacts, refs, hashes, and
assignment SHAs are the truth.

## Current Position

Last reoriented: 2026-05-16 18:42 EDT.

Current added focus: tournament scheduling must become a bounded online service.
All-pairs remains useful for small audits, but it is not the default path for
many checkpoints. The active research question is how to introduce many new
checkpoints, give them useful games quickly, protect strong new policies from
early false drops, and keep the trainer-facing top pool trustworthy.

The latest local work closed the policy-observation proof for the active gray64
model tensor: training and tournament evaluation use controlled-player
`self`/`other` semantics on a global board, tournament checkpoint loading fails
closed on incompatible or contradictory observation metadata, and trainer
frozen-opponent provider loading now enforces the same checkpoint observation
metadata contract. Trainer step telemetry now also records the selected learner
seat and controlled-player observation perspective so future run artifacts do
not require reconstructing env internals to answer "which player did this policy
see/control?".

The required lineage stage names now have current-code call sites, including
rating spawn claims and rating latest writes/skips. Training-candidate refresh
now refuses to materialize a leaderboard checkpoint into a trainer assignment
unless that checkpoint has explicit policy-observation metadata, and it writes a
clean metadata sidecar beside the copied control-volume checkpoint.

A focused local lifecycle proof now exists. It writes two synthetic LightZero
checkpoint files with policy-observation sidecars, sends them through the real
intake seed/submit/drain functions, records the rating spawn claim, runs the
actual rating-round reducer locally with a fast fake game mapper, publishes a
leaderboard snapshot, refreshes the trainer assignment from that leaderboard,
resolves the assignment through trainer code, and records trainer load/apply
lineage with the same assignment SHA. This is stronger than the previous
assignment-only proof, but it is still not a deployed Modal canary.

The immediate proof chain is now:

1. keep the local lifecycle proof green while code changes;
2. deploy the cleaned tournament app so scheduled current-lane refresh uses
   the same constants that passed local tests;
3. run a deployed current-code canary on v2 volumes;
4. verify the same artifact chain in deployed Volume/Dict state;
5. only then use a larger run to study survival improvement and ranking quality.

Do not reuse old live proof as proof of the current code after contract changes.
Old proof is a template for what to prove again. The current checked-in default
lane must also match actual Volume state before deployment; on 2026-05-16 the
local constants briefly pointed at `slot64-*` refresh pointers that did not
exist in the live r18fresh control volume. That is now fixed back to the three
actual live r18fresh pointers:

- `blank10-wall10-rank2_25-rank1_55`
- `blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5`
- `blank20-wall5-rank1_70-rank1imm5`

## Current Live Canary

Active canary ids:

- trainer run: `curvy-e2e-current-contract-20260516a`
- trainer attempt: `try-e2e-current-contract-20260516a`
- tournament: `curvy-e2e-current-contract-live-20260516a`
- rating: `elo-e2e-current-contract-live-20260516a`
- runs volume: `curvyzero-runs-v2`
- tournament volume: `curvyzero-curvytron-tournaments-v2`
- control volume: `curvyzero-curvytron-control-v2`

Current status:

- local focused proof is green: `272 passed, 11 skipped` across intake repair,
  tournament, online simulation, checkpoint opponent provider, opponent mixture,
  launch manifest, feedback-loop lineage, shared contracts, and training
  candidate controller tests;
- rating-round recovery from existing game summary files is fixed. If progress
  was counted from game summaries, the reducer now reduces those summaries
  directly instead of incorrectly trying to rebuild from missing shard tallies;
- live r18fresh current lane has `585` rating rows, `100` active rows, max
  checkpoint iteration `310893`, and rating latest at `round-000062`;
- trainer-facing training-candidate export manually validated with the same
  function used by the scheduled tick: generation `38`, snapshot
  `auto-r000062-g38-3f471334`, three pointer rewrites, active count `100`;
- the current code now names the actual live refresh pointer paths. A regression
  test locks the status contract to those paths;
- tournament app was redeployed after fixing a false submit-validation blocker
  where `intake-submit` treated current policy render defaults as non-default
  scheduler overrides;
- the canary trainer produced `iteration_0.pth.tar` and
  `iteration_5.pth.tar` with sidecars;
- `iteration_5` was found by latest-checkpoint intake seed;
- `iteration_0` was explicitly submitted and accepted;
- intake drain spawned rating call `fc-01KRS7G7RFQSEECDV4FXFRVY6R`;
- rating artifacts advanced at least through `round-000040`; `latest.json`
  reports `stable=true`, `checkpoint_count=2`, `pair_count=1`,
  `game_count=3`, and both canary checkpoints active;
- the trainer/public leaderboard file
  `e2e-current-contract-leaderboard-20260516a/latest.json` is stale: it is
  still generation `0`, sourced from `round-000000`, while rating latest had
  advanced to round `40`;
- the trainer's current control pointer still points at the initial blank/rank1
  starter assignment:
  `e2e-current-contract-initial-blank-20260516a`;
- tournament lineage shows a training-candidate assignment write and pointer
  rewrite, but those writes targeted the separate
  `e2e-rankslot-proof-control-20260516a` pointer, not the trainer pointer for
  `curvy-e2e-current-contract-20260516a`;
- therefore the older deployed Modal canary proves checkpoint creation, intake,
  rating rounds, and rating latest publication, but does not yet prove the full
  loop back into the same running trainer under the latest code.

Current immediate gap: after deploying the cleaned tournament app, rerun the
deployed proof with the tournament training-candidate refresh pointed at the
exact refresh pointer consumed by the canary trainer, then verify trainer
lineage contains a post-rating `trainer_assignment_loaded`/assignment
consumption event whose SHA matches the new assignment written from the
tournament leaderboard.

## Current Critique

The main failure mode in this work has been letting one bug or dashboard symptom
turn into a broad, manual monitoring loop. The corrective pattern is:

- write the plain contract first;
- add a focused local regression for each bug;
- prove the full chain with artifacts, not browser impressions;
- keep the task board current so side lanes do not vanish.
