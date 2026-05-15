# Full Loop Validation Matrix

## 2026-05-15 Current Result

This page contains older validation history below. The current top-line result
is newer and should be read first:

- Fresh post-cleanup canary
  `curvy-e2e-clean-canary-long-20260515c` closed the deployed loop at canary
  scale.
- Concrete chain: trainer wrote checkpoints -> live run-id intake admitted them
  -> tournament `curvy-e2e-clean-canary-long-live-20260515c` rated them ->
  promotion materialized assignment sha
  `58e7e60cb39fe7b7777626458af089abeec228637c65776035570f7e5441fc30` ->
  the control-volume pointer was repaired -> the same running trainer applied
  that sha at train iter `2750` -> later env rows used it with provider load
  OK.
- The real bug found by this proof was pointer upload path handling:
  `control:` was accidentally included in the destination path for
  `modal volume put`. `scripts/promote_curvytron_rating_round.py` is patched and
  regression-tested.
- Remaining hardening work is not “prove the loop exists”; it is duplicate
  rating claim cleanup, storage/inode cleanup, broad regression, and survival
  measurement before a larger launch.

Date: 2026-05-14

## Plain Goal

Separate what we can prove now from the one missing external link.

The full desired loop is:

```text
trainers write checkpoints
-> intake discovers them
-> tournament rates them
-> public leaderboard is published
-> Coach materializes a new assignment
-> trainer uses that assignment
```

The last step has two meanings:

- launch/resume/new attempt uses the assignment;
- an already-running trainer refreshes to the new assignment.

Launch/new-attempt assignment consumption works. A direct running-trainer
refresh hook also exists and has a tiny Modal A -> B proof. On 2026-05-14,
`refresh-direct-canary-20260514a` remotely proved the trainer can apply a new
immutable assignment at a refresh boundary and then use that assignment in env
telemetry. The production public-leaderboard/controller path is still not
integrated, and the active 18-row batches did not enable refresh.

There is also a separate start-weight contract:

- old/static batches started from fresh model weights unless same-run
  auto-resume applied;
- the next production-shaped batch must start every fresh learner from the
  trusted rank-1 tournament checkpoint using a model-only champion bootstrap;
- that is not the same thing as using the champion as an opponent.

## Current Validation Status

| Link | Can test now? | Current status | Next proof |
| --- | --- | --- | --- |
| Trainer writes checkpoints | Yes | Proven in live smokes; 159 checkpoint files were produced. | Keep a small multi-trainer smoke with frequent checkpoints. |
| Intake discovers checkpoints | Yes | Works for broad run-id scans; live smoke saw 159 checkpoints. | Verify every new checkpoint enters the durable manifest once. |
| Drain starts rating from manifest | Yes | Live smoke exposed a bug: rating used 9 players while manifest later saw 159. Local regression now forces live run watches into continuation mode and prevents old partial claims from blocking the larger pool. | Prove the fix remotely on a live watch. |
| Tournament runs games | Yes | Works for small pools; 9-player and other tiny one-frame smokes ran games. | Run a bounded seeded pool and confirm games start quickly in parallel. |
| Leaderboard publish | Yes | Remote smokes passed; one-frame gating exists. | Publish only after rating covers the intended pool. |
| Assignment materialization | Yes | Pure/local tests and tiny manual smoke passed. | Materialize from the new public leaderboard and inspect slot choices. |
| Trainer launch consumes assignment | Yes | Remote tiny train smokes consumed assignment refs. | Launch a new/resumed trainer from the generated assignment. |
| Running trainer refreshes assignment | Narrowly | Direct assignment-ref refresh hook has local tests and `refresh-direct-canary-20260514a` remotely applied assignment B after starting on assignment A. Static live batches did not enable it, and the public leaderboard/controller handoff is still not proven. | Prove controller-owned next attempt or pointer-backed refresh against a fresh post-promotion assignment from the live arena. |
| Fresh trainer starts from tournament winner | Narrowly proven | Same-run auto-resume guard exists; local champion-bootstrap guardrails exist; remote smoke `champion-bootstrap-smoke-20260514f` loaded rank-1 checkpoint weights into `model` and `target_model`, preserved fresh optimizer, then collected and trained. | Integrate this into a production-shaped launch bundle and run a 1-2 row canary from that bundle. |

## Current Loop18c3 Live Lane - 2026-05-14 14:17 EDT

This is the active proof lane:

```text
training matrix: curvy-loop18-clean3-20260514e
training prefix: curvy-loop18c3
tournament: curvy-loop18-live-mixed-20260514g
rating: elo-loop18-live-mixed-20260514g
```

| Link | Current evidence | Status |
| --- | --- | --- |
| Trainers write checkpoints | All 18 trainers are alive and have checkpoint refs. Several latest refs are already `iteration_30000`. | Passing |
| Intake discovers checkpoints | Intake finds all 18 exact current latest refs with no missing run IDs. Manifest `seen_checkpoint_count=67`. | Passing |
| Intake queues new work | Queue length is 51 while the active rating writer is still busy. | Passing, waiting for continuation |
| Tournament runs games | Round 0 is 595 pairs / 12495 games. At 14:17 EDT, lightweight progress saw about 12054 games and 0 failures. | Passing, not final |
| Final rating snapshot | No final trusted `latest.json` for this live mixed round yet. | Waiting |
| Public leaderboard | Not published from this live mixed round yet. | Waiting |
| Fresh assignment | Not materialized from this live mixed round yet. | Waiting |
| Trainer consumes fresh assignment | Not run yet. | Waiting |

Do not call the full loop validated until the final trainer-consumption smoke
passes from a fresh assignment built out of this live mixed leaderboard.

Also do not call this lane a champion-start proof. The launched manifest did not
start learners from the tournament winner.

Update at 14:24 EDT:

- Round 0 finished cleanly with 35 active rows, 595 pairs, 12495 games, and 0
  failures.
- A continuation claim already existed, and `round-000001` is now running.
- Round 1 has admitted the queued newer checkpoints into the existing rating
  run: roughly 82 players, 3321 pairs, 69741 games, first poll showed 9 started
  pairs and 0 failures.
- The loop is still not complete until round 1 publishes, materializes, and a
  trainer consumes the resulting assignment.

Update at 14:28 EDT:

- Round 1 advanced to 71 started pairs and about 1491 seen games.
- Failures remain 0.
- Intake queue is down to 3 events, with 73 seen checkpoints and the 18 current
  latest loop18c3 refs present.
- All 18 trainers remain alive, with latest checkpoints between iteration
  20000 and 40000.
- Still not validated end to end: wait for final round-1 artifacts, publish,
  materialize, write assignment, then run the tiny trainer smoke.
- Command audit correction: use `--round-index 1` for progress, fetch the
  round-1 input/progress/results/ratings files before publish, materialize from
  the immutable public snapshot, and enforce zero failures in artifact review
  before publishing.

Update at 14:42 EDT:

- Round 1 advanced to 330 started pairs and about 6930 seen games.
- Estimated completion is about 10%.
- Failures remain 0.
- Intake sees all 18 current latest run refs; queue is 19 because trainers are
  still producing checkpoints while the active rating writer is busy.
- Trainers remain alive with latest checkpoints between 20000 and 50000.

Update at 17:27 EDT:

- The original 18 loop18c3 trainers are still running and producing checkpoints.
- Eval survival is mixed, not a clean monotonic win: 13/18 loop18c3 rows have
  latest eval mean survival above their first eval, and 18/18 have some best
  checkpoint above their first eval, but many latest values have regressed from
  their own best.
- Round 0 finished cleanly and is now the promotion source for the contract
  proof: 35 active rows, 595 pairs, 12495 games, zero failed games, one-frame
  settings.
- Round 1 admitted many newer checkpoints and expanded into a huge all-pairs
  backlog. Do not wait on it for the immediate proof; use it as scale/backlog
  evidence.
- Public snapshot `loop18-mixed-r0-20260514a` was published from round 0 with
  sha256 `b36c52d628042be19ec7ad71472f82dc11508eccf7e6b273d26fbca74e78ec5d`.
- Assignment `loop18-mixed-r0-assignment-20260514a` was materialized and written
  to the training Volume with assignment sha256
  `8a8afdd07b0d0012b5d38a88ae32a6806ce1b50994203e3d40f23acf9dfcfbf0`.
- Tiny trainer smoke `loop18-mixed-r0-assignment-consume-smoke-20260514a` is
  running with that assignment ref and the rank-1 checkpoint as
  `initial_policy_checkpoint_ref`.

Update at 17:35 EDT:

- The tiny trainer smoke completed and proves the launch/new-attempt feedback
  contract for this generation.
- It loaded the rank-1 checkpoint as model-only start weights:
  `loaded=true`, `loaded_module_count=1`, `meaningful_model_load=true`,
  `fresh_optimizer_preserved=true`, and same-run auto-resume `found=false`.
- It consumed assignment sha256
  `8a8afdd07b0d0012b5d38a88ae32a6806ce1b50994203e3d40f23acf9dfcfbf0`.
- It wrote 275 env telemetry rows; rows show the exact assignment sha, frozen
  checkpoint opponent refs from the assignment, and
  `opponent_provider_load_ok=true`.
- This is a real contract proof, not an in-run refresh proof. The manual
  promotion/controller step still needs to become one repeatable automation
  command or service.

Update at 17:54 EDT:

- The repeatable controller path now passed on loop18c3 round 0:
  `scripts/promote_curvytron_rating_round.py`.
- The controller verified round artifacts, published snapshot
  `loop18-controller-r0-20260514b`, materialized/wrote assignment
  `loop18-controller-r0-assignment-20260514b`, and launched
  `loop18-controller-smoke-20260514b`.
- Smoke evidence: `mode=train`, `ok=true`, `called_train_muzero=true`,
  232 env steps, 1 learner train call, 335 env telemetry rows,
  champion checkpoint load `loaded=true`, `meaningful_model_load=true`,
  `fresh_optimizer_preserved=true`, auto-resume `found=false`, and all 335
  env rows showed the assignment sha with `opponent_provider_load_ok=true`.
- The first controller replay found a real false-positive risk: dry-mode smoke
  could have been mislabeled as success. The controller now passes `--mode train`
  and verifies both `summary.json` and `env_steps.jsonl`.

## What We Should Test Next

Do a bounded proof that does not depend on live trainer refresh:

1. Start two or three tiny trainers with frequent checkpoints.
2. Start the tournament watch at the same time.
3. Confirm intake manifest count matches produced checkpoint count.
4. Confirm rating config/latest count matches the intended pool count.
5. Confirm game workers start and complete.
6. Publish a public leaderboard snapshot.
7. Materialize `stable_slots_v1` from that snapshot.
8. Launch a fresh tiny trainer using the materialized assignment.

This proves every implemented link. It does not prove automatic in-run refresh.
For a production-shaped batch, the fresh trainer must also start from
`initial_policy_checkpoint_ref=<rank1 active checkpoint ref>`, and opponents
must come from an immutable `stable_slots_v1` assignment. Without both, it is
only a diagnostic assignment-consumption proof.

Launch rule for step 5: if a `modal run` command spawns background game/rating
workers and then returns, it must be detached or must wait for those workers.
Do not accept "round scheduled" as proof. Check that `latest.json` advanced and
completed game summaries exist.

## Current Hard Failures

1. The live smoke produced 159 checkpoint files, but the rating run only used 9
   players. This was a tournament continuation/control-plane failure before game
   scheduling. A local fix/test now exists; remote proof is still pending.
2. Leaderboard promotion cannot affect a running trainer until the external
   refresh lane lands.
3. The only visible source with at least 100 active checkpoint players is the old
   `curvytron-latest212-smoke-20260513` leaderboard. The current one-frame
   leaderboard has 33 rows, so it cannot seed a top-100 tournament by itself.

## Do Not Claim

- Do not claim the loop is fully closed until a trainer actually consumes a new
  assignment after leaderboard promotion.
- Do not claim a 9-player rating covers a 159-player manifest.
- Do not claim a tournament round succeeded only because progress/input files
  were written. Empty game dirs and missing summaries mean the workers did not
  finish.
- Do not use the old latest-212 leaderboard as one-frame truth. It can seed a new
  one-frame tournament, but the new tournament must rerate those checkpoints
   under current settings.

## Active Stress-Test Lane

Run a separate top-100 tournament stress test:

1. Seed from the top 100 active checkpoint refs in
   `curvytron-latest212-smoke-20260513`.
2. Rerate them under current one-frame settings.
3. Only after that tournament is visibly healthy, add one recent checkpoint from
   about 100 current trainer runs as a second intake wave.

This tests tournament scale and intake continuation. It still does not prove
running trainer refresh.

## Champ18a Live Proof Gate

The `curvy-champ18a` batch is the current champion-started live lane. It is not
the full refresh proof by itself.

Required evidence chain:

1. A `curvy-champ18a` trainer writes an exact immutable checkpoint ref.
2. The subscriber/intake manifest contains that exact ref and the intended run
   id.
3. The tournament round input contains that exact ref.
4. Completed game summaries include that checkpoint, with zero unexplained
   failures.
5. The final rating snapshot includes the checkpoint in the active roster.
6. The public leaderboard immutable snapshot includes the same checkpoint.
7. A new immutable assignment is materialized from that public snapshot and its
   audit points at the immutable snapshot ref/hash.
8. A refresh-enabled trainer applies that assignment through
   `opponent_assignment_refresh_events.jsonl`.
9. Post-refresh `env_steps.jsonl` rows report the new assignment ref/hash and
   successful frozen-opponent provider loads.
10. Survival/eval evidence improves under the same eval settings.

Current status:

- Small `champ18a` proof now closes the implemented handoff at diagnostic
  scale:
  trainer checkpoint -> intake/tournament -> rating -> public leaderboard ->
  `stable_slots_v1` assignment -> assignment writer -> tiny trainer consumes the
  assignment. The trainer also model-only bootstrapped from the small snapshot's
  rank-1 checkpoint.
- Direct trainer refresh mechanics are proven by `refresh-direct-canary-20260514a`.
- The small proof is only 8 checkpoints, so it is not final strength evidence
  and does not cover the larger 18-run / 28-checkpoint pool.
- Full running-trainer refresh from a public-leaderboard-derived assignment is
  still missing because the launched 18 rows use static assignment refs. Pointer
  upload is still blocked by Modal Volume layer pressure.
- The next proof should let `champ18a` round 2 finish, publish that larger
  final snapshot, materialize/write a fresh assignment, and repeat the trainer
  consumption smoke. Pointer-backed refresh should be retried when Volume layer
  pressure clears.
