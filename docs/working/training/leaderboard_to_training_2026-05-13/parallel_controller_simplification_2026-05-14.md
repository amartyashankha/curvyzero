# Parallel Controller Simplification - 2026-05-14

## Plain Goal

The system should feel boring:

```text
trainer checkpoints
-> intake sees exact refs
-> tournament rates them
-> public leaderboard snapshot is written
-> Coach materializes one immutable assignment
-> trainer consumes that assignment
```

Do not call the loop closed until the last line is proven by trainer telemetry.

## Simplest Robust Shape

Use one small Coach-side controller around the existing pieces:

1. Read the latest final tournament artifacts from the Volume.
2. Refuse to publish if the round artifacts disagree, have failed games, use the
   wrong one-frame settings, or have no active rows.
3. Publish one immutable public leaderboard snapshot.
4. Materialize one immutable `stable_slots_v1` assignment from that snapshot.
5. Write the assignment and audit to the training Volume.
6. Launch a tiny trainer smoke that consumes that assignment.
7. Launch or refresh real trainers only after the smoke proves the assignment
   ref/hash appears in trainer telemetry with loaded opponents.

Volume JSON is truth. Modal Dict and Queue are pointers and wakeups.

## Parallel Rule

If the intended path is:

```text
1 -> 2 -> 3 -> 4 -> 5
```

run safe versions of `1`, `2`, `3`, `4`, and `5` in parallel when possible.

This is allowed even when later results may be thrown away. The win is speed:
if all gates pass, we are already ahead; if gate 3 fails, discard 4/5 and keep
the evidence that helped find the failure.

## What To Run In Parallel

| Lane | Why it can run now | How to treat the result |
| --- | --- | --- |
| Main live tournament progress | It is already rating trainer checkpoints. | Promote only after final artifact checks. |
| Intake status | It is read-only and catches discovery/queue drift early. | Use as a health signal, not rating proof. |
| Trainer run status and curves | It is read-only and tells us if learning is alive. | Survival trend is separate from loop closure. |
| Assignment materialization from last clean snapshot | Cheap and deterministic. | Discard if a newer publish gate fails. |
| Tiny trainer assignment smoke | Cheap way to prove consumption. | Valid for contract proof, not policy strength. |
| Fallback small arena | Exercises the same interfaces faster. | Label as fallback; do not oversell. |

## Promotion Gates

Before promoting a result, check:

- rating `input`, `progress`, `results`, `ratings`, and `latest` agree;
- one-frame settings are present;
- failed games are zero or explicitly accepted;
- active rows are nonzero and not provisional-only;
- public snapshot ref/hash is immutable;
- assignment audit points to that exact snapshot ref/hash;
- trainer telemetry shows the exact assignment ref/hash and successful opponent
  provider loads;
- if claiming in-run refresh, refresh events show `decision=applied` before the
  new assignment appears in `env_steps.jsonl`.

## Current Critical Gap

The live training/tournament activity is not automatically proof of feedback.
The key question is:

```text
Did a newly rated checkpoint come back into a trainer as an opponent?
```

If not, the next action is not more staring at the tournament. The next action
is to publish from the clean rating, materialize an assignment, and prove a
trainer consumes it.

## Current Proof Lane - 2026-05-14 17:27 EDT

Use the clean completed round as the truth source:

```text
tournament: curvy-loop18-live-mixed-20260514g
rating: elo-loop18-live-mixed-20260514g
round: round-000000
leaderboard snapshot: loop18-mixed-r0-20260514a
snapshot sha256: b36c52d628042be19ec7ad71472f82dc11508eccf7e6b273d26fbca74e78ec5d
assignment id: loop18-mixed-r0-assignment-20260514a
assignment ref: training/lightzero-curvytron-visual-survival/curvy-loop18-live-assignment-r0-20260514a/attempts/try-loop18-live-assignment-r0-20260514a/opponents/assignments/loop18-mixed-r0-assignment-20260514a/assignment.json
assignment sha256: 8a8afdd07b0d0012b5d38a88ae32a6806ce1b50994203e3d40f23acf9dfcfbf0
trainer smoke: loop18-mixed-r0-assignment-consume-smoke-20260514a
```

Round 1 is not the promotion source right now. It admitted many newer
checkpoints and expanded into a very large all-pairs backlog. Treat it as scale
evidence and cleanup work, not as the immediate contract proof.

The immediate gate is narrow: the smoke must show the exact assignment ref/hash,
the rank-1 initial checkpoint load, and successful frozen-opponent provider
loads in trainer artifacts.

Result at 17:35 EDT: the gate passed.

- Trainer smoke completed with `ok=true`, `called_train_muzero=true`, one learner
  train call, and 275 env telemetry rows.
- Same-run auto-resume found nothing, so champion bootstrap was not silently
  skipped.
- The rank-1 checkpoint loaded before `before_run` with
  `loaded=true`, `loaded_module_count=1`, `meaningful_model_load=true`, and
  `fresh_optimizer_preserved=true`.
- Trainer command used the exact assignment ref above.
- Env telemetry rows carried assignment sha256
  `8a8afdd07b0d0012b5d38a88ae32a6806ce1b50994203e3d40f23acf9dfcfbf0`.
- Env telemetry rows used both frozen checkpoint slots from the assignment and
  reported `opponent_provider_load_ok=true`.

This proves the contract path through a manual/controller-like promotion step:

```text
clean rating -> public snapshot -> stable_slots_v1 assignment -> assignment write
-> fresh trainer starts from rank-1 checkpoint and uses assignment opponents
```

It does not yet prove the production controller is automatic, and it does not
prove an already-running trainer refreshes from a public-leaderboard-derived
assignment.
