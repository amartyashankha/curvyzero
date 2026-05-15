# Promotion Controller Plan - 2026-05-14

## Goal

Replace the manual operator glue with one boring Coach-owned command.

The command should take a trusted tournament/rating round and produce a trainer
ready assignment bundle:

```text
rating artifacts
-> public leaderboard snapshot
-> stable_slots_v1 assignment + audit
-> assignment written to training Volume
-> optional tiny trainer smoke
-> durable decision log
```

The controller should not be inside `train_muzero`. The trainer still consumes
only immutable assignment refs and optional model-only initial checkpoint refs.

## Minimal Interface

Inputs:

- `tournament_id`
- `rating_run_id`
- `round_id` or `round_index`
- `leaderboard_id`
- `snapshot_id`
- `assignment_id`
- `assignment_bank_run_id`
- `assignment_bank_attempt_id`
- `run_smoke`: yes/no

Outputs:

- public leaderboard snapshot ref/hash;
- assignment ref/hash;
- assignment audit ref;
- champion checkpoint ref;
- optional smoke run id/attempt id;
- one `promotion_decision.json` that records every input, output, and gate.

## Required Gates

Before publishing:

- round input/progress/results/ratings/latest all name the same round;
- one-frame tournament settings are present;
- completed pair/game counts match the intended round;
- failed game count is zero unless explicitly overridden with a diagnostic flag;
- active row count is nonzero;
- no provisional-only publish;
- source rating snapshot hash is recorded.

Before writing an assignment:

- materializer reads the immutable public snapshot, not the mutable latest
  pointer;
- assignment audit records the exact snapshot ref/hash;
- selected checkpoint refs are immutable `iteration_*.pth.tar` refs.

Before marking the promotion usable:

- if smoke is enabled, the trainer must show:
  - exact assignment ref/hash in command or env telemetry;
  - rank-1 `initial_policy_checkpoint_ref` loaded with model-only mode;
  - same-run auto-resume did not skip the bootstrap;
  - `opponent_provider_load_ok=true` in env telemetry.

## Current Manual Proof

Loop18c3 round 0 has passed these gates manually:

```text
tournament: curvy-loop18-live-mixed-20260514g
rating: elo-loop18-live-mixed-20260514g
snapshot: loop18-mixed-r0-20260514a
snapshot sha256: b36c52d628042be19ec7ad71472f82dc11508eccf7e6b273d26fbca74e78ec5d
assignment: loop18-mixed-r0-assignment-20260514a
assignment sha256: 8a8afdd07b0d0012b5d38a88ae32a6806ce1b50994203e3d40f23acf9dfcfbf0
smoke: loop18-mixed-r0-assignment-consume-smoke-20260514a
```

The smoke proved trainer consumption and champion bootstrap. It did not prove
automatic controller operation or public-leaderboard-derived in-run refresh.

## Simplest Implementation Shape

Start with a local script, not a new service:

```text
scripts/promote_curvytron_rating_round.py
```

It is a replayable promotion transaction. It should mostly orchestrate existing
tools:

1. Fetch or read rating artifacts.
2. Validate gates locally.
3. Call the existing tournament `leaderboard-publish` mode.
4. Fetch the immutable public snapshot.
5. Reuse `scripts/materialize_curvytron_leaderboard_assignment.py` logic.
6. Call the existing trainer `write-assignment` mode.
7. Optionally launch the bounded trainer smoke.
8. Write `promotion_decision.json` locally, and later optionally to Volume.

Keep this script small. Do not add a scheduler yet. Once the command is reliable,
it can become the body of a scheduled/controller function.

Implementation status:

- `scripts/promote_curvytron_rating_round.py` now exists.
- `curvytron_opponent_leaderboard_publish` accepts expected source guards:
  round id, round index, rating context hash, roster hash, and rating snapshot
  sha256.
- Focused local tests cover the stale-source guard and controller command
  construction.
- Remote controller replay `20260514b` against loop18c3 round 0 passed.

Controller replay evidence:

```text
source tournament: curvy-loop18-live-mixed-20260514g
source rating: elo-loop18-live-mixed-20260514g
source round: round-000000
rating snapshot sha256: c90fdc8e222ae0b5fad04ac8de41f615f94d86ad883a9f883e02b529d9109561
leaderboard: loop18-controller-replay-20260514b
snapshot: loop18-controller-r0-20260514b
snapshot sha256: e077373d4179e288f8e2e03159f400997c6329fc1f79ac913212f52d41182752
assignment: loop18-controller-r0-assignment-20260514b
assignment sha256: 8ed54fe941f7d59a4b6615081e913e8a17d73e5caef4ad73011314622c594d8e
assignment ref: training/lightzero-curvytron-visual-survival/loop18-controller-assignment-bank-20260514b/attempts/try-loop18-controller-assignment-bank-20260514b/opponents/assignments/loop18-controller-r0-assignment-20260514b/assignment.json
smoke: loop18-controller-smoke-20260514b
```

The smoke was a real train run, not dry mode:

```text
ok=true
mode=train
called_train_muzero=true
env_steps_collected=232
learner_train_calls=1
summary initial checkpoint loaded=true
meaningful_model_load=true
fresh_optimizer_preserved=true
auto_resume.found=false
env telemetry rows=335
opponent_provider_load_ok rows with assignment sha=335
```

The first controller replay `20260514a` exposed a bug: the smoke command omitted
`--mode train`, so it completed in dry mode. The controller now passes
`--mode train`, parses the real compact summary instead of nested command JSON,
and verifies `summary.json` plus `env_steps.jsonl` when `--run-smoke` is used.

## Red-Team Rules

The controller treats mutable refs as convenience only:

- it may fetch `latest.json`, but it records the exact sha and passes that sha
  back to the publisher;
- if `latest.json` changes between preflight and publish, the publish must
  refuse;
- the assignment must be materialized from an immutable public snapshot ref,
  not from a live pointer;
- a promotion is usable only if the written assignment sha matches the local
  assignment sha;
- champion start weights and opponent assignment are two separate fields in the
  decision log.

## Next Honest Experiment

Run the controller against one clean completed round and compare its outputs to
the already proven manual loop18c3 round-0 artifacts.

If that passes, run it again on the next clean current-checkpoint round where the
assignment includes checkpoints produced after the training batch started. That
is stronger evidence that trainer output came back into a trainer as an
opponent.

Run these lanes in parallel:

1. Controller replay of the known-good loop18c3 round 0.
2. Tiny fresh closed-loop canary where a new trainer checkpoint flows into a
   tiny tournament, then back into a fresh trainer assignment.
3. Passive observation of the larger live 18-run lanes. These can provide extra
   evidence but must not block the small honest proof.
