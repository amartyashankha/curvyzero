# CurvyTron No-Learning Postmortem - 2026-05-12

Purpose: short human-readable postmortem. Detailed artifacts remain in the
working audit and research docs.

## What Happened

We launched large CurvyTron runs through `--mode two-seat-selfplay`. They saved
artifacts, changed weights, and produced GIF/eval outputs, but survival curves
did not show useful learning.

The important discovery is that this path was not the same trusted path that
made Pong replication credible. It used LightZero policy/search and
`learn_mode.forward`, but it did not call stock `train_muzero` and did not use
LightZero's collector, `GameSegment`, or `MuZeroGameBuffer` target builder.

## Why We Moved There

The move was not random. CurvyTron needs both players' actions before one
physical tick. Stock LightZero envs normally receive one scalar action per env
step.

Fixed/frozen opponent wrappers kept stock LightZero, but did not provide live
same-policy two-seat self-play. Turn-commit tried to fake two scalar turns
inside stock LightZero, but replay saw fake pending rows. The custom two-seat
path solved action collection by choosing both seat actions first.

The mistake was treating that collector bridge as a full trainer before replay
and target semantics were proven.

## What Is Still Trusted

- Stock LightZero `train_muzero` is still the trusted loop.
- `source_state_fixed_opponent` is a valid stock control, not live self-play.
- `source_state_joint_action` is a valid stock centralized-control test, not
  competitive self-play.
- `source_state_turn_commit` is profile/smoke only until fake rows are removed
  from training replay.

Tiny canary status:

- `stock-frozen-canary-source-state-s304-20260512` proved the stock
  frozen-opponent route can call `train_muzero` on CPU and strictly load a real
  checkpoint opponent.
- `stock-frozen-gpu-base-canary-source-state-s304-20260512b` proved the same
  tiny route can run on an L4 GPU with `env_manager_type=base`.
- These are plumbing proofs, not learning proofs.

Native bridge status:

- A tiny hand-authored two-seat trace now passes native LightZero
  `GameSegment` / `MuZeroGameBuffer` reward/value/policy target assertions in
  Modal.
- This proves a route for a future trusted two-seat bridge. It does not make
  the old custom trainer trustworthy by itself.

## Current Main Lesson

For CurvyTron, the next useful proof is not a bigger custom run. It is a
stock-loop proof or a native replay bridge proof.

The exact custom-vs-stock seam is now known: the custom two-seat path diverged
at collection, `GameSegment`, replay buffer, target sampling, learner entry,
and checkpointing. The intended change was simultaneous action collection, but
the scaled path changed the training contract too.

Related detail docs:

- [failure audit](../curvytron_no_learning_failure_audit_2026-05-12.md)
- [train-muzero reconciliation](../curvytron_train_muzero_reconciliation_2026-05-12.md)
- [research folder](../curvytron_architecture_research_2026-05-12/README.md)
