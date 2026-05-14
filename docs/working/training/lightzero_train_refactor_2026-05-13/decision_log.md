# Decision Log

## 2026-05-13: Scope The Refactor To Training Code

Decision: this lane is owned by the training coach. It covers trainer
scaffolding, not environment redesign.

Why: the immediate bugs and complexity are in checkpoint discovery, resume,
status, poller, eval/GIF scheduling, manifests, and Modal wrappers.

Consequence: read environment code only to understand the public contract the
trainer consumes.

## 2026-05-13: Tests Before Refactor

Decision: add regression tests before moving code.

Why: the trainer file is large and has live bugs. Moving code without tests
will hide whether we fixed or moved the bug.

Consequence: first implementation phase is test lockdown, then minimal bugfix,
then extraction.

## 2026-05-13: First Bug Target Is Broad Checkpoint Discovery

Decision: the first bugfix target is the fixed-path checkpoint discovery issue.

Why: it affects multiple training-scaffolding surfaces and can make healthy
runs look stale.

Consequence: checkpoint discovery should become the first pure helper contract.

## 2026-05-13: Extract Only Pure Checkpoint Path Helpers First

Decision: extract LightZero exp-dir sibling discovery and iteration-name
parsing into `src/curvyzero/training/lightzero_checkpoints.py`, while leaving
selection and Modal side effects near their current callers for now.

Why: this removes duplicated fixed-path assumptions without moving unrelated
training behavior. It also gives tests a small module to pin directly.

Consequence: later refactors should build on this module instead of adding new
`lightzero_exp` string scans inside Modal files.

## 2026-05-13: Opponent Control Should Be Assignment Snapshots

Decision: future tournament-fed opponents should enter training as a frozen
opponent assignment snapshot, not as live polling inside the running trainer.

Why: a running `train_muzero` job should be reproducible. Modal Dict can be a
pointer/cache surface, but the exact consumed assignment should be durable and
copied into run metadata.

Consequence: trainer code should validate and pass through a resolved opponent
spec. Tournament ranking, refresh cadence, and registry layout stay outside the
trainer.
