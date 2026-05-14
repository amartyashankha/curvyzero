# Regression Test Lockdown Plan

## Goal

Before refactoring, make the current important behavior testable. The first
bugfix target is checkpoint discovery across LightZero timestamped experiment
directories.

## Tests To Add Or Strengthen

### Checkpoint Discovery

Use a temp run root with:

```text
train/lightzero_exp/ckpt/iteration_0.pth.tar
train/lightzero_exp_260513_123802/ckpt/iteration_180000.pth.tar
```

Expected behavior: broad discovery finds `iteration_180000`.

Cases:

- select latest numbered checkpoint across all `lightzero_exp*`;
- ignore invalid names;
- ignore missing or empty files if the current helpers treat them as unsafe;
- deterministic tie-break when the same iteration appears in multiple dirs.

### Status And Progress

Given the same temp tree, the latest-progress payload should report the broad
latest checkpoint, not the fixed-path checkpoint.

### Auto-Resume

Auto-resume should pick the broad latest checkpoint and the matching sidecar
when available.

### Poller

The checkpoint poller should discover new checkpoint candidates from
timestamped dirs and not mark a run stale only because `lightzero_exp/ckpt`
stopped moving.

### Eval/GIF

Background eval/GIF scheduling should receive the actual checkpoint ref. Tests
can use stubs for the worker call and assert the ref passed to the stub.

### Manifest And Tournament Inputs

Any helper that freezes a "recent", "mid", or "latest" checkpoint ref should
use broad discovery or clearly reject fixed stale refs.

### Opponent Assignment Inputs

Any frozen opponent consumed by training should be an exact immutable
`iteration_N.pth.tar` ref by the time it reaches the trainer.

Cases:

- top-level frozen opponent ref rejects `latest.pth.tar`;
- top-level frozen opponent ref rejects `ckpt_best.pth.tar`;
- top-level frozen opponent ref rejects non-iteration names;
- mixture frozen refs keep the same rule;
- future assignment snapshots are copied into run metadata after resolution.

## End-To-End Local Contract

Build a fake run attempt in a temp directory, then assert one high-level reader
returns a coherent view:

- latest checkpoint is timestamped `iteration_180000`;
- status payload agrees;
- resume selection agrees;
- poller candidates include it;
- generated eval/GIF job metadata points to it.

This should not require Modal.
