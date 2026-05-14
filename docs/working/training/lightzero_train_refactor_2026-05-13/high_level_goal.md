# High-Level Goal

Make the CurvyTron stock LightZero training lane boring and trustworthy.

## What "Boring" Means

- The trainer calls stock `lzero.entry.train_muzero`.
- CurvyZero adds only the scaffolding we need around that call.
- Checkpoints, status, resume, poller, eval/GIF, and manifests agree about the
  same run state.
- The code is split by responsibility, not by accident.
- Tests catch the known bugs before code is moved.

## Main Success Condition

After this refactor, a future agent should be able to answer:

- where checkpoints are discovered;
- how resume selects a checkpoint;
- how status reports latest progress;
- how checkpoint eval/GIF jobs are triggered;
- which tests protect those behaviors;
- where to change one behavior without touching unrelated pieces.

## Current Priority

Lock down the trainer scaffolding with regression tests, then fix the known
checkpoint-discovery bug, then refactor in small tested cuts.

