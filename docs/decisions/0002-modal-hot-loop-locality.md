# ADR-0002: Keep Modal Network Primitives Out Of The Hot Loop

Date: 2026-05-08
Status: Proposed

## Context

CurvyZero needs very high-throughput environment stepping, self-play, model inference, and eventually MCTS node evaluation. Modal provides useful distributed primitives such as Functions, Queues, Dicts, Volumes, CloudBucketMounts, Sandboxes, and snapshots, but many of these are networked coordination tools.

## Decision

Keep environment ticks, MCTS search, inference batches, replay sampling, and training updates inside one process/container hot loop whenever possible. Use Modal Functions for coarse jobs, Volumes or buckets for chunked artifacts, Queues for coarse work dispatch, Dicts for tiny metadata, and snapshots only as startup/debug accelerators.

## Evidence

- `docs/research/modal_patterns.md`
- The handoff notes that Modal Queue/Dict operations add network latency that is inappropriate per action, tick, or MCTS node.

## Consequences

- First Modal jobs should look like local scripts wrapped in Modal, not distributed microservices.
- Actor/evaluator/trainer distribution should happen at coarse granularity.
- Replay must be chunked and checkpointed rather than streamed through tiny queue messages.

## Reversal Conditions

- If profiling proves the hot loop is not latency-sensitive or a Modal primitive gains a low-latency local mode, specific uses can be reconsidered with a benchmark.

