# Experiment Grid Principles

## Purpose

This doc sets the operating rules for Optimizer architecture re-exploration runs. The goal is to compare launch shapes, row layouts, and readback behavior in a way that produces clear decisions, not just more traces.

Keep experiments small, named, and repeatable. Each run should answer one practical question about throughput, scheduling, collection cost, or data shape.

## Current Trusted Path

The trusted path is the current launcher/profile surface in this working area:

- Use stock LightZero `train_muzero` through `--mode profile`.
- Use `env_variant=source_state_fixed_opponent`.
- Use `opponent_policy_kind=frozen_lightzero_checkpoint` with the frozen
  opponent on CPU unless the row is explicitly an opponent-device test.
- Use the current profile tensor shape as the readback contract.
- Treat the old custom two-seat path as historical only. Do not optimize around it or revive assumptions from it.
- Do not interfere with live Coach training. Optimizer experiments must be isolated from active Coach runs, ports, checkpoints, and shared resources.

## Rows Must Be a Structured Grid/Tensor

Rows are not loose log lines. They must form a structured grid/tensor so we can compare experiments without hand interpretation.

Each row should have stable axes such as:

- launcher mode
- worker count
- actor count
- environment count
- collection mode
- batch/readback size
- seed
- run name

This matters because useful readback depends on slicing the same tensor across runs. If a field is missing, renamed casually, or encoded inside free text, the result becomes harder to trust and harder to compare.

## Run Naming Rules

Run names should be short, stable, and sortable.

Use:

```text
optgrid_<date>_<question>_<axis-summary>_<index>
```

Example:

```text
optgrid_2026-05-12_fanout_workers-w8-a32_001
```

Rules:

- Include the date in `YYYY-MM-DD`.
- Include the main question or hypothesis.
- Include the most important axis values.
- Use a numeric suffix for repeats.
- Do not encode every parameter in the name. The row tensor is the source of truth.
- Do not reuse a run name for a changed config.

## Launch and Readback Policy

Launch one controlled grid at a time. Prefer small grids that finish cleanly over broad grids that are hard to explain.

Current profile readback shape:

```text
parent launch: `modal run --detach`
child launch: `--profile-spawn`
result: `modal.FunctionCall.from_id(function_call_id).get()`
profile volume commit: off
```

Do not use non-detached `--profile-spawn` for wide runs. It can print a call id
and then let the parent app stop before the child profile finishes.

Before launch:

- confirm the run name
- confirm the grid axes
- confirm output location
- confirm no live Coach training resource overlap

After launch:

- read back through the structured rows/tensor
- check completion and failure counts first
- compare only runs with matching axes
- mark partial runs as partial, not failed by default
- record the decision the run supports

Do not treat ad hoc terminal output as the result. It can help debug, but the structured readback is the experiment record.

## Useful Results

A result is useful if it changes a decision or narrows the next grid.

Useful results include:

- a clear winner for a launch shape under the same workload
- a confirmed bottleneck axis, such as workers, actors, envs, or readback size
- a failure mode that rules out a configuration
- evidence that two settings are equivalent enough to choose the simpler one
- a smaller follow-up grid with a clear reason

A result is not useful if it only says "faster" or "slower" without comparable axes, completion status, and workload shape.

## What Not To Profile Right Now

Do not profile these yet:

- live Coach training behavior
- the old custom two-seat path
- checkpoint quality or policy strength
- long training convergence
- model architecture changes
- UI or visualization overhead
- broad system-level profiling that is not tied to the current launcher/readback question

For now, stay focused on launcher shape, structured collection, row tensor integrity, and readback clarity.
