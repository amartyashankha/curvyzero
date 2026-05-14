# Checkpoint Helper Contract

This is the helper contract for the checkpoint/refactor lane. The first slice
is implemented in `src/curvyzero/training/lightzero_checkpoints.py`.

## Problem

Multiple trainer-side readers need the same answer:

> What LightZero iteration checkpoints exist for this run or attempt, and which
> one is the latest?

Today each reader answers that differently.

## Desired Plain Data Shape

Partly implemented as `LightZeroCheckpointCandidate`:

```text
iteration
path
mtime
mtime_ns
size_bytes
checkpoint_name
exp_dir_name
```

Modal refs and source labels are intentionally still caller-owned.

One checkpoint candidate should be represented as plain data:

```text
iteration
name
path
ref
source_kind
exp_dir_name
size_bytes
mtime_ns
```

No Modal function calls. No policy loading. No training behavior.

## Discovery Roots

Implemented for one configured exp dir:

```text
lightzero_exp
lightzero_exp_*
```

The helper exposes checkpoint dirs and resume-state dirs for those exp roots.
Run/attempt selection still lives near its callers.

For one attempt train root, scan:

```text
train/lightzero_exp*/ckpt/iteration_*.pth.tar
```

For resume across a run, scan:

```text
<current attempt>/train/lightzero_exp*/ckpt/iteration_*.pth.tar
<prior attempts>/train/lightzero_exp*/ckpt/iteration_*.pth.tar
<run>/checkpoints/lightzero/iteration_*.pth.tar
```

For sidecar resume state, scan matching state dirs:

```text
train/lightzero_exp*/<resume_state_dir>/iteration_*.resume_state.pkl
<run>/checkpoints/<resume_state_dir>/iteration_*.resume_state.pkl
```

## Selection Rule

Partly centralized for checkpoint candidates:

1. iteration;
2. mtime_ns;
3. size;
4. path string as deterministic tie-break.

The helper supports optional empty-file filtering, but callers must choose
whether to require non-empty files. Do not silently change caller behavior
without tests.

Latest checkpoint is the max by:

1. iteration;
2. mtime;
3. size;
4. ref/path string as deterministic tie-break.

Ignore:

- invalid names;
- non-files;
- zero-byte files;
- mutable `ckpt_best` for resume/eval scheduling unless a caller explicitly
  asks for it.

## Callers To Migrate

- progress writer;
- auto-resume;
- resume sidecar lookup;
- live checkpoint publisher;
- checkpoint eval poller;
- run status summary;
- manifest checkpoint ref freezing.

## First Implementation Shape

Before creating a new module, implement or patch the behavior behind the
existing functions with tests. After the tests pass, extract the helper without
changing behavior.

Status: done for path discovery, filename parsing, candidate collection, and
latest checkpoint ordering. Payload construction remains in caller code.
