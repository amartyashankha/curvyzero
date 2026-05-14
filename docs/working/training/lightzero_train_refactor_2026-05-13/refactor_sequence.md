# Refactor Sequence

This is the intended order. Do not skip the test gates.

## Step 1: Add Tests

Status: done for Bug 1.

Add focused tests that fail on the fixed checkpoint path bug.

Expected files:

- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`
- `tests/test_lightzero_curvytron_run_status.py` if a new focused status test
  file is cleaner
- maybe `tests/test_opponent_mixture.py` for manifest/ref freezing

First bundle:

- progress latest selects timestamped checkpoint;
- auto-resume selects timestamped checkpoint;
- resume sidecar scans timestamped state dir;
- poller schedules timestamped checkpoint;
- status summary scans timestamped dirs.

## Step 2: Minimal Bugfix In Current Layout

Patch the existing functions to use broad checkpoint discovery. Keep this patch
small.

Status: done for Bug 1.

## Step 3: Extract Checkpoint Helpers

Move pure checkpoint parsing/discovery/selection into a small helper module.
Keep wrappers in the old trainer file.

Status: partly done. Pure exp-dir discovery, iteration-name parsing,
checkpoint candidate collection, and latest-selection ordering are now in
`src/curvyzero/training/lightzero_checkpoints.py`.

Not done: full run/attempt resume policy and poller scheduling policy. Those
still stay near the trainer because they include Modal volume reloads, source
labels, sleeps, remote calls, and live status writes.

## Step 4: Extract Resume Helpers

Move resume sidecar discovery and selection after checkpoint helper behavior is
stable.

Current stance: do not extract full resume yet. First decide whether the next
small cut is exact-iteration sidecar candidate lookup, with Modal/state payloads
left in the trainer.

## Step 4A: Opponent Assignment Guardrails

Status: started. Before implementing any registry, the trainer now enforces
immutable exact checkpoint refs for both mixture and top-level frozen-opponent
paths.

Pure assignment parsing now exists in `opponent_registry.py`, but it is not
wired into the trainer. Do not read Modal Dict inside the trainer.

Next small cut: add trainer-consumption tests for an assignment snapshot before
threading that snapshot into manifest/config building.

## Step 4B: Stock Boundary Gate

Status: started. A focused test now builds the trainer config, instantiates the
registered LightZero-facing source-state env, and steps one scalar action with
an opponent mixture. This protects the boundary that LightZero sees one action
and opponent behavior stays inside the env.

## Step 5: Extract Progress/Status Payload Helpers

Separate pure payload construction from writes.

## Step 6: Extract Poller Candidate Logic

Keep Modal remote calls in the Modal file. Move candidate discovery and
stability logic behind tests.

## Step 7: Clean Tests And Names

Delete or rename stale tests after the new contract tests pass.
