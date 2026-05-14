# Refactor Targets

This is a first-pass target list. Do not move code until regression tests are
in place.

## Target 1: Checkpoint Discovery

Current pain: checkpoint lookup logic is spread across trainer progress,
resume, poller, status, live publishing, and external scripts. Some readers
assume `train/lightzero_exp/ckpt`, which misses timestamped
`lightzero_exp_*` directories.

Desired shape:

- one pure helper for parsing checkpoint names;
- one pure helper for discovering all LightZero experiment checkpoint dirs;
- one pure helper for selecting latest/all/specific checkpoints;
- callers pass roots and receive plain data, not Modal objects.

Candidate future module:

```text
src/curvyzero/training/lightzero_checkpoints.py
```

## Target 2: Resume Selection

Current pain: checkpoint and sidecar resume lookup should agree on the same
iteration and discovery roots.

Desired shape:

- one helper that returns the selected checkpoint;
- one helper that finds the matching sidecar;
- trainer launcher only applies the selected path to LightZero config.

## Target 3: Progress And Status Payloads

Current pain: progress/status writing mixes payload construction, file writes,
checkpoint discovery, and display fields.

Desired shape:

- pure payload builders;
- thin file-writing wrappers;
- status readers use the same checkpoint discovery contract.

## Target 4: Background Eval/GIF Scheduling

Current pain: poller scanning, checkpoint stability, worker request payloads,
and Modal calls sit together.

Desired shape:

- pure candidate discovery and stability state;
- pure request payload builders;
- Modal remote calls stay in the Modal file.

## Target 5: Modal Entrypoints

Current pain: the Modal trainer file holds too many local pure helpers.

Desired shape:

- Modal functions remain, but mostly parse kwargs, call pure helpers, call
  stock `train_muzero`, and write final artifacts.
- Heavy pure logic becomes importable without Modal.

## Target 6: Opponent Registry Boundary

Current pain: frozen checkpoint opponents and scripted opponent mixtures are
threaded through hard-coded strings, manifest defaults, trainer validation,
eval/GIF metadata, and environment config. This is the exact shape that would
make the trainer file grow again if a tournament-fed opponent pool is added
directly.

Desired shape:

- tournament/control plane writes opponent registry data;
- pure training helper parses and validates registry data;
- trainer receives a resolved opponent spec and passes it into env config;
- checkpoint path resolution uses `lightzero_checkpoints.py`;
- Modal Dict or volume JSON access is an adapter, not trainer logic.

Candidate future module:

```text
src/curvyzero/training/opponent_registry.py
```

Design note:

- [opponent_registry_design.md](opponent_registry_design.md)

## Explicit Non-Targets

- Environment mechanics.
- Reward redesign.
- Opponent redesign.
- Old custom `two-seat-selfplay` learning path.
- Large trainer rewrite.
