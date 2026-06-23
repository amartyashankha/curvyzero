# Trainer Bloat Critique

Last updated: 2026-05-19

## Plain Question

Why is `lightzero_curvyzero_stacked_debug_visual_survival_train.py` so large if the intended design is mostly stock LightZero plus small CurvyZero changes?

## Short Answer

The file grew because it became both the training entrypoint and the operational control plane. It owns:

- CLI/defaults/manifests;
- Modal app and volume wiring;
- LightZero config launch defaults and compatibility facades;
- env contract metadata;
- checkpoint discovery, resume, progress, and sidecars;
- live checkpoint publishing;
- background eval and GIF scheduling;
- opponent assignment refresh;
- initial-policy loading;
- learner metrics and target audit hooks;
- status summaries and artifact payloads;
- the final `train_muzero` call.

That is too much. The LightZero call is small; the surrounding operational scaffolding swallowed the file.

## Main Self-Critique

- I treated each urgent runtime issue as a local patch to the giant trainer instead of repeatedly extracting the pure contract underneath it.
- I let private helper imports from the trainer become an informal API for eval, tests, and tournament code.
- I overused compatibility aliases and hidden fallbacks instead of making the current intended path explicit.
- I allowed operational concerns and algorithmic concerns to mix: reward support, opponent mixture, checkpoint metadata, and Modal lifecycle are all in the same file.
- I should have kept a stricter rule: the Modal trainer may orchestrate remote work, but pure reward/config/opponent/hook contracts belong in small modules.

## Why Some Complexity Is Real

Some of the size is not accidental:

- LightZero does not expose every callback we need, so checkpoint publishing, resume, opponent refresh, and target audit currently require hooks around LightZero internals.
- Modal volume semantics, detached jobs, background eval/GIF, and live checkpoint publishing are real operational concerns.
- The tournament feedback loop needs durable metadata on every checkpoint.
- Frozen opponents require checkpoint metadata, policy loading, observation contract compatibility, and assignment refresh.

But this does not justify one 13k-line file. It just means those concerns need explicit modules and contracts.

## Desired Shape

The Modal trainer should become:

1. parse/validate launch inputs;
2. resolve Modal volumes and checkpoint refs;
3. build LightZero config through a pure builder;
4. install a hook bundle;
5. call stock `train_muzero`;
6. write final status.

Everything else should live behind smaller modules:

- `reward_contracts.py`;
- `lightzero_config_builder.py`;
- `lightzero_hooks.py`;
- `opponent_assignment_runtime.py`;
- `checkpoint_publishers.py`;
- `training_extensions/`.

## Acceptance Checklist For Trainer Edits

Any future trainer edit should satisfy at least one of these:

- It removes a responsibility from the trainer.
- It moves a public contract into a pure module.
- It keeps a real Modal side effect in the Modal layer and names why it belongs there.
- It is a temporary compatibility shim with a ledger entry and deletion criteria.

Before merging the edit:

- focused local tests pass;
- no new private trainer helper import is introduced outside entrypoint/integration tests;
- any fallback in the touched path is deleted or ledgered;
- the task docs state whether behavior changed.

## Open Critique Questions

- Which parts are pure and can move immediately?
- Which parts are Modal side effects and should stay near Modal wrappers?
- Which private trainer helpers have leaked into other modules?
- Which hook patches are only needed because LightZero lacks an extension point?
- Which defaults are obsolete and should be deleted rather than preserved?
- Which tests currently lock in accidental structure instead of real behavior?

## Current Critique Synthesis

- The file is large because pure contracts and operational side effects were allowed to live in the same module.
- Tests and eval code import the trainer as a library, which keeps the bloat alive.
- Modal should own remote execution and volume/job operations, not reward/config/opponent/hook contracts.
- The first cleanup moves are already reducing dependency on the Modal trainer as an API: reward contracts are public, config patch/surface helpers are public, the visual-survival config builder has a typed public boundary, and eval no longer imports trainer-private config helpers.

Current extraction order:

1. Reward contracts: current cut wired through env/trainer/eval.
2. Opponent assignment vocabulary and unit names: started; split metadata proof is pinned.
3. LightZero config builder: typed public boundary landed; broad facades remain ledgered for protected callers.
4. Checkpoint metadata payloads.
5. Status/progress payloads.
6. Hook bundle/runtime publisher.
7. Background eval/GIF planning.
8. Thin Modal trainer adapter.

Key conclusion from critique agents:

The trainer is not big because the project needs a custom MuZero implementation. The stock call is small. The bloat is from operational and research scaffolding living in the same file as the stock call, then becoming an informal API for eval/tests/tournament/GIF code.
