# Training Code Inventory

Created: 2026-05-19.

Purpose: keep a plain inventory of what the training code is doing, why it is
not just a tiny LightZero hookup, and which files are bloated enough to deserve
cleanup.

## Short Answer

The actual LightZero call is still simple:

```text
lzero.entry.train_muzero([main_config, create_config], seed, max_train_iter, max_env_step)
```

The code is not simple because the repo has built a large experiment harness
around that call:

- custom CurvyTron environment and observation surfaces;
- frozen checkpoint opponents, blank-board opponents, scripted opponents, and
  immortal-opponent flags;
- dynamic tournament-to-trainer assignment refresh;
- checkpoint metadata, sidecars, lineage, and Modal Volume durability;
- background checkpoint evals and GIF capture;
- resume state beyond stock LightZero checkpoints;
- audits for policy observations, target rows, action diversity, and assignment
  provenance.

That harness is real work, but too much of it currently lives in one or two
launcher files.

## Biggest Files

Current line-count inventory from 2026-05-19:

| File | Lines | What It Contains |
| --- | ---: | --- |
| `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py` | 13225 | Modal trainer app, LightZero config patching, hooks, checkpoint mirroring, assignment refresh, resume, eval/GIF spawning, CLI. |
| `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py` | 12458 | Modal tournament app, checkpoint discovery/intake, online rating, leaderboard publishing, web/status modes, CLI. |
| `src/curvyzero/env/vector_runtime.py` | 6975 | Core vectorized game runtime. Large but closer to domain logic than orchestration. |
| `src/curvyzero/env/vector_visual_observation.py` | 5147 | Visual observation/rendering path. Large; needs careful modular cleanup later. |
| `src/curvyzero/training/curvytron_two_seat_lightzero_train_smoke.py` | 4743 | Older/two-seat LightZero smoke lane. Likely not the current mainline. |
| `src/curvyzero/tournament/curvytron_checkpoint_tournament.py` | 4143 | Pure tournament/rating helpers used by the Modal wrapper. |
| `src/curvyzero/env/vector_multiplayer_env.py` | 4054 | Multiplayer environment wrapper. |
| `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py` | 3567 | Main custom LightZero env wrapper for source-state visual survival. |

## What Is Actually LightZero

LightZero provides:

- the MuZero policy/model/learner/collector loop;
- replay sampling and learner minibatches;
- checkpoint save calls from `BaseLearner.save_checkpoint`;
- config compilation and env-manager integration.

CurvyZero currently injects itself at the edges:

- build patched LightZero configs from CurvyTron settings;
- provide a custom environment class;
- monkey-patch or wrap LightZero internals for progress files, checkpoint
  publishing, resume sidecars, target audits, learner metrics, and assignment
  refresh;
- mirror LightZero outputs from its experiment directory into durable Modal
  Volume paths.

## Why It Is Not A Simple Hookup

### Custom Environment Contract

The policy observation is not a stock Atari frame. It is a CurvyTron source-state
visual surface with explicit trail-render mode, bonus-render mode, backend,
seat perspective, and metadata sidecars. Tournament eval and trainer eval need
to reject incompatible checkpoints instead of silently loading the wrong visual
language.

### Opponent System

Training is not single-agent self-play in stock LightZero terms. The learner
controls one policy while the environment supplies an opponent. That opponent
can be a blank board, scripted policy, frozen LightZero checkpoint, or an
immortal variant of those. The current contract uses integer slot-count bags
that are materialized over collector envs.

### Live Feedback Loop

The desired system is closed-loop:

```text
trainer checkpoint -> intake -> tournament games -> leaderboard ->
trainer assignment -> trainer refresh
```

LightZero knows nothing about Modal Dicts, tournament leaderboards, assignment
refs, or safe refresh boundaries. That is repo-owned glue.

### Modal Durability

LightZero writes locally inside its experiment directory. The system also needs
durable Modal Volume artifacts with refs, hashes, sidecars, status files, and
lineage. That is why checkpoint save hooks and final mirror steps exist.

### Observability And Debugging

Because many previous runs failed in confusing ways, the trainer now records
lots of proof data: action telemetry, target audits, compiled config surface,
assignment load/apply lineage, checkpoint metadata, and background eval/GIF
status. This is useful, but it has made the launcher file too broad.

## Clean Boundary We Want

The current shape should move toward these modules:

- `training/lightzero_config.py`: build and validate LightZero configs.
- `training/lightzero_hooks.py`: install/restore checkpoint, resume, metrics,
  and target-audit hooks.
- `training/opponent_assignment_refresh.py`: resolve assignment refs and build
  collector-env reset params.
- `training/artifact_writer.py`: write status, command, lineage, and summary
  files.
- `infra/modal/train_app.py`: thin Modal entrypoint that wires the above
  modules together and calls LightZero.
- `tournament/*`: keep pure scheduling/rating logic separate from Modal web and
  CLI wrappers.

The rule should be: Modal files orchestrate; training modules define contracts;
environment modules implement game behavior; tournament modules rate policies.

## Current Risk

The code is functional enough to test, but the main risk is comprehension risk:

- one file owns too many unrelated behaviors;
- old smoke lanes and current lanes live beside each other;
- behavior changes and observability changes are mixed together;
- defaults exist in several places;
- tests cover many contracts, but the file layout makes it hard to see the
  current mainline.

The next cleanup should be boundary cleanup, not a huge rewrite. Extract pure
helpers first, keep tests green, and leave launch behavior unchanged until the
new boundaries are obvious.
