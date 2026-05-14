# Source File Inventory

Purpose: list files relevant to this training refactor lane.

## Primary Training Files

| File | Role | Current stance |
| --- | --- | --- |
| `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py` | Main Modal trainer launcher, LightZero config, checkpoint hooks, poller, eval/GIF, CLI. | Primary refactor target. |
| `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py` | Run status reader and summary table. | Needs checkpoint-discovery tests and likely patch. |
| `src/curvyzero/training/lightzero_checkpoints.py` | Pure LightZero checkpoint/resume-state path and filename helpers. | First extracted helper; keep Modal-free. |
| `src/curvyzero/training/opponent_registry.py` | Pure opponent assignment snapshot parser. | First assignment helper; keep Modal-free and tournament-free. |
| `src/curvyzero/training/opponent_leaderboard.py` | Pure public leaderboard snapshot, pointer, assignment selector, and audit helpers. | Keep Modal-free; current selector source of truth. |
| `src/curvyzero/training/lightzero_checkpoint_opponent_provider.py` | Frozen LightZero checkpoint opponent provider. | Loads trusted internal checkpoints and infers support-head config from state dict shape. |
| `scripts/materialize_curvytron_leaderboard_assignment.py` | Local materializer for exported rating/API JSON into snapshot/pointer/assignment/audit files. | Operator helper; no Modal writes. |
| `scripts/build_curvytron_opponent_mixture_manifest.py` | Builds mixture manifests that can freeze checkpoint refs. | Relevant to stale-ref bug; avoid broad redesign now. |
| `scripts/build_curvytron_stock_train_manifest.py` | Older stock train manifest builder with explicit frozen checkpoint refs. | Candidate for registry/assignment cleanup later. |
| `scripts/build_curvytron_survivaldiag_manifest.py` | Survival diagnostic manifest builder with explicit frozen checkpoint refs. | Candidate for registry/assignment cleanup later. |
| `tests/test_curvytron_live_checkpoint_eval_plumbing.py` | Main trainer plumbing test file. | Likely first place for regression tests. |
| `tests/test_lightzero_timestamped_checkpoint_discovery.py` | Focused regression coverage for broad `lightzero_exp*` discovery. | Current source of truth for Bug 1. |
| `tests/test_opponent_mixture.py` | Mixture parsing/threading tests. | Keep; only touch if manifest/ref selection tests belong here. |
| `tests/test_opponent_registry.py` | Opponent assignment snapshot parser tests. | Current source of truth for assignment snapshot parsing. |
| `tests/test_opponent_leaderboard.py` | Public leaderboard and assignment selector tests. | Current source of truth for pure leaderboard-to-assignment behavior. |
| `tests/test_lightzero_checkpoint_opponent_provider.py` | Checkpoint opponent provider tests. | Current source of truth for support-head inference. |

## Adjacent But Not Main

| File | Role | Current stance |
| --- | --- | --- |
| `src/curvyzero/tournament/*` | Tournament ranking/eval. | Adjacent consumer of checkpoint refs; not the training refactor target. |
| `tests/test_curvytron_checkpoint_tournament.py` | Tournament tests including broad checkpoint discovery. | Evidence and model for tests, but avoid coupling trainer refactor to tournament. |
| `src/curvyzero/training/multiplayer_source_state_trainer_surface.py` | Environment/trainer surface adapter. | Read only for interface contracts unless a trainer test exposes mismatch. |
| `tests/test_multiplayer_source_state_trainer_surface.py` | Env surface tests. | Not part of this refactor unless interface contract changes. |

## Candidate New Helper Modules

Candidate modules still require tests before extraction:

- `src/curvyzero/training/lightzero_progress.py`
- `src/curvyzero/training/lightzero_resume.py`
- `src/curvyzero/training/lightzero_background_eval.py`
