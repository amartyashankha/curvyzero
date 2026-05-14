# Current State

## Implemented Today

| Area | State | Evidence |
| --- | --- | --- |
| Tournament ratings | Implemented. Final ratings are written to the tournament Volume as `latest.json` plus per-round `ratings.json`. | `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py::_write_rating_round_outputs` |
| Website/API reader | Implemented. `/api/rating-standings` reads final/provisional snapshots. | `curvytron_tournament_browser`, `_read_best_rating_snapshot_for_run` |
| Intake Dict/Queue | Implemented for V0 guarded batch intake. | `checkpoint_intake_state`, `checkpoint_intake_queue`, `curvytron_checkpoint_intake_*` |
| Scheduled intake ticks | Implemented: subscriber tick and drain tick exist on a 60s schedule. | `curvytron_checkpoint_intake_subscriber_tick`, `curvytron_checkpoint_intake_drain_tick` |
| Broad checkpoint discovery | Implemented in tournament and trainer paths. | `train/lightzero_exp*/ckpt/iteration_*.pth.tar` |
| Rating context hash | Implemented for tournament evaluator/rating context. | `rating_context_hash` |
| Checkpoint roster guard | Implemented for rating evidence reuse. | `checkpoint_roster` in rating snapshots |
| Opponent assignment parser | Implemented as pure parser. | `src/curvyzero/training/opponent_registry.py` |
| Training scripted/blank/passive opponent modes | Implemented as training opponent-mixture/env settings. | `opponent_mixture.py`, `curvyzero_source_state_visual_survival_lightzero_env.py` |
| Pure leaderboard-to-assignment bridge | Implemented locally: build/validate public leaderboard snapshots, live pointers, 3-5 slot assignments, and assignment audits. | `src/curvyzero/training/opponent_leaderboard.py`, `tests/test_opponent_leaderboard.py` |
| Local materialization CLI | Implemented locally: turns exported rating/API JSON into snapshot, pointer, assignment, and audit JSON artifacts. | `scripts/materialize_curvytron_leaderboard_assignment.py` |
| Assignment artifact writer | Implemented and smoke-tested: stores `assignment.json` and optional `audit.json` under a training attempt. | `mode=write-assignment`, `_write_opponent_assignment_artifacts`, focused regression |
| Trainer assignment-ref plumbing | Implemented locally: train and poller command construction accept an assignment ref, resolve it through the existing mixture resolver, and propagate metadata. | `_resolve_opponent_assignment_for_env`, `_run_visual_survival_train`, `_checkpoint_eval_poller_command`, focused tests |
| Tournament-side leaderboard publisher | Implemented and remote-smoked: reads rating `latest.json`, writes public snapshot/latest refs, commits the Volume, then updates Dict pointer. Refuses provisional ratings unless explicitly allowed. | `curvytron_opponent_leaderboard_publish`, `test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer`, smoke `curvytron-latest212-smoke-20260513` |
| Manual closed-loop smoke | Proven manually: trainer assignment -> train checkpoints -> discovery/intake -> rating -> public leaderboard -> new assignment -> trainer smoke. | `implementation_log.md`, smoke runs `leaderboard-assignment-train-smoke-20260513a/c` and `arena-closed-loop-smoke-20260513b` |

## Designed But Not Fully Wired

| Area | Desired state | Current gap |
| --- | --- | --- |
| Public leaderboard snapshot production policy | Immutable `curvyzero_opponent_leaderboard_snapshot/v0` written to Volume from rating snapshots. | Smoke path works; still need production naming/retention/repair policy. |
| Modal Dict live pointer repair | Small pointer to current leaderboard snapshot, not durable truth. | Publisher writes pointer; repair/fallback command not implemented. |
| Assignment selector persistence | Deterministic selector writes assignment and audit under a training attempt. | Smoke writer exists; still needs production runbook and scheduled/operator policy. |
| Periodic assignment refresh | Long-running trainers should refresh slot assignments at safe boundaries. | Not automated; manual second-generation smoke proves mechanics. |
| Online Elo continuation | New checkpoints enter existing pool, get placement, then adaptive rounds continue from `latest.json`. | Local Inspector-side continuation plumbing now has focused tests, but no production-scale remote smoke yet. Queue repair and stale-claim repair are still open. |
| Public active status | Rows become training-eligible only with enough games/opponents/context. | Active/provisional exists in rating rows, but public training contract not published. |
| One-frame tournament parity | Official leaderboard should match current one-frame train semantics. | Tournament has cadence fields/hash, but new one-frame leaderboard launch needs explicit validation. |
| Non-checkpoint tournament players | Scripted/hand-coded players should eventually be representable if we want them ranked. | Current tournament/rating player specs require `checkpoint_ref`; policy loading goes through checkpoint loading only. |

## Current Architecture Boundary

Training should not poll Modal Dict or tournament state inside `train_muzero`.

Correct flow:

```text
leaderboard snapshot -> assignment selector -> immutable assignment.json -> trainer launch config
```

Incorrect flow:

```text
trainer step loop -> live Elo / Modal Dict / tournament API
```

## Current Decision

Before launching long-running leaderboard-fed training:

1. Add repair/fallback tooling for stale or missing leaderboard Dict pointers.
2. Promote the assignment writer/operator flow from smoke path to documented
   production runbook.
3. Validate tournament one-frame evaluator semantics.
4. Add safe refresh/continuation policy for long-running trainers.
5. Harden online Elo continuation and queue/dedupe repair.
6. Run a larger bounded closed-loop smoke.
7. Only then launch leaderboard-derived long-running training.

## Important Representation Split

Training already supports non-neural opponent concepts through mixture entries:

- `fixed_straight`;
- `proactive_wall_avoidant`;
- `blank_canvas_noop`;
- `opponent_death_mode=immortal`;
- frozen LightZero checkpoints.

The tournament/rating stack today is neural-checkpoint-centric:

- every normalized player requires a `checkpoint_ref`;
- `rating_roster_by_checkpoint` stores checkpoint refs and model/render metadata;
- game workers call checkpoint policy loading;
- resume-from-latest restores players from checkpoint refs.

So scripted/hand-coded policies can be **training opponents now**, but they are
not first-class **leaderboard players** yet.
