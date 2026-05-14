# Current State

## Implemented Today

| Area | State | Evidence |
| --- | --- | --- |
| Tournament ratings | Implemented. Final ratings are written to the tournament Volume as `latest.json` plus per-round `ratings.json`. | `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py::_write_rating_round_outputs` |
| Website/API reader | Implemented. `/api/rating-standings` reads final/provisional snapshots. | `curvytron_tournament_browser`, `_read_best_rating_snapshot_for_run` |
| Intake Dict/Queue | Implemented for V0 guarded batch intake. | `checkpoint_intake_state`, `checkpoint_intake_queue`, `curvytron_checkpoint_intake_*` |
| Scheduled intake ticks | Implemented: subscriber tick and drain tick exist on a 60s schedule. | `curvytron_checkpoint_intake_subscriber_tick`, `curvytron_checkpoint_intake_drain_tick` |
| Submit-only tournament service surface | Implemented locally: `intake-seed` configures policy; `tournament-submit` / `intake-submit` accepts candidate refs/run IDs without scheduler knobs. | `curvytron_checkpoint_intake_submit`, `src/curvyzero/tournament/checkpoint_intake_service.py`, focused tests |
| Broad checkpoint discovery | Implemented in tournament and trainer paths. | `train/lightzero_exp*/ckpt/iteration_*.pth.tar` |
| Rating context hash | Implemented for tournament evaluator/rating context. | `rating_context_hash` |
| Checkpoint roster guard | Implemented for rating evidence reuse. | `checkpoint_roster` in rating snapshots |
| Active tournament pool | Implemented locally: rating snapshots and public leaderboard snapshots default to a top-100 active pool. Mature rows below the cutoff are marked `retired`; provisional/new rows are not retired early; scheduler excludes retired rows but history remains in `latest.json`. | `DEFAULT_RATING_ACTIVE_POOL_LIMIT`, `rating_snapshot_from_pair_results`, `build_rating_round_pair_specs`, `build_leaderboard_snapshot_from_rating_snapshot`, focused tests |
| Opponent assignment parser | Implemented as pure parser. | `src/curvyzero/training/opponent_registry.py` |
| Training scripted/blank/passive opponent modes | Implemented as training opponent-mixture/env settings. | `opponent_mixture.py`, `curvyzero_source_state_visual_survival_lightzero_env.py` |
| Pure leaderboard-to-assignment bridge | Implemented locally: build/validate public leaderboard snapshots, live pointers, `top_slots_v0` smoke assignments, `stable_slots_v1` production-direction assignments, and assignment audits. `slot_rules_v0` is purged as the production path. | `src/curvyzero/training/opponent_leaderboard.py`, `tests/test_opponent_leaderboard.py`, `slot_architecture_feedback.md` |
| Local materialization CLI | Implemented locally: turns exported rating/API JSON into snapshot, pointer, `stable_slots_v1` assignment, and audit JSON artifacts by default. | `scripts/materialize_curvytron_leaderboard_assignment.py` |
| Checkpoint recency metadata | Locally repaired: tournament checkpoint normalization, intake manifest -> rating spec, rating rows, public leaderboard rows, and `stable_slots_v1` now preserve the metadata needed for a real `recent_strong` slot. | `checkpoint_metadata_from_ref`, `_intake_manifest_rating_checkpoints`, focused metadata tests |
| Assignment artifact writer | Implemented and smoke-tested: stores `assignment.json` and optional `audit.json` under a training attempt. | `mode=write-assignment`, `_write_opponent_assignment_artifacts`, focused regression |
| Trainer assignment-ref plumbing | Implemented locally: train and poller command construction accept an assignment ref, resolve it through the existing mixture resolver, and propagate metadata. | `_resolve_opponent_assignment_for_env`, `_run_visual_survival_train`, `_checkpoint_eval_poller_command`, focused tests |
| Tournament-side leaderboard publisher | Implemented and remote-smoked: reads rating `latest.json`, writes public snapshot/latest refs, commits the Volume, then updates Dict pointer. Refuses provisional ratings unless explicitly allowed. | `curvytron_opponent_leaderboard_publish`, `test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer`, smoke `curvytron-latest212-smoke-20260513` |
| Manual closed-loop smoke | Proven manually through `stable_slots_v1`: trainer assignment -> train checkpoints -> discovery/intake -> rating -> public leaderboard -> new assignment -> trainer smoke. Caveat: the first stable-slot smoke exposed a metadata gap where `recent_strong` could choose an older high-ranked checkpoint. Local code/tests now carry run/attempt/iteration/latest metadata through rating rows; rerun the remote smoke before trusting automatic refresh. | `implementation_log.md`, `gaps_and_tests.md` |

## Designed But Not Fully Wired

| Area | Desired state | Current gap |
| --- | --- | --- |
| Public leaderboard snapshot production policy | Immutable `curvyzero_opponent_leaderboard_snapshot/v0` written to Volume from rating snapshots. | Smoke path works; still need production naming/retention/repair policy. |
| Modal Dict live pointer repair | Small pointer to current leaderboard snapshot, not durable truth. | Local repair command exists and rebuilds `current:<leaderboard_id>` from validated immutable Volume snapshots. Needs remote operator smoke. |
| Assignment materializer persistence | `stable_slots_v1` writes concrete assignment entries and audit locally; assignment writer stores assignment/audit under a training attempt. | Production runbook and scheduled/operator policy are still needed. |
| Run-scoped slot recipe control | Desired slot recipes should live in a Modal Dict keyed by training run id, then materialize to immutable assignment JSON. | Designed in `run_slot_control_design.md`; not implemented yet. |
| Periodic assignment refresh | Long-running trainers should refresh slot assignments at safe boundaries. | Not automated; manual second-generation smoke proves mechanics. |
| Online Elo continuation | New checkpoints enter existing pool, get placement, then adaptive rounds continue from `latest.json`. | Local tournament-job continuation, queue-loss repair, and stale-claim repair now have focused tests, but no production-scale remote smoke yet. |
| Public active status | Rows become training-eligible only with enough games/opponents/context. | Active/provisional exists in rating rows. Publisher now refuses training-facing snapshots with no active rows unless diagnostic-only output is explicit. |
| One-frame tournament parity | Official leaderboard should match current one-frame train semantics. | Publisher now fails closed unless `rating_spec.decision_source_frames == 1` or a diagnostic/legacy flag is explicit. Remote public-source smoke still needed. |
| Non-checkpoint tournament players | Scripted/hand-coded players should eventually be representable if we want them ranked. | Current tournament/rating player specs require `checkpoint_ref`; policy loading goes through checkpoint loading only. |

## Current Architecture Boundary

Training should not poll Modal Dict or tournament state inside `train_muzero`.

Correct flow:

```text
leaderboard snapshot -> stable_slots_v1 materializer -> immutable assignment.json -> trainer launch config
```

Incorrect flow:

```text
trainer step loop -> live Elo / Modal Dict / tournament API
```

Concrete ownership:

- Coach writes exact checkpoint files and consumes immutable assignments.
- The tournament job discovers exact checkpoint refs, rates checkpoints, and publishes
  public leaderboard snapshots.
- Coach decides if those snapshots are good enough to create/refresh a training
  assignment.

Checkpoint discovery must use `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
timestamped `lightzero_exp_*` folders are expected.

## Current Decision

Before launching long-running leaderboard-fed training:

1. Remote-smoke repair/fallback tooling for stale or missing leaderboard Dict
   pointers.
2. Promote `stable_slots_v1` and the assignment writer/operator flow from local
   helper/smoke path to documented production runbook.
3. Remote-smoke tournament one-frame evaluator semantics and public publish.
4. Add safe refresh/continuation policy for long-running trainers.
5. Prove online Elo continuation and queue/dedupe repair remotely at bounded scale.
6. Rerun the bounded closed-loop smoke after the checkpoint-recency metadata fix.
7. Only then launch leaderboard-derived long-running training.

Continuation note:

- Local tests now protect the two critical rules: run-id scans are live but
  explicit checkpoint refs are frozen, and existing rating runs require
  `continue_from_latest=True`.
- Local tests also cover missing-queue-event repair from manifest queued refs
  and stale claim repair.
- Local tests now protect the active-pool rule: keep the full known checkpoint
  history, mark mature tail rows as `retired`, exclude retired rows from future
  scheduling, and keep unplaced tail rows `provisional` so new entrants can
  still get placement games.
- The remaining work is production-scale proof and operator runbooks, not the
  basic local contract.

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
