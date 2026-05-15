# Current State

For plain-language definitions of terms like online tournament continuation,
queue-loss repair, stale-claim repair, one-frame tournament validation,
assignment command path, and safe refresh, see `README.md`.

## Implemented Today

| Area | State | Evidence |
| --- | --- | --- |
| Tournament ratings | Implemented. Final ratings are written to the tournament Volume as `latest.json` plus per-round `ratings.json`. | `src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py::_write_rating_round_outputs` |
| Website/API reader | Implemented. `/api/rating-standings` reads final/provisional snapshots. | `curvytron_tournament_browser`, `_read_best_rating_snapshot_for_run` |
| Intake Dict/Queue | Implemented for V0 guarded batch intake. | `checkpoint_intake_state`, `checkpoint_intake_queue`, `curvytron_checkpoint_intake_*` |
| Scheduled intake ticks | Implemented: subscriber tick and drain tick exist on a 10s schedule. | `curvytron_checkpoint_intake_subscriber_tick`, `curvytron_checkpoint_intake_drain_tick` |
| Submit-only tournament service surface | Implemented locally: `intake-seed` configures policy; `tournament-submit` / `intake-submit` accepts candidate refs/run IDs without scheduler knobs. | `curvytron_checkpoint_intake_submit`, `src/curvyzero/tournament/checkpoint_intake_service.py`, focused tests |
| Intake manifest repair | Implemented locally: when a tournament/rating id is supplied, submit/status/tick/drain can rebuild a missing Dict manifest from the durable Volume manifest. | `_load_intake_manifest`, `test_intake_drain_rebuilds_missing_dict_manifest_from_volume` |
| Pool-scoped rating claims | Implemented locally: rating claims include claim mode and desired checkpoint-pool hash, so a fresh/small-pool claim does not block a later larger-pool continuation. | `_intake_rating_claim_key`, `test_expanded_pool_claim_is_not_blocked_by_partial_pool_claim`, `test_live_watch_drain_continues_past_partial_existing_rating_and_old_claim` |
| Broad checkpoint discovery | Implemented in tournament and trainer paths. | `train/lightzero_exp*/ckpt/iteration_*.pth.tar` |
| Rating context hash | Implemented for tournament evaluator/rating context. | `rating_context_hash` |
| Checkpoint roster guard | Implemented for rating evidence reuse. | `checkpoint_roster` in rating snapshots |
| Active tournament pool | Implemented locally: rating snapshots and public leaderboard snapshots default to a top-100 active pool. Mature rows below the cutoff are marked `retired`; provisional/new rows are not retired early; scheduler excludes retired rows but history remains in `latest.json`. | `DEFAULT_RATING_ACTIVE_POOL_LIMIT`, `rating_snapshot_from_pair_results`, `build_rating_round_pair_specs`, `build_leaderboard_snapshot_from_rating_snapshot`, focused tests |
| New-checkpoint placement | Implemented locally: fresh checkpoints prefer established active opponents before spending placement games on new-vs-new pairs. | `select_adaptive_v0_pair_slots`, `test_adaptive_v0_new_batch_gets_existing_opponents_in_first_round`, `test_adaptive_v0_gives_new_checkpoints_many_parallel_placement_pairs` |
| Opponent assignment parser | Implemented as pure parser. | `src/curvyzero/training/opponent_registry.py` |
| Training scripted/blank/passive opponent modes | Implemented as training opponent-mixture/env settings. | `opponent_mixture.py`, `curvyzero_source_state_visual_survival_lightzero_env.py` |
| Pure leaderboard-to-assignment bridge | Implemented locally: build/validate public leaderboard snapshots, live pointers, `top_slots_v0` smoke assignments, `stable_slots_v1` production-direction assignments, and assignment audits. `slot_rules_v0` is purged as the production path. | `src/curvyzero/training/opponent_leaderboard.py`, `tests/test_opponent_leaderboard.py`, `slot_architecture_feedback.md` |
| Local materialization CLI | Implemented locally: turns exported rating/API JSON into snapshot, pointer, `stable_slots_v1` assignment, and audit JSON artifacts by default. | `scripts/materialize_curvytron_leaderboard_assignment.py` |
| Checkpoint recency metadata | Locally repaired: tournament checkpoint normalization, explicit refs, intake manifest continuation, rating spec restoration, rating rows, public leaderboard rows, and `stable_slots_v1` now preserve the metadata needed for a real `recent_strong` slot. | `checkpoint_metadata_from_ref`, `_intake_manifest_rating_checkpoints`, `rating_roster_by_checkpoint`, focused metadata tests |
| Assignment artifact writer | Implemented and smoke-tested: stores `assignment.json` and optional `audit.json` under a training attempt. | `mode=write-assignment`, `_write_opponent_assignment_artifacts`, focused regression |
| Trainer assignment-ref plumbing | Implemented locally: train and poller command construction accept an assignment ref, resolve it through the existing mixture resolver, and propagate metadata. | `_resolve_opponent_assignment_for_env`, `_run_visual_survival_train`, `_checkpoint_eval_poller_command`, focused tests |
| Tournament-side leaderboard publisher | Implemented and remote-smoked: reads rating `latest.json`, writes public snapshot/latest refs, commits the Volume, then updates Dict pointer. Refuses provisional ratings unless explicitly allowed. | `curvytron_opponent_leaderboard_publish`, `test_opponent_leaderboard_publish_writes_snapshot_latest_and_pointer`, smoke `curvytron-latest212-smoke-20260513` |
| All-v2 closed-loop canary | Proven remotely through `stable_slots_v1`/assignment refresh: trainer assignment -> train checkpoints -> v2 discovery/intake -> v2 rating -> public leaderboard -> new assignment -> v2 control pointer -> same running trainer refresh -> provider-ok env rows. Caveat: the canary used relaxed/provisional gates and selected `iteration_0.pth.tar` as champion, so it proves wiring, not production-quality ranking or survival improvement. | `FULL_LOOP_PROOF.md`, `NOW.md`, `TODO.md` |

## Designed But Not Fully Wired

| Area | Desired state | Current gap |
| --- | --- | --- |
| Public leaderboard snapshot production policy | Immutable `curvyzero_opponent_leaderboard_snapshot/v0` written to Volume from rating snapshots. | Smoke path works; still need production naming/retention/repair policy. |
| Modal Dict live pointer repair | Small pointer to current leaderboard snapshot, not durable truth. | Local repair command exists, rebuilds `current:<leaderboard_id>` from validated immutable Volume snapshots, has a tiny remote smoke, and now has a minimal operator runbook. |
| Assignment writer flow | `stable_slots_v1` writes concrete assignment entries and audit locally; assignment writer stores assignment/audit under a training attempt. | Need one clear command path for writing a frozen `assignment.json` from a trusted leaderboard snapshot. |
| Run-scoped slot recipe control | Desired slot recipes should live in a Modal Dict keyed by training run id, then materialize to immutable assignment JSON. | Designed in `run_slot_control_design.md`; not implemented yet. |
| Safe assignment refresh | Long-running trainers should refresh opponents only at clean boundaries. | Direct pointer refresh is proven at all-v2 canary scale; run-control automation and production refresh policy remain launch-hardening work. |
| Run-scoped reward recipe control | Operators should be able to choose survival, bonus, and final-outcome reward weights per training run. | Designed in `run_reward_control_design.md`; not implemented as live run-control. Current code exposes named launch-time reward variants: `sparse_outcome`, `dense_survival_plus_outcome`, `survival_plus_bonus_no_outcome`, and `survival_plus_bonus_plus_outcome`. |
| Online tournament continuation | New checkpoints enter an existing tournament, get games, and ratings continue from `latest.json` instead of starting over. | Bounded Modal proof passed for a small pool on 2026-05-14. It proved continuation, queue-loss repair, stale-claim takeover, and one-frame round output. It is still not the final deployed always-on service proof. |
| Public active status | Rows become training-eligible only with enough games/opponents/context. | Active/provisional exists in rating rows. Publisher now refuses training-facing snapshots with no active rows unless diagnostic-only output is explicit. |
| One-frame tournament parity | Official leaderboard should match current one-frame train semantics. | Publisher now fails closed unless `rating_spec.decision_source_frames == 1` or a diagnostic/legacy flag is explicit. Tiny two-checkpoint remote rating/publish smoke passed; larger current-source validation still needed. |
| Non-checkpoint tournament players | Scripted/hand-coded players should eventually be representable if we want them ranked. | Current tournament/rating player specs require `checkpoint_ref`; policy loading goes through checkpoint loading only. |

## Current Architecture Boundary

Training should not poll tournament state inside `train_muzero`.

Correct flow:

```text
leaderboard snapshot -> stable_slots_v1 materializer -> immutable assignment.json -> trainer launch config
```

Incorrect flow:

```text
trainer step loop -> live Elo / Modal Dict / tournament API
```

Refined refresh rule:

- The trainer may do a bounded control-plane read from a run-control Modal Dict
  at a clean collection boundary.
- That Dict read is only for a prepared pending assignment pointer.
- The trainer must not read leaderboard rows, tournament state, slot recipes, or
  reward weights while learning.
- The first target cadence is about every 50 learner iterations, applied before
  a future `Collector.collect` call.

Concrete ownership:

- Coach writes exact checkpoint files and consumes immutable assignments.
- The tournament job discovers exact checkpoint refs, rates checkpoints, and publishes
  public leaderboard snapshots.
- Coach decides if those snapshots are good enough to create/refresh a training
  assignment.

Hard boundary:

- The trainer step loop must not poll live Elo, Modal Dict, Modal Queue,
  tournament browser routes, or leaderboard pointers.
- The trainer may read one frozen assignment at launch, resume, or another clean
  refresh boundary.
- The trainer may read one small run-control Dict pointer at that clean boundary,
  but only to find and verify a prepared immutable assignment.
- The trainer may read one frozen reward recipe at launch or a deliberate
  new-attempt boundary. It should not read live reward weights during learner
  updates.
- Tournament code may publish leaderboard snapshots and pointer caches, but it
  must not silently mutate a running trainer's opponents.

Checkpoint discovery must use `train/lightzero_exp*/ckpt/iteration_*.pth.tar`;
timestamped `lightzero_exp_*` folders are expected.

## Current Decision

Current focus:

1. Stop the current v2 real18 run because it is invalid enough to end as a
   training candidate, while preserving artifacts for diagnosis.
2. Keep the docs clear and truthful.
3. Keep the Coach/training boundary clean: trainer reads frozen assignment files,
   not live tournament state.
4. Do not launch new training until the restart blockers are tested:
   randomized learner seat/perspective, no-op/straight action checks,
   tournament eval parity, and cleanup of stale Modal app/artifact confusion.
5. Prove the online tournament path at bounded scale:
   - add new checkpoint candidates;
   - schedule many games for each new candidate against the active pool;
   - run those games in parallel on Modal;
   - keep old ratings/history;
   - recover if Queue or claim state is stale.

Restart manifest rule:

- Use `random_per_episode` learner seat/perspective handling.
- Public opponent specs use `opponent_immortal` for death immunity. Do not
  hand-author `opponent_death_mode` in public slot recipes, assignments, or
  manifests; the env-facing runtime field is derived at the trainer boundary.
- Include at least about `20%` blank/immortal pressure globally, with some
  higher-immortal variants.
- Keep enough leaderboard-checkpoint exposure for real policy learning.
- Do not repeat the previous weak `5%` wall-avoidant immortal recipes as the
  main restart plan.

Latest live-loop finding:

- Waited training smokes did run LightZero and wrote real checkpoints.
- The subscriber/intake manifest later saw 159 checkpoint players.
- The rating run that actually launched contained only 9 checkpoint players and
  finished 36 games.
- The current break is therefore before game scheduling: intake/continuation
  launched from a partial pool, then a fresh rating claim blocked the later
  full-pool continuation.
- Local repair now added: run-id/prefix scans are treated as live watches, live
  watches force continuation, and the drain uses a continuation claim/spec even
  when the original manifest default was false.
- Local regression now covers the exact shape: partial existing rating, later
  larger live-watch manifest, empty Queue, and an old claim. The drain must
  rebuild missing queue events and spawn a continuation over the full known pool.
- Do not call the automated loop proven until a live watch starts from a seeded
  pool, admits later checkpoints, and continues ratings using the full known
  `seen_checkpoint_refs` pool.
- Higher-level control-plane direction is documented in
  `tournament_control_invariants.md`: durable manifests define the desired
  player pool; Queue/Dict only wake or lease work; partial-pool ratings must not
  block a later full-pool rating.

Latest bounded online proof:

- Tournament/rating ids:
  `inspector-online-continuation-proof-20260514c /
  elo-oneframe-online-continuation-proof`.
- Started with 3 old checkpoint refs and wrote `round-000000`.
- Seeded intake, submitted 3 new checkpoint refs, then drained with stale-claim
  takeover enabled for the proof.
- The Queue was empty before the drain, so the drain rebuilt the missing events
  from the durable manifest.
- The spawned rating continued from `latest.json` and wrote `round-000001`.
- Final `latest.json` had 6 checkpoint rows, 9 scheduled pairs, 9 games, and
  `decision_source_frames=1`.
- Local follow-up fixed two bugs found during this proof:
  - `round_index=0` now continues to round 1, not round 0 again;
  - explicit continuation specs now preserve old checkpoint ids from
    `latest.json` when new refs are added.
- Local follow-up added two control-plane repairs from the Modal critique:
  - Dict manifest misses can be repaired from the Volume manifest when a
    tournament/rating id is supplied;
  - rating claims are keyed by claim mode and desired checkpoint pool, so an old
    fresh/small-pool claim does not block a later larger-pool continuation.

Remaining caveat:

- The proof used manual `modal run` commands and forced stale-claim takeover, so
  it proves the bounded data path but not the final always-on service shape.
  Next proof should use one deployed app, one active rating claim per
  `(tournament_id, rating_run_id)`, and fewer overlapping scheduled/manual
  writers.
- New Modal launch-lifetime finding: a non-detached `modal run` ephemeral app is
  not a safe parent for background tournament workers. We saw round
  input/progress written, then child game workers were killed when the local
  entrypoint/app stopped. Logs showed `RemoteError`, `KeyboardInterrupt`, and
  `Runner terminated`; the Volume had empty game dirs and no completed
  summaries. Anything that spawns child tournament workers and is expected to
  continue after the command returns must use `modal run --detach` or wait for
  the child work to finish.
- A separate top-100 tournament stress lane is now active. It should first rerate
  the top 100 active checkpoints from `curvytron-latest212-smoke-20260513` under
  current one-frame settings, then add recent checkpoints only after the base
  tournament is healthy.
- The tournament website still needs cleanup. One local bug is fixed: a failed
  Volume reload no longer throttles the next retry. Remaining work is fast shell,
  token refresh, paged panels, versioned GIF URLs, and no web-request fanout that
  writes progress files.

Separate trainer-refresh gap:

- Leaderboard promotion is not enough by itself.
- The trainer consumes `opponent_assignment_ref` at launch/config construction
  and turns it into an opponent mixture.
- A direct running-trainer refresh path now exists:
  `opponent_assignment_refresh_interval_train_iter` plus
  `opponent_assignment_refresh_ref`. It checks at a collect boundary, resolves
  the pending immutable assignment, resets every collector env with the new
  mixture/context, proves every env reports the new assignment, logs the event,
  then collects.
- This direct path is not yet the final run-control/`ready.json` path.
- Current best design is documented in `coarse_opponent_refresh_design.md`:
  Coach materializes a new immutable assignment, writes a pending-assignment
  pointer into run-control, and the trainer checks that pointer at a coarse
  collection boundary.
- The dangerous parts are split-brain subprocess env updates, partial
  Volume/Dict writes, multiple attempts racing on one run id, and stale loaded
  frozen opponent objects keyed only by slot name.
- The current preferred V1 shape is simple pause-and-reset at the start of a
  future `Collector.collect`: verify the immutable pending assignment, reset
  every collector env with the new mixture/context, prove fresh ready
  observations, then collect. If the pending assignment is bad before any env
  changes, keep the old assignment and log it. If worker state may be split, do
  not collect that batch.
- Local helper and trainer-level tests now cover the direct pause-and-reset
  path: retry on pre-reset pending-assignment failure, retry on missing hash,
  fail before collect on reset-proof failure, fail before training if the hook
  cannot be installed, write refresh JSONL, and include refresh summary data.
- Tiny Modal train proof `refresh-e2e-smoke-20260514/train-refresh-d` completed
  with `ok=true`. It launched with assignment A and pending assignment B, wrote
  one `applied` refresh event, proved both envs ready with assignment B, and
  telemetry switched from A to B after the initial LightZero random-collect
  startup rows.
- Until the run-control lane lands, full-loop testing can validate checkpoint
  production, intake, rating, leaderboard publish, assignment materialization,
  launch/resume consumption, and direct in-run assignment refresh, but not the
  final run-control/`ready.json` control plane.

Before launching long-running leaderboard-fed training:

1. Prove online tournament continuation and queue/claim repair on Modal.
2. Run a larger current-source one-frame tournament and public publish.
3. Rerun the bounded closed-loop smoke after the checkpoint-recency metadata fix.
4. Write one clear assignment command path for Coach.
5. Run a tiny direct-refresh trainer smoke with assignment A -> B and inspect
   refresh events/telemetry.
6. Only then launch leaderboard-derived long-running training.

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
- Local tests now protect the first-placement rule: a batch of new checkpoints
  gets established active opponents before new-vs-new placement pairs.
- Local tests now protect checkpoint id stability when an explicit continuation
  adds a ref that sorts before existing refs.
- The remaining work is production-scale proof and operator runbooks, not the
  basic local contract.

Remote proof update:

- A small deployed online continuation proof now passes when the drain is
  launched with `modal run --detach`.
- Proof id: `inspector-detached-online-proof-20260514b` /
  `elo-oneframe-detached-proof`.
- It completed baseline round 0 with three checkpoints, then accepted three new
  checkpoints through intake and completed round 1 with six checkpoints.
- Round 1 wrote 9/9 game summaries and advanced `latest.json` to round 1.
- This is not yet the big 100-player or 212-run proof. It is the narrow proof
  that the service can continue an existing rating run after new checkpoint
  intake without dropping the new pool.
- A public leaderboard publish from that completed snapshot also passed:
  `inspector-detached-proof-leaderboard-20260514b`, 6 rows, 6 active rows, Dict
  pointer published. The publish used tiny proof thresholds, so it proves the
  plumbing, not rating quality.

## Important Representation Split

Training already supports non-neural opponent concepts through mixture entries:

- `fixed_straight`;
- `proactive_wall_avoidant`;
- `blank_canvas_noop`;
- public `opponent_immortal=true` death immunity;
- frozen LightZero checkpoints.

The env-facing `opponent_death_mode=immortal` field is derived runtime
plumbing. Public slot recipes, assignments, and manifests should not hand-write
it.

The tournament/rating stack today is neural-checkpoint-centric:

- every normalized player requires a `checkpoint_ref`;
- `rating_roster_by_checkpoint` stores checkpoint refs and model/render metadata;
- game workers call checkpoint policy loading;
- resume-from-latest restores players from checkpoint refs.

So scripted/hand-coded policies can be **training opponents now**, but they are
not first-class **leaderboard players** yet.
