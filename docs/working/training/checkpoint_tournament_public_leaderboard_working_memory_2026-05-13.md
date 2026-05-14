# Checkpoint Tournament Public Leaderboard Working Memory, 2026-05-13

## Plain Goal

The public leaderboard is not the same thing as the current tournament website.

The website is for humans to inspect rankings, battles, and GIFs. The public
leaderboard is a future trainer/coach contract: it should publish a trusted pool
of opponent candidates that a separate selector can turn into a frozen opponent
assignment for a training run.

## Current Boundary

- Current tournament runs are evidence and plumbing tests, not the final public
  leaderboard.
- Some current policies may be weak or trained under odd settings. Do not treat
  them as automatically useful frozen opponents.
- A future leaderboard may need to start clean for a new training line and may
  be seeded with hard-coded/scripted baselines, champions, or other anchors.
- The trainer should not poll live Elo, Modal Dict, or tournament state during
  training. It should consume an immutable assignment snapshot.

## Interface Shape

Use the coach-side design already documented in:

- `docs/working/training/lightzero_train_refactor_2026-05-13/opponent_leaderboard_interface.md`
- `docs/working/training/lightzero_train_refactor_2026-05-13/opponent_registry_design.md`
- `src/curvyzero/training/opponent_registry.py`

Existing names:

- public leaderboard snapshot:
  `curvyzero_opponent_leaderboard_snapshot/v0` (documented, not yet
  implemented as code validators/builders);
- live pointer:
  `curvyzero_opponent_leaderboard_pointer/v0` (documented, small Dict pointer
  only);
- trainer assignment:
  `curvyzero_opponent_assignment/v0` (implemented in
  `src/curvyzero/training/opponent_registry.py`);
- existing trainer mixture:
  `curvyzero_episode_opponent_mixture/v0`.

V0 storage:

- Durable truth:
  `tournaments/curvytron/leaderboards/<leaderboard_id>/snapshots/<snapshot_id>.json`
- Convenience pointer:
  `tournaments/curvytron/leaderboards/<leaderboard_id>/latest.json`
- Optional live/provisional pointer:
  `tournaments/curvytron/leaderboards/<leaderboard_id>/provisional_latest.json`
- Modal Dict:
  `curvyzero-curvytron-opponent-leaderboard-live`, key
  `current:<leaderboard_id>`, containing only a small pointer to the Volume
  snapshot.

Do not put full leaderboard rows in Modal Dict. Dict is a cache/pointer, not the
audit trail.

Do not put public leaderboard rows directly into trainer picks. A leaderboard
row is a rich evidence record. A trainer pick is an assignment entry: `name`,
positive `weight`, `opponent_policy_kind`, and either an immutable
`opponent_checkpoint_ref` or validated scripted-policy settings.

Recommended V0: `assignment.json` plus `audit.json`. The assignment remains
tiny and trainer-ready. The audit carries source snapshot hash, selector
strategy, row hashes, buckets/fallbacks, file facts, and rating context.

Current gap: `--opponent-assignment-ref` is documented but not wired into the
trainer launch path yet. The trainer today consumes resolved
`opponent_mixture_spec`.

## Row Data Needed

Each public row should carry enough information for a selector and for a human
to know whether it is safe to use:

- checkpoint id, exact immutable checkpoint ref, label, run id, attempt id,
  iteration, and experiment directory name;
- rank, rating, status, previous rating or last delta;
- valid games, wins, losses, draws, battle count, distinct opponents, failure
  count/rate, draw/timeout rate if available;
- tournament id, rating run id, round id, formula version, policy mode,
  evaluator context hash, roster/pool hash;
- freshness and provenance, including checkpoint mtime/size and broad discovery
  source.

Checkpoint discovery must scan `train/lightzero_exp*/ckpt/iteration_*.pth.tar`.
Do not publish rows from fixed `train/lightzero_exp/ckpt` discovery alone.

## What Is Not Safe Yet

- Do not expose live/provisional Elo as trusted best policy.
- Do not let the trainer sample from raw rating `latest.json` directly.
- Do not expose confidence intervals unless they are actually computed.
- Do not claim outside-lineage strength until outside-lineage evidence exists.
- Do not mix incompatible environment, reward, render, policy-mode, or evaluator
  contexts in one default leaderboard.

## Small Implementation Steps

1. Add pure leaderboard snapshot and pointer validators.
2. Add a publisher that converts a verified rating snapshot into an immutable
   public leaderboard snapshot.
3. Write `latest.json` plus a small Modal Dict pointer after the snapshot is
   durable.
4. Add a selector that writes a frozen opponent assignment from one verified
   snapshot.
5. Later, thread `--opponent-assignment-ref` into the trainer launch path and
   record the consumed assignment ref/hash.

## Tests To Add

- Reject mutable checkpoint refs such as `latest.pth.tar` and `ckpt_best.pth.tar`.
- Snapshot hash is stable and Dict pointer hash mismatch blocks or repairs.
- Active rows require the intended distinct-opponent evidence floor.
- Publisher uses broad checkpoint discovery provenance.
- Selector is deterministic and excludes provisional rows by default.
- Trainer resolves assignment once and does not read live tournament state.
