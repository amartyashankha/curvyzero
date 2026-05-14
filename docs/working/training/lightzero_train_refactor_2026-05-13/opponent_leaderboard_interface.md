# Opponent Leaderboard Interface

Date: 2026-05-13

## Recommendation

Use a hybrid interface:

1. Tournament reducers publish immutable leaderboard JSON snapshots on the
   tournament Volume.
2. A Modal Dict stores only the live pointer/cache for the current leaderboard
   snapshot.
3. A selection controller turns one leaderboard snapshot into an immutable
   opponent assignment snapshot under the training attempt.
4. The trainer consumes that frozen assignment at launch or explicit refresh
   boundaries and passes the resolved mixture into the existing environment
   config.

Do not make `train_muzero` rank policies, mutate the leaderboard, or poll a
live Dict during training. That keeps the learning loop close to stock
LightZero: CurvyZero chooses config and opponent plumbing before LightZero owns
collector, replay, search, and learner.

## Plain Goal

Training should be able to say: "use this frozen set of opponents, chosen from
the public tournament leaderboard, and record exactly what I used."

Current tournament runs are not automatically the public leaderboard. They are
mostly plumbing, rating, and inspection evidence. A new training line may need a
new clean leaderboard, and that leaderboard may be seeded with scripted or
hard-coded anchor policies before neural checkpoints are mature enough.

Tournament and scheduling code can be clever. The trainer should be boring. It
should receive concrete opponent refs, validate that frozen checkpoints are
exact `iteration_N.pth.tar` files, and start training with a stable opponent
mixture.

Recent checkpoints should be strongly represented because they are the most
useful pressure for the current training frontier, but they should not erase
anchors, strong established policies, or low/easy sentinels.

## Terms

| Term | Meaning |
| --- | --- |
| Leaderboard snapshot | Immutable public tournament output: rows, ranks, evidence, source tournament refs, and context hashes. |
| Live pointer | Small Modal Dict value that says which leaderboard snapshot is current. It is a cache, not truth. |
| Selection strategy | Deterministic rule for choosing opponents from one leaderboard snapshot. |
| Assignment snapshot | Immutable per-training-attempt JSON containing the exact opponent mixture consumed by training. |
| Refresh boundary | A launch, resume, or explicit attempt boundary where a new assignment may be created. |
| Active row | Leaderboard row with enough evidence for default training use. Current target: at least 20 distinct opponents. |
| Provisional row | Useful candidate, but not yet mature enough to sample unless the strategy explicitly allows it. |
| Anchor | Frozen baseline, champion, median, collapse sentinel, or historical checkpoint kept for continuity. |

## Storage Model

Tournament Volume, durable public ledger:

```text
tournaments/curvytron/leaderboards/<leaderboard_id>/snapshots/<snapshot_id>.json
tournaments/curvytron/leaderboards/<leaderboard_id>/latest.json
tournaments/curvytron/leaderboards/<leaderboard_id>/provisional_latest.json
```

Training Volume, immutable per-attempt record:

```text
training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/opponents/assignments/<assignment_id>/assignment.json
training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/opponents/assignments/<assignment_id>/audit.json
```

`assignment.json` is the small trainer-consumed contract. `audit.json` is the
richer provenance record for humans and reproducibility.

The attempt `command.json` or equivalent launch metadata should record:

```json
{
  "opponent_assignment_ref": "training/.../opponents/assignments/<assignment_id>/assignment.json",
  "opponent_assignment_sha256": "<canonical-json-sha256>",
  "opponent_assignment_audit_ref": "training/.../opponents/assignments/<assignment_id>/audit.json"
}
```

## Modal Storage Critique

| Option | Good at | Bad at | Verdict |
| --- | --- | --- | --- |
| Modal Dict only | Fast current pointer, leases, claims, small live state. The tournament app already uses this pattern for `checkpoint_intake_state`. | Dict entries expire after 7 days of inactivity, reads/writes are network calls, values should stay small, and it is a poor audit trail. | Do not use as long-term truth. |
| Volume JSON only | Durable, inspectable, hashable, easy to copy into run metadata, natural fit for existing tournament artifacts. | Readers need commit/reload discipline; `latest.json` is a same-file write path; listing large trees can be slow. | Good durable source, slightly clunky live pointer. |
| Hybrid append-only snapshots | Immutable snapshots are truth; Dict points to the newest snapshot; `latest.json` is a convenience pointer for repair and non-Dict readers. | Needs one publisher path, generation checks, and hash validation. | Recommended. |

Official Modal docs matter here:

- Modal Dict entries expire after 7 days without reads or writes, so a Dict must
  not be the only long-term source of truth:
  https://modal.com/docs/reference/modal.Dict
- Modal recommends Dicts for small objects, with network latency and size limits
  to keep in mind:
  https://modal.com/docs/guide/dicts
- Modal Volumes require explicit commit/reload, and concurrent writes to the
  same file are last-writer-wins:
  https://modal.com/docs/reference/modal.Volume
- Volumes v2 support many distinct-file writes, but single-file writes still
  need a single writer:
  https://modal.com/docs/guide/volumes
- A scheduled publisher/repair job can use Modal Period or Cron, but that job
  only republishes pointers from Volume truth:
  https://modal.com/docs/guide/cron

## Data Contracts

### 1. Leaderboard Snapshot

Path:

```text
tournaments/curvytron/leaderboards/<leaderboard_id>/snapshots/<snapshot_id>.json
```

Schema:

```json
{
  "schema_id": "curvyzero_opponent_leaderboard_snapshot/v0",
  "leaderboard_id": "curvytron-main-eval",
  "snapshot_id": "20260513T231500Z-000042",
  "generation": 42,
  "created_at": "2026-05-13T23:15:00Z",
  "source": {
    "kind": "checkpoint_tournament_rating",
    "tournament_id": "arena-curvytron-allckpt-adaptive-...",
    "rating_run_id": "elo-allckpt-adaptive-...",
    "rating_snapshot_ref": "tournaments/curvytron/.../ratings/.../latest.json",
    "rating_snapshot_sha256": "<sha256>"
  },
  "context": {
    "policy_mode": "eval",
    "rating_formula_version": "batch_elo_v0",
    "checkpoint_roster_hash": "<hash>",
    "rating_context_hash": "<hash>",
    "pool_status": "provisional|active|retired"
  },
  "maturity_policy": {
    "active_min_distinct_opponents": 20,
    "active_min_valid_games": 300,
    "max_failure_rate": 0.02,
    "allow_provisional_for_recent_bucket": true
  },
  "rows": [
    {
      "checkpoint_id": "survivaldiag-v1b-20260513h-r042-i120000",
      "checkpoint_ref": "training/.../attempts/.../train/lightzero_exp_260513_.../ckpt/iteration_120000.pth.tar",
      "run_id": "survivaldiag-v1b-20260513h-r042",
      "attempt_id": "attempt_001",
      "iteration": 120000,
      "label": "survivaldiag-v1b-r042 i120000",
      "rank": 7,
      "rating": 1568.25,
      "rating_delta": 12.5,
      "status": "active",
      "eligibility": {
        "eligible_for_training_default": true,
        "reasons": []
      },
      "evidence": {
        "valid_games": 420,
        "battle_count": 20,
        "distinct_opponents": 20,
        "outside_lineage_opponents": 18,
        "failure_rate": 0.0,
        "draw_rate": 0.12,
        "timeout_rate": 0.03,
        "last_rated_at": "2026-05-13T23:10:00Z"
      },
      "recency": {
        "checkpoint_mtime_ns": 1778710500000000000,
        "latest_for_run": true,
        "age_seconds_at_snapshot": 1800
      },
      "source_refs": {
        "checkpoint_battle_index_ref": "tournaments/curvytron/.../checkpoints/<checkpoint_id>/battle_index.json"
      }
    }
  ],
  "snapshot_sha256": "<canonical-json-sha256-with-this-field-empty>"
}
```

Rules:

- `checkpoint_ref` must be Volume-relative, never an absolute mount path.
- Frozen checkpoint refs must point at exact `iteration_N.pth.tar` files.
- Rows with too little evidence may appear, but must be `provisional` and must
  not look equivalent to active rows.
- The snapshot is append-only. Never edit it after publishing.

### 2. Modal Dict Live Pointer

Dict name:

```text
curvyzero-curvytron-opponent-leaderboard-live
```

Key:

```text
current:<leaderboard_id>
```

Value:

```json
{
  "schema_id": "curvyzero_opponent_leaderboard_pointer/v0",
  "leaderboard_id": "curvytron-main-eval",
  "generation": 42,
  "snapshot_id": "20260513T231500Z-000042",
  "snapshot_ref": "tournaments/curvytron/leaderboards/curvytron-main-eval/snapshots/20260513T231500Z-000042.json",
  "snapshot_sha256": "<sha256>",
  "published_at": "2026-05-13T23:15:05Z",
  "writer": {
    "kind": "tournament_leaderboard_publisher",
    "modal_call_id": "fc-..."
  },
  "compact_summary": {
    "row_count": 424,
    "active_count": 80,
    "provisional_count": 344,
    "top_checkpoint_ids": ["...", "..."]
  }
}
```

Rules:

- Store primitive JSON-like values only.
- Keep the value small. A compact top-K cache is fine; the full rows live in
  the Volume snapshot.
- Every reader must verify `snapshot_ref` exists and matches
  `snapshot_sha256`. If verification fails, fall back to Volume `latest.json` or
  the newest snapshot.
- Because Dict entries expire after inactivity, a missing Dict key is a cache
  miss, not data loss.

### 3. Trainer-Consumed Assignment Snapshot

Path:

```text
training/.../<run_id>/attempts/<attempt_id>/opponents/assignments/<assignment_id>/assignment.json
```

Schema stays compatible with the current pure parser
`parse_opponent_assignment_snapshot`:

```json
{
  "schema_id": "curvyzero_opponent_assignment/v0",
  "assignment_id": "run123-attempt001-refresh000-5d8e4b7c",
  "source_epoch": 42,
  "source_ref": "tournaments/curvytron/leaderboards/curvytron-main-eval/snapshots/20260513T231500Z-000042.json",
  "created_at": "2026-05-13T23:16:00Z",
  "seed": 913713,
  "entries": [
    {
      "name": "recent_active_001",
      "weight": 18,
      "age_label": "recent",
      "tags": ["leaderboard", "recent", "active"],
      "opponent_policy_kind": "frozen_lightzero_checkpoint",
      "opponent_checkpoint_ref": "training/.../iteration_120000.pth.tar"
    },
    {
      "name": "champion_anchor",
      "weight": 10,
      "tags": ["leaderboard", "anchor", "champion"],
      "opponent_policy_kind": "frozen_lightzero_checkpoint",
      "opponent_checkpoint_ref": "training/.../iteration_180000.pth.tar"
    },
    {
      "name": "blank_canvas",
      "weight": 2,
      "tags": ["scripted", "sentinel"],
      "opponent_policy_kind": "fixed_straight",
      "opponent_runtime_mode": "blank_canvas_noop"
    }
  ]
}
```

Rules:

- This file is what the trainer reads.
- It contains only the resolved mixture, not ranking logic.
- `entries[*].opponent_checkpoint_ref` must be immutable and exact.
- Resuming a training run reuses this exact file unless the operator creates a
  new assignment at a refresh boundary.

### 4. Assignment Audit Snapshot

Path:

```text
training/.../<run_id>/attempts/<attempt_id>/opponents/assignments/<assignment_id>/audit.json
```

Schema:

```json
{
  "schema_id": "curvyzero_opponent_assignment_audit/v0",
  "assignment_id": "run123-attempt001-refresh000-5d8e4b7c",
  "assignment_ref": "training/.../assignment.json",
  "assignment_sha256": "<sha256>",
  "training": {
    "run_id": "run123",
    "attempt_id": "attempt001",
    "refresh_index": 0,
    "created_for": "launch|resume_refresh"
  },
  "source_leaderboard": {
    "leaderboard_id": "curvytron-main-eval",
    "generation": 42,
    "snapshot_ref": "tournaments/curvytron/leaderboards/.../snapshots/20260513T231500Z-000042.json",
    "snapshot_sha256": "<sha256>",
    "dict_pointer_generation": 42
  },
  "selection": {
    "strategy_id": "recent_active_anchor_mix_v0",
    "strategy_version": 0,
    "seed": 913713,
    "selector_git_sha": "<optional>",
    "weights": {
      "recent_active": 0.45,
      "strong_active": 0.25,
      "near_rating_or_diverse": 0.15,
      "anchors": 0.10,
      "scripted_sentinels": 0.05
    },
    "filters": {
      "default_requires_status": ["active"],
      "allow_recent_provisional": true,
      "recent_max_age_seconds": 172800,
      "min_distinct_opponents_for_active": 20
    }
  },
  "selected_rows": [
    {
      "entry_name": "recent_active_001",
      "checkpoint_id": "survivaldiag-v1b-r042-i120000",
      "bucket": "recent_active",
      "leaderboard_rank": 7,
      "leaderboard_status": "active",
      "row_sha256": "<sha256>",
      "checkpoint_ref": "training/.../iteration_120000.pth.tar",
      "file_summary": {
        "size_bytes": 123456789,
        "mtime_ns": 1778710500000000000,
        "sha256": "<optional-expensive-checkpoint-hash>"
      }
    }
  ]
}
```

Rules:

- `audit.json` can evolve faster than the trainer payload.
- It carries the strategy, row hashes, file facts, and source snapshot hash.
- It is written once and never updated.

## Selection Strategy

Default strategy: `recent_active_anchor_mix_v0`.

Inputs:

- one verified leaderboard snapshot;
- training run id and attempt id;
- deterministic seed;
- optional current-run lineage metadata;
- target entry count or total weight.

Default bucket mix:

| Bucket | Share | Intent |
| --- | ---: | --- |
| `recent_active` | 45% | Keep newest useful checkpoints strongly represented. |
| `strong_active` | 25% | Pressure the learner against the current top policies. |
| `near_rating_or_diverse` | 15% | Avoid overfitting to only champions or only recency. |
| `anchors` | 10% | Preserve continuity across leaderboard generations. |
| `scripted_sentinels` | 5% | Keep simple behavior regressions visible. |

Eligibility:

- Default sampling uses `status=active`.
- Recent provisional rows may be sampled only when they satisfy a lower explicit
  floor, for example at least 5 distinct opponents and no systemic failures.
- Rows with missing checkpoint files, mutable refs, incompatible
  `rating_context_hash`, or high failure rates are excluded.
- Prefer outside-lineage opponents unless the strategy is explicitly a lineage
  diagnostic.

Determinism:

- Sort candidates by stable keys before random sampling.
- Use the assignment seed for all bucket draws.
- If a bucket is underfilled, record the fallback in `audit.json` and borrow
  from active anchors or strong active rows.

## Writer And Reader Ownership

| Component | Writes | Reads | Must not do |
| --- | --- | --- | --- |
| Tournament reducer/publisher | leaderboard snapshots, `latest.json`, Dict pointer | tournament rating artifacts | write training assignments |
| Selection controller | per-attempt `assignment.json` and `audit.json` | Dict pointer, leaderboard snapshot | mutate tournament ratings |
| Trainer scaffolding | consumed-assignment refs in attempt metadata | one `assignment.json` | rank, sample, or poll Dict inside `train_muzero` |
| Eval/GIF jobs | their own artifacts | same resolved assignment as training | silently choose different opponents |
| Website/status readers | nothing important | snapshots, pointer, assignments | become source of truth |

## Update Cadence

Leaderboard:

- Final snapshot after each rating reduction.
- Provisional snapshot at a bounded cadence while a large tournament is running,
  currently about every minute if the provisional writer is active.
- Dict pointer update only after the snapshot is written and committed.

Assignment:

- Created at training attempt launch.
- Reused for resume by default.
- Optional refresh only at explicit attempt/checkpoint boundaries. A refresh
  writes a new assignment id and records `refresh_index > 0`.
- No per-step or in-loop refresh.

Pointer repair:

- A scheduled publisher can periodically read Volume `latest.json` and republish
  the Dict pointer. This is convenience and keeps the pointer warm, but training
  correctness cannot depend on it because Dict entries expire after inactivity.

## Failure Modes

| Failure | Behavior |
| --- | --- |
| Dict key expired or missing | Treat as cache miss. Read Volume `latest.json` or newest immutable snapshot. |
| Dict pointer stale | Verify generation/hash against Volume. Prefer the newest valid committed snapshot. |
| Snapshot ref missing | Block assignment creation and mark the leaderboard unhealthy. Do not train from a guessed row set. |
| Same-file Volume write race | Avoid by single publisher. Immutable snapshot paths make duplicate writes obvious. |
| Volume reload fails because files are open | Retry outside hot paths. Selection is launch-time work, so slow reload is acceptable. |
| Provisional row looks strong after one opponent | Keep `status=provisional`; default strategy excludes it unless recent-provisional sampling is explicitly enabled. |
| Checkpoint ref is `latest`, `ckpt_best`, or non-iteration | Reject before assignment is written and again in trainer parsing. |
| Checkpoint file disappeared or changed | Assignment creation validates existence and file facts; audit records size/mtime/hash where available. |
| Strategy underfills a bucket | Borrow from active rows and record fallback reason in audit. |
| Leaderboard context changed | Start a new leaderboard id or require explicit bridge policy; do not mix incompatible ratings into one default assignment. |

## Reproducibility Story

To reproduce a training run's opponents:

1. Read the attempt metadata and find `opponent_assignment_ref`.
2. Verify `opponent_assignment_sha256`.
3. Read `audit.json` for the source leaderboard snapshot, strategy id, seed, and
   selected row hashes.
4. Verify every frozen checkpoint ref is an exact `iteration_N.pth.tar` file.
5. Use the assignment entries exactly as written.

The live Dict is irrelevant for replay. Even if the Dict entry has expired, the
training attempt still has its own immutable assignment copy.

## Staying Close To Stock LightZero

- Stock `lzero.entry.train_muzero` still owns the learning loop.
- CurvyZero resolves one opponent mixture before building the config.
- The environment/opponent provider executes the mixture during episodes, which
  is already the current boundary.
- No tournament imports, Modal Dict reads, leaderboard ranking, or policy
  sampling happen inside LightZero collector/search/learner code.
- A refresh is modeled as a new attempt or explicit resume boundary, not as a
  hidden mid-loop mutation.

## Testing Plan

Pure tests:

- leaderboard snapshot schema accepts active/provisional rows and rejects mutable
  checkpoint refs;
- Dict pointer payload validates `snapshot_ref`, generation, and hash;
- selector deterministically produces the same assignment for the same snapshot,
  strategy, and seed;
- selector gives recent checkpoints the requested share when eligible;
- selector excludes immature rows unless `allow_recent_provisional` is true;
- assignment parser keeps rejecting `latest`, `ckpt_best`, and non-iteration
  refs;
- assignment audit hash matches the trainer-consumed assignment JSON.

Trainer plumbing tests:

- trainer reads an assignment ref and passes the existing opponent-mixture
  contract into env config;
- resume reuses the prior assignment by default;
- explicit refresh writes and consumes a new assignment id;
- eval/GIF jobs receive the same resolved assignment metadata.

Modal smoke tests:

- publisher writes an immutable leaderboard snapshot, commits, updates Dict, and
  can recover after deleting the Dict key;
- selector reads through Dict when present and through Volume when Dict is
  missing;
- assignment creation writes both `assignment.json` and `audit.json` under a
  training attempt.

Do not add trainer-source behavior until these tests exist.

## Phased Implementation Plan

Phase 0, docs and contract:

- Adopt this interface as the contract.
- Keep `opponent_registry_design.md` as the narrower parser/trainer boundary
  note; this document owns the leaderboard and assignment ledger surface.

Phase 1, pure contracts:

- Add pure schema validators/builders for leaderboard snapshot, live pointer,
  and assignment audit.
- Keep Modal imports out of the pure module.
- Reuse the existing `curvyzero_opponent_assignment/v0` parser for the
  trainer-consumed assignment.

Phase 2, publisher:

- Add one tournament-side publisher that writes immutable leaderboard snapshots
  and a small `latest.json`.
- Update the existing Modal Dict pattern with a new named Dict for the current
  pointer.
- Add a repair command that rebuilds the Dict pointer from Volume snapshots.

Phase 3, selector:

- Add a pure selection controller for `recent_active_anchor_mix_v0`.
- Write `assignment.json` and `audit.json` under the training attempt.
- Add tests for deterministic sampling and recent-checkpoint representation.

Phase 4, trainer consumption:

- Add trainer plumbing tests first.
- Thread `--opponent-assignment-ref` through launch/config construction.
- Keep the trainer's job to parse, validate, record, and pass through the
  resolved mixture.

Phase 5, refresh boundaries:

- Add an operator-controlled refresh command that creates a new assignment for a
  resume/attempt boundary.
- Record refresh lineage in `audit.json`.
- Do not add in-loop polling unless there is a separate design and tests proving
  it does not break reproducibility.
