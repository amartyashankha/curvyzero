# Refresh Canary Plan - 2026-05-14

Purpose: prove a running trainer can pick up a new immutable opponent
assignment through the refresh pointer schema
`curvyzero_opponent_assignment_refresh_pointer/v0`.

This is intentionally a one-run canary. It does not launch Modal work by
itself; the command block below is the reviewable launch path.

## Chosen A/B Assignments

Use two already materialized champ18a immutable assignments:

```bash
ASSIGN_A=training/lightzero-curvytron-visual-survival/curvy-champ18a-assignments/attempts/try-champ18a-assignments/opponents/assignments/curvy-champion-canary18-20260514a-blank5-wall5-rank2_25-rank1_65/assignment.json
SHA_A=69dc079401e5dcbded01b165170e8ab5d0a76e8488ce78c2b92406456e36c5f1

ASSIGN_B=training/lightzero-curvytron-visual-survival/curvy-champ18a-assignments/attempts/try-champ18a-assignments/opponents/assignments/curvy-champion-canary18-20260514a-blank10-wall5-rank4_10-rank3_20-rank2_20-rank1_35/assignment.json
SHA_B=a014400f6e1ca7bc62abedfc3cc871bb84aed244461a20debdde62c20f3807b5
```

Assignment A starts the trainer. The refresh pointer initially points to A, so
the first refresh check should log `decision=unchanged`. After that event is
visible, overwrite the same pointer file to B before the next refresh bucket.
That proves the running process reloaded the pointer, not just a static
assignment ref.

## Minimal Command Path

Set shared names:

```bash
D=artifacts/local/refresh_canary_20260514
RUN=refresh-canary-20260514a
ATT=try-refresh-canary-20260514a
PTR=training/lightzero-curvytron-visual-survival/refresh-canary-pointers/refresh-canary-20260514a/pointer.json
TRAIN_REF=training/lightzero-curvytron-visual-survival/$RUN/attempts/$ATT/train
mkdir -p "$D"
```

Verify both immutable assignments are readable from the training Volume:

```bash
uv run --extra modal modal volume get --force curvyzero-runs "$ASSIGN_A" "$D/assignment_A.json"
uv run --extra modal modal volume get --force curvyzero-runs "$ASSIGN_B" "$D/assignment_B.json"
```

Prime the pointer to assignment A:

```bash
jq -n --arg ref "$ASSIGN_A" --arg sha "$SHA_A" \
  '{schema_id:"curvyzero_opponent_assignment_refresh_pointer/v0", assignment_ref:$ref, assignment_sha256:$sha}' \
  > "$D/pointer_A.json"

uv run --extra modal modal volume put --force curvyzero-runs "$D/pointer_A.json" "$PTR"
```

Launch the short canary only after the main thread approves this command:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode train \
  --compute cpu \
  --run-id "$RUN" \
  --attempt-id "$ATT" \
  --env-variant source_state_fixed_opponent \
  --reward-variant sparse_outcome \
  --opponent-assignment-ref "$ASSIGN_A" \
  --opponent-assignment-refresh-interval-train-iter 4 \
  --opponent-assignment-refresh-ref "$PTR" \
  --max-env-step 8192 \
  --max-train-iter 32 \
  --source-max-steps 1024 \
  --collector-env-num 1 \
  --n-episode 1 \
  --evaluator-env-num 1 \
  --n-evaluator-episode 1 \
  --num-simulations 1 \
  --batch-size 8 \
  --lightzero-eval-freq 0 \
  --save-ckpt-after-iter 1 \
  --commit-on-checkpoint \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --output-detail compact
```

`save_ckpt_after_iter=1` plus `commit_on_checkpoint=true` is deliberate for
the canary: it makes the refresh JSONL visible from the Volume while the run is
still alive, so the operator can wait for the A/unchanged event before
overwriting the pointer to B.

Poll until the first refresh event exists and shows A was retained:

```bash
uv run --extra modal modal volume get --force curvyzero-runs \
  "$TRAIN_REF/opponent_assignment_refresh_events.jsonl" \
  "$D/opponent_assignment_refresh_events.before_B.jsonl"

jq -r '[.decision, .train_iter, .bucket, .assignment_sha256, .refresh_pointer.assignment_ref] | @tsv' \
  "$D/opponent_assignment_refresh_events.before_B.jsonl"
```

Expected first proof point: `decision` is `unchanged`, `assignment_sha256` is
`$SHA_A`, and `refresh_pointer.assignment_ref` is `$ASSIGN_A`.

After that first event is visible, overwrite the same pointer to assignment B:

```bash
jq -n --arg ref "$ASSIGN_B" --arg sha "$SHA_B" \
  '{schema_id:"curvyzero_opponent_assignment_refresh_pointer/v0", assignment_ref:$ref, assignment_sha256:$sha}' \
  > "$D/pointer_B.json"

uv run --extra modal modal volume put --force curvyzero-runs "$D/pointer_B.json" "$PTR"
```

Then fetch the canary telemetry:

```bash
uv run --extra modal modal volume get --force curvyzero-runs \
  "$TRAIN_REF/opponent_assignment_refresh_events.jsonl" \
  "$D/opponent_assignment_refresh_events.jsonl"

uv run --extra modal modal volume get --force curvyzero-runs \
  "$TRAIN_REF/env_steps.jsonl" \
  "$D/env_steps.jsonl"

uv run --extra modal modal volume get --force curvyzero-runs \
  "$TRAIN_REF/summary.json" \
  "$D/summary.json"
```

Success checks:

```bash
jq -r 'select(.decision=="applied") |
  [.train_iter, .bucket, .refresh_index, .assignment_sha256,
   .refresh_pointer.assignment_ref, .refresh_pointer.resolved_assignment_sha256] | @tsv' \
  "$D/opponent_assignment_refresh_events.jsonl"

jq -r --arg sha "$SHA_B" 'select(.opponent_assignment_sha256==$sha) |
  [.env_id, .opponent_assignment_ref, .opponent_assignment_sha256,
   .opponent_assignment_refresh_index, .opponent_provider_load_ok] | @tsv' \
  "$D/env_steps.jsonl" | head

jq '.opponent_assignment_refresh' "$D/summary.json"
```

The canary passes only if:

- `opponent_assignment_refresh_events.jsonl` has an `applied` event for
  assignment B.
- That event includes the pointer metadata and
  `resolved_assignment_sha256 == $SHA_B`.
- Later `env_steps.jsonl` rows show `opponent_assignment_sha256 == $SHA_B` and
  `opponent_assignment_refresh_index == 1`.
- `summary.json.opponent_assignment_refresh.event_count` is nonzero and its
  latest/events tail includes the applied B pointer.

## One-Row Manifest Alternative

If the operator wants to launch through the grouped manifest submitter instead
of `modal run`, make a one-row canary manifest derived from the champ18a
manifest and edit only that row's `train_kwargs`:

```json
{
  "opponent_assignment_ref": "<ASSIGN_A>",
  "opponent_assignment_refresh_interval_train_iter": 4,
  "opponent_assignment_refresh_ref": "<PTR>",
  "max_env_step": 8192,
  "max_train_iter": 32,
  "source_max_steps": 1024,
  "collector_env_num": 1,
  "n_episode": 1,
  "num_simulations": 1,
  "batch_size": 8,
  "save_ckpt_after_iter": 1,
  "commit_on_checkpoint": true,
  "background_eval_enabled": false,
  "background_gif_enabled": false
}
```

Keep `poller_kwargs.opponent_assignment_ref` on A if a poller is used, but the
safest canary disables background eval/GIF and uses the direct trainer command
above. The submitter writes only assignment refs selected by
`train_kwargs.opponent_assignment_ref`, so the refresh target B must already
exist on the Volume or be written separately before pointer update.

## Blockers And Gaps

- The live 18-row champ18a batch has
  `assignment_refresh_interval_train_iter=0` and no refresh ref, so it cannot
  prove running-trainer refresh.
- The current trainer path supports direct pointer refresh, not the final
  run-control/`ready.json` schema. This canary proves the resolver/hook/reset
  mechanics only.
- The pointer file is mutable by design. The assignments remain immutable; the
  proof is that the running trainer follows a changed pointer to a different
  immutable assignment at a collect boundary.
- If the trainer finishes before the pointer is changed to B, rerun with a
  larger `max_train_iter` or a larger interval. Do not count a pointer that was
  B before the first collect as refresh proof.
- A `failed_after_reset_attempt` event means the split-process collector reset
  path is not safe enough yet; do not collect that batch as successful proof.
- The grouped submitter has no first-class pointer publication/validation path.
  It passes extra train kwargs through, but it does not manage the refresh
  pointer or ensure refresh-target assignments are present.
