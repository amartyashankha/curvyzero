# Checkpoint Anchor Policy

Status: active policy. No Modal jobs were launched while writing this.

## Why This Exists

"Best known checkpoint" is not a single universal label. For this campaign,
checkpoint refs can play three different roles:

| Role | What "best" means | Default readout |
| --- | --- | --- |
| Initial policy seed | Strong starting policy with exact immutable ref and known load shape | Tournament/leaderboard champion, then Modal existence audit |
| Opponent curriculum ref | Diverse frozen opponents that exist now and are not moving aliases | Curated active nonzero refs, then Modal existence audit |
| Promotion candidate | New checkpoint worth preserving or tournament exposure | Eval AUC/best/retention plus later tournament exposure |

Do not use `latest`, `ckpt_best`, or a moving leaderboard pointer as the final
checkpoint identity for any of these roles.

## Current Best-Known Seed

The strongest historical seed anchor remains the r18fresh plus-outcome
checkpoint pinned in the old next-batch seeding doc:

```text
training/lightzero-curvytron-visual-survival/curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/attempts/try-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/train/lightzero_exp/ckpt/iteration_180000.pth.tar
```

Evidence:

- `docs/working/training/r18fresh_postmortem_2026-05-16/NEXT_BATCH_SEEDING.md`
  locked it as the shared rank-1 launch seed from snapshot
  `auto-r000032-g22-555c999b`.
- `docs/working/training/leaderboard_to_training_2026-05-13/tournament_elo_trajectory_2026-05-16.md`
  shows the same checkpoint climbing to rank 1 by round 30, with `2562` games
  and `118` distinct opponents.
- The same r18fresh family has strong sibling checkpoints, including
  `iteration_260000`, but the pinned launch seed is `iteration_180000`.

Caveat: before using this as the next actual seed, rerun a Modal existence audit
against `curvyzero-runs-v2`. Historical rank evidence is not a current
existence proof.

Current status: that audit has now passed for the prepared bestseed Wave A
family. The bestseed manifests use this historical ref as the learner seed and
still use `top4nz` refs only for opponent rank slots.

## Tooling Contract

The manifest builder now has two independent checkpoint knobs:

- `--checkpoint-refs-file`: selects frozen opponent rank-slot refs.
- `--initial-policy-checkpoint-ref`: selects the learner initial policy seed.

If `--initial-policy-checkpoint-ref` is omitted, non-scratch manifests fall
back to rank1 from the ratings snapshot or checkpoint refs file. For medium and
long learning runs, prefer the explicit seed flag with the historical r18fresh
`iteration_180000` ref unless the launch note explicitly chooses the top4nz
repair seed.

## Current Launchable Repair Anchor

The repaired Wave A non-RND manifests currently use the `top4nz` rank1 ref as
their initial policy seed:

```text
training/lightzero-curvytron-visual-survival/curvy-r18fresh-sparse-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5-cl-da1a498fd8/attempts/try-r18fresh-sparse-blank10-wall10-rank4_10-rank3_15-rank2_20-rank1_30-rank1imm5-clea-6132ae9835/train/lightzero_exp/ckpt/iteration_40000.pth.tar
```

The full `top4nz` refs file has four currently audited refs:

```text
artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.txt
```

Its Modal audit passes:

```text
artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.ref_audit.modal.json
```

Interpretation: `top4nz` is a believable currently launchable repair source and
opponent-curriculum source. It is not the global best-known seed by historical
tournament evidence. The slot called rank4 is really the fourth active nonzero
curated ref, not literal global rank 4.

## Current Audit Result

The anchor audit tool is:

```bash
uv run python scripts/audit_curvytron_checkpoint_anchor_policy.py --output artifacts/local/curvytron_checkpoint_anchor_policy_audit_20260623a.json
```

Saved result:

```text
artifacts/local/curvytron_checkpoint_anchor_policy_audit_20260623a.json
```

Current result:

- `ok=true`
- historical best seed: r18fresh plus-outcome `iteration_180000`
- `static_top4nz_ref_count=4`
- `static_top4nz_modal_audit.ok=true`
- repaired non-RND manifest count: `10`
- `historical_best_seed_manifest_count=0`
- `top4nz_seed_manifest_count=10`
- warning: repaired manifests do not use the historical r18fresh rank-1
  checkpoint as their initial seed

This warning is intentional. It forces an operator choice:

1. Launch the repaired top4nz-seeded manifests because they are currently
   audited and available.
2. Regenerate the non-RND manifests with the historical best-known seed, then
   rerun manifest, ref, packet, capacity, and staged-profile audits.

For long `8h+` runs, prefer option 2 unless there is a clear reason to value
current repair availability over the historically strongest seed.

## Bestseed Repair Result

The bestseed non-RND family has been regenerated with:

- opponent refs from
  `artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.txt`
- learner seed from the historical r18fresh `iteration_180000` checkpoint

Gates:

```bash
uv run python scripts/audit_curvytron_checkpoint_anchor_policy.py --non-rnd-seed-profile bestseed --require-best-known-seed --output artifacts/local/curvytron_checkpoint_anchor_policy_audit_bestseed_20260623a.json
uv run python scripts/audit_curvytron_wave_a_launch_packet.py --non-rnd-seed-profile bestseed --output artifacts/local/curvytron_wave_a_launch_packet_audit_bestseed_20260623a.json
```

Saved results:

- `artifacts/local/curvytron_checkpoint_anchor_policy_audit_bestseed_20260623a.json`
- `artifacts/local/curvytron_wave_a_launch_packet_audit_bestseed_20260623a.json`

Current result:

- anchor audit: `ok=true`, `historical_best_seed_manifest_count=10`,
  `top4nz_seed_manifest_count=0`
- packet audit: `ok=true`, `actual_total_selected_rows=90`, `error_count=0`
- every bestseed non-RND Modal ref audit has `ref_count=5`: four opponent refs
  plus the independent historical learner seed

## How To Decide "Best" Programmatically

Use the authority that matches the question:

- tournament champion: current rating or published leaderboard snapshot
- eval best: `lightzero_curvytron_run_status --output eval-json` plus
  `scripts/analyze_curvytron_eval_curves.py`
- launchability: `scripts/audit_curvytron_launch_manifest_refs.py` against the
  exact immutable refs
- current Wave A seed compliance:
  `scripts/audit_curvytron_checkpoint_anchor_policy.py`

There is no single source of truth yet that joins all of these. Until there is,
write the authority used next to every "best checkpoint" claim.

## Launch Gate

Before any medium or long learning run, record:

- initial policy seed ref
- why that ref is the right anchor for this run
- whether it is the historical best seed or the top4nz repair seed
- Modal existence audit result
- whether all rows in the lane share the same initial seed
- opponent refs file and audit result

If this information is missing, the launch can still be a short plumbing
preflight, but it should not be described as a best-known-checkpoint learning
run.
