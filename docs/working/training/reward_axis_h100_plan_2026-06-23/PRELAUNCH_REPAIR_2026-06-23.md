# Prelaunch Repair - 2026-06-23

Status: non-launch repair pass. No training jobs were launched.

## Summary

The original exact-ref non-RND Wave A manifests remain ref-blocked because their
shared `curvy-n18conn-*` checkpoint refs are not visible in `curvyzero-runs-v2`.
That failure is recorded in `PRELAUNCH_AUDIT_2026-06-23.md`.

A repaired non-RND family now exists locally using currently visible
`curvy-r18fresh-*` checkpoint refs:

```text
artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.txt
```

These refs are believable as a curated static exact-ref repair source. They are
not a stable production leaderboard. The source was derived from the bounded
`r18fresh` rating snapshot for no-tournament static controls; that snapshot was
already documented as an exact frozen-ref source rather than launch-quality
leaderboard truth.

Important seed caveat: these repaired manifests use the top4nz rank1 sparse
`iteration_40000` checkpoint as their initial policy seed. The historical
best-known seed from the old r18fresh tournament snapshot is the plus-outcome
`iteration_180000` checkpoint recorded in `CHECKPOINT_ANCHOR_POLICY.md`. Launch
approval must decide whether the current repair seed is acceptable or whether
the manifests should be regenerated with the historical best-known seed.

## Source Ref Evidence

Refs-file Modal audit:

```text
artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.ref_audit.modal.json
```

Result:

- `ok=true`
- `ref_count=4`
- `missing_ref_count=0`
- `modal_parent_error_count=0`

Provenance notes:

- The refs are active nonzero mid-run checkpoints from the copied bounded
  `r18fresh` snapshot.
- Three of the four refs come from the same clean sparse ladder run; the fourth
  is the same sparse recipe with `so10`.
- The source is suitable for static no-refresh repair and matched controls, not
  for a claim that the leaderboard is now production-ready.

## Repaired Manifest Family

| Lane | Manifest family | Intended active rows | Gate result |
| --- | --- | ---: | --- |
| Static exact-ref reward isolate | `reward-static-top4nz-h100-wave-a-repair-20260623a` | 18 | syntax audit, Modal ref audit, and full submit dry-run pass |
| Long-horizon pretrained replicas | `reward-lhpre-top4nz-rep01` through `rep06` | 18 selected | syntax audit, Modal ref audit, and selected-row dry-runs pass |
| Cadence/support panel | `reward-csupport-top4nz-s25-b128-td25-cap1024`, `reward-csupport-top4nz-s25-b128-td25-cap2048`, `reward-csupport-top4nz-s25-b256-td25-cap2048` | 9 selected | syntax audit, Modal ref audit, and selected-row dry-runs pass |

All repaired manifests:

- use `opponent_source=mixture`
- use `assignment_refresh_interval_train_iter=0`
- use `compute=gpu-h100-cpu40`
- use exact immutable `iteration_N.pth.tar` checkpoint refs
- write zero assignments and zero refresh pointers in submitter dry-runs

## Artifact Checklist

Static repair:

```text
artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.json
artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.ref_audit.syntax.json
artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.ref_audit.modal.json
artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.submit.dryrun.json
```

Long-horizon repair pattern:

```text
artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a.json
artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a.ref_audit.modal.json
artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a.selected-r005-r011-r017.submit.dryrun.json
```

Cadence/support repair pattern:

```text
artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-*-wave-a-repair-20260623a/reward-csupport-top4nz-*-wave-a-repair-20260623a.json
artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-*-wave-a-repair-20260623a/reward-csupport-top4nz-*-wave-a-repair-20260623a.ref_audit.modal.json
artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-*-wave-a-repair-20260623a/reward-csupport-top4nz-*-wave-a-repair-20260623a.selected-r005-r011-r017.submit.dryrun.json
```

RND durable dry-run artifact:

```text
artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.submit.dryrun.json
```

## Current Launch Readiness

The repaired Wave A shape is locally ready for approval review:

- RND blank sweep: `45` rows, dry-run saved.
- Static non-RND repair: `18` rows, Modal ref audit and dry-run saved.
- Long-horizon non-RND repair: `18` selected rows, Modal ref audits and
  dry-runs saved.
- Cadence/support non-RND repair: `9` selected rows, Modal ref audits and
  dry-runs saved.

Total prepared rows: `90`, balanced between RND and non-RND lanes. Treat this
as a short-sweep launch menu unless the operator explicitly narrows it for a
longer horizon.

Packet audit:

```text
artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json
```

Result: `ok=true`, `actual_total_selected_rows=90`,
`expected_total_selected_rows=90`, `launch_artifacts=[]`, `error_count=0`.
The audit is produced by
`scripts/audit_curvytron_wave_a_launch_packet.py` and should be rerun before any
launch approval.

Capacity proxy:

```text
artifacts/local/curvytron_wave_a_capacity_snapshot_20260623a.json
```

Result: `ok=true`, CurvyTron train/status apps idle, but
`approval_recommendation=operator_capacity_review_required` because current
Modal task count plus `90` requested rows exceeds the coarse `100`-task
envelope proxy. Current conservative capacity proxy room is `22` additional
rows unless active tasks are classified as non-H100/non-conflicting. This must
be rerun before approval and read as capacity context, not as launch permission.

This is still not a launch. Remaining gates:

- explicit human approval for the exact launch command and row count
- fresh active H100 capacity check and capacity proxy review
- fresh checkpoint-anchor policy audit and seed decision
- runtime-tier choice: broad `<=2h`, at most 40 rows for `2h-8h`, and 10-20
  rows for `8h+`
- stage below the capacity proxy room or wait when active task type is ambiguous
- fresh packet audit pass with no launch artifacts
- launch note with run-id prefixes, status command, first health note path, and
  stop/cleanup procedure
- confirmation whether to launch the full repaired `90` rows or a staged
  subset

## Launch Command Shapes

RND full sweep:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.json --allow-launch --output artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.submit.launch.json
```

Static non-RND full repair:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.json --allow-launch --output artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.submit.launch.json
```

Long-horizon selected repair pattern:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py <reward-lhpre-top4nz-repNN-manifest.json> --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output <reward-lhpre-top4nz-repNN>.selected-r005-r011-r017.submit.launch.json
```

Cadence/support selected repair pattern:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py <reward-csupport-top4nz-manifest.json> --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output <reward-csupport-top4nz>.selected-r005-r011-r017.submit.launch.json
```

Do not run any of these launch commands without explicit approval.
