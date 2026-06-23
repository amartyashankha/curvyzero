# Wave A Long17 Bestseed Approval Packet - 2026-06-23

Status: approval packet draft. No Modal training jobs were launched while
writing this note.

## Scope

Profile:

```text
artifacts/local/curvytron_wave_a_staged_launch_long17_no_highest_weight_bestseed_20260623a.json
```

Intent: conservative `8h+` long-read launch that fits the current capacity
proxy while preserving RND controls and independent non-RND reward/cadence
coverage.

Rows:

- total H100 rows: `17`
- RND rows: `8`
- non-RND rows: `9`
- seed profile: `bestseed`
- capacity snapshot:
  `artifacts/local/curvytron_wave_a_capacity_snapshot_long17_no_highest_weight_bestseed_20260623a.json`
- current capacity result: `capacity_proxy_clear`,
  `projected_total_tasks=100`

RND rows are `r001-r008`: stock, meter, and positive RND weights through
`0.6`. The profile deliberately drops only the highest RND weight, `1.0`, to
fit the current proxy room. Non-RND rows are `r005/r011/r017` from static,
long-horizon rep01, and the first cadence/support knob.

## Approval Boundary

Do not run any command below with `--allow-launch` until the operator explicitly
approves this exact packet, row count, runtime tier, and status note path.

Required launch statement:

```text
Approve Wave A long17_no_highest_weight_bestseed: 17 H100 rows, 8h+ tier,
bestseed learner seed, no highest RND weight, status note path
docs/working/training/reward_axis_h100_plan_2026-06-23/live_notes/wave_a_long17_bestseed_health_2026-06-23.md
```

## Preflight Commands

Run immediately before approval or launch:

```bash
uv run python scripts/audit_curvytron_wave_a_launch_packet.py --non-rnd-seed-profile bestseed --output artifacts/local/curvytron_wave_a_launch_packet_audit_bestseed_20260623a.json
```

```bash
uv run python scripts/audit_curvytron_checkpoint_anchor_policy.py --non-rnd-seed-profile bestseed --require-best-known-seed --output artifacts/local/curvytron_checkpoint_anchor_policy_audit_bestseed_20260623a.json
```

```bash
uv run python scripts/audit_curvytron_wave_a_capacity.py --requested-h100-rows 17 --output artifacts/local/curvytron_wave_a_capacity_snapshot_long17_no_highest_weight_bestseed_20260623a.json
```

Proceed only if the packet and anchor audits report `ok=true` with no errors,
the packet audit reports `launch_artifacts=[]`, and capacity is either
`capacity_proxy_clear` or explicitly accepted by the operator as a capacity
override.

## Launch Commands

RND rows, `8`:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.json --row-id r001 --row-id r002 --row-id r003 --row-id r004 --row-id r005 --row-id r006 --row-id r007 --row-id r008 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.selected-r001-r008.submit.launch.json
```

Static bestseed rows, `3`:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-static-bestseed-top4nz-h100-wave-a-20260623a/reward-static-bestseed-top4nz-h100-wave-a-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-static-bestseed-top4nz-h100-wave-a-20260623a/reward-static-bestseed-top4nz-h100-wave-a-20260623a.selected-r005-r011-r017.submit.launch.json
```

Long-horizon bestseed rep01 rows, `3`:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-lhpre-bestseed-top4nz-rep01-h100-wave-a-20260623a/reward-lhpre-bestseed-top4nz-rep01-h100-wave-a-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-lhpre-bestseed-top4nz-rep01-h100-wave-a-20260623a/reward-lhpre-bestseed-top4nz-rep01-h100-wave-a-20260623a.selected-r005-r011-r017.submit.launch.json
```

Cadence/support bestseed rows, `3`:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-csupport-bestseed-top4nz-s25-b128-td25-cap1024-wave-a-20260623a/reward-csupport-bestseed-top4nz-s25-b128-td25-cap1024-wave-a-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-csupport-bestseed-top4nz-s25-b128-td25-cap1024-wave-a-20260623a/reward-csupport-bestseed-top4nz-s25-b128-td25-cap1024-wave-a-20260623a.selected-r005-r011-r017.submit.launch.json
```

## First Status Read

Use this run-id set after launch:

```bash
RUN_IDS='rnd-blank-h100-wave-a-20260623a-no-bonus-copy00-s20260519,rnd-blank-h100-wave-a-20260623a-measure-only-copy00-s20260519,rnd-blank-h100-wave-a-20260623a-bonus-0p003-copy00-s20260519,rnd-blank-h100-wave-a-20260623a-bonus-0p01-copy00-s20260519,rnd-blank-h100-wave-a-20260623a-bonus-0p03-copy00-s20260519,rnd-blank-h100-wave-a-20260623a-bonus-0p10-copy00-s20260519,rnd-blank-h100-wave-a-20260623a-bonus-0p30-copy00-s20260519,rnd-blank-h100-wave-a-20260623a-bonus-0p60-copy00-s20260519,reward-stbest-sparse-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s405954346,reward-stbest-survbonusnoout-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s814824342,reward-stbest-survbonusout-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s1851432079,reward-lhbest-r01-sparse-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s1078440301,reward-lhbest-r01-survbonusnoout-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s1723125752,reward-lhbest-r01-survbonusout-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s947375703,reward-csbest-1024-sparse-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s1214788951,reward-csbest-1024-survbonusnoout-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s994388710,reward-csbest-1024-survbonusout-slot64-blank12-wall4-rank1_46-rank1imm2-clean-s1333097153'
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --output table
```

Then follow with:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --output curve-summary
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --output eval-summary
```

First note path:

```text
docs/working/training/reward_axis_h100_plan_2026-06-23/live_notes/wave_a_long17_bestseed_health_2026-06-23.md
```

## First Decisions

- `0-30m`: health only. Check heartbeats, progress, checkpoint/eval poller,
  RND metrics, and missing-run rows.
- `30k-50k`: weak signal only. Look for action collapse, eval wiring, and
  obviously broken RND metric scale.
- `100k-170k`: first useful AUC/best/retention read.
- `240k-300k` or `8h+`: retention and promotion-read candidate selection.

Do not interpret RND positives unless stock, meter, and the bestseed non-RND
rows are healthy.
