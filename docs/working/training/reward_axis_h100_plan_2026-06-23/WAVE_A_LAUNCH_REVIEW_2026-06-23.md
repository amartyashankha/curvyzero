# Wave A Launch Review - 2026-06-23

Status: approval packet draft. No Modal training jobs were launched.

Current recommendation: use the bestseed non-RND repair family for medium and
long learning reads. The top4nz-seeded packet in this note remains launchable
only if the approval explicitly chooses the top4nz repair seed as the comparison
anchor.

## Scope

This note packages the original repaired top4nz Wave A launch candidate:

- RND blank sweep: `45` rows.
- Static top4nz exact-ref repair: `18` rows.
- Long-horizon top4nz replicas: `18` selected rows.
- Cadence/support top4nz panel: `9` selected rows.

Total prepared H100 rows: `90`, balanced as `45` RND and `45` non-RND.

A preferred bestseed sibling now exists with the same row shape, but with the
learner initial policy pinned to the historical r18fresh plus-outcome
`iteration_180000` checkpoint while keeping the top4nz exact refs as static
opponent rank slots. Its packet audit is:

```text
artifacts/local/curvytron_wave_a_launch_packet_audit_bestseed_20260623a.json
```

Use the bestseed packet unless the launch note explicitly says the experiment is
testing the top4nz repair seed itself.

Use this only with:

- `LAUNCH_QUEUE.md` for current lane state
- `WAVE_A_MANIFESTS.md` for manifest ledger
- `PRELAUNCH_AUDIT_2026-06-23.md` for the blocked original refs
- `PRELAUNCH_REPAIR_2026-06-23.md` for repaired non-RND artifact checklist
- `CHECKPOINT_ANCHOR_POLICY.md` for the learner-seed versus opponent-ref split
- `MONITORING_SIGNALS.md` and `CONTINGENCY_PLANS.md` for first reads and
  responses

## Approval Boundary

Do not run any command with `--allow-launch` until the operator explicitly
approves the exact command set and intended row count.

Do not launch positive RND alone. The full repaired packet is intentionally
balanced as `45` RND rows and `45` non-RND rows.

Before approval, rerun the no-launch packet audit. Preferred bestseed form:

```bash
uv run python scripts/audit_curvytron_wave_a_launch_packet.py --non-rnd-seed-profile bestseed --output artifacts/local/curvytron_wave_a_launch_packet_audit_bestseed_20260623a.json
```

Top4nz comparison form, only if the launch note chooses top4nz seeding:

```bash
uv run python scripts/audit_curvytron_wave_a_launch_packet.py --output artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json
```

Expected packet audit result:

- `ok=true`
- `actual_total_selected_rows=90`
- `expected_total_selected_rows=90`
- `expected_non_rnd_selected_rows=45`
- `expected_rnd_selected_rows=45`
- `launch_artifacts=[]`
- `error_count=0`

Then rerun active capacity through the capacity auditor:

```bash
uv run python scripts/audit_curvytron_wave_a_capacity.py --output artifacts/local/curvytron_wave_a_capacity_snapshot_20260623a.json
```

For staged profiles, use the profile-specific capacity command embedded in the
staged artifact. Current bestseed frontier:

- `artifacts/local/curvytron_wave_a_capacity_snapshot_long17_no_highest_weight_bestseed_20260623a.json`:
  `17` rows, `capacity_proxy_clear`, `projected_total_tasks=100`
- `artifacts/local/curvytron_wave_a_capacity_snapshot_long18_all_weights_bestseed_20260623a.json`:
  `18` rows, `operator_capacity_review_required`, `projected_total_tasks=101`
- `artifacts/local/curvytron_wave_a_capacity_snapshot_long19_low_weight_replicated_bestseed_20260623a.json`:
  `19` rows, `operator_capacity_review_required`, `projected_total_tasks=101`
- `artifacts/local/curvytron_wave_a_capacity_snapshot_mid36_bestseed_20260623a.json`:
  `36` rows, `operator_capacity_review_required`, `projected_total_tasks=118`
- `artifacts/local/curvytron_wave_a_capacity_snapshot_short90_bestseed_20260623a.json`:
  `90` rows, `operator_capacity_review_required`, `projected_total_tasks=172`

Treat this capacity read as volatile context and a coarse task-count proxy only.
It does not prove H100 availability or unavailability because Modal task counts
can include non-H100 work. Rerun immediately before approval and make an
explicit operator capacity decision. Also choose the runtime tier before
approval: the 90-row packet is appropriate for a short `<=2h` breadth/health
sweep if capacity is clear, but `2h-8h` runs should launch or retain at most 40
active rows and `8h+` runs should launch or retain only 10-20 rows.

Then rerun the checkpoint-anchor policy audit. Preferred bestseed form:

```bash
uv run python scripts/audit_curvytron_checkpoint_anchor_policy.py --non-rnd-seed-profile bestseed --require-best-known-seed --output artifacts/local/curvytron_checkpoint_anchor_policy_audit_bestseed_20260623a.json
```

Last bestseed anchor read,
`artifacts/local/curvytron_checkpoint_anchor_policy_audit_bestseed_20260623a.json`:

- `ok=true`
- historical best seed: r18fresh plus-outcome `iteration_180000`
- repaired bestseed non-RND manifests audited: `10`
- `historical_best_seed_manifest_count=10`
- `top4nz_seed_manifest_count=0`

Top4nz comparison form, only if the launch note chooses top4nz seeding:

```bash
uv run python scripts/audit_curvytron_checkpoint_anchor_policy.py --output artifacts/local/curvytron_checkpoint_anchor_policy_audit_20260623a.json
```

Last no-launch anchor read,
`artifacts/local/curvytron_checkpoint_anchor_policy_audit_20260623a.json`:

- `ok=true`
- historical best seed: r18fresh plus-outcome `iteration_180000`
- repaired non-RND manifests audited: `10`
- `historical_best_seed_manifest_count=0`
- `top4nz_seed_manifest_count=10`
- warning: current repaired manifests do not use the historical r18fresh rank-1
  checkpoint as their initial seed

Approval must explicitly choose bestseed or top4nz seed policy. If the approval
is silent, do not launch.

## Launch Commands

For runtime-tier staged commands, prefer the generated bestseed planner
artifacts:

```bash
uv run python scripts/plan_curvytron_wave_a_staged_launch.py --profile mid36_bestseed --output artifacts/local/curvytron_wave_a_staged_launch_mid36_bestseed_20260623a.json
uv run python scripts/plan_curvytron_wave_a_staged_launch.py --profile long17_no_highest_weight_bestseed --output artifacts/local/curvytron_wave_a_staged_launch_long17_no_highest_weight_bestseed_20260623a.json
uv run python scripts/plan_curvytron_wave_a_staged_launch.py --profile long18_all_weights_bestseed --output artifacts/local/curvytron_wave_a_staged_launch_long18_all_weights_bestseed_20260623a.json
uv run python scripts/plan_curvytron_wave_a_staged_launch.py --profile long19_low_weight_replicated_bestseed --output artifacts/local/curvytron_wave_a_staged_launch_long19_low_weight_replicated_bestseed_20260623a.json
```

The commands below are the full prepared top4nz packet shape, not the preferred
bestseed launch shape. For bestseed launches, generate exact commands from the
bestseed staged artifacts or from the bestseed manifest paths recorded in
`WAVE_A_MANIFESTS.md`.

RND full sweep, `45` rows:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.json --allow-launch --output artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.submit.launch.json
```

Static non-RND repair, `18` rows:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.json --allow-launch --output artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.submit.launch.json
```

Long-horizon repair, `18` selected rows:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep01-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep01-h100-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep01-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep01-h100-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep02-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep02-h100-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep02-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep02-h100-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep03-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep03-h100-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep03-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep03-h100-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep04-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep04-h100-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep04-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep04-h100-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep05-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep05-h100-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep05-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep05-h100-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep06-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep06-h100-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep06-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep06-h100-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
```

Cadence/support repair, `9` selected rows:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b128-td25-cap2048-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b128-td25-cap2048-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b128-td25-cap2048-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b128-td25-cap2048-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b256-td25-cap2048-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b256-td25-cap2048-wave-a-repair-20260623a.json --row-id r005 --row-id r011 --row-id r017 --allow-launch --allow-partial-launch --output artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b256-td25-cap2048-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b256-td25-cap2048-wave-a-repair-20260623a.selected-r005-r011-r017.submit.launch.json
```

## Run Prefixes

Expected run-id prefixes:

- `rnd-blank-h100-wave-a-20260623a-`
- `reward-st4nz-repair-`
- `lhpre-st4nz-r01-` through `lhpre-st4nz-r06-`
- `csup-st4nz-s25-b128-c1024-`
- `csup-st4nz-s25-b128-c2048-`
- `csup-st4nz-s25-b256-c2048-`

After launch, derive the exact comma-separated run list from the launch JSON
files, not from memory.

## First Health Commands

Set `RUN_IDS` from launch JSON files after approval and submission:

```bash
RUN_IDS="$(jq -rs '[.[].records[].run_id] | join(",")' artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.submit.launch.json artifacts/local/curvytron_tonight18_manifests/*repair-20260623a/*.submit.launch.json)"
```

Then read health:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --output table
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --output curve-summary
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --output eval-summary
```

Suggested health-note path:

```text
docs/working/training/reward_axis_h100_plan_2026-06-23/WAVE_A_STATUS_2026-06-23.md
```

## Long-Sleep Monitoring Pattern

Use sleeps only after jobs are launched and first health has been checked:

```bash
while true; do
  date
  uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --output curve-summary
  sleep 1800
done
```

At the longer decision horizons, switch to `eval-summary` and lane-specific
artifact reads. Do not read flat survival before 100k as failure if health is
clean.

## Stop And Cleanup Boundary

The current repo path does not expose a row-level stop command. The safe cleanup
procedure is:

1. Preserve launch JSON files and status output.
2. Extract `function_call_id` and poller ids from the launch JSON.
3. Inspect logs and current status before stopping anything.
4. Stop only the affected Modal work by exact app id or Modal control surface;
   do not use `modal app stop` by app description alone.
5. Record the exact reason for stop in `WAVE_A_STATUS_2026-06-23.md`.

This boundary is intentional. App-level stopping is too blunt unless the exact
impact is verified first.
