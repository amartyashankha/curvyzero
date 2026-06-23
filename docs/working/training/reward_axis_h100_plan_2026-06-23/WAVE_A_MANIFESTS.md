# Wave A Manifest Ledger

Status: local manifests prepared, packet-audited, and submitter dry-runs
passed. The original `curvy-n18conn-*` exact-ref non-RND manifests are
remote-ref blocked, but a repaired `top4nz` non-RND family now has passing
Modal ref audits and dry-runs. No Modal launch was performed from this doc set.

## Prepared Manifests

| Lane | Manifest | Rows | Claim class | Notes |
| --- | --- | ---: | --- | --- |
| RND wide blank sweep | `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.json` | 45 | manifest-ready, packet-audited | 5 replicas over stock, meter, and 7 positive RND weights. Saved submitter dry-run exists. |
| Static top4nz exact-ref repair | `artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.json` | 18 | manifest-ready, ref-audited, packet-audited | Current non-RND launch partner candidate. Full tonight18 reward/recipe/noise matrix, no refresh, seeded from audited `curvy-r18fresh-*` refs. |
| Long-horizon top4nz replicas | `artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-rep01-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-rep01-h100-wave-a-repair-20260623a.json` through `rep06` | 18 selected | manifest-ready, ref-audited, packet-audited | Six repaired full 18-row manifests; intended launch selects `r005/r011/r017` from each. |
| Cadence/support top4nz panel | `artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a.json` plus two sibling knob manifests | 9 selected | manifest-ready, ref-audited, packet-audited | Three repaired full 18-row knob manifests; intended launch selects `r005/r011/r017` from each. |
| Original static exact-ref reward isolate | `artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.json` | 18 | manifest-ready, ref-blocked | Historical candidate. Current Modal audit fails because shared `curvy-n18conn-*` refs are missing. |

The older `reward-static-h100-wave-a-20260623a` manifest also exists locally,
but it is not the current Wave A launch candidate. Prefer the repaired
`top4nz` exact-ref manifest row above for current launch review.

Some plus-outcome-only long-horizon and cadence/support manifests also exist
locally from exploration. They are secondary artifacts, not the recommended Wave
A queue. The primary non-RND expansion is the reward-triad row selection
`r005/r011/r017`.

Use `PRELAUNCH_REPAIR_2026-06-23.md` as the current artifact checklist for the
repaired non-RND family. Use `PRELAUNCH_AUDIT_2026-06-23.md` as the record of
why the original exact-ref family remains blocked.

Use `CHECKPOINT_ANCHOR_POLICY.md` before launching medium or long rows. The
current repaired non-RND manifests are Modal-ref-audited but top4nz-seeded, not
historical-best-seeded.

## Packet Audit

The current repaired packet is checked by:

```bash
uv run python scripts/audit_curvytron_wave_a_launch_packet.py --output artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json
```

Saved result:

```text
artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json
```

Result:

- `ok=true`
- `actual_total_selected_rows=90`
- `expected_total_selected_rows=90`
- `expected_non_rnd_selected_rows=45`
- `expected_rnd_selected_rows=45`
- `launch_artifacts=[]`
- `error_count=0`

The auditor is no-launch only. By default it rejects existing
`*.submit.launch.json` artifacts, broad `*wave-a*` legacy globs, missing Modal
ref audits, RND metrics-guard drift, and selected-row mismatches.

## RND Wide Blank Sweep

Builder intent:

- matrix name: `rnd-blank-h100-wave-a-20260623a`
- compute: `gpu-h100-cpu40`
- rows: 45
- max train iter: 300000
- save checkpoint every 2500 iterations
- opponent runtime: `blank_canvas_noop`
- opponent refresh interval: `0`
- tournament: disabled
- reward variant: `survival_plus_bonus_no_outcome`
- RND update cadence: `exploration_bonus_rnd_update_per_collect=100`

Modes and weights:

- `none` at weight `0.0`
- `rnd_meter_v0` at weight `0.0`
- `rnd_replay_target_v0` at weights `0.003`, `0.01`, `0.03`, `0.1`,
  `0.3`, `0.6`, `1.0`

Guard read:

- `guards.expected_row_count=45`
- `guards.modal_launch_performed=false`
- `guards.operator_launch_gate_required=true`
- `guards.assignment_refresh_enabled=false`
- `guards.all_rows_blank_canvas_noop=true`
- stock rows have `require_rnd_metrics=false`
- meter and positive rows have `require_rnd_metrics=true`

Submitter dry-run:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.json
```

Dry-run result:

- `dry_run=true`
- `status=dry_run`
- `row_count=45`
- `selected_row_count=45`
- `assignment_write_count=0`
- `refresh_pointer_write_count=0`
- `train_function=lightzero_curvytron_visual_survival_h100_cpu40`
- `poller_function=lightzero_curvytron_visual_survival_checkpoint_eval_poller`
- saved artifact:
  `artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.submit.dryrun.json`

Launch command, only after explicit operator approval:

```bash
uv run --extra modal python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.json --allow-launch
```

## Repaired Top4NZ Non-RND Family

Source refs:

```text
artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.txt
```

Refs-file Modal audit:

```text
artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.ref_audit.modal.json
```

Result:

- `ok=true`
- `ref_count=4`
- `missing_ref_count=0`
- `modal_parent_error_count=0`

Interpretation caveat: the top4 nonzero refs are a believable static repair
source because they exist now in `curvyzero-runs-v2` and came from nonzero
mid-run `r18fresh` checkpoints. They are not a stable production leaderboard;
the old no-tournament control plan already framed them as exact frozen refs.

Repaired static manifest:

```text
artifacts/local/curvytron_tonight18_manifests/reward-static-top4nz-h100-wave-a-repair-20260623a/reward-static-top4nz-h100-wave-a-repair-20260623a.json
```

Gate results:

- syntax audit: `ok=true`
- Modal ref audit: `ok=true`, `ref_count=4`, `missing_ref_count=0`
- submitter dry-run: `dry_run=true`, `selected_row_count=18`
- assignment writes: `0`
- refresh pointer writes: `0`
- train function: `lightzero_curvytron_visual_survival_h100_cpu40`

Repaired long-horizon replicas:

```text
artifacts/local/curvytron_tonight18_manifests/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a/reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a.json
```

For `rep01` through `rep06`:

- syntax audit: `ok=true`
- Modal ref audit: `ok=true`, `ref_count=4`, `missing_ref_count=0`
- selected-row dry-run: `dry_run=true`, `selected_row_count=3`
- selected rows: `r005`, `r011`, `r017`
- assignment writes: `0`
- refresh pointer writes: `0`

Repaired cadence/support manifests:

```text
artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b128-td25-cap1024-wave-a-repair-20260623a.json
artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b128-td25-cap2048-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b128-td25-cap2048-wave-a-repair-20260623a.json
artifacts/local/curvytron_tonight18_manifests/reward-csupport-top4nz-s25-b256-td25-cap2048-wave-a-repair-20260623a/reward-csupport-top4nz-s25-b256-td25-cap2048-wave-a-repair-20260623a.json
```

For all three:

- syntax audit: `ok=true`
- Modal ref audit: `ok=true`, `ref_count=4`, `missing_ref_count=0`
- selected-row dry-run: `dry_run=true`, `selected_row_count=3`
- selected rows: `r005`, `r011`, `r017`
- assignment writes: `0`
- refresh pointer writes: `0`

## Static Exact-Ref Reward Isolate

Historical/ref-blocked original candidate. Do not use this section's manifest
for launch unless the old `curvy-n18conn-*` refs are restored into
`curvyzero-runs-v2` and a fresh Modal existence audit passes. Use the repaired
top4nz section above for current launch review.

Builder intent:

- matrix name: `reward-static-exactref-h100-wave-a-20260623a`
- compute: `gpu-h100-cpu40`
- rows: 18
- max train iter: 300000
- save checkpoint every 10000 iterations
- opponent source: `mixture`
- assignment refresh interval: `0`
- assignment bank: none
- own-checkpoint opponent refresh: disabled
- initial policy source: `rank1_checkpoint_from_checkpoint_refs_file`
- shared initial checkpoint: rank1 `iteration_240000.pth.tar`

Checkpoint ref source:

```text
artifacts/local/curvytron_restart_source_refs/restart18-source-loop18-top96-nonzero-20260515a/refs.txt
```

The refs file contains 96 lines. The manifest uses four curated exact refs for
rank slots and one shared exact initial checkpoint for all 18 rows.

Syntax-only ref audit:

```bash
uv run python scripts/audit_curvytron_launch_manifest_refs.py artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.json --syntax-only --output artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.ref_audit.syntax.json
```

Audit result:

- `ok=true`
- `bad_ref_count=0`
- `missing_ref_count=0`
- `ref_count=4`
- `syntax_only=true`
- `existence_checked=false`

Submitter dry-run:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.json
```

Dry-run result:

- `dry_run=true`
- `status=dry_run`
- `row_count=18`
- `selected_row_count=18`
- `assignment_write_count=0`
- `refresh_pointer_write_count=0`
- `training_candidate_refresh_config_record=null`
- `train_function=lightzero_curvytron_visual_survival_h100_cpu40`
- `poller_function=lightzero_curvytron_visual_survival_checkpoint_eval_poller`

Launch command, only after explicit operator approval:

```bash
uv run python scripts/submit_curvytron_survivaldiag_manifest.py artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.json --allow-launch
```

## Legacy Non-Repair Replica/Support Manifests

The old non-`top4nz` long-horizon and cadence/support manifests still exist
under names such as `reward-lhpre-repNN-h100-wave-a-20260623a` and
`reward-csupport-s25-*-wave-a-20260623a`. Treat those as historical and
ref-blocked for current operations. Their syntax-only audits are not enough,
because the shared old exact refs are not visible in `curvyzero-runs-v2`.

For current launch review, use only:

- `reward-lhpre-top4nz-repNN-h100-wave-a-repair-20260623a`
- `reward-csupport-top4nz-*-wave-a-repair-20260623a`

Those repaired families are covered by the packet audit above and by
`PRELAUNCH_REPAIR_2026-06-23.md`.

## Remaining Gates Before Launch

Do not promote any lane from `manifest-ready` to `launched` until:

- a human explicitly approves the launch command and intended row count
- active H100 usage leaves capacity for the requested rows under the chosen
  runtime tier
- the launch note says whether this is a short `<=2h` breadth sweep, a
  `2h-8h` run capped at 40 rows, or an `8h+` run capped at 10-20 rows
- the launch note says whether non-RND rows use the current top4nz repair seed
  or regenerated historical best-known seed
- all exact-ref non-RND manifests selected for launch pass a Modal existence
  audit, not just syntax
- `scripts/audit_curvytron_wave_a_launch_packet.py` passes with
  `launch_artifacts=[]` unless the audit is intentionally being run after a
  real launch
- launch commands reference the repaired `top4nz` manifests in
  `PRELAUNCH_REPAIR_2026-06-23.md`, or the failing original audit in
  `PRELAUNCH_AUDIT_2026-06-23.md` is explicitly resolved by restored refs
- the operator confirms whether to launch all prepared lanes together or in
  staged groups
- the first monitoring command and artifact note path are chosen before launch
- cleanup/stop commands are known for the exact run-id prefixes

The exact-ref existence audit should use the same audit script with the Modal
existence check enabled. Keep the syntax-only result as a weaker preflight, not
as proof that remote checkpoint refs exist.

Current failing original audit artifact:
`artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.ref_audit.modal.json`.

## First Post-Launch Reads

At 0-30 minutes:

- every row has a Modal FunctionCall or a clear submission failure
- `progress_latest.json` exists or the startup failure is classified
- checkpoint eval poller artifacts exist
- RND rows with meter/positive modes write RND metrics
- stock RND rows are not blocked on RND metrics

At 30k-50k iterations:

- no interpretation from flat survival alone
- action histograms and GIFs are used only to detect collapse
- learner metrics must be advancing and finite
- RND predictor metrics must be finite and nontrivial

At 100k-170k iterations:

- start reading AUC, best-so-far, and latest-vs-best retention
- compare RND positive weights to both stock and meter controls
- compare reward variants only on survival/eval metrics, not raw trainer reward

At 240k-300k iterations:

- decide which rows earn replicas or bridge experiments
- preserve best checkpoints even if latest regresses
- attach diagnostic tournament only after nonzero checkpoints exist
