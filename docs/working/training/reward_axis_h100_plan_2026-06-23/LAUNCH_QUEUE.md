# Launch Queue

Status: operator queue for the active reward/RND H100 campaign. No Modal jobs
were launched while writing this note.

Use this file as the front door when deciding what to do next. Follow the
details in:

- `OPERATING_PATTERNS.md` for claim classes, broad-sweep posture, launch gates,
  and artifact discipline
- `WAVE_A_MANIFESTS.md` for prepared manifest paths and dry-run evidence
- `WAVE_A_LAUNCH_REVIEW_2026-06-23.md` for the approval packet draft, command
  shapes, first status commands, and cleanup boundary
- `artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json` for the
  latest no-launch packet audit result
- `artifacts/local/curvytron_wave_a_capacity_snapshot_20260623a.json` for the
  latest no-launch Modal capacity proxy result
- `artifacts/local/curvytron_checkpoint_anchor_policy_audit_20260623a.json`
  for the current best-known-checkpoint seed audit
- `artifacts/local/curvytron_wave_a_staged_launch_mid36_20260623a.json` and
  `artifacts/local/curvytron_wave_a_staged_launch_long19_low_weight_replicated_20260623a.json`
  for staged launch command plans
- `EXPERIMENT_PLAN.md` and `AGGRESSIVE_REORIENTATION.md` for lane intent
- `RND_LANE.md` and `STOCK_PATH_RND_REORIENTATION.md` for RND-specific rules
- `MONITORING_SIGNALS.md` and `CONTINGENCY_PLANS.md` for readout and response

## Current Queue

| Lane | Rows | Claim class | Current state | Next gate |
| --- | ---: | --- | --- | --- |
| RND wide blank sweep | 45 | manifest-ready | Local manifest and saved submitter dry-run exist. Interpret only alongside healthy stock/meter controls and non-RND lanes. | Human launch approval, active capacity check, status note path, and matching non-RND launch or already-healthy non-RND rows. |
| Static top4nz exact-ref repair | 18 | manifest-ready, ref-audited | Repaired local manifest, syntax audit, Modal ref audit, and full submitter dry-run pass. Uses currently visible `curvy-r18fresh-*` refs. | Human launch approval, active capacity check, launch note, and stop/cleanup procedure. |
| Long-horizon top4nz replicas | 18 | manifest-ready, ref-audited | Six repaired 18-row manifests exist; selected rows `r005/r011/r017` from each pass Modal ref audit and dry-run. | Human launch approval, active capacity check, launch with row filters and `--allow-partial-launch`. |
| Cadence/support top4nz panel | 9 | manifest-ready, ref-audited | Three repaired 18-row knob manifests exist; selected rows `r005/r011/r017` from each pass Modal ref audit and dry-run. | Human launch approval, active capacity check, launch with row filters and `--allow-partial-launch`. |
| Buffer | 10 | reserved | Capacity held for relaunch/debug/fixed-opponent RND bridge. | Spend only after first health read or an explicit bridge decision. |

The prepared Wave A menu remains 90 H100 rows. All 90 repaired rows have local
manifest and dry-run coverage. The repaired non-RND exact-ref rows also have
passing Modal existence audits. No Modal jobs were launched from this doc set.
The current packet audit reports `ok=true`, `actual_total_selected_rows=90`,
and `launch_artifacts=[]`.

The current capacity proxy reports `ok=true` but
`approval_recommendation=operator_capacity_review_required`: CurvyTron train and
status apps are idle, while current Modal task count plus `90` requested rows
exceeds the coarse `100`-task envelope proxy. The current conservative proxy
allows only `22` additional rows unless existing tasks are classified as
non-H100/non-conflicting. This does not prove H100 unavailability because Modal
task counts can include non-H100 work, but it does require a fresh operator
capacity decision before launch.

The original `reward-static-exactref-*`, `reward-lhpre-repNN-*`, and
`reward-csupport-s25-*` manifests that point at `curvy-n18conn-*` refs remain
historical/ref-blocked by `PRELAUNCH_AUDIT_2026-06-23.md`. Use the `top4nz`
repair manifests in `PRELAUNCH_REPAIR_2026-06-23.md` for current launch review.

Checkpoint-anchor caveat: the repaired top4nz non-RND manifests currently use
the top4nz rank1 sparse `iteration_40000` checkpoint as their initial policy
seed. The historical best-known seed remains the r18fresh plus-outcome
`iteration_180000` checkpoint. The current anchor audit is `ok=true` but warns
about this difference. Medium/long launch approval must explicitly choose
between accepting the launchable repair seed and regenerating with the
historical best-known seed.

Non-RND coverage is required. The static top4nz exact-ref reward isolate is the
prepared non-RND launch partner for the RND sweep. The long-horizon replicas and
cadence/support panel are also prepared as row-filtered local manifests, not as
separate native 18-row and 9-row schemas.

## Prepared Launch Pair

The two primary prepared lanes are independent and can launch broadly together
only after explicit approval and a fresh capacity check:

1. `RND wide blank sweep`
2. `Static top4nz exact-ref repair`

This remains the broad-sweep default after ref repair. Do not serialize these
into many days of tiny canaries. The canary logic is embedded in the broad RND
manifest through stock and meter controls; health failures should trigger the
contingencies, not prevent broad preparation.

Required before launch:

- explicit human approval for the exact launch command and row count
- fresh `scripts/audit_curvytron_wave_a_launch_packet.py` pass
- fresh `scripts/audit_curvytron_wave_a_capacity.py` pass and active H100
  capacity decision
- fresh `scripts/audit_curvytron_checkpoint_anchor_policy.py` pass and an
  explicit seed-anchor decision
- generated `scripts/plan_curvytron_wave_a_staged_launch.py` profile for the
  intended runtime tier
- intended runtime tier:
  - `<=2h`: broad 90-row packet is allowed if capacity is explicitly clear;
    more than 100 simultaneous H100s needs explicit operator override and a
    short timeout
  - `2h-8h`: at most 40 active H100 rows
  - `8h+`: 10-20 active H100 rows
- if capacity is ambiguous, stage at or below the capacity audit's
  `max_additional_rows_under_task_proxy` or wait
- confirmation that the launch includes or preserves a healthy non-RND lane
- chosen run note path for first health read
- known cleanup/stop commands keyed by run-id prefix
- for exact-ref non-RND lanes: Modal existence audit, not only syntax audit
- launch package points at `PRELAUNCH_REPAIR_2026-06-23.md`, not the ref-blocked
  original manifests

## Prepared Non-RND Expansion

The non-RND expansion rows are intentionally row-filtered from full tonight18
builder outputs because the builder validates exactly 18 rows. Record row ids in
the launch note and use `--allow-partial-launch` only after explicit approval.

1. Long-horizon pretrained replicas: six replica manifests, selected rows
   `r005/r011/r017`, total intended active rows `18`.
2. Cadence/support panel: three knob manifests, selected rows `r005/r011/r017`,
   total intended active rows `9`.

Long-horizon replicas should reuse exact immutable initial checkpoint refs and
simple static mixtures. They should not introduce leaderboard refresh,
assignment pointers, or new opponent-source ambiguity.

Current caveat: use only the repaired `top4nz` replica/support manifests for
launch review. The older non-repair manifests reuse missing exact refs and stay
blocked unless those refs are restored and audited.

Cadence/support rows should be labeled as training-dynamics tests. A win there
does not prove a reward variant by itself; it says the next reward matrix should
inherit the stabilized knobs.

## Launch Readout Order

0-30 minutes:

- read only health
- require RND metrics on enabled rows
- require checkpoint/eval/GIF artifacts or classify the missing artifact

30k-50k:

- do not reject healthy flat rows
- inspect learner metrics and action collapse
- prepare, but do not overread, early AUC/best signals

100k-170k:

- first useful broad comparison
- require RND positive weights to beat both stock and meter controls
- require non-RND static reward rows to have comparable health and horizon
- compare extrinsic rewards by survival/eval metrics, not raw trainer reward

240k-300k:

- retention decision
- preserve best checkpoints even if latest regresses
- decide replicas, fixed-opponent RND bridge, support/cadence expansion, or
  diagnostic tournament attachment

## Do Not Cross These Wires

- RND is an intrinsic lane, not an extrinsic reward variant.
- Stock-ish LightZero RND success does not promote compact RND.
- Compact speed wins do not promote reward or RND claims.
- Non-RND extrinsic reward/cadence lanes must run even if RND looks exciting.
- Diagnostic tournament selection comes after nonzero checkpoints exist.
- Trainer-facing leaderboard refresh is Wave C, not Wave A.
