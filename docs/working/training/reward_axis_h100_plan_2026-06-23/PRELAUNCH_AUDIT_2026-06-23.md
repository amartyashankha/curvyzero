# Prelaunch Audit - 2026-06-23

Status: non-launch preflight. No training jobs were launched.

## Checks Run

Modal scope:

- `modal profile current` returned `modal-labs`.
- `modal environment list` showed active environment `shankha-dev`.
- `modal volume list --json` showed `curvyzero-runs-v2` exists in
  `shankha-dev`.
- `modal volume ls --env main curvyzero-runs-v2 ...` failed because
  `curvyzero-runs-v2` does not exist in `main`.
- `modal volume ls curvyzero-runs ...` failed because the legacy
  `curvyzero-runs` volume is not present in `shankha-dev`.

Capacity surface:

```bash
modal app list --json | jq '{app_count:length, total_tasks:(map((.Tasks // "0")|tonumber)|add), detached_running:([.[] | select(.State=="ephemeral (detached)")] | length), curvy_train:[.[] | select(.Description=="curvyzero-lightzero-curvytron-visual-survival-train-v2") | {state:.State,tasks:.Tasks,created_at:."Created at"}], curvy_status:[.[] | select(.Description=="curvyzero-lightzero-curvytron-run-status") | {state:.State,tasks:.Tasks,created_at:."Created at"}]}'
```

Result:

- `app_count=135`
- `total_tasks=71`
- `detached_running=47`
- CurvyTron train app `curvyzero-lightzero-curvytron-visual-survival-train-v2`
  was deployed with `tasks=0`
- CurvyTron status app `curvyzero-lightzero-curvytron-run-status` was deployed
  with `tasks=0`

This is not a launch clearance. It only says the CurvyTron app itself was idle
at audit time; unrelated detached apps were active, and H100 capacity still
needs a fresh check at launch time.

Exact-ref manifest audit:

```bash
uv run python scripts/audit_curvytron_launch_manifest_refs.py artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.json --check-modal --output artifacts/local/curvytron_tonight18_manifests/reward-static-exactref-h100-wave-a-20260623a/reward-static-exactref-h100-wave-a-20260623a.ref_audit.modal.json
```

Result:

- `ok=false`
- `ref_count=4`
- `missing_ref_count=4`
- `modal_parent_error_count=2`
- the two checkpoint parent dirs for the `curvy-n18conn-*` source runs do not
  currently exist in `curvyzero-runs-v2`

The audit helper was updated to preserve Modal parent lookup errors in
`modal_parent_errors`; before this, parent lookup failures were collapsed into
plain missing refs.

## Affected Lanes

All primary exact-ref non-RND manifests share the same four checkpoint refs:

- static exact-ref reward isolate
- six long-horizon pretrained replica manifests
- three cadence/support knob manifests

Therefore the single failed static exact-ref Modal audit blocks the whole
primary exact-ref non-RND family until refs are repaired or rebuilt.

## Decision

Do not launch the original 90-row Wave A campaign from the `curvy-n18conn-*`
exact-ref manifests.

RND wide blank sweep remains locally manifest-ready, but a positive-RND learning
campaign must not launch alone. If RND is launched before non-RND ref repair, it
must be explicitly scoped as a plumbing-only preflight and not interpreted as a
learning or reward result.

## Repair Follow-Up

The non-RND exact-ref family was rebuilt from currently visible
`curvy-r18fresh-*` refs in `curvyzero-runs-v2`. The repaired launch package is
recorded in `PRELAUNCH_REPAIR_2026-06-23.md`.

Repair result:

- refs source:
  `artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.txt`
- refs-file Modal audit: `ok=true`, `ref_count=4`, `missing_ref_count=0`
- static repair: 18-row manifest, Modal ref audit pass, submit dry-run pass
- long-horizon repair: six manifests, selected `r005/r011/r017`, Modal ref
  audits pass, dry-runs pass
- cadence/support repair: three manifests, selected `r005/r011/r017`, Modal ref
  audits pass, dry-runs pass

This supersedes the old blocker for the repaired `top4nz` manifests only. It
does not make the original `curvy-n18conn-*` manifests launchable, and it does
not turn the `r18fresh` source snapshot into a stable production leaderboard.
No training jobs were launched during the repair pass.

## Recovery Options

Preferred recovery: rebuild the non-RND exact-ref manifests from checkpoint refs
that currently exist in `curvyzero-runs-v2`, likely using the present
`curvy-r18fresh-*` or CZ26-era run dirs. Then rerun syntax audit, Modal
existence audit, and submitter dry-runs before restoring launch readiness.

Alternate recovery: if the old `curvy-n18conn-*` refs are still desired, locate
or restore the legacy source volume/environment, rematerialize those refs into
`curvyzero-runs-v2`, and rerun the same manifest-level Modal existence audit.

Do not use stale `source-refs-v2-target-after-copy-audit.json` files as current
launch evidence. They conflict with the current `modal volume ls` result and are
now historical breadcrumbs only.
