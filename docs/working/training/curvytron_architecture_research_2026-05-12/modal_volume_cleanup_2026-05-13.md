# Modal Volume Cleanup - 2026-05-13

Task path: `training/lightzero-curvytron-visual-survival/` on Modal volume
`curvyzero-runs`.

## Preserved

- Current 300-row batch from
  `artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513a.json`.
  - Matrix: `curvy-survive-bonus-large-20260513a`.
  - Run IDs: 300 exact manifest run IDs.
  - First: `curvy-survive-bonus-blank-fast-steady-base-r001-s1110011`.
  - Last: `curvy-survive-bonus-blank-browser-heavy-batch64-r300-s1141671`.
- Previous last-50 v1b batch from
  `artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.rows.jsonl`.
  - Matrix/prefix: `survivaldiag-v1b-20260513h`.
  - Run IDs: 50 exact row-manifest run IDs.
  - First: `survivaldiag-v1b-20260513h-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40`.
  - Last: `survivaldiag-v1b-20260513h-050-survbonusnoout-blanknoop-browser-armed-c01-sim16-s950443-l4t4c40`.
- Protective prefixes used in addition to exact IDs:
  `curvy-survive-bonus`, `survivaldiag-v1b-20260513h`.

Dry-run reported `preserved_count=350`, `missing_preserved_id_count=0`,
`skipped_count=0`, and `prefix_only_preserved_count=0`.

## Deleted

Deleted 263 stale direct run directories under the task path.

Delete categories from the cleanup report:

- `stock_lightzero`: 86.
- `curvytron_visual_survival_other`: 71.
- `survivaldiag_other`: 11.
- `other`: 95.

Delete report status check: `deleted_status_count=263`,
`non_deleted_action_count=0`.

Post-purge verification reported `direct_run_dir_count=350`,
`preserved_count=350`, `action_count=0`, `missing_preserved_id_count=0`, and
`skipped_count=0`.

## Commands And Reports

Script used:
`scripts/cleanup_curvytron_modal_runs.py`

Dry-run command:

```text
uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py --purge-unpreserved --preserve-manifest artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513a.json,artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.rows.jsonl --preserve-prefix curvy-survive-bonus,survivaldiag-v1b-20260513h --report-path artifacts/local/curvytron_survivaldiag_manifests/modal_volume_cleanup_20260513_allowlist_dry_run.json --output-detail compact
```

Delete command:

```text
uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py --purge-unpreserved --preserve-manifest artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513a.json,artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.rows.jsonl --preserve-prefix curvy-survive-bonus,survivaldiag-v1b-20260513h --delete --yes --report-path artifacts/local/curvytron_survivaldiag_manifests/modal_volume_cleanup_20260513_allowlist_delete.json --output-detail compact
```

Post-verify report:
`artifacts/local/curvytron_survivaldiag_manifests/modal_volume_cleanup_20260513_allowlist_post_verify.json`.

## Ambiguity

No cleanup-blocking ambiguity remained. The v1b batch was identified cleanly
from the 50-row row manifest and the launch note
`survivaldiag_v1b_launch_2026-05-13.md`; no preserved IDs were missing from the
volume, and no extra prefix-only directories were preserved.
