# Modal Volume Cleanup Follow-Up Dry Run - 2026-05-13

Context update: the `curvy-survive-bonus-large-20260513a` 300-run launch
failed immediately because the train calls crashed from missing kwargs. Pollers
still wrote `checkpoint_eval_poller.json`, and the GIF browser may still show
old GIFs. Therefore every run whose ID starts with `curvy-survive-bonus-` is
hard-protected until a relaunch succeeds and the new run IDs/attempt IDs are
confirmed.

## Dry-Run Report

Report file:
`artifacts/local/curvytron_survivaldiag_manifests/modal_volume_cleanup_20260513_followup_failed300_dry_run.json`

Command:

```text
uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py --purge-unpreserved --preserve-manifest artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513a.json,artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.rows.jsonl --preserve-prefix curvy-survive-bonus,survivaldiag-v1b-20260513h --report-path artifacts/local/curvytron_survivaldiag_manifests/modal_volume_cleanup_20260513_followup_failed300_dry_run.json --output-detail compact
```

Result:

- `dry_run=true`, `delete=false`.
- `direct_run_dir_count=350`.
- `preserved_count=350`.
- `action_count=0`.
- `missing_preserved_id_count=0`.
- `prefix_only_preserved_count=0`.
- `skipped_count=0`.
- `delete_categories={}`.

No old training run roots, checkpoint dirs, eval manifests, or GIF artifacts
would be removed by the current allowlist dry run.

## Preserved Identification

Current failed 300 batch:

- Manifest source:
  `artifacts/local/curvytron_survivaldiag_manifests/curvy-survive-bonus-large-20260513a.json`.
- Matrix: `curvy-survive-bonus-large-20260513a`.
- Exact manifest run IDs: 300.
- Protective prefix: `curvy-survive-bonus`.
- First run ID:
  `curvy-survive-bonus-blank-fast-steady-base-r001-s1110011`.
- Last run ID:
  `curvy-survive-bonus-blank-browser-heavy-batch64-r300-s1141671`.
- Preservation reason: exact manifest ID plus hard prefix guard because the
  failed launch still owns poller artifacts and may be needed for diagnosis.

Previous ugly 50 batch:

- Row manifest source:
  `artifacts/local/curvytron_survivaldiag_manifests/survivaldiag-v1b-20260513h.rows.jsonl`.
- Launch note source:
  `docs/working/training/curvytron_architecture_research_2026-05-12/survivaldiag_v1b_launch_2026-05-13.md`.
- Exact row-manifest run IDs: 50.
- Protective prefix: `survivaldiag-v1b-20260513h`.
- First run ID:
  `survivaldiag-v1b-20260513h-001-survbonusnoout-blanknoop-fast-armed-c00-s910001-l4t4c40`.
- Last run ID:
  `survivaldiag-v1b-20260513h-050-survbonusnoout-blanknoop-browser-armed-c01-sim16-s950443-l4t4c40`.
- Preservation reason: this is the documented last 50-row v1b batch the user
  asked to keep.

## Scope Guard

The cleanup script is scoped to direct run directories under:

`training/lightzero-curvytron-visual-survival/`

Old website/browser support paths are not targeted by this dry run and must not
be added to a delete scope. Any future destructive cleanup needs a fresh
manifest-backed allowlist after relaunch, with the current failed
`curvy-survive-bonus-` prefix still protected unless explicitly cleared.

Cleanup is deferred until after the rescue relaunch is healthy: current
artifacts are under preserved prefixes, and website clutter is nonblocking.
