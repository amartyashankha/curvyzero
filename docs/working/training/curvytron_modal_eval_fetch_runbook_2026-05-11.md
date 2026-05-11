# CurvyTron Modal Eval Fetch Runbook - 2026-05-11

Purpose: keep Modal operations separate from the local inspector.

## Rule

The inspector should read local files and report evidence. It should not call
Modal by default.

Use Modal only when the local files are missing the action trace, the replay hash
does not match, or the checkpoint itself must be rerun with newer instrumentation.

## Clean Pattern

1. List the remote checkpoint or eval directory.

```bash
uv run --extra modal modal volume ls curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run_id>/checkpoints/lightzero
```

2. Rerun explicit old checkpoint refs with a new eval id.

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_visual_survival_eval \
  --compute cpu \
  --run-id <run_id> \
  --attempt-id <attempt_id> \
  --eval-id <new_eval_id> \
  --selected-iterations 0,128,256,384 \
  --eval-seeds 1297473639,1657157601,31836349 \
  --max-eval-steps 1024 \
  --parallel \
  --summary-only \
  --quiet-framework-logs
```

3. Fetch only the eval output needed for local inspection.

```bash
uv run --extra modal modal volume get curvyzero-runs \
  training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/eval/<new_eval_id> \
  artifacts/local/curvytron-eval-manifests/<new_eval_id> \
  --force
```

4. Inspect locally.

```bash
uv run python -m curvyzero.training.curvytron_inspector \
  --eval-manifest artifacts/local/curvytron-eval-manifests/<new_eval_id>/<manifest>.json
```

## Old Checkpoint Locations

The local manifests point to checkpoints in the Modal Volume `curvyzero-runs`,
mounted remotely at `/runs`.

Known refs:

- `curvytron-visual-survival-player-aware-fixed-s101-262144/checkpoints/lightzero/iteration_{0,256,512,768,1024,1071}.pth.tar`
- `curvytron-visual-survival-player-aware-fixed-s102-sim8-131072/checkpoints/lightzero/iteration_{0,128,256,384,512,540}.pth.tar`
- `curvytron-visual-survival-player-aware-fixed-s100-131072/checkpoints/lightzero/iteration_{0,128,256,384,512,520}.pth.tar`
- `curvytron-visual-survival-debug-lz-refresh-s92iter384-s93-131072/checkpoints/lightzero/iteration_{0,128,256,384,512,584}.pth.tar`

No matching checkpoint weights were found locally in this checkout.

## Avoid

- Do not reuse an old `eval_id` for a new run.
- Do not infer "latest" checkpoint when the question is about a specific old
  artifact.
- Do not compare runs with different seeds, caps, opponent types, or load status
  as if they were one clean curve.
- Do not make `curvytron_inspector.py` fetch remote files or launch Modal runs.
- Do not make one Modal call per timestep or per episode.

## Source Notes

Modal docs used for this runbook:

- Volumes guide: https://modal.com/docs/guide/volumes
- Volume CLI reference: https://modal.com/docs/reference/cli/volume
- App and entrypoint guide: https://modal.com/docs/guide/apps
- Function reference: https://modal.com/docs/reference/modal.Function
