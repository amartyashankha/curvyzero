# 2026-05-08 Modal Dummy Survival Smoke

## Question

Can a tiny CPU Modal Function import `curvyzero.training.dummy_survival` and run the existing dummy survival training CLI-equivalent?

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_survival --iterations 1 --episodes-per-iter 2 --seed 0 --eval-episodes 2
```

## Result

- Smoke completed successfully.
- Modal app: `curvyzero-dummy-survival`.
- Run URL: `https://modal.com/apps/modal-labs/shankha-dev/ap-mc0EKw9CeAsjxmgWpLQMoM`
- Client elapsed: 3.744544 seconds.
- Remote elapsed: 0.024301 seconds.
- Final eval: 2 episodes, mean steps 4.0, max steps 4.
- Final eval crash rate: 1.0.
- Final eval survival rate: 0.0.
- Model summary: 11 tabular states, 12 learned dynamics edges.

## Artifacts

Artifacts were written to the remote ephemeral filesystem only; no Modal Volume was attached.

- `/tmp/artifacts/curvyzero/dummy_survival/seed-0-iters-1-episodes-2/summary.json`
- `/tmp/artifacts/curvyzero/dummy_survival/seed-0-iters-1-episodes-2/checkpoint.npz`
- `/tmp/artifacts/curvyzero/dummy_survival/seed-0-iters-1-episodes-2/iteration_metrics.jsonl`
