# 2026-05-08 Modal Dummy Line Duel Smoke

## Question

Can a tiny CPU Modal Function import `curvyzero.training.dummy_line_duel` and run the existing dummy line-duel training CLI-equivalent?

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.dummy_line_duel --iterations 1 --episodes-per-iter 2 --seed 0 --eval-episodes 2
```

## Results

- Smoke completed successfully.
- Modal app: `curvyzero-dummy-line-duel`.
- Run URL: `https://modal.com/apps/modal-labs/shankha-dev/ap-WM8c9YEPStqnz6oWmH7WFy`
- Client elapsed: 2.998249 seconds.
- Remote elapsed: 0.038942 seconds.
- Final eval: 2 episodes, 2 `player_0` wins, 0 `player_1` wins, 0 draws, 0 truncations.
- Final eval mean steps: 6.0.
- Model summary: 11 tabular states, 14 learned dynamics edges.

## Artifacts

Artifacts were written to the remote ephemeral filesystem only; no Modal Volume was attached.

- `/tmp/artifacts/curvyzero/dummy_line_duel/seed-0-iters-1-episodes-2/summary.json`
- `/tmp/artifacts/curvyzero/dummy_line_duel/seed-0-iters-1-episodes-2/checkpoint.npz`
- `/tmp/artifacts/curvyzero/dummy_line_duel/seed-0-iters-1-episodes-2/iteration_metrics.jsonl`
- `/tmp/artifacts/curvyzero/dummy_line_duel/seed-0-iters-1-episodes-2/replay_rows.jsonl`
