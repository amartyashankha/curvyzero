# 2026-05-09 Modal Volume Dummy Survival Smoke

## Question

Can a coarse CPU Modal job run the existing dummy survival training scaffold,
write its training/eval artifacts to a durable Modal Volume, commit them, and
return compact artifact refs?

## Setup

- Modal app: `curvyzero-volume-dummy-survival`.
- Entrypoint: `curvyzero.infra.modal.volume_dummy_survival`.
- Volume: `curvyzero-runs`.
- Mounted path in the container: `/runs`.
- Artifact layout:

```text
training/dummy-survival/volume-smoke/seed-<seed>/iterations-<iterations>/episodes-per-iter-<episodes_per_iter>/eval-episodes-<eval_episodes>/
```

This is intentionally a small persistence smoke, not a resume test or
orchestrator.

## Command

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.volume_dummy_survival \
  --iterations 1 \
  --episodes-per-iter 2 \
  --seed 0 \
  --eval-episodes 2
```

## Results

- Smoke completed successfully.
- Modal run URL:
  `https://modal.com/apps/modal-labs/shankha-dev/ap-vf2Dm8jJ9SkOxI0eWSPp7K`.
- Volume path:
  `training/dummy-survival/volume-smoke/seed-0/iterations-1/episodes-per-iter-2/eval-episodes-2`.
- Remote elapsed: 1.619024 seconds.
- Client elapsed: 4.716766 seconds.
- Final eval: 2 episodes, mean steps 4.0, max steps 4.
- Final eval crash rate: 1.0.
- Final eval survival rate: 0.0.
- Model summary: 11 tabular states, 12 learned dynamics edges.

Returned artifact refs:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `summary.json` | 1961 | `54914e5023cd47625578e8df56da1679e488bb875f9a2f70ede89ed712228595` |
| `checkpoint.npz` | 2003 | `3da1fb0e7b205edfe6086d43fea352b20b58b032db22bbed2b4b3a76aad7fab6` |
| `iteration_metrics.jsonl` | 480 | `55ce2292f2239fca4b2d343b2337a66d2e6cca3b36dac8419d15b262964e7403` |

## Interpretation

The dummy survival stack can write its normal training artifacts under a stable
Modal Volume path and return enough JSON metadata for downstream fetches or
manual inspection.

The deterministic path is useful for a small smoke, but it means rerunning the
same seed/iteration/episode tuple writes the same directory. A real training
attempt should add a run or attempt id plus resume/latest semantics.

## Artifacts

- Volume: `curvyzero-runs`.
- Volume path:
  `training/dummy-survival/volume-smoke/seed-0/iterations-1/episodes-per-iter-2/eval-episodes-2`.
- Artifact refs:
  - `training/dummy-survival/volume-smoke/seed-0/iterations-1/episodes-per-iter-2/eval-episodes-2/summary.json`
  - `training/dummy-survival/volume-smoke/seed-0/iterations-1/episodes-per-iter-2/eval-episodes-2/checkpoint.npz`
  - `training/dummy-survival/volume-smoke/seed-0/iterations-1/episodes-per-iter-2/eval-episodes-2/iteration_metrics.jsonl`

## Follow-ups

- Keep this wrapper tiny until checkpoint resume and run ids are needed.
- Reuse the single `curvyzero-runs` Volume for future coarse training/eval
  artifact smokes.
