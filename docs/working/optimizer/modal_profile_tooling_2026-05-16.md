# Modal Profile Tooling

Date: 2026-05-16

Purpose: keep CurvyTron optimizer profiling boring. This is the small, current
way to launch, collect, and summarize profile grids without losing results in
Modal app lifecycle weirdness.

## Plain Rule

Use one of two patterns:

1. Direct blocking rows for small or medium grids.
2. Detached parent plus `--profile-spawn` for background grids.

Do not use `--profile-spawn` without `--detach`. That can print a function call
id and then let the ephemeral parent app stop before the child finishes. We hit
that failure on 2026-05-16.

Current manifest rows must use the explicit Modal entrypoint:

```text
curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train::main
```

The manifest runner now preflights this, requires `--manifest`, normalizes
numeric row selectors like `--rows 1` to `001`, and writes a result record even
when a launch record is missing a function call id.

## Current Tools

Hybrid boundary carve-out:

```text
For scripts/build_curvytron_hybrid_observation_profile_grid.py manifests, do not
run commands.sh directly and do not add --detach for single smokes. Use the
durable runner:

uv run python scripts/run_curvytron_hybrid_observation_profile_manifest.py \
  --manifest artifacts/local/curvytron_hybrid_observation_profile_manifests/<id>/manifest.json
```

Why: raw `modal run --detach` can leave only an "App completed" line locally.
The durable runner uses blocking Modal commands, captures stdout, extracts the
profile JSON, and writes:

```text
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/row_<id>_stdout.log
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/row_<id>_result.json
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/rows.jsonl
artifacts/local/curvytron_hybrid_observation_profile_results/<id>/collected_results.json
```

Build a flexible manifest:

```text
uv run python scripts/build_curvytron_profile_grid.py ...
```

Run or collect a manifest:

```text
uv run python scripts/run_curvytron_optimizer_profile_manifest.py \
  --manifest artifacts/local/curvytron_optimizer_profile_manifests/<id>.json \
  --action launch-and-collect
```

Summarize local result JSON files:

```text
uv run python scripts/summarize_curvytron_optimizer_profile_results.py \
  --results-dir artifacts/local/curvytron_optimizer_profile_results/<id>
```

## Current-Surface Grid Example

This is the next useful grid for the Coach hardware/batch recommendation:

```text
uv run python scripts/build_curvytron_profile_grid.py \
  --experiment-id opt-current-hw-batch-20260516a \
  --family current_hw_batch \
  --run-prefix optcur0516a \
  --attempt-prefix profcur0516a \
  --seed 309 \
  --computes gpu-l4-t4-cpu40,gpu-h100-cpu40 \
  --collectors 128,256 \
  --batch-sizes 32,64 \
  --num-simulations 8 \
  --source-max-steps 512 \
  --max-train-iter 96 \
  --max-env-step 200000 \
  --save-ckpt-after-iter 999999 \
  --stop-after-learner-train-calls 12 \
  --env-telemetry-stride 256 \
  --disable-death-for-profile
```

It writes:

```text
artifacts/local/curvytron_optimizer_profile_manifests/opt-current-hw-batch-20260516a.json
```

This grid means:

- current trusted stock path;
- current policy surface: `browser_lines + simple_symbols`;
- current reliable backend: `cpu_oracle`;
- no LightZero eval, no background eval, no GIF;
- no expected checkpoint write because `save_ckpt_after_iter=999999`;
- no-death profile so long trajectories stress collection/render/search.

## Canary

The tooling canary passed:

```text
artifacts/local/curvytron_optimizer_profile_results/opt-tooling-canary-20260516a
```

Result:

| row | C | sims | death | steps | wall | steps/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `001` | 8 | 2 | no-death | 512 | 10.22 | 50.10 |

This only proves tooling. It is not a speed recommendation.

## What To Tell Coach Right Now

Use L4 for the next broad run:

```text
gpu-l4-t4-cpu40
collector_env_num=256
n_episode=256
batch_size=64
num_simulations=8
browser_lines + simple_symbols + cpu_oracle
```

Why: the completed current-surface grid below measured best L4 `713.83` steps/s
and best H100 `1001.94` steps/s. L4 throughput is about `28.8%` lower than H100,
inside the concern threshold and probably worth it for cheaper broader
experiments.

Important caveat: batch64 helped the L4/C256 row and hurt H100. This is not a
universal "batch64 is better" rule.

Do not tell Coach that GPU observation rendering is active. It is not. These
rows use CPU `cpu_oracle` observations while LightZero model/search/learner use
the requested GPU.

## Next Rows

Completed generated `opt-current-hw-batch-20260516a` grid:

- L4 vs H100;
- C128 vs C256;
- batch32 vs batch64;
- sim8;
- no-death512.

Results:

| row | compute | collectors | batch | steps | wall | steps/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `001` | L4/T4 CPU40 | 128 | 32 | 65,536 | 110.82 | 591.39 |
| `002` | L4/T4 CPU40 | 128 | 64 | 65,536 | 112.40 | 583.06 |
| `003` | L4/T4 CPU40 | 256 | 32 | 131,072 | 217.76 | 601.92 |
| `004` | L4/T4 CPU40 | 256 | 64 | 131,072 | 183.62 | 713.83 |
| `005` | H100 CPU40 | 128 | 32 | 65,536 | 86.77 | 755.28 |
| `006` | H100 CPU40 | 128 | 64 | 65,536 | 125.11 | 523.83 |
| `007` | H100 CPU40 | 256 | 32 | 131,072 | 130.82 | 1001.94 |
| `008` | H100 CPU40 | 256 | 64 | 131,072 | 165.15 | 793.63 |

Plain read:

- L4 is not catastrophically slower. Best L4 row is `713.83` steps/s; best H100
  row is `1001.94` steps/s. L4 is about `28.8%` slower than H100 on best row,
  inside the `40%` concern threshold.
- On L4, C256/batch64 was the best row. C128/batch32 and C128/batch64 were
  nearly tied, and C256/batch32 was only slightly above C128. This says batch64
  is worth using specifically for the C256/L4 shape, but not as a global rule.
- On H100, batch64 was worse than batch32 at both C128 and C256.
- Recommendation to Coach for an L4 run: use `gpu-l4-t4-cpu40`,
  `collector_env_num=256`, `n_episode=256`, `batch_size=64`,
  `num_simulations=8`, current `browser_lines + simple_symbols + cpu_oracle`,
  and sparse checkpoint/eval cadence. Keep a C128/batch32 row as the cheap
  fallback if stability or resource pressure matters more than throughput.
