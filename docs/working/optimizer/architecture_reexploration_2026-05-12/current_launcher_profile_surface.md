# Current Launcher Profile Surface

> [!IMPORTANT]
> Superseded/archive note (2026-05-15): production policy observation is CPU `cpu_oracle` `browser_lines + simple_symbols`; GPU `browser_lines + simple_symbols` remains lab-only until trainer contract parity. `body_circles_fast` is historical/control only.

Date: 2026-05-12

Scope: read-only audit of the current CurvyTron LightZero Modal launcher and
profile summarizer. No Modal jobs were launched.

## Use This Path Now

Use the canonical Modal module:

```text
curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train
```

For optimizer timing profiles, use `--mode profile`. This still builds the
stock LightZero config and calls `lzero.entry.train_muzero`, but installs the
optimizer phase profiler and enables profile env timing. The trusted frozen
opponent surface is:

```text
--mode profile
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar
--no-opponent-use-cuda
```

Tiny base-manager attribution template:

```text
uv run --extra modal modal run --quiet -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute gpu-l4-t4 \
  --seed 304 \
  --run-id opt-stock-frozen-l4-base-c1-b16-sim8-browser-s304-YYYYMMDDx \
  --attempt-id profile-base-c1-b16-sim8-browser \
  --env-variant source_state_fixed_opponent \
  --reward-variant sparse_outcome \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar \
  --no-opponent-use-cuda \
  --env-manager-type base \
  --collector-env-num 1 \
  --n-episode 1 \
  --batch-size 16 \
  --num-simulations 8 \
  --source-max-steps 256 \
  --lightzero-eval-freq 0 \
  --skip-lightzero-eval-in-profile \
  --save-ckpt-after-iter 9999 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --output-detail compact
```

Subprocess worker-timing template:

```text
uv run --extra modal modal run --quiet -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_train \
  --mode profile \
  --compute gpu-l4-t4-cpu40 \
  --seed 304 \
  --run-id opt-stock-frozen-l4cpu40-subproc-c4-b16-sim8-nodeath-browser-worker-s304-YYYYMMDDx \
  --attempt-id profile-subproc-c4-b16-sim8-nodeath-browser \
  --env-variant source_state_fixed_opponent \
  --reward-variant sparse_outcome \
  --opponent-policy-kind frozen_lightzero_checkpoint \
  --opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar \
  --no-opponent-use-cuda \
  --env-manager-type subprocess \
  --collector-env-num 4 \
  --n-episode 4 \
  --batch-size 16 \
  --num-simulations 8 \
  --source-max-steps 256 \
  --disable-death-for-profile \
  --env-telemetry-stride 1 \
  --lightzero-eval-freq 0 \
  --skip-lightzero-eval-in-profile \
  --save-ckpt-after-iter 9999 \
  --no-background-eval-enabled \
  --no-background-gif-enabled \
  --output-detail compact
```

Use `--mode train` only when the goal is a real stock training run. The current
dry-run matrix builder for review-only train commands is
`scripts/build_curvytron_stock_train_manifest.py`; it refuses
`two-seat-selfplay`, restricts rows to stock source-state env variants, and
requires immutable frozen checkpoint refs.

## Avoid These Paths

Do not use `--mode two-seat-selfplay` for current optimizer evidence. It is a
historical custom adapter path, not the trusted learning/profile lane after the
May 12 pivot.

Do not run these historical launchers except for explicit postmortem
reproduction:

```text
scripts/launch_curvytron_overnight40_20260512.zsh
scripts/launch_curvytron_mixpast_20260512.zsh
```

Both scripts are guarded by `ALLOW_HISTORICAL_CUSTOM_TWO_SEAT_RERUN=1` and emit
`--mode two-seat-selfplay` commands. Their old frozen flags such as
`--two-seat-frozen-opponent-*` belong to the custom adapter, not the stock
frozen-opponent path.

Also avoid `source_state_turn_commit` for training. The launcher blocks train
mode for it because its pending/commit scalar rewards have untrusted credit
semantics. Treat `fixed_opponent`, `turn_commit`, and older stacked/debug names
as historical/control surfaces unless a specific audit says otherwise.

## Important Flags

Core stock surface:

- `--env-variant source_state_fixed_opponent`: trusted single-agent LightZero
  wrapper with an internal fixed/frozen opponent.
- `--opponent-policy-kind frozen_lightzero_checkpoint`: frozen checkpoint
  opponent.
- `--opponent-checkpoint-ref REF`: immutable checkpoint path, not `latest` or
  `ckpt_best`.
- `--no-opponent-use-cuda`: current safe default; keeps frozen checkpoint
  inference out of subprocess CUDA workers.
- `--reward-variant sparse_outcome` or `dense_survival_plus_outcome`: fixed
  opponent supports both. Use sparse for clean timing unless the run is
  specifically a dense-reward comparison.

Width/search/profile knobs:

- `--env-manager-type base|subprocess`: `base` gives best local attribution;
  `subprocess` is the scaling path and needs telemetry for worker internals.
- `--collector-env-num C` and `--n-episode C`: keep these aligned for
  subprocess width sweeps.
- `--num-simulations N`: live MuZero MCTS simulations per root; the frozen
  opponent config also receives this as opponent simulations.
- `--batch-size B`: learner batch size; also copied into frozen-opponent config.
- `--source-max-steps 256`: per-episode source tick cap and target `td_steps`.
- `--source-state-trail-render-mode browser_lines|body_circles_fast`:
  `browser_lines` is the richer source-state visual; `body_circles_fast` is the
  faster render-path comparison and should be labeled as a fidelity tradeoff.

Death and timing:

- Normal death is default.
- `--disable-death-for-profile` is allowed only with `--mode profile`; use it
  for long-survival Amdahl profiles.
- `--env-telemetry-stride 1` records sampled worker timing every env step; this
  matters most with `--env-manager-type subprocess`.
- `--profile-cuda-sync-enabled` makes CUDA timing more synchronous but adds
  overhead. Leave it off for throughput profiles; turn it on only for a timing
  attribution question.
- `--profile-allow-auto-resume` should stay off unless the run is intentionally
  profiling resumed state. Fresh `run_id`s are safer.

Eval, GIF, and checkpoint cadence:

- `--lightzero-eval-freq 0` suppresses periodic stock eval by patching cadence
  beyond `max_train_iter`, but stock initial eval is separate.
- `--skip-lightzero-eval-in-profile` skips the stock evaluator in profile mode;
  default is true.
- `--save-ckpt-after-iter 9999` avoids checkpoint saves in small profiles.
  Lower values intentionally add checkpoint, mirror, eval, or GIF surface.
- `--no-background-eval-enabled` disables CurvyZero background checkpoint eval.
- `--no-background-gif-enabled` disables self-play GIF generation and gif
  browser run-marker clutter.
- `--background-eval-launch-kind poller` is the default train-side background
  path; in profile mode, prefer disabling background eval entirely.

Compute and detach:

- `--compute cpu`, `cpu64`, `gpu-l4-t4`, `gpu-l4-t4-cpu40`,
  `gpu-h100-cpu40`, or `gpu-h100x2-cpu40`.
- Use `gpu-l4-t4` for small GPU/base attribution and `gpu-l4-t4-cpu40` for
  subprocess width sweeps that need CPU headroom.
- Keep `--no-opponent-use-cuda` with GPU compute unless the experiment is
  explicitly about opponent CUDA.
- Do not add Modal `--detach` for short readback profiles if the next step is
  immediate summary capture. Generated training manifests include `--detach` by
  default, with `--no-detach` available only in the manifest builder.

## Side-Effect Suppression

For clean optimizer profiles, include:

```text
--lightzero-eval-freq 0
--skip-lightzero-eval-in-profile
--save-ckpt-after-iter 9999
--no-background-eval-enabled
--no-background-gif-enabled
--output-detail compact
```

These suppress stock eval overhead, background checkpoint eval, GIF generation,
gif-browser marker clutter, and checkpoint-heavy artifact churn. They also keep
the profile summary focused on collect/search/learner/replay/env timing.

## Summarizing Profiles

Use:

```text
python3 scripts/summarize_curvytron_lightzero_profiles.py \
  --attempt RUN_ID:ATTEMPT_ID
```

The summarizer expects
`training/lightzero-curvytron-visual-survival/RUN_ID/attempts/ATTEMPT_ID/train/summary.json`
and reports the important parse columns: compute, manager, collectors,
episodes, opponent CUDA, batch, sims, steps/sec, collect, search, learner,
checkpoint, telemetry observation/opponent/vector-step timing, render mode,
death mode, and GPU sampling.

## Run ID Metadata

For 50+ jobs, every `run_id` and `attempt_id` should include enough metadata to
parse without opening JSON:

```text
opt-stock-frozen-{compute}-{mgr}-c{C}-b{B}-sim{S}-{death}-{render}-{reward}-oppcpu-s{seed}-{date}{suffix}
```

Minimum fields:

- lane: `opt-stock-frozen`;
- compute: `l4`, `l4cpu40`, `h100cpu40`, `cpu`, etc.;
- env manager and collector count: `base-c1`, `subproc-c32`;
- batch and simulations: `b16-sim8`;
- death mode: `normal` or `nodeath`;
- render mode: `browser` or `fast`;
- reward: `sparse` or `dense`;
- opponent device: `oppcpu` unless explicitly testing `oppcuda`;
- seed and date/suffix: for uniqueness and replayability.

Keep attempt IDs shorter but aligned, for example
`profile-subproc-c32-b16-sim8-nodeath-browser-s304`.
