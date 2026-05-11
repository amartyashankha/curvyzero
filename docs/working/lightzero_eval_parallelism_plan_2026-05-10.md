# LightZero Eval Parallelism Plan - 2026-05-10

Scope: eval queue performance only. Do not change eval meaning.

## Current Shape

The live queue is already parallel at two levels:

- `scripts/lightzero_live_eval_queue.py` prints Modal eval commands with
  `--parallel`, so each Modal run enters the parallel branch in
  `curvyzero.infra.modal.lightzero_pong_eval_smoke`.
- Inside that branch, `lightzero_pong_eval_smoke.main()` calls
  `eval_fn.starmap(...)` over the checkpoint/seed jobs, so jobs in one Modal
  call fan out to independent remote workers.
- The queue helper also has local process fan-out: `--execute` plus
  `--max-parallel-launches N` keeps up to `N` separate `modal run` commands in
  flight.

The current recipe defaults to `--group-size 4`, so one Modal app call can
submit several checkpoint/seed jobs while writing one combined manifest. Use
`--group-size 1` only when fastest first-checkpoint signal matters more than
repeated Modal app startup.

## Remaining Per-Checkpoint Cost

Each checkpoint still does real work, even when launched alone:

- load checkpoint from the Modal Volume;
- compile patched LightZero config and instantiate `MuZeroPolicy`;
- strict-load the policy model;
- create a real ALE Pong evaluator env;
- run the stock `lzero.worker.MuZeroEvaluator` episode up to the cap;
- write the per-checkpoint artifact and commit the Volume.

With `--num-simulations 50`, each environment step performs 50 MuZero MCTS
simulations. A `2048` cap is therefore about 102,400 MCTS simulations for one
stock evaluator episode. That dominates the runtime; Modal app/container
startup, checkpoint/model/env setup, Volume reads/writes, and Volume commits add
noticeable fixed overhead around it.

The older broad-wave path also ran a manual rollout before the stock evaluator.
That duplicated the policy/env/MCTS work for every checkpoint/seed. The queue
helper now defaults to `--stock-only`, which skips the manual rollout and keeps
the stock evaluator fields (`stock_steps_survived`, `stock_return`, reward
counts, artifacts, and manifests) as the primary readout. Use `--no-stock-only`
only for parity/debug evals where manual actions, manual/stock mismatch, or
observation/action-mask traces matter.

## Default Eval Wave

Use aggressive queue fan-out by default:

- `--compute gpu-l4-t4-cpu40`
- `--group-size 4`
- `--max-parallel-launches 64`
- `--stock-only`
- `--slim-manifest`

This is a resource/tooling setting. It does not change checkpoint selection,
strict load, seed, step cap, stock evaluator use, artifacts, or metric meaning.
The slim manifest setting only keeps combined root manifests compact; full
per-checkpoint artifacts are still written under the eval root.

Recommended command shape:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run-id> \
  --attempt-id <attempt-id> \
  --eval-id <eval-id> \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --update-per-collect -1 \
  --seed 0 \
  --stock-only \
  --group-size 4 \
  --max-parallel-launches 64 \
  --slim-manifest \
  --execute
```

`--max-parallel-launches 64` is a ceiling, not a minimum. If only 9 checkpoints
are pending after duplicate filtering, the helper launches only 9 Modal calls.

Limits and risks:

- Modal account/GPU concurrency may queue or throttle if too many L4/T4 workers
  are requested at once.
- Cost scales close to linearly with concurrent workers.
- Concurrent Volume reads/writes should be safe because output dirs are
  checkpoint/step/seed keyed, but many simultaneous `runs_volume.commit()` calls
  can add tail latency.
- Local `modal run` subprocess logs can interleave. The recent
  `--summary-only` and `--quiet-framework-logs` flags make this tolerable.
- Combined root manifests can otherwise get bulky on 2048-step multi-seed
  curves. Keep `--slim-manifest` on for broad waves and fetch the raw
  per-checkpoint JSONs when full detail is needed.

Operational posture: start live eval waves at `64` with `--group-size 4` and
`--stock-only`. If Modal or local process limits bite, temporarily reduce the
launch ceiling and record that as capacity friction. Use `--no-stock-only` only
for narrow debug reruns.

## Stale CPU8/CPU16 Notes

Earlier notes about defaulting eval to `gpu-l4-t4-cpu8`, adding CPU16 eval
compute, or trying launch ceilings below 64 are stale for the current queue.
The attempted CPU64 path was also wrong: Modal rejected it because this
workspace's function CPU limit is 40 cores. The current default large-wave
posture is `--compute gpu-l4-t4-cpu40` with
`--max-parallel-launches 64`.

CPU8/CPU16 may still appear in historical run ids, training speed notes, or old
eval artifacts. Treat those as labels for past work, not current eval guidance.

## Reporting Signal

For Pong, lead with survival time:

- `stock_steps_survived`
- `delta_stock_steps_survived` versus same-run `iteration_0`
- `stock_survival_fraction`
- manual `steps_survived` when useful and clearly labeled

Score/return is secondary context. Include `stock_return`, manual `return`,
rewards, and action histograms after the survival read.

## Best Next Step

Fast trend read:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run-id> \
  --attempt-id <attempt-id> \
  --eval-id <eval-id>-telemetry \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 2 \
  --eval-seed-count 4 \
  --eval-speed-profile telemetry \
  --update-per-collect -1 \
  --stock-only \
  --group-size 8 \
  --max-parallel-launches 64 \
  --skip-eval-root-listing \
  --slim-manifest \
  --execute
```

This is the cheap curve read: 5 MCTS simulations/action, fewer seeds, compact
manifests, and one known-new eval-root listing skipped. If resuming the same
eval id after preemption or a partial wave, omit `--skip-eval-root-listing` so
completed checkpoint/seed dirs are skipped.

Serious confirmation read:

```sh
uv run python scripts/lightzero_live_eval_queue.py \
  --run-id <run-id> \
  --attempt-id <attempt-id> \
  --eval-id <eval-id>-serious \
  --compute gpu-l4-t4-cpu40 \
  --eval-pass custom \
  --max-eval-steps 2048 \
  --max-episode-steps 2048 \
  --step-detail-limit 8 \
  --eval-seed-count 16 \
  --eval-speed-profile serious \
  --update-per-collect -1 \
  --stock-only \
  --group-size 4 \
  --max-parallel-launches 64 \
  --slim-manifest \
  --execute
```

This is the proof path: 50 MCTS simulations/action. Report survival first; use
return only as secondary context.
