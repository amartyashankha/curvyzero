# Modal Training Run Lifecycle Footguns - 2026-05-11

Purpose: compact operating pattern for long Modal LightZero training runs after
the `s111`/`s112` and `s130`-`s133` no-root/no-evidence launches.

## Short Rule

A Modal launch is not a training run until the `curvyzero-runs` Volume shows
`train/progress/latest.json` or a checkpoint root. A no-wait
`Function.spawn` call id is only a scheduling receipt.

Prefer `--wait-for-train` for evidence-producing long runs. If no-wait spawn is
used for capacity reasons, poll the Volume once soon after launch and relaunch
with the wait pattern if no progress path appears after the grace window.

## Safe Launch Pattern

Use a unique `run_id` and `attempt_id`, include the surface in the ids, and run
the wrapper with `--wait-for-train` from a long-lived shell/session:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_exact_reproduction \
  --mode train \
  --compute gpu-h100-cpu40 \
  --seed 142 \
  --run-id lz-visual-pong-replication-matrix-20260511-s142-h100-repeat \
  --attempt-id train-stock-surface-100k-ckpt1000-h100cpu40-detached-wait \
  --progress-interval-sec 180 \
  --max-env-step-override 100000 \
  --save-ckpt-after-iter-override 1000 \
  --wait-for-train
```

For exact stock-cadence controls, omit `--max-env-step-override` and
`--save-ckpt-after-iter-override`. Exact stock controls may show only
`iteration_0` and `ckpt_best` for a while; that can be normal. For survival
curve scouting, keep the stock surface but use a shorter env-step cap and
`--save-ckpt-after-iter-override 1000` so `1k/2k/...` checkpoints become
observable.

The wrapper writes under:

```text
training/lightzero-official-visual-pong/<run_id>/attempts/<attempt_id>/train
```

Expected refs:

```text
.../train/progress/latest.json
.../train/lightzero_exp/ckpt
.../train/summary.json
```

## Immediate Verification

First use the status helper:

```bash
uv run python scripts/lightzero_replication_status.py
```

For a new run, check the exact Volume refs directly:

```bash
uv run --extra modal modal volume get curvyzero-runs \
  training/lightzero-official-visual-pong/<run_id>/attempts/<attempt_id>/train/progress/latest.json -
```

```bash
uv run --extra modal modal volume ls --json curvyzero-runs \
  training/lightzero-official-visual-pong/<run_id>/attempts/<attempt_id>/train/lightzero_exp/ckpt
```

Minimum "alive" evidence:

- `progress/latest.json` exists with `phase` `starting` or `running`; or
- checkpoint dir exists with `iteration_0.pth.tar` and usually `ckpt_best.pth.tar`.

Minimum "evaluable" evidence:

- same-run `iteration_0.pth.tar` exists;
- at least one later checkpoint exists for a curve claim, such as
  `iteration_5000.pth.tar`, `iteration_10000.pth.tar`, or
  `iteration_12000.pth.tar`;
- strict eval records the same checkpoint refs and does not use fallback model
  loading.

For no-wait `spawn` launches, poll after a short grace period. If both
`progress/latest.json` and `lightzero_exp/ckpt` are absent, label the row
`spawned/no-evidence`, not `running`.

## What Not To Trust

Do not trust these as proof that training is alive:

- a local JSON line with `"status": "spawned"`;
- a Modal `function_call_id`;
- an active Modal app count;
- a command returning before any Volume path exists;
- a run name in a planning table;
- a missing path immediately after launch as a policy failure;
- a failed Modal image build as a model result;
- score alone while survival steps are moving.

The `s130`-`s133` rows had no visible Volume checkpoint roots after spawn and
were correctly treated as no-evidence. The `s111`/`s112` names were not present
in the Volume at the audit point. By contrast, `s114`, `s120`, `s121`, and
`s122` became real rows only because the Volume showed progress and
`iteration_0`/later checkpoint files.

## Poll Cadence

Use Volume evidence, not local optimism.

- First poll: shortly after launch, enough to see `starting` or `iteration_0`.
- Live poll: about every 12 minutes for checkpoint inventory.
- Near-term gates: `5k`, `10k`, `20k`, `50k`, then `100k` if still running.
- Long budget reality: a `500k` run can be a one-to-two-day job. Do not wait
  for the final budget before checking survival movement.

Observed May 11 stock64 speeds were roughly `180`-`220` checkpoint iterations
per minute. H100 was only about `1.2x` faster than L4/T4+CPU40 in that sample,
so choose it for queue/capacity reasons rather than assuming dramatic speedup.

## Evaluation Pattern

Use same-run `iteration_0` as the baseline. The core early signal is strict
stock survival steps; score can remain near `-21` while survival improves.

Fast telemetry evals may use lower-cost settings only to decide where to look
next. Claim evals should use stock settings, fixed seeds, strict checkpoint
loading, and no fallback.

Do not mix surfaces:

- installed LightZero 0.2.0 stock64 `train_muzero`;
- Agent96 model-card lane;
- current GitHub upstream segment lane;
- survival-shaped ablations.

Each lane needs its own same-surface `iteration_0` comparison.

## Docs To Record

Every launch row should record:

- exact launch command;
- `run_id`, `attempt_id`, seed, compute, env-step cap, checkpoint cadence, and
  whether `--wait-for-train` was used;
- expected `progress/latest.json` ref;
- expected checkpoint dir ref;
- first Volume poll timestamp and result;
- first `iteration_0` timestamp, if present;
- latest checkpoint inventory;
- eval manifest refs and whether strict loading/no fallback was used;
- plain status: `spawned/no-evidence`, `starting`, `running`, `evaluable`,
  `completed`, or `failed`;
- interpretation boundary: stock64 claim, Agent96 claim, upstream segment,
  shaped ablation, or plumbing-only.

If a row has a call id but no Volume progress, write exactly that. Do not let it
silently become a failed policy row or a running row.

## Source Notes

- Failure audit: `docs/working/training/pong_replication_failure_audit_2026-05-11.md`
- Monitor: `docs/working/lightzero_pong_replication_monitor_2026-05-11.md`
- Launcher: `src/curvyzero/infra/modal/lightzero_pong_exact_reproduction.py`
