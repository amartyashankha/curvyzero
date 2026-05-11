# CurvyTron Checkpoint Eval + Inspection Handoff - 2026-05-11

Purpose: tell the coach what plumbing now exists and where to look.

## What Changed

Checkpoint observability is now a background pipeline:

1. The trainer runs normally.
2. A separate Modal poller watches the LightZero checkpoint directory.
3. For each stable `iteration_*.pth.tar`, the poller spawns a separate eval +
   inspection job.
4. The eval job writes the eval artifact, manifest, and inspector reports.

The trainer loop does not run eval/inspection and does not commit the Volume in
the checkpoint loop.

## What The Coach Can Read

For each checkpoint, look under the attempt eval directory:

```text
training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/eval/live_checkpoint_iteration_*/ 
```

Useful files:

- `manifest_*.json`: eval table and per-seed artifact refs
- `inspection/report.json`: structured inspector report
- `inspection/report.md`: coach-readable inspector report

The poller status is here:

```text
training/lightzero-curvytron-visual-survival/<run_id>/attempts/<attempt_id>/train/checkpoint_eval_poller.json
```

## Proof Run

Run:

```text
run_id: curvytron-inspector-poller-sanity-20260511
attempt_id: train-gpu100-poller-ckpt50-20260511
```

The poller scheduled and completed eval+inspection for:

- `iteration_0`: 14 steps, death cause `own_trail`
- `iteration_50`: 17 steps, death cause `wall`
- `iteration_100`: 16 steps, death cause `wall`
- `iteration_105`: 17 steps, death cause `wall`

This proves the plumbing works. It does not prove learning.

## Relevant Files

- `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  - trainer launcher
  - checkpoint poller
  - eval+inspection Modal worker
- `src/curvyzero/infra/modal/lightzero_curvytron_visual_survival_eval.py`
  - checkpoint eval harness
- `src/curvyzero/training/curvytron_inspector.py`
  - inspector report builder
- `src/curvyzero/training/curvytron_visual_survival_replay_inspector.py`
  - replay/death inspection helper
- `tests/test_curvytron_live_checkpoint_eval_plumbing.py`
  - poller and launcher tests
- `tests/test_curvytron_visual_survival_replay_inspector.py`
  - replay inspector tests

## Operational Notes

Use `--background-eval-enabled` to turn this on. The default launch kind is the
separate poller.

If launching a background training run without `--wait-for-train`, use Modal
detached mode so spawned background work survives:

```text
modal run --detach ...
```

With `--wait-for-train`, the launcher waits for the trainer and the poller, which
is the cleanest smoke-test path.

## Cautions

- Current eval is still a fixed-opponent survival read, not self-play proof.
- These reports should be used to explain what happened to each checkpoint, not
  to claim the policy learned without baselines and stronger evals.
- The eval itself can change later; the plumbing should stay useful.
