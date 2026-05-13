# Checkpoint Tournament Discovery Handoff

Date: 2026-05-13

Short version: checkpoint discovery must scan all LightZero experiment
directories, not only `train/lightzero_exp/ckpt`.

## Footgun

DI-engine can mutate `cfg.exp_name` inside `compile_config(...)` when the target
directory already exists. In that case the active training directory becomes
something like:

```text
train/lightzero_exp_260513_123802/ckpt
```

while CurvyZero's status/poller code may still look at:

```text
train/lightzero_exp/ckpt
```

This can make a run look stuck at `iteration_0` even though later checkpoints
exist in the timestamped directory.

## Concrete Example

Run:

```text
curvy-mix2clean-r50-scr50-rf-s8-c32-l32-repH-k10-c1-s2104011
```

The fixed path had only:

```text
train/lightzero_exp/ckpt/iteration_0.pth.tar
```

But this timestamped path had checkpoints through `iteration_180000`:

```text
train/lightzero_exp_260513_123802/ckpt/
```

## Tournament Rule

For policy tournaments, discover candidates by scanning:

```text
train/lightzero_exp*/ckpt/iteration_*.pth.tar
```

Then take the highest valid numbered checkpoint per run, unless the tournament
is intentionally evaluating an earlier checkpoint.

Do not rank or select a run from `train/lightzero_exp/ckpt` alone.

Current nuance: the Modal tournament discovery helper already uses this broad
shape when it discovers from run roots:

```text
src/curvyzero/infra/modal/curvyzero_checkpoint_tournament.py
```

Look for the `train_root.glob("lightzero_exp*/ckpt")` scan. That part is good.
The remaining footgun is any caller or prep script that bypasses discovery and
passes a fixed `train/lightzero_exp/ckpt/...` checkpoint ref, or anything that
relies on the stable mirror/status/poller output from the trainer app.

This was verified with a targeted `--mode discover` run on the six rows that
fixed-path status showed as `iteration_0`; broad discovery found all six and
returned timestamped `lightzero_exp_260513_*` checkpoint refs.

## Main Investigation Doc

Use this as the source of truth:

```text
docs/working/training/curvytron_architecture_research_2026-05-12/stale_checkpoint_bug_investigation_2026-05-13.md
```

Relevant code in:

```text
src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py
```

Watch these functions:

- `_prepare_lightzero_auto_resume`
- `_latest_lightzero_iteration_checkpoint`
- `_write_checkpoint_progress_latest`
- `_run_checkpoint_eval_poller`
- `_scan_lightzero_artifacts`

Also check any tournament input manifest or selector that directly stores
checkpoint refs. If it was built from the fixed status path, rebuild it from a
broad `lightzero_exp*/ckpt` scan before trusting the results.

Adjacent risk: opponent-mixture manifests can also store fixed checkpoint refs.
For example, `scripts/build_curvytron_opponent_mixture_manifest.py` has default
refs under `train/lightzero_exp/ckpt`. Those refs may be fine for old rows that
really saved there, but they should not be copied as a general pattern. Any new
manifest that selects "recent", "mid", or "old" checkpoints should use the same
broad discovery rule before freezing refs.
