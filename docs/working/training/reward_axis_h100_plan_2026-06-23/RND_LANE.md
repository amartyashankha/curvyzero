# RND Lane

Status: implementation exists; successful learning result is not yet proven.

## What Happened To RND

RND did not disappear. It was implemented, tested as plumbing, and then left in
a cautious state because the evidence stopped short of "positive RND improves
training quality."

Current facts:

- Core implementation:
  `src/curvyzero/training/exploration_bonus.py`
- Manifest builder:
  `scripts/build_curvytron_rnd_blank_sweep_manifest.py`
- Historical planning and run notes:
  `docs/working/training/exploration_bonus_rnd_2026-05-19/`
- Compact-owned reward contract rejects RND:
  `src/curvyzero/training/compact_reward_rnd_contract.py`

The previous status is best summarized as: RND was real enough to test and
diagnose, but not real enough to promote.

The old "positive RND is blocked" posture should now be read precisely:
positive RND was blocked for recommendation, not because the implementation was
absent. Under this H100 plan, positive RND is allowed as a controlled,
stock-and-meter-controlled experiment. It remains blocked for promotion until
retained extrinsic quality and fixed-opponent transfer are shown.

## Most Believable Implementation

The credible path is the existing source-state visual RND implementation:

| Mode | Effect | How to read it |
| --- | --- | --- |
| `none` | Stock extrinsic target only | Baseline. |
| `rnd_meter_v0` | Trains/logs RND metrics, leaves reward target unchanged | Instrumentation/control row. |
| `rnd_replay_target_v0` | Adds normalized intrinsic reward to replay target | Real positive RND experiment. |

Important details:

- Feature source: `policy_gray64_latest/v0`, derived from the latest visual
  stack into a single gray 64x64 frame.
- Predictor/target architecture is internal to `CurvyRNDRewardModel`; the
  target network is frozen and the predictor is trained online.
- Positive RND mutates the learner target reward. It must be judged against
  stock and meter controls, not against raw trainer reward.
- Compact training currently has an extrinsic-only no-RND contract. RND should
  stay in the source-state visual lane until that contract changes.

## Recommended First RND Sweep

Use the existing no-tournament blank-canvas builder. This lane answers a narrow
question: does intrinsic novelty pressure produce better early exploration and
survival than stock training under a controlled blank-canvas setup?

Wide H100 sweep:

```bash
uv run --extra modal python scripts/build_curvytron_rnd_blank_sweep_manifest.py \
  --profile rnd_blank_sweep \
  --matrix-name rnd-blank-h100-wave-a-20260623a \
  --compute gpu-h100-cpu40 \
  --replicas 5 \
  --weights 0.003 0.01 0.03 0.10 0.30 0.60 1.00 \
  --rnd-update-per-collect 100 \
  --save-ckpt-after-iter 2500 \
  --background-eval-seed-count 8
```

Expected row count: 45 rows.

Saved submitter dry-run:

```text
artifacts/local/curvytron_rnd_blank_sweep_manifests/rnd-blank-h100-wave-a-20260623a/rnd-blank-h100-wave-a-20260623a.submit.dryrun.json
```

Dry-run result: `dry_run=true`, `selected_row_count=45`,
`assignment_write_count=0`, `refresh_pointer_write_count=0`, and all rows use
`lightzero_curvytron_visual_survival_h100_cpu40`.

The RND sweep is also part of the repaired Wave A packet audit:

```text
artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json
```

Current packet result: `ok=true`, with `45` RND rows and `45` non-RND rows. Do
not launch positive RND without healthy stock, meter, and non-RND comparison
lanes.

Optional fast meter gate:

```bash
uv run --extra modal python scripts/build_curvytron_rnd_blank_sweep_manifest.py \
  --profile rnd_blank_meter_gate \
  --matrix-name rnd-meter-h100-gate-20260623a \
  --compute gpu-h100-cpu40 \
  --replicas 3 \
  --rnd-update-per-collect 100 \
  --save-ckpt-after-iter 2500 \
  --background-eval-seed-count 8
```

Expected row count: 6 rows.

The full sweep already contains stock and meter rows. The meter gate is only for
an even faster health check if we want to validate metrics before launching all
positive weights.

Historical caution: the 2026-05-19 fast-checkpoint RND sweep proved startup,
checkpoint cadence, eval/GIF flow, and JSONL RND metrics beyond iteration 0. It
did not prove that positive RND improves policy quality. This Wave A sweep adds
replicas so seed noise can be separated from real signal.

Cadence caution: pass `--rnd-update-per-collect 100` explicitly. The builder's
low default is useful for diagnostic plumbing, but the old RND notes identify
around `100` as the serious training cadence unless cadence itself is the
ablation.

## Signals To Watch

Early operational signal, first 30 minutes:

- RND rows write `rnd_reward_model_metrics_latest.json`.
- Stock rows do not require RND metrics.
- Predictor loss and intrinsic reward stats are finite.
- Background eval poller evaluates nonzero checkpoints.
- Action distribution is not immediately collapsed.

First learning signal, roughly 30k-50k:

- Low weights, especially `0.003`, `0.01`, or `0.03`, improve best-so-far or AUC
  over both stock and meter rows.
- Meter rows look like stock rows. If meter changes behavior, the measurement
  path is not passive.
- High weights do not dominate only by producing erratic action noise.

Useful decision signal, roughly 100k-170k:

- Positive RND has higher survival AUC than stock and meter across replicas.
- Best checkpoint is not a one-off spike.
- Latest remains within the same retention band as the best stock control.
- Intrinsic reward scale does not swamp extrinsic reward components.

Retention signal, roughly 240k-300k:

- Best positive RND weight keeps latest near best.
- The winning weight is reproducible across at least two replicas.
- Action-collapse and GIF review do not show degenerate novelty seeking.

## Promotion Gate

Promote RND only if a positive weight beats both `none` and `rnd_meter_v0` on
most of:

- survival best-so-far
- survival AUC
- latest-vs-best retention
- action-collapse rate
- finite, stable RND metrics
- reproducibility across replicas

Blank-canvas promotion means "RND deserves the next experiment," not "RND is
production reward." The next experiment should add fixed opponents or tournament
exposure with a builder that makes assignment refresh and checkpoint refs
explicit.

## Failure Modes

- Metric-only success: RND metrics look healthy but survival is unchanged.
- Novelty collapse: predictor learns too quickly and intrinsic reward vanishes.
- Overpowering intrinsic scale: high weights improve novelty but damage
  survival or action stability.
- Blank-canvas false positive: RND works against noop opponents but not against
  real checkpoint mixtures.
- Resume gap: RND model state is not preserved or compared correctly across
  resumed attempts.
