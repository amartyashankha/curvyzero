# LightZero Dummy Pong Next-Run Checklist - 2026-05-09

Use this only after the LightZero feature-fit/config-import smoke passes. The
tiny Modal LightZero MuZero train smoke has now passed:

```text
run_id: lz-dpong-20260509T141607Z-3696aa333028
attempt_id: attempt-20260509T141607Z-98662e4917b4
ok: true
called_train_muzero: true
algorithm: LightZero MuZero
env-side terminal rows: 5
random_uniform result: 4 wins, 1 loss, 0 truncations
mirrored checkpoints: ckpt_best.pth.tar, iteration_0.pth.tar, iteration_2.pth.tar
independent scorecard: blocked because current scoreboard cannot load .pth.tar
ckpt_best keys: last_iter, last_step, model, optimizer, target_model
policy head shape: (3, 32)
```

The command that produced the tiny run was:

```sh
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke \
  --mode progression \
  --env dummy_pong_lag1 \
  --feature-mode tabular_ego \
  --max-env-step 64 \
  --max-train-iter 2 \
  --num-simulations 2 \
  --batch-size 8 \
  --seed 0
```

## Inspect In Modal Output

- Top-level result says `algorithm: "LightZero MuZero"` and
  `called_train_muzero: true`.
- Top-level `ok` is `true`, `status` is `completed`, and `problems` is empty.
- `summary_ref`, `attempt_manifest`, `latest_attempt`, and `artifact_refs`
  point to `curvyzero-runs` Volume records.
- `train_result.ok` inside the summary is `true`, with LightZero log signals
  showing at least one `Training Iteration`, learner/evaluator metric mentions,
  and checkpoint-save evidence where available.
- `lightzero_artifacts.checkpoint_files` and
  `checkpoint_mirror.copied_checkpoints` show at least one LightZero checkpoint
  mirrored into the CurvyZero run tree.
- `artifact_refs.summary`, `artifact_refs.episodes`,
  `artifact_refs.training_signals`, and `artifact_refs.lightzero_artifacts`
  exist and have nonzero sizes where expected.
- `pong_scorecard` is present and includes wins, losses, score return, mean and
  median survival steps, p90 survival steps, survival standard deviation,
  truncation rate, shaped loss-delay return, and shaped-return standard
  deviation.
- `episodes.jsonl` contains CurvyZero env-side episode rows with seed, step,
  joint action, winner, terminated/truncated, reward/score, opponent policy id,
  and trace metadata visible enough for later debugging.
- `independent_scorecard.status` is expected to be `blocked` for this smoke
  until standalone CurvyZero inference for LightZero checkpoints exists; note it
  plainly instead of treating LightZero evaluator output as an independent
  scorecard.

## Tiny Train Pass/Fail

Pass if this tiny run calls LightZero's real MuZero trainer on the custom dummy
Pong env, finishes within the tiny caps, returns `ok: true`, writes durable
summary and artifact refs, emits learner/evaluator signals, mirrors at least
one LightZero checkpoint, and preserves the Pong telemetry bundle in CurvyZero
artifacts.

Fail if LightZero training crashes, `called_train_muzero` is false, the run
silently falls back to a stock LightZero env, required Pong telemetry is absent,
checkpoint discovery or mirroring fails, run records cannot be mapped into
`curvyzero-runs`, or the adapter hides seed/action/reward traces needed for the
later CurvyZero scorecard.

Treat `independent_scorecard.status: "blocked"` as an explicit limitation, not
as a run crash. It still blocks any claim that the learned checkpoint has been
validated outside LightZero.

## What Not To Claim

- Do not claim project-owned Pong MuZero has run. This is LightZero MuZero.
- Do not claim CurvyZero/CurvyTron MuZero has run.
- Do not claim the policy learned Pong from a tiny capped smoke.
- Do not claim the mirrored `.pth.tar` checkpoints were independently scored.
- Do not report only wins or a win fraction.
- Do not treat survival, shaped loss-delay return, or variance as the real
  environment reward.
- Do not claim independent checkpoint evaluation until a CurvyZero standalone
  scorecard can load and score the LightZero checkpoint.
- Do not compare this directly against CEM-v2 or raster-only MLP as if they are
  the same algorithm. They are baselines, not MuZero.
