# Learner Observability Gap, 2026-05-16

Scope: focused observability audit plus implementation. Initial subagent pass
was docs-only; main thread implemented and deployed the smallest safe patch at
2026-05-16 11:23 EDT.

## Question

The current runs produce checkpoints, tournament entries, background evals, and
GIFs, but we still cannot consistently answer a simpler question for every
active row:

> Is the LightZero learner actually updating, what losses/entropies is it
> seeing, and when did action behavior change?

The gap is not that no signal exists. The gap is that the signal is scattered
across checkpoint progress, final summaries, stderr tails, env telemetry, and
eval manifests instead of one durable machine-readable learner metric stream.

## Current Durable Artifacts

For the stock LightZero visual-survival trainer:

- `status_heartbeat.json`: live stage/status written before and during trainer
  setup. Useful for "is the job alive?", not learning.
- `progress_latest.json`: written by checkpoint hooks only. Current schema is
  `curvyzero_lightzero_curvytron_train_progress_latest/v0`; it records run,
  attempt, checkpoint ref/name, checkpoint iteration, `learner_train_iter`,
  elapsed time, source hook, checkpoint policy metadata sidecar, and optional
  own-checkpoint opponent refresh result. It does not currently include
  `last_learner`, losses, policy entropy, grad norm, or action counts.
  Relevant code: `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  lines 2064-2265.
- `progress.jsonl`: the status tool knows how to read it, but the current stock
  trainer path does not write it. It is used by the experimental two-seat path.
  The status reader expects `last_learner` there, but for stock runs this is
  usually absent/empty.
- `summary.json`: final artifact after the `train_muzero` call exits. It includes
  command/config, surface, target audit summary, action summary, phase profile,
  checkpoint mirror, and parsed text log signals. Not useful as live proof for
  long running jobs.
- `stdout_tail.txt` and `stderr_tail.txt`: written only after `train_muzero`
  returns or fails. The main trainer redirects stdout/stderr to in-memory
  buffers during training, then writes compact tails at the end. Relevant code:
  lines 4654-4790 and 5030-5054.
- `target_audit.json`: final passive collect/replay audit. It records collect
  call counts, replay sample counts, compact GameSegment samples, and compact
  replay target samples when accessible. It does not record every learner update.
  Relevant code: `_LightZeroTargetAudit` and `_install_lightzero_target_audit`.
- `action_observability.json`: final summary of `env_steps.jsonl`, including
  action histograms and terminal reasons. Live action telemetry exists in
  `env_steps.jsonl`, but the status path only gets a summarized view after the
  summary artifact is written.
- `opponent_assignment_refresh_events.jsonl`: live durable event stream for
  assignment refresh decisions. Good control-plane evidence, not learner loss
  evidence.
- checkpoint files and checkpoint metadata sidecars: durable proof that
  checkpoints were saved and what observation surface they use.

## What The Status Tool Expects

`lightzero_curvytron_run_status.py` already has the shape we want to feed:

- `_run_status` reads `progress_latest.json`, then looks for
  `progress["last_learner"]["model_parameters_changed"]`.
- `_progress_curve` reads `progress.jsonl`, then also looks for
  `last_learner.model_parameters_changed`.
- Both paths also look for progress-level action counts and survival fields:
  `mean_completed_episode_steps`, `completed_episode_count`,
  `survival_reward_sum`, `action_counts`, and `effective_action_noop_probability`.
  Relevant code: `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`
  lines 1096-1224.

So the reader is ahead of the writer: it expects a learner/action progress
surface that stock training currently does not populate.

## Where Learner Metrics Currently Come From

The only broadly visible learner metrics today are text logs in stderr tails.
Kant's latest read found stderr tails for only 7 active `r18fresh` rows and none
of the sampled static-control rows. Those tails showed nontrivial signal:

- policy entropy near `1.094`
- target policy entropy around `0.864`
- policy loss around `6.45`
- grad norm around `1.53`
- nonconstant target/predicted values in at least one dense row

That is useful evidence, but it is weak operationally because:

- the tails are compact text, not JSON;
- they are missing for many rows;
- they are only written at the end of the main trainer path;
- regexing stderr is brittle across LightZero/DI-engine versions;
- there is no stable per-iteration key for joining learner loss to checkpoints,
  assignment refreshes, eval points, and tournament promotion.

## Smallest Safe Patch

Add a passive learner metric writer around `BaseLearner.train` in the stock
trainer path.

Implementation status, 2026-05-16 11:23 EDT:

- Implemented `_LearnerMetricsRecorder` and
  `_install_lightzero_learner_metrics_recorder` in
  `src/curvyzero/infra/modal/lightzero_curvyzero_stacked_debug_visual_survival_train.py`.
- The trainer now writes `learner_metrics.jsonl` plus
  `learner_metrics_latest.json`, embeds `last_learner` into checkpoint progress
  when available, and preserves original train return/exception behavior.
- `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py` now falls back
  to `learner_metrics_latest.json` and reports train-call index, train iter
  before/after, collector envstep, elapsed time, and numeric metric count/map.
- Targeted tests cover successful train-result preservation, exception
  preservation, checkpoint progress embedding, and status fallback.
- Verification before deploy: `156 passed, 3 skipped`; ruff clean.

Implementation shape:

1. Add `_install_lightzero_learner_metrics_writer(...)` next to the existing
   checkpoint/progress hooks.
2. Find `BaseLearner` the same way `_install_checkpoint_progress_writer` and the
   phase profiler do.
3. Wrap `BaseLearner.train`:
   - record `train_iter` before and after;
   - time the call;
   - call the original method and return its exact result;
   - extract a compact numeric metric map from the returned object if it is a
     dict-like result;
   - also try compact known learner/log-buffer surfaces if present, without
     failing if they are absent;
   - do not parse stderr as the primary metric source.
4. Write sampled rows to
   `attempt_train_root / "learner_metrics.jsonl"` and update
   `attempt_train_root / "learner_metrics_latest.json"`.
5. Also attach the latest sampled row to `progress_latest.json` as
   `last_learner` whenever a checkpoint progress write happens.
6. Keep this non-blocking: all writer/extractor exceptions become an
   `observability_error` row or a printed warning; they must never stop training.

Suggested default sampling:

- Always record the first 5 learner calls.
- Then record every `1000` learner calls by default.
- Also force-record on checkpoint writes, so every checkpoint has a nearby
  learner row.
- Commit Volume writes on the existing checkpoint commit path, plus optionally
  every `1000` recorded metric rows. Do not commit every learner call.

Suggested row schema:

```json
{
  "schema_id": "curvyzero_lightzero_learner_metrics/v0",
  "run_id": "...",
  "attempt_id": "...",
  "created_at": "...",
  "learner_train_call_index": 1234,
  "train_iter_before": 49999,
  "train_iter_after": 50000,
  "train_iter_delta": 1,
  "collector_envstep": 1234567,
  "elapsed_sec": 0.0432,
  "result_type": "dict",
  "numeric_metrics": {
    "total_loss": 12.3,
    "policy_loss": 6.4,
    "value_loss": 2.1,
    "reward_loss": 0.8,
    "policy_entropy": 1.09,
    "target_policy_entropy": 0.86,
    "grad_norm": 1.5,
    "learning_rate": 0.0002
  },
  "model_update": {
    "hash_checked": false,
    "model_hash_before": null,
    "model_hash_after": null,
    "model_parameters_changed": null
  },
  "source": "BaseLearner.train"
}
```

Model-change proof should be optional and sampled. Reusing the existing
`_model_hash` pattern from
`src/curvyzero/training/curvyzero_stacked_debug_visual_survival_profile.py`
lines 1287-1303 is fine, but hashing the full model every learner update is too
expensive. Safer defaults:

- `learner_model_hash_interval_train_calls=0` by default, meaning disabled for
  ordinary long runs;
- force hash only in canary/debug runs, or every checkpoint if overhead is
  acceptable after measurement;
- in normal runs, use `train_iter_delta > 0`, nonzero loss rows, and changing
  checkpoint file hashes as the low-overhead learning evidence.

## Minimal Code Touch Points

Small patch, no algorithm change:

- `lightzero_curvyzero_stacked_debug_visual_survival_train.py`
  - add compact scalar extraction helpers;
  - add `_LearnerMetricsRecorder`;
  - add `_install_lightzero_learner_metrics_writer`;
  - install it alongside the existing resume/progress/live-publisher hooks;
  - pass the recorder/latest row into `_write_checkpoint_progress_latest`.
- `lightzero_curvytron_run_status.py`
  - read `learner_metrics_latest.json` when `progress_latest.last_learner` is
    missing;
  - add `learner_metrics_ref`, `learner_metrics_point_count`, and selected loss
    fields to `curve-json`/status output.
- Tests:
  - fake `BaseLearner.train` returns a dict with numeric metrics; assert one
    JSONL row, latest JSON, and exact original return value.
  - fake `BaseLearner.train` raises; assert the original exception is preserved
    and the wrapper does not hide it.
  - fake checkpoint write attaches `last_learner` from the recorder into
    `progress_latest.json`.
  - status-reader fallback reads `learner_metrics_latest.json` if
    `progress_latest` lacks `last_learner`.

## What This Would Prove

After the patch, for every active run we should be able to query:

- "The learner is still training": `learner_train_call_index` and
  `train_iter_after` advance.
- "The update is nontrivial": losses/entropies/grad norm are present and not
  constant where LightZero exposes them.
- "The checkpoint has nearby learner evidence": `progress_latest.last_learner`
  joins checkpoint iteration to learner metrics.
- "The policy probably changed": sampled model hash can prove it in canaries;
  normal runs can use cheaper evidence unless deeper proof is needed.

This does not replace eval/tournament survival curves. It only fills the missing
inner-loop evidence so we stop relying on stderr tails and guesswork.
