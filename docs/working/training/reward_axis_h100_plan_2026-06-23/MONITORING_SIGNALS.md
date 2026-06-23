# Monitoring Signals

Status: active monitoring contract. This file defines what to watch during the
H100 reward-axis
runs and how soon each signal should be trusted.

## Signal Priority

Primary learning signals:

1. Best-so-far eval survival versus iteration 0.
2. AUC of eval survival across checkpoints.
3. Latest-vs-best retention.
4. Action-collapse rate and top-action fraction.
5. Tournament head-to-head exposure and game duration, if attached.
6. For RND rows only: intrinsic reward scale, predictor loss, and positive
   weight performance versus both stock and meter controls.

Secondary health signals:

- learner train-call index advances
- train iter advances
- learner numeric metrics exist
- checkpoint files and metadata sidecars are written
- background eval poller sees and evaluates checkpoints
- status heartbeat remains fresh

Misleading if read alone:

- latest checkpoint only
- raw trainer reward across reward variants
- job still running
- checkpoint exists
- GIF looks different
- weights changed
- tournament Elo without exposure counts

## Status Commands

Typical health/status query:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "<comma-separated-run-ids>" \
  --output fast-summary
```

Eval curve summary:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "<comma-separated-run-ids>" \
  --output curve-summary
```

Eval JSON for local aggregation:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "<comma-separated-run-ids>" \
  --output curve-json
```

Assignment-refresh proof for no-refresh controls:

```bash
uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status \
  --run-ids "<comma-separated-run-ids>" \
  --output assignment-proof-json
```

RND artifacts to inspect in the run volume:

```text
training/<task_id>/<run_id>/attempts/<attempt_id>/train/rnd_reward_model_metrics_latest.json
training/<task_id>/<run_id>/attempts/<attempt_id>/train/rnd_reward_model_metrics.jsonl
```

## Early Health Gate

Check within the first 30 minutes:

- `progress_latest.json` readable or expected not yet present
- `status_heartbeat.json` fresh
- `checkpoint_eval_poller.json` exists
- selected rows have no assignment refresh events for static controls
- learner metrics latest exists after learner starts
- no obvious config mismatch in command/config artifacts

Do not infer learning here.

## First Learning Gate

Check at the first few nonzero checkpoints, roughly 30k-50k train iterations:

- eval means exist for iteration 0 and at least one nonzero checkpoint
- top-action fraction stays below collapse threshold
- learner train iter and train-call index advance
- policy entropy/target entropy/loss rows are present when exposed
- survival is not catastrophically below iteration 0 across all rows
- RND stock, meter, and positive rows can be grouped by weight and replica
- non-RND static reward/cadence rows are grouped by reward variant, recipe, and
  knob set

Interpretation:

- One row up is only a hint.
- A flat healthy row at 30k-50k is not a failure signal; CurvyTron has often
  needed much longer horizons before useful training signal appears.
- All rows flat plus clean learner metrics means reward/cadence may still be
  wrong, not that infrastructure is dead.
- Missing evals or unreadable progress are operational failures first.

## Useful Decision Gate

Check at 100k-170k:

- best-so-far survival versus iteration 0
- AUC by reward arm
- latest versus best
- action collapse history
- reward component sanity within each variant
- sparse outcome telemetry side by side with survival
- for RND, positive weights beat both `none` and `rnd_meter_v0` on AUC or
  best-so-far without worse collapse
- independent non-RND static/cadence rows are healthy enough to be real
  baselines, not missing controls

This is the first horizon where reward-arm decisions can be meaningful.

## Retention Gate

Check at 240k-300k:

- latest within 90 percent of best, or best-to-latest drop materially lower than
  old matched rows
- best checkpoint is not only an isolated spike
- top action distribution remains noncollapsed
- if tournament-attached, top policies are nonzero and exposure-mature
- top tournament rows are not dominated by iteration 0 or stale latest-only
  artifacts
- RND winner retains latest near best and repeats across at least two replicas

If the best checkpoint is strong but latest regresses, preserve the best
checkpoint and mark the run as a retention failure rather than a no-learning
failure.

## Tournament Signals

Tournament is useful after there are enough nonzero checkpoints. Read it as
relative head-to-head evidence, not as an absolute learning curve.

Trust more:

- games and distinct opponents per row
- active/provisional status
- top rows are nonzero checkpoints
- game duration distributions rise over rounds
- top-ranked policies are associated with longer games
- ratings stabilize or max delta becomes small enough for the intended use

Trust less:

- mean Elo across the whole pool
- rank without game count
- a top row with one opponent
- top iteration-0 policies
- stale `latest.json`

## RND-Specific Signals

RND is promising when:

- `rnd_meter_v0` behaves like stock on survival, proving the meter is passive.
- low positive weights improve survival AUC or best-so-far over both stock and
  meter rows.
- predictor loss and intrinsic reward stats are finite and nontrivial.
- intrinsic scale does not swamp extrinsic reward components.
- action histograms do not collapse or turn into obvious novelty seeking.
- independent non-RND lanes are healthy through the same horizon.

RND is not promising when:

- only RND metrics improve while survival stays flat.
- high weights win only by causing erratic action distributions.
- meter rows diverge from stock rows.
- blank-canvas gains vanish when fixed opponents or tournament exposure are
  attached.
- independent non-RND lanes are missing or failed before the useful horizon.

## Stop Or Pivot Conditions

Stop a row or avoid widening when:

- no nonzero eval appears after checkpointing should be active
- learner metrics do not advance after training starts
- action collapse appears and persists across consecutive checkpoints
- latest survival is below iteration 0 and best never improves by 100k-170k
- assignment refresh fires in a no-refresh control
- checkpoint metadata has the wrong observation/reward contract

Do not stop solely because survival is flat before 100k when health, eval, and
learner metrics are clean.

See `CONTINGENCY_PLANS.md` for the fallback ladder after a stop or pause
decision. The important operational rule is to relaunch with one changed
variable, not a bundled fix.

Pivot from reward to cadence/support when:

- all reward arms show mid-run lift and late regression
- plus-outcome is volatile while no-outcome is stable but weak
- dense variants saturate support indicators or show huge terminal reward
  swings
- sparse has good tournament outcomes but poor survival retention

Pivot from RND to implementation work when:

- RND metrics are missing or nonfinite on enabled rows
- resume/checkpoint semantics cannot preserve the RND model state
- positive rows cannot be compared because stock or meter controls failed
- blank-canvas positives look good but fixed-opponent extension is unavailable

Do not promote RND when:

- independent non-RND reward/cadence lanes were not launched
- non-RND lanes failed health before the useful horizon
- positive RND is only better than weak or broken controls

## Reporting Template

For each monitoring pass, report:

```text
run group:
iteration window:
rows checked:
health:
eval first / best / latest:
AUC:
latest-best drop:
action collapse:
learner metrics:
reward components:
tournament exposure:
decision:
next action:
```
