# Contingency Plans

Status: planning doc. No Modal jobs were launched while writing this.

## Operating Principle

Do not let a broad H100 sweep turn one uncertainty into five coupled
uncertainties. When a lane misbehaves, preserve the artifacts, stop widening
that lane, and compare against the lane-local control before touching other
lanes.

The default response should be:

1. Identify whether the failure is launch, instrumentation, training dynamics,
   evaluation, or interpretation.
2. Preserve the exact manifest, row ids, run ids, and latest status output.
3. Stop only the affected row group when possible.
4. Relaunch with a single changed variable.
5. Record the decision in this doc set or a run-specific child note.

## Prelaunch Risks

| Risk | Detection | Contingency |
| --- | --- | --- |
| Wrong H100 function or app target | Manifest `deployed_app_submission.train_function` is not `lightzero_curvytron_visual_survival_h100_cpu40` | Do not launch. Fix builder args or deployed app mapping, regenerate manifest. |
| RND rows do not require metrics | RND manifest guard fails, or enabled rows have `require_rnd_metrics != true` | Do not launch positive RND. Regenerate with the RND builder. |
| Stock rows accidentally require RND metrics | Stock `none` rows have `require_rnd_metrics=true` | Regenerate. Stock rows must remain clean controls. |
| Static controls use refresh | Any static reward row has `opponent_assignment_refresh_interval_train_iter > 0` or assignment refresh refs | Do not launch static controls. Regenerate with no refresh. |
| No non-RND lane is active | Launch plan contains only RND positives or only RND stock/meter rows | Hold expansion. Launch or preserve static exact-ref reward/cadence coverage so RND has real context. |
| Exact-ref Modal audit fails | Manifest-level audit reports missing refs or Modal parent lookup errors | Do not launch exact-ref lanes. Rebuild manifests from currently present v2 checkpoints, or rematerialize the intended refs from a verified source environment and rerun the audit. |
| Repair refs are overclaimed | Docs or launch notes describe top4nz repair refs as stable leaderboard truth | Correct the claim before launch. The refs are audited static exact refs, not a production leaderboard source. |
| Wave A packet audit fails | `scripts/audit_curvytron_wave_a_launch_packet.py` reports stale refs, wrong row counts, missing dry-runs, launch artifacts, or selected-row drift | Do not launch. Repair the manifest/dry-run/audit artifact first, then rerun the packet audit before checking capacity. |
| Capacity proxy requires operator review | `scripts/audit_curvytron_wave_a_capacity.py` reports `operator_capacity_review_required` | Do not treat this as either launch approval or hard H100 denial. Rerun near approval time, identify whether active tasks are H100-relevant, and either launch within the remaining room, stage the launch, or wait. |
| Row count exceeds intended runtime tier | Generated row count or retained active jobs exceed the chosen tier: broad `<=2h`, 40 rows for `2h-8h`, or 10-20 rows for `8h+` | Launch by row group using `--row-id` or `--limit`, split into Wave A/B, shorten timeout, or wait. Do not let a broad short sweep silently become a long run. |
| Run id collision | Dry-run or manifest validation reports duplicate run ids, or row ids map to old active runs | Regenerate with a new matrix name and run prefix. |
| Checkpoint refs are stale or shape-mismatched | Seeded rows fail dry-run or first load | Switch that lane to scratch bootstrap, or use exact immutable refs from a freshly audited file. |
| Submitter partial launch guard trips | `--allow-launch` with selected rows fails without `--allow-partial-launch` | Treat as a useful guard. Add `--allow-partial-launch` only after row ids are pasted into the run note. |

## First 30 Minutes

These are health checks, not learning checks.

| Symptom | Likely class | Contingency |
| --- | --- | --- |
| No heartbeat or progress file | Launch/runtime failure | Stop affected rows, inspect app logs, relaunch one row from the same manifest only after root cause is known. |
| Checkpoints never appear | Trainer did not enter learning loop, save cadence bug, or path bug | Check learner train-call index and checkpoint path. If train calls advance but no ckpt appears, reduce to one row and verify save cadence. |
| Background eval poller absent | Instrumentation failure | Keep training rows only if checkpoints are written; relaunch pollers or run status/eval manually before judging learning. |
| RND metrics missing on enabled rows | RND instrumentation failure | Stop positive RND rows. Keep stock controls. Relaunch a 2-row meter gate before reopening positive weights. |
| RND metrics nonfinite | RND numerical failure | Stop positive weights at and above the first bad weight. Relaunch low weights only with smaller learning rate or weight decay audit. |
| All rows OOM or crash | Config too large for actual app shape | Relaunch a one-row canary with lower batch/eval seed count. Do not shrink all lanes blindly. |
| CPU eval backlog grows | Poller bottleneck | Reduce background eval seed count or poll interval for future rows; do not read missing eval as no learning. |
| Volume writes are slow or missing | Storage/commit bottleneck | Extend poller runtime, reduce GIF frequency/steps, preserve trainer progress as primary health. |

Immediate triage rule: if more than 25 percent of a lane fails health in the
first 30 minutes, freeze that lane and continue only the healthy independent
lanes.

## First Learning Horizon, 30k-50k

| Symptom | Interpretation risk | Contingency |
| --- | --- | --- |
| No row beats iteration 0 | Could be reward/cadence failure, eval bug, or too early | Wait until 100k if health is clean; simultaneously prepare the cadence/support panel. |
| Every row improves then immediately collapses | Training dynamics likely dominate reward | Do not add new reward variants. Shift GPUs to support cap, TD horizon, batch size, and search sims. |
| One row spikes once | Selection bias | Preserve checkpoint, but require AUC or a second nonzero checkpoint before promotion. |
| Action collapse appears | Policy degeneracy | Stop widening that recipe. Compare collapse timing by reward/RND weight, then lower intrinsic/terminal scale or change cadence. |
| RND meter diverges from stock | Meter is not passive or seeds are too noisy | Run more stock/meter replicas before judging positive RND. Do not promote positive weights from this sweep. |
| High RND weights look best early | Novelty may be overpowering survival | Require action histograms, GIF review, and retention before expansion. Add low-weight replicas rather than high-weight expansion. |
| Plus-outcome wins raw trainer reward only | Cross-variant reward scale trap | Ignore as promotion evidence. Use survival AUC, retention, and tournament exposure. |

## Useful Decision Horizon, 100k-170k

| Symptom | Contingency |
| --- | --- |
| Plus-outcome beats controls but regresses hard | Sweep lower `reward_outcome_alpha` and higher support cap before production widening. Preserve best checkpoint. |
| No-outcome is stable but weak | Keep as control/default compact reward; test plus-outcome alpha rather than replacing all lanes. |
| Sparse outcome has strong best checkpoint but poor latest | Preserve best sparse checkpoint for tournaments; treat as credit-assignment/cadence lead. |
| All extrinsic rewards are flat | Move to cadence/support/search first. Reward axis has low explanatory power. |
| Low-weight RND beats stock and meter on AUC | Add replicas for the best 2-3 low weights and implement fixed-opponent RND manifest extension. |
| RND improves blank-canvas only | Treat as exploration-specific, not game-strength proof. Next test must use fixed opponents or tournament exposure. |
| RND intrinsic scale swamps extrinsic components | Cut maximum weight, add lower weights, and consider clipping/normalization audit before more GPUs. |
| RND looks good but non-RND lanes are missing or unhealthy | Do not promote RND. Relaunch or repair the non-RND static/cadence controls first. |
| Later leaderboard/refresh slice disagrees with static reward isolate | Assume opponent refresh/assignment is the confound. Run a no-refresh bridge row before picking a winner. |

## Retention Horizon, 240k-300k

| Symptom | Contingency |
| --- | --- |
| Best is strong, latest is weak | Mark retention failure, preserve best checkpoint, and test cadence/support. Do not call the reward done. |
| Latest remains within 90 percent of best | Candidate can be promoted to tournament or wider recipe exposure. |
| Best checkpoint is isolated | Require another seed or adjacent checkpoint before promotion. |
| Tournament top rows are iteration 0 | Tournament pool/exposure is not measuring learning. Increase nonzero checkpoint exposure and exclude stale rows from interpretation. |
| Tournament winner has low game count | Not enough exposure. Continue tournament only, not training, until exposure matures. |
| RND winner does not reproduce across replicas | Treat as noise. Add replicas around lower weights before implementation changes. |

## Lane-Specific Fallbacks

### RND Lane

Fallback ladder:

1. Keep stock `none` controls running if healthy.
2. If positive rows fail, run `rnd_meter_v0` versus stock only.
3. If meter is healthy, relaunch positive rows at low weights only:
   `0.001`, `0.003`, `0.01`, `0.03`.
4. If low weights help blank-canvas survival, add a fixed-opponent RND builder
   extension.
5. If fixed opponents erase the gain, archive RND as an exploration diagnostic.

### Extrinsic Reward Lane

Fallback ladder:

1. Preserve the full 18-row no-refresh read.
2. If all arms regress, do not add rewards. Move to support/cadence.
3. If plus-outcome is volatile, sweep `reward_outcome_alpha`.
4. If no-outcome is stable, use it as the control and compact default.
5. If sparse produces tournament winners, save those checkpoints but keep
   survival retention as the promotion gate.

### Cadence/Support Lane

Fallback ladder:

1. Increase `batch_size` before inventing a new reward.
2. Increase `num_simulations` for quality reads, accepting slower wall time.
3. Set explicit `td_steps` to reduce target ambiguity.
4. Set explicit `model_support_cap` if dense rewards saturate support.
5. If stabilized knobs help all rewards, move those knobs into the next matrix.

### Tournament Lane

Fallback ladder:

1. Keep tournaments diagnostic until checkpoint quality is visible.
2. Require nonzero checkpoint exposure before ranking claims.
3. If ratings are stale or sparse, report exposure first and rank second.
4. Do not feed tournament output into training-candidate refresh for static
   controls.

## Stop Conditions

Stop or pause a lane when:

- health artifacts are missing after the expected startup window
- more than 25 percent of rows in the lane crash early
- RND metrics are missing or nonfinite on enabled rows
- action collapse persists across consecutive checkpoints
- a no-refresh control writes assignment refresh artifacts
- row configs do not match the manifest hypothesis
- storage or eval backlog makes the learning signal unreadable

Keep running when:

- training is healthy but eval is delayed
- one row fails inside an otherwise healthy replicated lane
- early survival is flat before 50k and learner metrics look normal
- tournament ratings are immature but checkpoint evals are available

## Communication Template

Use this shape for incident notes:

```text
lane:
rows affected:
first bad time / iteration:
health artifacts present:
learning artifacts present:
control rows status:
suspected class:
decision:
single changed variable for relaunch:
artifacts preserved:
```
