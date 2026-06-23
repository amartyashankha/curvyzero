# Next Hours Operating Plan - 2026-06-23

Status: active planning note. No Modal jobs were launched while writing this.

## High-Level Objective

The objective is not to find the final CurvyTron reward in one launch. The next
few hours should produce clean, auditable evidence for three questions:

1. Does any extrinsic reward arm, especially
   `survival_plus_bonus_plus_outcome`, show better survival AUC, best-so-far,
   and retention than `survival_plus_bonus_no_outcome` and `sparse_outcome`
   under static exact-ref controls?
2. Does stock-path RND with low positive weights beat both `none` and
   `rnd_meter_v0` on survival, not only on RND metrics?
3. If learning lifts and then regresses, is the next move reward scaling or
   cadence/support/search?

The outcome we want is a decision-quality map of which lanes deserve longer
runs, replicas, or fixed-opponent bridge tests. Leaderboard feedback comes
after nonzero checkpoints exist.

Primary references:

- `REWARD_INVENTORY.md` for what each reward arm means.
- `RND_LANE.md` and `STOCK_PATH_RND_REORIENTATION.md` for RND scope.
- `MONITORING_SIGNALS.md` for readout gates.
- `CHECKPOINT_ANCHOR_POLICY.md` for best-known checkpoint seed policy.
- `CONTINGENCY_PLANS.md` for response ladders.
- `LAUNCH_QUEUE.md` and `WAVE_A_LAUNCH_REVIEW_2026-06-23.md` for launch state.

## Signal Hierarchy

Health signals come first:

- launch artifacts and run ids exist
- heartbeat/progress/checkpoint/eval poller artifacts appear
- learner train iter and train-call index advance
- RND rows write finite RND metrics
- no-refresh controls do not write assignment refresh artifacts

Weak early signals begin around `30k-50k` train iterations:

- nonzero evals exist
- action distribution is not collapsed
- survival is not catastrophically below iteration 0 across all rows
- stock, meter, and positive RND groups can be compared

Useful learning signals begin around `100k-170k`:

- best-so-far survival above iteration 0
- survival AUC by reward arm or RND weight
- latest-vs-best drop
- RND positive weights beat both stock and meter controls
- non-RND static and cadence rows are healthy enough to serve as real controls

Retention signals require `240k-300k`:

- latest remains near best or the best-to-latest drop is lower than controls
- best checkpoint is not a one-off spike
- action collapse is absent or materially better than matched controls
- RND winners reproduce across replicas

Flat survival before `100k` is not a failure when health, eval, and learner
metrics are clean.

## Runtime Tiers

Use these as operating caps, not scientific conclusions:

| Tier | Intended runtime | Active H100 posture | Purpose |
| --- | --- | --- | --- |
| Short breadth | `<=2h` | Broad launch up to the prepared 90-row packet if capacity is explicitly clear. More than 100 simultaneous H100s needs explicit operator override and timeout. | Startup health, metrics, first weak signal, broad failure discovery. |
| Medium read | `2h-8h` | At most 40 active H100 rows. | Reach useful learning signal while preserving RND and non-RND controls. |
| Long read | `8h+` | 10-20 active H100 rows. | Retention and reproducibility for the most informative rows only. |

The capacity auditor is a conservative task-count proxy. If it says
`operator_capacity_review_required`, the next action is a capacity decision,
not launch permission and not proof that H100s are unavailable.

## Prepared Profiles

Use these as staged subsets of the already-audited Wave A menu.

| Profile | Rows | Use when | Shape |
| --- | ---: | --- | --- |
| `short90` | 90 | `<=2h` breadth/health sweep with explicit capacity approval | Full packet: 45 RND blank rows, 18 static top4nz rows, 18 long-horizon selected rows, 9 cadence/support selected rows. |
| `mid36` | 36 | `2h-8h` medium run | RND rows `r001-r018` from the blank sweep plus all 18 static top4nz non-RND rows. |
| `long17_no_highest_weight` | 17 | `8h+` when capacity is tight | RND rows `r001-r008`, dropping only the highest RND weight, plus the same 9-row non-RND triad as `long18_all_weights`. |
| `long18_all_weights` | 18 | `8h+` when preserving all RND weights matters | RND rows `r001-r009`, static `r005/r011/r017`, long-horizon rep01 `r005/r011/r017`, and first cadence/support knob `r005/r011/r017`. |
| `long19_low_weight_replicated` | 19 | `8h+` recommended default if RND low-weight read is the priority | RND rows `r001-r005` and `r010-r014`, plus the same 9-row non-RND triad as `long18_all_weights`. |

Interpretation notes:

- `mid36` keeps the two most important first reads alive: RND stock/meter/low
  positive coverage and the full static extrinsic reward isolate.
- `long18_all_weights` is better for detecting whether high RND weights are
  pathological.
- `long17_no_highest_weight` is the conservative capacity-fit fallback when the
  proxy room drops below 18-19 rows; it preserves stock, meter, low/mid RND
  weights, and the non-RND triad.
- `long19_low_weight_replicated` is better for deciding whether low RND weights
  deserve fixed-opponent bridge experiments.
- Exact launch commands must be written from manifest paths and `--row-id`
  filters before approval; do not reconstruct row ids from memory.
- Current generated profile artifacts:
  `artifacts/local/curvytron_wave_a_staged_launch_mid36_20260623a.json`,
  `artifacts/local/curvytron_wave_a_staged_launch_long19_low_weight_replicated_20260623a.json`,
  and `artifacts/local/curvytron_wave_a_staged_launch_short90_20260623a.json`.
- Preferred bestseed profile artifacts:
  `artifacts/local/curvytron_wave_a_staged_launch_mid36_bestseed_20260623a.json`,
  `artifacts/local/curvytron_wave_a_staged_launch_long17_no_highest_weight_bestseed_20260623a.json`,
  `artifacts/local/curvytron_wave_a_staged_launch_long18_all_weights_bestseed_20260623a.json`,
  `artifacts/local/curvytron_wave_a_staged_launch_long19_low_weight_replicated_bestseed_20260623a.json`,
  and `artifacts/local/curvytron_wave_a_staged_launch_short90_bestseed_20260623a.json`.
- Current profile-specific capacity frontier:
  `artifacts/local/curvytron_wave_a_capacity_snapshot_long17_no_highest_weight_bestseed_20260623a.json`
  is `capacity_proxy_clear` for 17 rows; 18, 19, 36, and 90-row bestseed
  profiles currently require operator capacity review under the conservative
  task-count proxy.

## Time-Scale Plan

### 5 Minutes

- Rerun or inspect the latest packet audit and capacity proxy.
- Rerun or inspect the checkpoint-anchor audit.
- Choose the intended runtime tier before any launch approval request.
- Choose historical best-known seed versus current top4nz repair seed for
  non-RND rows. For medium/long, default to the bestseed profiles unless the
  launch note explicitly chooses top4nz repair seeding.
- Confirm that the next command set includes non-RND controls or that healthy
  non-RND controls already exist.
- Record the proposed profile name and row count in `LAUNCH_QUEUE.md` or a
  status child note.

### 15 Minutes

- If launch is being requested, paste the exact command set, output paths, row
  count, timeout, and profile name into the approval note.
- If launch is not being requested, prepare the staged row subset for
  `mid36`, `long19_low_weight_replicated`, or the capacity-fit
  `long17_no_highest_weight` profile using
  `scripts/plan_curvytron_wave_a_staged_launch.py`.
- Confirm first status command and health note path before the first
  FunctionCall is submitted.

### 30 Minutes

- If rows are launched, read health only: heartbeat, progress, checkpoint/eval
  poller, learner metrics, and RND metrics.
- Classify missing artifacts as operational failures before interpreting
  learning.
- If more than 25 percent of a lane fails health, freeze that lane and keep
  independent healthy lanes moving.

### 1 Hour

- Confirm startup stability and first checkpoint/eval flow.
- Check RND metrics are finite and stock rows are not blocked by RND guards.
- Check action histograms only for obvious collapse.
- Do not call reward or RND winners at this horizon.

### 2 Hours

- If a short broad launch is still running, decide whether to stop it or narrow
  to a medium profile at 40 rows or fewer.
- First weak learning read is allowed only if enough rows reached nonzero
  checkpoints.
- Preserve best checkpoints from any early spikes; do not promote them yet.

### 4 Hours

- Ensure no more than 40 active H100 rows remain.
- Compare health-normalized AUC/best hints across RND stock/meter/positive and
  non-RND reward arms.
- If all reward arms lift then regress, prioritize cadence/support over new
  reward variants.
- If low-weight RND looks better without collapse, prepare fixed-opponent RND
  bridge work but wait for retention before claiming success.

### 8 Hours

- Ensure only 10-20 active H100 rows remain for any longer continuation.
- Read retention only for rows that reached the planned iteration horizon.
- Preserve best checkpoints, record latest-vs-best drops, and decide which
  rows deserve replicas or diagnostic tournament exposure.
- Do not introduce trainer-facing leaderboard refresh until nonzero
  checkpoints are selected and documented.

## Default Next Move

If approval is requested for a short breadth launch, use `short90` only with a
clear timeout and fresh capacity review. If the intent is a real learning run
beyond two hours, prefer `mid36`. If the intent is to leave jobs running through
the day, prefer `long19_low_weight_replicated`.
