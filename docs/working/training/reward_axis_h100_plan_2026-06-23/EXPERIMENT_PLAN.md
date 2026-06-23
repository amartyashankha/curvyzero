# Initial H100 Experiment Plan

Status: active plan, reoriented for aggressive parallel H100 use. Build and
dry-run first, then launch separated lanes whose signals remain interpretable
under the intended runtime tier.

## Hypotheses

H1: `survival_plus_bonus_plus_outcome` is the best current main reward arm, but
its terminal term needs alpha/support/cadence controls to avoid late regression.

H2: `survival_plus_bonus_no_outcome` gives the cleanest survival shaping, but it
may underselect policies that win or retain skill against stronger checkpoints.

H3: `sparse_outcome` can produce strong mid-run checkpoints, but current
`td_steps`, shallow search, and large collect chunks make latest retention poor.

H4: reward is not the only issue. The same reward can look bad if support
saturation, `num_simulations=8`, `collector_env_num=256`, or `batch_size=32`
dominates the training dynamics.

H5: RND is worth a real parallel push. The implementation is believable enough
for a positive-weight sweep, but its result must be judged against stock and
meter controls and must not be folded into compact-owned reward claims.

H6: leaderboard should come later. Early signal should come from static exact
checkpoint curricula and long horizons; tournament/leaderboard is for later
selection and feedback.

H7: non-RND controls must remain broad. RND can only be interpreted against
healthy stock/meter controls and independent extrinsic reward/cadence lanes.

## Fixed Principles

- Start with no tournament refresh for the first reward isolate.
- Prefer exact immutable `iteration_N.pth.tar` checkpoint refs for quality rows.
  Use scratch only for explicit cold-start or RND blank-canvas diagnostics.
- Always consider starting non-RND quality rows from the best-known checkpoint.
  If using the current top4nz repair seed instead, record that as an explicit
  availability/repair decision.
- Keep learner seed and opponent refs separate. `--checkpoint-refs-file`
  selects frozen opponent rank slots; `--initial-policy-checkpoint-ref` selects
  the learner initial policy.
- Keep `learner_seat_mode=random_per_episode`.
- Keep background eval and learner metrics on.
- Use H100 aggressively, but keep reward, RND, cadence/support, and tournament
  questions in separate lanes.
- Use runtime-tier capacity, not one flat rule: short `<=2h` breadth can use the
  prepared 90-row packet if capacity is clear; `2h-8h` runs should narrow to at
  most 40 active H100 rows; `8h+` runs should narrow to 10-20 rows.
- Treat the capacity audit as a conservative task-count proxy. If active
  capacity is ambiguous, stage below the proxy room or wait for an explicit
  operator capacity decision.
- Always run non-RND experiments too: static exact-ref reward isolate,
  long-horizon pretrained replicas, and cadence/support rows are first-class
  lanes, not optional controls.
- Judge best-so-far, AUC, and latest-vs-best retention, not latest alone.
- Let healthy rows run to at least 240k-300k before interpreting lack of
  learning as failure.
- Keep trainer-facing leaderboard refresh out of Wave A.
- Do not compare raw trainer reward across reward variants.
- Treat RND as first-class but experimental: positive RND must beat stock and
  `rnd_meter_v0`, not just produce interesting metrics.

## Wave A - Prepared Launch Menu

Recommended first prepared menu: 90 H100 rows, balanced as 45 RND and 45
non-RND. This is a launch packet, not a commitment to run all rows through every
wall-clock horizon.

Operating stance: launch broad when the manifest and health gates are clear.
Do not turn every hypothesis into a serial canary. Canaries should be embedded
in broad manifests or run as same-hour health gates with the broad launch ready.

| Lane | Rows | Source | First useful read |
| --- | ---: | --- | --- |
| RND wide blank sweep | 45 | `scripts/build_curvytron_rnd_blank_sweep_manifest.py` with 9 points x 5 replicas | RND metrics in 30 minutes; survival/AUC by 50k-170k. |
| Static exact-ref reward isolate | 18 | `scripts/build_curvytron_tonight18_manifest.py`, no refresh, `--checkpoint-refs-file` | Reward-arm read by 100k-300k. |
| Long-horizon pretrained replicas | 18 | six exact-ref manifests, selected rows `r005/r011/r017` from each | Seed robustness and slow-signal retention across sparse, no-outcome, and plus-outcome. |
| Cadence/support panel | 9 | three exact-ref knob manifests, selected rows `r005/r011/r017` from each | Whether reward failure is really support/cadence failure. |

Runtime staging is tracked in `NEXT_HOURS_OPERATING_PLAN_2026-06-23.md`.
Current launch state is tracked in `LAUNCH_QUEUE.md`. The RND wide blank sweep
has a saved dry-run artifact. The original exact-ref non-RND manifests remain
blocked by `PRELAUNCH_AUDIT_2026-06-23.md`, but the repaired top4nz non-RND
family in `PRELAUNCH_REPAIR_2026-06-23.md` has passing Modal ref audits and
dry-run evidence. The bestseed repair family in `WAVE_A_MANIFESTS.md` has
passing Modal ref audits, dry-run evidence, and a best-known-seed anchor audit.
The current top4nz packet audit
`artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json` reports
`ok=true`; the bestseed packet audit
`artifacts/local/curvytron_wave_a_launch_packet_audit_bestseed_20260623a.json`
also reports `ok=true` for the repaired 90-row no-launch package. Real launch
still requires human approval, exact command and row-count review, and active
H100 capacity confirmation. The current capacity proxy artifact
`artifacts/local/curvytron_wave_a_capacity_snapshot_20260623a.json` requires
operator capacity review because unrelated Modal task load is present; under
the conservative task-count proxy, the current room is `22` additional rows
unless existing tasks are classified as non-H100/non-conflicting.

Do not launch a positive-RND-only campaign. The prepared broad pair is RND plus
the non-RND static exact-ref reward isolate. If runtime or capacity forces a
split, preserve enough non-RND rows to keep the RND interpretation honest.

The old 3-row reward isolate is still useful as a conceptual minimum, but it is
not the default operating posture for a short broad sweep when capacity is
available.

The old CZ26-style refresh grid is not Wave A. Reintroduce it after static
exact-ref lanes produce nonzero checkpoints worth ranking.

## Phase 0 - Manifest And Health Dry Run

Goal: prove the launch surface and selected rows are exactly what we intend.

Recommended builder path:

- `scripts/build_curvytron_tonight18_manifest.py`
- current repaired ref source:
  `--checkpoint-refs-file artifacts/local/curvytron_no_tournament_control_20260516/source/static_top4_nonzero_refs.txt`
- for bestseed non-RND rows:
  `--initial-policy-checkpoint-ref training/lightzero-curvytron-visual-survival/curvy-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/attempts/try-r18fresh-survbonusout-blank20-wall5-rank1_70-rank1imm5-so10rep10-s134842423/train/lightzero_exp/ckpt/iteration_180000.pth.tar`
- `--opponent-source mixture`
- `--assignment-refresh-interval-train-iter 0`
- `--compute gpu-h100-cpu40`
- selected rows only through `scripts/submit_curvytron_survivaldiag_manifest.py`

Required dry-run checks:

- packet audit:
  `uv run python scripts/audit_curvytron_wave_a_launch_packet.py --non-rnd-seed-profile bestseed --output artifacts/local/curvytron_wave_a_launch_packet_audit_bestseed_20260623a.json`
- capacity proxy:
  `uv run python scripts/audit_curvytron_wave_a_capacity.py --output artifacts/local/curvytron_wave_a_capacity_snapshot_20260623a.json`
- checkpoint anchor policy:
  `uv run python scripts/audit_curvytron_checkpoint_anchor_policy.py --non-rnd-seed-profile bestseed --require-best-known-seed --output artifacts/local/curvytron_checkpoint_anchor_policy_audit_bestseed_20260623a.json`
- staged launch profile:
  `uv run python scripts/plan_curvytron_wave_a_staged_launch.py --profile mid36_bestseed --output artifacts/local/curvytron_wave_a_staged_launch_mid36_bestseed_20260623a.json`
- selected rows only
- `assignment_write_count=0`
- `refresh_pointer_write_count=0`
- no `opponent_assignment_refresh_ref`
- exact initial checkpoint refs for static quality rows
- one shared initial policy seed for matched comparisons
- background eval poller present

Launch note: partial row launch with `--allow-launch` must include
`--allow-partial-launch`.

## Phase 1 - Clean Reward Isolate

Goal: compare the three main extrinsic reward arms while holding opponent recipe
and noise fixed. In Wave A, launch the full 18-row static matrix or at least the
three clean rank1-heavy rows below.

Use the strongest current recipe shape, the current script's rank1-heavy row:

- `slot64-blank12-wall4-rank1_46-rank1imm2`
- clean noise only
- no assignment refresh

Initial row set from the 18-row generator:

| Row | Reward | Purpose |
| --- | --- | --- |
| `r005` | `sparse_outcome` | Clean outcome baseline. |
| `r011` | `survival_plus_bonus_no_outcome` | Survival-only shaping control. |
| `r017` | `survival_plus_bonus_plus_outcome` | Current main candidate. |

Default-shape run:

- `collector_env_num=256`
- `n_episode=256`
- `num_simulations=8`
- `batch_size=32`
- `td_steps=None`
- `model_support_cap=None` (current effective cap path)

Decision: if `r017` beats `r011` and `r005` on best-so-far, AUC, and retention
by 240k-300k, keep plus-outcome as main. If `r011` is more stable while `r017`
is volatile, move to alpha/support controls before widening. A flat 50k read is
not enough to reject a healthy row.

## Phase 2 - Reward Plus Cadence/Support Panel

Goal: distinguish reward effect from obvious training-dynamics confounds.

Run a second small panel on the same row set or the top two rewards only.

Suggested stabilized knobs:

- `collector_env_num=64`
- `n_episode=64`
- `num_simulations=25`
- `batch_size=128` or `256`
- `td_steps=25`
- `model_support_cap=1024` or `2048`

Suggested alpha panel for `survival_plus_bonus_plus_outcome`:

- alpha `1.0`: current semantics
- alpha `0.5`: less terminal volatility
- alpha `0.25`: more survival-dominant

Because `reward_outcome_alpha` is a manifest-level knob in the current
tonight18 builder, alpha tests should be separate manifests or a builder change.

Decision: if plus-outcome only works under lower alpha or higher support, the
next implementation should make reward scaling/support explicit before the
bigger sweep.

## Phase 3 - Opponent Recipe And Noise Confirmation

Goal: confirm the selected reward under multiple opponent recipes and decide
whether stochasticity is helpful or a credit-assignment hazard.

Rows to widen after Phase 1/2:

- all clean rows for selected reward across the three recipes
- only then add `straight_override_p10_repeat_p10` rows as a separate axis

Interpretation rule:

- Stochastic rows can be useful regularizers, but they also train on requested
  actions while transitions may reflect overridden/repeated actions. Treat a
  stochastic win as a clue, not a default policy, until executed-action replay
  semantics are audited.

## Phase 4 - Tournament Attachment

Goal: after nonzero checkpoints exist, evaluate best checkpoints head-to-head
without feeding rankings back into the trainers.

Use a fresh diagnostic tournament id and rating id. Do not run
training-candidate refresh for the Phase 1/2 controls.

Trainer-facing leaderboard refresh is a later Wave C feature. Diagnostic
tournament can select checkpoint refs for the next static wave; it should not
drive Wave A trainers.

Signals:

- top checkpoints are nonzero and not only latest-for-run
- enough games and distinct opponents
- tournament game duration distribution rises
- top-ranked policies correlate with longer survival games
- no iteration-zero checkpoints dominate the active top band

## Time-To-Decision

Rough expectations from prior H100-ish and r18fresh runs:

| Window | Expected signal |
| --- | --- |
| 0-30 minutes | Modal health, status heartbeat, first checkpoint/eval poller wiring. Not learning proof. |
| 30k checkpoints | First weak directional read. Check eval exists, action distribution not collapsed, learner metrics advancing. |
| 50k-100k | First useful reward comparison. Best-so-far and AUC start to matter. |
| 170k | Stronger early decision point. Prior CZ26 readouts used this horizon. |
| 240k-300k | Retention decision. Prior r18fresh comparisons used this grid and many best checkpoints appeared mid-run. |

For sim25/batch128+ runs, wall-clock may be materially slower than the old
sim8/B32 rows. Treat iteration horizon as the comparison unit and wall time as
operational planning only.

## Promotion Gate

A reward arm is promising only if it satisfies most of:

- best survival materially above iteration 0
- latest survival remains within 90 percent of best, or regression is clearly
  lower than matched baseline
- AUC beats the baseline arm
- action collapse is rare or absent
- learner metrics advance with nontrivial losses/entropy/grad norm
- tournament exposure does not contradict survival
- best checkpoint is reproducible across at least two seeds or matched rows

## Explicit Non-Goals

- Do not merge positive RND into the extrinsic reward-axis conclusion.
- Do not call latest checkpoint a winner without retention.
- Do not feed diagnostic tournament output back into Phase 1/2 trainers.
- Do not mix reward recipes within one attempt.
- Do not promote a reward based on raw trainer reward across variants.
