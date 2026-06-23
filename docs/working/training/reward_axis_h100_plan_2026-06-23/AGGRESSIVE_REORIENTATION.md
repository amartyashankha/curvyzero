# Aggressive H100 Reorientation

Status: planning doc. No Modal jobs were launched while writing this.

## Bottom Line

Use the H100 budget as a parallel readout farm, not as one giant entangled
sweep. The best first move is to prepare a broad, timeboxed menu of separate
lanes whose signals can be read independently:

1. RND blank-canvas sweep with real positive intrinsic reward.
2. Static exact-ref extrinsic reward isolate across the current reward/opponent
   matrix.
3. Long-horizon pretrained replicas against simple exact-ref curricula.
4. Cadence/support controls around the same reward arms.

RND should be first-class now. It should not be folded into the compact speed
claim or used to declare an extrinsic reward winner.

## Current Belief

The most believable main extrinsic reward remains
`survival_plus_bonus_plus_outcome`, but only as a hypothesis. The most believable
RND implementation is the existing `rnd_replay_target_v0` lane, guarded by
`rnd_meter_v0` and stock no-bonus controls. We have implementation/test proof
for RND plumbing, not successful learning proof.

The key risk is not underusing GPUs. The key risk is spending a large H100
batch, especially for multiple hours, on a matrix where a win cannot be
attributed to reward, intrinsic bonus, support range, cadence, or opponent
source.

Leaderboard comes later. Wave A should not depend on live tournament ranking,
trainer-facing leaderboard refresh, or moving assignment pointers. Use exact
trained checkpoint refs and static mixtures first.

## Wave A - 90-Row Prepared Menu

This is the first aggressive menu I would prepare after dry-run review. It
contains 90 H100 rows and is balanced for a short breadth/health launch. It
should narrow before it becomes a medium or long learning campaign.

| Lane | Rows | Shape | Why it exists |
| --- | ---: | --- | --- |
| RND wide blank sweep | 45 | 9 RND points x 5 replicas | Makes RND real again, with enough replicas to separate noise from signal. |
| Static exact-ref reward isolate | 18 | `tonight18` full matrix, no refresh, seeded exact refs | Tests sparse, no-outcome, and plus-outcome across recipes/noise without leaderboard feedback. |
| Long-horizon pretrained replicas | 18 | top reward/recipe candidates, same trained seed, repeated rows | Gives slow training signal enough runway and separates seed noise from reward/recipe signal. |
| Cadence/support panel | 9 | selected clean rows x stabilized knob sets | Separates reward failure from support/cadence failure. |
| Buffer | 10 | relaunch/debug/fixed-opponent RND bridge | Avoids spending every short-sweep slot before first health read. |

Runtime staging:

- `<=2h`: broad packet can run if capacity is explicitly clear and the timeout
  is understood.
- `2h-8h`: narrow to at most 40 active H100 rows.
- `8h+`: narrow to 10-20 active H100 rows.

Use `NEXT_HOURS_OPERATING_PLAN_2026-06-23.md` for concrete staged profiles.

Interpretation rule: lanes can run together, but they should not be promoted
together. Each lane needs its own baseline, metric horizon, and failure mode.
Use `CONTINGENCY_PLANS.md` as the response playbook if a lane fails health,
drifts into ambiguity, or produces a suspicious win.
Use `OPERATING_PATTERNS.md` to classify every claim before reporting it as a
result.

## Wave B - Conditional Expansion

After the first useful horizon, expand only the lanes that produce readable
signal.

| Condition | Expansion |
| --- | --- |
| Low-weight RND beats stock and meter on survival AUC without collapse | Add more replicas for the best 2-3 weights, then extend RND beyond blank canvas with static exact checkpoint opponents. |
| `survival_plus_bonus_plus_outcome` beats controls and retains latest | Launch a larger static exact-ref recipe matrix before adding trainer-facing leaderboard refresh. |
| All reward arms improve mid-run then regress | Spend next GPUs on support, TD horizon, search sims, and batch/cadence rather than new rewards. |
| Plus-outcome is volatile but no-outcome is stable | Sweep lower `reward_outcome_alpha` before widening opponent recipes. |
| Static exact-ref lanes produce good nonzero checkpoints | Attach diagnostic tournament for selection only; do not feed it back into trainers yet. |

## Launch Discipline

Before any launch:

- Build manifests locally.
- Dry-run the grouped submitter.
- Read the lane-specific contingencies before launching.
- Confirm row counts and row ids.
- Confirm no-refresh controls have `assignment_refresh_interval_train_iter=0`.
- Confirm RND rows require RND metrics and stock rows do not.
- Confirm static quality rows use exact refs from a checked refs file, not a
  live leaderboard pointer.
- Confirm H100 function names resolve to
  `lightzero_curvytron_visual_survival_h100_cpu40`.
- Capture the generated manifest path and exact submit command in this doc set
  or in a run-specific child note.

## What Not To Do

- Do not wait for a tiny 3-row read before preparing the RND lane.
- Do not mix RND into compact-owned speed claims.
- Do not make leaderboard quality a Wave A blocker.
- Do not feed tournament/leaderboard output back into Wave A trainers.
- Do not declare a positive RND result from blank-canvas survival alone.
- Do not promote latest checkpoint if best checkpoint regressed hard.
- Do not compare raw trainer reward across reward variants or RND weights.
