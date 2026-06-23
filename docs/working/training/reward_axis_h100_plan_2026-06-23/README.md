# Reward Axis H100 Plan - 2026-06-23

Status: active campaign source of truth. No Modal jobs were launched from this
doc set.

## Plain Read

The current learning question is not "does CurvyTron train at all?" The useful
prior evidence says it often finds better mid-run checkpoints and then regresses.
The reward axis should therefore be judged by survival retention, tournament
exposure, action collapse, and learner metrics, not by latest checkpoint or raw
trainer reward alone.

Most likely next main reward arm: `survival_plus_bonus_plus_outcome`, with a
small alpha/support/cadence control around it. It was the least-bad latest
survival arm in the r18fresh readouts and is favored by the later reward-axis
notes. It is not a free win: dense support saturation and terminal-outcome
volatility remain the main risks.

Most important control arm: `survival_plus_bonus_no_outcome`. It is the clean
survival-plus-bonus objective, the compact default, and the easiest way to
separate "survival shaping works" from "terminal outcome is stabilizing or
destabilizing learning."

Most important diagnostic arm: `sparse_outcome`. It is the clean game-outcome
baseline. It can find strong mid-run checkpoints, but long-horizon credit and
late retention are weak under current settings.

RND reorientation: RND should now run as a first-class parallel lane. The code
and plumbing exist, and focused tests have historically passed, but no positive
RND learning result is proven yet. Treat RND as its own hypothesis, not as an
extrinsic reward variant and not as part of the compact speed claim.

Curriculum reorientation: leaderboard comes much later. Wave A should lean on
exact trained checkpoint refs, simple static opponent mixtures, and long
horizons. Tournament/leaderboard is a later selection layer after nonzero
checkpoints exist, not the first curriculum source.

## Files

- `AGGRESSIVE_REORIENTATION.md`: broad-lane allocation and operating stance.
- `LONG_HORIZON_CURRICULUM.md`: long-run interpretation, exact-ref curriculum,
  and the simplified no-leaderboard Wave A shape.
- `NEXT_HOURS_OPERATING_PLAN_2026-06-23.md`: high-level objective, signal
  hierarchy, runtime-tier profiles, and the next 5-minute through 8-hour plan.
- `LAUNCH_QUEUE.md`: operator-facing queue of manifest-ready and reserved
  Wave A lanes.
- `WAVE_A_MANIFESTS.md`: prepared local manifests, dry-run results, launch
  commands, and remaining launch gates.
- `PRELAUNCH_AUDIT_2026-06-23.md`: current non-launch audit showing that the
  original exact-ref non-RND manifests are remote-ref blocked in Modal.
- `PRELAUNCH_REPAIR_2026-06-23.md`: repaired exact-ref non-RND manifests using
  currently visible `curvy-r18fresh-*` refs, with Modal ref audits and dry-runs.
- `WAVE_A_LAUNCH_REVIEW_2026-06-23.md`: approval packet draft with intended
  row counts, launch command shapes, status commands, and cleanup boundary.
- `artifacts/local/curvytron_wave_a_launch_packet_audit_20260623a.json`: latest
  no-launch packet audit for the repaired 90-row Wave A package.
- `artifacts/local/curvytron_wave_a_capacity_snapshot_20260623a.json`: latest
  no-launch Modal app-list capacity proxy for the Wave A approval gate.
- `STOCK_PATH_RND_REORIENTATION.md`: why RND belongs on the original
  stock-ish LightZero path now, and why compact remains no-RND.
- `CONTINGENCY_PLANS.md`: what can go wrong and the fallback ladder by horizon
  and lane.
- `OPERATING_PATTERNS.md`: how to run, monitor, delegate, stop, and promote
  lanes without mixing claims.
- `CRITICAL_RETROSPECTIVE.md`: the past lessons that should constrain the new
  campaign.
- `RND_LANE.md`: what happened to RND, the most believable implementation, and
  the first aggressive sweep.
- `REWARD_INVENTORY.md`: all reward functions and where they are wired.
- `EXPERIMENT_PLAN.md`: staged H100 experiment plan and initial rows.
- `MONITORING_SIGNALS.md`: what to watch, when to believe it, and when to stop.

## Source Anchors

- Reward constants: `src/curvyzero/contracts/curvytron.py`
- Reward contracts/supports: `src/curvyzero/training/reward_contracts.py`
- Source-state reward implementation:
  `src/curvyzero/training/curvyzero_source_state_visual_survival_lightzero_env.py`
- Manifest builder:
  `scripts/build_curvytron_tonight18_manifest.py`
- Grouped submitter:
  `scripts/submit_curvytron_survivaldiag_manifest.py`
- Wave A packet auditor:
  `scripts/audit_curvytron_wave_a_launch_packet.py`
- Wave A capacity auditor:
  `scripts/audit_curvytron_wave_a_capacity.py`
- RND blank-canvas builder:
  `scripts/build_curvytron_rnd_blank_sweep_manifest.py`
- CZ26 next-batch builder, later live-refresh lane:
  `scripts/build_curvytron_next_batch_manifest.py`
- Status reader:
  `src/curvyzero/infra/modal/lightzero_curvytron_run_status.py`
- Prior r18fresh/postmortem docs:
  `docs/working/training/r18fresh_postmortem_2026-05-16/`
- Current learning-loop docs:
  `docs/working/training/leaderboard_to_training_2026-05-13/`

## Current Open Work

1. Use `LAUNCH_QUEUE.md` as the front door for current lane state.
2. Treat the original `curvy-n18conn-*` exact-ref non-RND manifests as
   historical/ref-blocked until their missing refs are restored.
3. Use `PRELAUNCH_REPAIR_2026-06-23.md` for the current repaired non-RND launch
   candidates; they are manifest-ready and Modal-ref-audited, but not launched.
4. Before launching prepared manifests, choose the runtime tier, confirm active
   H100 capacity, and get explicit human approval for exact commands, row
   counts, and intended timeout. Rerun
   `scripts/audit_curvytron_wave_a_launch_packet.py` first and require
   `ok=true` with no launch artifacts. Then rerun
   `scripts/audit_curvytron_wave_a_capacity.py`; treat
   `operator_capacity_review_required` as a required human capacity decision,
   not as launch approval.
5. Prefer seeded exact checkpoint refs for static quality rows; keep scratch as
   an explicit diagnostic.
6. Defer CZ26-style leaderboard/refresh slices until after static lanes show
   trainable nonzero checkpoints.
7. Keep RND positive rows separate from compact speed and extrinsic reward
   claims until retained extrinsic quality is proven.
8. Keep non-RND static reward and cadence/support lanes alive alongside RND;
   never interpret positive RND alone.
