# v2refresh18p Max-Step / Resume Audit - 2026-05-15

Read-only audit before raising trainer/tournament step caps from `65_536` to
`1_048_576`.

## Current Metrics

- Manifest:
  `artifacts/local/curvytron_tonight18_manifests/curvy-v2refresh18p-20260514b/curvy-v2refresh18p-20260514b.json`.
- Current manifest settings: 18 rows, `source_max_steps=65536`,
  `max_train_iter=300000`, `opponent_assignment_refresh_interval_train_iter=50`,
  and all rows still carry `initial_policy_checkpoint_ref`.
- Latest checkpoint range from `curvyzero-runs-v2` is roughly `70k-130k`.
- Eval-summary command:
  `uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvytron_run_status --run-ids "$RUN_IDS" --attempt-ids "$ATTEMPTS" --output eval-summary`.
- Latest eval read: average first mean `131.25`, average latest mean `159.88`,
  average best mean `215.94`; `14/18` rows latest > first and `17/18` rows best
  > first. This is improving but noisy; several rows have collapsed-action
  flags or peak-then-regress behavior.

## Resume Safety

- Same-run auto-resume is guarded against silently skipping champion bootstrap.
  With the current manifest unchanged, relaunching the same run IDs would find
  existing checkpoints and then block because `initial_policy_checkpoint_ref` is
  still set.
- Auto-resume selection scans current/prior `lightzero_exp*/ckpt` dirs and the
  run checkpoint mirror, and selects the highest non-empty `iteration_*.pth.tar`.
- Stock LightZero checkpoint resume is expected to restore learner/policy state,
  including model, target model, and optimizer. CurvyZero resume sidecars add
  collector/evaluator counters, policy wrapper extras, and RNG. Raw replay
  GameSegments and live env-manager internals are not restored.
- Assignment refresh state is not a full historical resume surface. On a resumed
  process, the refresh hook starts from the launch assignment context, then
  checks the pointer on the first bucket and applies/marks unchanged from there.
  Refresh index telemetry can reset even when assignment sha is correct.

Recommendation: use fresh run IDs for the `1_048_576` cap run. Same-run resume
is only appropriate for intentional interrupted-run recovery after removing
`initial_policy_checkpoint_ref` and canarying one row.
