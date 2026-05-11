# Pong Discrepancy Action Plan - 2026-05-11

Purpose: turn the new Pong survival signal into a short investigation plan.
This is the bridge between "we saw learning" and "we know what not to break
when moving toward CurvyTron."

## Current Plain Read

We have real Pong survival signal now. The cleanest row is installed LightZero
0.2.0 stock64 `s122`: mean survival moved
`761.5 -> 1378.06 -> 1591.69 -> 1977.62` by `iteration_26000`. Score improved
more slowly, so survival is the lead metric.

The broader stock64 lane also improved after longer waits: `s114`, `s120`,
`s121`, `s142`, and exact controls `s113/s123` all show later survival gains.
Do not harden the old story that `s122` is uniquely good. The current question
is stability and speed of learning, not whether the basic stock Pong setup can
learn at all.

## What Probably Went Wrong Before

- We judged sparse-reward Pong too early. Many early `1k` or `5k` reads were
  not enough.
- We let score hide survival signal. Survival moved before score looked good.
- We mixed proof lanes with custom/debug lanes. Custom Pong, shaped reward
  runs, and CurvyTron adapter smokes are useful, but they are not stock Pong
  replication.
- Some Modal launches produced no visible Volume root. That is a run-lifecycle
  problem, not a policy result.
- Custom CurvyTron/two-seat code skipped parts of the native LightZero
  `train_muzero` path. Direct `learn_mode.forward` is not the same as the
  stock collector/replay/learner loop.
- Slow or serial evals delayed decisions and made us argue from stale data.

## Active Questions

- Does `s142` keep improving toward the stronger `s122` curve?
- Do the weaker stock rows keep improving by `20k/50k`, or plateau early?
- Is the current GitHub upstream segment path learning when evaluated with a
  proper cap and parallel seed panel?
- Which custom CurvyTron path pieces are safe diagnostics, and which must be
  replaced by native-compatible GameSegment/replay/learner behavior?
- What is the minimum CurvyTron training surface that preserves the stock
  LightZero contract while using survival reward?

## Near-Term Work

- Keep `s122` and `s142` maturing; evaluate survival curves at later
  checkpoints. Use a higher eval cap for strong checkpoints because `s122`
  is already near the `2048` cap.
- After a long wait, re-evaluate the newest checkpoints before making a
  stronger claim about which stock rows work.
- Use the failure audit to retire no-evidence runs rather than re-litigating
  them.
- Use the custom-vs-stock contract autopsy to decide which CurvyTron code path
  can be trusted.
- Record every claim as: same-run baseline, survival curve, score, checkpoint
  refs, eval settings, and non-claim.

## Linked Docs

- [Pong replication failure audit](pong_replication_failure_audit_2026-05-11.md)
- [Stock64 signal comparison](pong_stock64_signal_comparison_2026-05-11.md)
- [MuZero training footguns](muzero_training_footguns_2026-05-11.md)
- [LightZero Pong replication monitor](../lightzero_pong_replication_monitor_2026-05-11.md)
- [Active board](../training_coach_active_board_2026-05-10.md)
