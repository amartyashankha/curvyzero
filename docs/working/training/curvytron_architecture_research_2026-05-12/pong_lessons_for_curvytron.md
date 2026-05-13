# Pong Lessons For CurvyTron

Purpose: capture the earlier Pong pattern because it looks uncomfortably
similar to the CurvyTron failure.

Status: paper-trail pass completed 2026-05-12.

## Short Analogy

In Pong, custom or near-custom training paths looked plausible but left the
learning story ambiguous. They proved wrappers, loaders, replay artifacts,
scorecards, and failure visibility. They did not give a clean policy-quality
answer.

The first credible learning signal came from a path much closer to stock
LightZero MuZero, judged with strict checkpoint loading and survival curves
after enough training horizon.

CurvyTron appears to be repeating the same shape:

```text
custom path solves one obvious problem
  -> many extra training-contract changes sneak in
  -> artifacts exist but learning is unclear or flat
  -> stock/closer framework path is needed as the control
```

## CurvyTron Comparison

| Historical Pong pattern | CurvyTron repeat |
| --- | --- |
| Custom dummy/self-play Pong was useful plumbing but weak policy evidence. | Custom two-seat CurvyTron solved simultaneous action collection but bypassed stock replay/targets. |
| LightZero dummy Pong used real MuZero pieces but custom env/eval/replay choices made quality ambiguous. | `two-seat-selfplay` used `MuZeroPolicy` and `learn_mode.forward`, but not stock `train_muzero`, collector, or `MuZeroGameBuffer`. |
| Stock/near-stock Atari Pong eventually showed survival movement after longer horizon. | CurvyTron needs a stock-loop control (`source_state_fixed_opponent`, frozen opponent, or centralized joint-action) or a native-buffer bridge before large claims. |
| Early Pong rows were over-read before strict eval and survival curves. | May 12 CurvyTron flat curves should be read as custom-contract failure evidence, not as a game-learnability verdict. |

## Pong Timeline

1. **Custom dummy/self-play Pong proved plumbing, not quality.**
   Modal train-to-scoreboard worked, but learned checkpoints stayed weak against
   `track_ball` and needed survival/loss-delay reporting to avoid over-reading
   win counts. See
   `docs/experiments/2026-05-09-modal-pong-selfplay-repair-run.md:54` and
   `docs/experiments/2026-05-09-modal-pong-selfplay-repair-run.md:102`.

2. **LightZero dummy Pong used MuZero but remained a custom lane.**
   It added env adapters, checkpoint loading, MCTS scoreboards, frozen/scripted
   opponents, and target telemetry. Independent scorecards repeatedly found
   degenerate policies such as all-up or all-stay, or results too small to claim
   quality. See
   `docs/experiments/2026-05-09-lightzero-dummy-pong-longer-run.md:366`,
   `docs/experiments/2026-05-09-lightzero-dummy-pong-lagged-opponent-smoke.md:173`,
   and
   `docs/experiments/2026-05-09-lightzero-dummy-pong-frozen-checkpoint-selfplay-iter16.md:161`.

3. **Stock Atari Pong first cleared infrastructure, then parity.**
   The wrapper moved to installed LightZero `0.2.0`,
   `zoo.atari.config.atari_muzero_config`, `PongNoFrameskip-v4`, conv MuZero,
   and `lzero.entry.train_muzero`. Early 4096/sim10 and 8192/sim25 rungs still
   failed quality gates, but strict-load and stock-evaluator parity were made
   explicit. See
   `docs/experiments/2026-05-09-modal-lightzero-pong-tiny-train-smoke.md:22`,
   `src/curvyzero/infra/modal/lightzero_pong_tiny_train_smoke.py:697`, and
   `docs/experiments/2026-05-09-modal-lightzero-pong-4096-sim10.md:270`.

4. **Credible signal arrived late on the stock64 lane.**
   The useful read was stock/near-stock visual Pong with sparse reward, strict
   stock eval, same-run `iteration_0` baselines, enough checkpoint horizon, and
   survival-first curves. Later evals showed survival gains across multiple
   seeds, not just `s122`. See
   `docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:13`
   and `docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:66`.

## What Went Wrong

- Custom/debug lanes were treated as if they could answer stock-Pong learning
  questions. They could not; they mostly tested wrappers, telemetry, checkpoint
  loading, and eval harnesses. See
  `docs/working/training/pong_replication_failure_audit_2026-05-11.md:26`.
- Sparse Pong was judged too early. Many `1k`/`5k` reads were flat; later
  checkpoints moved. See
  `docs/working/training/pong_discrepancy_action_plan_2026-05-11.md:20`.
- Score hid early progress. Survival steps moved before score looked good. See
  `docs/working/training/pong_stock64_signal_comparison_2026-05-11.md:70`.
- Trainer-side rewards and checkpoints were not enough. Independent strict eval
  could reveal constant-action policies that train-side summaries made look
  alive.

## What Worked

- Use the stock LightZero spine when possible: collector, GameSegment,
  GameBuffer, learner, checkpoints, and evaluator.
- Compare same-run checkpoints under the same eval contract, especially
  `iteration_0` versus later checkpoints.
- Make strict load and no-fallback mandatory for learning claims. The eval queue
  enforces `--no-allow-model-fallback --run-stock-evaluator`, with `--stock-only`
  for scorecard work; see `scripts/lightzero_live_eval_queue.py:211`.
- Lead with survival curves and keep score, positive rewards, terminal reason,
  action histogram, and fallback/strict-load status beside them. See
  `src/curvyzero/infra/modal/lightzero_pong_eval_smoke.py:2577` and
  `scripts/summarize_lightzero_pong_eval_manifest.py:382`.

## Direct CurvyTron Lessons

- Do not treat `--mode two-seat-selfplay` as the main learning lane yet. It solves
  simultaneous action collection, but it does not call stock `train_muzero` and
  does not use LightZero's collector/GameBuffer target path. See
  `docs/working/training/curvytron_train_muzero_reconciliation_2026-05-12.md:116`.
- The Pong-like CurvyTron control is a stock-loop scalar-action route:
  `source_state_fixed_opponent`, recent frozen/checkpoint opponent, or
  centralized `source_state_joint_action`. Label each claim honestly. See
  `docs/working/training/curvytron_architecture_research_2026-05-12/path_matrix.md:8`.
- Before scaling true two-seat again, pass one gate: call stock `train_muzero`,
  feed native `GameSegment` / `MuZeroGameBuffer`, or prove the custom target
  builder against tiny known trajectories.
- CurvyTron eval claims should mirror the Pong discipline: same-run baseline,
  checkpoint refs, strict-load/no-fallback, fixed seed panel, opponent source,
  survival curve, sparse outcome, terminal cause, action histogram, and explicit
  non-claims.

## Claims To Keep Smaller

- Pong stock64 signal supports the process template, not a CurvyTron result.
- Pong custom-path failures did not prove MuZero was bad; CurvyTron custom-path
  flatness does not prove CurvyTron is bad.
- "Self-play" must name the opponent source: fixed, frozen, recent checkpoint,
  live same-policy two-seat, or centralized joint-action.
- A checkpoint curve is credible only when the trainer path, strict-load status,
  eval opponent, and reward/objective are stated beside it.
