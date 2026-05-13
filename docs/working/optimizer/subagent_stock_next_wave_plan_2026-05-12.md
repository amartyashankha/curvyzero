# Stock Frozen Next-Wave Profile Plan

Date: 2026-05-12

Scope: Optimizer-only plan for the trusted stock LightZero path:

```text
lzero.entry.train_muzero
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--opponent-use-cuda=false
```

Do not launch these from this note. This is a profile/scaling plan, not a
learning recommendation.

## Current Bottleneck Read

The stock frozen path is not currently "just GPU-bound." The short normal-death
profiles say subprocess collection scales through `collector_env_num=16` on the
40-CPU L4 shape, while learner time stays small and named MCTS/search is not the
dominant wall bucket. That means more collector width is still plausible, but
only as a throughput knob.

Base-manager attribution says the stock frozen path has several CPU-visible
terms: env step, stack update, render, and frozen-checkpoint opponent inference.
The frozen opponent costs real time versus fixed-straight, but it is not the
only limiter. `body_circles_fast` helps a small short-episode row a little, but
does not yet justify changing the stock visual surface.

The long-survival/no-death warning is the sharpest Amdahl clue: a subprocess C2
row spent almost all wall time in collector collect, while named MCTS and named
policy forward were much smaller. Because subprocess hides worker internals, the
likely story is env/render/opponent worker work, but the next proof needs
base-manager attribution before scaling that case.

Old custom two-seat B64/B128/H100 rows are postmortem evidence only. They can
inform Amdahl intuition, especially that bigger GPU does not erase CPU terms,
but they should not be used as Coach training settings.

## Common Controls

Use these controls unless a row below overrides them:

```text
--mode profile
--env-variant source_state_fixed_opponent
--opponent-policy-kind frozen_lightzero_checkpoint
--opponent-checkpoint-ref training/lightzero-curvytron-visual-survival/curvytron-dense-ckpt1-iter10000-sanity-20260512a/checkpoints/lightzero/iteration_32.pth.tar
--opponent-use-cuda=false
--reward-variant sparse_outcome
--source-state-trail-render-mode browser_lines
--batch-size 16
--num-simulations 8
--source-max-steps 256
--max-train-iter 64
--max-env-step 8192
--evaluator-env-num 1
--n-evaluator-episode 1
--lightzero-eval-freq 0
--skip-lightzero-eval-in-profile=true
--stop-after-learner-train-calls 5
--save-ckpt-after-iter 9999
--no-background-eval-enabled
--no-background-gif-enabled
--profile-allow-auto-resume=false
```

Every accepted row must report `called_train_muzero=true`, env manager type,
collector/evaluator counts, learner device, opponent device, `opponent_use_cuda`,
phase timings, MCTS root count, env steps, learner train calls, and checkpoint /
eval / artifact suppression.

## Cells

| Cell | Lens | Exact knob changes | Why it exists |
| --- | --- | --- | --- |
| A1 base long browser | Attribution | `--compute gpu-l4-t4`, `--env-manager-type base`, `--collector-env-num 1`, `--n-episode 1`, `--source-max-steps 512`, `--disable-death-for-profile=true`, `--stop-after-learner-train-calls 1`, `--source-state-trail-render-mode browser_lines` | Expose long-survival env/render/stack/opponent buckets that subprocess hides. This is the denominator for the no-death Amdahl read. |
| A2 base long fast-render ablation | Attribution | Same as A1, except `--source-state-trail-render-mode body_circles_fast` | Matched render attribution only. If wall barely moves, long-survival cost is not mostly trail rendering. If it moves a lot, render is a real stock bottleneck under long survival. |
| S1 C32 normal L4 | Scaling | `--compute gpu-l4-t4-cpu40`, `--env-manager-type subprocess`, `--collector-env-num 32`, `--n-episode 32` | Extends the passed C8/C16 stock frozen scaling line. Helps only if roots/sec or env steps/sec improve without proportional wall growth or worker failures. |
| S2 C16 sim16 L4 | Scaling | `--compute gpu-l4-t4-cpu40`, `--env-manager-type subprocess`, `--collector-env-num 16`, `--n-episode 16`, `--num-simulations 16` | Separates search-budget pressure from collector-width pressure at a known-good collector width. |
| S3 C32 sim16 L4 | Scaling | `--compute gpu-l4-t4-cpu40`, `--env-manager-type subprocess`, `--collector-env-num 32`, `--n-episode 32`, `--num-simulations 16` | Tests whether wider roots make the GPU/search term material, or whether CPU env/opponent/render still dominates. Run after S1 and S2 are healthy. |
| G1 C32 sim16 H100 | Scaling | Same as S3, except `--compute gpu-h100-cpu40` | Paired GPU-type check at the first shape where H100 has a fair chance to matter. If policy/MCTS gets faster but wall barely changes, the limiter is CPU/worker side. |
| B1 C16 B32 L4 | Attribution | `--compute gpu-l4-t4-cpu40`, `--env-manager-type subprocess`, `--collector-env-num 16`, `--n-episode 16`, `--batch-size 32`, `--num-simulations 8` | Clarifies stock `batch-size` cost. In this path, collector width is the main self-play-width analogue; `batch-size` mostly probes learner/replay/policy config pressure, not old two-seat env width. |

## Do Not Test Yet

- Do not relaunch or scale `--mode two-seat-selfplay` as learning evidence.
- Do not use old custom B128/H100 rows as stock frozen recommendations unless
  the table is explicitly labeled postmortem/custom-adapter.
- Do not turn `--opponent-use-cuda=true` on with subprocess workers. The current
  trusted split is GPU learner, CPU frozen opponent.
- Do not jump to C64/C128, H100x2, multi-GPU, or distributed collectors before
  C32 and the no-death attribution rows are interpretable.
- Do not mix reward, learning rate, opponent age, checkpoint-pool refresh,
  background eval/GIF, or frequent checkpoint cadence into optimizer timing
  cells. Those are Coach or artifact-cost questions.
- Do not treat `body_circles_fast` as a training surface recommendation. It is
  only a render attribution lens.
- Do not compare raw `steps/s` between stock frozen and old custom two-seat
  runs; the denominators are different.

## Interpreting Larger Width

In stock frozen training, larger "self-play width" should mostly mean more
collector envs: more concurrent ego-vs-frozen games and more MCTS roots per
collection wave. It is not true live same-current-policy two-seat self-play.

More parallelism helps when roots/sec, env steps/sec, or completed segments/sec
rise faster than wall time, while learner time, replay/sample time, and artifact
time stay secondary. It stops helping when wall grows roughly with collector
count, subprocess failures appear, or H100 improves named policy/MCTS time
without moving total wall time.

Read every scaling row as throughput evidence only. A wider collector can make
more stock replay data per wall-clock minute, but it does not prove better
targets, better opponent curriculum, better exploration, or better learning.
Coach owns those claims through checkpoint curves, held-out eval, survival,
sparse outcome, reward components, and action distributions.
