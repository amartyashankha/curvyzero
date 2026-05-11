# LightZero 8192/sim25 Postmortem Critic - 2026-05-09

Scope: installed `LightZero==0.2.0` official Atari Pong, run
`lz-visual-pong-8192-sim25-s0`, attempt
`train-8192-sim25-b64-env4-auto`. This is not GitHub-main exact upstream, not
the older 96x96 pretrained checkpoint lane, not custom dummy Pong, and not
CurvyTron. No code and no pytest.

## Bottom Line

The run is a useful mechanical control and a weak learning result.

It proved that the installed 64x64 LightZero Atari path can run on Modal L4,
write many strict-loadable checkpoints, and be evaluated through both the
manual policy path and the stock `MuZeroEvaluator` path. It did not prove Atari
Pong learning. The periodic checkpoints still look like brittle early MuZero
snapshots that choose one action for the whole capped eval window.

The strongest explanation is not "LightZero cannot learn Pong." The strongest
explanation is that this was still a small, off-recipe, hard-to-account run:
`8192` env steps and `25` simulations are much closer than earlier probes, but
still far below installed-package stock `200000` env steps and current
GitHub-style `500000` env steps. Batch size, collectors, evaluator count,
episode caps, segment length, and checkpoint cadence were also still not stock.

## What Collapsed

The periodic checkpoint curve did not improve:

- `iteration_0`: all action `3`, return `-6`.
- `iteration_100`: all action `0`, return `-6`.
- `iteration_500`: all action `5`, return `-6`.
- `iteration_900`: all action `0`, return `-6`.
- `iteration_932`: all action `1`, return `-6`.

Those checkpoints are not all collapsing to the same action. They are
collapsing to the same kind of behavior: one deterministic action for the whole
256-step eval. That is action collapse, even when the chosen action changes
between checkpoints.

The likely cause is noisy early learning on too little data. Restoring stock
`update_per_collect=None` let LightZero compute many learner updates from the
replay-ratio accounting, but the data budget was still only `8192` env steps.
That produced `iteration_0..932`, not because the run became stock-sized, but
because update accounting expanded inside a small collection budget. In plain
terms: many updates, a small and early replay pool, cheap eval/checkpoint
cadence, and deterministic MCTS eval are a good recipe for arbitrary root
preferences to harden into one-action policies.

This also explains the checkpoint burst. The wrapper label
`max_train_iter=64` was misleading once `update_per_collect=None` was restored.
Stock LightZero update accounting took over, and `save_ckpt_after_iter=1`
mirrored roughly 934 checkpoint files, around 90 GB. That is a setup-fidelity
lesson: stock update semantics were closer, but our checkpoint retention and
reporting were not bounded enough.

## Why `ckpt_best` Is Suspicious

`ckpt_best` does not behave like the periodic checkpoints:

- Manual eval used all six actions and got `0.0` over 256 steps.
- Stock evaluator still got `-6.0`.
- Manual and stock first-32 actions did not match.
- Logged policy logits in the `ckpt_best` step details were all zero.
- The experiment note says the `ckpt_best` file was smaller.

That is not a quality win. It is a parity warning.

Follow-up checkpoint diff now makes this much clearer. A Modal state diff
compared `iteration_932` against `ckpt_best`:

```text
app: ap-yIGfkon1zNYV11hsjyxWO6
artifact: training/lightzero-official-visual-pong/lz-visual-pong-8192-sim25-s0/attempts/train-8192-sim25-b64-env4-auto/eval/checkpoint_state_diff/iteration_932_vs_ckpt_best_20260509T223817Z.json
sha256: 8bfe73bcacf4f4fa72f0cb96dc5838f098b75c40ad574e790582e184371a2fbf
```

The diff found the same model key set and shapes, but reset-looking metadata and
state:

- `iteration_932`: file size `96,211,827`, `last_iter=932`,
  `last_step=3728`, optimizer state count `97`, first norm batch counter
  `5592`, nonzero running means, trained running variances.
- `ckpt_best`: file size `64,190,491`, `last_iter=0`, `last_step=0`,
  optimizer state count `0`, first norm batch counter `0`, running means all
  `0`, running variances all `1`, norm weights all `1`.

Plain read: `ckpt_best` looks like an initial or reset-style checkpoint, not the
best learned policy from this run. Do not use the manual `ckpt_best` return `0`
as learning evidence.

The most likely reasons are mundane:

- `ckpt_best` is probably being created before meaningful training state is
  present, or through a save path that does not reflect the final/best learned
  model for this wrapper.
- Manual eval and stock eval still differ in reset/evaluator protocol, frame
  stack lifecycle, episode handling, and possibly seed path.
- All-zero logged logits mean the row is not explaining its own decisions well;
  MCTS tie-breaking, value heads, visit counts, or instrumentation may be doing
  more work than the policy-logit field suggests.

The right read is: ignore `ckpt_best` for this run's quality curve unless the
LightZero best-checkpoint save path is explained. Use periodic checkpoints for
the learning read.

## Trainer Reward `-21` Versus Stock Eval `-6`

These are different measurements.

`-21` is the trainer-side final reward. In Atari Pong, `-21` means losing a
full game. It is a bad result, but it is probably closer to a complete-episode
read.

`-6` is from the post-train checkpoint curve with a 256-step eval cap. The
policy lost points at steps like `60, 95, 130, 165, 200, 235`, then the eval
window stopped. So `-6` mostly says "this bad policy lost every point available
inside the short window." It is not a full Pong score.

There is no contradiction. The same weak policy can score `-6` in a short
capped eval and `-21` in a longer trainer-side/full-game eval. The `-6` rows are
useful for comparing checkpoint behavior cheaply; they are not the final
quality metric.

## Worthwhile Next Checks

Do these before any larger train:

1. Run one `ckpt_best` parity eval with manual and stock paths forced as close
   as possible: same reset seed, same cap, same evaluator env count if feasible,
   same frame-stack source, and logged first actions, rewards, root visits,
   values, and logits.
2. Re-evaluate a tiny selected set at a longer cap, for example 1024 steps:
   `iteration_0`, one mid checkpoint, `iteration_932`, and `ckpt_best`. This
   separates "short-window `-6` artifact" from "full-game loss."
3. Add bounded accounting before another train: learner updates performed,
   env steps collected, replay size, checkpoint cadence, retained checkpoint
   count, and total artifact size.
4. If training again, choose a fidelity target explicitly: either installed
   `0.2.0` non-segment stock toward `200000`, or exact GitHub/upstream toward
   `500000`. Do not call another hybrid run "stock" without naming the
   remaining gaps.

## Rabbit Holes

Do not spend the next loop on these unless the checks above produce new
evidence:

- More action-mapping work. The six ALE actions are valid; action `5` is
  `LEFTFIRE`, not an out-of-range bug.
- More manual evaluator rewrites. Stock evaluator parity now exists well enough
  to show that the periodic checkpoints are bad.
- Jumping to sim50 or a bigger run immediately. Larger search will not fix
  unbounded update/checkpoint accounting or a suspicious `ckpt_best`.
- Treating `ckpt_best` manual `0.0` as proof of learning. It is currently an
  inconsistency to explain.
- Treating the 96x96 OpenDILab pretrained checkpoint mismatch as the same
  incident. That is a separate config/checkpoint-surface problem.
- Reporting this as CurvyTron or custom dummy Pong progress. It is only a stock
  Atari control lane.

## Decision Read

The result is not a reason to abandon LightZero, but it is a reason to stop
scaling casually. The immediate problem is observability and setup control:
we need to know what each checkpoint means, why `ckpt_best` differs, how many
updates happened, and whether a longer, explicitly faithful recipe changes the
curve.

Until then, call 8192/sim25 an infrastructure pass, a checkpoint-accounting
warning, and a policy-quality fail.
