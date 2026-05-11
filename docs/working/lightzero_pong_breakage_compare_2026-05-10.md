# LightZero Pong Breakage Compare - 2026-05-10

Purpose: simple compare between the earlier official/control Pong weak signal
and the newer frequent-checkpoint attempts that looked broken.

## Claim

The earlier `32768` official/control run showed a weak but real same-run
learning signal: `iteration_9092` was less bad than that run's own
`iteration_0` under strict no-fallback stock-ish eval.

The frequent-checkpoint lane has not shown a policy-quality failure. The first
cadence attempt failed before `iteration_1000` because the Modal local client
disconnected. The current clean attempt is the post-patch `spawn` launch and is
too early to judge until it produces `iteration_1000+` and that checkpoint is
evaluated against its own `iteration_0`.

## Non-claim

This does not prove solved Pong, exact upstream reproduction, or CurvyTron
readiness.

This does not prove LightZero failed on Pong.

This does not prove the current spawn run will finish or improve; it only says
the currently visible evidence is launch/cadence lifecycle evidence, not a
learning result.

## Terms

- Checkpoint: a saved model snapshot.
- `iteration_0`: the starting checkpoint for an attempt, before useful learner
  updates.
- Same-run baseline: compare a later checkpoint only to `iteration_0` from the
  same attempt.
- Frequent checkpoint: save normal checkpoints every `1000` learner iterations
  instead of the stock `10000`.
- Broken: code or infrastructure failed before producing the artifact needed
  for the intended comparison.
- Early: training has started, but there is not yet a later checkpoint to
  compare.

## What Changed

Earlier successful weak-signal run:

- Attempt:
  `train-faithful-short-installed-0.2.0-s0-32768-relpath`.
- Checkpoint cadence stayed stock, so it produced `iteration_0` and later
  `iteration_9092`.
- Strict eval compared `iteration_0` and `iteration_9092` from the same run.
- Result: both survived the 512-step cap, so `delta_steps_survived=0`, but
  return and loss events improved: manual return `-13 -> -8`, stock return
  `-13 -> -10`, nonzero rewards `13 -> 8`, positive rewards stayed `0`.

Frequent-checkpoint changes:

- Same installed `LightZero==0.2.0` Atari Pong lane.
- Same faithful-short env-step cap style.
- Added `--save-ckpt-after-iter-override 1000`.
- Wrapper patch sets
  `policy.learn.learner.hook.save_ckpt_after_iter = 1000`.
- Current wrapper also launches train mode with `train_fn.spawn(...)`, so a
  long remote training call is not tied to a local `.remote()` caller.

## What Actually Broke

There were two real infrastructure issues.

1. The first cadence-override launch failed before training with
   `KeyError('learn')`. The wrapper now creates
   `policy.learn.learner.hook` before setting the checkpoint cadence.

2. The next cadence attempt
   `train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-relpath` started
   training but stopped at about `340.736` seconds. Its progress artifact says
   `phase=failed`, `actual_save_ckpt_after_iter=1000`, and
   `checkpoint_count=2` (`iteration_0` plus `ckpt_best`). Logs reported local
   client disconnect. It never reached `iteration_1000`, so it never tested
   checkpoint cadence as a learning curve.

The intermediate detached attempt
`train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-detached-relpath` still
went through the old `.remote()` path. Modal logs warned that `.remote()` in
detached apps may be canceled when the local caller disconnects. Its latest
progress artifact at `2026-05-10T00:50:08Z` showed `phase=running`,
`train_elapsed_sec=363.230029879`, and only `iteration_0` plus `ckpt_best`.
Treat it as lifecycle-noisy, not as the clean run.

## Current Run Read

The current clean run named by the board is:

```text
app: ap-h7bpMSwDDW6f0eIOv7Cfdl
function_call: fc-01KR7NR7XQNTQ4GF75EQ3Z6TE0
attempt: train-faithful-short-installed-0.2.0-s0-32768-ckpt1000-spawn-relpath
```

This is the post-patch `spawn` launch. At the narrow check here, the app was
listed as detached with one task, the checkpoint directory had `ckpt_best`, and
logs showed Atari env activity (`one episode done`). The progress file still
showed only the initial `starting` snapshot. That is not enough to call the run
broken or good. It is simply before the first useful normal checkpoint.

## Bottom Line

The old run looked weak because the policy was weak, but the comparison was
valid: later same-run checkpoint versus same-run `iteration_0`.

The frequent-checkpoint lane looked broken because the first attempts were
blocked by real launch/caller-lifecycle bugs before `iteration_1000`. The
current clean `spawn` attempt should be judged only after it has
`iteration_1000+` and a strict same-run eval.

## Next Action

Poll only the current spawn attempt. Run strict eval when
`iteration_1000.pth.tar` or a later normal `iteration_*.pth.tar` appears, and
compare it against that same spawn attempt's `iteration_0`. Do not eval or
interpret `ckpt_best` as the signal-curve checkpoint.

No pytest was run.
