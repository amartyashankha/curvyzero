# LightZero Debug And Scale Plan - 2026-05-10

Purpose: keep the Pong lane from drifting while current training and eval jobs
continue.

## Plain Goal

We are trying to prove one simple thing first:

Can our normal LightZero Atari Pong setup train a real Pong policy that gets
better than its own starting checkpoint under the stock evaluator?

That is the proof lane. It uses normal LightZero Atari Pong, stock reward, and
same-run checkpoint comparisons. It is not CurvyTron, not custom dummy Pong, and
not survival-shaped reward.

The main measurement is survival time. Report stock steps survived first.
Return changes matter only after survival is visible. A move from `-21` to
`-20` is weak by itself; the real question is whether the checkpoint survives
longer than its own starting checkpoint and whether that improvement lasts.

If this proof is credible, then CurvyTron gets a better next step. If Pong stays
unclear, moving to CurvyTron mostly moves the confusion.

## What We Have Seen

- Old normal seed `1` and seed `3` runs did show later same-run improvement.
  The useful movement appeared around `10000` to `18000` learner iterations,
  not at the first checkpoint.
- Seed `1` L4/T4 had the clearest normal result: late checkpoints survived much
  longer than `iteration_0`, and the latest strict rerun still showed a big
  same-run gain even though the exact stock return changed.
- Seed `3` had a middle checkpoint bump, then partly fell back.
- Current broad early sweep rows, mostly `1000` to `3000`, are mostly flat.
  Later current rows show survival bumps in seeds `13`, `18`, and `19`, but
  not stable survival improvement yet.
- Seeds `17` and `19` showed weak survival bumps, but not stock-return learning.
- Survival-shaped rows are side-lane telemetry. Some shaped runs show small
  survival movement, but shaped reward is not proof that normal Pong learns.
- H100 was faster in short profiles, but the run still looked mostly blocked by
  collection/eval/env work rather than GPU learning. Treat it as possibly
  CPU-bound until better profiles say otherwise.
- Quiet eval stdout landed, so evals can now run without flooding the console.

Plain read: early flat rows are not enough to call failure, because the older
positive rows were later. But the current evidence is still mixed and not a
stable Pong-learning claim.

## Top Hypotheses

1. Undertrained sparse reward.
   Pong gives little useful reward early. `1000` to `3000` may simply be too
   early for this setup.

2. Eval seed or episode determinism.
   One eval episode can hide or exaggerate a checkpoint. We need compact curves
   and, when possible, repeat eval seeds before trusting exact numbers.

3. Not enough later checkpoints.
   The old signal appeared near `10000` to `18000`. Early-only curves answer a
   different question.

4. Possible eval/tooling mismatch.
   Manual rollout and stock evaluator can disagree. Stock evaluator fields are
   the main proof surface; manual survival is useful telemetry, not a substitute.

5. CPU/collection bottleneck.
   H100 did not look fully used. More GPU may not buy much unless collection,
   eval, and MCTS work speed up too.

6. Reward shaping is not proof.
   Survival-shaped reward may help inspect signal, but it changes the objective.
   It cannot be reported as normal stock/control Pong progress.

## Next 5 Actions

1. Build later compact eval curves for normal Pong.
   For each live normal run, eval same-run `iteration_0` plus later checkpoints
   such as `1000`, `5000`, `10000`, `16000`, and latest/final under strict
   no-fallback stock evaluator. Lead with stock survival, stock episode length,
   action histogram, then stock return and reward counts.

2. Run parallel shaped evals, but keep them boxed.
   Eval shaped checkpoints against their own shaped `iteration_0`. Always label
   them as survival-shaped side-lane telemetry, never as normal Pong proof.

3. Compare speed across CPU, L4, L4+CPU, H100, and H100+CPU with the same
   profile metadata.
   Record wall time, collection time, eval time, learner time, GPU use, CPU
   label, checkpoint cadence, env count, simulation count, and run id. Do this
   for normal LightZero and, later, repo-native CurvyTron so the lanes are
   comparable.

4. Launch larger normal runs only after cadence is clear.
   Do not start another big 199k normal wave until we know checkpoints appear
   reliably, eval queue names are clear, and compact curves can be fetched and
   summarized without confusion.

5. Prepare CurvyTron adapter only after Pong proof is credible.
   Keep adapter notes ready: visual frames, discrete ego actions, reset seed,
   action mask, `to_play=-1`, survival-time reward, done/info, and joint-action
   logging. Do not treat adapter work as the main proof lane yet.

## Stop Doing

- Stop reading early `1000` rows as proof of failure.
- Stop mixing shaped reward claims with normal stock/control Pong claims.
- Stop comparing manual rollout and stock evaluator as if they are identical.
- Stop mixing stock Atari Pong, custom dummy Pong, and CurvyTron in one score
  story.
- Stop reporting `-21` to `-20` as the main signal. Survival time comes first.
- Stop using unclear names. Run ids, attempt ids, eval ids, and summaries must
  say normal versus survival-shaped, seed, compute, checkpoint range, eval cap,
  and stock evaluator.
- Stop launching larger runs just because a GPU is available. First make the
  checkpoint/eval/reporting loop boring and repeatable.

## Decision Rule

Call Pong credible only when normal stock/control runs show later same-run
survival improvement under the stock evaluator, with clear checkpoint refs and
clean names, and with enough repeats that it is not just one lucky row. Stock
return can support the claim, but it is not the first signal.
