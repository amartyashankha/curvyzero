# Training Iteration Loop Bottlenecks - 2026-05-09

Role: iteration-loop bottleneck critic. Scope is docs-only critique of why the
Coach loop is slow, especially LightZero train/eval waits. No code changes and
no pytest.

## Bottom Line

Coach iteration is slow because the loop is mostly serialized at whole-job and
whole-checkpoint boundaries. The expensive part is no longer "can we run one
LightZero job?" It is the wait between a hypothesis, a Modal train run, artifact
mirror, checkpoint selection, independent MCTS scorecard, manual fetch, manual
table summary, and docs promotion.

Do not weaken claims by shrinking eval below usefulness or relying on trainer
telemetry. Instead, reduce avoidable waiting: cache/build less often, evaluate
selected checkpoints in parallel, stop scoring every nearby checkpoint, automate
summary promotion, and keep detailed diagnostics behind explicit gates.

## Current Bottlenecks

1. Modal image/build/startup tax
   - Tiny LightZero jobs can finish trainer work in roughly the same order as
     Modal startup, import/config, artifact scan, and wrapper overhead.
   - A dry config smoke and small 512/8 train were around 12 seconds each,
     which means fixed overhead is a large share at smoke scale.
   - A packaging failure from changing `__pycache__` during build shows the
     loop can lose a full launch before training or eval starts.

2. Eval is serialized after train
   - The common path is train, inspect refs, launch one scorecard, fetch
     summary, read it, then launch another probe or scorecard.
   - This is defensible for first plumbing, but it is now the wall-clock drag.
     Independent checkpoints, opponents, seeds, and sim counts are naturally
     parallel jobs.

3. Per-checkpoint scorecard sprawl
   - Recent reports often score `iteration_0`, one or more intermediate
     checkpoints, final, and `ckpt_best` across multiple opponents.
   - That is useful for diagnosis, but most rows repeat the same decision:
     action collapse, zero or tiny `down`, no held-out quality win.
   - The main thread needs fewer checkpoint curves and more lane decisions.

4. MCTS simulations multiply every wait
   - `num_simulations` is both a quality knob and a direct cost multiplier for
     collection and independent eval.
   - Raising sims is legitimate when testing root-target quality, but broad
     scorecards at higher sims should be reserved for selected checkpoints.

5. Eval caps and episode counts are doing two jobs
   - Caps/episodes need to be high enough to support the claim, but many early
     rows only need a stop/go read: action entropy, `down` usage, scoreability,
     no-fallback status, and gross score movement.
   - Running full detail on every checkpoint makes weak branches expensive to
     reject.

6. Artifact fetch, Volume I/O, and checkpoint mirroring
   - LightZero checkpoints are large enough that scanning, copying, committing,
     and fetching repeated `.pth.tar` artifacts is noticeable.
   - Current artifacts are useful, but too many retained checkpoints turn every
     run into a storage and summary-management task.

7. Worker orchestration is too manual
   - Workers are producing good notes, but the handoff loop still depends on
     humans copying commands, refs, tables, and stop/continue decisions.
   - This creates idle time between train completion and the next credible eval.

8. Docs promotion is part of the critical path
   - The source ledger pattern is right: durable claims should cite promoted
     evidence, not raw impressions.
   - But every run currently asks for manual docs synthesis before the next
     worker knows which refs and claims matter.

9. Wrapper overhead and repeated constants
   - Repeated LightZero image, Volume, mount, checkpoint, and summary logic is
     tolerable for exploration, but now slows every change and increases the
     chance of mismatched eval config, horizon, or artifact path.

## Speedups That Preserve Claim Strength

1. Batch eval launch after each train
   - As soon as a train run writes checkpoints, launch one eval batch containing
     the agreed checkpoint set, opponents, seats, seeds, and sim count.
   - Use Modal `map`/`starmap` for independent scorecard rows.
   - Keep each row honest: same checkpoint hash, eval cap, split id, opponent,
     paired-seat setting, and no-fallback status.

2. Score fewer checkpoints by default
   - Default set: `iteration_0`, final/latest, and selected `best` only when a
     real selection pointer exists.
   - Add intermediate checkpoints only when the hypothesis is explicitly about
     curve shape or collapse timing.
   - For dead branches, stop after the first selected/final scorecard proves the
     gate failed.

3. Use two-tier eval detail
   - Tier 1 gate: small but honest scorecard with action histogram, score,
     survival, truncation, no-fallback status, and checkpoint hash.
   - Tier 2 diagnostic: larger episode count, multiple sim counts, root target
     sidecars, policy logits, and per-state oracle probes.
   - Promote to Tier 2 only when Tier 1 changes a decision or exposes a narrow
     bug worth isolating.

4. Parallelize root/sim sweeps
   - If comparing `num_simulations=2,8,16,25`, run those as independent eval
     rows, not serial follow-ups.
   - Keep broad sim sweeps on fixed states or selected checkpoints; do not run
     every checkpoint through every sim count.

5. Cache Modal images and reduce packaging churn
   - Keep the pinned LightZero image stable and avoid copying volatile files
     into the image context.
   - Build/cache once per dependency set; vary run config through arguments.
   - Add a cache Volume only when measured repeated dependency or compile cost
     appears, not as a default artifact sink.

6. Retain checkpoints selectively
   - Always retain init, latest/final, and any selected best.
   - Retain intermediate checkpoints only by cadence or trigger, for example
     first nonzero reward, action entropy change, root-target gate, or declared
     curve experiment.
   - Write `latest.json` and `best.json` pointers so eval workers do not need to
     scan manifests manually.

7. Automate report collation
   - Extend the existing scorecard summarizer pattern into the default
     post-run report: compact rows, warnings, missing refs, and stop/continue
     recommendation.
   - Emit one shared reporting-contract JSON per train/eval batch so docs can
     cite the artifact instead of reassembling facts.

8. Lower routine logging detail
   - Keep raw episodes and sidecars for selected diagnostics.
   - For normal gate runs, write compact summaries plus enough raw refs to
     reproduce. Avoid detailed per-step logs unless the question is about action
     selection, root visits, or seed/reset pathology.

9. Split worker roles by wait type
   - One worker owns train launch and immutable refs.
   - One worker owns parallel eval batches.
   - One worker owns summary collation and docs promotion.
   - The main Coach thread should read the compact decision table, not babysit
     each checkpoint probe.

## Practical Next Loop

Use this shape for the next LightZero train iteration:

```text
train whole job
  -> write latest/final/init refs and hashes
  -> launch parallel eval batch for the agreed gate rows
  -> summarize all rows into one compact table
  -> promote only the decision and source refs
  -> continue, stop, or escalate diagnostics
```

Recommended default gate rows:

- checkpoints: init and latest/final; add selected best only if pointer exists.
- opponents: `random_uniform`, `lagged_track_ball_1`, `track_ball`.
- metrics: raw score, shaped diagnostic, survival mean/p90, truncations, action
  histogram, entropy, no-fallback status, checkpoint hash.
- sim count: one agreed default for the lane; run sim sweeps only for root-target
  questions or selected checkpoints.

This keeps claims honest because independent MCTS eval remains mandatory. It
gets faster because the waiting moves from serial human-driven probes to one
parallel, reproducible batch with a compact decision artifact.

