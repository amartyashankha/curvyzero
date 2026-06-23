# Coach Recommendation

Date: 2026-05-23

## Short Version

Use stock LightZero/Coach for real learning runs right now. Do not switch the
overnight run to `direct_ctree_gpu_latent` as a speed default.

Do not claim compact ownership is faster for real training yet. It has useful
profile-side infrastructure, but it has not been promoted into the Coach
training loop.

## What Is Proven

- The trusted training lane is still stock LightZero `train_muzero`.
- `direct_ctree_gpu_latent` can now be requested in stock `mode=train` under a
  fail-closed hook/proof contract. Local focused tests pass.
- A real capped train canary with RND meter required passed: direct calls were
  positive, fallback was zero, replay/learner/RND were exercised.
- A warmer matched H100 train A/B did not show a speed win:
  stock `121.90` env steps/sec versus direct `117.83` env steps/sec.
- The optimizer has profile-side wins:
  - direct full-loop profile rows around `1.28x-1.31x`;
  - batched GPU observation profile around `1.5x`;
  - MCTX/JAX compact rows around `2x+`, but not semantic parity.
- No actual Coach run has proven a material speedup over the CZ26/L4 baseline.

## What Not To Do

- Do not use profile-only roots/sec as a Coach speed claim.
- Do not launch compact/MCTX as production training.
- Do not change the normal Coach train defaults just to test optimizer ideas.
- Do not mix learner iterations/hour, profile steps/sec, and roots/sec in one
  comparison.

## What To Do Next

Run two separate proof gates:

1. Compact ownership proof gate.
   - Isolated profile lane, no live Coach runs.
   - Compact arrays own actions/search/replay refs.
   - Learner/RND are real or loudly mocked.
   - Metric: compact closed-loop steps/sec plus identity checks.

2. If still chasing Lane 1, use it only as a diagnostic A/B:
   - same trainer shape, same hardware, same RND/reward/noise/death/sidecars;
   - compare stock path against the direct hook;
   - do not promote unless total `train_muzero_wall` moves, not just a sub-bucket.

Only after those pass should Coach consider a small opt-in training proof.

## Current Recommendation To Coach

Keep running learning experiments on stock Coach. The optimizer should keep the
direct hook as an opt-in diagnostic, not as the next training default. Ask
optimizer for a Coach-facing speed change only after it can say:

```text
called_train_muzero=true or deliberately isolated profile proof
same knobs as baseline
learner/RND/replay semantics checked
speed currency named
promotion status named
```
