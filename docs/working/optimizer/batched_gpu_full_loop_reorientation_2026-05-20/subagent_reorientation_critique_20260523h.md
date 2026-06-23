# Subagent Reorientation Critique, 2026-05-23h

Scope: docs-only critique. I did not edit code, touch live training runs,
start Modal jobs, mutate checkpoints, or change trainer defaults.

## One-Sentence Verdict

The current optimizer plan is still aimed at the right wall, but only if it
stays profile-only until a real immutable checkpoint passes PyTorch-to-JAX model
parity and the compact MCTX path is tested with the real model.

## High-Level Goal Right Now

- The goal is to make CurvyTron training faster without breaking the trusted
  Coach training lane.
- The trusted Coach lane remains stock LightZero `train_muzero` style work, not
  the old custom `--mode two-seat-selfplay` trainer.
- The optimizer lane is trying to remove the current search/control/dataflow
  wall.
- The strongest current speed signal is profile-only: MCTX/JAX behind the
  compact slab is faster than direct LightZero CTree on the same compact profile
  denominator.
- That speed signal is not yet a Coach training claim because the current MCTX
  path used a toy JAX model, not the real LightZero PyTorch model.

Plainly:

```text
The plan is not "polish rendering" anymore.
The plan is "prove a faster compact search path can run the real CurvyTron
model safely, then compare it honestly to the trusted loop."
```

## Is PyTorch-to-JAX Shadow Parity Useful Or A Rabbit Hole?

It is useful, with a strict time box.

Why it is useful:

- The MCTX/JAX profile rows are the clearest multi-x search speed signal in the
  current optimizer docs.
- Those rows cannot be promoted while they use a toy model.
- A raw PyTorch-to-JAX model parity harness is the smallest gate that connects
  that speed signal to the real CurvyTron policy model.
- Fresh-model parity already passed on Modal L4 GPU with explicit tolerance
  `5e-4`, required-key accounting, BatchNorm eval state, and nonzero output
  heads.
- The next missing proof is not philosophical. It is concrete: run the same
  parity check on a current immutable `iteration_N.pth.tar` checkpoint.

Why it could become a rabbit hole:

- Exact bit-for-bit GPU parity is not needed. Chasing that would waste time.
- Search-result parity with LightZero CTree is not the immediate gate. Raw model
  inference parity comes first.
- If the JAX model bridge keeps expanding into a full second trainer before
  checkpoint parity passes, the lane has drifted.
- If checkpoint parity fails because of model-surface mismatch and needs major
  LightZero reimplementation, we should stop and rethink instead of sinking more
  days into a shadow stack.

Current critique:

```text
Useful gate: yes.
Rabbit hole risk: real, but manageable.
Stop condition: checkpoint parity or real-model compact-service smoke cannot be
made clean without rewriting too much LightZero behavior.
```

## Next 3 Proof Gates Before Anything Touches Coach Training

1. **Current immutable checkpoint parity**
   - Find a current fixed `iteration_N.pth.tar` checkpoint.
   - Reject mutable refs like `latest.pth.tar` and `ckpt_best.pth.tar`.
   - Run the PyTorch LightZero model and the JAX shadow model on the same
     zero/ramp/random/checker-like observation fixtures and actions.
   - Require intermediate comparisons: representation latent, prediction
     policy/value, recurrent next latent, recurrent reward, recurrent
     policy/value.
   - Keep tolerances practical. The fresh GPU result showed latent max abs near
     `2e-4`, so `5e-4` is a reasonable first GPU tolerance.

2. **Real-model MCTX compact-service profile smoke**
   - Plug the real JAX shadow model into `MctxCompactSearchServiceV1` behind
     `CompactSearchServiceV1`.
   - Keep the row clearly labeled:
     `profile_only=true`, `calls_train_muzero=false`,
     `touches_live_runs=false`, `promotion_eligible=false`.
   - Prove selected actions are legal, active-root masks behave, no inactive
     root leaks into search, and compact replay index rows match the selected
     action handles.
   - This is still not Coach training. It only proves the faster backend can use
     the real model in the compact profile shell.

3. **Matched same-denominator full-loop-shaped profile**
   - Compare direct CTree, real-model MCTX compact service, and service-tax/mock
     ceilings on the same batch, actor count, simulation count, warmup,
     measured steps, RND setting, death/reset setting, observation backend, and
     checkpoint.
   - Break down time into env/observation, host-to-device, model, search,
     action readback, replay row commit, RND/sample materialization, and learner
     materialization if present.
   - Only after this profile shows a real full-loop-shaped gain should Coach
     consider a capped training smoke.

## What Would Make Us Stop This Lane And Rethink

- A current immutable checkpoint fails raw model parity in a way that is not a
  small tolerance or known GPU numeric issue.
- The JAX shadow model requires large, brittle reimplementation of LightZero
  internals beyond inference.
- The real-model MCTX compact service loses most of the toy-model speedup
  because of JAX/PyTorch/host transfer tax.
- The matched profile shows search is no longer the wall and the remaining wall
  is env mechanics, observation packaging, replay materialization, RND, or
  learner/sample construction.
- The lane starts changing Coach defaults before profile-only gates pass.
- The lane starts optimizing old render/body-circle/custom-trainer paths instead
  of the current compact search/dataflow wall.
- The MCTX semantics difference becomes a learning-quality risk that Coach does
  not explicitly accept.

## Stale Or Dangerous Docs / Claims

- Any doc that treats old `--mode two-seat-selfplay` scaled failures as evidence
  against CurvyTron or stock LightZero is dangerous. The reset decision says
  those failures mostly indict the custom adapter.
- Any doc that recommends old renderer names such as `body_circles_fast`,
  `fast_gray64_direct`, or old browser/render defaults as current launch advice
  is dangerous unless it points to the current source-of-truth constants.
- Any old checkpoint ref is suspect. One documented
  `iteration_32.pth.tar` ref was already missing from the current runs volume.
- Any MCTX speed doc is dangerous if read as Coach-ready. The correct label is:
  profile-only, faster compact search denominator, toy model until checkpoint
  parity passes.
- Any flat-A3 CTree doc is dangerous if read as the next big lever. The matched
  docs say it was roughly flat or slightly worse, not a 5x path.
- The optimizer folder has many useful sidecar docs, but the active hierarchy
  should stay anchored on `task_board.md`, `world_model.md`, `experiment_log.md`,
  and this parity status doc. Older sidecars should be treated as evidence, not
  current instructions.

## Concrete Recommendation

Keep this lane, but narrow it.

Do not touch Coach training yet. The next optimizer action should be:

```text
Find a current immutable checkpoint.
Run checkpoint PyTorch-to-JAX raw model parity.
If it passes, wire the real JAX shadow model into the profile-only MCTX compact
search service.
Then run a matched same-denominator profile against direct CTree.
```

If any of those gates fail for structural reasons, stop the MCTX/JAX bridge and
return to the broader search/dataflow architecture question.
